"""Smoke test: the starter graph (text -> llm -> image) end to end against the
mock NanoGPT harness, asserting BOTH request payloads exactly per SPEC-engine.md."""

import unittest

from tests import fixture
from tests.harness import MockNanoGPT, chat_response, image_response

from nanoodle import MediaRef, Workflow

STARTER_SYSTEM = ("You write image prompts. Turn the idea into one vivid, detailed "
                  "image prompt — scene, lighting, mood, style. Reply with the prompt only.")
LLM_REPLY = "A neon-soaked alley cat under sodium lights, cinematic, 35mm."
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


class SmokeTest(unittest.TestCase):
    def setUp(self):
        self.mock = MockNanoGPT().start()
        self.addCleanup(self.mock.stop)

    def _workflow(self):
        return Workflow.load(fixture("starter-graph.json"), api_key="test-key",
                             base_url=self.mock.base_url)

    def test_derive_inputs_outputs_settings(self):
        wf = self._workflow()
        self.assertEqual(wf.warnings, [])
        inputs = {s.key: s for s in wf.inputs}
        self.assertEqual(set(inputs), {"Text", "System prompt"})
        text = inputs["Text"]
        self.assertEqual((text.node_id, text.field, text.kind, text.optional),
                         ("n1", "text", "textarea", False))
        self.assertEqual(text.default, "a cozy ramen shop on a rainy night")
        system = inputs["System prompt"]
        self.assertEqual((system.node_id, system.field, system.optional), ("n2", "system", True))
        self.assertEqual(system.default, STARTER_SYSTEM)  # node value wins over spec default

        # n2.prompt and n3.prompt are wired -> hidden
        self.assertNotIn(("n2", "prompt"), [(s.node_id, s.field) for s in wf.inputs])
        self.assertNotIn(("n3", "prompt"), [(s.node_id, s.field) for s in wf.inputs])

        outputs = wf.outputs
        self.assertEqual(len(outputs), 1)
        self.assertEqual((outputs[0].key, outputs[0].node_id, outputs[0].type),
                         ("Image", "n3", "image"))
        self.assertEqual(outputs[0].ports, ["image"])

        settings = {s.key: s for s in wf.settings}
        self.assertIn("n2.model", settings)
        self.assertIn("n3.size", settings)
        self.assertEqual(settings["n3.size"].default, "1k")
        # n3.prompt is IO (wired), never a setting
        self.assertNotIn("n3.prompt", settings)

    def test_run_with_overridden_text_input(self):
        wf = self._workflow()
        self.mock.script("POST", "/api/v1/chat/completions",
                         chat_response(LLM_REPLY, cost_usd=0.0002, balance=4.51))
        self.mock.script("POST", "/v1/images/generations",
                         image_response(b64_list=[PNG_B64], cost=0.01, balance=4.50))

        result = wf.run({"Text": "a neon alley cat"})

        # ---- request 1: chat completions, SPEC-engine payload EXACTLY --------
        chats = self.mock.requests_to("/api/v1/chat/completions", "POST")
        self.assertEqual(len(chats), 1)
        chat = chats[0]
        self.assertEqual(chat.json, {
            "model": "zai-org/glm-5.2",
            "messages": [
                {"role": "system", "content": STARTER_SYSTEM},
                {"role": "user", "content": "a neon alley cat"},
            ],
            "temperature": 0.8,
        })
        self.assertNotIn("stream", chat.json)  # engine is NON-streaming
        self.assertEqual(chat.headers.get("authorization"), "Bearer test-key")
        self.assertEqual(chat.headers.get("x-api-key"), "test-key")
        self.assertTrue(chat.headers.get("content-type", "").startswith("application/json"))

        # ---- request 2: image generations, SPEC-engine payload EXACTLY -------
        imgs = self.mock.requests_to("/v1/images/generations", "POST")
        self.assertEqual(len(imgs), 1)
        img = imgs[0]
        self.assertEqual(img.json, {
            "model": "nano-banana-2-lite",
            "size": "1k",
            "n": 1,
            "response_format": "b64_json",
            "prompt": LLM_REPLY,   # the llm's text feeds the image prompt via the wire
        })
        self.assertEqual(img.headers.get("authorization"), "Bearer test-key")
        self.assertEqual(img.headers.get("x-api-key"), "test-key")
        self.assertEqual(len(self.mock.requests), 2)  # nothing else was called

        # ---- result keys and values ------------------------------------------
        image = result["Image"]
        self.assertIsInstance(image, MediaRef)
        self.assertEqual(image.url, "data:image/png;base64," + PNG_B64)  # b64 mime sniffed
        self.assertIs(result["n3"], image)                # also keyed by node id
        self.assertEqual(result.get("nope", "dflt"), "dflt")
        self.assertTrue(image.bytes().startswith(b"\x89PNG"))

        # per-node records + cost accounting
        self.assertEqual(result.nodes["n1"].status, "done")
        self.assertEqual(result.nodes["n2"].status, "done")
        self.assertEqual(result.nodes["n2"].out["text"], LLM_REPLY)
        self.assertEqual(result.nodes["n3"].status, "done")
        self.assertEqual(result.errors, [])
        self.assertAlmostEqual(result.cost_usd, 0.0102, places=6)
        self.assertTrue(result.cost_exact)
        self.assertEqual(result.remaining_balance, 4.50)

    def test_bare_scalar_input_single_required(self):
        wf = self._workflow()
        self.mock.script("POST", "/api/v1/chat/completions", chat_response(LLM_REPLY))
        self.mock.script("POST", "/v1/images/generations", image_response(b64_list=[PNG_B64]))
        result = wf.run("just a scalar")  # exactly one REQUIRED input -> allowed
        chat = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(chat.json["messages"][1]["content"], "just a scalar")
        self.assertIn("Image", result.outputs)
        # image endpoint returned no cost -> total is inexact
        self.assertFalse(result.cost_exact)

    def test_default_input_used_when_omitted(self):
        wf = self._workflow()
        self.mock.script("POST", "/api/v1/chat/completions", chat_response(LLM_REPLY))
        self.mock.script("POST", "/v1/images/generations", image_response(b64_list=[PNG_B64]))
        wf.run()
        chat = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(chat.json["messages"][1]["content"], "a cozy ramen shop on a rainy night")

    def test_settings_override(self):
        wf = self._workflow()
        self.mock.script("POST", "/api/v1/chat/completions", chat_response(LLM_REPLY))
        self.mock.script("POST", "/v1/images/generations", image_response(b64_list=[PNG_B64]))
        wf.run({"Text": "x"}, settings={"n3.size": "2k", "n2.maxTokens": "60"})
        chat = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(chat.json["max_tokens"], 60)
        img = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(img.json["size"], "2k")


if __name__ == "__main__":
    unittest.main()
