"""Prompt length caps — the wired prompt nobody can shorten.

Many image/video models reject an over-long prompt at NanoGPT's route (400
prompt_too_long). In a graph the prompt is WRITTEN by an upstream LLM, so a headless
caller has nothing to shorten and no hint the limit exists. These tests pin the deal:
trim to fit, always report it, learn a cap we did not know — and never rewrite the
graph's own prompts to avoid any of it. Everything runs against the mock server.
Port of nanoodle-js tests/prompt-caps.test.mjs.
"""

import copy
import unittest

from tests._util import MockedTest
from tests.harness import chat_response, image_response

from nanoodle import (PROMPT_CAPS, NanoodleError, RunError, fit_prompt_text,
                      is_prompt_too_long, prompt_cap, prompt_cap_from_error)

PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
           "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

# text -> LLM -> Image(qwen-image-3, cap 800): the reported shape, minimally
GRAPH = {
    "nodes": [
        {"id": "t1", "type": "text", "fields": {"text": "Album: Neon Mirage"}},
        {"id": "m1", "type": "llm",
         "fields": {"model": "gpt-5o", "system": "You are an art director.", "prompt": ""}},
        {"id": "i1", "type": "image",
         "fields": {"model": "qwen-image-3", "prompt": "", "size": "1024x1024"}},
    ],
    "links": [
        {"id": "l1", "from": {"node": "t1", "port": "text"}, "to": {"node": "m1", "port": "prompt"}},
        {"id": "l2", "from": {"node": "m1", "port": "text"}, "to": {"node": "i1", "port": "prompt"}},
    ],
}

TOO_LONG_400 = {
    "status": 400,
    "json": {"error": "Your prompt is too long for Brand New Model. Please shorten it to "
                      "640 characters or less (current: 900 characters).",
             "code": "prompt_too_long"},
}


class CapTableTest(unittest.TestCase):
    def test_table_for_image_and_video_catalog_for_audio_never_for_chat(self):
        self.assertEqual(prompt_cap({"type": "image", "fields": {"model": "qwen-image-3"}}), 800)
        self.assertEqual(prompt_cap({"type": "edit", "fields": {"model": "step-image-edit-2"}}), 512)
        self.assertEqual(prompt_cap({"type": "tvideo", "fields": {"model": "minimax-hailuo-02"}}), 2000)
        # an uncapped (or never-probed) model stays unconstrained — the permissive-gate rule
        self.assertIsNone(prompt_cap({"type": "image", "fields": {"model": "flux-kontext"}}))
        # an LLM's own limit is tokens, and it is never the node being fitted
        self.assertIsNone(prompt_cap({"type": "llm", "fields": {"model": "gpt-5o"}}))
        # audio caps ARE catalog metadata upstream, so they are read live, never listed here
        self.assertNotIn("audio", PROMPT_CAPS)
        catalog = {"audio": [{"id": "tts-1", "supported_parameters": {"max_chars": 4096}}]}
        node = {"type": "tts", "fields": {"model": "tts-1"}}
        self.assertEqual(prompt_cap(node, catalog=catalog), 4096)
        self.assertIsNone(prompt_cap(node), "no catalog -> no cap, never a guess")
        # a learned cap beats the baked table: the table is a snapshot, a live 400 is today
        self.assertEqual(prompt_cap({"type": "image", "fields": {"model": "qwen-image-3"}},
                                    learned={"image:qwen-image-3": 640}), 640)


class FitPromptTextTest(unittest.TestCase):
    def test_boundaries_fallbacks_and_the_one_hard_cut(self):
        s = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."
        cut = fit_prompt_text(s, 30)
        self.assertLessEqual(len(cut), 30)
        self.assertTrue(cut.endswith("."))
        self.assertEqual(fit_prompt_text("short", 100), "short")   # inside the cap -> untouched
        words = fit_prompt_text("aa " * 100, 50)
        self.assertLessEqual(len(words), 50)
        self.assertFalse(words.endswith(" "), "no sentence end -> word boundary, never mid-word")
        self.assertEqual(len(fit_prompt_text("x" * 500, 100)), 100,
                         "one unbroken token -> a hard cut is the last resort")


class ErrorShapeTest(unittest.TestCase):
    def test_all_three_live_shapes_parse_and_current_is_never_the_cap(self):
        shapes = [
            ('{"error":"Your prompt is too long for Qwen Image 3. Please shorten it to 800 '
             'characters or less (current: 831 characters).","code":"prompt_too_long"}', 800),
            ('{"error":{"message":"Your prompt is too long for Step Image Edit 2. Please shorten '
             'it to 512 characters or less (current: 5999 characters).","code":"prompt_too_long"}}', 512),
            ('{"error":"Prompt is too long. Please keep it under 3000 characters."}', 3000),
        ]
        for body, want in shapes:
            self.assertTrue(is_prompt_too_long(body), body[:60])
            self.assertEqual(prompt_cap_from_error(body), want)
        # to learn the "current" length would grow the limit with every failure — the model
        # would never be prevented, only re-broken
        self.assertNotEqual(prompt_cap_from_error(shapes[1][0]), 5999)
        self.assertIsNone(prompt_cap_from_error('{"error":"Invalid image input."}'))
        self.assertFalse(is_prompt_too_long('{"error":"Invalid image input."}'))


class TrimAndDiscloseTest(MockedTest):
    def _wf(self, graph=None):
        return self.wf_dict(copy.deepcopy(graph or GRAPH))

    def _script(self, chat_text):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response(chat_text))
        self.mock.script("POST", "/v1/images/generations", image_response(b64_list=[PNG_B64]))

    def test_the_graphs_own_prompts_reach_the_model_exactly_as_written(self):
        self._script("a neon city")
        self._wf().run()
        messages = self.mock.requests_to("/api/v1/chat/completions")[0].json["messages"]
        self.assertEqual(messages[0]["content"], "You are an art director.",
                         "the system prompt is untouched — no length directive is injected")
        self.assertEqual(len(messages), 2, "and nothing extra is appended to the conversation")

    def test_a_downstream_cap_never_leaks_upstream(self):
        self._script("a neon city")
        graph = copy.deepcopy(GRAPH)
        graph["nodes"][2]["fields"]["model"] = "step-image-edit-2"   # tightest cap (512)
        self._wf(graph).run()
        chat = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(chat.json["messages"][0]["content"], "You are an art director.")
        self.assertNotIn("512", chat.body.decode("utf-8"),
                         "the cap is applied to the image call, and nowhere else")

    def test_an_overflowing_prompt_is_fitted_at_a_sentence_boundary_and_reported(self):
        # an LLM that ignores the budget entirely (they do overshoot; this one does it wildly)
        runaway = "A neon city at night. " + "Rain slicks the street. " * 60
        self._script(runaway)
        events = []
        with self.assertWarns(RuntimeWarning):
            result = self._wf().run(on_progress=events.append)

        sent = self.mock.requests_to("/v1/images/generations")[0].json["prompt"]
        self.assertLessEqual(len(sent), 800, "sent %d characters, the cap is 800" % len(sent))
        self.assertTrue(sent.endswith("."), "the cut lands on a sentence boundary")
        self.assertGreaterEqual(len(sent), 560,
                                "a boundary cut never throws away more than the overflow did")

        trims = [e for e in events if e["type"] == "prompt-trimmed"]
        self.assertEqual(len(trims), 1, "a trimmed prompt is announced — never silent")
        self.assertEqual(trims[0]["node_id"], "i1")
        self.assertEqual(trims[0]["cap"], 800)
        self.assertEqual(trims[0]["from"], len(runaway))
        self.assertEqual(trims[0]["to"], len(sent))
        # the result carries the same disclosure for a caller that passed no on_progress
        self.assertEqual(result.prompt_trims,
                         [{"node_id": "i1", "name": "Image", "from": len(runaway),
                           "to": len(sent), "cap": 800}])

    def test_a_prompt_that_fits_is_sent_verbatim_and_nothing_is_announced(self):
        self._script("a neon city at 3am")
        events = []
        result = self._wf().run(on_progress=events.append)
        self.assertEqual(self.mock.requests_to("/v1/images/generations")[0].json["prompt"],
                         "a neon city at 3am")
        self.assertEqual([e for e in events if e["type"] == "prompt-trimmed"], [])
        self.assertEqual(result.prompt_trims, [])


class LearnedCapTest(MockedTest):
    def test_an_unknown_models_cap_is_learned_from_the_live_400(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("a neon city"))
        # a model NOT in the table — exactly the case the table cannot cover
        self.mock.script("POST", "/v1/images/generations", TOO_LONG_400)
        graph = copy.deepcopy(GRAPH)
        graph["nodes"][2]["fields"]["model"] = "brand-new-model"
        wf = self.wf_dict(graph)

        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("running again should succeed", str(ctx.exception.result.errors[0]["message"]))
        self.assertIn("640 characters or less", ctx.exception.result.errors[0]["message"],
                      "without hiding what the API actually said")
        self.assertEqual(wf._prompt_caps.get("image:brand-new-model"), 640,
                         "the cap is banked for the retry")

        # second run: the same Workflow now trims to the learned 640 before sending
        self.mock.reset()
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("A neon city. " * 80))
        self.mock.script("POST", "/v1/images/generations", image_response(b64_list=[PNG_B64]))
        events = []
        wf.run(on_progress=events.append)
        sent = self.mock.requests_to("/v1/images/generations")[0].json["prompt"]
        self.assertLessEqual(len(sent), 640, "the learned cap is applied, not rediscovered")
        self.assertEqual([e["cap"] for e in events if e["type"] == "prompt-trimmed"], [640])

    def test_an_unrelated_400_is_relayed_untouched(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("a neon city"))
        self.mock.script("POST", "/v1/images/generations",
                         {"status": 400, "json": {"error": "Invalid image input."}})
        wf = self.wf_dict(copy.deepcopy(GRAPH))
        with self.assertRaises(NanoodleError) as ctx:
            wf.run()
        self.assertNotIn("running again should succeed", str(ctx.exception))
        self.assertEqual(wf._prompt_caps, {})


if __name__ == "__main__":
    unittest.main()
