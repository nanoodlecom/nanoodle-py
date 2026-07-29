"""Author-marked optional input nodes (fields.optional).

The editor's "optional" checkbox on an input node makes every input that node surfaces
skippable: the run proceeds and the node yields an empty value instead of failing.
Consumers drop the empty value. Port of the nanoodle-js io.test.mjs optional cases —
the same graph must run headlessly in Python and in JavaScript.
"""

import unittest

from tests._util import MockedTest, tripwire_http
from tests.harness import chat_response

from nanoodle import NanoodleError, RunError, Workflow

PNG_DATA_URL = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
                "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _upload_graph(optional):
    fields = {"optional": True} if optional else {}
    return {"nodes": [{"id": "u1", "type": "upload", "name": "Style reference",
                       "fields": fields}], "links": []}


def _vision_graph(optional):
    """optional Image input -> an llm reference port (img1)."""
    return {
        "nodes": [
            {"id": "u1", "type": "upload", "name": "Style reference",
             "fields": ({"optional": True} if optional else {})},
            {"id": "m1", "type": "llm",
             "fields": {"model": "m", "prompt": "describe the style", "system": ""}},
        ],
        "links": [{"id": "l1", "from": {"node": "u1", "port": "image"},
                   "to": {"node": "m1", "port": "img1"}}],
    }


class OptionalDerivationTest(unittest.TestCase):
    def _wf(self, data):
        return Workflow.from_dict(data, api_key="k", http=tripwire_http)

    def test_author_optional_node_marks_every_input_it_surfaces(self):
        wf = self._wf({"nodes": [
            {"id": "n1", "type": "text", "fields": {}},
            {"id": "n2", "type": "text", "name": "Extra notes", "fields": {"optional": True}},
        ], "links": []})
        by_node = {s.node_id: s for s in wf.inputs}
        self.assertTrue(by_node["n2"].optional)
        self.assertFalse(by_node["n1"].optional)
        # a single optional input still takes the node's custom name as its key
        self.assertEqual(by_node["n2"].key, "Extra notes")

    def test_string_form_from_a_checkbox_round_trip_counts(self):
        wf = self._wf({"nodes": [{"id": "n1", "type": "text",
                                  "fields": {"optional": "true"}}], "links": []})
        self.assertTrue(wf.inputs[0].optional)

    def test_an_llms_prompt_becomes_optional_too(self):
        wf = self._wf({"nodes": [{"id": "n1", "type": "llm", "name": "Notes",
                                  "fields": {"model": "m", "optional": True}}], "links": []})
        self.assertEqual([(s.field, s.optional) for s in wf.inputs],
                         [("prompt", True), ("system", True)])


class OptionalRunTest(unittest.TestCase):
    def test_omitted_optional_media_yields_an_empty_value(self):
        wf = Workflow.from_dict(_upload_graph(True), api_key="k", http=tripwire_http)
        self.assertTrue(wf.inputs[0].optional)
        self.assertEqual(wf.inputs[0].key, "Style reference")
        result = wf.run()
        self.assertEqual(result.nodes["u1"].status, "done")
        self.assertEqual(result.nodes["u1"].out["image"], "")

    def test_the_same_graph_without_the_checkbox_still_demands_the_input(self):
        wf = Workflow.from_dict(_upload_graph(False), api_key="k", http=tripwire_http)
        self.assertFalse(wf.inputs[0].optional)
        with self.assertRaises(NanoodleError) as ctx:
            wf.run()
        self.assertIn("Style reference", str(ctx.exception))
        self.assertIn("missing required input", str(ctx.exception))


class OptionalConsumerTest(MockedTest):
    def test_a_skipped_optional_reference_is_dropped_by_the_consumer(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("ok"))
        wf = self.wf_dict(_vision_graph(True))
        result = wf.run()
        self.assertEqual(result.nodes["u1"].out["image"], "")
        self.assertEqual(result.nodes["m1"].status, "done")
        messages = self.mock.requests_to("/api/v1/chat/completions")[0].json["messages"]
        self.assertEqual(messages[-1]["content"], "describe the style",
                         "no empty image part is posted — the reference is simply dropped")

    def test_a_supplied_optional_reference_is_still_used(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("ok"))
        wf = self.wf_dict(_vision_graph(True))
        wf.run({"Style reference": PNG_DATA_URL})
        messages = self.mock.requests_to("/api/v1/chat/completions")[0].json["messages"]
        parts = messages[-1]["content"]
        self.assertEqual([p["type"] for p in parts], ["text", "image_url"])
        self.assertEqual(parts[1]["image_url"]["url"], PNG_DATA_URL)

    def test_without_the_checkbox_the_missing_reference_fails_the_run(self):
        wf = self.wf_dict(_vision_graph(False))
        with self.assertRaises((NanoodleError, RunError)) as ctx:
            wf.run()
        self.assertIn("Style reference", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
