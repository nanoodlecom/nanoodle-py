"""Graph loading (aliases / migrations / unknown types / cycles) and the
inputs/outputs/settings public interface incl. key resolution."""

import unittest

from tests import fixture

from nanoodle import NanoodleError, UnsupportedNodeError, Workflow
from nanoodle.graph import topo_order


class GraphLoadTest(unittest.TestCase):
    def test_audio_alias_and_text_port_migration(self):
        wf = Workflow.from_dict({
            "nodes": [
                {"id": "n1", "type": "text", "fields": {"text": "hi"}},
                {"id": "n2", "type": "audio", "fields": {"model": "m"}},  # legacy alias
            ],
            "links": [
                {"id": "l1", "from": {"node": "n1", "port": "text"},
                 "to": {"node": "n2", "port": "text"}},  # legacy port
            ],
        }, api_key="k")
        self.assertEqual(wf.graph.node("n2").type, "tts")
        self.assertEqual(wf.graph.links[0].to_port, "prompt")  # migrated

    def test_orphaned_links_dropped_and_minimal_form(self):
        wf = Workflow.from_dict({"nodes": [{"id": "n1", "type": "text", "fields": {}}],
                                 "links": [{"id": "l1", "from": {"node": "nX", "port": "text"},
                                            "to": {"node": "n1", "port": "text"}}]}, api_key="k")
        self.assertEqual(wf.graph.links, [])

    def test_unknown_type_warns_on_load_fails_on_run(self):
        wf = Workflow.from_dict({"nodes": [{"id": "n1", "type": "flurble", "fields": {}}]},
                                api_key="k")
        self.assertTrue(any("flurble" in w for w in wf.warnings))
        with self.assertRaises(UnsupportedNodeError) as ctx:
            wf.run()
        self.assertIn("flurble", str(ctx.exception))
        self.assertEqual(ctx.exception.node_id, "n1")

    def test_json_string_accepted(self):
        wf = Workflow.from_dict('{"nodes": [{"id":"n1","type":"text","fields":{"text":"x"}}]}',
                                api_key="k")
        self.assertEqual(wf.graph.node("n1").fields["text"], "x")

    def test_cycle_error_names_nodes(self):
        wf = Workflow.load(fixture("cycle.json"), api_key="k")
        with self.assertRaises(NanoodleError) as ctx:
            wf.run()
        msg = str(ctx.exception)
        self.assertIn("cycle", msg)
        self.assertIn("n1", msg)
        self.assertIn("n2", msg)

    def test_topo_order_respects_deps(self):
        wf = Workflow.load(fixture("starter-graph.json"), api_key="k")
        self.assertEqual(topo_order(wf.graph), ["n1", "n2", "n3"])

    def test_comment_nodes_never_run(self):
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "text", "fields": {"text": "hi"}},
            {"id": "n2", "type": "comment", "fields": {"text": "a note"}},
        ]}, api_key="k")
        result = wf.run()
        self.assertNotIn("n2", result.nodes)
        self.assertEqual(result["Text"], "hi")


class KeyResolutionTest(unittest.TestCase):
    def test_custom_name_single_required_input_names_it(self):
        wf = Workflow.load(fixture("field-override.json"), api_key="k")
        keys = {s.key for s in wf.inputs}
        self.assertIn("Persona", keys)   # PR #138 flat-label rule
        self.assertIn("Prompt", keys)    # n2's unwired prompt
        # n2.system is wired -> hidden
        self.assertNotIn(("n2", "system"), [(s.node_id, s.field) for s in wf.inputs])

    def test_resolution_orders(self):
        from nanoodle.iodef import resolve_input_key
        wf = Workflow.load(fixture("field-override.json"), api_key="k")
        specs = wf.inputs
        by_name = resolve_input_key(specs, " persona ", wf.graph)   # trimmed, case-insensitive
        self.assertEqual((by_name.node_id, by_name.field), ("n1", "text"))
        by_id_field = resolve_input_key(specs, "n1.text", wf.graph)
        self.assertIs(by_id_field, by_name)
        bare_id = resolve_input_key(specs, "n2", wf.graph)          # single input on the node
        self.assertEqual(bare_id.field, "prompt")
        by_label = resolve_input_key(specs, "PROMPT", wf.graph)
        self.assertIs(by_label, bare_id)
        by_field = resolve_input_key(specs, "text", wf.graph)
        self.assertIs(by_field, by_name)

    def test_unknown_key_lists_available(self):
        wf = Workflow.load(fixture("field-override.json"), api_key="k")
        with self.assertRaises(NanoodleError) as ctx:
            wf.run({"Bogus": "x"})
        msg = str(ctx.exception)
        self.assertIn("Bogus", msg)
        self.assertIn("Persona", msg)
        self.assertIn("Prompt", msg)

    def test_duplicate_names_ambiguity_and_key_fallback(self):
        wf = Workflow.load(fixture("duplicate-names.json"), api_key="k")
        keys = sorted(s.key for s in wf.inputs)
        # duplicate "Text" labels fall back to nodeId.field keys
        self.assertIn("n1.text", keys)
        self.assertIn("n2.text", keys)
        with self.assertRaises(NanoodleError) as ctx:
            wf.run({"Writer": "x"})   # two nodes named Writer -> ambiguous
        self.assertIn("ambiguous", str(ctx.exception))

    def test_duplicate_output_names_suffixed_in_topo_order(self):
        wf = Workflow.load(fixture("duplicate-names.json"), api_key="k")
        self.assertEqual([o.key for o in wf.outputs], ["Writer", "Writer 2"])
        self.assertEqual([o.node_id for o in wf.outputs], ["n3", "n4"])

    def test_bare_scalar_refused_with_multiple_required(self):
        wf = Workflow.load(fixture("llm-vision.json"), api_key="k")
        with self.assertRaises(NanoodleError) as ctx:
            wf.run("hello")
        self.assertIn("exactly one required input", str(ctx.exception))

    def test_missing_required_input_named(self):
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "llm", "fields": {"model": "m"}},
        ]}, api_key="k")
        with self.assertRaises(NanoodleError) as ctx:
            wf.run()
        self.assertIn("missing required input", str(ctx.exception))
        self.assertIn("Prompt", str(ctx.exception))

    def test_wired_setting_refused(self):
        wf = Workflow.load(fixture("starter-graph.json"), api_key="k")
        with self.assertRaises(NanoodleError) as ctx:
            wf.run({"Text": "x"}, settings={"n3.prompt": "sneaky"})
        self.assertIn("wired", str(ctx.exception))

    def test_choice_input_options_and_validation(self):
        wf = Workflow.load(fixture("join-choice.json"), api_key="k")
        choice = next(s for s in wf.inputs if s.kind == "choice")
        self.assertEqual(choice.options, ["red", "blue", "green"])
        self.assertEqual(choice.default, "blue")
        with self.assertRaises(NanoodleError) as ctx:
            wf.run({"Choice": "purple"})
        self.assertIn("options", str(ctx.exception))


class LocalNodesTest(unittest.TestCase):
    def test_join_choice_chain_runs_offline(self):
        wf = Workflow.load(fixture("join-choice.json"), api_key=None)
        import unittest.mock as mock
        with mock.patch.dict("os.environ", {}, clear=False):
            result = wf.run()   # local-only graph needs no API key
        self.assertEqual(result["Join"], "hello - blue")
        result2 = wf.run({"Choice": "green", "Text": "hey"})
        self.assertEqual(result2["Join"], "hey - green")

    def test_join_sep_backslash_n_and_empty_filter(self):
        # n2 is an inputless join -> emits "" at runtime, which b must filter out
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "text", "fields": {"text": "a"}},
            {"id": "n2", "type": "join", "fields": {}},
            {"id": "n3", "type": "join", "fields": {"sep": "\\n"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "text"}, "to": {"node": "n3", "port": "a"}},
            {"id": "l2", "from": {"node": "n2", "port": "text"}, "to": {"node": "n3", "port": "b"}},
        ]})
        self.assertEqual(wf.run()["Join"], "a")   # empty b filtered, literal \n sep

    def test_empty_required_text_input_is_named_error(self):
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "text", "fields": {"text": ""}},
        ]})
        with self.assertRaises(NanoodleError) as ctx:
            wf.run()
        self.assertIn("missing required input", str(ctx.exception))
        self.assertIn("Text", str(ctx.exception))
        wf2 = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "text", "fields": {"text": "a"}},
            {"id": "n2", "type": "text", "fields": {"text": "b"}},
            {"id": "n3", "type": "join", "fields": {"sep": "\\n"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "text"}, "to": {"node": "n3", "port": "a"}},
            {"id": "l2", "from": {"node": "n2", "port": "text"}, "to": {"node": "n3", "port": "b"}},
        ]})
        self.assertEqual(wf2.run()["Join"], "a\nb")

    def test_choice_bad_selected_falls_back_to_first(self):
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "choice", "fields": {"options": "x\ny", "selected": "gone"}},
        ]})
        self.assertEqual(wf.run()["Choice"], "x")

    def test_no_api_key_with_network_nodes_fails_upfront(self):
        import unittest.mock as mock
        with mock.patch.dict("os.environ", {"NANOGPT_API_KEY": ""}, clear=False):
            import os
            os.environ.pop("NANOGPT_API_KEY", None)
            wf = Workflow.load(fixture("starter-graph.json"), api_key=None)
            with self.assertRaises(NanoodleError) as ctx:
                wf.run({"Text": "x"})
        self.assertIn("API key", str(ctx.exception))

    def test_unsupported_node_fails_fast(self):
        calls = []

        def tripwire_http(*a, **kw):  # any network call here is a spec violation
            calls.append(a)
            raise AssertionError("network call before fail-fast check")

        wf = Workflow.load(fixture("unsupported-node.json"), api_key="k",
                           http=tripwire_http)
        self.assertEqual(wf.warnings, [])   # load only warns for UNKNOWN types
        with self.assertRaises(UnsupportedNodeError) as ctx:
            wf.run()
        msg = str(ctx.exception)
        self.assertIn("node type 'resize' does local media processing that requires "
                      "the nanoodle browser app; not supported by this library yet", msg)
        self.assertEqual(ctx.exception.node_type, "resize")
        self.assertEqual(calls, [])   # fail-fast happened BEFORE any network call


if __name__ == "__main__":
    unittest.main()
