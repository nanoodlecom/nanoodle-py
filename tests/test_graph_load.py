"""Graph loading: aliases, link migration, malformed records, unknown types, topo
order, cycle errors, comment nodes (SPEC-format loader semantics)."""

import unittest

from tests import fixture
from tests._util import tripwire_http

from nanoodle import NanoodleError, UnsupportedNodeError, Workflow
from nanoodle.graph import display_name, materialize, topo_order


class DuplicateNodeIdTest(unittest.TestCase):
    def test_duplicate_node_id_raises_instead_of_silently_overwriting(self):
        # Regression (D4): the second node with an id used to silently
        # replace the first, so a graph ran with a node its author never
        # saw. JS graph.mjs throws; Py now throws the same error.
        graph = {
            "nodes": [
                {"id": "a", "type": "text", "fields": {"text": "first"}},
                {"id": "a", "type": "text", "fields": {"text": "second"}},
            ],
            "links": [],
        }
        with self.assertRaises(NanoodleError) as ctx:
            Workflow.from_dict(graph, api_key="<redacted>")
        self.assertIn('duplicate node id "a"', str(ctx.exception))
        with self.assertRaises(NanoodleError):
            materialize(graph)


class LoaderSemanticsTest(unittest.TestCase):
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

    def test_music_text_port_migration(self):
        wf = Workflow.from_dict({
            "nodes": [
                {"id": "n1", "type": "text", "fields": {"text": "hi"}},
                {"id": "n2", "type": "music", "fields": {"model": "m"}},
            ],
            "links": [{"id": "l1", "from": {"node": "n1", "port": "text"},
                       "to": {"node": "n2", "port": "text"}}],
        }, api_key="k")
        self.assertEqual(wf.graph.links[0].to_port, "prompt")

    def test_llm_text_port_not_migrated(self):
        # the migration is scoped to music/tts targets only
        wf = Workflow.from_dict({
            "nodes": [
                {"id": "n1", "type": "text", "fields": {"text": "hi"}},
                {"id": "n2", "type": "llm", "fields": {"model": "m", "prompt": "p"}},
            ],
            "links": [{"id": "l1", "from": {"node": "n1", "port": "text"},
                       "to": {"node": "n2", "port": "text"}}],
        }, api_key="k")
        self.assertEqual(wf.graph.links[0].to_port, "text")

    def test_dangling_links_are_retained_but_do_not_shape_io(self):
        for role, source, target in (("source", "nX", "n1"), ("target", "n1", "nX")):
            with self.subTest(role=role):
                wf = Workflow.from_dict({"nodes": [{"id": "n1", "type": "text", "fields": {}}],
                                         "links": [{"id": "l1", "from": {"node": source, "port": "text"},
                                                    "to": {"node": target, "port": "text"}}]}, api_key="k")
                self.assertEqual([l.id for l in wf.graph.links], ["l1"])
                self.assertEqual([s.field for s in wf.inputs], ["text"])
                self.assertEqual([s.node_id for s in wf.outputs], ["n1"])
                self.assertIn('missing %s node "nX"' % role, wf.warnings[0])
                with self.assertRaises(NanoodleError) as ctx:
                    wf.run()
                self.assertEqual(str(ctx.exception), wf.warnings[0])

    def test_minimal_form(self):
        # {nodes} only (no links/v/nid/lid/view) is a valid save
        wf = Workflow.from_dict({"nodes": [{"id": "n1", "type": "text",
                                            "fields": {"text": "x"}}]}, api_key="k")
        self.assertEqual(wf.graph.node("n1").fields["text"], "x")
        self.assertEqual(wf.graph.links, [])

    def test_layout_and_editor_keys_ignored(self):
        wf = Workflow.load(fixture("starter-graph.json"), api_key="k")
        self.assertEqual(wf.warnings, [])
        self.assertEqual(list(wf.graph.nodes), ["n1", "n2", "n3"])
        self.assertEqual(len(wf.graph.links), 2)

    def test_unknown_type_warns_on_load_fails_on_run(self):
        wf = Workflow.from_dict({"nodes": [{"id": "n1", "type": "flurble", "fields": {}}]},
                                api_key="k")
        self.assertTrue(any("flurble" in w for w in wf.warnings))
        with self.assertRaises(UnsupportedNodeError) as ctx:
            wf.run()
        self.assertIn("flurble", str(ctx.exception))
        self.assertEqual(ctx.exception.node_id, "n1")

    def test_draw_node_now_fails_as_unknown_type(self):
        # draw was removed (NanoGPT retired the chat-image contract) — a graph
        # carrying one must warn on load and fail fast BEFORE any API spend.
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "draw", "fields": {"model": "m", "prompt": "p"}}]},
            api_key="k")
        self.assertTrue(any("draw" in w for w in wf.warnings))
        with self.assertRaises(UnsupportedNodeError) as ctx:
            wf.run()
        self.assertEqual(ctx.exception.node_id, "n1")

    def test_non_object_json_rejected(self):
        with self.assertRaises(NanoodleError):
            materialize([1, 2, 3])

    def test_json_string_accepted(self):
        wf = Workflow.from_dict('{"nodes": [{"id":"n1","type":"text","fields":{"text":"x"}}]}',
                                api_key="k")
        self.assertEqual(wf.graph.node("n1").fields["text"], "x")

    def test_invalid_json_string_reports_line_and_column(self):
        data = '{\n  "nodes": [}'
        for load in (Workflow.from_dict, materialize):
            with self.subTest(load=load.__name__), self.assertRaisesRegex(
                    NanoodleError, r"invalid workflow JSON .*line 2 column 13"):
                load(data)

    def test_display_name_resolution(self):
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "llm", "fields": {}, "name": "  Poet  "},
            {"id": "n2", "type": "llm", "fields": {}},
            {"id": "n3", "type": "llm", "fields": {}, "name": "   "},
        ]}, api_key="k")
        self.assertEqual(display_name(wf.graph.node("n1")), "Poet")   # custom, trimmed
        self.assertEqual(display_name(wf.graph.node("n2")), "LLM")    # type title
        self.assertEqual(display_name(wf.graph.node("n3")), "LLM")    # blank name -> title


class MalformedNodeRecordTest(unittest.TestCase):
    def test_invalid_records_report_their_position(self):
        invalid_ids = (None, "", " ", [], {}, True, False, float("inf"), float("nan"))
        invalid_types = (None, "", "\t", [], {}, 1, True)
        cases = [(entry, "must be an object") for entry in (None, [], "text", 1, False)]
        cases += [({"type": "text"}, "invalid id"), ({"id": "bad"}, "invalid type")]
        cases += [({"id": nid, "type": "text"}, "invalid id") for nid in invalid_ids]
        cases += [({"id": "bad", "type": ntype}, "invalid type") for ntype in invalid_types]
        cases += [({"id": "bad", "type": "text", "fields": 1}, "invalid fields")]
        for entry, expected in cases:
            with self.subTest(entry=entry):
                with self.assertRaises(NanoodleError) as ctx:
                    materialize({"nodes": [{"id": "first", "type": "text"}, entry]})
                self.assertIn("node at position 2", str(ctx.exception))
                self.assertIn(expected, str(ctx.exception))

    def test_invalid_node_list_shape_reports_graph_error(self):
        for nodes in (None, "text", {}, False):
            with self.subTest(nodes=nodes), self.assertRaisesRegex(NanoodleError, "'nodes' list"):
                materialize({"nodes": nodes})

    def test_numeric_ids_and_node_type_aliases_remain_addressable(self):
        graph = materialize({"nodes": [
            {"id": 0, "type": "text", "fields": {"text": "speak"}},
            {"id": 12, "type": "audio", "fields": None},
            {"id": 5.5, "type": "decide-branch"},
        ], "links": [
            {"from": {"node": 0, "port": "text"}, "to": {"node": 12, "port": "text"}},
        ]})
        self.assertEqual(list(graph.nodes), [0, 12, 5.5])
        self.assertEqual(graph.node(12).type, "tts")
        self.assertEqual(graph.links[0].to_port, "prompt")
        self.assertEqual(graph.node(5.5).type, "decide")
        self.assertTrue(graph.node(5.5).fields["branch"])
        self.assertEqual(graph.warnings, [])


class SourceWireTest(unittest.TestCase):
    @staticmethod
    def _graph(node_type, port):
        return {"nodes": [
            {"id": "source", "type": node_type, "fields": {"text": "upstream"}},
            {"id": "sink", "type": "text", "fields": {"text": "stored fallback"}},
        ], "links": [
            {"id": "wire", "from": {"node": "source", "port": port},
             "to": {"node": "sink", "port": "text"}},
        ]}

    def test_undeclared_source_is_retained_for_inspect_and_rejected_before_running(self):
        wf = Workflow.from_dict(self._graph("text", "obsoletePort"), api_key="", http=tripwire_http)
        issue = 'wire source.obsoletePort → sink.text uses undeclared source port "obsoletePort" on Text'
        self.assertEqual(wf.warnings, [issue])
        self.assertEqual(wf.graph.links[0].id, "wire")
        events = []
        with self.assertRaises(NanoodleError) as ctx:
            wf.run(on_progress=events.append)
        self.assertEqual(str(ctx.exception), issue)
        self.assertEqual(events, [])

    def test_structural_checks_accept_dynamic_alias_and_optional_outputs(self):
        possible_ports = {
            "audio": ("audio",),
            "decide-branch": ("yes", "no", "decision"),
            "edit": ("images", "layers"),
            "endpoint": ("text", "image", "images", "audio", "video"),
            "image": ("images",),
            "inpaint": ("images", "layers"),
            "slice-3d": ("sheet", "layout", "image"),
            "sprite-stack-render": ("video",),
            "vframes": ("frame1", "frame48"),
        }
        for node_type, ports in possible_ports.items():
            for port in ports:
                with self.subTest(node_type=node_type, port=port):
                    self.assertEqual(materialize(self._graph(node_type, port)).warnings, [])

    def test_impossible_frame_spellings_are_rejected(self):
        for port in ("frame0", "frame01", "frame49", "frame1\n", "frame١", None, []):
            with self.subTest(port=port):
                graph = materialize(self._graph("vframes", port))
                self.assertEqual(len(graph.warnings), 1)
                self.assertIn("undeclared source port", graph.warnings[0])


class TargetWireTest(unittest.TestCase):
    @staticmethod
    def _graph(node_type, port):
        return {"nodes": [
            {"id": "source", "type": "text", "fields": {"text": "upstream"}},
            {"id": "sink", "type": node_type, "fields": {"promtp": "typo in stored fields"}},
        ], "links": [
            {"id": "wire", "from": {"node": "source", "port": "text"},
             "to": {"node": "sink", "port": port}},
        ]}

    def test_unrecognized_targets_fail_even_with_a_matching_stored_field(self):
        for node_type, port in (("llm", "promtp"), ("llm", "text"), ("llm", "ref1"),
                                ("decide", "branch"), ("join", None), ("join", [])):
            with self.subTest(node_type=node_type, port=port):
                wf = Workflow.from_dict(self._graph(node_type, port), api_key="", http=tripwire_http)
                self.assertEqual([l.id for l in wf.graph.links], ["wire"])
                self.assertIn("undeclared target port", wf.warnings[0])
                events = []
                with self.assertRaises(NanoodleError) as ctx:
                    wf.run(on_progress=events.append)
                self.assertEqual(str(ctx.exception), wf.warnings[0])
                self.assertEqual(events, [])

    def test_runtime_inputs_and_known_field_overrides_are_structurally_valid(self):
        targets = {
            "llm": ("prompt", "system", "model", "img", "img2", "audio"),
            "edit": ("prompt", "image", "image2"),
            "decide": ("text", "img2", "question", "options", "levels"),
            "combine": ("vid1", "clip2"),
            "tvideo": ("prompt", "ref2"),
            "ivideo": ("image", "endframe"),
            "model3d": ("model", "image", "prompt"),
            "endpoint": ("url", "mode", "model", "system"),
            "text": ("text",),
            "choice": ("options", "selected"),
            "music": ("text", "lyrics", "negative_prompt"),
            "tts": ("text", "instructions"),
        }
        for node_type, ports in targets.items():
            for port in ports:
                with self.subTest(node_type=node_type, port=port):
                    self.assertEqual(materialize(self._graph(node_type, port)).warnings, [])

    def test_browser_only_inputs_report_the_python_runtime_limit(self):
        for node_type, port in (("llm", "video"), ("vedit", "image"), ("vedit", "ref2")):
            with self.subTest(node_type=node_type, port=port):
                media = "video" if port == "video" else "image"
                graph = self._graph(node_type, port)
                graph["nodes"][0]["type"] = "vupload" if media == "video" else "upload"
                graph["links"][0]["from"]["port"] = media
                wf = Workflow.from_dict(graph, api_key="k", http=tripwire_http)
                self.assertIn(f'input "{port}", which the Python SDK cannot send', wf.warnings[0])
                with self.assertRaises(NanoodleError) as ctx:
                    wf.run()
                self.assertEqual(str(ctx.exception), wf.warnings[0])


class CompetingInputTest(unittest.TestCase):
    def test_legacy_and_current_prompt_ports_compete_after_migration(self):
        wf = Workflow.from_dict({"nodes": [
            {"id": "first", "type": "text", "fields": {"text": "first"}},
            {"id": "second", "type": "text", "fields": {"text": "second"}},
            {"id": "sink", "type": "audio", "fields": {"model": "m"}},
        ], "links": [
            {"id": "legacy", "from": {"node": "first", "port": "text"},
             "to": {"node": "sink", "port": "text"}},
            {"id": "current", "from": {"node": "second", "port": "text"},
             "to": {"node": "sink", "port": "prompt"}},
        ]}, api_key="k", http=tripwire_http)
        self.assertEqual([(l.id, l.to_port) for l in wf.graph.links],
                         [("legacy", "prompt"), ("current", "prompt")])
        issue = ("input sink.prompt has competing wires from first.text, second.text. "
                 "Leave one active wire for this input.")
        self.assertEqual(wf.warnings, [issue])
        with self.assertRaises(NanoodleError) as ctx:
            wf.run()
        self.assertEqual(str(ctx.exception), issue)


class TopoOrderTest(unittest.TestCase):
    def test_topo_order_respects_deps(self):
        wf = Workflow.load(fixture("starter-graph.json"), api_key="k")
        self.assertEqual(topo_order(wf.graph), ["n1", "n2", "n3"])

    def test_cycle_error_names_nodes(self):
        wf = Workflow.load(fixture("cycle.json"), api_key="k")
        with self.assertRaises(NanoodleError) as ctx:
            wf.run()
        msg = str(ctx.exception)
        self.assertIn("cycle", msg)
        self.assertIn("n1", msg)
        self.assertIn("n2", msg)

    def test_two_links_between_same_pair(self):
        # text feeding two fields of the same node is one topo edge, both overrides apply
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "text", "fields": {"text": "hi"}},
            {"id": "n2", "type": "join", "fields": {"sep": "+"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "text"}, "to": {"node": "n2", "port": "a"}},
            {"id": "l2", "from": {"node": "n1", "port": "text"}, "to": {"node": "n2", "port": "b"}},
        ]})
        self.assertEqual(topo_order(wf.graph), ["n1", "n2"])
        self.assertEqual(wf.run()["Join"], "hi+hi")

    def test_comment_nodes_never_run_but_are_recorded_skipped(self):
        # cross-language parity: comment nodes appear in result.nodes with
        # status 'skipped' (never executed), same as the JS library
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "text", "fields": {"text": "hi"}},
            {"id": "n2", "type": "comment", "fields": {"text": "a note"}},
        ]}, api_key="k")
        result = wf.run()
        self.assertIn("n2", result.nodes)
        self.assertEqual(result.nodes["n2"].status, "skipped")
        self.assertIsNone(result.nodes["n2"].out)
        self.assertIsNone(result.nodes["n2"].error)
        self.assertEqual(result["Text"], "hi")


if __name__ == "__main__":
    unittest.main()
