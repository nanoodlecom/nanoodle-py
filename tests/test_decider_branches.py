"""Run the shared conditional graph against a local NanoGPT stub."""
import json
from pathlib import Path

from nanoodle import RunError
from tests._util import MockedTest
from tests.harness import chat_response
from tests.test_decide import decision

FIXTURE = Path(__file__).parent / "fixtures" / "decider-branches.json"


class BranchTest(MockedTest):
    def test_conditional_graph_and_retry(self):
        graph = json.loads(FIXTURE.read_text())
        cases = graph.pop("cases")
        graph.pop("legacy")
        workflow = self.wf_dict(json.loads(json.dumps(graph)))
        models = {node["fields"]["model"]: node["id"]
                  for node in graph["nodes"] if node["type"] == "llm"}
        for scenario in cases:
            with self.subTest(scenario=scenario["name"]):
                self.mock.reset()
                self.mock.script("POST", "/api/v1/decisions", [
                    decision(None if value is None else {"type": "noul", "noul": value}, 0.001)
                    for value in scenario["answers"]
                ])
                chat_ids = [nid for nid in scenario["calls"] if nid in models.values()]
                self.mock.script("POST", "/api/v1/chat/completions", [
                    chat_response(nid, cost_usd=0.001) for nid in chat_ids
                ])
                if scenario.get("error"):
                    with self.assertRaises(RunError) as caught:
                        workflow.run()
                    result = caught.exception.result
                    failed_decider = "n5" if len(scenario["answers"]) > 1 else "n2"
                    self.assertIn("valid yes/no answer", result.nodes[failed_decider].error)
                else:
                    result = workflow.run()
                    self.assertEqual(result.get("Answer"), scenario["output"])
                    for nid in scenario["skipped"]:
                        self.assertEqual(result.nodes[nid].status, "skipped")
                        self.assertEqual(result.nodes[nid].skip_reason, "branch")
                        self.assertIsNone(result.nodes[nid].cost_usd)
                decisions = self.mock.requests_to("/api/v1/decisions")
                chat = self.mock.requests_to("/api/v1/chat/completions")
                calls = ["n2", "n5"][:len(decisions)] + [models[r.json["model"]] for r in chat]
                self.assertCountEqual(calls, scenario["calls"])
                self.assertEqual(len(self.mock.requests), len(calls))
                self.assertAlmostEqual(result.cost_usd, 0.001 * len(calls))
                self.assertTrue(result.cost_exact)

    def test_legacy_gate_round_trip(self):
        legacy = json.loads(FIXTURE.read_text())["legacy"]
        self.mock.script("POST", "/api/v1/decisions", decision({"type": "noul", "noul": 0}, 0.001))
        workflow = self.wf_dict(json.loads(json.dumps(legacy)))
        result = workflow.run()
        self.assertNotIn("branch", workflow.graph.node("n2").fields)
        self.assertEqual(result.nodes["n2"].status, "gated")
        self.assertEqual(result.nodes["n4"].gated_by, "n2")
        self.assertEqual(len(self.mock.requests), 1)
        self.assertAlmostEqual(result.cost_usd, 0.001)
