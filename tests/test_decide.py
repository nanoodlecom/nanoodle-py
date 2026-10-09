"""⚖️ Decide (NanoGPT /api/v1/decisions) — the twin of nanoodle-js decide and the editor node.

Pure question/merge/output logic runs without ffmpeg; the image path (fit_image_jpeg) needs it.
Gate semantics: a closed yes/no gate is status "gated" (not an error), downstream "skipped".
"""

import base64
import json
import shutil
import tempfile
import os
import unittest

from tests._util import MockedTest
from tests.harness import chat_response, error_response
from tests.test_cli import run_cli

from nanoodle import NanoodleError, RunError
from nanoodle.decide import (DECIDE_IMG_FALLBACK, DecideGateClosed, decide_image_limits,
                             decide_merge_orders, decide_question_for, decide_run)
from nanoodle.errors import GatedOutputError

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
       "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def usage(cost):
    return {"input_tokens": 40, "output_tokens": 0, "cost": cost}


def decision(answer, cost):
    return {"status": 200, "json": {"answers": {"answer": answer}, "usage": usage(cost)}}


def gate_graph(gate=True, extra_nodes=(), extra_links=()):
    return {
        "nodes": [
            {"id": "t", "type": "text", "fields": {"text": "a dog"}},
            {"id": "g", "type": "decide", "fields": {"model": "liquid/d1", "mode": "yesno",
                                                     "gate": gate, "question": "Is it a cat?"}},
            {"id": "l", "type": "llm", "fields": {"model": "gpt-x", "prompt": "write a cat poem"}},
        ] + list(extra_nodes),
        "links": [
            {"id": "l1", "from": {"node": "t", "port": "text"}, "to": {"node": "g", "port": "text"}},
            {"id": "l2", "from": {"node": "g", "port": "text"}, "to": {"node": "l", "port": "prompt"}},
        ] + list(extra_links),
    }


class DecidePureTest(unittest.TestCase):
    def test_question_shapes_and_refusals(self):
        p = decide_question_for("pick", {}, 3)
        self.assertEqual(list(p["criteria"]), ["image_1", "image_2", "image_3"])
        self.assertEqual(p["type"], "choice")
        self.assertIn("best matches", p["instructions"])
        with self.assertRaisesRegex(NanoodleError, "at least two images"):
            decide_question_for("pick", {}, 1)
        self.assertEqual(list(decide_question_for("choose", {"options": "a\n b \n\na"}, 0)["criteria"]),
                         ["a", "b"])
        with self.assertRaisesRegex(NanoodleError, "two labels"):
            decide_question_for("choose", {"options": "only"}, 0)
        self.assertEqual(decide_question_for("score", {}, 0)["criteria"], ["poor", "okay", "good", "great"])
        with self.assertRaisesRegex(NanoodleError, "2 to 10"):
            decide_question_for("score", {"levels": "\n".join("abcdefghijk")}, 0)
        self.assertEqual(decide_question_for("yesno", {"question": "Cat?"}, 0)["type"], "noul")
        self.assertEqual(decide_image_limits(None, False), DECIDE_IMG_FALLBACK)
        self.assertIsNone(decide_image_limits({"image_input": False}, True))
        self.assertEqual(decide_image_limits(
            {"image_input": True, "image_limits": {"maxImages": 2}}, True)["maxImages"], 2)

    def test_merge_maps_reversed_slots_back_and_sums_cost(self):
        m = decide_merge_orders([
            {"answers": {"answer": {"probabilities": {"image_1": 0.45, "image_2": 0.05, "image_3": 0.5}}}, "usage": usage(1)},
            {"answers": {"answer": {"probabilities": {"image_1": 0.8, "image_2": 0.05, "image_3": 0.15}}}, "usage": usage(2)},
        ], [[0, 1, 2], [2, 1, 0]])
        a = m["answers"]["answer"]
        self.assertEqual(a["choice"], "image_3")
        self.assertAlmostEqual(a["probabilities"]["image_3"], 0.65)
        self.assertAlmostEqual(a["probabilities"]["image_1"], 0.3)
        self.assertEqual(m["usage"]["cost"], 3)

    def test_decide_run_pick_two_requests_winner_out(self):
        sent = []

        def send(body):
            sent.append(body)
            probs = ({"image_1": 0.45, "image_2": 0.05, "image_3": 0.5} if len(sent) == 1
                     else {"image_1": 0.8, "image_2": 0.05, "image_3": 0.15})
            return {"answers": {"answer": {"type": "choice", "choice": "image_1", "probabilities": probs}},
                    "usage": usage(0.00001)}
        imgs = ["data:a", "https://x/b.png", "data:c"]
        out = decide_run({"mode": "pick", "question": "best?"}, "pplx", "brief", imgs, DECIDE_IMG_FALLBACK,
                         send, lambda u, d, b: "data:image/jpeg;base64,QUJD")
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[1]["state"][0], "brief")
        self.assertEqual(sent[1]["state"][1], "image_1:")
        self.assertEqual(out["text"], "image 3")
        self.assertEqual(out["image"], "data:c")
        self.assertAlmostEqual(out["decision"]["cost"], 0.00002)
        self.assertTrue(out["decision"]["rows"][2]["win"])

    def test_outputs_score_and_refusals(self):
        out = decide_run({"mode": "score"}, "m", "an essay", [], None,
                         lambda b: {"answers": {"answer": {"type": "score", "score": 2.78,
                                                           "probabilities": {"0": 0, "1": 0.03, "2": 0.16, "3": 0.81}}},
                                    "usage": usage(0)}, None)
        self.assertEqual(out["text"], "3.78")
        self.assertTrue(out["decision"]["rows"][3]["win"])
        with self.assertRaisesRegex(NanoodleError, "can.t see images"):
            decide_run({"mode": "pick"}, "d1", "", ["data:a", "data:b"], None, None, None)
        with self.assertRaisesRegex(NanoodleError, "at most 2 images"):
            decide_run({"mode": "pick"}, "m", "", ["a", "b", "c"],
                       {"maxImages": 2, "maxDimension": 512, "maxEncodedBytes": 240000}, None, None)
        with self.assertRaisesRegex(NanoodleError, "nothing to judge"):
            decide_run({"mode": "yesno"}, "m", "", [], None, None, None)

    def test_gate_raises_a_gate_not_a_failure(self):
        with self.assertRaises(DecideGateClosed) as cm:
            decide_run({"mode": "yesno", "gate": "true", "question": "Cat?"}, "m", "a dog", [], None,
                       lambda b: {"answers": {"answer": {"type": "noul", "noul": 0.2}}, "usage": usage(0)}, None)
        self.assertEqual(cm.exception.code, "decide-gate")
        self.assertTrue(cm.exception.gate)
        self.assertEqual(cm.exception.out["text"], "no")
        self.assertIn("gate closed", str(cm.exception))
        out = decide_run({"mode": "yesno", "question": "Cat?"}, "m", "a dog", [], None,
                         lambda b: {"answers": {"answer": {"type": "noul", "noul": 0.1}}, "usage": usage(0)}, None)
        self.assertEqual(out["text"], "no")


class DecideRunTest(MockedTest):
    def test_choose_exact_body_label_out_cost_billed(self):
        self.mock.script("POST", "/api/v1/decisions", decision(
            {"type": "choice", "choice": "billing", "confidence": 0.4,
             "probabilities": {"billing": 0.44, "shipping": 0.39, "sales": 0.17}}, 0.0000023))
        wf = self.wf_dict({"nodes": [
            {"id": "t", "type": "text", "fields": {"text": "I was charged twice"}},
            {"id": "d", "type": "decide", "fields": {"model": "liquid/d1", "mode": "choose",
                                                     "question": "Team?", "options": "billing\nshipping\nsales"}},
        ], "links": [{"id": "l1", "from": {"node": "t", "port": "text"}, "to": {"node": "d", "port": "text"}}]})
        result = wf.run()
        req = self.mock.requests_to("/api/v1/decisions")
        self.assertEqual(len(req), 1)
        self.assertEqual(req[0].json, {
            "model": "liquid/d1", "state": "I was charged twice",
            "questions": {"answer": {"type": "choice", "instructions": "Team?",
                                     "criteria": {"billing": None, "shipping": None, "sales": None}}}})
        self.assertEqual(result["Decide"], "billing")
        self.assertAlmostEqual(result.cost_usd, 0.0000023)
        self.assertTrue(result.cost_exact)
        self.assertEqual(result.nodes["d"].status, "done")
        self.assertEqual(result.gated, [])

    def test_closed_gate_is_gated_not_failed_and_downstream_skips_unbilled(self):
        self.mock.script("POST", "/api/v1/decisions", decision({"type": "noul", "noul": 0.1}, 0.000001))
        events = []
        result = self.wf_dict(gate_graph()).run(on_progress=events.append)   # no RunError
        self.assertEqual(self.mock.requests_to("/api/v1/chat/completions"), [])
        g, l = result.nodes["g"], result.nodes["l"]
        self.assertEqual(g.status, "gated")
        self.assertIsNone(g.error)
        self.assertEqual(g.out["text"], "no")
        self.assertAlmostEqual(g.gate["yes"], 0.1)
        self.assertIn("gate closed", g.gate["message"])
        self.assertAlmostEqual(g.cost_usd, 0.000001)
        self.assertEqual(l.status, "skipped")
        self.assertEqual(l.gated_by, "g")
        self.assertIsNone(l.cost_usd)
        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.gated), 1)
        self.assertEqual(result.gated[0]["node_id"], "g")
        self.assertEqual(result.gated[0]["name"], "Decide")
        self.assertEqual(result.gated[0]["skipped"], ["l"])
        self.assertAlmostEqual(result.cost_usd, 0.000001)
        self.assertNotIn("LLM", result.outputs)
        self.assertIsNone(result.get("LLM"))
        with self.assertRaises(GatedOutputError) as cm:
            result["LLM"]
        self.assertIn("skipped — the gate 'Decide' answered no", str(cm.exception))
        self.assertTrue(any(e["type"] == "node-gated" and e["node_id"] == "g" for e in events))
        self.assertTrue(any(e["type"] == "node-skipped" and e["gated_by"] == "g" for e in events))
        self.assertFalse(any(e["type"] == "node-error" for e in events))

    def test_skips_cascade_name_the_gate_other_lanes_run_open_gate_runs(self):
        self.mock.script("POST", "/api/v1/decisions", decision({"type": "noul", "noul": 0.2}, 0.000001))
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("poem"))
        extra = [{"id": "j", "type": "join", "fields": {}},
                 {"id": "s", "type": "decide", "fields": {"model": "liquid/d1", "mode": "yesno", "question": "Side?"}}]
        links = [{"id": "l3", "from": {"node": "l", "port": "text"}, "to": {"node": "j", "port": "a"}}]
        r = self.wf_dict(gate_graph(True, extra, links)).run()
        self.assertEqual(r.nodes["j"].status, "skipped")
        self.assertEqual(r.nodes["j"].gated_by, "g")
        self.assertEqual(sorted(r.gated[0]["skipped"]), ["j", "l"])
        self.assertEqual(r.nodes["s"].status, "done")
        self.assertEqual(r["s"], "no")
        opened = self.wf_dict(gate_graph(False, extra, links)).run()
        self.assertEqual(opened.nodes["g"].status, "done")
        self.assertEqual(opened.nodes["l"].status, "done")
        self.assertEqual(opened.gated, [])

    def test_gate_plus_real_failure_is_an_error(self):
        self.mock.script("POST", "/api/v1/decisions", decision({"type": "noul", "noul": 0.2}, 0.000001))
        self.mock.script("POST", "/api/v1/chat/completions", error_response(500, b'{"error":"boom"}'))
        g = {"nodes": [
            {"id": "g", "type": "decide", "fields": {"model": "liquid/d1", "mode": "yesno", "gate": "true", "question": "ok?"}},
            {"id": "bad", "type": "llm", "fields": {"model": "gpt-x", "prompt": "x"}},
            {"id": "j", "type": "join", "fields": {}},
        ], "links": [
            {"id": "l1", "from": {"node": "g", "port": "text"}, "to": {"node": "j", "port": "a"}},
            {"id": "l2", "from": {"node": "bad", "port": "text"}, "to": {"node": "j", "port": "b"}},
        ]}
        with self.assertRaises(RunError) as cm:
            self.wf_dict(g).run()
        self.assertEqual(cm.exception.result.nodes["j"].status, "error")
        self.assertEqual(cm.exception.result.nodes["g"].status, "gated")

    def test_text_only_model_refuses_images_before_any_request(self):
        catalog = {"chat": [{"id": "liquid/d1", "decision_input": {"image_input": False}}]}
        g = {"nodes": [
            {"id": "a", "type": "upload", "fields": {"image": PNG}},
            {"id": "b", "type": "upload", "fields": {"image": PNG}},
            {"id": "d", "type": "decide", "fields": {"model": "liquid/d1"}},
        ], "links": [
            {"id": "l1", "from": {"node": "a", "port": "image"}, "to": {"node": "d", "port": "img1"}},
            {"id": "l2", "from": {"node": "b", "port": "image"}, "to": {"node": "d", "port": "img2"}},
        ]}
        with self.assertRaises(RunError) as cm:
            self.wf_dict(g, catalog=catalog).run()
        self.assertIn("can’t see images", cm.exception.result.nodes["d"].error)
        self.assertEqual(self.mock.requests_to("/api/v1/decisions"), [])

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg not on PATH")
    def test_pick_shrinks_wired_images_to_jpeg_within_budget(self):
        self.mock.script("POST", "/api/v1/decisions", [   # forward, then reversed (slot 1 = orig 2)
            decision({"type": "choice", "probabilities": {"image_1": 0.3, "image_2": 0.7}}, 0.00001),
            decision({"type": "choice", "probabilities": {"image_1": 0.4, "image_2": 0.6}}, 0.00001)])
        g = {"nodes": [
            {"id": "a", "type": "upload", "fields": {"image": PNG}},
            {"id": "b", "type": "upload", "fields": {"image": PNG}},
            {"id": "d", "type": "decide", "fields": {"model": "perplexity/pplx-decider-v1.1-27b"}},
        ], "links": [
            {"id": "l1", "from": {"node": "a", "port": "image"}, "to": {"node": "d", "port": "img1"}},
            {"id": "l2", "from": {"node": "b", "port": "image"}, "to": {"node": "d", "port": "img2"}},
        ]}
        result = self.wf_dict(g).run()
        reqs = self.mock.requests_to("/api/v1/decisions")
        self.assertEqual(len(reqs), 2, "pick asks in order + reversed")
        for r in reqs:
            parts = [p for p in r.json["state"] if isinstance(p, dict)]
            self.assertEqual(len(parts), 2)
            for p in parts:
                self.assertTrue(p["image_url"]["url"].startswith("data:image/jpeg;base64,"))
                self.assertLessEqual(len(p["image_url"]["url"]), int(240000 * 0.95 // 2))
        self.assertEqual(result["Decide"], "image 2")   # orig 1: (0.3+0.6)/2, orig 2: (0.7+0.4)/2
        self.assertAlmostEqual(result.cost_usd, 0.00002)
        self.assertEqual(result.nodes["d"].out["image"].url, PNG)


class DecideCliTest(MockedTest):
    def test_closed_gate_exits_0_and_reports_gated(self):
        self.mock.script("POST", "/api/v1/decisions", decision({"type": "noul", "noul": 0.12}, 0.000003))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gate.json")
            with open(path, "w") as f:
                json.dump(gate_graph(), f)
            code, out, err = run_cli(["run", path, "--api-key", "k", "--base-url",
                                      self.mock.base_url, "--json"])
            self.assertEqual(code, 0, err)
            s = json.loads(out)
            self.assertEqual(s["errors"], [])
            self.assertEqual(s["gated"][0]["node_id"], "g")
            self.assertEqual(s["gated"][0]["skipped"], ["l"])
            self.assertEqual(s["nodes"]["g"]["status"], "gated")
            self.assertEqual(s["nodes"]["g"]["gate"]["yes"], 0.12)
            self.assertEqual(s["nodes"]["l"]["status"], "skipped")
            self.assertEqual(s["nodes"]["l"]["gatedBy"], "g")
            self.assertIsNone(s["outputs"]["LLM"])
            self.assertAlmostEqual(s["costUsd"], 0.000003)
            code, out, err = run_cli(["run", path, "--api-key", "k", "--base-url", self.mock.base_url])
            self.assertEqual(code, 0, err)
            self.assertIn("⛔ Decide (g) gated: gate closed", err)
            self.assertIn("⤼ LLM (l) skipped — gate g said no", err)
            self.assertIn("LLM: (skipped — the gate 'Decide' answered no)", out)
            self.assertIn("cost: $0.000003", err)


if __name__ == "__main__":
    unittest.main()
