"""CLI: python -m nanoodle run|inspect — flags, --json shape, media saving,
@file inputs, error exit codes. All against the mock harness."""

import base64
import contextlib
import io
import json
import os
import struct
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from tests import fixture
from tests._util import MockedTest
from tests.harness import chat_response, image_response

from nanoodle.__main__ import main

PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
           "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class SpriteFilesTest(unittest.TestCase):
    def test_at_file_slices_gltf_and_binary_ply_and_preserves_upload_extension(self):
        graph = json.loads(Path(fixture("sprite-stack.json")).read_text())
        ply = base64.b64decode(graph["nodes"][0]["fields"]["model"].partition(",")[2])
        graph["nodes"][0]["fields"] = {}
        graph["nodes"][2]["fields"]["turntable"] = False

        # Unpack the fixture GLB into a self-contained JSON glTF with its original animation.
        glb = Path(fixture("media/sprite-knight.glb")).read_bytes()
        size = struct.unpack_from("<I", glb, 12)[0]
        model = json.loads(glb[20:20 + size])
        model["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(glb[28 + size:]).decode()
        gltf = json.dumps(model).encode()

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            saved = root / "graph.json"
            for name, data, mime in (("knight.GLTF", gltf, "model/gltf+json"), ("cloud.PLY", ply, "model/ply")):
                with self.subTest(name=name):
                    mesh = root / name
                    mesh.write_bytes(data)
                    graph["nodes"][1]["fields"]["frames"] = 2 if name.endswith("GLTF") else 1
                    saved.write_text(json.dumps(graph))
                    code, out, err = run_cli(["run", str(saved), "--input", "m.model=@" + str(mesh), "--json"])
                    self.assertEqual(code, 0, err or out)
                    self.assertEqual(json.loads(out)["nodes"]["r"]["status"], "done")

                    saved.write_text(json.dumps({"nodes": [dict(graph["nodes"][0], name="Model")]}))
                    code, out, err = run_cli(["run", str(saved), "--input", "m.model=@" + str(mesh),
                                              "--out", d, "--json"])
                    self.assertEqual(code, 0, err or out)
                    self.assertEqual((root / ("Model" + mesh.suffix.lower())).read_bytes(), data)
                    self.assertEqual(json.loads(out)["outputs"]["Model"]["mime"], mime)

    def test_at_file_reads_sprite_palettes_and_animated_sheet(self):
        from nanoodle.sprite_codecs import encode_png
        from tests.test_sprite_bmp import icon

        graph = json.loads(Path(fixture("sprite-stack.json")).read_text())
        graph["nodes"].append({"id": "p", "type": "upload", "fields": {}})
        graph["nodes"][1]["fields"]["palette"] = 2
        graph["nodes"][2]["fields"]["turntable"] = False
        graph["links"].append({"from": {"node": "p", "port": "image"},
                               "to": {"node": "s", "port": "paletteImage"}})
        ico = icon([(1, 1, 32, encode_png(1, 1, bytes((235, 77, 41, 255))))])
        cur = b"\0\0\x02\0" + ico[4:]
        avif = Path(fixture("media/sprite-alpha.avif")).read_bytes()

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            saved = root / "graph.json"
            saved.write_text(json.dumps(graph))
            for name, data in (("palette.AVIF", avif), ("palette.ico", ico), ("palette.cur", cur)):
                with self.subTest(name=name):
                    path = root / name
                    path.write_bytes(data)
                    code, out, err = run_cli(["run", str(saved), "--input", "p.image=@" + str(path), "--json"])
                    self.assertEqual(code, 0, err or out)
                    self.assertEqual(json.loads(out)["nodes"]["s"]["status"], "done")

            saved.write_text(json.dumps({
                "nodes": [graph["nodes"][-1], graph["nodes"][2]],
                "links": [{"from": {"node": "p", "port": "image"}, "to": {"node": "r", "port": "sheet"}}],
            }))
            code, out, err = run_cli(["run", str(saved), "--input",
                                      "p.image=@" + fixture("media/sprite-sequence.apng"), "--json"])
            self.assertEqual(code, 0, err or out)
            self.assertEqual(json.loads(out)["nodes"]["r"]["status"], "done")



class InspectTest(unittest.TestCase):
    def test_inspect_prints_inputs_outputs_settings_nodes(self):
        code, out, err = run_cli(["inspect", fixture("starter-graph.json")])
        self.assertEqual(code, 0)
        self.assertIn("Inputs:", out)
        self.assertIn("Text", out)
        self.assertIn("System prompt", out)
        self.assertIn("Outputs:", out)
        self.assertIn("Image", out)
        self.assertIn("Settings:", out)
        self.assertIn("n2.model", out)
        self.assertIn("Nodes:", out)
        self.assertIn("n3", out)


class EnvFileTest(MockedTest):
    def _write_env(self, d, body):
        path = os.path.join(d, ".env")
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        return path

    def test_run_reads_key_from_env_file(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("a vivid prompt"))
        self.mock.script("POST", "/v1/images/generations", image_response(b64_list=[PNG_B64]))
        with tempfile.TemporaryDirectory() as d:
            env_path = self._write_env(d, "# comment\nNANOGPT_API_KEY=key-from-file\n")
            with unittest.mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NANOGPT_API_KEY", None)
                code, _, _ = run_cli(["run", fixture("starter-graph.json"),
                                      "--base-url", self.mock.base_url,
                                      "--env-file", env_path,
                                      "--input", "Text=x", "--json"])
        self.assertEqual(code, 0)
        chat = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(chat.headers.get("authorization"), "Bearer key-from-file")

    def test_existing_env_var_wins_over_env_file(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("a vivid prompt"))
        self.mock.script("POST", "/v1/images/generations", image_response(b64_list=[PNG_B64]))
        with tempfile.TemporaryDirectory() as d:
            env_path = self._write_env(d, 'NANOGPT_API_KEY="file-key"\n')
            with unittest.mock.patch.dict(os.environ, {"NANOGPT_API_KEY": "env-key"}):
                code, _, _ = run_cli(["run", fixture("starter-graph.json"),
                                      "--base-url", self.mock.base_url,
                                      "--env-file", env_path,
                                      "--input", "Text=x", "--json"])
        self.assertEqual(code, 0)
        chat = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(chat.headers.get("authorization"), "Bearer env-key")

    def test_missing_env_file_exits_1_before_any_call(self):
        code, _, err = run_cli(["run", fixture("starter-graph.json"),
                                "--base-url", self.mock.base_url,
                                "--env-file", "/nonexistent/.env",
                                "--input", "Text=x"])
        self.assertEqual(code, 1)
        self.assertIn("env-file", err)
        self.assertEqual(self.mock.requests, [])

    def test_inspect_accepts_env_file(self):
        with tempfile.TemporaryDirectory() as d:
            env_path = self._write_env(d, "export NANOGPT_API_KEY=inspect-key\n")
            with unittest.mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NANOGPT_API_KEY", None)
                code, out, _ = run_cli(["inspect", fixture("starter-graph.json"),
                                        "--env-file", env_path])
                self.assertEqual(os.environ.get("NANOGPT_API_KEY"), "inspect-key")
        self.assertEqual(code, 0)
        self.assertIn("Inputs:", out)


class RunCliTest(MockedTest):
    def _script_ok(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("a vivid prompt"))
        self.mock.script("POST", "/v1/images/generations", image_response(b64_list=[PNG_B64]))

    def _argv(self, *extra):
        return ["run", fixture("starter-graph.json"),
                "--api-key", "cli-key", "--base-url", self.mock.base_url] + list(extra)

    def test_run_json_output_shape(self):
        self._script_ok()
        code, out, err = run_cli(self._argv("--input", "Text=hello", "--json"))
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["outputs"]["Image"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(payload["nodes"]["n3"]["status"], "done")
        self.assertEqual(payload["errors"], [])
        chat = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(chat.json["messages"][1]["content"], "hello")
        self.assertEqual(chat.headers.get("authorization"), "Bearer cli-key")

    def test_malformed_wires_are_inspectable_and_stop_run_before_paid_work(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("ok"))
        data = {
            "nodes": [
                {"id": "source", "type": "text", "fields": {"text": "upstream"}},
                {"id": "sink", "type": "llm", "fields": {"model": "m", "prompt": "stored fallback"}},
            ],
        }
        cases = [
            ("source", "obsoletePort", "sink", "prompt", 'undeclared source port "obsoletePort"'),
            ("source", "text", "sink", "promtp", 'undeclared target port "promtp"'),
            ("absent", "text", "sink", "prompt", 'missing source node "absent"'),
            ("source", "text", "absent", "prompt", 'missing target node "absent"'),
        ]
        with tempfile.TemporaryDirectory() as d:
            saved = Path(d) / "graph.json"
            argv = ["run", str(saved), "--api-key", "cli-key",
                    "--base-url", self.mock.base_url, "--json"]
            for source, from_port, target, to_port, issue in cases:
                with self.subTest(issue=issue):
                    data["links"] = [{"id": "wire", "from": {"node": source, "port": from_port},
                                      "to": {"node": target, "port": to_port}}]
                    saved.write_text(json.dumps(data), encoding="utf-8")
                    code, _, err = run_cli(["inspect", str(saved)])
                    self.assertEqual(code, 0)
                    self.assertIn(issue, err)
                    code, out, err = run_cli(argv)
                    self.assertEqual(code, 1)
                    self.assertIn(issue, err)
                    payload = json.loads(out)
                    self.assertIn(issue, payload["errors"][0]["message"])
                    self.assertIn(f"{source}.{from_port} → {target}.{to_port}",
                                  payload["errors"][0]["message"])
                    self.assertEqual(payload["nodes"], {})
                    self.assertIsNone(payload["outputs"]["LLM"])
                    self.assertEqual(payload["costUsd"], 0.0)
                    self.assertEqual(self.mock.requests, [])

            data["links"] = [{"id": "wire", "from": {"node": "source", "port": "text"},
                              "to": {"node": "sink", "port": "prompt"}}]
            saved.write_text(json.dumps(data), encoding="utf-8")
            code, out, err = run_cli(argv)
        self.assertEqual(code, 0, err or out)
        req, = self.mock.requests_to("/api/v1/chat/completions")
        self.assertEqual(req.json["messages"][-1]["content"], "upstream")

    def test_run_set_overrides_setting(self):
        self._script_ok()
        code, _, _ = run_cli(self._argv("--input", "Text=x", "--set", "n3.size=2k", "--json"))
        self.assertEqual(code, 0)
        img = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(img.json["size"], "2k")

    def test_competing_prompt_wires_fail_before_paid_work_in_either_order(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("ok"))
        data = {"nodes": [
            {"id": "first", "type": "text", "fields": {"text": "first"}},
            {"id": "second", "type": "text", "fields": {"text": "second"}},
            {"id": "sink", "type": "llm", "fields": {"model": "m", "prompt": "stored fallback"}},
        ]}
        left = {"id": "left", "from": {"node": "first", "port": "text"},
                "to": {"node": "sink", "port": "prompt"}}
        right = {"id": "right", "from": {"node": "second", "port": "text"},
                 "to": {"node": "sink", "port": "prompt"}}
        issue = ("input sink.prompt has competing wires from first.text, second.text. "
                 "Leave one active wire for this input.")
        with tempfile.TemporaryDirectory() as d:
            saved = Path(d) / "graph.json"
            argv = ["run", str(saved), "--api-key", "cli-key",
                    "--base-url", self.mock.base_url, "--json"]
            for links in ([left, right], [right, left]):
                data["links"] = links
                saved.write_text(json.dumps(data), encoding="utf-8")
                code, _, err = run_cli(["inspect", str(saved)])
                self.assertEqual(code, 0)
                self.assertIn(issue, err)
                code, out, err = run_cli(argv)
                self.assertEqual(code, 1)
                self.assertIn(issue, err)
                result = json.loads(out)
                self.assertEqual(result["errors"][0]["message"], issue)
                self.assertEqual(result["nodes"], {})
                self.assertEqual(result["costUsd"], 0.0)
                self.assertEqual(self.mock.requests, [])

            data["links"] = [left, dict(left, id="copy")]
            saved.write_text(json.dumps(data), encoding="utf-8")
            code, out, err = run_cli(argv)
        self.assertEqual(code, 0, err or out)
        req, = self.mock.requests_to("/api/v1/chat/completions")
        self.assertEqual(req.json["messages"][-1]["content"], "first")

    def test_run_out_dir_saves_media(self):
        self._script_ok()
        with tempfile.TemporaryDirectory() as d:
            out_dir = os.path.join(d, "out")
            code, out, _ = run_cli(self._argv("--input", "Text=x", "--out", out_dir, "--json"))
            self.assertEqual(code, 0)
            payload = json.loads(out)
            path = payload["outputs"]["Image"]["file"]
            self.assertTrue(path.endswith("Image.png"))
            with open(path, "rb") as f:
                self.assertTrue(f.read().startswith(b"\x89PNG"))

    def test_run_at_file_input(self):
        self._script_ok()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "idea.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("from a file")
            code, _, _ = run_cli(self._argv("--input", "Text=@" + p, "--json"))
            self.assertEqual(code, 0)
            chat = self.mock.requests_to("/api/v1/chat/completions")[0]
            self.assertEqual(chat.json["messages"][1]["content"], "from a file")

    def test_bad_input_syntax_exits_1(self):
        code, _, err = run_cli(self._argv("--input", "no-equals-sign"))
        self.assertEqual(code, 1)
        self.assertIn("error:", err)
        self.assertEqual(self.mock.requests, [])

    def test_unknown_input_key_exits_1_and_lists_keys(self):
        code, _, err = run_cli(self._argv("--input", "Bogus=x"))
        self.assertEqual(code, 1)
        self.assertIn("Bogus", err)
        self.assertIn("Text", err)
        self.assertEqual(self.mock.requests, [])

    def test_plain_run_prints_outputs_and_progress(self):
        self._script_ok()
        code, out, err = run_cli(self._argv("--input", "Text=x"))
        self.assertEqual(code, 0)
        self.assertIn("Image:", out)
        self.assertIn("cost:", err)
        self.assertIn("LLM", err)   # progress lines on stderr

    def test_plain_run_prints_balance_when_api_reports_it(self):
        # image_response with balance → remaining_balance on result → stderr cost line
        self.mock.script("POST", "/api/v1/chat/completions",
                         chat_response("a vivid prompt", cost_usd=0.001, balance=9.5))
        self.mock.script("POST", "/v1/images/generations",
                         image_response(b64_list=[PNG_B64], cost=0.02, balance=9.48))
        code, _, err = run_cli(self._argv("--input", "Text=x"))
        self.assertEqual(code, 0)
        self.assertIn("cost:", err)
        self.assertIn("balance:", err)
        self.assertIn("9.48", err)


class FailedRunJsonTest(MockedTest):
    """A failed run must still answer --json. An agent caller reads the per-node status,
    the per-node error, the partial outputs and the cost already spent from stdout."""

    def _argv(self, *extra):
        return ["run", fixture("partial-failure.json"),
                "--api-key", "cli-key", "--base-url", self.mock.base_url] + list(extra)

    def _script_one_lane_fails(self):
        # the llm lane completes and costs money; the image lane fails
        self.mock.script("POST", "/api/v1/chat/completions",
                         chat_response("some notes", cost_usd=0.002, balance=9.5))
        self.mock.script("POST", "/v1/images/generations",
                         {"status": 400, "json": {"error": "Invalid image input."}})

    def test_failed_run_with_json_prints_the_same_shape_and_exits_1(self):
        self._script_one_lane_fails()
        code, out, err = run_cli(self._argv("--json"))
        self.assertEqual(code, 1)
        payload = json.loads(out)
        # per-node status and error map
        self.assertEqual(payload["nodes"]["n2"]["status"], "done")
        self.assertEqual(payload["nodes"]["n3"]["status"], "error")
        self.assertIn("Invalid image input", payload["nodes"]["n3"]["error"])
        self.assertEqual([e["node_id"] for e in payload["errors"]], ["n3"])
        # partial outputs: the lane that finished keeps its value, the failed one is null
        self.assertEqual(payload["outputs"]["Notes"], "some notes")
        self.assertIsNone(payload["outputs"]["Picture"])
        # the money already spent is reported, not lost with the exception
        self.assertEqual(payload["costUsd"], 0.002)
        self.assertEqual(payload["remainingBalance"], 9.5)
        self.assertIn("run failed:", err)

    def test_failed_run_with_json_still_saves_the_partial_media(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("some notes"))
        self.mock.script("POST", "/v1/images/generations",
                         {"status": 400, "json": {"error": "Invalid image input."}})
        with tempfile.TemporaryDirectory() as d:
            out_dir = os.path.join(d, "out")
            code, out, _ = run_cli(self._argv("--json", "--out", out_dir))
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertEqual(payload["outputs"]["Notes"], "some notes")

    def test_failed_run_without_json_keeps_the_plain_error_path(self):
        self._script_one_lane_fails()
        code, out, err = run_cli(self._argv())
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("error: output node", err)


class PreRunFailureJsonTest(MockedTest):
    """--json must answer the failures that happen BEFORE the first node runs.

    A missing required input is the commonest way an agent's call fails, and it is caught
    during validation, so there is no RunResult and RunError is never raised. The CLI used
    to print nothing at all on stdout for it: exit 1, empty stdout, one line on stderr. The
    docs promised JSON. An agent caller needs it most exactly here.
    """

    def _argv(self, graph, *extra):
        return ["run", fixture(graph), "--api-key", "cli-key",
                "--base-url", self.mock.base_url] + list(extra)

    def test_missing_required_input_prints_json_and_exits_1(self):
        code, out, err = run_cli(self._argv("needs-input.json", "--json"))
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertEqual([e["message"] for e in payload["errors"]],
                         ["missing required input: Answer"])
        self.assertIsNone(payload["errors"][0]["node_id"], "no node ran, so none is named")
        self.assertEqual(payload["nodes"], {}, "nothing executed")
        self.assertEqual(payload["costUsd"], 0.0, "and nothing was spent")
        self.assertEqual(payload["outputs"], {"Answer": None},
                         "the interface is still described, with no values")
        self.assertEqual(payload["promptTrims"], [])
        self.assertEqual(payload["payments"], [], "no node ran, so no deposit exists")
        self.assertIn("error: missing required input", err)
        self.assertEqual(self.mock.requests, [], "the API was never called")

    def test_unknown_input_key_prints_json_and_exits_1(self):
        code, out, _ = run_cli(self._argv("needs-input.json", "--json",
                                          "--input", "Bogus=x"))
        self.assertEqual(code, 1)
        self.assertIn("Bogus", json.loads(out)["errors"][0]["message"])

    def test_a_graph_that_cannot_be_read_still_answers_json(self):
        code, out, err = run_cli(["run", os.path.join(tempfile.gettempdir(), "no-such-graph.json"),
                                  "--json", "--api-key", "k"])
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertIn("no-such-graph.json", payload["errors"][0]["message"])
        self.assertEqual(payload["outputs"], {}, "no interface is known, so none is described")

    def test_the_human_path_is_unchanged(self):
        code, out, err = run_cli(self._argv("needs-input.json"))
        self.assertEqual(code, 1)
        self.assertEqual(out, "", "no JSON leaks into the plain path")
        self.assertIn("error: missing required input: Answer", err)


if __name__ == "__main__":
    unittest.main()
