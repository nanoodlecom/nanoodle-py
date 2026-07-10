"""CLI: python -m nanoodle run|inspect — flags, --json shape, media saving,
@file inputs, error exit codes. All against the mock harness."""

import contextlib
import io
import json
import os
import tempfile
import unittest

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

    def test_run_set_overrides_setting(self):
        self._script_ok()
        code, _, _ = run_cli(self._argv("--input", "Text=x", "--set", "n3.size=2k", "--json"))
        self.assertEqual(code, 0)
        img = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(img.json["size"], "2k")

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


if __name__ == "__main__":
    unittest.main()
