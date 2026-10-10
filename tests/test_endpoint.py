"""🔌 Custom endpoint. The request goes to the graph's URL, never to NanoGPT."""

import os
import unittest
from unittest import mock

from tests._util import FAST, MockedTest
from tests.harness import binary_response, chat_response

from nanoodle import MediaRef, NanoodleError, RunError
from nanoodle.endpoint import endpoint_fetch_error, endpoint_resolve_target, endpoint_url_ok
from nanoodle.graph import Node


class EndpointUrlTest(unittest.TestCase):
    def test_http_is_local_only_and_credentials_are_refused(self):
        self.assertIs(endpoint_url_ok("https://example.com/v1/chat/completions"), True)
        self.assertIs(endpoint_url_ok("http://127.0.0.1:8787/v1"), True)
        self.assertIs(endpoint_url_ok("http://localhost/x"), True)
        self.assertIs(endpoint_url_ok("http://[::1]:8787/x"), True)
        self.assertIs(endpoint_url_ok("http://10.1.2.3/x"), True)
        self.assertIs(endpoint_url_ok("http://192.168.1.9/x"), True)
        self.assertIs(endpoint_url_ok("http://172.16.0.4/x"), True)
        self.assertIs(endpoint_url_ok("http://169.254.1.1/x"), True)
        self.assertIs(endpoint_url_ok("http://printer.local/x"), True)
        for bad, needle in (
            ("http://example.com/v1", "localhost"),
            ("http://user:pw@127.0.0.1/x", "credentials"),
            ("ftp://127.0.0.1/x", "isn"),
            ("", "URL required"),
        ):
            msg = endpoint_url_ok(bad)
            self.assertIsInstance(msg, str)
            self.assertIn(needle, msg)

    def test_choice_path_joins_the_authored_host(self):
        node = Node("e", "endpoint", {"url": "http://127.0.0.1:8787/v1/chat/completions", "mode": "chat"})
        got = endpoint_resolve_target(node, {"url": "/post"})
        self.assertEqual(got, {"url": "http://127.0.0.1:8787/post", "mode": "chat"})
        got = endpoint_resolve_target(node, {"url": "/post", "mode": "image"})
        self.assertEqual(got["url"], "http://127.0.0.1:8787/post")
        self.assertEqual(got["mode"], "image")
        got = endpoint_resolve_target(node, {"mode": "image"})
        self.assertEqual(got["url"], "http://127.0.0.1:8787/v1/chat/completions")
        self.assertEqual(got["mode"], "image")

    def test_fetch_errors_name_cors_without_leaking_a_procedure(self):
        local = endpoint_fetch_error(NanoodleError("could not reach it"), "http://127.0.0.1:9/x")
        public = endpoint_fetch_error(NanoodleError("could not reach it"), "https://example.com/x")
        self.assertIn("CORS", local)
        self.assertIn("refused", local)
        self.assertIn("blocked by CORS", public)


class EndpointRunTest(MockedTest):
    def test_chat_does_not_send_the_nanogpt_key(self):
        self.mock.script("POST", "/v1/chat/completions", chat_response("pong", cost_usd=9))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NANOGPT_API_KEY", None)
            wf = self.wf_dict({"nodes": [
                {"id": "e", "type": "endpoint", "name": "Hook", "fields": {
                    "url": self.mock.base_url + "/v1/chat/completions",
                    "mode": "chat", "prompt": "ping", "model": "local",
                    "system": "be brief",
                }},
            ]}, api_key="test-key", **FAST)
            result = wf.run()
        req = self.mock.requests[0]
        self.assertEqual(req.path, "/v1/chat/completions")
        self.assertNotIn("x-api-key", req.headers)
        self.assertNotIn("authorization", req.headers)
        self.assertEqual(req.json["model"], "local")
        self.assertEqual(req.json["messages"][0], {"role": "system", "content": "be brief"})
        self.assertEqual(req.json["messages"][1], {"role": "user", "content": "ping"})
        self.assertEqual(result["Hook"], "pong")
        self.assertEqual(result.cost_usd, 0.0)  # the custom server's price is not NanoGPT's
        self.assertTrue(result.cost_exact)
        self.assertEqual(wf.outputs[0].ports, ["text"])

    def test_choice_path_is_posted_and_auth_is_a_bearer(self):
        self.mock.script("POST", "/post", chat_response("joined"))
        wf = self.wf_dict({"nodes": [
            {"id": "ch", "type": "choice", "fields": {"options": "/post\n/other", "selected": "/post"}},
            {"id": "e", "type": "endpoint", "fields": {
                "url": self.mock.base_url + "/v1/chat/completions",
                "mode": "chat", "prompt": "hi", "auth": "sekrit",
            }},
        ], "links": [
            {"id": "l", "from": {"node": "ch", "port": "text"}, "to": {"node": "e", "port": "url"}},
        ]}, api_key="test-key", **FAST)
        # the authored host survives; only the path came from the Choice
        self.assertTrue(wf.graph.node("e").fields["url"].endswith("/v1/chat/completions"))
        self.assertEqual(wf.run()["Custom endpoint"], "joined")
        req = self.mock.requests_to("/post")[0]
        self.assertEqual(req.headers.get("authorization"), "Bearer sekrit")
        self.assertNotIn("x-api-key", req.headers)
        # a wired url is an input port, so the setting is hidden and the field is untouched
        self.assertFalse(any(s.field == "url" and s.node_id == "e" for s in wf.settings))
        self.assertTrue(wf.graph.node("e").fields["url"].endswith("/v1/chat/completions"))

    def test_image_mode_output_port_and_404_hint(self):
        self.mock.script("POST", "/img",
                         {"status": 200, "json": {"data": [{"url": "https://cdn/out.png"}]}})
        wf = self.wf_dict({"nodes": [
            {"id": "e", "type": "endpoint", "fields": {
                "url": self.mock.base_url + "/img", "mode": "image", "prompt": "a cup",
            }},
        ]}, api_key="", **FAST)
        self.assertEqual(wf.outputs[0].ports, ["image"])
        image = wf.run()["Custom endpoint"]
        self.assertIsInstance(image, MediaRef)
        self.assertEqual(image.url, "https://cdn/out.png")
        body = self.mock.requests_to("/img")[0].json
        self.assertEqual(body["prompt"], "a cup")
        self.assertEqual(body["response_format"], "b64_json")

        self.mock.reset()
        self.mock.script("POST", "/missing",
                         {"status": 404, "json": {"error": "no such route"}})
        wf = self.wf_dict({"nodes": [
            {"id": "e", "type": "endpoint", "fields": {
                "url": self.mock.base_url + "/missing", "mode": "json", "prompt": "x",
            }},
        ]}, api_key="", **FAST)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("no such route", str(ctx.exception))
        self.assertIn("check the custom endpoint URL", str(ctx.exception))

    def test_audio_binary_and_json_mode(self):
        self.mock.script("POST", "/wav", binary_response(b"ID3fake", mime="audio/mpeg"))
        wf = self.wf_dict({"nodes": [
            {"id": "e", "type": "endpoint", "fields": {
                "url": self.mock.base_url + "/wav", "mode": "audio", "prompt": "say hi",
            }},
        ]}, api_key="", **FAST)
        self.assertEqual(wf.outputs[0].ports, ["audio"])
        audio = wf.run()["Custom endpoint"]
        self.assertEqual(audio.bytes(), b"ID3fake")
        self.assertEqual(audio.mime, "audio/mpeg")

        self.mock.script("POST", "/j", {"status": 200, "json": {"data": {"ok": True}}})
        wf = self.wf_dict({"nodes": [
            {"id": "t", "type": "text", "fields": {"text": "wired"}},
            {"id": "e", "type": "endpoint", "fields": {
                "url": self.mock.base_url + "/j", "mode": "json",
            }},
        ], "links": [
            {"id": "l", "from": {"node": "t", "port": "text"}, "to": {"node": "e", "port": "text"}},
        ]}, api_key="", **FAST)
        prompt = [s for s in wf.inputs if s.field == "prompt"]
        self.assertEqual(len(prompt), 1)
        self.assertTrue(prompt[0].optional)
        text = wf.run()["Custom endpoint"]
        self.assertIn('"ok": true', text)
        self.assertEqual(self.mock.requests_to("/j")[0].json, {"text": "wired"})

    def test_public_http_is_refused_before_a_request(self):
        wf = self.wf_dict({"nodes": [
            {"id": "e", "type": "endpoint", "fields": {
                "url": "http://example.com/v1", "mode": "chat", "prompt": "hi",
            }},
        ]}, api_key="test-key", **FAST)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("localhost", str(ctx.exception))
        self.assertEqual(self.mock.requests, [])


if __name__ == "__main__":
    unittest.main()
