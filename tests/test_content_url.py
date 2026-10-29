import json
import unittest

from nanoodle import NanoodleError, RunError
from nanoodle.engine import Engine
from nanoodle.media import MEDIA_INLINE_MAX
from nanoodle.transport import HttpResponse, default_http
from tests._util import FAST, MockedTest
from tests.harness import MockNanoGPT, video_status

RELATIVE = "/api/generate-video/content?runId=grok1&model=grok-imagine-video-1.5-lite"
SIGNED = "https://files.example/clip.mp4?X-Amz-Signature=secret"


class ContentUrlTest(MockedTest):
    def test_grok_and_two_h3_extends_resolve_before_next_submit(self):
        self.mock.script("POST", "/api/generate-video", [
            {"status": 200, "json": {"runId": "r%d" % i, "cost": 0.1}} for i in range(3)])
        self.mock.script("GET", "/api/video/status", [
            video_status("COMPLETED", url=RELATIVE),
            video_status("COMPLETED", url="/api/generate-video/content?runId=h3"),
            video_status("COMPLETED", url="https://cdn.example/final.mp4")])
        self.mock.script("GET", "/api/generate-video/content", [
            {"status": 307, "headers": {"Location": SIGNED}, "body": b""},
            {"status": 200, "headers": {"Content-Type": "video/mp4"}, "body": b"small-mp4"}])
        nodes = [{"id": "open", "type": "tvideo", "fields": {"model": "grok-imagine-video-1.5-lite", "prompt": "a boat"}}]
        nodes += [{"id": name, "type": "vedit", "fields": {"model": "minimax/h3-max-turbo/extend-video",
                   "prompt": "keep sailing", "modelOpts": {"output": "extended"}}} for name in ("ext1", "ext2")]
        links = [{"id": str(i), "from": {"node": a, "port": "video"}, "to": {"node": b, "port": "video"}}
                 for i, (a, b) in enumerate((("open", "ext1"), ("ext1", "ext2")))]
        result = self.wf_dict({"nodes": nodes, "links": links}, **FAST).run()
        submits = self.mock.requests_to("/api/generate-video", "POST")
        self.assertEqual(len(submits), 3)
        self.assertEqual(submits[1].json["videoUrl"], SIGNED)
        self.assertNotIn("videoDataUrl", submits[1].json)
        self.assertTrue(submits[2].json["videoDataUrl"].startswith("data:video/mp4;base64,"))
        self.assertNotIn("videoUrl", submits[2].json)
        for req in self.mock.requests_to("/api/generate-video/content"):
            self.assertEqual(req.headers["authorization"], "Bearer test-key")
            self.assertNotIn("x-api-key", req.headers)
        self.assertAlmostEqual(result.cost_usd, 0.3)

    def test_default_transport_never_follows_authenticated_redirect_to_file_host(self):
        cdn = MockNanoGPT().start()
        self.addCleanup(cdn.stop)
        self.mock.script("GET", "/api/generate-video/content", {
            "status": 307, "headers": {"Location": cdn.base_url + "/clip.mp4?sig=secret"}, "body": b""})
        engine = Engine("test-key", base_url=self.mock.base_url, http=default_http)
        self.assertEqual(engine.resolve_content_url(RELATIVE), cdn.base_url + "/clip.mp4?sig=secret")
        self.assertEqual(cdn.requests, [])

    def test_failed_content_fetch_preserves_submitted_cost(self):
        self.mock.script("POST", "/api/generate-video", {"status": 200, "json": {"runId": "r1", "cost": 0.1}})
        self.mock.script("GET", "/api/video/status", video_status("COMPLETED", url=RELATIVE))
        self.mock.script("GET", "/api/generate-video/content", {"status": 403, "body": b"secret"})
        wf = self.wf_dict({"nodes": [{"id": "n", "type": "tvideo", "fields": {"model": "grok", "prompt": "a boat"}}]}, **FAST)
        with self.assertRaises(RunError) as caught:
            wf.run()
        self.assertAlmostEqual(caught.exception.result.cost_usd, 0.1)
        self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(len(self.mock.requests_to("/api/generate-video", "POST")), 1)


class ResolverTest(unittest.TestCase):
    def engine(self, response):
        self.calls = []
        def http(method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            if isinstance(response, Exception):
                raise response
            return response
        return Engine("test-key", base_url="https://nano-gpt.com", http=http)

    def test_passthrough_does_not_fetch(self):
        engine = self.engine(None)
        for url in (SIGNED, "data:video/mp4;base64,AAAA", "blob:example"):
            self.assertEqual(engine.resolve_content_url(url), url)
        self.assertEqual(engine.resolve_content_url("/files/video.mp4"), "https://nano-gpt.com/files/video.mp4")
        self.assertEqual(self.calls, [])

    def test_redirect_errors_limits_and_secret_safe_transport_failures(self):
        for response, error in [
            (HttpResponse(307, {}, b""), "no location"),
            (HttpResponse(307, {"location": RELATIVE}, b""), "did not resolve"),
            (HttpResponse(307, {"location": "javascript:bad"}, b""), "no usable location"),
            (HttpResponse(500, {}, b"Bearer test-key?X-Amz-Signature=secret"), "HTTP 500"),
            (HttpResponse(200, {"content-type": "video/mp4"}, b"x" * MEDIA_INLINE_MAX), "too large"),
            (RuntimeError("test-key " + SIGNED), "could not fetch"),
        ]:
            with self.subTest(error=error):
                with self.assertRaisesRegex(NanoodleError, error) as caught:
                    self.engine(response).resolve_content_url(RELATIVE)
                self.assertNotIn("test-key", str(caught.exception))
                self.assertNotIn("secret", str(caught.exception))

    def test_same_origin_api_hops_and_relative_locations(self):
        engine = self.engine(HttpResponse(302, {"location": "/files/clip.mp4?sig=secret"}, b""))
        self.assertEqual(engine.resolve_content_url("https://nano-gpt.com/v1/video/content?id=1"),
                         "https://nano-gpt.com/files/clip.mp4?sig=secret")
        self.assertEqual(len(self.calls), 1)
