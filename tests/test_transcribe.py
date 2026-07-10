"""Transcribe node: multipart body (file field MUST be named "file"), auth
headers, response-key priority, size guard, https source download."""

import base64
import unittest

from tests._util import FAST, MockedTest

from nanoodle import NanoodleError, RunError


class TranscribeTest(MockedTest):
    def test_multipart_payload_and_parse(self):
        self.mock.script("POST", "/api/v1/audio/transcriptions",
                         {"status": 200, "json": {"text": "hi there",
                                                  "metadata": {"cost": 0.002}}})
        wf = self.wf("transcribe.json", **FAST)
        result = wf.run()
        req = self.mock.requests_to("/api/v1/audio/transcriptions")[0]
        ctype = req.headers.get("content-type", "")
        self.assertTrue(ctype.startswith("multipart/form-data; boundary="))
        boundary = ctype.split("boundary=")[1]
        parts = req.body.split(b"--" + boundary.encode())
        joined = req.body.decode("latin-1")
        # the audio form field MUST be named "file"
        self.assertIn('name="file"; filename="audio.wav"', joined)
        self.assertRegex(joined, r'name="model"\r\n\r\nwhisper-1')
        self.assertRegex(joined, r'name="language"\r\n\r\nauto')
        self.assertGreaterEqual(len(parts), 4)
        # decoded wav bytes ride in the file part
        self.assertIn(base64.b64decode("UklGRgAAAABXQVZF"), req.body)
        self.assertEqual(req.headers.get("authorization"), "Bearer test-key")
        self.assertEqual(req.headers.get("x-api-key"), "test-key")
        self.assertEqual(result["Transcribe"], "hi there")
        self.assertAlmostEqual(result.cost_usd, 0.002)   # metadata.cost path

    def test_transcription_key_priority(self):
        self.mock.script("POST", "/api/v1/audio/transcriptions",
                         {"status": 200, "json": {"transcription": "primary", "text": "secondary"}})
        wf = self.wf("transcribe.json", **FAST)
        self.assertEqual(wf.run()["Transcribe"], "primary")

    def test_nested_data_fallbacks(self):
        for j, expected in (({"data": {"transcription": "deep"}}, "deep"),
                            ({"data": {"text": "deeper"}}, "deeper")):
            self.mock.reset()
            self.mock.script("POST", "/api/v1/audio/transcriptions", {"status": 200, "json": j})
            wf = self.wf("transcribe.json", **FAST)
            self.assertEqual(wf.run()["Transcribe"], expected)

    def test_no_transcription_in_response_errors(self):
        self.mock.script("POST", "/api/v1/audio/transcriptions", {"status": 200, "json": {}})
        wf = self.wf("transcribe.json", **FAST)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("no transcription in response", str(ctx.exception))

    def test_oversized_clip_is_local_error_no_network(self):
        from nanoodle.media import TRANSCRIBE_MAX_BYTES, make_data_url
        big = make_data_url(b"\x00" * (TRANSCRIBE_MAX_BYTES + 1), "audio/wav")
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "aupload", "fields": {"audio": big}},
            {"id": "n2", "type": "transcribe", "fields": {"model": "whisper-1"}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "audio"},
                      "to": {"node": "n2", "port": "audio"}}]}, **FAST)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("too big to transcribe", str(ctx.exception))
        self.assertEqual(self.mock.requests, [])

    def test_https_source_downloaded_then_uploaded(self):
        clip = b"ID3\x03\x00remoteclip"
        self.mock.script("GET", "/media/clip.mp3",
                         {"status": 200, "body": clip,
                          "headers": {"Content-Type": "audio/mpeg"}})
        self.mock.script("POST", "/api/v1/audio/transcriptions",
                         {"status": 200, "json": {"text": "remote words"}})
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "aupload",
             "fields": {"audio": self.mock.base_url + "/media/clip.mp3"}},
            {"id": "n2", "type": "transcribe", "fields": {"model": "whisper-1"}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "audio"},
                      "to": {"node": "n2", "port": "audio"}}]}, **FAST)
        result = wf.run()
        self.assertEqual(result["Transcribe"], "remote words")
        # the media GET must NOT carry auth headers (provider CDNs, not NanoGPT)
        get = self.mock.requests_to("/media/clip.mp3")[0]
        self.assertNotIn("authorization", get.headers)
        self.assertIn(clip, self.mock.requests_to("/api/v1/audio/transcriptions")[0].body)

    def test_missing_model_error(self):
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "aupload", "fields": {"audio": "data:audio/wav;base64,UklG"}},
            {"id": "n2", "type": "transcribe", "fields": {}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "audio"},
                      "to": {"node": "n2", "port": "audio"}}]}, **FAST)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("pick a model first", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
