"""MediaRef, data-URL helpers, mime sniffing, and the local media-size guards."""

import base64
import os
import tempfile
import unittest

from tests._util import MockedTest, tripwire_http
from tests.harness import chat_response

from nanoodle import MediaRef, NanoodleError, RunError, Workflow, media_from_file
from nanoodle.media import (MEDIA_INLINE_MAX, b64_image_mime, make_data_url,
                            parse_data_url, sniff_mime)


class MediaRefTest(unittest.TestCase):
    def test_bytes_save_and_str(self):
        data = b"\x89PNG\r\n\x1a\nrest"
        ref = MediaRef("data:image/png;base64," + base64.b64encode(data).decode())
        self.assertEqual(ref.mime, "image/png")
        self.assertEqual(ref.bytes(), data)
        self.assertTrue(str(ref).startswith("data:image/png;base64,"))
        with tempfile.TemporaryDirectory() as d:
            p = ref.save(os.path.join(d, "x.png"))
            with open(p, "rb") as f:
                self.assertEqual(f.read(), data)

    def test_media_from_file_and_repr_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "clip.wav")
            with open(p, "wb") as f:
                f.write(b"RIFF\x00\x00\x00\x00WAVEdata" + b"\x00" * 200)
            ref = media_from_file(p)
            self.assertEqual(ref.mime, "audio/wav")
            self.assertLess(len(repr(ref)), 120)   # repr never dumps the payload

    def test_media_from_file_unknown_ext_sniffs_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "mystery.bin")
            with open(p, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\npixels")
            self.assertEqual(media_from_file(p).mime, "image/png")

    def test_media_from_file_explicit_mime_wins(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.dat")
            with open(p, "wb") as f:
                f.write(b"abc")
            self.assertEqual(media_from_file(p, mime="audio/flac").mime, "audio/flac")

    def test_https_ref_without_fetcher_errors_without_query(self):
        ref = MediaRef("https://cdn/x.png?sig=abc")
        with self.assertRaises(NanoodleError) as ctx:
            ref.bytes()
        self.assertNotIn("sig=abc", str(ctx.exception))   # query (may carry tokens) not echoed

    def test_equality_with_ref_and_str(self):
        a = MediaRef("data:image/png;base64,AAA=")
        b = MediaRef("data:image/png;base64,AAA=")
        self.assertEqual(a, b)
        self.assertEqual(a, "data:image/png;base64,AAA=")
        self.assertEqual(hash(a), hash(b))

    def test_suggested_extension(self):
        self.assertEqual(MediaRef("data:image/png;base64,AAA=").suggested_extension(), "png")
        self.assertEqual(MediaRef("https://x/y", mime="video/mp4").suggested_extension(), "mp4")
        self.assertEqual(MediaRef("https://x/y").suggested_extension(), "bin")


class MediaRefFetchTest(MockedTest):
    def test_result_media_bytes_fetched_through_engine(self):
        # an https output URL downloads through the run's transport
        payload = b"\x89PNG\r\n\x1a\nremote"
        self.mock.script("POST", "/v1/images/generations",
                         {"status": 200,
                          "json": {"data": [{"url": self.mock.base_url + "/cdn/out.png"}]}})
        self.mock.script("GET", "/cdn/out.png",
                         {"status": 200, "body": payload,
                          "headers": {"Content-Type": "image/png"}})
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "image", "fields": {"model": "m", "prompt": "p"}}]})
        ref = wf.run()["Image"]
        self.assertEqual(ref.bytes(), payload)
        self.assertEqual(ref.mime, "image/png")   # filled from the response
        # media downloads carry NO auth headers (provider CDNs)
        get = self.mock.requests_to("/cdn/out.png")[0]
        self.assertNotIn("authorization", get.headers)
        self.assertNotIn("x-api-key", get.headers)


class HelpersTest(unittest.TestCase):
    def test_sniff_mime_magic_table(self):
        cases = [
            (b"\x89PNG\r\n\x1a\nxx", "image/png"),
            (b"\xff\xd8\xffdd", "image/jpeg"),
            (b"GIF89a...", "image/gif"),
            (b"RIFF\x00\x00\x00\x00WEBPVP8", "image/webp"),
            (b"RIFF\x00\x00\x00\x00WAVEfmt", "audio/wav"),
            (b"ID3\x04\x00tag", "audio/mpeg"),
            (b"\xff\xfbframe", "audio/mpeg"),
            (b"OggSpage", "audio/ogg"),
            (b"fLaCmeta", "audio/flac"),
            (b"\x00\x00\x00\x18ftypmp42data", "video/mp4"),
            (b"\x1a\x45\xdf\xa3webm", "video/webm"),
            (b"plain text here", "application/octet-stream"),
        ]
        for data, mime in cases:
            self.assertEqual(sniff_mime(data), mime, data)

    def test_b64_image_mime_table(self):
        self.assertEqual(b64_image_mime("/9j/xxxx"), "image/jpeg")
        self.assertEqual(b64_image_mime("iVBORxxx"), "image/png")
        self.assertEqual(b64_image_mime("R0lGxxx"), "image/gif")
        self.assertEqual(b64_image_mime("UklGRxxx"), "image/webp")
        self.assertEqual(b64_image_mime("QUJD"), "image/png")   # default

    def test_parse_data_url_roundtrip_and_text_branch(self):
        mime, data = parse_data_url(make_data_url(b"\x01\x02", "audio/wav"))
        self.assertEqual((mime, data), ("audio/wav", b"\x01\x02"))
        mime, data = parse_data_url("data:text/plain,hello%20world")
        self.assertEqual((mime, data), ("text/plain", b"hello world"))

    def test_parse_data_url_errors(self):
        with self.assertRaises(NanoodleError):
            parse_data_url("https://not-a-data-url")
        with self.assertRaises(NanoodleError):
            parse_data_url("data:image/png;base64,!!!not-base64!!!")


class InlineSizeGuardTest(unittest.TestCase):
    def test_oversized_json_body_is_local_error_before_any_network_call(self):
        big = "data:image/png;base64," + "A" * (MEDIA_INLINE_MAX + 100)
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": big}},
            {"id": "n2", "type": "vision", "fields": {"model": "m"}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "image"},
                      "to": {"node": "n2", "port": "image"}}]},
            api_key="k", http=tripwire_http)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("too large", str(ctx.exception))

    def test_oversized_wired_llm_audio_is_local_error(self):
        big = "data:audio/mpeg;base64," + "A" * (MEDIA_INLINE_MAX + 100)
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "aupload", "fields": {"audio": big}},
            {"id": "n2", "type": "llm", "fields": {"model": "m", "prompt": "p"}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "audio"},
                      "to": {"node": "n2", "port": "audio"}}]},
            api_key="k", http=tripwire_http)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("too large", str(ctx.exception))


class NoKeyLeakTest(MockedTest):
    def test_workflow_and_result_reprs_never_carry_the_key(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("ok"))
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "llm", "fields": {"model": "m", "prompt": "p"}}]},
            api_key="hunter2-key")
        result = wf.run()
        for blob in (repr(wf.__dict__.get("graph")), repr(result),
                     repr(result.nodes["n1"]), repr(wf.inputs), repr(wf.settings)):
            self.assertNotIn("hunter2-key", blob or "")


if __name__ == "__main__":
    unittest.main()
