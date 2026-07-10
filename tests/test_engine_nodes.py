"""Per-node payload assertions, polling loops, audio branches, transcribe
multipart, cost extraction, HTTP error mapping, RunError partials, concurrency."""

import base64
import json
import re
import unittest

from tests import fixture
from tests.harness import MockNanoGPT, chat_response, image_response

from nanoodle import MediaRef, NanoodleError, RunError, Workflow

FAST = {"poll_intervals": {"video": 0.01, "audio": 0.01},
        "timeouts": {"video": 1.0, "audio": 1.0}}


class MockedTest(unittest.TestCase):
    def setUp(self):
        self.mock = MockNanoGPT().start()
        self.addCleanup(self.mock.stop)

    def wf(self, name, **opts):
        opts.setdefault("api_key", "test-key")
        opts.setdefault("base_url", self.mock.base_url)
        return Workflow.load(fixture(name), **opts)


class LlmVisionTest(MockedTest):
    def test_llm_vision_payload(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("a red fox"))
        wf = self.wf("llm-vision.json")
        result = wf.run()
        req = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(req.json, {
            "model": "gpt-5o",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What is in this picture?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ]}],
            "temperature": 0.8,
        })
        self.assertEqual(result["LLM"], "a red fox")

    def test_llm_json_mode_and_reasoning_effort(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response('{"ok":1}'))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "llm",
             "fields": {"model": "m", "prompt": "p", "format": "JSON",
                        "reasoningEffort": "high", "maxTokens": "60",
                        "temperature": "0.2"}},
        ]}, api_key="k", base_url=self.mock.base_url)
        wf.run()
        req = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(req.json, {
            "model": "m",
            "messages": [{"role": "user", "content": "p"}],
            "temperature": 0.2,
            "max_tokens": 60,
            "response_format": {"type": "json_object"},
            "reasoning_effort": "high",
        })

    def test_llm_content_parts_joined(self):
        self.mock.script("POST", "/api/v1/chat/completions",
                         chat_response(None, content_parts=[{"type": "text", "text": "he"},
                                                            {"type": "text", "text": "llo"}]))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "llm", "fields": {"model": "m", "prompt": "p"}}]},
            api_key="k", base_url=self.mock.base_url)
        self.assertEqual(wf.run()["LLM"], "hello")

    def test_llm_null_content_is_error(self):
        self.mock.script("POST", "/api/v1/chat/completions",
                         {"status": 200, "json": {"choices": [{"message": {"content": None}}]}})
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "llm", "fields": {"model": "m", "prompt": "p"}}]},
            api_key="k", base_url=self.mock.base_url)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("no text in response", str(ctx.exception))

    def test_llm_wired_audio_becomes_input_audio_part(self):
        wav = "data:audio/wav;base64,UklGRgAAAABXQVZF"
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("heard it"))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "aupload", "fields": {"audio": wav}},
            {"id": "n2", "type": "llm", "fields": {"model": "m", "prompt": "what is this?"}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "audio"},
                      "to": {"node": "n2", "port": "audio"}}]},
            api_key="k", base_url=self.mock.base_url)
        wf.run()
        req = self.mock.requests_to("/api/v1/chat/completions")[0]
        content = req.json["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "what is this?"})
        self.assertEqual(content[1], {"type": "input_audio",
                                      "input_audio": {"data": "UklGRgAAAABXQVZF",
                                                      "format": "wav"}})

    def test_vision_payload(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("a cat"))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": "data:image/png;base64,AAA="}},
            {"id": "n2", "type": "vision", "fields": {"model": "m"}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "image"},
                      "to": {"node": "n2", "port": "image"}}]},
            api_key="k", base_url=self.mock.base_url)
        result = wf.run()
        req = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(req.json["messages"], [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
        ]}])
        self.assertEqual(result["Vision"], "a cat")


class ImageFamilyTest(MockedTest):
    def test_edit_multi_image_payload(self):
        self.mock.script("POST", "/v1/images/generations",
                         image_response(urls=["https://cdn.example/out.png"], cost=0.02))
        wf = self.wf("edit-multi.json")
        result = wf.run()
        req = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(req.json, {
            "model": "nano-banana-2",
            "size": "1k",
            "n": 1,
            "response_format": "b64_json",
            "prompt": "blend them",
            "imageDataUrl": ["data:image/png;base64,AAA=", "data:image/png;base64,BBB="],
        })
        self.assertEqual(result["Edit"].url, "https://cdn.example/out.png")

    def test_edit_single_image_is_string_not_array(self):
        self.mock.script("POST", "/v1/images/generations", image_response(urls=["https://x/y.png"]))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": "data:image/png;base64,AAA="}},
            {"id": "n2", "type": "edit", "fields": {"model": "m", "prompt": "p"}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "image"},
                      "to": {"node": "n2", "port": "image"}}]},
            api_key="k", base_url=self.mock.base_url)
        wf.run()
        req = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(req.json["imageDataUrl"], "data:image/png;base64,AAA=")

    def test_image_variations_seed_and_jpeg_sniff(self):
        jpeg_b64 = base64.b64encode(b"\xff\xd8\xffmockjpeg").decode()
        self.mock.script("POST", "/v1/images/generations",
                         image_response(b64_list=[jpeg_b64, jpeg_b64]))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "image",
             "fields": {"model": "m", "prompt": "p", "variations": "2", "seed": "42"}}]},
            api_key="k", base_url=self.mock.base_url)
        result = wf.run()
        req = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(req.json["n"], 2)
        self.assertEqual(req.json["seed"], 42)
        self.assertNotIn("size", [k for k in req.json if req.json[k] is None])
        self.assertEqual(req.json["size"], "1024x1024")   # default when unset
        self.assertTrue(result["Image"].url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(len(result.nodes["n1"].out["images"]), 2)

    def test_inpaint_passes_source_and_mask(self):
        self.mock.script("POST", "/v1/images/generations", image_response(urls=["https://x/o.png"]))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "inpaint",
             "fields": {"model": "m", "prompt": "a hat",
                        "image": "data:image/png;base64,SRC=",
                        "mask": "data:image/png;base64,MASK="}}]},
            api_key="k", base_url=self.mock.base_url)
        wf.run()
        req = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(req.json["imageDataUrl"], "data:image/png;base64,SRC=")
        self.assertEqual(req.json["maskDataUrl"], "data:image/png;base64,MASK=")
        self.assertEqual(req.json["prompt"], "a hat")

    def test_draw_parses_message_images(self):
        img_url = "data:image/png;base64,DRAWN="
        self.mock.script("POST", "/api/v1/chat/completions",
                         chat_response("here you go",
                                       images=[{"image_url": {"url": img_url}}]))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "draw", "fields": {"model": "m", "prompt": "a boat"}}]},
            api_key="k", base_url=self.mock.base_url)
        result = wf.run()
        req = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(req.json, {"model": "m",
                                    "messages": [{"role": "user", "content": "a boat"}],
                                    "temperature": 0.8})
        self.assertNotIn("response_format", req.json)
        self.assertEqual(result["Draw"].url, img_url)
        self.assertEqual(result.nodes["n1"].out["text"], "here you go")


class VideoTest(MockedTest):
    def test_submit_payload_poll_loop_and_url(self):
        self.mock.script("POST", "/api/generate-video",
                         {"status": 200, "json": {"runId": "r-77", "cost": 0.25,
                                                  "remainingBalance": 3.75}})
        self.mock.script("GET", "/api/video/status", [
            {"status": 200, "json": {"status": "PENDING"}},
            {"status": 200, "json": {"status": "processing"}},
            {"status": 200, "json": {"data": {"status": "COMPLETED",
                                              "output": {"video": {"url": "https://cdn/v.mp4"}}}}},
        ])
        polls = []
        wf = self.wf("video-poll.json", **FAST)
        result = wf.run(on_progress=lambda e: polls.append(e) if e["type"] == "poll" else None)
        submit = self.mock.requests_to("/api/generate-video")[0]
        self.assertEqual(submit.json, {
            "model": "seedance-2.0",
            "prompt": "a drifting paper boat",
            "duration": "5",
            "aspect_ratio": "16:9",   # aspect field -> aspect_ratio wire name
            "resolution": "720p",
            "seed": 7,                # fields.modelOpts merged verbatim
        })
        status_reqs = self.mock.requests_to("/api/video/status")
        self.assertEqual(len(status_reqs), 3)
        self.assertEqual(status_reqs[0].query, "requestId=r-77")
        video = result["Text→Video"]
        self.assertIsInstance(video, MediaRef)
        self.assertEqual(video.url, "https://cdn/v.mp4")
        self.assertAlmostEqual(result.cost_usd, 0.25)
        self.assertEqual(result.remaining_balance, 3.75)
        self.assertGreaterEqual(len(polls), 2)

    def test_video_failed_status_raises(self):
        self.mock.script("POST", "/api/generate-video", {"status": 200, "json": {"id": "r-1"}})
        self.mock.script("GET", "/api/video/status",
                         {"status": 200, "json": {"data": {"status": "FAILED",
                                                           "error": "nsfw filter"}}})
        wf = self.wf("video-poll.json", **FAST)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("video failed: nsfw filter", str(ctx.exception))

    def test_video_poll_timeout(self):
        self.mock.script("POST", "/api/generate-video", {"status": 200, "json": {"runId": "r-2"}})
        self.mock.script("GET", "/api/video/status", {"status": 200, "json": {"status": "PENDING"}})
        wf = self.wf("video-poll.json", poll_intervals={"video": 0.01},
                     timeouts={"video": 0.05})
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("timed out", str(ctx.exception))

    def test_video_poll_garbage_is_skipped(self):
        self.mock.script("POST", "/api/generate-video", {"status": 200, "json": {"runId": "r-3"}})
        self.mock.script("GET", "/api/video/status", [
            {"status": 500, "body": b"boom"},
            {"status": 200, "body": b"not json"},
            {"status": 200, "json": {"status": "SUCCEEDED", "output": {"url": "https://cdn/x.mp4"}}},
        ])
        wf = self.wf("video-poll.json", **FAST)
        result = wf.run()
        self.assertEqual(result["Text→Video"].url, "https://cdn/x.mp4")

    def test_ivideo_sources_and_endframe(self):
        self.mock.script("POST", "/api/generate-video", {"status": 200, "json": {"runId": "r"}})
        self.mock.script("GET", "/api/video/status",
                         {"status": 200, "json": {"status": "COMPLETED",
                                                  "output": {"url": "https://cdn/i.mp4"}}})
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": "data:image/png;base64,FIRST"}},
            {"id": "n2", "type": "upload", "fields": {"image": "data:image/png;base64,LAST"}},
            {"id": "n3", "type": "ivideo", "fields": {"model": "m", "prompt": "morph"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "image"}, "to": {"node": "n3", "port": "image"}},
            {"id": "l2", "from": {"node": "n2", "port": "image"}, "to": {"node": "n3", "port": "endframe"}},
        ]}, api_key="k", base_url=self.mock.base_url, **FAST)
        wf.run()
        submit = self.mock.requests_to("/api/generate-video")[0]
        self.assertEqual(submit.json["imageDataUrl"], "data:image/png;base64,FIRST")
        self.assertEqual(submit.json["last_image"], "data:image/png;base64,LAST")

    def test_vedit_data_vs_https_source(self):
        self.mock.script("POST", "/api/generate-video", {"status": 200, "json": {"runId": "r"}})
        self.mock.script("GET", "/api/video/status",
                         {"status": 200, "json": {"status": "COMPLETED",
                                                  "output": {"url": "https://cdn/e.mp4"}}})
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "vupload", "fields": {"video": "data:video/mp4;base64,VID"}},
            {"id": "n2", "type": "vedit", "fields": {"model": "m", "prompt": "p"}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "video"},
                      "to": {"node": "n2", "port": "video"}}]},
            api_key="k", base_url=self.mock.base_url, **FAST)
        wf.run()
        submit = self.mock.requests_to("/api/generate-video")[0]
        self.assertEqual(submit.json["videoDataUrl"], "data:video/mp4;base64,VID")
        self.assertNotIn("videoUrl", submit.json)

    def test_tvideo_reference_images_ordered(self):
        self.mock.script("POST", "/api/generate-video", {"status": 200, "json": {"runId": "r"}})
        self.mock.script("GET", "/api/video/status",
                         {"status": 200, "json": {"status": "COMPLETED",
                                                  "output": {"url": "https://cdn/r.mp4"}}})
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": "data:image/png;base64,ONE"}},
            {"id": "n2", "type": "upload", "fields": {"image": "data:image/png;base64,TWO"}},
            {"id": "n3", "type": "tvideo", "fields": {"model": "m", "prompt": "p"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n2", "port": "image"}, "to": {"node": "n3", "port": "ref2"}},
            {"id": "l2", "from": {"node": "n1", "port": "image"}, "to": {"node": "n3", "port": "ref1"}},
        ]}, api_key="k", base_url=self.mock.base_url, **FAST)
        wf.run()
        submit = self.mock.requests_to("/api/generate-video")[0]
        self.assertEqual(submit.json["reference_images"],
                         ["data:image/png;base64,ONE", "data:image/png;base64,TWO"])


class AudioTest(MockedTest):
    def test_tts_binary_response_with_header_cost(self):
        mp3 = b"ID3\x03\x00fakemp3"
        self.mock.script("POST", "/api/v1/audio/speech",
                         {"status": 200, "body": mp3,
                          "headers": {"Content-Type": "audio/mpeg",
                                      "x-cost": "0.0011", "x-remaining-balance": "4.2"}})
        wf = self.wf("tts-binary.json", **FAST)
        result = wf.run()
        req = self.mock.requests_to("/api/v1/audio/speech")[0]
        self.assertEqual(req.json, {"model": "kokoro-v1", "input": "hello world",
                                    "voice": "af_bella"})   # speed 1 omitted
        audio = result["Speech"]
        self.assertIsInstance(audio, MediaRef)
        self.assertEqual(audio.mime, "audio/mpeg")
        self.assertEqual(audio.bytes(), mp3)
        self.assertAlmostEqual(result.cost_usd, 0.0011)
        self.assertTrue(result.cost_exact)
        self.assertEqual(result.remaining_balance, 4.2)

    def test_tts_generic_content_type_pinned_from_format(self):
        self.mock.script("POST", "/api/v1/audio/speech",
                         {"status": 200, "body": b"RIFFxxxxWAVEdata",
                          "headers": {"Content-Type": "application/octet-stream"}})
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "tts",
             "fields": {"model": "m", "prompt": "hi",
                        "extraJson": "{\"response_format\": \"wav\"}"}}]},
            api_key="k", base_url=self.mock.base_url, **FAST)
        result = wf.run()
        req = self.mock.requests_to("/api/v1/audio/speech")[0]
        self.assertEqual(req.json["response_format"], "wav")
        self.assertEqual(result["Speech"].mime, "audio/wav")

    def test_audio_json_url_branch(self):
        self.mock.script("POST", "/api/v1/audio/speech",
                         {"status": 200, "json": {"audioUrl": "https://cdn/a.mp3", "cost": 0.03}})
        wf = self.wf("tts-binary.json", **FAST)
        result = wf.run()
        self.assertEqual(result["Speech"].url, "https://cdn/a.mp3")
        self.assertAlmostEqual(result.cost_usd, 0.03)

    def test_music_run_id_poll_branch(self):
        self.mock.script("POST", "/api/v1/audio/speech",
                         {"status": 200, "json": {"runId": "job-9", "cost": 0.05,
                                                  "paymentSource": "balance",
                                                  "isApiRequest": True}})
        self.mock.script("GET", "/api/tts/status", [
            {"status": 200, "json": {"status": "pending", "queuePosition": 2}},
            {"status": 200, "json": {"status": "completed", "audioUrl": "https://cdn/song.mp3"}},
        ])
        wf = self.wf("music-poll.json", **FAST)
        result = wf.run()
        submit = self.mock.requests_to("/api/v1/audio/speech")[0]
        self.assertEqual(submit.json, {"model": "suno-v5", "input": "lofi beats",
                                       "instrumental": True, "duration": 30})
        poll = self.mock.requests_to("/api/tts/status")[0]
        params = dict(p.split("=", 1) for p in poll.query.split("&"))
        self.assertEqual(params, {"runId": "job-9", "model": "suno-v5", "cost": "0.05",
                                  "paymentSource": "balance", "isApiRequest": "true"})
        self.assertEqual(result["Music"].url, "https://cdn/song.mp3")

    def test_audio_poll_failure_raises(self):
        self.mock.script("POST", "/api/v1/audio/speech", {"status": 200, "json": {"id": "j"}})
        self.mock.script("GET", "/api/tts/status",
                         {"status": 200, "json": {"status": "content_policy_violation",
                                                  "error": "nope"}})
        wf = self.wf("music-poll.json", **FAST)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("audio failed: nope", str(ctx.exception))

    def test_music_extra_json_merges_and_song_count_purged(self):
        self.mock.script("POST", "/api/v1/audio/speech",
                         {"status": 200, "json": {"url": "https://cdn/x.mp3"}})
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "music",
             "fields": {"model": "m", "prompt": "p",
                        "extraJson": "{\"style\": \"jazz\", \"number_of_songs\": 4}"}}]},
            api_key="k", base_url=self.mock.base_url, **FAST)
        wf.run()
        req = self.mock.requests_to("/api/v1/audio/speech")[0]
        self.assertEqual(req.json.get("style"), "jazz")
        self.assertNotIn("number_of_songs", req.json)

    def test_remix_audio_source_and_invalid_extra_json(self):
        self.mock.script("POST", "/api/v1/audio/speech",
                         {"status": 200, "json": {"url": "https://cdn/r.mp3"}})
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "aupload", "fields": {"audio": "https://host/src.mp3"}},
            {"id": "n2", "type": "remix", "fields": {"model": "m", "prompt": "make it disco"}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "audio"},
                      "to": {"node": "n2", "port": "audio"}}]},
            api_key="k", base_url=self.mock.base_url, **FAST)
        wf.run()
        req = self.mock.requests_to("/api/v1/audio/speech")[0]
        self.assertEqual(req.json["audio"], "https://host/src.mp3")   # https rides as-is
        self.assertEqual(req.json["input"], "make it disco")

        wf_bad = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "tts",
             "fields": {"model": "m", "prompt": "x", "extraJson": "{nope"}}]},
            api_key="k", base_url=self.mock.base_url, **FAST)
        with self.assertRaises(RunError) as ctx:
            wf_bad.run()
        self.assertIn("advanced params: invalid JSON", str(ctx.exception))


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


class CostAndErrorTest(MockedTest):
    def _one_llm(self, **fields):
        base = {"model": "m", "prompt": "p"}
        base.update(fields)
        return Workflow.from_dict({"nodes": [{"id": "n1", "type": "llm", "fields": base}]},
                                  api_key="secret-key-123", base_url=self.mock.base_url)

    def test_zero_cost_is_known_free(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("ok", cost_usd=0))
        result = self._one_llm().run()
        self.assertEqual(result.cost_usd, 0.0)
        self.assertTrue(result.cost_exact)   # present-but-zero = known-free

    def test_pricing_amount_fallback_and_balance_header_wins(self):
        j = {"choices": [{"message": {"content": "ok"}}],
             "x_nanogpt_pricing": {"amount": 0.004, "remainingBalance": 9.0}}
        self.mock.script("POST", "/api/v1/chat/completions",
                         {"status": 200, "json": j,
                          "headers": {"x-remaining-balance": "8.5"}})
        result = self._one_llm().run()
        self.assertAlmostEqual(result.cost_usd, 0.004)
        self.assertEqual(result.remaining_balance, 8.5)   # header beats the JSON field

    def test_missing_cost_marks_inexact(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("ok"))
        result = self._one_llm().run()
        self.assertEqual(result.cost_usd, 0.0)
        self.assertFalse(result.cost_exact)

    def test_401_maps_to_auth_error_and_never_leaks_key(self):
        self.mock.script("POST", "/api/v1/chat/completions",
                         {"status": 401, "body": b'{"error":"bad key"}'})
        with self.assertRaises(RunError) as ctx:
            self._one_llm().run()
        msg = str(ctx.exception)
        self.assertIn("API key rejected", msg)
        self.assertNotIn("secret-key-123", msg)
        self.assertNotIn("secret-key-123", repr(ctx.exception.result))

    def test_402_and_balance_body_map_to_out_of_funds(self):
        self.mock.script("POST", "/api/v1/chat/completions",
                         {"status": 402, "body": b"payment required"})
        with self.assertRaises(RunError) as ctx:
            self._one_llm().run()
        self.assertIn("out of balance", str(ctx.exception))
        self.mock.script("POST", "/api/v1/chat/completions",
                         {"status": 500, "body": b"insufficient funds for model"})
        with self.assertRaises(RunError) as ctx2:
            self._one_llm().run()
        self.assertIn("out of balance", str(ctx2.exception))

    def test_500_maps_to_status_prefixed_error_truncated(self):
        self.mock.script("POST", "/api/v1/chat/completions",
                         {"status": 500, "body": b"X" * 500})
        with self.assertRaises(RunError) as ctx:
            self._one_llm().run()
        self.assertIn("500: " + "X" * 160, str(ctx.exception))
        self.assertNotIn("X" * 161, str(ctx.exception))

    def test_run_error_carries_partial_results(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("prompt text", cost_usd=0.001))
        self.mock.script("POST", "/v1/images/generations", {"status": 500, "body": b"exploded"})
        wf = Workflow.load(fixture("starter-graph.json"), api_key="k",
                           base_url=self.mock.base_url)
        with self.assertRaises(RunError) as ctx:
            wf.run({"Text": "x"})
        result = ctx.exception.result
        self.assertEqual(result.nodes["n2"].status, "done")     # llm succeeded
        self.assertEqual(result.nodes["n2"].out["text"], "prompt text")
        self.assertEqual(result.nodes["n3"].status, "failed")
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["node_id"], "n3")
        self.assertAlmostEqual(result.cost_usd, 0.001)          # partial cost kept

    def test_upstream_failure_names_the_upstream(self):
        self.mock.script("POST", "/api/v1/chat/completions", {"status": 500, "body": b"dead"})
        wf = Workflow.load(fixture("starter-graph.json"), api_key="k",
                           base_url=self.mock.base_url)
        with self.assertRaises(RunError) as ctx:
            wf.run({"Text": "x"})
        result = ctx.exception.result
        self.assertEqual(result.nodes["n3"].status, "failed")
        self.assertEqual(result.nodes["n3"].error, "upstream failed: LLM")
        # the image endpoint was never called
        self.assertEqual(self.mock.requests_to("/v1/images/generations"), [])

    def test_non_sink_failure_no_dependent_sink_is_warning_only(self):
        # n1 -> (n2 llm FAILS), n1 -> n3 llm OK; n2 has no downstream sink of its own
        self.mock.script("POST", "/api/v1/chat/completions", [
            {"status": 500, "body": b"boom"},
            chat_response("fine"),
        ])
        wf = Workflow.load(fixture("parallel-lanes.json"), api_key="k",
                           base_url=self.mock.base_url)
        # Both lanes ARE sinks here, so instead craft: failing lane must reject.
        with self.assertRaises(RunError):
            wf.run()


class ConcurrencyTest(MockedTest):
    def test_parallel_lanes_overlap(self):
        self.mock.script("POST", "/api/v1/chat/completions",
                         dict(chat_response("slow"), delay=0.25))
        wf = Workflow.load(fixture("parallel-lanes.json"), api_key="k",
                           base_url=self.mock.base_url)
        import time
        t0 = time.monotonic()
        result = wf.run()
        elapsed = time.monotonic() - t0
        self.assertEqual(len(self.mock.requests_to("/api/v1/chat/completions")), 2)
        self.assertGreaterEqual(self.mock.max_concurrent, 2)   # both lanes in flight at once
        self.assertLess(elapsed, 0.48)                          # ran concurrently, not serially
        self.assertEqual(result["Lane A"], "slow")
        self.assertEqual(result["Lane B"], "slow")


class FieldOverrideTest(MockedTest):
    def test_wire_into_textarea_field_overrides_typed_value(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("arr"))
        wf = self.wf("field-override.json")
        wf.run()
        req = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(req.json["messages"][0],
                         {"role": "system", "content": "You are a pirate."})
        # override the Persona (text node) input -> new system prompt
        self.mock.reset()
        wf.run({"Persona": "You are a knight."})
        req2 = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(req2.json["messages"][0]["content"], "You are a knight.")


class MediaRefTest(unittest.TestCase):
    def test_bytes_save_and_str(self):
        import os
        import tempfile
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
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "clip.wav")
            with open(p, "wb") as f:
                f.write(b"RIFF\x00\x00\x00\x00WAVEdata" + b"\x00" * 200)
            from nanoodle import media_from_file
            ref = media_from_file(p)
            self.assertEqual(ref.mime, "audio/wav")
            self.assertLess(len(repr(ref)), 120)

    def test_https_ref_without_fetcher_errors_clearly(self):
        ref = MediaRef("https://cdn/x.png?sig=abc")
        with self.assertRaises(NanoodleError) as ctx:
            ref.bytes()
        self.assertNotIn("sig=abc", str(ctx.exception))   # query (may carry tokens) not echoed


if __name__ == "__main__":
    unittest.main()
