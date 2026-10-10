"""🎧 Clean voice: public-URL source, duration quote, speech submit + tts poll.

No live NanoGPT calls. The length probe is patched — it must not ffprobe a URL.
"""

import unittest
from unittest import mock

from tests._util import FAST, MockedTest

from nanoodle import MediaRef, NanoodleError, RunError, estimate_graph_cost
from nanoodle.cleanvoice import (CLEANVOICE_DEFAULT_MODEL, clean_voice_error,
                                 clean_voice_seconds, clean_voice_source)
from nanoodle.engine import NodeCancelled


def _graph(nodes, links=None):
    return {"nodes": nodes, "links": links or []}


class CleanVoiceRunTest(MockedTest):
    def setUp(self):
        super(CleanVoiceRunTest, self).setUp()
        patcher = mock.patch("nanoodle.cleanvoice.clean_voice_media_seconds", return_value=6.05)
        self.seconds = patcher.start()
        self.addCleanup(patcher.stop)

    def _cv(self, fields, links=None, nodes=None):
        base = nodes or []
        base.append({"id": "c", "type": "cleanvoice", "name": "Clean", "fields": fields})
        return self.wf_dict(_graph(base, links), **FAST)

    def test_audio_beats_video_and_pasted_link(self):
        self.mock.script("POST", "/api/v1/audio/speech",
                         {"status": 200, "json": {"url": "https://cdn/clean.mp3", "cost": 0.02}})
        wf = self._cv(
            {"model": "elevenlabs/audio-isolation", "url": "https://cdn/pasted.mp3"},
            links=[
                {"id": "l1", "from": {"node": "a", "port": "audio"}, "to": {"node": "c", "port": "audio"}},
                {"id": "l2", "from": {"node": "v", "port": "video"}, "to": {"node": "c", "port": "video"}},
            ],
            nodes=[
                {"id": "a", "type": "aupload", "fields": {"audio": "https://cdn/clip.mp3"}},
                {"id": "v", "type": "vupload", "fields": {"video": "https://cdn/clip.mp4"}},
            ])
        audio = wf.run()["Clean"]
        body = self.mock.requests_to("/api/v1/audio/speech")[0].json
        self.assertEqual(body["audio"], "https://cdn/clip.mp3")
        self.assertEqual(body["model"], "elevenlabs/audio-isolation")
        self.assertEqual(body["input"], "")
        self.assertEqual(body["duration"], 6.05)
        self.assertIsInstance(audio, MediaRef)
        self.assertEqual(audio.url, "https://cdn/clean.mp3")

    def test_video_then_pasted_link(self):
        self.mock.script("POST", "/api/v1/audio/speech",
                         {"status": 200, "json": {"url": "https://cdn/out.mp3"}})
        wf = self._cv(
            {"model": "m", "url": "https://cdn/pasted.mp3"},
            links=[{"id": "l1", "from": {"node": "v", "port": "video"},
                    "to": {"node": "c", "port": "video"}}],
            nodes=[{"id": "v", "type": "vupload", "fields": {"video": "https://cdn/clip.mp4"}}])
        wf.run()
        self.assertEqual(self.mock.requests_to("/api/v1/audio/speech")[0].json["audio"],
                         "https://cdn/clip.mp4")

        self.mock.reset()
        self.mock.script("POST", "/api/v1/audio/speech",
                         {"status": 200, "json": {"url": "https://cdn/out.mp3"}})
        wf = self._cv({"model": "m", "url": "https://cdn/pasted.ogg"})
        wf.run()
        self.assertEqual(self.mock.requests_to("/api/v1/audio/speech")[0].json["audio"],
                         "https://cdn/pasted.ogg")

    def test_inline_clip_is_refused_before_any_request(self):
        wf = self._cv({"model": "m", "url": "data:audio/mpeg;base64,AAAA"})
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("hosted file", str(ctx.exception))
        self.assertEqual(self.mock.requests, [])

    def test_duration_quote_clamps_and_falls_back(self):
        self.assertEqual(clean_voice_seconds(0.1), 0.6)
        self.assertEqual(clean_voice_seconds(99999), 3600)
        self.assertEqual(clean_voice_seconds(None), 60)
        self.assertEqual(clean_voice_seconds(6.05), 6.05)
        for secs, want in ((0.1, 0.6), (None, 60), (4000, 3600)):
            self.mock.reset()
            self.seconds.return_value = secs
            self.mock.script("POST", "/api/v1/audio/speech",
                             {"status": 200, "json": {"url": "https://cdn/o.mp3"}})
            wf = self._cv({"model": "m", "url": "https://cdn/in.mp3"})
            wf.run()
            self.assertEqual(self.mock.requests_to("/api/v1/audio/speech")[0].json["duration"], want)

    def test_blank_model_defaults_and_veed_is_kept(self):
        self.mock.script("POST", "/api/v1/audio/speech",
                         {"status": 200, "json": {"url": "https://cdn/o.mp3"}})
        self._cv({"url": "https://cdn/in.mp3"}).run()
        self.assertEqual(self.mock.requests_to("/api/v1/audio/speech")[0].json["model"],
                         CLEANVOICE_DEFAULT_MODEL)
        self.mock.reset()
        self.mock.script("POST", "/api/v1/audio/speech",
                         {"status": 200, "json": {"url": "https://cdn/o.mp3"}})
        self._cv({"model": "veed/clean-audio", "url": "https://cdn/in.mp3"}).run()
        self.assertEqual(self.mock.requests_to("/api/v1/audio/speech")[0].json["model"],
                         "veed/clean-audio")

    def test_202_polls_tts_status_and_keeps_submit_cost(self):
        self.mock.script("POST", "/api/v1/audio/speech",
                         {"status": 202, "json": {"runId": "cv-1", "cost": 0.0121,
                                                  "paymentSource": "balance", "isApiRequest": True}})
        self.mock.script("GET", "/api/tts/status", [
            {"status": 200, "json": {"status": "pending", "queuePosition": 1}},
            {"status": 200, "json": {"status": "completed", "audioUrl": "https://cdn/clean.mp3"}},
        ])
        result = self._cv({"model": CLEANVOICE_DEFAULT_MODEL, "url": "https://cdn/in.mp3"}).run()
        self.assertEqual(result["Clean"].url, "https://cdn/clean.mp3")
        self.assertAlmostEqual(result.cost_usd, 0.0121)
        poll = self.mock.requests_to("/api/tts/status")[0]
        self.assertIn("runId=cv-1", poll.query)
        self.assertIn("model=" + CLEANVOICE_DEFAULT_MODEL.replace("/", "%2F"), poll.query)

    def test_provider_errors_are_reworded(self):
        cases = (
            ("please verify the source audio duration", "couldn't read this file's length"),
            ("unable to download the file", "must be public"),
            ("needs a public http(s) source", "public https link"),
        )
        for body, needle in cases:
            self.mock.reset()
            self.mock.script("POST", "/api/v1/audio/speech",
                             {"status": 400, "body": body})
            wf = self._cv({"model": "m", "url": "https://cdn/in.mp3"})
            with self.assertRaises(RunError) as ctx:
                wf.run()
            self.assertIn(needle, str(ctx.exception))

    def test_cancelled_passes_through(self):
        err = NodeCancelled("run cancelled")
        self.assertIs(clean_voice_error(err), err)

    def test_source_helper_rejects_a_missing_file(self):
        with self.assertRaises(NanoodleError) as ctx:
            clean_voice_source({}, {})
        self.assertIn("no audio", str(ctx.exception))


ELEVEN = {"per_second": 0.002016666666666667, "minimum": 0.00121}
VEED = {"per_billing_interval": 0.01375, "billing_interval_seconds": 60, "minimum": 0.01375}


class CleanVoiceEstimateTest(unittest.TestCase):
    def _est(self, nodes, links, model_id, pricing):
        catalogs = {"audio": [{"id": model_id, "pricing": pricing}]}
        return estimate_graph_cost(_graph(nodes, links), catalogs)

    def test_uses_wired_duration_else_thirty_seconds(self):
        nodes = [
            {"id": "v", "type": "tvideo", "fields": {"duration": "6", "model": "vid"}},
            {"id": "c", "type": "cleanvoice", "fields": {"model": "elevenlabs/audio-isolation"}},
        ]
        links = [{"id": "l", "from": {"node": "v", "port": "video"},
                  "to": {"node": "c", "port": "video"}}]
        est = self._est(nodes, links, "elevenlabs/audio-isolation", ELEVEN)
        self.assertAlmostEqual(est.usd, 0.0121, places=6)
        self.assertFalse(est.exact)
        self.assertEqual(est.unpriced, 1)  # the video node has no catalog row

        alone = self._est(
            [{"id": "c", "type": "cleanvoice", "fields": {"model": "elevenlabs/audio-isolation"}}],
            [], "elevenlabs/audio-isolation", ELEVEN)
        self.assertAlmostEqual(alone.usd, 0.0605, places=6)
        self.assertEqual(alone.unpriced, 0)

    def test_minimum_and_veed_started_minutes(self):
        from nanoodle.estimate import audio_unit_usd
        self.assertAlmostEqual(audio_unit_usd(ELEVEN, None, 0.3), 0.00121)
        self.assertAlmostEqual(
            audio_unit_usd(ELEVEN, None, 23.562448979591835), 0.047517605442176876, places=9)
        self.assertAlmostEqual(audio_unit_usd(VEED, None, 23.5), 0.01375)
        self.assertAlmostEqual(audio_unit_usd(VEED, None, 61), 0.0275)
        est = self._est(
            [{"id": "s", "type": "tts", "fields": {"duration": "61", "model": "ignored"}},
             {"id": "c", "type": "cleanvoice", "fields": {"model": "veed/clean-audio"}}],
            [{"id": "l", "from": {"node": "s", "port": "audio"}, "to": {"node": "c", "port": "audio"}}],
            "veed/clean-audio", VEED)
        # tts is unpriced (no catalog row for "ignored"); clean voice bills two started minutes
        self.assertAlmostEqual(est.usd, 0.0275)
        self.assertEqual(est.priced, 1)
