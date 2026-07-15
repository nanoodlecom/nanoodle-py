"""Local media nodes: resize / vframes / combine / soundtrack / trim / extractaudio.

Soft dependency: ffmpeg on PATH. Mirrors nanoodle-js/tests/local-media.test.mjs.
"""

import os
import shutil
import unittest

from tests import fixture

from nanoodle import MediaRef, NanoodleError, Workflow, media_from_file
from nanoodle.local_media import resize_plan

HAS_FFMPEG = shutil.which("ffmpeg") is not None
MEDIA = lambda name: fixture(os.path.join("media", name))


def skip_no_ffmpeg(fn):
    return unittest.skipUnless(HAS_FFMPEG, "ffmpeg not on PATH")(fn)


class ResizePlanTest(unittest.TestCase):
    def test_fit_never_upscales(self):
        p = resize_plan(200, 100, "fit", 100, 100)
        self.assertEqual(p["cw"], 100)
        self.assertEqual(p["ch"], 50)

    def test_exact_stretches(self):
        p = resize_plan(200, 100, "exact", 50, 50)
        self.assertEqual((p["cw"], p["ch"]), (50, 50))

    def test_fill_covers(self):
        p = resize_plan(200, 100, "fill", 50, 50)
        self.assertEqual((p["cw"], p["ch"]), (50, 50))
        self.assertGreaterEqual(p["dw"], 50)

    def test_missing_dims(self):
        self.assertIsNone(resize_plan(10, 10, "fit", 0, 0))


class ResizeNodeTest(unittest.TestCase):
    @skip_no_ffmpeg
    def test_fit_shrinks_png(self):
        with open(MEDIA("nn-red.png"), "rb") as f:
            data = f.read()
        import base64
        png = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": png}},
            {"id": "n2", "type": "resize", "fields": {"mode": "fit", "width": "32", "height": "32"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "image"}, "to": {"node": "n2", "port": "image"}},
        ]})
        out = wf.run()["Resize / crop"]
        self.assertIsInstance(out, MediaRef)
        self.assertTrue(out.url.startswith("data:image/"))
        self.assertGreater(len(out.bytes()), 20)

    @skip_no_ffmpeg
    def test_missing_dims_errors(self):
        with open(MEDIA("nn-red.png"), "rb") as f:
            data = f.read()
        import base64
        png = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": png}},
            {"id": "n2", "type": "resize", "fields": {"mode": "fit"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "image"}, "to": {"node": "n2", "port": "image"}},
        ]})
        from nanoodle import RunError
        with self.assertRaises((NanoodleError, RunError)) as ctx:
            wf.run()
        self.assertIn("width or height", str(ctx.exception).lower())


class TrimNodeTest(unittest.TestCase):
    @skip_no_ffmpeg
    def test_trim_wav(self):
        wav = media_from_file(MEDIA("nn-tone.wav"))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "aupload", "fields": {}},
            {"id": "n2", "type": "trim", "fields": {"start": "0", "length": "0.25"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "audio"}, "to": {"node": "n2", "port": "audio"}},
        ]})
        out = wf.run({"Audio": wav})["Trim audio"]
        self.assertIsInstance(out, MediaRef)
        raw = out.bytes()
        self.assertEqual(raw[:4], b"RIFF")
        self.assertLess(len(raw), 32078)

    @skip_no_ffmpeg
    def test_start_past_end(self):
        wav = media_from_file(MEDIA("nn-tone.wav"))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "aupload", "fields": {}},
            {"id": "n2", "type": "trim", "fields": {"start": "99", "length": "1"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "audio"}, "to": {"node": "n2", "port": "audio"}},
        ]})
        from nanoodle import RunError
        with self.assertRaises((NanoodleError, RunError)) as ctx:
            wf.run({"Audio": wav})
        self.assertIn("past the end", str(ctx.exception).lower())


class ExtractAudioNodeTest(unittest.TestCase):
    @skip_no_ffmpeg
    def test_extract_from_mp4(self):
        vid = media_from_file(MEDIA("clipA.mp4"))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "vupload", "fields": {}},
            {"id": "n2", "type": "extractaudio", "fields": {"start": "0"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "video"}, "to": {"node": "n2", "port": "video"}},
        ]})
        out = wf.run({"Video": vid})["Extract audio"]
        self.assertEqual(out.bytes()[:4], b"RIFF")


class VframesNodeTest(unittest.TestCase):
    @skip_no_ffmpeg
    def test_extract_two_frames(self):
        vid = media_from_file(MEDIA("clipA.mp4"))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "vupload", "fields": {}},
            {"id": "n2", "type": "vframes", "name": "Frames",
             "fields": {"frames": "2", "gap": "0.2", "dir": "start"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "video"}, "to": {"node": "n2", "port": "video"}},
        ]})
        result = wf.run({"Video": vid})
        f1 = result["Frames"]
        self.assertIsInstance(f1, MediaRef)
        self.assertTrue(f1.url.startswith("data:image/"))
        self.assertIn("frame2", result.nodes["n2"].out)


class CombineNodeTest(unittest.TestCase):
    @skip_no_ffmpeg
    def test_concat_two_clips(self):
        a = media_from_file(MEDIA("clipA.mp4"))
        b = media_from_file(MEDIA("clipB.mp4"))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "vupload", "name": "A", "fields": {}},
            {"id": "n2", "type": "vupload", "name": "B", "fields": {}},
            {"id": "n3", "type": "combine", "fields": {"dedup": "false"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "video"}, "to": {"node": "n3", "port": "clip1"}},
            {"id": "l2", "from": {"node": "n2", "port": "video"}, "to": {"node": "n3", "port": "clip2"}},
        ]})
        out = wf.run({"A": a, "B": b})["Combine videos"]
        self.assertIsInstance(out, MediaRef)
        self.assertGreater(len(out.bytes()), 1000)

    @skip_no_ffmpeg
    def test_one_clip_errors(self):
        a = media_from_file(MEDIA("clipA.mp4"))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "vupload", "fields": {}},
            {"id": "n2", "type": "combine", "fields": {}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "video"}, "to": {"node": "n2", "port": "clip1"}},
        ]})
        from nanoodle import RunError
        with self.assertRaises((NanoodleError, RunError)) as ctx:
            wf.run({"Video": a})
        self.assertIn("at least two clips", str(ctx.exception).lower())


class SoundtrackNodeTest(unittest.TestCase):
    @skip_no_ffmpeg
    def test_mux_wav_onto_video(self):
        vid = media_from_file(MEDIA("clipA.mp4"))
        wav = media_from_file(MEDIA("nn-tone.wav"))
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "vupload", "fields": {}},
            {"id": "n2", "type": "aupload", "fields": {}},
            {"id": "n3", "type": "soundtrack", "fields": {"loop": "false"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "video"}, "to": {"node": "n3", "port": "video"}},
            {"id": "l2", "from": {"node": "n2", "port": "audio"}, "to": {"node": "n3", "port": "audio"}},
        ]})
        out = wf.run({"Video": vid, "Audio": wav})["Soundtrack"]
        self.assertIsInstance(out, MediaRef)
        self.assertGreater(len(out.bytes()), 500)


if __name__ == "__main__":
    unittest.main()
