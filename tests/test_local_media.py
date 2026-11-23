"""Local media nodes: resize / vframes / combine / soundtrack / trim / extractaudio.

Soft dependency: ffmpeg on PATH. Mirrors nanoodle-js/tests/local-media.test.mjs.
"""

import io
import os
import shutil
import subprocess
import tempfile
import unittest

from PIL import Image

from tests import fixture
from tests._util import tripwire_http

from nanoodle import MediaRef, NanoodleError, RunError, Workflow, media_from_file
from nanoodle.local_media import extract_video_frames, resize_plan
from nanoodle.media import parse_data_url

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
    def test_authored_count_exposes_up_to_frame48(self):
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "vframes", "fields": {"frames": "49"}},
        ]})
        self.assertEqual(wf.outputs[0].ports, ["frame%d" % i for i in range(1, 49)])
        frames_knob = next(s for s in wf.settings if s.field == "frames")
        self.assertEqual((frames_knob.min, frames_knob.max), (1, 48))

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

    @skip_no_ffmpeg
    def test_wired_frame13_runs_with_a_lower_authored_count(self):
        # A saved graph may wire frame13 with an authored count of 1. The
        # executor must raise the count and extract the wired frame.
        from nanoodle.graph import wired_frames_floor
        from nanoodle.iodef import derive_settings

        vid = media_from_file(MEDIA("clipA.mp4"))
        data = {
            "nodes": [
                {"id": "n1", "type": "vupload", "fields": {}},
                {"id": "n2", "type": "vframes", "name": "Frames",
                 "fields": {"frames": "1", "gap": "0.05", "dir": "start"}},
                {"id": "n3", "type": "resize", "name": "Thumb",
                 "fields": {"mode": "fit", "width": "16", "height": "16"}},
            ],
            "links": [
                {"id": "l1", "from": {"node": "n1", "port": "video"},
                 "to": {"node": "n2", "port": "video"}},
                {"id": "l2", "from": {"node": "n2", "port": "frame13"},
                 "to": {"node": "n3", "port": "image"}},
            ],
        }
        wf = Workflow.from_dict(data)
        self.assertEqual(wired_frames_floor(wf.graph, "n2"), 13)
        frames_knob = next(s for s in derive_settings(wf.graph)
                           if s.node_id == "n2" and s.field == "frames")
        self.assertEqual((frames_knob.min, frames_knob.max), (13, 48))

        result = wf.run({"Video": vid})
        self.assertEqual(len(result.nodes["n2"].out), 13)
        self.assertTrue(result.nodes["n2"].out["frame13"].url.startswith("data:image/"))
        thumb = result["Thumb"]
        self.assertIsInstance(thumb, MediaRef)
        self.assertTrue(thumb.url.startswith("data:image/"))
        self.assertGreater(len(thumb.bytes()), 20)

    @skip_no_ffmpeg
    def test_wired_count_respects_frame_floor_and_higher_values(self):
        video = media_from_file(MEDIA("clipA.mp4"))
        wf = Workflow.from_dict({"nodes": [
            {"id": "video", "type": "vupload"},
            {"id": "count", "type": "text", "fields": {"text": "1"}},
            {"id": "frames", "type": "vframes", "fields": {"frames": "1", "dir": "start", "gap": "0.04"}},
            {"id": "thumb", "type": "resize", "fields": {"width": "16", "height": "16"}},
        ], "links": [
            {"from": {"node": "video", "port": "video"}, "to": {"node": "frames", "port": "video"}},
            {"from": {"node": "count", "port": "text"}, "to": {"node": "frames", "port": "frames"}},
            {"from": {"node": "frames", "port": "frame3"}, "to": {"node": "thumb", "port": "image"}},
        ]}, api_key="", http=tripwire_http)

        for value, count in (("1", 3), ("5", 5), ("invalid", 3)):
            with self.subTest(value=value):
                result = wf.run({"Video": video, "count.text": value})
                self.assertEqual(list(result.nodes["frames"].out), [f"frame{i}" for i in range(1, count + 1)])
                self.assertIsInstance(result["thumb"], MediaRef)
        self.assertEqual(wf.graph.node("frames").fields["frames"], "1")


@unittest.skipUnless(HAS_FFMPEG and shutil.which("ffprobe"), "ffmpeg and ffprobe not on PATH")
class IndexedVframesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            for index in range(24):
                image = Image.new("RGB", (16, 16), (index * 10, 0, 0))
                image.save(os.path.join(directory, f"f{index:02d}.png"))
            pattern = os.path.join(directory, "f%02d.png")
            for name, fps, timestamps in [
                ("cfr", "24", []),
                ("vfr", "100", [
                    "-vf", "setpts='if(lt(N,4),N*4,16+(N-4)*13)'",
                    "-fps_mode", "vfr", "-output_ts_offset", "2",
                ]),
            ]:
                path = os.path.join(directory, f"{name}.mp4")
                subprocess.run([
                    "ffmpeg", "-loglevel", "error", "-framerate", fps, "-i", pattern,
                    *timestamps, "-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-q:v", "1", path,
                ], check=True, timeout=30)
                setattr(cls, name, media_from_file(path))

    def assert_indices(self, output, expected):
        self.assertEqual(list(output), [f"frame{i}" for i in range(1, len(expected) + 1)])
        indices = []
        for frame in output.values():
            data = frame.bytes() if isinstance(frame, MediaRef) else parse_data_url(frame)[1]
            with Image.open(io.BytesIO(data)) as image:
                red, _, _ = image.convert("RGB").getpixel((8, 8))
            indices.append(round(red / 10))
        self.assertEqual(indices, expected)

    def test_cfr_stride_selects_exact_display_indices(self):
        self.assert_indices(
            extract_video_frames(self.cfr, count=4, stride=3, dir="start"), [0, 3, 6, 9])
        self.assert_indices(
            extract_video_frames(self.cfr, count=3, stride="2", dir="start", start_sec="0.5"),
            [12, 14, 16])

    def test_vfr_uses_relative_timestamps_in_both_directions(self):
        # Relative PTS are 0, .04, .08, .12, .16, .29, .42, .55, ...
        bounds = {"count": 8, "start_sec": 0.071, "end_sec": 0.42}
        for direction, sampling, expected in [
            ("start", {"stride": 2}, [2, 4, 6]),
            ("end", {"stride": 2}, [6, 4, 2]),
            ("start", {"gap": 0.1}, [2, 5, 6]),
            ("end", {"gap": 0.1}, [6, 5, 4]),
        ]:
            with self.subTest(direction=direction, sampling=sampling):
                self.assert_indices(
                    extract_video_frames(self.vfr, dir=direction, **bounds, **sampling), expected)

    def test_workflow_keeps_ports_and_does_not_pad_a_short_range(self):
        graph = {
            "nodes": [
                {"id": "v", "type": "vupload", "fields": {}},
                {"id": "f", "type": "vframes", "name": "Frames", "fields": {
                    "frames": "8", "stride": "2", "startSec": "0.071", "endSec": "0.21",
                    "dir": "start",
                }},
            ],
            "links": [
                {"from": {"node": "v", "port": "video"}, "to": {"node": "f", "port": "video"}},
            ],
        }
        wf = Workflow.from_dict(graph)
        self.assertEqual(wf.outputs[0].ports, [f"frame{i}" for i in range(1, 9)])
        self.assert_indices(wf.run({"Video": self.cfr}).nodes["f"].out, [2, 4])
        result = wf.run({"Video": self.cfr}, settings={"f.dir": "end"})
        self.assert_indices(result.nodes["f"].out, [5, 3])
        narrowed = wf.run({"Video": self.cfr}, settings={"f.endSec": 0.13})
        self.assert_indices(narrowed.nodes["f"].out, [2])
        settings = {spec.field: spec for spec in wf.settings if spec.node_id == "f"}
        self.assertEqual(settings["stride"].min, 1)
        self.assertEqual(settings["startSec"].default, "0.071")
        self.assertEqual(settings["endSec"].default, "0.21")

        graph["nodes"].extend([
            {"id": "r", "type": "resize", "fields": {"width": "16"}},
            {"id": "count", "type": "text", "fields": {"text": "1"}},
        ])
        graph["links"].extend([
            {"from": {"node": "f", "port": "frame3"}, "to": {"node": "r", "port": "image"}},
            {"from": {"node": "count", "port": "text"}, "to": {"node": "f", "port": "frames"}},
        ])
        with self.assertRaisesRegex(RunError, "no image input") as error:
            Workflow.from_dict(graph, api_key="", http=tripwire_http).run({"Video": self.cfr})
        self.assert_indices(error.exception.result.nodes["f"].out, [2, 4])

    def test_invalid_and_empty_ranges_fail(self):
        for fields, message in [
            ({"start_sec": -1}, "startSec must be a non-negative"),
            ({"end_sec": "Infinity"}, "endSec must be a non-negative"),
            ({"start_sec": 0.8, "end_sec": 0.2}, "endSec must be at or after startSec"),
            ({"start_sec": 0.02, "end_sec": 0.03}, "no video frames"),
            ({"stride": 0}, "stride must be a positive integer"),
            ({"stride": 1.5}, "stride must be a positive integer"),
            ({"stride": 2**53}, "stride must be a positive integer"),
        ]:
            with self.subTest(fields=fields), self.assertRaisesRegex(NanoodleError, message):
                extract_video_frames(self.cfr, **fields)


class MediaInlineGuardTest(unittest.TestCase):
    def test_local_only_accepts_oversized_data_url_input(self):
        from nanoodle.media import MEDIA_INLINE_MAX
        big = "data:video/mp4;base64," + "A" * (MEDIA_INLINE_MAX + 50)
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "vupload", "fields": {}},
        ]})
        result = wf.run({"Video": big})
        self.assertEqual(result["Video input"].url, big)

    def test_network_graph_refuses_oversized_media_input(self):
        from nanoodle.media import MEDIA_INLINE_MAX
        from tests._util import tripwire_http
        big = "data:image/png;base64," + "A" * (MEDIA_INLINE_MAX + 50)
        wf = Workflow.from_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {}},
            {"id": "n2", "type": "vision", "fields": {"model": "m"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "image"},
             "to": {"node": "n2", "port": "image"}},
        ]}, api_key="k", http=tripwire_http)
        with self.assertRaises(NanoodleError) as ctx:
            wf.run({"Image": big})
        self.assertIn("too large", str(ctx.exception))


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
