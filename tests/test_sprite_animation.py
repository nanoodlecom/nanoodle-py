import io
import json
import struct
import unittest
from pathlib import Path

from PIL import Image

from nanoodle import NanoodleError, Workflow
from nanoodle.media import make_data_url
from nanoodle.sprite_animation import encode_stack_animation, read_stack_animation
from nanoodle.sprite_codecs import _chunk, decode_image
from nanoodle.sprite_stack import _Projection
from nanoodle.sprite_video import encode_mp4
from tests import fixture


def stack_graph(model_fields, render_fields):
    graph = json.loads(Path(fixture("sprite-stack.json")).read_text())
    knight = Path(fixture("media/sprite-knight.glb")).read_bytes()
    graph["nodes"][0]["fields"] = {"model": make_data_url(knight, "model/gltf-binary")}
    graph["nodes"][1]["fields"] = model_fields
    graph["nodes"][2]["fields"] = render_fields
    return Workflow.from_dict(graph)


def render_upload(raw, fields):
    graph = {
        "nodes": [
            {
                "id": "u",
                "type": "upload",
                "fields": {"image": make_data_url(raw, "image/png")},
            },
            {"id": "r", "type": "sprite-stack-render", "fields": fields},
        ],
        "links": [
            {
                "from": {"node": "u", "port": "image"},
                "to": {"node": "r", "port": "sheet"},
            }
        ],
    }
    return Workflow.from_dict(graph).run().nodes["r"].out


def reference_video(sheets, plan, fields, duration):
    views = [_Projection(sheet, fields) for sheet in sheets]
    frames = (views[pose].frame(yaw, lambda: None) for pose, yaw in plan)
    return encode_mp4(views[0].width, views[0].height, frames, duration)


class StackAnimationTest(unittest.TestCase):
    def test_real_rig_clip_has_stable_bounds_and_distinct_full_poses(self):
        workflow = stack_graph(
            {
                "preset": "character",
                "resolution": 24,
                "slices": 16,
                "outline": False,
                "clip": "Walk",
                "frames": 4,
            },
            {"scale": 1},
        )
        settings = {s.field: s for s in workflow.settings if s.node_id == "s"}
        self.assertEqual(settings["palette"].default, "0")
        self.assertIs(settings["outline"].default, False)
        self.assertEqual(settings["frames"].max, 64)
        result = workflow.run()
        out = result.nodes["s"].out
        layout = json.loads(out["layout"])
        self.assertEqual(
            layout["animation"],
            {"clip": "Walk", "frames": 4, "duration": 1, "times": [0, 0.25, 0.5, 0.75]},
        )
        for actual, expected in zip(layout["bounds"]["max"], [5.06, 3.56, -2.61]):
            self.assertAlmostEqual(actual, expected, places=5)
        raw = out["sheet"].bytes()
        sequence = read_stack_animation(raw)
        self.assertEqual(
            (sequence.width, sequence.height, sequence.delays), (384, 24, [0.25] * 4)
        )
        with Image.open(io.BytesIO(raw)) as image:
            self.assertEqual(image.n_frames, 4)
            sheets = []
            for index in range(4):
                image.seek(index)
                sheets.append(sequence.sheet(index))
                self.assertEqual(image.convert("RGBA").tobytes(), sheets[-1][2])
        # The knight lifts then lowers its shield; phases 1 and 3 meet halfway.
        self.assertEqual(len({s[2] for s in sheets}), 3)
        self.assertEqual(sheets[1], sheets[3])
        self.assertEqual(
            result.nodes["r"].out["video"].bytes(),
            reference_video(sheets, [(i, 35) for i in range(4)], {"scale": 1}, 1),
        )

    def test_single_pose_is_a_static_png_and_character_defaults_are_visible(self):
        wf = stack_graph({"preset": "character"}, {})
        settings = {s.field: s.default for s in wf.settings if s.node_id == "s"}
        self.assertEqual(
            (
                settings["resolution"],
                settings["slices"],
                settings["palette"],
                settings["outline"],
            ),
            ("128", "80", "0", True),
        )
        result = wf.run()
        layout = json.loads(result.nodes["s"].out["layout"])
        self.assertNotIn("animation", layout)
        self.assertEqual((layout["tileWidth"], layout["columns"]), (128, 80))
        self.assertNotIn("video", result.nodes["r"].out)
        self.assertIsNone(read_stack_animation(result.nodes["s"].out["sheet"].bytes()))

    def test_cross_runtime_apng_full_frames_and_turntable_timing(self):
        raw = Path(fixture("media/sprite-sequence.apng")).read_bytes()
        seq = read_stack_animation(raw)
        sheets = [seq.sheet(i) for i in range(3)]
        self.assertEqual(seq.delays, [0.125] * 3)
        offset = (7 * 128 + 6) * 4
        self.assertEqual(
            [s[2][offset : offset + 4] for s in sheets],
            [bytes(c) for c in [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]],
        )
        with Image.open(
            io.BytesIO(
                encode_stack_animation(128, 16, (s[2] for s in sheets), 3, 0.375)
            )
        ) as im:
            self.assertEqual(im.n_frames, 3)
            for i in range(3):
                im.seek(i)
                self.assertEqual(im.tobytes(), sheets[i][2])
        fields = {
            "scale": 1,
            "turntable": True,
            "yawStart": 0,
            "yawEnd": 180,
            "fps": 8,
            "duration": 0.75,
        }
        out = render_upload(raw, fields)
        plan = [(i % 3, i * 36) for i in range(6)]
        self.assertEqual(
            out["video"].bytes(), reference_video(sheets, plan, fields, 0.75)
        )

    def test_unmarked_apng_is_a_still_and_corrupt_marked_poses_fail(self):
        raw = Path(fixture("media/sprite-sequence.apng")).read_bytes()
        plain = raw.replace(_chunk(b"nsTA", b"\x01"), b"")
        self.assertIsNone(read_stack_animation(plain))
        self.assertNotIn("video", render_upload(plain, {}))
        self.assertEqual(decode_image(plain)[2], read_stack_animation(raw).sheet(0)[2])
        at = raw.index(b"fdAT")
        length = struct.unpack_from(">I", raw, at - 4)[0]
        bad = _chunk(b"fdAT", struct.pack(">I", 100) + raw[at + 8 : at + 4 + length])
        with self.assertRaisesRegex(NanoodleError, "frame data"):
            read_stack_animation(raw[: at - 4] + bad + raw[at + 8 + length :])
