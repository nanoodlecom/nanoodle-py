"""Exercise glTF animation and skinning with vertex coordinates known analytically."""

import base64
import json
import math
import struct
import unittest

from nanoodle.errors import NanoodleError
from nanoodle.sprite_gltf_tracks import Track
from nanoodle.sprite_model import read_model, read_model_poses


def _rig():
    views, accessors = [], []
    binary = bytearray()

    def view(data):
        binary.extend(bytes(-len(binary) % 4))
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
        binary.extend(data)
        return len(views) - 1

    def accessor(rows, shape, fmt="f", normalized=False):
        data = b"".join(struct.pack("<" + fmt*len(row), *row) for row in rows)
        accessors.append({"bufferView": view(data), "type": shape, "componentType": 5126 if fmt == "f" else 5121,
                          "count": len(rows), "normalized": normalized})
        return len(accessors) - 1

    pos = accessor([(1, 0, 0), (0, 1, 0), (0, 0, 1)], "VEC3")
    # Two numbered sets, mixed weight encodings and non-unit total weights.
    attrs = {"POSITION": pos}
    for i in range(2):
        attrs[f"JOINTS_{i}"] = accessor([(i, 0, 0, 0)] * 3, "VEC4", "B")
        attrs[f"WEIGHTS_{i}"] = (accessor([(85, 0, 0, 0)] * 3, "VEC4", "B", True)
                                  if i == 0 else accessor([(1.0, 0, 0, 0)] * 3, "VEC4"))

    morph = len(accessors)
    accessors.append({"type": "VEC3", "count": 3, "componentType": 5126,
                      "sparse": {"count": 1, "indices": {"bufferView": view(b"\0"), "componentType": 5121},
                                 "values": {"bufferView": view(struct.pack("<3f", 0, 2, 0))}}})
    inverse = accessor([(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, offset, 0, 0, 1) for offset in (-4, -8)], "MAT4")
    times = accessor([(2,), (4,)], "SCALAR")
    trans = accessor([(5, 0, 0), (7, 0, 0)], "VEC3")
    weights = accessor([(1,), (0,)], "SCALAR")
    return {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"translation": [3, 0, 0], "children": [1, 2, 3]},
                  {"translation": [-100, 0, 0], "mesh": 0, "skin": 0, "weights": [0.5]},
                  {"translation": [4, 0, 0]}, {"translation": [8, 0, 0]}],
        "skins": [{"joints": [2, 3], "inverseBindMatrices": inverse}],
        "meshes": [{"primitives": [{"attributes": attrs, "targets": [{"POSITION": morph}]}], "weights": [0.25]}],
        "animations": [{"channels": [{"target": {"node": 2, "path": "translation"}, "sampler": 0},
                                     {"target": {"node": 1, "path": "weights"}, "sampler": 1}],
                        "samplers": [{"input": times, "output": trans}, {"input": times, "output": weights}]}],
        "accessors": accessors, "bufferViews": views,
        "buffers": [{"uri": "data:application/octet-stream;base64," + base64.b64encode(binary).decode(),
                     "byteLength": len(binary)}]
    }


class GltfPosesTest(unittest.TestCase):
    def test_bind_pose_and_default_morph_weights(self):
        doc = _rig()
        for use_node_weights, expected_y in [(True, 1), (False, 0.5)]:
            if not use_node_weights:
                del doc["nodes"][1]["weights"]
            kind, meshes = read_model(json.dumps(doc).encode(), None)
            self.assertEqual(kind, "mesh")
            self.assertEqual(meshes[0].positions, [(4, expected_y, 0), (3, 1, 0), (3, 0, 1)])

    def test_unnamed_clip_and_animated_sparse_morph(self):
        rig = json.dumps(_rig()).encode()
        for selected in [None, "Animation 1"]:
            _, name, start, duration, pose = read_model_poses(rig, None, selected, 4)
            self.assertEqual((name, start, duration), ("Animation 1", 2, 2))
            expected = [(4.25, 2, 0), (4.375, 1.5, 0), (4.5, 1, 0), (4.625, 0.5, 0)]
            self.assertEqual([pose(i)[0].positions[0] for i in range(4)], expected)
            self.assertEqual(pose(0)[0].positions[0], expected[0])
        self.assertEqual(read_model_poses(rig, None, "Animation 1", 1)[-1](0)[0].positions[0], expected[0])
        with self.assertRaisesRegex(NanoodleError, "Available clips: Animation 1"):
            read_model_poses(rig, None, "animation 1", 1)

    def test_reject_invalid_skin_and_clip(self):
        doc = _rig()
        del doc["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_1"]
        with self.assertRaisesRegex(NanoodleError, "paired"):
            read_model(json.dumps(doc).encode(), None)
        del doc["animations"]
        with self.assertRaisesRegex(NanoodleError, "frames=1"):
            read_model_poses(json.dumps(doc).encode(), None, None, 8)

    def test_step_and_spline_segments(self):
        track = Track(0, "translation", [1, 2, 5], [(1,), (2,), (7,)], "STEP")
        self.assertEqual([track.sample(t) for t in (0, 1.5, 2, 6)], [(1,), (1,), (2,), (7,)])
        # Outgoing tangent of 4 multiplied by a segment length of 2.
        track = Track(0, "weights", [2, 4], [(0,), (0,), (4,), (0,), (2,), (0,)], "CUBICSPLINE")
        self.assertEqual(track.sample(3), (2,))
        self.assertEqual(track.sample(4), (2,))

    def test_rotation_interpolators(self):
        # Antipodal encoding of a positive 120 degree rotation.
        q = (0, 0, -math.sqrt(3)/2, -0.5)
        track = Track(0, "rotation", [0, 2], [(0, 0, 0, 1), q], "LINEAR")
        halfway = track.sample(1)
        self.assertAlmostEqual(halfway[2], 0.5)
        self.assertAlmostEqual(halfway[3], math.sqrt(3)/2)
        # Cubic uses the authored quaternion signs, followed by normalization.
        zero = (0, 0, 0, 0)
        track = Track(0, "rotation", [0, 2], [zero, (0, 0, 0, 1), zero, zero, q, zero], "CUBICSPLINE")
        halfway = track.sample(1)
        self.assertAlmostEqual(halfway[2], -math.sqrt(3)/2)
        self.assertAlmostEqual(halfway[3], 0.5)
