import hashlib
import json
import struct
import unittest
from unittest.mock import patch

from nanoodle import Workflow
from tests import fixture


class SpriteStackTest(unittest.TestCase):
    def test_splat_to_sheet_preview_and_turntable(self):
        from nanoodle.sprite_codecs import decode_image

        with open(fixture("sprite-stack.json"), encoding="utf-8") as source:
            graph = json.load(source)
        with patch("subprocess.Popen", side_effect=AssertionError("unexpected external sprite tool")):
            result = Workflow.from_dict(graph).run()
        sliced = result.nodes["s"].out
        rendered = result.nodes["r"].out
        self.assertEqual(struct.unpack(">II", sliced["sheet"].bytes()[16:24]), (128, 16))
        self.assertEqual(json.loads(sliced["layout"])["upAxis"], "z")
        self.assertEqual(sliced["image"].bytes(), rendered["image"].bytes())
        # The fixture includes translucent Gaussian footprints on multiple layers.
        width, height, pixels = decode_image(rendered["image"].bytes())
        self.assertEqual((width, height), (72, 70))
        self.assertEqual(hashlib.sha256(pixels).hexdigest(), "94a2e13aaf7fe9dd78318dc2502224fb81b17baad031a0f819d7162437f25300")
        video = rendered["video"].bytes()
        self.assertEqual(video[4:8], b"ftyp")
        self.assertIn(b"avc1", video)
        self.assertIn(b"mdat", video)
