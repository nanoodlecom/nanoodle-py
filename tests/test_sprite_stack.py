import json
import shutil
import struct
import unittest

from nanoodle import Workflow
from tests import fixture


class SpriteStackTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
    def test_splat_to_sheet_preview_and_turntable(self):
        with open(fixture("sprite-stack.json"), encoding="utf-8") as source:
            result = Workflow.from_dict(json.load(source)).run()
        sliced = result.nodes["s"].out
        rendered = result.nodes["r"].out
        self.assertEqual(struct.unpack(">II", sliced["sheet"].bytes()[16:24]), (128, 16))
        self.assertEqual(json.loads(sliced["layout"])["upAxis"], "z")
        self.assertEqual(sliced["image"].bytes(), rendered["image"].bytes())
        self.assertEqual(rendered["video"].bytes()[4:8], b"ftyp")
