import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

from nanoodle.errors import NanoodleError
from nanoodle.sprite_codecs import decode_image
from tests import fixture


class AVIFTest(unittest.TestCase):
    def test_color_and_alpha_without_external_tools(self):
        with patch("subprocess.Popen", side_effect=AssertionError("unexpected external AVIF tool")):
            w, h, pixels = decode_image(Path(fixture("media/sprite-alpha.avif")).read_bytes())
        self.assertEqual((w, h), (37, 41))
        self.assertEqual(hashlib.sha256(pixels[3::4]).hexdigest(),
                         "86fed21888f90588261242598e9b4971dc23e0aa0f7a03465bcc32ea53d51e69")
        self.assertTrue(all(abs(a - b) <= 3 for a, b in zip(pixels[:3], (245, 46, 49))))

    def test_truncated_avif_reports_a_decoder_error(self):
        with self.assertRaisesRegex(NanoodleError, "Invalid AVIF"):
            decode_image(Path(fixture("media/sprite-alpha.avif")).read_bytes()[:64])
