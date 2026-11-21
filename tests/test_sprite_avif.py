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

    def test_clean_aperture_then_orientation(self):
        # Chromium sprite uploads use a top-left clean aperture, ignore an
        # offset aperture, and rotate/mirror the resulting image in both cases.
        cases = [
            ("sprite-aperture", (17, 15), 21, (88, 217, 40),
             "9ed8cce51fbcd2934bc44a69b5e0473f885cfdc53c87216cf1afc7d943f8239b"),
            ("sprite-offset-aperture", (16, 16), 64, (37, 163, 63),
             "4cae390b9391dd1022f76ab499973835a0ae48a8e3164168fa8d6ee1a7df794e"),
            ("sprite-rotated", (11, 16), 0, (9, 182, 43),
             "4add79e75ac001a6bb02a45194e4577b257f9b17fd95834adfe5baa1cda6da1b"),
        ]
        for name, size, sample, rgb, alpha_digest in cases:
            with self.subTest(name=name):
                w, h, pixels = decode_image(Path(fixture(f"media/{name}.avif")).read_bytes())
                self.assertEqual((w, h), size)
                self.assertEqual(hashlib.sha256(pixels[3::4]).hexdigest(), alpha_digest)
                self.assertTrue(all(abs(a - b) <= 3 for a, b in zip(pixels[4 * sample:], rgb)))

    def test_truncated_avif_reports_a_decoder_error(self):
        with self.assertRaisesRegex(NanoodleError, "Invalid AVIF"):
            decode_image(Path(fixture("media/sprite-alpha.avif")).read_bytes()[:64])
