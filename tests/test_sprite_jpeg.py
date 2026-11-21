"""Odd-sized, chroma-subsampled textures shared with the Rust executor."""

import unittest
from pathlib import Path

from nanoodle.errors import NanoodleError
from nanoodle.sprite_codecs import decode_image


class SpriteJpegTest(unittest.TestCase):
    directory = Path(__file__).parent / "fixtures" / "media"

    def test_sequential_and_progressive_restart_scans_match(self):
        width, height, pixels = decode_image((self.directory / "sprite-baseline.jpg").read_bytes())
        self.assertEqual((width, height), (35, 19))
        self.assertEqual(
            (width, height, pixels), decode_image((self.directory / "sprite-progressive.jpg").read_bytes())
        )
        self.assertEqual(pixels[3::4], b"\xff" * (width * height))
        # Independently decoded libjpeg-turbo colors, including the final partial MCU.
        for x, y, rgb in [
            (0, 0, (0, 34, 27)),
            (4, 7, (56, 127, 121)),
            (20, 11, (206, 240, 163)),
            (34, 18, (30, 97, 105)),
        ]:
            offset = (y * width + x) * 4
            self.assertTrue(all(abs(a - b) <= 3 for a, b in zip(pixels[offset : offset + 3], rgb)))

    def test_truncated_scan_is_rejected(self):
        encoded = (self.directory / "sprite-progressive.jpg").read_bytes()
        with self.assertRaisesRegex(NanoodleError, "JPEG"):
            decode_image(encoded[:-13])
