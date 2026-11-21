import hashlib
import struct
import unittest
from pathlib import Path

from nanoodle.errors import NanoodleError
from nanoodle.sprite_codecs import decode_image
from nanoodle.sprite_webp import _alpha

MEDIA = Path(__file__).parent / "fixtures" / "media"


class SpriteWebPTest(unittest.TestCase):
    def test_lossless_transforms_and_straight_alpha(self):
        w, h, rgba = decode_image((MEDIA / "sprite-lossless.webp").read_bytes())
        self.assertEqual((w, h), (37, 29))
        # Independently decoded with libwebp, preserving hidden RGB.
        self.assertEqual(hashlib.sha256(rgba).hexdigest(),
                         "e23cb32963638f393ad0ae18ca86d1f69e355283475fabfcb824508233b9f15c")

    def test_lossy_keyframe_and_compressed_alpha(self):
        for name, w, h, first, center, alpha_hash in [
            ("lossy", 41, 33, (249, 34, 42), (99, 190, 80),
             "6b7c37085dd4cc1c51e5216faeaee37e7d761001efe29c49d79ec3233339d5dd"),
            ("alpha", 39, 35, (236, 36, 35), (39, 202, 81),
             "0d81e98e05d1e48899b525f5534e8ab6b6c167364d31de26587df91462b6f07d"),
        ]:
            with self.subTest(name=name):
                width, height, rgba = decode_image((MEDIA / f"sprite-{name}.webp").read_bytes())
                self.assertEqual((width, height), (w, h))
                self.assertEqual(hashlib.sha256(rgba[3::4]).hexdigest(), alpha_hash)
                for offset, expected in [(0, first), (4 * ((h // 2) * w + w // 2), center)]:
                    for actual, reference in zip(rgba[offset:offset + 3], expected):
                        self.assertLessEqual(abs(actual - reference), 1)

    def test_first_animation_frame_composites_at_offset(self):
        w, h, rgba = decode_image((MEDIA / "sprite-animation.webp").read_bytes())
        self.assertEqual((w, h), (8, 8))
        self.assertEqual(rgba[:4], bytes(4))
        self.assertEqual(rgba[4 * (w * 2 + 3):4 * (w * 2 + 4)], bytes((150, 40, 70, 127)))
        self.assertEqual(rgba[3::4].count(127), 15)
        self.assertEqual(rgba[3::4].count(0), 49)

    def test_alpha_predictors_wrap_and_handle_first_row_and_column(self):
        expected = bytes([1, 4, 9, 13, 10, 8, 50, 60, 80])
        for predictor, encoded in enumerate([
            expected,
            [1, 3, 5, 12, 253, 254, 37, 10, 20],
            [1, 3, 5, 12, 6, 255, 37, 50, 72],
            [1, 3, 5, 12, 250, 249, 37, 13, 22],
        ]):
            self.assertEqual(_alpha(bytes([predictor << 2]) + bytes(encoded), 3, 3), expected)

    def test_truncated_chunks_and_oversized_canvas(self):
        good = (MEDIA / "sprite-lossless.webp").read_bytes()
        bad_canvas = b"VP8X" + struct.pack("<I", 10) + bytes(4) + b"\xff" * 6
        oversized = b"RIFF" + struct.pack("<I", len(bad_canvas) + 4) + b"WEBP" + bad_canvas
        for invalid in [good[:-1], good[:4] + bytes(4) + good[8:], oversized]:
            with self.assertRaises(NanoodleError):
                decode_image(invalid)
