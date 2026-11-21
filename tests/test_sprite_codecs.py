import base64
import struct
import unittest

from nanoodle.errors import NanoodleError
from nanoodle.sprite_codecs import decode_image, encode_png
from nanoodle.sprite_video import encode_mp4


class SpriteCodecTest(unittest.TestCase):
    def test_adam7_two_bit_palette_and_transparency(self):
        raw = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAsAAAAJAgMAAAHuDQ4oAAAADFBMVEVkPBTwCgsP3H0IGfeFddTT"
            "AAAAA3RSTlMAUJZZoa/7AAAAIUlEQVR4nGNggIMVYNjRAELlDgx3DyDIY8eOMOTk5MBoAFFIEJD6"
            "kU3lAAAAAElFTkSuQmCC")
        palette = ((100, 60, 20, 0), (240, 10, 11, 80), (15, 220, 125, 150), (8, 25, 247, 255))
        expected = bytes(c for y in range(9) for x in range(11) for c in palette[(x + 3*y) % 4])
        self.assertEqual(decode_image(raw), (11, 9, expected))
        self.assertEqual(decode_image(encode_png(11, 9, expected)), (11, 9, expected))
        with self.assertRaises(NanoodleError):
            decode_image(raw[:-20])

    def test_interlaced_gif_with_transparent_pixels(self):
        raw = base64.b64decode(
            "R0lGODlhEwAUAIEAAGQ8FPAKCw/cfQgZ9yH5BAEAAAAALAAAAAATABQAQAhNAAEEEDBAIEGDBQcm"
            "FICwoUKHDB9KPDgR4sKLBi1qHEixI8aHHjceDDkRI8mTBVFq/KgyQEmRA1jKFPiyJsGZNW3KxEl"
            "Sp0WeH32WDAgAOw==")
        palette = ((0, 0, 0, 0), (240, 10, 11, 255), (15, 220, 125, 255), (8, 25, 247, 255))
        expected = bytes(c for y in range(20) for x in range(19) for c in palette[(x + 3*y) % 4])
        self.assertEqual(decode_image(raw), (19, 20, expected))

    def test_native_video_records_skipped_frames_and_refreshes(self):
        frame = bytes((100, 13, 55, 255)) * (66 * 50)
        video = encode_mp4(66, 50, (frame for _ in range(61)), 2.5)
        self.assertIn(b"avc1", video)
        key_table = video.index(b"stss") + 8
        self.assertEqual(struct.unpack_from(">3I", video, key_table), (2, 1, 61))
        sizes_at = video.index(b"stsz") + 8
        default_size, count = struct.unpack_from(">II", video, sizes_at)
        self.assertEqual((default_size, count), (0, 61))
        sizes = struct.unpack_from(">61I", video, sizes_at + 8)
        self.assertLess(max(sizes[1:60]), 16)
        self.assertTrue(384 < sizes[60] < 1000)
