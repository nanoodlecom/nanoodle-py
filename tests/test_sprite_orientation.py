import struct
import unittest
import zlib
from pathlib import Path

from nanoodle.errors import NanoodleError
from nanoodle.sprite_codecs import decode_image, encode_png


def tiff(value, endian, tag_type=3, count=1):
    prefix = b"II" if endian == "<" else b"MM"
    return (
        prefix
        + struct.pack(endian + "HI", 42, 12)
        + b"\0" * 4
        + struct.pack(endian + "HHHIH", 1, 0x112, tag_type, count, value)
        + b"\0" * 6
    )


def png_exif(data, metadata):
    # The 13-byte IHDR ends after byte 33 of the file.
    chunk = b"eXIf" + metadata
    return (
        data[:33] + struct.pack(">I", len(metadata)) + chunk + struct.pack(">I", zlib.crc32(chunk)) + data[33:]
    )


def jpeg_exif(data, metadata):
    chunk = b"Exif\0\0" + metadata
    return data[:2] + b"\xff\xe1" + struct.pack(">H", len(chunk) + 2) + chunk + data[2:]


class SpriteOrientationTest(unittest.TestCase):
    def test_all_png_orientations_in_both_byte_orders(self):
        colors = {ord(ch): bytes([ord(ch), 41, 201, i * 17]) for i, ch in enumerate("ABCDEF")}
        raw = lambda text: b"".join(colors[ch] for ch in text)
        png = encode_png(3, 2, raw(b"ABCDEF"))
        for i, ordered in enumerate(
            [b"ABCDEF", b"CBAFED", b"FEDCBA", b"DEFABC", b"ADBECF", b"DAEBFC", b"FCEBDA", b"CFBEAD"], 1
        ):
            size = (3, 2) if i <= 4 else (2, 3)
            for endian in "<>":
                with self.subTest(orientation=i, endian=endian):
                    self.assertEqual(decode_image(png_exif(png, tiff(i, endian))), (*size, raw(ordered)))

    def test_jpeg_app1_matches_png_exif_for_every_orientation(self):
        fixture = Path(__file__).parent / "fixtures" / "media" / "sprite-baseline.jpg"
        jpg = fixture.read_bytes()
        png = encode_png(*decode_image(jpg))
        for i in range(1, 9):
            metadata = tiff(i, ">" if i % 2 else "<")
            self.assertEqual(decode_image(jpeg_exif(jpg, metadata)), decode_image(png_exif(png, metadata)))

    def test_invalid_optional_metadata_is_ignored(self):
        rgba = bytes(range(16))
        png = encode_png(2, 2, rgba)
        for metadata in (
            b"",
            b"II*\x00\xff\xff\xff\xff",
            tiff(6, ">")[:-8],
            tiff(6, "<", tag_type=4),
            tiff(6, ">", count=2),
            tiff(9, ">"),
        ):
            self.assertEqual(decode_image(png_exif(png, metadata)), (2, 2, rgba))

    def test_jpeg_pixel_cap_checked_before_image_allocation(self):
        jpeg = bytearray((Path(__file__).parent / "fixtures/media/sprite-baseline.jpg").read_bytes())
        frame = jpeg.index(b"\xff\xc0")
        struct.pack_into(">HH", jpeg, frame + 5, 4097, 4096)
        with self.assertRaisesRegex(NanoodleError, "16 megapixels"):
            decode_image(jpeg)
