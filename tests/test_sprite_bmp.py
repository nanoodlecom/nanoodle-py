import struct
import unittest

from nanoodle.errors import NanoodleError
from nanoodle.sprite_bmp import decode_bmp, decode_ico
from nanoodle.sprite_codecs import encode_png


def dib(width, height, depth, body, *, compression=0, table=b"", used=0):
    header = struct.pack("<IiiHH6I", 40, width, height, 1, depth,
                         compression, len(body), 0, 0, used, 0)
    return header + table + body


def bitmap(dib_bytes, pixel_offset):
    return struct.pack("<2sIHHI", b"BM", len(dib_bytes) + 14, 0, 0, 14 + pixel_offset) + dib_bytes


def icon(entries):
    directory = bytearray(struct.pack("<3H", 0, 1, len(entries)))
    data = bytearray()
    for width, height, depth, payload in entries:
        offset = 6 + len(entries) * 16 + len(data)
        directory.extend(struct.pack("<4B2H2I", width % 256, height % 256, 0, 0, 1, depth,
                                     len(payload), offset))
        data.extend(payload)
    return directory + data


class SpriteBitmapTest(unittest.TestCase):
    def test_core_palette_and_bottom_up_padding(self):
        core = struct.pack("<I4H", 12, 2, 2, 1, 1)
        palette = bytes((0, 0, 255, 255, 0, 0))  # red and blue in BGR
        rows = b"\x80\0\0\0\x40\0\0\0"
        red, blue = b"\xff\0\0\xff", b"\0\0\xff\xff"
        self.assertEqual(decode_bmp(bitmap(core + palette + rows, 18)), (2, 2, red + blue + blue + red))

    def test_top_down_and_rgb565(self):
        row = bytes((5, 9, 21, 13, 29, 37, 0, 0))
        image = bitmap(dib(2, -1, 24, row), 40)
        self.assertEqual(decode_bmp(image), (2, 1, bytes((21, 9, 5, 255, 37, 29, 13, 255))))
        masks = struct.pack("<3I", 0xF800, 0x7E0, 0x1F)
        packed = struct.pack("<2H", 0xF800, 0x7E0)
        self.assertEqual(decode_bmp(bitmap(dib(2, 1, 16, packed, compression=3, table=masks), 52)),
                         (2, 1, b"\xff\0\0\xff\0\xff\0\xff"))

    def test_rle_absolute_and_run_packets(self):
        palette = b"\0\0\0\0\xff\0\0\0\0\0\xff\0"
        packets = bytes((4, 1, 0, 0, 0, 4, 2, 0, 2, 0, 0, 1))
        data = bitmap(dib(4, 2, 8, packets, compression=1, table=palette, used=3), 52)
        blue, red, black = bytes((0, 0, 255, 255)), bytes((255, 0, 0, 255)), bytes((0, 0, 0, 255))
        self.assertEqual(decode_bmp(data), (4, 2, (red + black) * 2 + blue * 4))
        # Delta, EOL and EOB can leave pixels unpainted. Their display color
        # remains black, even if the file supplied a blue palette entry zero.
        skip = bytes((0, 2, 2, 0, 2, 1, 0, 1))
        data = bitmap(dib(4, 1, 8, skip, compression=1, table=palette[4:], used=2), 48)
        self.assertEqual(decode_bmp(data), (4, 1, black * 2 + red * 2))
        with self.assertRaisesRegex(NanoodleError, "row"):
            decode_bmp(bitmap(dib(1, 2, 8, packets, compression=1, table=palette, used=3), 52))

    def test_ico_resolution_and_and_mask(self):
        # The small PNG comes first; the 24-bit DIB is the preferred 2×1 frame.
        png = encode_png(1, 1, b"\0\0\xff\xff")
        row = bytes((255, 255, 255, 0, 0, 255, 0, 0))
        data = dib(2, 2, 24, row + bytes((0x80, 0, 0, 0)))
        ico = icon([(1, 1, 32, png), (2, 1, 24, data)])
        self.assertEqual(decode_ico(ico), (2, 1, bytes((0, 0, 0, 0, 255, 0, 0, 255))))
        self.assertEqual(decode_ico(icon([(1, 1, 32, png)])), (1, 1, b"\0\0\xff\xff"))
        cursor = bytes((0, 0, 2, 0)) + ico[4:]
        self.assertEqual(decode_ico(cursor), decode_ico(ico))

    def test_ico_alpha_and_zero_alpha_fallback(self):
        row = bytes((3, 2, 1, 64, 7, 6, 5, 255))
        mask = bytes((0xC0, 0, 0, 0))
        ico = icon([(2, 1, 32, dib(2, 2, 32, row + mask))])
        self.assertEqual(decode_ico(ico), (2, 1, bytes((1, 2, 3, 64, 5, 6, 7, 255))))
        row = bytes((3, 2, 1, 0, 7, 6, 5, 0))
        mask = bytes((0x40, 0, 0, 0))
        ico = icon([(2, 1, 32, dib(2, 2, 32, row + mask))])
        self.assertEqual(decode_ico(ico), (2, 1, bytes((1, 2, 3, 255, 0, 0, 0, 0))))


if __name__ == "__main__":
    unittest.main()
