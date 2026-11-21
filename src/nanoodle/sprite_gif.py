"""Decode the first displayed image of a GIF palette or sprite sheet."""

import struct

from .errors import NanoodleError


def _subblocks(data, pos):
    joined = bytearray()
    while pos < len(data):
        size = data[pos]
        pos += 1
        if not size:
            return joined, pos
        if pos + size > len(data):
            break
        joined.extend(data[pos:pos + size])
        pos += size
    raise NanoodleError("Truncated GIF data sub-block")


def _indices(data, minimum, count):
    if not 2 <= minimum <= 8:
        raise NanoodleError("Invalid GIF LZW minimum code size")
    clear = 1 << minimum
    end = clear + 1
    first = [bytes([i]) for i in range(clear)] + [None, None]
    table = first.copy()
    width = minimum + 1
    bit = 0
    previous = None
    out = bytearray()
    while bit + width <= len(data) * 8:
        pos, shift = divmod(bit, 8)
        word = data[pos]
        if shift + width > 8:
            word |= data[pos + 1] << 8
        if shift + width > 16:
            word |= data[pos + 2] << 16
        code = (word >> shift) & ((1 << width) - 1)
        bit += width
        if code == clear:
            table = first.copy()
            previous = None
            width = minimum + 1
            continue
        if code == end:
            if len(out) != count:
                raise NanoodleError("GIF frame has the wrong number of pixels")
            return bytes(out)
        if code < len(table):
            phrase = table[code]
        elif code == len(table) and previous:
            phrase = previous + previous[:1]
        else:
            raise NanoodleError("Invalid GIF LZW dictionary reference")
        if len(out) + len(phrase) > count:
            raise NanoodleError("GIF frame exceeds its declared dimensions")
        out.extend(phrase)
        if previous and len(table) < 4096:
            table.append(previous + phrase[:1])
            if len(table) == 1 << width and width < 12:
                width += 1
        previous = phrase
    raise NanoodleError("Missing GIF LZW end code")


def decode_gif(data):
    from .sprite_codecs import _dimensions
    if len(data) < 13:
        raise NanoodleError("Truncated GIF header")
    width, height, flags, background, _ = struct.unpack_from("<HHBBB", data, 6)
    _dimensions(width, height)
    pos = 13

    def read_colors(flags, pos):
        size = 3 * (2 << (flags & 7))
        if pos + size > len(data):
            raise NanoodleError("Truncated GIF color table")
        return data[pos:pos + size], pos + size

    global_colors = None
    if flags & 128:
        global_colors, pos = read_colors(flags, pos)
    transparent = None
    while pos < len(data):
        marker = data[pos]
        pos += 1
        if marker == 0x3B:
            break
        if marker == 0x21 and pos < len(data):
            label = data[pos]
            payload, pos = _subblocks(data, pos + 1)
            if label == 0xF9:
                if len(payload) != 4:
                    raise NanoodleError("Invalid GIF graphic control block")
                transparent = payload[3] if payload[0] & 1 else None
            continue
        if marker != 0x2C or pos + 9 > len(data):
            raise NanoodleError("Invalid GIF image descriptor")
        x, y, w, h, flags = struct.unpack_from("<4HB", data, pos)
        pos += 9
        if not w or not h or x + w > width or y + h > height:
            raise NanoodleError("GIF frame exceeds its logical screen")
        colors = global_colors
        if flags & 128:
            colors, pos = read_colors(flags, pos)
        if not colors or pos >= len(data):
            raise NanoodleError("Missing GIF image data or color table")
        minimum = data[pos]
        payload, _ = _subblocks(data, pos + 1)
        indices = _indices(payload, minimum, w * h)
        color_count = len(colors) // 3
        if max(indices) >= color_count:
            raise NanoodleError("Invalid GIF color index")
        tables = [colors[i::3].ljust(256, b"\0") for i in range(3)]
        alpha = bytearray(b"\xff" * 256)
        if transparent is not None:
            alpha[transparent] = 0
            tables = [table[:transparent] + b"\0" + table[transparent + 1:] for table in tables]
        tables.append(bytes(alpha))
        background_pixel = bytes(4)
        if transparent is None and global_colors and background * 3 + 3 <= len(global_colors):
            background_pixel = global_colors[3 * background:3 * background + 3] + b"\xff"
        rgba = bytearray(background_pixel * (width * height))
        row_order = ((0, 8), (4, 8), (2, 4), (1, 2)) if flags & 64 else ((0, 1),)
        offset = 0
        for start, step in row_order:
            for line in range(start, h, step):
                row = indices[offset:offset + w]
                offset += w
                start_pixel = ((line + y) * width + x) * 4
                for channel, table in enumerate(tables):
                    a = start_pixel + channel
                    rgba[a:a + w * 4:4] = row.translate(table)
        return width, height, bytes(rgba)
    raise NanoodleError("GIF contains no image")
