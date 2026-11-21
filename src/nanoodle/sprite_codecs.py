"""In-process texture decoding and sprite sheet encoding for the local nodes."""

import struct
import zlib

from .errors import NanoodleError

_PNG = b"\x89PNG\r\n\x1a\n"
_DEPTHS = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
          (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def _dimensions(width, height):
    max_width = 16384 if height <= 128 else 8192
    if not 1 <= width <= max_width or not 1 <= height <= 8192:
        raise NanoodleError("Sprite image dimensions are out of range")


def encode_png(width, height, rgba):
    """Encode straight-alpha 8-bit RGBA rows as a lossless PNG."""
    _dimensions(width, height)
    stride = width * 4
    if len(rgba) != stride * height:
        raise NanoodleError("Invalid sprite frame dimensions")

    def chunk(kind, body):
        crc = zlib.crc32(body, zlib.crc32(kind))
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    # Sprite sheets contain long transparent and repeated pixel runs. The
    # None filter compresses those directly and avoids a per-pixel encoder.
    pixels = memoryview(rgba)
    compressor = zlib.compressobj()
    compressed = bytearray()
    for offset in range(0, len(pixels), stride):
        compressed.extend(compressor.compress(b"\0"))
        compressed.extend(compressor.compress(pixels[offset:offset + stride]))
    compressed.extend(compressor.flush())
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return _PNG + chunk(b"IHDR", header) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


def _unfilter(kind, row, prev, bpp):
    if kind == 0:
        return
    if kind == 1:
        for i in range(bpp, len(row)):
            row[i] = (row[i] + row[i - bpp]) & 255
    elif kind == 2:
        for i, above in enumerate(prev):
            row[i] = (row[i] + above) & 255
    elif kind == 3:
        for i, above in enumerate(prev):
            left = row[i - bpp] if i >= bpp else 0
            row[i] = (row[i] + ((left + above) >> 1)) & 255
    elif kind == 4:
        for i, above in enumerate(prev):
            left = row[i - bpp] if i >= bpp else 0
            corner = prev[i - bpp] if i >= bpp else 0
            dl, da = above - corner, left - corner
            dc = abs(dl + da)
            dl, da = abs(dl), abs(da)
            predictor = left if dl <= da and dl <= dc else (above if da <= dc else corner)
            row[i] = (row[i] + predictor) & 255
    else:
        raise NanoodleError("Invalid PNG scanline filter")


def _rgba_row(row, width, depth, mode, palette, transparent):
    if mode == 6 and depth == 8:
        return row
    channels = _CHANNELS[mode]
    if depth < 8:
        mask = (1 << depth) - 1
        samples = bytes((row[i * depth // 8] >> (8 - depth - i * depth % 8)) & mask
                        for i in range(width))
    else:
        samples = row
    if mode == 3:
        if max(samples) >= len(palette[0]):
            raise NanoodleError("Invalid PNG palette index")
        result = bytearray(width * 4)
        for channel, table in enumerate(palette):
            result[channel::4] = bytes(samples).translate(table.ljust(256, b"\0"))
        return result

    # Apply transparency against the original sample precision before
    # reducing to the destination's eight bits per channel.
    pixel_size = channels * (2 if depth == 16 else 1)
    if transparent is None:
        alpha = b"\xff" * width
    else:
        alpha = bytes(0 if samples[i:i + pixel_size] == transparent else 255
                      for i in range(0, len(samples), pixel_size))
    if depth == 16:
        samples = bytes((s[0] + 128) // 257 for s in struct.iter_unpack(">H", row))
    elif depth < 8:
        samples = bytes(s * (255 // ((1 << depth) - 1)) for s in samples)
    if mode == 6:
        return samples
    result = bytearray(width * 4)
    if mode in (0, 4):
        gray = samples[::channels]
        result[0::4] = result[1::4] = result[2::4] = gray
        if mode == 4:
            alpha = samples[1::2]
    else:
        for channel in range(3):
            result[channel::4] = samples[channel::3]
    result[3::4] = alpha
    return result


def _decode_png(data):
    pos = len(_PNG)
    header = colors = transparency = None
    compressed = bytearray()
    while pos + 12 <= len(data):
        size, kind = struct.unpack_from(">I4s", data, pos)
        end = pos + 8 + size
        if end + 4 > len(data):
            raise NanoodleError("Truncated PNG chunk")
        body = memoryview(data)[pos + 8:end]
        checksum = struct.unpack_from(">I", data, end)[0]
        if zlib.crc32(body, zlib.crc32(kind)) != checksum:
            raise NanoodleError("Invalid PNG checksum")
        if kind == b"IHDR":
            if header is not None or pos != 8 or size != 13:
                raise NanoodleError("Invalid PNG header")
            header = struct.unpack(">IIBBBBB", body)
        elif header is None:
            raise NanoodleError("Missing PNG header")
        elif kind == b"PLTE":
            if size == 0 or size > 768 or size % 3:
                raise NanoodleError("Invalid PNG palette")
            colors = body.tobytes()
        elif kind == b"tRNS":
            transparency = body.tobytes()
        elif kind == b"IDAT":
            compressed.extend(body)
        elif kind == b"IEND":
            if size:
                raise NanoodleError("Invalid PNG end marker")
            break
        elif not kind[0] & 32:
            raise NanoodleError("Unsupported critical PNG chunk")
        pos = end + 4
    else:
        raise NanoodleError("Missing PNG end marker")

    width, height, depth, mode, compress, filtering, interlace = header
    _dimensions(width, height)
    if depth not in _DEPTHS.get(mode, ()) or compress or filtering or interlace not in (0, 1):
        raise NanoodleError("Unsupported PNG header")
    palette = transparent = None
    if mode == 3:
        if not colors or len(colors) // 3 > (1 << depth):
            raise NanoodleError("Missing or oversized PNG palette")
        alpha = transparency or b""
        if len(alpha) > len(colors) // 3:
            raise NanoodleError("Invalid PNG palette transparency")
        palette = [colors[i::3] for i in range(3)] + [alpha.ljust(len(colors) // 3, b"\xff")]
    elif transparency is not None:
        if mode not in (0, 2) or len(transparency) != 2 * _CHANNELS[mode]:
            raise NanoodleError("Invalid PNG transparency")
        keys = [entry[0] for entry in struct.iter_unpack(">H", transparency)]
        if any(k >= 1 << depth for k in keys):
            raise NanoodleError("Invalid PNG transparency sample")
        transparent = transparency if depth == 16 else bytes(keys)

    bits = depth * _CHANNELS[mode]
    passes = []
    expected_size = 0
    for x, y, dx, dy in _ADAM7 if interlace else ((0, 0, 1, 1),):
        columns, rows = len(range(x, width, dx)), len(range(y, height, dy))
        if columns and rows:
            row_bytes = (columns * bits + 7) // 8
            expected_size += (1 + row_bytes) * rows
            passes.append((x, y, dx, dy, columns, row_bytes))
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(compressed, expected_size + 1)
    except zlib.error as exc:
        raise NanoodleError("Invalid compressed PNG data") from exc
    if not inflater.eof or len(raw) != expected_size:
        raise NanoodleError("PNG scanline length does not match its dimensions")
    rgba = bytearray(width * height * 4)
    offset = 0
    for x, first_y, dx, dy, columns, row_bytes in passes:
        prev = bytearray(row_bytes)
        for y in range(first_y, height, dy):
            kind = raw[offset]
            offset += 1
            row = bytearray(raw[offset:offset + row_bytes])
            offset += row_bytes
            _unfilter(kind, row, prev, max(1, (bits + 7) // 8))
            expanded = _rgba_row(row, columns, depth, mode, palette, transparent)
            start = (y * width + x) * 4
            if dx == 1:
                rgba[start:start + columns * 4] = expanded
            else:
                for channel in range(4):
                    a = start + channel
                    rgba[a:a + columns * dx * 4:dx * 4] = expanded[channel::4]
            prev = row
    return width, height, bytes(rgba)


def decode_image(raw):
    """Decode an embedded texture or palette into straight RGBA pixels."""
    if raw.startswith(_PNG):
        return _decode_png(raw)
    if raw.startswith(b"\xff\xd8"):
        from .sprite_jpeg import decode_jpeg
        return decode_jpeg(raw)
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        from .sprite_gif import decode_gif
        return decode_gif(raw)
    raise NanoodleError("Expected a PNG, JPEG or GIF sprite texture")
