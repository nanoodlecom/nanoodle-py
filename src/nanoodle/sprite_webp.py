"""Decode a WebP sprite texture or the first displayed canvas of an animation."""

import struct

from .errors import NanoodleError
from .sprite_codecs import _dimensions


def _invalid(detail):
    return NanoodleError(f"Invalid WebP image: {detail}")


def _uint24(data, offset):
    return data[offset] | data[offset + 1] << 8 | data[offset + 2] << 16


def _chunks(data):
    pos = 0
    while pos < len(data):
        if len(data) - pos < 8:
            raise _invalid("incomplete chunk header")
        kind = bytes(data[pos:pos + 4])
        size = struct.unpack_from("<I", data, pos + 4)[0]
        pos += 8
        end = pos + size
        next_pos = end + (size & 1)
        if next_pos > len(data):
            raise _invalid("incomplete chunk payload")
        yield kind, data[pos:end]
        pos = next_pos


def _size(kind, payload):
    if kind == b"VP8L":
        if len(payload) < 5 or payload[0] != 0x2F:
            raise _invalid("invalid lossless frame header")
        bits = struct.unpack_from("<I", payload, 1)[0]
        if bits >> 29:
            raise _invalid("unknown lossless version")
        w, h = (bits & 0x3FFF) + 1, (bits >> 14 & 0x3FFF) + 1
    else:
        if len(payload) < 10 or payload[0] & 1 or payload[3:6] != b"\x9d\x01\x2a":
            raise _invalid("invalid VP8 keyframe")
        w, h = struct.unpack_from("<HH", payload, 6)
        w, h = w & 0x3FFF, h & 0x3FFF
    _dimensions(w, h)
    return w, h


def _alpha(payload, width, height):
    if not payload:
        raise _invalid("empty ALPH chunk")
    flags = payload[0]
    coding, predictor = flags & 3, flags >> 2 & 3
    if coding == 1:
        from .sprite_webp_vp8l import decode_alpha
        pixels = bytearray(decode_alpha(payload[1:], width, height))
    elif coding == 0:
        pixels = bytearray(payload[1:])
    else:
        raise _invalid("unsupported alpha compression")
    if len(pixels) != width * height:
        raise _invalid("wrong alpha plane size")
    if predictor == 0:
        return pixels

    # All three filters predict across the first row, and down the first
    # column. The remaining samples use the left, top, or clamped gradient.
    for x in range(1, width):
        pixels[x] = (pixels[x] + pixels[x - 1]) & 255
    for row in range(width, len(pixels), width):
        pixels[row] = (pixels[row] + pixels[row - width]) & 255
        if predictor == 1:
            for i in range(row + 1, row + width):
                pixels[i] = (pixels[i] + pixels[i - 1]) & 255
        elif predictor == 2:
            for i in range(row + 1, row + width):
                pixels[i] = (pixels[i] + pixels[i - width]) & 255
        else:
            for i in range(row + 1, row + width):
                pred = pixels[i - 1] + pixels[i - width] - pixels[i - width - 1]
                pixels[i] = (pixels[i] + max(0, min(255, pred))) & 255
    return pixels


def _decode_frame(kind, payload, alpha, size):
    w, h = _size(kind, payload)
    if size is not None and size != (w, h):
        raise _invalid("frame dimensions disagree with its header")
    if kind == b"VP8L":
        from .sprite_webp_vp8l import decode_vp8l
        if alpha is not None:
            raise _invalid("a lossless frame carries its own alpha")
        return decode_vp8l(payload)
    from .sprite_webp_vp8 import decode_vp8
    _, _, rgba = decode_vp8(payload)
    if alpha is not None:
        rgba = bytearray(rgba)
        rgba[3::4] = _alpha(alpha, w, h)
        rgba = bytes(rgba)
    return w, h, rgba


def _first_frame(data, canvas):
    if len(data) < 16:
        raise _invalid("incomplete animation frame header")
    x, y = 2 * _uint24(data, 0), 2 * _uint24(data, 3)
    width, height = _uint24(data, 6) + 1, _uint24(data, 9) + 1
    cw, ch = canvas
    if x + width > cw or y + height > ch:
        raise _invalid("animation frame exceeds its canvas")
    alpha = None
    for kind, payload in _chunks(data[16:]):
        if kind == b"ALPH":
            alpha = payload
        elif kind in (b"VP8 ", b"VP8L"):
            w, h, frame = _decode_frame(kind, payload, alpha, (width, height))
            if (w, h) == canvas:
                return w, h, frame
            # Browsers present the first subframe on a transparent canvas.
            # The ANIM background is an application hint and disposal applies
            # only when advancing to the next frame.
            out = bytearray(cw * ch * 4)
            for row in range(h):
                start = ((y + row) * cw + x) * 4
                out[start:start + w * 4] = frame[row * w * 4:(row + 1) * w * 4]
            return cw, ch, bytes(out)
    raise _invalid("animation frame has no image payload")


def decode_webp(raw):
    """Return (width, height, straight RGBA bytes) from RIFF WebP."""
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WEBP":
        raise _invalid("wrong RIFF signature")
    end = 8 + struct.unpack_from("<I", raw, 4)[0]
    if end < 12 or end > len(raw):
        raise _invalid("incomplete RIFF payload")

    canvas = None
    alpha = None
    animated = False
    for kind, payload in _chunks(memoryview(raw)[12:end]):
        if kind == b"VP8X":
            if len(payload) < 10 or canvas is not None:
                raise _invalid("invalid extended header")
            canvas = _uint24(payload, 4) + 1, _uint24(payload, 7) + 1
            _dimensions(*canvas)
            animated = bool(payload[0] & 2)
        elif kind == b"ALPH":
            alpha = payload
        elif kind in (b"VP8 ", b"VP8L"):
            return _decode_frame(kind, payload, alpha, canvas)
        elif kind == b"ANMF":
            if canvas is None or not animated:
                raise _invalid("missing animation canvas")
            return _first_frame(payload, canvas)
    raise _invalid("missing image payload")
