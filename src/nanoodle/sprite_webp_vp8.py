"""Decode VP8 keyframes for WebP (RFC 6386).

Adapted from golang.org/x/image/vp8 v0.31.0.
Copyright 2011 The Go Authors. BSD-3-Clause; see LICENSE-go-image.
"""

from .errors import NanoodleError
from .sprite_codecs import _dimensions
from .sprite_webp_vp8_filter import filter_frame, to_rgba
from .sprite_webp_vp8_pixels import add_residual, predict, wht
from .sprite_webp_vp8_tables import (
    DEFAULT_TOKEN,
    DEQUANT_AC,
    DEQUANT_DC,
    PRED_MODES,
    TOKEN_UPDATES,
)

_BANDS = (0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0)
_ZIGZAG = (0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15)
_CATEGORY_BITS = (
    (173, 148, 140), (176, 155, 140, 135), (180, 157, 141, 134, 130),
    (254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129),
)
_SHIFT = tuple(8 - (r + 1).bit_length() for r in range(127))
# Each scratch row fits a left border, 16 luma samples and a 4-sample overhang.
_STRIDE = 24
_ORIGIN = _STRIDE + 1


class _Partition:
    """VP8 Boolean range decoder, with one independent cursor per partition."""

    __slots__ = ("bits", "count", "data", "pos", "range_minus_one")

    def __init__(self, data):
        self.data = data
        self.pos = self.bits = self.count = 0
        self.range_minus_one = 254

    def bit(self, probability=128):
        if self.count < 8:
            if self.pos >= len(self.data):
                raise NanoodleError("Truncated VP8 entropy partition")
            self.bits |= self.data[self.pos] << (8 - self.count)
            self.pos += 1
            self.count += 8
        span = self.range_minus_one
        split = ((span * probability) >> 8) + 1
        one = self.bits >= split << 8
        if one:
            span -= split
            self.bits -= split << 8
        else:
            span = split - 1
        if span < 127:
            shift = _SHIFT[span]
            span = ((span + 1) << shift) - 1
            self.bits <<= shift
            self.count -= shift
        self.range_minus_one = span
        return one

    def uint(self, bits):
        value = 0
        for _ in range(bits):
            value = (value << 1) | self.bit()
        return value

    def optional_delta(self, bits):
        if not self.bit():
            return 0
        value = self.uint(bits)
        return -value if self.bit() else value


def _token_partitions(data, count):
    position = (count - 1) * 3
    if len(data) < position:
        raise NanoodleError("Truncated VP8 token partition lengths")
    parts = []
    for i in range(count - 1):
        length = int.from_bytes(data[3*i:3*i+3], "little")
        if position + length > len(data):
            raise NanoodleError("Invalid VP8 token partition length")
        parts.append(_Partition(data[position:position + length]))
        position += length
    parts.append(_Partition(data[position:]))
    return parts


def _quant_index(value, limit=127):
    return max(0, min(limit, value))


def _quantizers(control, segment_quant, absolute):
    base = control.uint(7)
    ydc, y2dc, y2ac, uvdc, uvac = (control.optional_delta(4) for _ in range(5))
    quants = []
    for delta in segment_quant:
        q = delta if absolute else base + delta

        quants.append((
            (DEQUANT_DC[_quant_index(q + ydc)], DEQUANT_AC[_quant_index(q)]),
            (DEQUANT_DC[_quant_index(q + y2dc)] * 2, max(8, DEQUANT_AC[_quant_index(q + y2ac)] * 155 // 100)),
            (DEQUANT_DC[_quant_index(q + uvdc, 117)], DEQUANT_AC[_quant_index(q + uvac)]),
        ))
    return quants


def _block_coefficients(reader, probabilities, plane, context, quant, start, coeff):
    """Decode DC/AC tokens into a 4×4 block; report its nonzero context."""
    base = plane * 8 * 3 * 11
    index = start
    p = base + (_BANDS[index] * 3 + context) * 11
    bit = reader.bit
    if not bit(probabilities[p]):
        return 0
    while index < 16:
        position = _ZIGZAG[index]
        index += 1
        next_band = base + _BANDS[index] * 33
        if not bit(probabilities[p + 1]):
            p = next_band
            continue
        if not bit(probabilities[p + 2]):
            magnitude = 1
            p = next_band + 11
        else:
            if not bit(probabilities[p + 3]):
                magnitude = (3 + bit(probabilities[p + 5])) if bit(probabilities[p + 4]) else 2
            elif not bit(probabilities[p + 6]):
                if bit(probabilities[p + 7]):
                    magnitude = 7 + 2 * bit(165) + bit(145)
                else:
                    magnitude = 5 + bit(159)
            else:
                high = bit(probabilities[p + 8])
                category = 2 * high + bit(probabilities[p + 9 + high])
                magnitude = 0
                for probability in _CATEGORY_BITS[category]:
                    magnitude = magnitude * 2 + bit(probability)
                magnitude += 3 + (8 << category)
            p = next_band + 22
        value = magnitude * quant[position != 0]
        if bit():
            value = -value
        coeff[position] = ((value + 32768) & 65535) - 32768
        if index == 16 or not bit(probabilities[p]):
            break
    return 1


def _luma_modes(reader, above, left, large):
    if large:
        if not reader.bit(156):
            mode = 2 if reader.bit(163) else 0
        else:
            mode = 1 if reader.bit() else 3
        above[:] = left[:] = [mode] * 4
        return [mode]
    modes = []
    for y in range(4):
        mode = left[y]
        for x in range(4):
            p = (above[x] * 10 + mode) * 9
            if not reader.bit(PRED_MODES[p]):
                mode = 0
            elif not reader.bit(PRED_MODES[p+1]):
                mode = 1
            elif not reader.bit(PRED_MODES[p+2]):
                mode = 2
            elif not reader.bit(PRED_MODES[p+3]):
                if not reader.bit(PRED_MODES[p+4]):
                    mode = 3
                else:
                    mode = 5 if reader.bit(PRED_MODES[p+5]) else 4
            elif not reader.bit(PRED_MODES[p+6]):
                mode = 6
            elif not reader.bit(PRED_MODES[p+7]):
                mode = 7
            else:
                mode = 9 if reader.bit(PRED_MODES[p+8]) else 8
            modes.append(mode)
            above[x] = mode
        left[y] = mode
    return modes


def _chroma_mode(reader):
    if not reader.bit(142):
        return 0
    if not reader.bit(114):
        return 2
    return 1 if reader.bit(183) else 3


def _workspace(plane, stride, mbx, mby, size):
    pos = (mby * stride + mbx) * size
    block = bytearray(_STRIDE * (size + 1))
    for row in range(size):
        block[(row + 1) * _STRIDE] = plane[pos + row*stride - 1] if mbx else 129
    if not mby:
        block[:size+5] = bytes([127]) * (size + 5)
    else:
        top = pos - stride
        block[0] = plane[top-1] if mbx else 129
        block[1:size+1] = plane[top:top+size]
        if size == 16:
            if (mbx + 1) * 16 == stride:
                block[17:21] = block[16:17] * 4
            else:
                block[17:21] = plane[top+16:top+20]
    if size == 16:
        for row in (4, 8, 12):
            off = row * _STRIDE + 17
            block[off:off + 4] = block[17:21]
    return block, pos


def _render_plane(plane, stride, mbx, mby, size, modes, groups):
    block, pos = _workspace(plane, stride, mbx, mby, size)
    if len(modes) == 1:
        predict(block, _STRIDE, _ORIGIN, size, modes[0], bool(mbx), bool(mby))
    cells = size // 4
    for i, coeff in enumerate(groups):
        offset = _ORIGIN + (i // cells) * 4 * _STRIDE + (i % cells) * 4
        if len(modes) > 1:
            predict(block, _STRIDE, offset, 4, modes[i], True, True)
        if coeff is not None and any(coeff):
            add_residual(block, _STRIDE, offset, coeff)
    for row in range(size):
        off = _ORIGIN + row * _STRIDE
        plane[pos:pos + size] = block[off:off + size]
        pos += stride


def _residuals(reader, probs, quant, large, skipped, above, left):
    groups = [None] * 24
    if skipped:
        left[:8] = above[:8] = [0] * 8
        if large:
            left[8] = above[8] = 0
        return groups, False
    dc = [0] * 16
    if large:
        nz = _block_coefficients(reader, probs, 1, left[8] + above[8], quant[1], 0, dc)
        left[8] = above[8] = nz
        if nz:
            dc = wht(dc)
    any_coeff = False
    for plane, offset, n, ctx, factor in ((0 if large else 3, 0, 4, 0, quant[0]),
                                          (2, 16, 2, 4, quant[2]), (2, 20, 2, 6, quant[2])):
        for row in range(n):
            for col in range(n):
                index = offset + row * n + col
                coeff = [0] * 16
                start = int(large and index < 16)
                if start:
                    coeff[0] = dc[index]
                nz = _block_coefficients(reader, probs, plane, left[ctx+row] + above[ctx+col], factor, start, coeff)
                above[ctx+col] = left[ctx+row] = nz
                if nz or coeff[0]:
                    any_coeff = True
                    groups[index] = coeff
    return groups, any_coeff


def decode_vp8(payload):
    """Decode the single visible intra frame in a WebP VP8 chunk into RGBA."""
    data = memoryview(payload)
    if len(data) < 10 or data[3:6] != b"\x9d\x01\x2a":
        raise NanoodleError("Invalid VP8 keyframe header")
    tag = int.from_bytes(data[:3], "little")
    profile = (tag >> 1) & 7
    if tag & 1 or not (tag & 16) or profile > 3:
        raise NanoodleError("Unsupported VP8 frame type or profile")
    width = int.from_bytes(data[6:8], "little") & 0x3fff
    height = int.from_bytes(data[8:10], "little") & 0x3fff
    _dimensions(width, height)
    mbw, mbh = (width + 15) // 16, (height + 15) // 16
    end_control = 10 + (tag >> 5)
    if end_control > len(data):
        raise NanoodleError("Invalid VP8 control partition length")
    control = _Partition(data[10:end_control])
    control.uint(2)  # Color-space and clamp flags; WebP keyframes use YUV420.
    segment_quant, segment_filter = [0] * 4, [0] * 4
    segment_probs = [255] * 3
    update_map = False
    use_segments = control.bit()
    absolute = use_segments  # A keyframe resets enabled segments to absolute mode.
    if use_segments:
        update_map = control.bit()
        if control.bit():
            absolute = control.bit()
            segment_quant = [control.optional_delta(7) for _ in range(4)]
            segment_filter = [control.optional_delta(6) for _ in range(4)]
        if update_map:
            segment_probs = [control.uint(8) if control.bit() else 255 for _ in range(3)]

    simple, level, sharpness = control.bit(), control.uint(6), control.uint(3)
    ref_delta = mode_delta = 0
    if control.bit() and control.bit():
        ref_delta, _, _, _ = (control.optional_delta(6) for _ in range(4))
        mode_delta, _, _, _ = (control.optional_delta(6) for _ in range(4))
    parts = _token_partitions(data[end_control:], 1 << control.uint(2))
    quantizers = _quantizers(control, segment_quant, absolute)
    control.bit()  # Refresh entropy probabilities for later frames.
    probabilities = bytearray(DEFAULT_TOKEN)
    for i, p in enumerate(TOKEN_UPDATES):
        if control.bit(p):
            probabilities[i] = control.uint(8)
    skip_prob = control.uint(8) if control.bit() else None

    ystride = 16 * mbw
    uvstride = 8 * mbw
    y = bytearray(ystride * 16 * mbh)
    u = bytearray(uvstride * 8 * mbh)
    v = bytearray(len(u))
    above_modes = [[0] * 4 for _ in range(mbw)]
    above_nz = [[0] * 9 for _ in range(mbw)]
    levels, inner = bytearray(mbw * mbh), bytearray(mbw * mbh)
    for mby in range(mbh):
        left_modes, left_nz = [0] * 4, [0] * 9
        reader = parts[mby % len(parts)]
        for mbx in range(mbw):
            segment = 0
            if update_map:
                if control.bit(segment_probs[0]):
                    segment = 2 + control.bit(segment_probs[2])
                else:
                    segment = int(control.bit(segment_probs[1]))
            skipped = skip_prob is not None and control.bit(skip_prob)
            large = control.bit(145)
            modes = _luma_modes(control, above_modes[mbx], left_modes, large)
            chroma = [_chroma_mode(control)]
            groups, residual = _residuals(reader, probabilities, quantizers[segment], large, skipped,
                                          above_nz[mbx], left_nz)
            _render_plane(y, ystride, mbx, mby, 16, modes, groups[:16])
            _render_plane(u, uvstride, mbx, mby, 8, chroma, groups[16:20])
            _render_plane(v, uvstride, mbx, mby, 8, chroma, groups[20:24])
            base = segment_filter[segment] + (0 if absolute else level)
            idx = mby * mbw + mbx
            levels[idx] = max(0, min(63, base + ref_delta + (0 if large else mode_delta)))
            inner[idx] = (not large) or residual
    if level:
        filter_frame(y, u, v, mbw, mbh, simple, sharpness, levels, inner)
    return width, height, to_rgba(y, u, v, width, height, mbw)
