"""VP8 intra-block prediction and inverse transforms (RFC 6386, §§ 12 and 14).

Adapted from golang.org/x/image/vp8 (predfunc.go and idct.go).
Copyright 2011 The Go Authors. BSD-3-Clause; see LICENSE-go-image.
"""

from itertools import pairwise

from .errors import NanoodleError

# A 256-entry window maps any pixel to clip8(pixel + delta), for a delta
# in [-255, 255]. Applying one true-motion row in bytes.translate is cheaper
# than clipping every sample in the Python interpreter.
_CLIPPED_RANGE = bytes(255) + bytes(range(256)) + b"\xff" * 255


def _clip(value):
    if 0 <= value <= 255:
        return value
    return 0 if value < 0 else 255


def _halves(edge):
    return [(a + b + 1) >> 1 for a, b in pairwise(edge)]


def _quarters(edge):
    return [(a + 2 * b + c + 2) >> 2 for a, b, c in zip(edge, edge[1:], edge[2:])]


def _vertical_right(top, left, corner):
    """Four rows in vertical-right mode. Horizontal-down transposes this."""
    halves = _halves([corner] + top[:4])
    quarters = _quarters(left[2::-1] + [corner] + top[:4])
    return [halves, quarters[2:6], [quarters[1]] + halves[:3], [quarters[0]] + quarters[2:5]]


def _directional_rows(mode, top, left, corner):
    if mode == 2:  # vertical
        return [_quarters([corner] + top[:5])] * 4
    if mode == 3:  # horizontal
        return [[v] * 4 for v in _quarters([corner] + left + left[-1:])]
    if mode == 4:  # right-down
        edge = _quarters(left[::-1] + [corner] + top[:4])
        return [edge[3-y:7-y] for y in range(4)]
    if mode == 5:  # vertical-right
        return _vertical_right(top, left, corner)
    if mode == 6:  # left-down
        edge = _quarters(top + top[-1:])
        return [edge[y:y+4] for y in range(4)]
    if mode == 7:  # vertical-left (bottom-right overhang uses three-sample averages)
        halves = _halves(top[:5])
        quarters = _quarters(top)
        return [halves, quarters[:4], halves[1:] + quarters[4:5], quarters[1:4] + quarters[5:6]]
    if mode == 8:  # horizontal-down
        return zip(*_vertical_right(left, top, corner))
    if mode == 9:  # horizontal-up
        halves = _halves(left)
        quarters = _quarters(left + left[-1:])
        edge = [v for pair in zip(halves, quarters) for v in pair] + left[-1:] * 4
        return [edge[2*y:2*y+4] for y in range(4)]
    raise NanoodleError("Invalid WebP prediction mode")


def predict(buf, stride, pos, n, mode, left_available, top_available):
    """Predict into an n×n block of a plane with initialized top and left borders."""
    top_pos = pos - stride
    top = buf[top_pos:top_pos + n]
    if mode == 0:
        # 4×4 blocks use both initialized borders, including at picture edges.
        use_left = left_available or n == 4
        use_top = top_available or n == 4
        count = (int(use_left) + int(use_top)) * n
        total = sum(top) if use_top else 0
        if use_left:
            total += sum(buf[pos + y * stride - 1] for y in range(n))
        value = (total + count // 2) // count if count else 128
        row = bytes([value]) * n
        for _ in range(n):
            buf[pos:pos + n] = row
            pos += stride
    elif mode == 1:  # true motion
        corner = buf[top_pos - 1]
        for _ in range(n):
            delta = buf[pos - 1] - corner
            start = 255 + delta
            buf[pos:pos + n] = top.translate(_CLIPPED_RANGE[start:start + 256])
            pos += stride
    elif n == 4:
        # The macroblock reader fills the top-right overhang in the workspace.
        top = list(buf[top_pos:top_pos + 8])
        left = [buf[pos + y * stride - 1] for y in range(4)]
        for row in _directional_rows(mode, top, left, buf[top_pos - 1]):
            buf[pos:pos + 4] = bytes(row)
            pos += stride
    elif mode == 2:
        for _ in range(n):
            buf[pos:pos + n] = top
            pos += stride
    elif mode == 3:
        for _ in range(n):
            buf[pos:pos + n] = bytes([buf[pos - 1]]) * n
            pos += stride
    else:
        raise NanoodleError("Invalid WebP prediction mode")


def _idct(a, b, c, d):
    # Fixed-point cos(pi/8)*sqrt(2) and sin(pi/8)*sqrt(2).
    even_sum, even_diff = a + c, a - c
    odd_diff = ((b * 35468) >> 16) - ((d * 85627) >> 16)
    odd_sum = ((b * 85627) >> 16) + ((d * 35468) >> 16)
    return even_sum + odd_sum, even_diff + odd_diff, even_diff - odd_diff, even_sum - odd_sum


def add_residual(buf, stride, pos, coeff):
    """Inverse-transform a row-major 4×4 block and add it to the prediction."""
    if not any(coeff[1:]):
        dc = (coeff[0] + 4) >> 3
        if dc:
            for _ in range(4):
                buf[pos:pos + 4] = bytes(_clip(v + dc) for v in buf[pos:pos + 4])
                pos += stride
        return

    columns = [_idct(*coeff[x::4]) for x in range(4)]
    for a, b, c, d in zip(*columns):
        row = _idct(a + 4, b, c, d)
        for i, v in enumerate(row):
            buf[pos + i] = _clip(buf[pos + i] + (v >> 3))
        pos += stride


def _wht(a, b, c, d):
    return a + d + b + c, a - d + b - c, a + d - b - c, a - d - b + c


def wht(coeff):
    """Inverse Walsh-Hadamard transform of the 16 Y2 coefficients into Y1 DCs."""
    columns = [_wht(*coeff[x::4]) for x in range(4)]
    out = []
    for a, b, c, d in zip(*columns):
        out.extend((value + 3) >> 3 for value in _wht(a, b, c, d))
    return out
