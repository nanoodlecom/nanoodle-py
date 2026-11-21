"""VP8 keyframe edge filtering and WebP's interpolated Y'CbCr-to-RGBA output.

The loop filters implement RFC 6386 sections 15.2–15.4 following Go's
x/image/vp8/filter.go (Copyright 2014 The Go Authors, see LICENSE-go-image).
The fixed-point color coefficients follow libwebp's src/dsp/yuv.h
(Copyright 2010 Google Inc., see LICENSE-libwebp).
"""


def _clip8(value):
    return max(0, min(255, value))


def _signed8(value):
    return max(-128, min(127, value))


def _filter_edge(plane, start, along, across, size, limit, interior, hev, simple, outer):
    for q in range(start, start + size * along, along):
        p1, p0 = plane[q - 2 * across], plane[q - across]
        q0, q1 = plane[q], plane[q + across]
        if 2 * abs(p0 - q0) + (abs(p1 - q1) >> 1) > limit:
            continue
        if not simple:
            p2, q2 = plane[q - 3 * across], plane[q + 2 * across]
            if (abs(plane[q - 4 * across] - p2) > interior or abs(p2 - p1) > interior
                    or abs(p1 - p0) > interior or abs(q1 - q0) > interior
                    or abs(q2 - q1) > interior or abs(plane[q + 3 * across] - q2) > interior):
                continue
        if simple or abs(p1 - p0) > hev or abs(q1 - q0) > hev:
            adjustment = 3 * (q0 - p0) + _signed8(p1 - q1)
        elif outer:
            adjustment = _signed8(3 * (q0 - p0) + _signed8(p1 - q1))
            d0 = (27 * adjustment + 63) >> 7
            d1 = (18 * adjustment + 63) >> 7
            d2 = (9 * adjustment + 63) >> 7
            plane[q - 3 * across], plane[q + 2 * across] = _clip8(p2 + d2), _clip8(q2 - d2)
            plane[q - 2 * across], plane[q + across] = _clip8(p1 + d1), _clip8(q1 - d1)
            plane[q - across], plane[q] = _clip8(p0 + d0), _clip8(q0 - d0)
            continue
        else:
            adjustment = 3 * (q0 - p0)
            half = ((_signed8(adjustment + 4) >> 3) + 1) >> 1
            plane[q - 2 * across], plane[q + across] = _clip8(p1 + half), _clip8(q1 - half)
        plane[q - across] = _clip8(p0 + (_signed8(adjustment + 3) >> 3))
        plane[q] = _clip8(q0 - (_signed8(adjustment + 4) >> 3))


def filter_frame(y, u, v, mbw, mbh, simple, sharpness, levels, inner_edges):
    """Filter reconstructed macroblocks in place; levels include segment/mode deltas."""
    planes = [(y, 16, 16 * mbw)]
    if not simple:
        planes.extend(((u, 8, 8 * mbw), (v, 8, 8 * mbw)))
    for my in range(mbh):
        for mx in range(mbw):
            mb = my * mbw + mx
            level = min(levels[mb], 63)
            if level <= 0:
                continue
            interior = level
            if sharpness:
                interior = max(1, min(interior >> (2 if sharpness > 4 else 1), 9 - sharpness))
            limit = 2 * level + interior
            hev = int(level >= 15) + int(level >= 40)
            for plane, size, stride in planes:
                start = (my * stride + mx) * size
                # All vertical edges must be filtered before horizontal edges.
                for adjacent, along, across in ((mx, stride, 1), (my, 1, stride)):
                    if adjacent:
                        _filter_edge(plane, start, along, across, size, limit + 4,
                                     interior, hev, simple, True)
                    if inner_edges[mb]:
                        for offset in range(4, size, 4):
                            _filter_edge(plane, start + across * offset, along, across, size,
                                         limit, interior, hev, simple, False)


# Fixed-point Rec. 601 studio-range conversion used by WebP decoders.
_Y = tuple((19077 * x) >> 8 for x in range(256))
_RV = tuple(((26149 * x) >> 8) - 14234 for x in range(256))
_GU = tuple(8708 - ((6419 * x) >> 8) for x in range(256))
_GV = tuple((13320 * x) >> 8 for x in range(256))
_BU = tuple(((33050 * x) >> 8) - 17685 for x in range(256))
# All converted channels fall in [-277, 535]. Look up their saturated
# byte values directly in the per-pixel loop.
_CLIPPED = bytes(280) + bytes(range(256)) + b"\xff" * 280


def to_rgba(y, u, v, width, height, mbw):
    """Convert the cropped visible 4:2:0 image using WebP's centered bilinear chroma."""
    ystride, cstride = 16 * mbw, 8 * mbw
    cw, ch = (width + 1) // 2, (height + 1) // 2
    out = bytearray(b"\xff") * (width * height * 4)
    # At each pixel the closest chroma sample has weight 9/16, the two
    # adjacent samples have 3/16 each and the opposite corner has 1/16.
    columns = [
        (x // 2, min(cw - 1, x // 2 + 1) if x & 1 else max(0, x // 2 - 1))
        for x in range(width)
    ]
    dst = 0
    for row in range(height):
        near_row = row // 2
        far_row = min(ch - 1, near_row + 1) if row & 1 else max(0, near_row - 1)
        near = near_row * cstride
        far = far_row * cstride
        ur = [3 * u[near + c] + u[far + c] for c in range(cw)]
        vr = [3 * v[near + c] + v[far + c] for c in range(cw)]
        yi = row * ystride
        for primary, secondary in columns:
            uu = (3 * ur[primary] + ur[secondary] + 8) >> 4
            vv = (3 * vr[primary] + vr[secondary] + 8) >> 4
            luminance = _Y[y[yi]]
            out[dst] = _CLIPPED[280 + ((luminance + _RV[vv]) >> 6)]
            out[dst + 1] = _CLIPPED[280 + ((luminance + _GU[uu] - _GV[vv]) >> 6)]
            out[dst + 2] = _CLIPPED[280 + ((luminance + _BU[uu]) >> 6)]
            dst += 4
            yi += 1
    return bytes(out)
