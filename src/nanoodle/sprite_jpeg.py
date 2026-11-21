"""Decode 8-bit Huffman-coded sequential and progressive JPEG textures (ITU-T T.81)."""

import math
from array import array

from .errors import NanoodleError

# Entropy-coded AC coefficients and quantization tables both use this order.
# fmt: off
_ZIGZAG = (
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
)
# fmt: on
_BASIS = tuple(
    tuple(math.cos(math.pi * (2 * x + 1) * k / 16) * (math.sqrt(0.5) if k == 0 else 1) / 2 for k in range(8))
    for x in range(8)
)


def _invalid(reason):
    raise NanoodleError("Invalid or unsupported JPEG: " + reason)


def _ceil(n, d):
    return (n + d - 1) // d


def _clip(value):
    if value < 0:
        return 0
    if value > 255:
        return 255
    return value


class _Bits:
    """Bits from one unstuffed entropy segment, ending at a restart or scan end."""

    def __init__(self, data):
        self.data = data
        self.pos = self.value = self.count = 0

    def fill(self, n):
        while self.count < n and self.pos < len(self.data):
            self.value = (self.value << 8) | self.data[self.pos]
            self.pos += 1
            self.count += 8

    def take(self, n):
        if self.count < n:
            self.fill(n)
            if self.count < n:
                _invalid("truncated entropy data")
        self.count -= n
        result = self.value >> self.count
        self.value &= (1 << self.count) - 1
        return result

    def signed(self, n):
        if n == 0:
            return 0
        bits = self.take(n)
        return bits if bits >= 1 << (n - 1) else bits + 1 - (1 << n)


class _Huffman:
    def __init__(self, counts, symbols):
        self.fast = [None] * 256
        self.long = {}
        code = pos = 0
        for size, count in enumerate(counts, 1):
            if code + count >= 1 << size:  # all-ones is reserved for padding
                _invalid("oversubscribed Huffman table")
            for _ in range(count):
                symbol = symbols[pos]
                self.long[size, code] = symbol
                if size <= 8:
                    start = code << (8 - size)
                    end = (code + 1) << (8 - size)
                    self.fast[start:end] = [(size, symbol)] * (end - start)
                pos += 1
                code += 1
            code <<= 1

    def read(self, bits):
        bits.fill(8)
        if bits.count >= 8:
            item = self.fast[bits.value >> (bits.count - 8)]
            if item is not None:
                size, value = item
                bits.take(size)
                return value
        code = 0
        for size in range(1, 17):
            code = (code << 1) | bits.take(1)
            value = self.long.get((size, code))
            if value is not None:
                return value
        _invalid("unknown Huffman symbol")


def _entropy_segments(data, pos):
    """Return bytes between restart markers and the offset of the next header."""
    segments = []
    buf = bytearray()
    while True:
        marker = data.find(b"\xff", pos)
        if marker < 0:
            _invalid("unterminated scan")
        buf.extend(data[pos:marker])
        pos = marker + 1
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos == len(data):
            _invalid("truncated marker")
        code = data[pos]
        pos += 1
        if code == 0:
            buf.append(0xFF)
        elif 0xD0 <= code <= 0xD7:
            if code != 0xD0 + len(segments) % 8:
                _invalid("out-of-order restart marker")
            segments.append(buf)
            buf = bytearray()
        else:
            segments.append(buf)
            return segments, marker


class _Component:
    def __init__(self, ident, sampling, quant_id):
        self.ident = ident
        self.h, self.v = sampling >> 4, sampling & 15
        if not 1 <= self.h <= 4 or not 1 <= self.v <= 4 or quant_id > 3:
            _invalid("invalid component sampling or quantizer")
        self.quant_id = quant_id
        self.quant = None
        self.coeffs = None
        self.plane = None
        self.levels = [-1] * 64

    def prepare(self, frame):
        self.width = _ceil(frame.width * self.h, frame.h)
        self.height = _ceil(frame.height * self.v, frame.v)
        self.block_cols = frame.cols * self.h
        self.block_rows = frame.rows * self.v
        self.stride = self.block_cols * 8
        self.plane = bytearray(self.stride * self.block_rows * 8)
        if frame.progressive:
            self.coeffs = array("h", [0]) * (self.block_cols * self.block_rows * 64)

    def draw_block(self, block, block_x, block_y):
        start = block_y * 8 * self.stride + block_x * 8
        if not any(block[1:]):
            value = _clip(math.floor(block[0] * self.quant[0] / 8 + 128.5))
            row = bytes([value]) * 8
            for i in range(8):
                at = start + i * self.stride
                self.plane[at : at + 8] = row
            return
        coeff = [0] * 64
        for i, val in enumerate(block):
            coeff[_ZIGZAG[i]] = val * self.quant[i]
        # Horizontal and vertical inverse transforms. On sparse sprite images
        # many rows have only DC frequency and can skip all multiplications.
        tmp = [0.0] * 64
        for row in range(0, 64, 8):
            a, b, c, d, e, f, g, h = coeff[row : row + 8]
            if not (b or c or d or e or f or g or h):
                tmp[row : row + 8] = [a * _BASIS[0][0]] * 8
            else:
                for x, w in enumerate(_BASIS):
                    tmp[row + x] = (
                        a * w[0] + b * w[1] + c * w[2] + d * w[3] + e * w[4] + f * w[5] + g * w[6] + h * w[7]
                    )
        for x in range(8):
            a, b, c, d, e, f, g, h = tmp[x::8]
            index = start + x
            for w in _BASIS:
                val = a * w[0] + b * w[1] + c * w[2] + d * w[3] + e * w[4] + f * w[5] + g * w[6] + h * w[7]
                self.plane[index] = _clip(math.floor(val + 128.5))
                index += self.stride

    def rows(self, frame):
        if self.h == frame.h and self.v == frame.v:
            source = memoryview(self.plane)
            for y in range(frame.height):
                yield source[y * self.stride : y * self.stride + frame.width]
            return

        # JPEG chroma samples are centered. Interpolate in integer arithmetic
        # and round once, after both axes, to avoid directional color fringes.
        def coords(size, count, factor, full):
            div = 2 * full
            for i in range(size):
                lo, frac = divmod((2 * i + 1) * factor - full, div)
                yield max(0, min(lo, count - 1)), max(0, min(lo + 1, count - 1)), div - frac, frac

        xs = list(coords(frame.width, self.width, self.h, frame.h))
        denom = 4 * frame.h * frame.v
        rows = {}
        for lo, hi, a, b in coords(frame.height, self.height, self.v, frame.v):
            for old in tuple(rows):
                if old != lo and old != hi:
                    del rows[old]
            for y in (lo, hi):
                if y not in rows:
                    row = memoryview(self.plane)[y * self.stride : y * self.stride + self.width]
                    rows[y] = [row[l] * wa + row[r] * wb for l, r, wa, wb in xs]
            upper, lower = rows[lo], rows[hi]
            yield bytes((u * a + v * b + denom // 2) // denom for u, v in zip(upper, lower))


class _Frame:
    def __init__(self, header, progressive):
        if len(header) < 6 or header[0] != 8:
            _invalid("only 8-bit DCT samples are supported")
        self.progressive = progressive
        self.height = int.from_bytes(header[1:3], "big")
        self.width = int.from_bytes(header[3:5], "big")
        from .sprite_codecs import _dimensions

        _dimensions(self.width, self.height)
        count = header[5]
        if count not in (1, 3, 4) or len(header) != 6 + 3 * count:
            _invalid("component count")
        self.components = {}
        for pos in range(6, len(header), 3):
            comp = _Component(*header[pos : pos + 3])
            if comp.ident in self.components:
                _invalid("duplicate component")
            self.components[comp.ident] = comp
        comps = self.components.values()
        self.h = max(c.h for c in comps)
        self.v = max(c.v for c in comps)
        self.cols = _ceil(self.width, self.h * 8)
        self.rows = _ceil(self.height, self.v * 8)

    def rgba(self, adobe_transform):
        comps = list(self.components.values())
        for c in comps:
            if c.plane is None or c.levels[0] < 0:
                _invalid("component missing DC scan")
            if c.coeffs is not None:
                for y in range(_ceil(c.height, 8)):
                    for x in range(_ceil(c.width, 8)):
                        at = (c.block_cols * y + x) * 64
                        c.draw_block(c.coeffs[at : at + 64], x, y)
                c.coeffs = None
        rows = [c.rows(self) for c in comps]
        out = bytearray(b"\xff") * (self.width * self.height * 4)
        pos = 0
        rgb = len(comps) == 3 and (adobe_transform == 0 or list(self.components) == list(b"RGB"))
        for channels in zip(*rows):
            if len(channels) == 1 or rgb:
                end = pos + self.width * 4
                for k in range(3):
                    out[pos + k : end : 4] = channels[min(k, len(channels) - 1)]
                pos = end
                continue
            if len(channels) == 4 and adobe_transform != 2:
                for c, m, y, k in zip(*channels):
                    if adobe_transform is None:
                        c, m, y, k = 255 - c, 255 - m, 255 - y, 255 - k
                    out[pos] = (c * k + 127) // 255
                    out[pos + 1] = (m * k + 127) // 255
                    out[pos + 2] = (y * k + 127) // 255
                    pos += 4
                continue
            for x, (y, cb, cr) in enumerate(zip(*channels[:3])):
                cb -= 128
                cr -= 128
                r = _clip(math.floor(y + 1.402 * cr + 0.5))
                g = _clip(math.floor(y - 0.344136 * cb - 0.714136 * cr + 0.5))
                b = _clip(math.floor(y + 1.772 * cb + 0.5))
                if len(channels) == 4:
                    k = channels[3][x]
                    r, g, b = (
                        ((255 - r) * k + 127) // 255,
                        ((255 - g) * k + 127) // 255,
                        ((255 - b) * k + 127) // 255,
                    )
                out[pos], out[pos + 1], out[pos + 2] = r, g, b
                pos += 4
        return self.width, self.height, bytes(out)


def _decode_block(bits, block, comp, dc, ac, first, last, high, low, eob):
    if first == 0:
        if high == 0:
            size = dc.read(bits)
            if size > 11:
                _invalid("DC coefficient size")
            comp.predictor += bits.signed(size)
            value = comp.predictor << low
            if not -32768 <= value <= 32767:
                _invalid("DC coefficient out of range")
            block[0] = value
        else:
            block[0] |= bits.take(1) << low
        first = 1
    if first > last:
        return eob
    if high == 0:
        if eob:
            return eob - 1
        index = first
        while index <= last:
            symbol = ac.read(bits)
            run, size = divmod(symbol, 16)
            if not size:
                if run != 15:
                    return (1 << run) + bits.take(run) - 1
                index += 16
                if index > last + 1:
                    _invalid("AC zero run outside block")
            else:
                index += run
                if index > last or size > 10:
                    _invalid("AC coefficient outside block")
                value = bits.signed(size) << low
                if not -32767 <= value <= 32767:
                    _invalid("AC coefficient out of range")
                block[index] = value
                index += 1
        return 0

    step = 1 << low

    def refine(index):
        val = block[index]
        if bits.take(1) and not (abs(val) & step):
            block[index] = val + step if val > 0 else val - step

    index = first
    if not eob:
        while index <= last:
            symbol = ac.read(bits)
            run, size = divmod(symbol, 16)
            if size > 1:
                _invalid("AC refinement size")
            new = 0
            if size:
                new = step if bits.take(1) else -step
            elif run < 15:
                eob = (1 << run) + bits.take(run)
                break
            # The run counts zero coefficients only. Refine nonzeros passed
            # on the way to the next insertion or the 16th zero of a ZRL.
            while index <= last:
                if block[index]:
                    refine(index)
                elif run == 0:
                    break
                else:
                    run -= 1
                index += 1
            if index > last:
                _invalid("AC refinement run outside block")
            if new:
                block[index] = new
            index += 1
    if eob:
        while index <= last:
            if block[index]:
                refine(index)
            index += 1
        return eob - 1
    return 0


def _scan(frame, header, segments, restart, quant, tables):
    count = header[0] if header else 0
    if not 1 <= count <= len(frame.components) or len(header) != 4 + 2 * count:
        _invalid("scan header")
    first, last, approx = header[-3:]
    high, low = divmod(approx, 16)
    if not frame.progressive:
        if (first, last, approx) != (0, 63, 0):
            _invalid("sequential scan parameters")
    elif (
        first > last
        or last > 63
        or low > 13
        or high > 13
        or (first == 0 and last != 0)
        or (first and count != 1)
        or (high and high != low + 1)
    ):
        _invalid("progressive scan parameters")
    components = []
    for i in range(count):
        cid, tid = header[1 + i * 2 : 3 + i * 2]
        comp = frame.components.get(cid)
        if comp is None or any(c is comp for c, _, _ in components):
            _invalid("unknown or duplicate scan component")
        dc = tables.get(tid >> 4)
        ac = tables.get(16 | (tid & 15))
        if (first == 0 and high == 0 and dc is None) or (last != 0 and ac is None):
            _invalid("missing Huffman table")
        if comp.quant is None:
            comp.quant = quant.get(comp.quant_id)
            if comp.quant is None:
                _invalid("missing quantization table")
        expected = high if high else -1
        if any(level != expected for level in comp.levels[first : last + 1]):
            _invalid("unexpected scan order")
        comp.levels[first : last + 1] = [low] * (last + 1 - first)
        components.append((comp, dc, ac))

    single = count == 1
    if single:
        comp = components[0][0]
        cols = _ceil(frame.width * comp.h, 8 * frame.h)
        rows = _ceil(frame.height * comp.v, 8 * frame.v)
    else:
        cols, rows = frame.cols, frame.rows
    total = cols * rows
    stride = restart or total
    if len(segments) != _ceil(total, stride):
        _invalid("restart interval does not match scan")
    # Every DC scan needs at least one entropy bit per component block.
    # Reject impossible buffers before allocating padded image planes.
    if first == 0:
        blocks_per_mcu = 1 if single else sum(c.h * c.v for c, _, _ in components)
        if sum(map(len, segments)) * 8 < total * blocks_per_mcu:
            _invalid("truncated DC scan")
    for comp, _, _ in components:
        if comp.plane is None:
            comp.prepare(frame)
    for seg_idx, segment in enumerate(segments):
        bits = _Bits(segment)
        for comp, _, _ in components:
            comp.predictor = 0
        eob = 0
        for mcu in range(seg_idx * stride, min(total, (seg_idx + 1) * stride)):
            my, mx = divmod(mcu, cols)
            for comp, dc, ac in components:
                nx, ny = (1, 1) if single else (comp.h, comp.v)
                for by in range(my * ny, (my + 1) * ny):
                    for bx in range(mx * nx, (mx + 1) * nx):
                        at = (by * comp.block_cols + bx) * 64
                        block = [0] * 64 if comp.coeffs is None else memoryview(comp.coeffs)[at : at + 64]
                        eob = _decode_block(bits, block, comp, dc, ac, first, last, high, low, eob)
                        if comp.coeffs is None:
                            if eob:
                                _invalid("progressive run in sequential scan")
                            comp.draw_block(block, bx, by)
        if eob:
            _invalid("end-of-band run crosses restart or scan end")


def decode_jpeg(data):
    """Return (width, height, tightly packed RGBA bytes) for a JPEG texture."""
    data = bytes(data)
    if not data.startswith(b"\xff\xd8"):
        _invalid("missing SOI")
    frame = adobe_transform = None
    quant, tables = {}, {}
    pos, restart = 2, 0
    while pos < len(data):
        if data[pos] != 0xFF:
            _invalid("expected marker")
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            break
        code = data[pos]
        pos += 1
        if code == 0xD9:
            if frame is None:
                _invalid("missing frame")
            return frame.rgba(adobe_transform)
        if code == 0x01:
            continue
        if pos + 2 > len(data) or code in (0, 0xD8) or 0xD0 <= code <= 0xD7:
            _invalid("unexpected marker")
        size = int.from_bytes(data[pos : pos + 2], "big")
        end = pos + size
        if size < 2 or end > len(data):
            _invalid("truncated segment")
        header = data[pos + 2 : end]
        pos = end
        if code in (0xC0, 0xC1, 0xC2):
            if frame is not None:
                _invalid("multiple frames")
            frame = _Frame(header, code == 0xC2)
        elif code == 0xDA:
            if frame is None:
                _invalid("scan before frame")
            segments, pos = _entropy_segments(data, pos)
            _scan(frame, header, segments, restart, quant, tables)
        elif code == 0xDB:
            at = 0
            while at < len(header):
                precision, ident = divmod(header[at], 16)
                at += 1
                n = 64 * (precision + 1)
                if precision > 1 or ident > 3 or at + n > len(header):
                    _invalid("quantization table")
                values = (
                    header[at : at + n]
                    if not precision
                    else tuple(int.from_bytes(header[k : k + 2], "big") for k in range(at, at + n, 2))
                )
                if 0 in values:
                    _invalid("zero quantizer")
                quant[ident] = values
                at += n
        elif code == 0xC4:
            at = 0
            while at < len(header):
                if at + 17 > len(header):
                    _invalid("Huffman table header")
                ident = header[at]
                counts = header[at + 1 : at + 17]
                n = sum(counts)
                at += 17
                if ident not in (0, 1, 2, 3, 16, 17, 18, 19) or not 1 <= n <= 256 or at + n > len(header):
                    _invalid("Huffman table symbols")
                tables[ident] = _Huffman(counts, header[at : at + n])
                at += n
        elif code == 0xDD:
            if len(header) != 2:
                _invalid("restart interval")
            restart = int.from_bytes(header, "big")
        elif code == 0xEE and header.startswith(b"Adobe") and len(header) >= 12:
            adobe_transform = header[11]
        elif (code & 0xF0) != 0xE0 and code != 0xFE:
            _invalid(f"unsupported coding marker FF{code:02X}")
    _invalid("missing EOI")
