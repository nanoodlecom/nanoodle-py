"""Decode WebP lossless image streams (RFC 9649, section 3) to straight RGBA."""

import sys
from array import array

from .errors import NanoodleError
from .sprite_codecs import _dimensions

# RFC 9649 section 3.6.2.2.1. Each byte encodes (row << 4) + column + 7.
_DISTANCE_PLANE = bytes.fromhex(
    "17 08 18 16 27 09 28 26 19 15 29 25 37 0a 38 36 1a 14 39 35 "
    "2a 24 47 0b 48 46 1b 13 3a 34 49 45 2b 23 57 4a 44 3b 33 0c "
    "58 56 1c 12 59 55 2c 22 4b 43 5a 54 3c 32 67 0d 68 66 1d 11 "
    "69 65 2d 21 5b 53 4c 42 6a 64 3d 31 77 0e 78 76 5c 52 1e 10 "
    "6b 63 4d 41 79 75 2e 20 7a 74 3e 30 6c 62 5d 51 0f 7b 73 4e "
    "40 1f 2f 6d 61 3f 7c 72 5e 50 4f 7d 71 6e 60 5f 7e 70 6f 7f "
)
_LENGTH_ORDER = (17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
_SIGNED = tuple(range(128)) + tuple(range(-128, 0))
_MASK = 0xff00ff


class _Bits:
    """A little-endian reservoir. Prefix lookup may peek past the last bit,
    but consuming an absent bit always raises a truncated-stream error.
    """
    __slots__ = ("count", "data", "index", "word")

    def __init__(self, source):
        self.data = memoryview(source)
        self.index = self.word = self.count = 0

    def fill(self):
        end = min(self.index + 4, len(self.data))
        self.word |= int.from_bytes(self.data[self.index:end], "little") << self.count
        self.count += (end - self.index) * 8
        self.index = end

    def read(self, size):
        if self.count < size:
            self.fill()
            if self.count < size:
                raise NanoodleError("Truncated WebP lossless stream")
        value = self.word & ((1 << size) - 1)
        self.word >>= size
        self.count -= size
        return value


class _Tree:
    """Canonical prefix codes with a short-code lookup and a sparse long-code table."""
    __slots__ = ("long", "mask", "max_bits", "sole", "table", "width")

    def __init__(self, lengths):
        counts = [0] * 16
        for n in lengths:
            counts[n] += 1
        n_used = len(lengths) - counts[0]
        if n_used < 1:
            raise NanoodleError("Empty WebP prefix code")
        if n_used == 1:
            self.sole = next(i for i, n in enumerate(lengths) if n)
            return
        self.sole = -1
        codes = [0] * 16
        free = 1
        counts[0] = 0
        for n in range(1, 16):
            free = 2 * free - counts[n]
            codes[n] = (codes[n - 1] + counts[n - 1]) << 1
            if free < 0:
                raise NanoodleError("Oversubscribed WebP prefix code")
        if free:
            raise NanoodleError("Incomplete WebP prefix code")
        self.max_bits = max(lengths)
        self.width = min(8, self.max_bits)
        self.mask = (1 << self.width) - 1
        self.table = [0] * (self.mask + 1)
        self.long = {}
        for symbol, n in enumerate(lengths):
            if not n:
                continue
            code = codes[n]
            codes[n] += 1
            reversed_code = int(format(code, f"0{n}b")[::-1], 2)
            if n <= self.width:
                entry = symbol << 4 | n
                for i in range(reversed_code, len(self.table), 1 << n):
                    self.table[i] = entry
            else:
                self.long[1 << n | reversed_code] = symbol

    def symbol(self, bits):
        if self.sole >= 0:
            return self.sole
        if bits.count < self.max_bits:
            bits.fill()
        entry = self.table[bits.word & self.mask]
        if entry:
            n = entry & 15
            if bits.count < n:
                raise NanoodleError("Truncated WebP prefix code")
            bits.count -= n
            bits.word >>= n
            return entry >> 4
        for n in range(self.width + 1, min(bits.count, self.max_bits) + 1):
            mark = 1 << n
            code = mark | (bits.word & (mark - 1))
            if code in self.long:
                bits.count -= n
                bits.word >>= n
                return self.long[code]
        raise NanoodleError("Invalid WebP prefix code")


def _read_tree(bits, size):
    lengths = [0] * size
    if bits.read(1):
        pair = bits.read(1)
        a = bits.read(8 if bits.read(1) else 1)
        b = bits.read(8) if pair else a
        if max(a, b) >= size:
            raise NanoodleError("WebP prefix symbol outside alphabet")
        lengths[a] = lengths[b] = 1
    else:
        alphabet_lengths = [0] * 19
        for sym in _LENGTH_ORDER[:4 + bits.read(4)]:
            alphabet_lengths[sym] = bits.read(3)
        alphabet = _Tree(alphabet_lengths)
        limit = 2 + bits.read(2 + 2 * bits.read(3)) if bits.read(1) else size
        if limit > size:
            raise NanoodleError("WebP code-length count exceeds alphabet")
        position = 0
        previous = 8
        for _ in range(limit):
            sym = alphabet.symbol(bits)
            if sym < 16:
                lengths[position] = sym
                position += 1
                if sym:
                    previous = sym
            else:
                if sym == 16:
                    length, repeats = previous, 3 + bits.read(2)
                elif sym == 17:
                    length, repeats = 0, 3 + bits.read(3)
                else:
                    length, repeats = 0, 11 + bits.read(7)
                if position + repeats > size:
                    raise NanoodleError("WebP code-length repeat exceeds alphabet")
                lengths[position:position + repeats] = [length] * repeats
                position += repeats
            if position == size:
                break
    return _Tree(lengths)


def _prefix_value(bits, prefix):
    if prefix < 4:
        return prefix + 1
    shift = (prefix - 2) >> 1
    return ((2 + (prefix & 1)) << shift) + bits.read(shift) + 1


def _tiles(n, shift):
    return (n + (1 << shift) - 1) >> shift


def _image(bits, width, height, main=False):
    ncache = bits.read(4) if bits.read(1) else None
    if ncache is not None and not 1 <= ncache <= 11:
        raise NanoodleError("Invalid WebP color cache size")
    cache = [0] * (1 << ncache) if ncache else []
    group_ids = None
    groups_used = {0}
    if main and bits.read(1):
        shift = 2 + bits.read(3)
        cols = _tiles(width, shift)
        entropy = _image(bits, cols, _tiles(height, shift))
        group_ids = array("H", ((color >> 8) & 65535 for color in entropy))
        groups_used = set(group_ids)
    groups = {}
    for i in range(max(groups_used) + 1):
        group = tuple(_read_tree(bits, size) for size in (280 + len(cache), 256, 256, 256, 40))
        if i in groups_used:
            groups[i] = group
    count = width * height
    out = array("I")
    position = 0
    planes = [max(1, (v >> 4) * width + (v & 15) - 7) for v in _DISTANCE_PLANE]
    if group_ids is None:
        green, red, blue, alpha, distance_tree = groups[0]
    while position < count:
        if group_ids is not None:
            y, x = divmod(position, width)
            gid = group_ids[(y >> shift) * cols + (x >> shift)]
            green, red, blue, alpha, distance_tree = groups[gid]
        token = green.symbol(bits)
        start = position
        if token < 256:
            r = red.symbol(bits)
            b = blue.symbol(bits)
            a = alpha.symbol(bits)
            out.append((a << 24) | (r << 16) | (token << 8) | b)
            position += 1
        elif token < 280:
            length = _prefix_value(bits, token - 256)
            code = _prefix_value(bits, distance_tree.symbol(bits))
            distance = planes[code - 1] if code <= 120 else code - 120
            if distance > position or length > count - position:
                raise NanoodleError("WebP backward reference outside image")
            # Repeated, overlapping copies are common on flat areas.
            block = out[position - distance:position - distance + min(length, distance)]
            if length > distance:
                block = (block * ((length + distance - 1) // distance))[:length]
            out.extend(block)
            position += length
        else:
            out.append(cache[token - 280])
            position += 1
        if cache:
            for i in range(start, position):
                pixel = out[i]
                cache[((pixel * 0x1e35a7bd) & 0xffffffff) >> (32 - ncache)] = pixel
    return out


def _add(a, b):
    # Add alternate lanes separately to discard carry at channel boundaries.
    return (((a & _MASK) + (b & _MASK)) & _MASK) | (((a & 0xff00ff00) + (b & 0xff00ff00)) & 0xff00ff00)


def _average(a, b):
    return (a & b) + (((a ^ b) & 0xfefefefe) >> 1)


def _predict(mode, left, top, tl, tr):
    if mode == 0:
        return 0xff000000
    if mode == 1:
        return left
    if mode == 2:
        return top
    if mode == 3:
        return tr
    if mode == 4:
        return tl
    if mode == 5:
        return _average(_average(left, tr), top)
    if mode == 6:
        return _average(left, tl)
    if mode == 7:
        return _average(left, top)
    if mode == 8:
        return _average(tl, top)
    if mode == 9:
        return _average(top, tr)
    if mode == 10:
        return _average(_average(left, tl), _average(top, tr))
    if mode == 11:
        # Compare each candidate with the unconstrained prediction L + T - TL.
        dl = dt = 0
        for s in (0, 8, 16, 24):
            a = (tl >> s) & 255
            dl += abs(((top >> s) & 255) - a)
            dt += abs(((left >> s) & 255) - a)
        return left if dl < dt else top
    if mode not in (12, 13):
        raise NanoodleError("Invalid WebP pixel predictor")
    value = 0
    if mode == 13:
        left = _average(left, top)
    for s in (0, 8, 16, 24):
        a, b, c = (left >> s) & 255, (top >> s) & 255, (tl >> s) & 255
        if mode == 12:
            k = a + b - c
        else:
            diff = a - c
            k = a + ((diff + (diff < 0)) >> 1)
        value |= max(0, min(255, k)) << s
    return value


def _inverse_predict(pixels, width, height, shift, controls):
    cols = _tiles(width, shift)
    for y in range(height):
        for x in range(width):
            i = y * width + x
            if y == 0:
                pred = pixels[i - 1] if x else 0xff000000
            elif x == 0:
                pred = pixels[i - width]
            else:
                mode = (controls[(y >> shift) * cols + (x >> shift)] >> 8) & 255
                pred = _predict(mode, pixels[i - 1], pixels[i - width],
                                pixels[i - width - 1], pixels[i - width + 1])
            pixels[i] = _add(pixels[i], pred)


def _inverse_color(pixels, width, height, shift, controls):
    cols = _tiles(width, shift)
    for y in range(height):
        tile_row = (y >> shift) * cols
        for x in range(width):
            coeff = controls[tile_row + (x >> shift)]
            i = y * width + x
            p = pixels[i]
            g = _SIGNED[(p >> 8) & 255]
            red = (((p >> 16) & 255) + ((_SIGNED[coeff & 255] * g) >> 5)) & 255
            blue = (p + ((_SIGNED[(coeff >> 8) & 255] * g) >> 5)
                    + ((_SIGNED[(coeff >> 16) & 255] * _SIGNED[red]) >> 5)) & 255
            pixels[i] = (p & 0xff00ff00) | (red << 16) | blue


def _unpack_palette(pixels, width, height, shift, table):
    # The first packed index in the green channel uses the least significant bits.
    step = 8 >> shift
    mask = (1 << step) - 1
    colors = list(table) + [0] * ((1 << step) - len(table))
    out = array("I")
    row = _tiles(width, shift)
    for y in range(height):
        for x in range(width):
            p = pixels[y * row + (x >> shift)] >> 8
            out.append(colors[(p >> ((x % (1 << shift)) * step)) & mask])
    return out


def _decode_pixels(payload, width, height):
    _dimensions(width, height)
    bits = _Bits(payload)
    transforms = []
    seen = set()
    while bits.read(1):
        kind = bits.read(2)
        if kind in seen:
            raise NanoodleError("Duplicate WebP transform")
        seen.add(kind)
        shift, controls = 0, None
        if kind in (0, 1):
            shift = 2 + bits.read(3)
            controls = _image(bits, _tiles(width, shift), _tiles(height, shift))
        elif kind == 3:
            count = 1 + bits.read(8)
            controls = _image(bits, count, 1)
            for i in range(1, count):
                controls[i] = _add(controls[i], controls[i - 1])
            shift = 3 if count <= 2 else 2 if count <= 4 else 1 if count <= 16 else 0
        transforms.append((kind, width, shift, controls))
        if kind == 3:
            width = _tiles(width, shift)
    pixels = _image(bits, width, height, main=True)
    for kind, width, shift, controls in reversed(transforms):
        if kind == 0:
            _inverse_predict(pixels, width, height, shift, controls)
        elif kind == 1:
            _inverse_color(pixels, width, height, shift, controls)
        elif kind == 2:
            for i, p in enumerate(pixels):
                g = (p >> 8) & 255
                pixels[i] = _add(p, (g << 16) | g)
        else:
            pixels = _unpack_palette(pixels, width, height, shift, controls)
    return pixels


def decode_vp8l(payload):
    """Decode a VP8L chunk payload, including its five-byte header."""
    if len(payload) < 5 or payload[0] != 0x2f:
        raise NanoodleError("Invalid WebP lossless header")
    header = int.from_bytes(payload[1:5], "little")
    if header >> 29:
        raise NanoodleError("Unsupported WebP lossless version")
    w, h = (header & 0x3fff) + 1, ((header >> 14) & 0x3fff) + 1
    pixels = _decode_pixels(payload[5:], w, h)
    if sys.byteorder == "big":
        pixels.byteswap()
    bgra = pixels.tobytes()
    rgba = bytearray(bgra)
    rgba[0::4], rgba[2::4] = bgra[2::4], bgra[0::4]
    return w, h, rgba


def decode_alpha(payload, width, height):
    """Decode the green plane of the lossless stream inside an ALPH chunk."""
    pixels = _decode_pixels(payload, width, height)
    if sys.byteorder == "big":
        pixels.byteswap()
    return pixels.tobytes()[1::4]
