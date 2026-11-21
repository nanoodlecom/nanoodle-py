"""Encode local sprite turntables as H.264 Baseline samples in an MP4.

Sprite frames tend to change within a small silhouette. A P slice copies
unchanged 16×16 blocks from its reference picture. Uniform neighbours share
a directional prediction. Other blocks carry their YUV420 samples verbatim. The single track has no reordering or audio.
"""

import io
import math
import re
import struct
from functools import lru_cache
from itertools import groupby

from .errors import NanoodleError


class _Bits:
    """RBSP bit writer. PCM macroblocks begin on byte boundaries."""

    def __init__(self):
        self.data = bytearray()
        self.pending = 0
        self.length = 0

    def put(self, value, width=1):
        self.pending = (self.pending << width) | value
        self.length += width
        while self.length >= 8:
            self.length -= 8
            self.data.append(self.pending >> self.length)
            self.pending &= (1 << self.length) - 1

    def ue(self, value):
        n = value + 1
        self.put(n, n.bit_length() * 2 - 1)

    def align(self):
        self.put(0, (-self.length) % 8)

    def nal(self, header):
        self.put(1)
        self.align()
        # Escape start-code prefixes inside the RBSP, including PCM bytes.
        escaped = re.sub(b"\x00\x00(?=[\x00-\x03])", b"\x00\x00\x03", self.data)
        return bytes([header]) + escaped


def _configuration(width, height, frame_count, duration, payload_bytes):
    across, down = (width + 15) // 16, (height + 15) // 16
    # Annex A's macroblocks per picture/second and baseline bitrate (kb/s).
    levels = ((30, 1620, 40500, 10000), (31, 3600, 108000, 14000),
              (32, 5120, 216000, 20000), (40, 8192, 245760, 20000),
              (41, 8192, 245760, 50000), (42, 8704, 522240, 50000),
              (50, 22080, 589824, 135000), (51, 36864, 983040, 240000),
              (52, 36864, 2073600, 240000), (60, 139264, 4177920, 240000),
              (61, 139264, 8355840, 480000), (62, 139264, 16711680, 800000))
    area = across * down
    for level, max_area, max_rate, max_kbps in levels:
        if (area <= max_area and max(across, down) ** 2 <= 8 * max_area
                and area * frame_count <= max_rate * duration
                and payload_bytes * 8 <= max_kbps * 1000 * duration):
            break
    else:
        raise NanoodleError("Sprite video exceeds the H.264 picture limits")
    sps = _Bits()
    sps.data.extend((66, 0xC0, level))
    for value in (0, 0, 2, 1):  # SPS id, 4-bit frame_num, POC type, one reference
        sps.ue(value)
    sps.put(0)
    sps.ue(across - 1)
    sps.ue(down - 1)
    sps.put(1)  # frame_mbs_only_flag
    sps.put(1)  # direct_8x8_inference_flag
    crop = width != across * 16 or height != down * 16
    sps.put(int(crop))
    if crop:
        for value in (0, (across * 16 - width) // 2, 0, (down * 16 - height) // 2):
            sps.ue(value)
    # Signal square pixels and the matrix/range used by the RGBA conversion.
    sps.put(1)  # vui_parameters_present_flag
    sps.put(1)
    sps.put(1, 8)  # aspect_ratio_idc: square pixels
    sps.put(0)  # no overscan
    sps.put(1)
    sps.put(5, 3)  # unspecified source video format
    sps.put(0)  # limited range
    sps.put(1)
    sps.put(1, 8)   # BT.709 primaries
    sps.put(13, 8)  # sRGB transfer
    sps.put(6, 8)   # SMPTE 170M / BT.601 matrix
    sps.put(1)
    sps.ue(1)  # centered 4:2:0 chroma
    sps.ue(1)
    sps.put(0, 5)  # timing comes from the MP4 track; no HRD or bitstream restrictions

    pps = _Bits()
    pps.ue(0)
    pps.ue(0)
    pps.put(0, 2)  # CAVLC and no bottom-field POC
    for _ in range(3):  # single slice group and default reference indices
        pps.ue(0)
    pps.put(0, 3)  # weighting disabled
    for _ in range(3):  # initial QP, QS and chroma offsets (signed zero)
        pps.ue(0)
    pps.put(4, 3)  # deblocking control present, constrained intra off, no redundant pictures
    sps, pps = sps.nal(0x67), pps.nal(0x68)
    return (bytes((1, 66, 0xC0, level, 0xFF, 0xE1)) + struct.pack(">H", len(sps))
            + sps + b"\x01" + struct.pack(">H", len(pps)) + pps)


def _block_encoder():
    # Fresh bounded caches for each export. Repeated backgrounds and the
    # small meshes' palettes make both conversions frequent cache hits.
    @lru_cache(maxsize=2048)
    def components(pixel):
        r, g, b, a = pixel
        if a != 255:
            r, g, b = (r * a + 127) // 255, (g * a + 127) // 255, (b * a + 127) // 255
        return (16 + ((66*r + 129*g + 25*b + 128) >> 8),
                -38*r - 74*g + 112*b, 112*r - 94*g - 18*b)

    @lru_cache(maxsize=512)
    def yuv(rgba):
        pixels = [components(p) for p in struct.iter_unpack("4B", rgba)]
        samples = bytearray(384)
        samples[:256] = bytes(p[0] for p in pixels)
        c = 0
        for y in range(0, 256, 32):
            for x in range(y, y + 16, 2):
                a, b, d, e = pixels[x], pixels[x + 1], pixels[x + 16], pixels[x + 17]
                samples[256 + c] = 128 + ((a[1] + b[1] + d[1] + e[1] + 512) >> 10)
                samples[320 + c] = 128 + ((a[2] + b[2] + d[2] + e[2] + 512) >> 10)
                c += 1
        flat = None
        if (samples.count(samples[0], 0, 256) == 256
                and samples.count(samples[256], 256, 320) == 64
                and samples.count(samples[320], 320) == 64):
            flat = (samples[0], samples[256], samples[320])
        return bytes(samples), flat

    return yuv


def _sample(frame, width, height, index, previous, to_yuv):
    if len(frame) != width * height * 4:
        raise NanoodleError("Invalid sprite video RGBA frame")
    across = (width + 15) // 16
    stride = width * 4
    padded = across * 16 - width
    bits = _Bits()
    key = index % 60 == 0
    bits.ue(0)
    bits.ue(2 if key else 0)
    bits.ue(0)
    bits.put((index % 60) % 16, 4)
    if key:
        bits.ue((index // 60) % 2)
        bits.put(0, 2)
    else:
        bits.put(0, 3)
    bits.ue(0)
    bits.ue(1)  # disable loop filtering, preserving reference samples

    colors = [None] * across
    pcm = bytearray(across)
    skipped = 0
    offset = 0
    for y in range(0, height, 16):
        rows = []
        for line in range(y, min(y + 16, height)):
            row = frame[line * stride:(line + 1) * stride]
            rows.append(row + row[-4:] * padded if padded else row)
        rows.extend([rows[-1]] * (16 - len(rows)))
        for col in range(across):
            x = col * 64
            block, color = to_yuv(b"".join(row[x:x + 64] for row in rows))
            match_left = col > 0 and color is not None and colors[col - 1] == color
            match_top = y > 0 and color is not None and colors[col] == color
            colors[col] = color
            if not key and block == previous[offset]:
                pcm[col] = 0
                skipped += 1
                offset += 1
                continue
            if not key:
                bits.ue(skipped)
                skipped = 0
            if match_left or match_top:
                bits.ue((2 if match_left else 1) + (0 if key else 5))
                bits.ue(1 if match_left else 2)  # horizontal / vertical intra chroma prediction
                bits.ue(0)  # mb_qp_delta
                # A PCM neighbour has 16 coefficients, selecting table 3 of
                # coeff_token even with a second, empty neighbour. Otherwise
                # the zero luma-DC coefficient token is a single set bit.
                if (col and pcm[col - 1]) or (y and pcm[col]):
                    bits.put(3, 6)
                else:
                    bits.put(1)
                pcm[col] = 0
            else:
                bits.ue(25 if key else 30)
                bits.align()
                bits.data.extend(block)
                pcm[col] = 1
            previous[offset] = block
            offset += 1
    if skipped:
        bits.ue(skipped)
    return bits.nal(0x65 if key else 0x41)


def _box(kind, *parts):
    length = 8 + sum(len(p) for p in parts)
    if length > 0xFFFFFFFF:
        raise NanoodleError("Sprite video exceeds the MP4 32-bit box limit")
    return b"".join((struct.pack(">I4s", length, kind), *parts))


def _fullbox(kind, flags, *parts):
    return _box(kind, struct.pack(">I", flags), *parts)


def _u32(*numbers):
    return struct.pack(f">{len(numbers)}I", *numbers)


def _movie(width, height, duration, sizes, avcc, chunk_offset):
    timescale = min(90000, max(1, math.floor(0xFFFFFFFF / duration)))
    ticks = math.floor(duration * timescale + 0.5)
    count = len(sizes)
    edges = [(i * ticks + count // 2) // count for i in range(count + 1)]
    runs = [(len(list(items)), delta) for delta, items in
            groupby(edges[i + 1] - edges[i] for i in range(count))]
    matrix = _u32(65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30)
    mvhd = _fullbox(b"mvhd", 0, _u32(0, 0, timescale, ticks, 65536),
                    b"\x01\x00" + bytes(10), matrix, bytes(24), _u32(2))
    tkhd = _fullbox(b"tkhd", 7, _u32(0, 0, 1, 0, ticks), bytes(16), matrix,
                    _u32(width << 16, height << 16))
    mdhd = _fullbox(b"mdhd", 0, _u32(0, 0, timescale, ticks), b"\x55\xc4\0\0")
    hdlr = _fullbox(b"hdlr", 0, bytes(4), b"vide", bytes(12), b"VideoHandler\0")
    entry = _box(b"avc1", bytes(6), b"\0\x01", bytes(16), struct.pack(">HH", width, height),
                 _u32(72 << 16, 72 << 16, 0), b"\0\x01", bytes(32), b"\0\x18\xff\xff",
                 _box(b"avcC", avcc))
    stsd = _fullbox(b"stsd", 0, _u32(1), entry)
    stts = _fullbox(b"stts", 0, _u32(len(runs)), b"".join(_u32(n, dt) for n, dt in runs))
    stsc = _fullbox(b"stsc", 0, _u32(1, 1, count, 1))  # all samples in a single chunk
    stco = _fullbox(b"stco", 0, _u32(1, chunk_offset))
    stsz = _fullbox(b"stsz", 0, _u32(0, count), _u32(*sizes))
    keys = range(1, count + 1, 60)
    stss = _fullbox(b"stss", 0, _u32(len(keys)), _u32(*keys))
    stbl = _box(b"stbl", stsd, stts, stsc, stsz, stco, stss)
    dinf = _box(b"dinf", _fullbox(b"dref", 0, _u32(1), _fullbox(b"url ", 1)))
    minf = _box(b"minf", _fullbox(b"vmhd", 1, bytes(8)), dinf, stbl)
    return _box(b"moov", mvhd, _box(b"trak", tkhd, _box(b"mdia", mdhd, hdlr, minf)))


def encode_mp4(width, height, frames, duration):
    """Consume packed RGBA frames once and return a seekable silent MP4."""
    if width < 2 or height < 2 or width % 2 or height % 2 or max(width, height) > 65535:
        raise NanoodleError("Sprite video needs even frame dimensions")
    if not math.isfinite(duration) or duration <= 0:
        raise NanoodleError("Invalid sprite video duration")
    previous = [None] * (((width + 15) // 16) * ((height + 15) // 16))
    encode_block = _block_encoder()
    data = io.BytesIO()
    sizes = []
    for index, frame in enumerate(frames):
        sample = _sample(frame, width, height, index, previous, encode_block)
        data.write(_u32(len(sample)))
        data.write(sample)
        sizes.append(4 + len(sample))
        if data.tell() + 8 > 0xFFFFFFFF:
            raise NanoodleError("Sprite turntable exceeds the MP4 32-bit box limit")
    if not sizes:
        raise NanoodleError("Sprite turntable has no frames")
    avcc = _configuration(width, height, len(sizes), duration, data.tell())
    ftyp = _box(b"ftyp", b"isom", _u32(512), b"isomiso2avc1mp41")
    # Write the sample table first so players can start without seeking.
    movie = _movie(width, height, duration, sizes, avcc, 0)
    movie = _movie(width, height, duration, sizes, avcc, len(ftyp) + len(movie) + 8)
    return b"".join((ftyp, movie, struct.pack(">I4s", data.tell() + 8, b"mdat"), data.getbuffer()))
