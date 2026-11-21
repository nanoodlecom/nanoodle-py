"""The Slice 3D APNG carries one full, independently decodable layer atlas per pose."""

import struct
import zlib
from dataclasses import dataclass
from math import floor, isfinite

from .errors import NanoodleError
from .sprite_codecs import (
    _PNG,
    _chunk,
    _compressed_rgba_rows,
    _decode_png,
    _rgba_header,
)


def encode_stack_animation(width, height, atlases, count, duration):
    if not 2 <= count <= 64:
        raise NanoodleError("An animated sprite sheet needs 2–64 poses.")
    delay = duration / count
    if not isfinite(delay) or not 0 < delay <= 65535:
        raise NanoodleError(
            "Slice 3D cannot represent this animation frame duration in APNG."
        )
    denominator = max(1, min(60000, floor(65535 / delay)))
    numerator = max(1, min(65535, floor(delay * denominator + 0.5)))
    out = bytearray(_rgba_header(width, height))
    out.extend(_chunk(b"nsTA", b"\x01"))
    out.extend(_chunk(b"acTL", struct.pack(">II", count, 0)))
    sequence = 0
    written = 0
    for rgba in atlases:
        if written == count:
            raise NanoodleError("Unexpected extra pose in sprite animation.")
        control = struct.pack(
            ">5I2H2B", sequence, width, height, 0, 0, numerator, denominator, 0, 0
        )
        out.extend(_chunk(b"fcTL", control))
        sequence += 1
        compressed = _compressed_rgba_rows(width, height, rgba)
        if not written:
            out.extend(_chunk(b"IDAT", compressed))
        else:
            out.extend(_chunk(b"fdAT", struct.pack(">I", sequence) + compressed))
            sequence += 1
        written += 1
    if written != count:
        raise NanoodleError("Missing pose in sprite animation.")
    out.extend(_chunk(b"IEND", b""))
    return bytes(out)


@dataclass
class StackAnimation:
    width: int
    height: int
    delays: list
    data: list

    def sheet(self, index):
        # Each fdAT/IDAT contains the same deflate stream as a regular PNG IDAT.
        # Decode only the requested pose so high-resolution clips fit in memory.
        chunks = [_rgba_header(self.width, self.height)]
        chunks.extend(_chunk(b"IDAT", segment) for segment in self.data[index])
        chunks.append(_chunk(b"IEND", b""))
        return _decode_png(b"".join(chunks))


def _marked(data):
    """Recognize the private chunk, including an invalid marker in a malformed sheet."""
    if not data.startswith(_PNG):
        return False
    offset = 8
    for _ in range(65536):
        if offset + 8 > len(data):
            break
        length, kind = struct.unpack_from(">I4s", data, offset)
        if kind == b"nsTA":
            return True
        if kind == b"IEND":
            break
        offset += 12 + length
    return False


def read_stack_animation(data):
    """Index compressed chunks for marked APNGs and leave other uploads to their image codec."""
    if not _marked(data):
        return None
    if len(data) > 544 * 1024 * 1024:
        raise NanoodleError("Sprite animation exceeds the size limit.")
    width = height = count = sequence = 0
    chunks = frame_bytes = 0
    frames, delays = [], []
    offset = 8
    seen_marker = False
    previous = None
    while offset + 12 <= len(data):
        length, kind = struct.unpack_from(">I4s", data, offset)
        chunks += 1
        if (
            not kind.isalpha()
            or not kind[2:3].isupper()
            or (chunks == 1 and kind != b"IHDR")
        ):
            raise NanoodleError("Invalid sprite animation chunk type or order.")
        if chunks > 65536:
            raise NanoodleError("Sprite animation has too many chunks.")
        end = offset + 8 + length
        if end + 4 > len(data):
            raise NanoodleError("Truncated sprite animation chunk.")
        body = memoryview(data)[offset + 8 : end]
        if zlib.crc32(body, zlib.crc32(kind)) != struct.unpack_from(">I", data, end)[0]:
            raise NanoodleError("Invalid sprite animation checksum.")
        if kind == b"IHDR":
            if offset != 8 or length != 13 or body[8:] != b"\x08\x06\x00\x00\x00":
                raise NanoodleError("Expected full RGBA8 sprite animation sheets.")
            width, height = struct.unpack_from(">2I", body)
            if not (
                16 <= height <= 128
                and width % height == 0
                and 8 <= width // height <= 128
            ):
                raise NanoodleError("Invalid sprite animation sheet dimensions.")
        elif kind == b"nsTA":
            if not width or seen_marker or count or body != b"\x01":
                raise NanoodleError("Invalid sprite animation marker.")
            seen_marker = True
        elif kind == b"acTL":
            if not seen_marker or count or length != 8:
                raise NanoodleError("Invalid sprite animation header.")
            count, plays = struct.unpack(">II", body)
            if not 2 <= count <= 64 or plays:
                raise NanoodleError("Invalid sprite animation count or loop setting.")
        elif kind == b"fcTL":
            if (
                not count
                or len(frames) >= count
                or length != 26
                or (frames and not frame_bytes)
            ):
                raise NanoodleError("Invalid sprite animation frame control.")
            seq, fw, fh, x, y, num, den, dispose, blend = struct.unpack(">5I2H2B", body)
            if (
                seq != sequence
                or (fw, fh) != (width, height)
                or x
                or y
                or dispose
                or blend
                or not num
                or not den
            ):
                raise NanoodleError(
                    "Invalid sprite animation sequence, rectangle or timing."
                )
            if delays and delays[0] != num / den:
                raise NanoodleError("Sprite animation frame delays must be uniform.")
            delays.append(num / den)
            frame_bytes = 0
            frames.append([])
            sequence += 1
        elif kind == b"IDAT":
            if len(frames) != 1 or (frames[0] and previous != b"IDAT"):
                raise NanoodleError("Invalid sprite animation default frame.")
            frame_bytes += length
            frames[0].append(body)
        elif kind == b"fdAT":
            if (
                len(frames) < 2
                or length < 4
                or sequence != struct.unpack_from(">I", body)[0]
            ):
                raise NanoodleError("Invalid sprite animation frame data.")
            frame_bytes += length - 4
            frames[-1].append(body[4:])
            sequence += 1
        elif kind == b"IEND":
            if (
                length
                or end + 4 != len(data)
                or len(frames) != count
                or not frame_bytes
            ):
                raise NanoodleError("Invalid sprite animation end.")
            return StackAnimation(width, height, delays, frames)
        elif kind != b"PLTE" and not kind[0] & 32:
            raise NanoodleError("Unsupported critical sprite animation chunk.")
        if frame_bytes > (width * 4 + 1) * height + 65536:
            raise NanoodleError(
                "Sprite animation frame exceeds the compressed size limit."
            )
        previous = kind
        offset = end + 4
    raise NanoodleError("Missing sprite animation end.")
