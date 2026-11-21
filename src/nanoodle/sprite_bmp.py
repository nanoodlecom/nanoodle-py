"""Read Windows/OS/2 bitmaps and the preferred image in a Windows icon."""

import struct

from .errors import NanoodleError
from .sprite_codecs import _PNG, _decode_png, _dimensions


def _mask_fields(masks, depth):
    fields = []
    used = 0
    for index, mask in enumerate(masks):
        if not mask:
            if index < 3:
                raise NanoodleError("Missing BMP color mask")
            fields.append((0, 0))
            continue
        shift = (mask & -mask).bit_length() - 1
        maximum = mask >> shift
        if mask >= 1 << depth or mask & used or maximum & (maximum + 1):
            raise NanoodleError("Invalid BMP color masks")
        fields.append((shift, maximum))
        used |= mask
    return fields


def _read_rle(data, start, width, height, depth, color_count, palette):
    """Decode Windows RLE4/RLE8 onto its initially black display surface."""
    rgba = bytearray(b"\0\0\0\xff" * (width * height))
    pos, x, y = start, 0, 0
    while pos + 2 <= len(data):
        count, value = data[pos:pos + 2]
        pos += 2
        if count:
            if y >= height or x + count > width:
                raise NanoodleError("BMP encoded run exceeds its row")
            if depth == 8:
                run = bytes([value]) * count
            else:
                run = (bytes((value >> 4, value & 15)) * ((count + 1) // 2))[:count]
        elif value == 0:
            x = 0
            y += 1
            if y > height:
                raise NanoodleError("Too many BMP rows")
            continue
        elif value == 1:
            return rgba, pos
        elif value == 2:
            if pos + 2 > len(data):
                break
            x += data[pos]
            y += data[pos + 1]
            pos += 2
            if x > width or y >= height:
                raise NanoodleError("Invalid BMP delta")
            continue
        else:
            count = value
            nbytes = count if depth == 8 else (count + 1) // 2
            end = pos + nbytes
            if end + nbytes % 2 > len(data) or y >= height or x + count > width:
                raise NanoodleError("Invalid absolute BMP run")
            raw = data[pos:end]
            pos = end + nbytes % 2
            if depth == 8:
                run = raw
            else:
                run = bytes((raw[i // 2] >> (4 if i % 2 == 0 else 0)) & 15 for i in range(count))
        if max(run) >= color_count:
            raise NanoodleError("BMP palette index out of range")
        dest = ((height - 1 - y) * width + x) * 4
        for ch, table in enumerate(palette):
            rgba[dest + ch:dest + ch + count * 4:4] = run.translate(table)
        x += count
    raise NanoodleError("Truncated BMP run-length data")


def _read_dib(data, start, *, pixel_offset=None, icon_size=None):
    if len(data) - start < 4:
        raise NanoodleError("Missing BMP header")
    header = struct.unpack_from("<I", data, start)[0]
    if header not in (12, 16, 40, 52, 56, 64, 108, 124) or start + header > len(data):
        raise NanoodleError("Unsupported or truncated BMP header")
    compression = image_size = colors_used = 0
    if header == 12:
        width, signed_h, planes, depth = struct.unpack_from("<4H", data, start + 4)
    else:
        width, signed_h, planes, depth = struct.unpack_from("<iiHH", data, start + 4)
        if header >= 40:
            compression, image_size = struct.unpack_from("<II", data, start + 16)
            colors_used = struct.unpack_from("<I", data, start + 32)[0]
    height = abs(signed_h)
    if icon_size:
        if height % 2 or (width, height // 2) != icon_size:
            raise NanoodleError("ICO dimensions do not match its DIB")
        height //= 2
    _dimensions(width, height)
    if planes != 1 or depth not in (1, 4, 8, 16, 24, 32):
        raise NanoodleError("Unsupported BMP bit depth or planes")
    pos = start + header
    mask_fields = None
    if compression in (3, 6) and header not in (12, 16, 64):
        if depth not in (16, 32) or (compression == 6 and 40 < header < 56):
            raise NanoodleError("Invalid BMP bitfields")
        mask_start = start + 40
        n = 4 if compression == 6 or header >= 56 else 3
        if mask_start + 4 * n > len(data):
            raise NanoodleError("Truncated BMP bitfields")
        masks = list(struct.unpack_from(f"<{n}I", data, mask_start))
        if n == 3:
            masks.append(0)
        mask_fields = _mask_fields(masks, depth)
        if header == 40:
            pos += 4 * n
    elif compression == 0:
        if depth == 16:
            mask_fields = _mask_fields((0x7c00, 0x03e0, 0x001f, 0), depth)
    elif (compression, depth) not in ((1, 8), (2, 4)) or signed_h < 0:
        raise NanoodleError("Unsupported BMP compression")

    color_count = colors_used or (1 << depth if depth <= 8 else 0)
    entry_size = 3 if header == 12 else 4
    if depth <= 8 and color_count > 1 << depth:
        raise NanoodleError("Oversized BMP palette")
    if color_count * entry_size > len(data) - pos:
        raise NanoodleError("Truncated BMP palette")
    palette = None
    if depth <= 8:
        table = data[pos:pos + color_count * entry_size]
        palette = [bytes(table[i::entry_size]).ljust(256, b"\0") for i in (2, 1, 0)]
    pos += color_count * entry_size
    if pixel_offset is not None:
        if not pos <= pixel_offset <= len(data):
            raise NanoodleError("Invalid BMP pixel offset")
        pos = pixel_offset

    stride = ((width * depth + 31) // 32) * 4
    compressed = compression in (1, 2)
    if compressed:
        rgba, end = _read_rle(data, pos, width, height, depth, color_count, palette)
        if image_size:
            if pos + image_size < end or pos + image_size > len(data):
                raise NanoodleError("Invalid BMP compressed length")
            end = pos + image_size
    else:
        end = pos + height * stride
        if end > len(data):
            raise NanoodleError("Truncated BMP pixels")
        rgba = bytearray(width * height * 4)
    has_icon_alpha = False
    maybe_icon_alpha = icon_size and depth == 32 and mask_fields is None
    for row_num in range(0 if compressed else height):
        y = height - 1 - row_num if signed_h > 0 else row_num
        row_at = pos + row_num * stride
        row = data[row_at:row_at + stride]
        out = memoryview(rgba)[y * width * 4:(y + 1) * width * 4]
        if depth <= 8:
            if depth == 8:
                indices = bytes(row[:width])
            else:
                mask = (1 << depth) - 1
                indices = bytes((row[i * depth // 8] >> (8 - depth - i * depth % 8)) & mask
                                for i in range(width))
            if max(indices) >= color_count:
                raise NanoodleError("BMP palette index out of range")
            for ch, table in enumerate(palette):
                out[ch::4] = indices.translate(table)
            out[3::4] = b"\xff" * width
        elif mask_fields is not None:
            samples = struct.iter_unpack("<H" if depth == 16 else "<I", row[:width * (depth // 8)])
            for x, (pixel,) in enumerate(samples):
                for channel, (shift, maximum) in enumerate(mask_fields):
                    out[4*x + channel] = (
                        (((pixel >> shift) & maximum) * 255 + maximum // 2) // maximum
                        if maximum else 255)
        else:
            channels = depth // 8
            row = row[:width * channels]
            for ch in range(3):
                out[ch::4] = row[2 - ch::channels]
            if maybe_icon_alpha:
                alpha = row[3::4]
                out[3::4] = alpha
                has_icon_alpha = has_icon_alpha or any(alpha)
            else:
                out[3::4] = b"\xff" * width

    if icon_size and not (has_icon_alpha or mask_fields and mask_fields[3][1]):
        and_stride = ((width + 31) // 32) * 4
        mask_left = len(data) - end
        if mask_left < and_stride * height and (mask_left != 0 or depth != 32):
            raise NanoodleError("Truncated ICO transparency mask")
        for row_num in range(height):
            y = height - 1 - row_num if signed_h > 0 else row_num
            row_at = end + row_num * and_stride
            for x in range(width):
                dst = 4 * (y * width + x)
                if mask_left and data[row_at + x // 8] & (0x80 >> (x % 8)):
                    rgba[dst:dst + 4] = b"\0\0\0\0"
                else:
                    rgba[dst + 3] = 255
    return width, height, bytes(rgba)


def decode_bmp(raw):
    if len(raw) < 18 or raw[:2] != b"BM":
        raise NanoodleError("Invalid BMP file header")
    size, _, _, offset = struct.unpack_from("<IHHI", raw, 2)
    if size and not offset <= size <= len(raw):
        raise NanoodleError("Invalid BMP file size")
    return _read_dib(raw[:size or len(raw)], 14, pixel_offset=offset)


def decode_ico(raw):
    if len(raw) < 6 or raw[:4] not in (b"\x00\x00\x01\x00", b"\x00\x00\x02\x00"):
        raise NanoodleError("Invalid ICO header")
    count = struct.unpack_from("<H", raw, 4)[0]
    directory_end = 6 + count * 16
    if not count or directory_end > len(raw):
        raise NanoodleError("Truncated ICO directory")
    best = None
    for i in range(count):
        w, h, _, _, _, depth, size, offset = struct.unpack_from("<4B2H2I", raw, 6 + i * 16)
        width, height = w or 256, h or 256
        if offset < directory_end or size < 4 or offset + size > len(raw):
            raise NanoodleError("Invalid ICO image offset")
        rank = (width * height, depth if raw[2] == 1 else 0)  # CUR stores a hotspot, not a bit depth
        if best is None or rank > best[0]:
            best = rank, offset, size, width, height
    _, offset, size, width, height = best
    image = raw[offset:offset + size]
    if image.startswith(_PNG):
        if len(image) < 24 or struct.unpack_from(">II", image, 16) != (width, height):
            raise NanoodleError("ICO dimensions do not match its PNG")
        return _decode_png(image)
    return _read_dib(image, 0, icon_size=(width, height))
