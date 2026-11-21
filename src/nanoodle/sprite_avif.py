"""Apply the clean aperture and orientation used by browser AVIF sprite uploads."""

import struct


def _boxes(data):
    pos = 0
    while pos + 8 <= len(data):
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if pos + 16 > len(data):
                return
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = len(data) - pos
        if size < header or size > len(data) - pos:
            return
        yield kind, data[pos + header:pos + size]
        pos += size


def _child(data, kind):
    return next((body for tag, body in _boxes(data) if tag == kind), memoryview(b""))


def _primary_properties(ipma, primary):
    if len(ipma) < 8 or ipma[0] > 1:
        return
    id_bytes = 4 if ipma[0] else 2
    index_bytes = 2 if ipma[3] & 1 else 1
    index_mask = (1 << (8 * index_bytes - 1)) - 1
    offset = 8
    for _ in range(int.from_bytes(ipma[4:8], "big")):
        if offset + id_bytes + 1 > len(ipma):
            return
        item = int.from_bytes(ipma[offset:offset + id_bytes], "big")
        count = ipma[offset + id_bytes]
        start = offset + id_bytes + 1
        offset = start + count * index_bytes
        if offset > len(ipma):
            return
        if item == primary:
            for pos in range(start, offset, index_bytes):
                yield int.from_bytes(ipma[pos:pos + index_bytes], "big") & index_mask


def _top_left_aperture(raw, size):
    meta = _child(memoryview(raw), b"meta")[4:]
    pitm = _child(meta, b"pitm")
    if len(pitm) < 6 or pitm[0] > 1:
        return None
    id_end = 8 if pitm[0] else 6
    if len(pitm) < id_end:
        return None
    primary = int.from_bytes(pitm[4:id_end], "big")
    iprp = _child(meta, b"iprp")
    apertures = {index: body for index, (tag, body) in enumerate(_boxes(_child(iprp, b"ipco")), 1)
                 if tag == b"clap" and len(body) == 32}
    if not apertures:
        return None
    for tag, ipma in _boxes(iprp):
        if tag != b"ipma":
            continue
        for index in _primary_properties(ipma, primary):
            if index not in apertures:
                continue
            wn, wd, hn, hd, xn, xd, yn, yd = struct.unpack(">8i", apertures[index])
            if min(wn, wd, hn, hd, xd, yd) <= 0 or wn % wd or hn % hd:
                return None
            w, h = wn // wd, hn // hd
            # Chromium crops only integer apertures whose top left is (0, 0).
            # The stored offsets are relative to the center of the coded image.
            if (w <= size[0] and h <= size[1] and 2 * xn == (w - size[0]) * xd
                    and 2 * yn == (h - size[1]) * yd):
                return w, h
            return None
    return None


def present_avif(image, raw):
    from PIL import ImageOps

    aperture = _top_left_aperture(raw, image.size)
    if aperture and aperture != image.size:
        image = image.crop((0, 0, *aperture))
    # Pillow represents the AVIF irot/imir properties as EXIF orientation.
    if image.getexif().get(274, 1) != 1:
        ImageOps.exif_transpose(image, in_place=True)
    return image
