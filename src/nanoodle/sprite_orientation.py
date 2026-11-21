"""Read the primary TIFF orientation from JPEG/PNG and orient decoded RGBA."""


def _tiff_orientation(tiff):
    if len(tiff) < 8:
        return 1
    if tiff[:4] == b"II\x2a\x00":
        order = "little"
    elif tiff[:4] == b"MM\x00\x2a":
        order = "big"
    else:
        return 1

    offset = int.from_bytes(tiff[4:8], order)
    if not 8 <= offset <= len(tiff) - 2:
        return 1
    count = int.from_bytes(tiff[offset : offset + 2], order)
    start = offset + 2
    if count > (len(tiff) - start) // 12:
        return 1
    # Orientation is a single SHORT stored inline in IFD0, not in the
    # thumbnail's directory or the EXIF camera settings subdirectory.
    for at in range(start, start + count * 12, 12):
        if int.from_bytes(tiff[at : at + 2], order) != 0x112:
            continue
        field_type = int.from_bytes(tiff[at + 2 : at + 4], order)
        elements = int.from_bytes(tiff[at + 4 : at + 8], order)
        value = int.from_bytes(tiff[at + 8 : at + 10], order)
        if field_type == 3 and elements == 1 and 1 <= value <= 8:
            return value
    return 1


def _image_orientation(raw):
    view = memoryview(raw)
    if raw.startswith(b"\xff\xd8"):
        pos = 2
        while pos + 4 <= len(raw) and raw[pos] == 0xFF:
            while pos < len(raw) and raw[pos] == 0xFF:
                pos += 1
            if pos + 3 > len(raw):
                break
            marker = raw[pos]
            pos += 1
            if marker in (0xDA, 0xD9):  # SOS or EOI
                break
            if marker == 0x01:  # standalone TEM
                continue
            length = int.from_bytes(view[pos : pos + 2], "big")
            end = pos + length
            if length < 2 or end > len(raw):
                break
            if marker == 0xE1 and length >= 8 and raw[pos + 2 : pos + 8] == b"Exif\x00\x00":
                return _tiff_orientation(view[pos + 8 : end])
            pos = end
    elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
        pos = 8
        while pos + 12 <= len(raw):
            length = int.from_bytes(view[pos : pos + 4], "big")
            kind = raw[pos + 4 : pos + 8]
            end = pos + 8 + length
            if end + 4 > len(raw) or kind == b"IEND":
                break
            if kind == b"eXIf":
                return _tiff_orientation(view[pos + 8 : end])
            pos = end + 4
    return 1


def orient_image(raw, image):
    """Apply display orientation to the decoded (width, height, RGBA) tuple."""
    orientation = _image_orientation(raw)
    if orientation == 1:
        return image
    width, height, rgba = image
    # Copy entire four-byte pixels. Sliced memoryviews perform the strided
    # copies in C for transposed as well as mirrored scanlines.
    src = memoryview(rgba).cast("I")
    out = bytearray(len(rgba))
    dest = memoryview(out).cast("I")
    transposed = orientation >= 5
    row_width, row_count = (height, width) if transposed else (width, height)
    for y in range(row_count):
        source_row = row_count - 1 - y if orientation in (3, 4, 7, 8) else y
        if transposed:
            row = src[source_row::width]
        else:
            offset = source_row * width
            row = src[offset : offset + width]
        if orientation in (2, 3, 6, 7):
            row = row[::-1]
        dest[y * row_width : (y + 1) * row_width] = row
    return row_width, row_count, bytes(out)
