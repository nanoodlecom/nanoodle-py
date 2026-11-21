"""Local sprite slicing, orthographic projection and turntable rendering."""
import json
import math

from .errors import NanoodleError
from .local_media import _url_bytes
from .media import make_data_url


def _number(fields, key, default, low, high, *, integer=False):
    raw = fields.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (ValueError, TypeError, OverflowError):
        value = math.nan
    if not low <= value <= high or (integer and not value.is_integer()):
        kind = "an integer " if integer else ""
        raise NanoodleError(f"{key} must be {kind}between {low} and {high}.")
    return int(value) if integer else value


class _Projection:
    """Precompute sheet occupancy and view dimensions once for all angles."""

    def __init__(self, sheet, fields):
        width, tile, data = sheet
        if not 16 <= tile <= 128 or width % tile or not 8 <= width // tile <= 128:
            raise NanoodleError("Expected a horizontal Slice 3D sheet with 8–128 square tiles of 16–128 pixels.")
        if len(data) != width * tile * 4:
            raise NanoodleError("Invalid sprite sheet dimensions.")
        self.tile = tile
        self.sheet_width = width
        self.data = data
        self.pixels = memoryview(data).cast("I")
        self.layers = width // tile
        pitch = _number(fields, "pitch", 30, 0, 90) * math.pi / 180
        self.scale = _number(fields, "scale", 3, 1, 8)
        spacing = _number(fields, "spacing", 1, 0, 4) * (tile - 2) / self.layers
        self.rise = spacing * math.cos(pitch)
        self.squash = max(1 / tile, math.sin(pitch))
        diagonal = tile * math.sqrt(2)
        self.width = 2 * math.ceil((diagonal * self.scale + 4) / 2)
        self.height = 2 * math.ceil(((diagonal * self.squash + self.rise * (self.layers - 1)) * self.scale + 4) / 2)
        # Exclude transparent layers and margins before projecting the pixels.
        alpha = data[3::4]
        self.bounds = []
        for layer in range(self.layers):
            left, top, right, bottom = tile, tile, 0, 0
            for y in range(tile):
                start = y * width + layer * tile
                row = alpha[start:start + tile]
                end = len(row.rstrip(b"\0"))
                if end:
                    left = min(left, tile - len(row.lstrip(b"\0")))
                    right = max(right, end)
                    top, bottom = min(top, y), y + 1
            if right:
                self.bounds.append((layer, left, top, right, bottom))

    def frame(self, yaw, check_cancel):
        width, height, scale = self.width, self.height, self.scale
        data = bytearray(width * height * 4)
        dest = memoryview(data).cast("I")
        angle = yaw * math.pi / 180
        c, s = math.cos(angle), math.sin(angle)
        tile, half = self.tile, self.tile / 2
        dx = [(x + 0.5 - width / 2) / scale for x in range(width)]
        cdx = [c * x for x in dx]
        sdx = [-s * x for x in dx]
        floor = math.floor
        source = self.data
        for layer, x0, y0, x1, y1 in self.bounds:
            check_cancel()
            cy = height / 2 + ((self.layers - 1) / 2 - layer) * self.rise * scale
            corners = [(u - half, v - half) for u in (x0, x1) for v in (y0, y1)]
            px = [(c * u - s * v) * scale + width / 2 for u, v in corners]
            py = [(s * u + c * v) * scale * self.squash + cy for u, v in corners]
            left, right = max(0, floor(min(px))), min(width, math.ceil(max(px)))
            top, bottom = max(0, floor(min(py))), min(height, math.ceil(max(py)))
            for y in range(top, bottom):
                if (y - top) % 32 == 0:
                    check_cancel()
                dy = (y + 0.5 - cy) / (scale * self.squash)
                sy, cy_rot = s * dy, c * dy
                row = y * width
                for x in range(left, right):
                    # Preserve JS's operation order at integer texel boundaries.
                    u = floor(cdx[x] + sy + half)
                    v = floor(sdx[x] + cy_rot + half)
                    if not (x0 <= u < x1 and y0 <= v < y1):
                        continue
                    src = v * self.sheet_width + layer * tile + u
                    src_byte = src * 4
                    alpha = source[src_byte + 3]
                    if not alpha:
                        continue
                    dst = row + x
                    dst_byte = dst * 4
                    if alpha == 255 or not data[dst_byte + 3]:
                        dest[dst] = self.pixels[src]
                    else:
                        # Straight-alpha source-over with ties-to-even byte clamping.
                        under = data[dst_byte + 3] * (1 - alpha / 255)
                        total = alpha + under
                        for k in range(3):
                            data[dst_byte + k] = round((source[src_byte + k] * alpha + data[dst_byte + k] * under) / total)
                        data[dst_byte + 3] = round(total)
        return data


def run_sprite(engine, node, inp, on_cost):
    from .sprite_codecs import decode_image, encode_png

    check_cancel = engine.check_cancel
    check_cancel()
    source_port = "model" if node.type == "slice-3d" else "sheet"
    if not inp.get(source_port):
        raise NanoodleError(f"Wire {source_port} into {node.type}.")
    fetch = engine.local_fetcher()
    raw = _url_bytes(inp[source_port], fetch)
    fields = node.fields
    out = {}

    def media(data, mime):
        return engine._media_ref(make_data_url(data, mime), mime)

    if node.type == "slice-3d":
        from .sprite_geometry import slice_model

        palette = None
        if _number(fields, "palette", 16, 0, 64, integer=True) and inp.get("paletteImage"):
            palette = decode_image(_url_bytes(inp["paletteImage"], fetch))
        width, height, pixels, layout = slice_model(raw, fields, palette, decode_image, check_cancel=check_cancel)
        sheet = (width, height, pixels)
        out["sheet"] = media(encode_png(*sheet), "image/png")
        out["layout"] = json.dumps(layout, separators=(",", ":"), allow_nan=False)
        view = _Projection(sheet, {})
        yaw = 35
    else:
        view = _Projection(decode_image(raw), fields)
        yaw = _number(fields, "yaw", 35, -3600, 3600)
    frame = view.frame(yaw, check_cancel)
    out["image"] = media(encode_png(view.width, view.height, frame), "image/png")

    if node.type == "sprite-stack-render" and (fields.get("turntable") is True or fields.get("turntable") == "true"):
        from .sprite_video import encode_mp4

        fps = _number(fields, "fps", 24, 1, 60, integer=True)
        duration = _number(fields, "duration", 3, 0.25, 30)
        count = max(2, math.floor(fps * duration + 0.5))
        start = _number(fields, "yawStart", 0, -3600, 3600)
        end = _number(fields, "yawEnd", 360, -3600, 3600)

        def frames():
            for i in range(count):
                check_cancel()
                angle = start + (end - start) * i / (count - 1)
                yield frame if angle == yaw else view.frame(angle, check_cancel)

        out["video"] = media(encode_mp4(view.width, view.height, frames(), duration), "video/mp4")
    check_cancel()
    return out
