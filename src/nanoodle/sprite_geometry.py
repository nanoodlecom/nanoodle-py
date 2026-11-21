"""Slice glTF poses or a Gaussian PLY into bottom-to-top RGBA tiles."""
from array import array
from itertools import chain, pairwise
from math import ceil, exp, floor, inf, isfinite
from struct import iter_unpack

from .errors import NanoodleError
from .sprite_model import read_model_poses

_AXES = {"x": (1, 2, 0, -1), "y": (0, 2, 1, 1), "z": (0, 1, 2, -1)}
_PLANES = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
_LINEAR = tuple(
    v / 12.92 if v < 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    for v in (i / 255 for i in range(256))
)


def _integer(fields, key, default, low, high):
    raw = fields.get(key)
    try:
        value = default if raw is None or raw == "" else float(raw)
        if low <= value <= high and int(value) == value:
            return int(value)
    except (TypeError, ValueError, OverflowError):
        pass
    raise NanoodleError(f"{key} must be an integer between {low} and {high}.")


class _Grid:
    def __init__(self, kind, primitives, axis, size, layers):
        try:
            self.u, self.v, self.h, self.sign = _AXES[axis]
        except (KeyError, TypeError):
            raise NanoodleError("Up axis must be auto, x, y or z.") from None
        self.size = size
        self.layers = layers
        self.low, self.high = [inf] * 3, [-inf] * 3
        for primitive in primitives:
            for axis_index in range(3):
                if kind == "mesh":
                    coords = [p[axis_index] for p in primitive.positions]
                    low, high = min(coords), max(coords)
                else:
                    center, extent = primitive.center[axis_index], primitive.extent[axis_index]
                    low, high = center - extent, center + extent
                self.low[axis_index] = min(self.low[axis_index], low)
                self.high[axis_index] = max(self.high[axis_index], high)
        span = max(b - a for a, b in zip(self.low, self.high))
        if not (span > 0 and isfinite(span)):
            raise NanoodleError("Model bounds have no usable extent.")
        self.unit = span / (size - 2)
        self.step = span / layers
        self.cu = (self.high[self.u] + self.low[self.u]) / 2
        self.cv = (self.high[self.v] + self.low[self.v]) / 2

    def project(self, p):
        return (
            (p[self.u] - self.cu) / self.unit + self.size / 2,
            self.sign * (p[self.v] - self.cv) / self.unit + self.size / 2,
            (p[self.h] - self.low[self.h]) / self.step,
        )


def _byte(v):
    # Uint8ClampedArray uses round-to-nearest, ties-to-even.
    return 0 if v <= 0 else 255 if v >= 255 else round(v)


def _srgb(value):
    return 12.92 * value if value < 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055


def _wrap(t, mode):
    if mode == 33071:
        return min(1, max(0, t))
    if mode == 33648:
        return 1 - abs(t % 2 - 1)
    return t - floor(t)


def _color_bytes(mesh, color):
    alpha = color[3]
    if mesh.alpha_mode == "MASK":
        alpha = int(alpha >= mesh.cutoff)
    elif mesh.alpha_mode == "OPAQUE":
        alpha = 1
    if alpha == 0:
        return None
    return bytes((_byte(_srgb(color[0]) * 255), _byte(_srgb(color[1]) * 255),
                  _byte(_srgb(color[2]) * 255), _byte(alpha * 255)))


def _sampler(mesh, ids):
    # Constant material colors need only one linear-to-sRGB conversion.
    if mesh.color is None and mesh.texture is None:
        color = _color_bytes(mesh, mesh.factor)
        return lambda u, v, w: color
    a, b, c = ids
    colors = None if mesh.color is None else (mesh.color[a], mesh.color[b], mesh.color[c])
    if mesh.texture:
        (width, height, pixels), sampler = mesh.texture
        us, vs = zip(mesh.uv[a], mesh.uv[b], mesh.uv[c])
        wrap_s, wrap_t = sampler.get("wrapS"), sampler.get("wrapT")

    def sample(u, v, w):
        color = list(mesh.factor)
        if colors:
            for k in range(len(colors[0])):
                color[k] *= u * colors[0][k] + v * colors[1][k] + w * colors[2][k]
        if mesh.texture:
            x = min(width - 1, floor(_wrap(u * us[0] + v * us[1] + w * us[2], wrap_s) * width))
            y = min(height - 1, floor(_wrap(u * vs[0] + v * vs[1] + w * vs[2], wrap_t) * height))
            p = (y * width + x) * 4
            color[0] *= _LINEAR[pixels[p]]
            color[1] *= _LINEAR[pixels[p + 1]]
            color[2] *= _LINEAR[pixels[p + 2]]
            color[3] *= pixels[p + 3] / 255
        return _color_bytes(mesh, color)
    return sample


def _scan(p, q, r, a, b, width, height):
    dx, dy = q[a] - p[a], q[b] - p[b]
    ex, ey = r[a] - p[a], r[b] - p[b]
    det = dx * ey - dy * ex
    if abs(det) < 1e-12:
        return
    sign = 1 if det > 0 else -1
    start_x = max(0, ceil(min(p[a], q[a], r[a]) - 0.5))
    end_x = min(width - 1, floor(max(p[a], q[a], r[a]) - 0.5))
    start_y = max(0, ceil(min(p[b], q[b], r[b]) - 0.5))
    end_y = min(height - 1, floor(max(p[b], q[b], r[b]) - 0.5))
    for y in range(start_y, end_y + 1):
        py = y + 0.5 - p[b]
        for x in range(start_x, end_x + 1):
            px = x + 0.5 - p[a]
            v, w = (px * ey - py * ex) / det, (dx * py - dy * px) / det
            u = 1 - v - w
            if u >= -1e-8 and v >= -1e-8 and w >= -1e-8:
                yield x, y, u, v, w, sign


def _fill_interiors(grid, rays, rgba, check_cancel):
    size, layers = grid.size, grid.layers
    width = size * layers
    for i, ray in enumerate(rays):
        if i % 256 == 0 and check_cancel:
            check_cancel()
        if not ray or len(ray) < 2:
            continue
        ray.sort(key=lambda hit: hit[0])
        hits = []
        for depth, sign, color in ray:
            if hits and abs(hits[-1][0] - depth) < 1e-6:
                hits[-1][1] += sign
            else:
                hits.append([depth, sign, color])
        for hit in hits:
            hit[1] = (hit[1] > 0) - (hit[1] < 0)
        # Open meshes and coincident faces keep their painted surface.
        if sum(hit[1] for hit in hits):
            continue
        winding = 0
        base = (i // size * width + i % size) * 4
        for (lo, sign, lower), (hi, _, upper) in pairwise(hits):
            winding += sign
            if not winding:
                continue
            middle = (lo + hi) / 2
            for layer in range(max(0, ceil(lo - 0.5)), min(layers, ceil(hi - 0.5))):
                pos = base + layer * size * 4
                color = lower if layer + 0.5 < middle else upper
                if color is not None and rgba[pos + 3] == 0:
                    rgba[pos:pos + 4] = color


def _paint_mesh(grid, meshes, rgba, check_cancel):
    size, layers = grid.size, grid.layers
    dims = (size, size, layers)
    # Strides for (column, row, layer) in the horizontal sprite strip.
    strides = (4, size * layers * 4, size * 4)
    rays = [None] * (size * size)
    for mesh in meshes:
        points = [grid.project(pos) for pos in mesh.positions]
        step = 3 if mesh.mode == 4 else 1
        for i in range(0, mesh.length - 2, step):
            if i % (64 * step) == 0 and check_cancel:
                check_cancel()
            first = 0 if mesh.mode == 6 else i
            corners = (i + 1, i, i + 2) if mesh.mode == 5 and i % 2 else (first, i + 1, i + 2)
            ids = corners if mesh.idx is None else tuple(mesh.idx[j] for j in corners)
            p, q, r = (points[j] for j in ids)
            areas = [abs((q[a] - p[a]) * (r[b] - p[b]) - (q[b] - p[b]) * (r[a] - p[a]))
                     for a, b, _ in _PLANES]
            largest = max(areas)
            if not largest:
                continue
            dominant = areas.index(largest)
            sample = _sampler(mesh, ids)
            painted = False
            # A dominant-axis projection captures vertical walls. Scanning the
            # floor projection as well yields signed intersections for closed solids.
            for plane in ((dominant, 0) if dominant else (0,)):
                a, b, c = _PLANES[plane]
                for x, y, u, v, w, sign in _scan(p, q, r, a, b, dims[a], dims[b]):
                    depth = u * p[c] + v * q[c] + w * r[c]
                    color = sample(u, v, w)
                    if plane == dominant:
                        painted = True
                        if color is not None:
                            cell = min(dims[c] - 1, max(0, floor(depth)))
                            pos = x * strides[a] + y * strides[b] + cell * strides[c]
                            rgba[pos:pos + 4] = color
                    if plane == 0:
                        cell = y * size + x
                        if rays[cell] is None:
                            rays[cell] = []
                        rays[cell].append((depth, sign, color))
            if not painted:
                color = sample(1/3, 1/3, 1/3)
                if color is not None:
                    pos = sum(strides[k] * min(dims[k] - 1, max(0, floor((p[k] + q[k] + r[k]) / 3)))
                              for k in range(3))
                    rgba[pos:pos + 4] = color
    _fill_interiors(grid, rays, rgba, check_cancel)


def _paint_splats(grid, splats, rgba, check_cancel):
    size, layers = grid.size, grid.layers
    width = size * layers
    weights = array("f", [0]) * (width * size)
    axes = grid.u, grid.v, grid.h
    dims = size, size, layers
    column = [(x + 0.5 - size / 2) * grid.unit + grid.cu for x in range(size)]
    row = [grid.sign * (y + 0.5 - size / 2) * grid.unit + grid.cv for y in range(size)]
    elevation = [grid.low[grid.h] + (l + 0.5) * grid.step for l in range(layers)]
    for splat in splats:
        center = grid.project(splat.center)
        radius = [splat.extent[a] / unit for a, unit in zip(axes, (grid.unit, grid.unit, grid.step))]
        lo = [max(0, floor(c - r)) for c, r in zip(center, radius)]
        hi = [min(d - 1, floor(c + r)) for d, c, r in zip(dims, center, radius)]
        cx, cy, cl = [min(d - 1, max(0, floor(c))) for c, d in zip(center, dims)]
        rx, ry, rz = splat.rotation[0:3], splat.rotation[4:7], splat.rotation[8:11]
        sx, sy, sz = splat.scales
        # Each axis samples a regular grid. Precompute its offsets just once
        # per splat instead of rebuilding a world transform for every voxel.
        dx = [column[x] - splat.center[grid.u] for x in range(lo[0], hi[0] + 1)]
        d = [0.0, 0.0, 0.0]
        red, green, blue = splat.color
        for layer in range(lo[2], hi[2] + 1):
            if check_cancel:
                check_cancel()
            d[grid.h] = elevation[layer] - splat.center[grid.h]
            for y in range(lo[1], hi[1] + 1):
                d[grid.v] = row[y] - splat.center[grid.v]
                base = y * width + layer * size
                for x, delta in enumerate(dx, lo[0]):
                    d[grid.u] = delta
                    at_center = layer == cl and y == cy and x == cx
                    if at_center:
                        alpha = splat.alpha
                    else:
                        a, b, c = d
                        xq = (a * rx[0] + b * rx[1] + c * rx[2]) / sx
                        yq = (a * ry[0] + b * ry[1] + c * ry[2]) / sy
                        zq = (a * rz[0] + b * rz[1] + c * rz[2]) / sz
                        distance = xq * xq + yq * yq + zq * zq
                        if distance > 9:
                            continue
                        alpha = splat.alpha * exp(-0.5 * distance)
                    if alpha < 0.5 / 255:
                        continue
                    idx = base + x
                    old = weights[idx]
                    total = old + alpha
                    pos = idx * 4
                    rgba[pos] = _byte((rgba[pos] * old + red * alpha) / total)
                    rgba[pos + 1] = _byte((rgba[pos + 1] * old + green * alpha) / total)
                    rgba[pos + 2] = _byte((rgba[pos + 2] * old + blue * alpha) / total)
                    rgba[pos + 3] = _byte(rgba[pos + 3] + alpha * (255 - rgba[pos + 3]))
                    weights[idx] = total


def _palette(data, limit, check_cancel):
    # Small palettes keep exact colors in encounter order, including near
    # neighbors within one 5-bit bin. Bound histogram memory for large images.
    exact = {}
    for i, (r, g, b, a) in enumerate(iter_unpack("4B", data)):
        if check_cancel and i % 32768 == 0:
            check_cancel()
        if a:
            exact[r, g, b] = None
            if len(exact) > limit:
                break
    else:
        if not exact:
            raise NanoodleError("Palette source has no visible pixels.")
        return list(exact)
    histogram = {}
    for i, (r, g, b, a) in enumerate(iter_unpack("4B", data)):
        if check_cancel and i % 32768 == 0:
            check_cancel()
        if not a:
            continue
        key = (r >> 3) * 1024 + (g >> 3) * 32 + (b >> 3)
        sums = histogram.get(key)
        if sums is None:
            histogram[key] = [r, g, b, 1]
        else:
            sums[0] += r
            sums[1] += g
            sums[2] += b
            sums[3] += 1
    colors = [(r / n, g / n, b / n, n) for r, g, b, n in histogram.values()]

    def box(colors):
        spans = [max(c[k] for c in colors) - min(c[k] for c in colors) for k in range(3)]
        count = sum(c[3] for c in colors)
        span = max(spans)
        return span * count, spans.index(span), count, colors

    boxes = [box(colors)]
    while len(boxes) < limit:
        boxes.sort(key=lambda b: b[0], reverse=True)
        score, axis, count, colors = boxes[0]
        if score == 0:
            break
        boxes.pop(0)
        colors.sort(key=lambda c: c[axis])
        cut, weight = 1, colors[0][3]
        while cut < len(colors) - 1 and weight < count / 2:
            weight += colors[cut][3]
            cut += 1
        boxes.extend((box(colors[:cut]), box(colors[cut:])))
    return [tuple(floor(sum(c[k] * c[3] for c in colors) / count + 0.5) for k in range(3))
            for _, _, count, colors in boxes]


def _quantize(rgba, palette_image, limit, outline, check_cancel):
    reserve_ink = outline and palette_image is None and limit > 1
    colors = _palette(rgba if palette_image is None else palette_image[2], limit - reserve_ink, check_cancel)
    cache = {c: bytes(c) for c in colors}
    for index, (r, g, b, a) in enumerate(iter_unpack("4B", rgba)):
        if check_cancel and index % 32768 == 0:
            check_cancel()
        if not a:
            continue
        key = r, g, b
        nearest = cache.get(key)
        if nearest is None:
            nearest = bytes(min(colors, key=lambda c: (r-c[0])**2 + (g-c[1])**2 + (b-c[2])**2))
            if len(cache) < 65536:
                cache[key] = nearest
        pos = index * 4
        rgba[pos:pos + 3] = nearest
    return bytes((24, 24, 24) if reserve_ink else min(colors, key=sum))


def _outline(rgba, size, width, ink, check_cancel):
    from PIL import Image, ImageChops, ImageOps

    # Dilation over a four-neighbour cross preserves thin silhouettes. Padding
    # each tile separately keeps colours from leaking across atlas seams.
    image = Image.frombytes("RGBA", (width, size), bytes(rgba))

    def neighbour(padded, dx, dy):
        x, y = 1 + dx, 1 + dy
        return padded.crop((x, y, x + size, y + size))

    for left in range(0, width, size):
        if check_cancel:
            check_cancel()
        box = (left, 0, left + size, size)
        alpha = image.crop(box).getchannel("A")
        padded = ImageOps.expand(alpha, border=1)
        horizontal = ImageChops.lighter(neighbour(padded, -1, 0), neighbour(padded, 1, 0))
        vertical = ImageChops.lighter(neighbour(padded, 0, -1), neighbour(padded, 0, 1))
        coverage = ImageChops.lighter(horizontal, vertical)
        halo = ImageChops.multiply(coverage, alpha.point([255] + [0] * 255))
        outline = Image.new("RGBA", (size, size), tuple(ink) + (255,))
        outline.putalpha(halo)
        image.paste(outline, box, halo.point([0] + [255] * 255))
    return image.tobytes()


_CHARACTER = {"resolution": 128, "slices": 80, "palette": 0, "outline": True}


def preset_fields(fields):
    """Apply the character defaults while preserving authored values, including zero/false."""
    if fields.get("preset") != "character":
        return fields
    return {**_CHARACTER, **{k: v for k, v in fields.items() if v is not None and v != ""}}


def _rasterize(kind, primitives, grid, palette_image, limit, outline, check_cancel):
    size, layers = grid.size, grid.layers
    width = size * layers
    rgba = bytearray(width * size * 4)
    if kind == "mesh":
        _paint_mesh(grid, primitives, rgba, check_cancel)
    else:
        _paint_splats(grid, primitives, rgba, check_cancel)
    ink = bytes((24, 24, 24))
    if limit and (palette_image is not None or any(rgba[3::4])):
        ink = _quantize(rgba, palette_image, limit, outline, check_cancel)
    if outline:
        return _outline(rgba, size, width, ink, check_cancel)
    return bytes(rgba)


def slice_sequence(model_bytes, fields, palette_image, decode_image, *, check_cancel=None):
    """Return a shared layout and an iterator of packed full-size RGBA layer atlases."""
    fields = preset_fields(fields)
    size = _integer(fields, "resolution", 48, 16, 128)
    layers = _integer(fields, "slices", 48, 8, 128)
    limit = _integer(fields, "palette", 16, 0, 64)
    count = _integer(fields, "frames", 1, 1, 64)
    kind, clip, start, duration, pose_at = read_model_poses(
        model_bytes, decode_image, fields.get("clip") or "", count, check_cancel=check_cancel)
    axis = fields.get("upAxis") or "auto"
    if axis == "auto":
        axis = "z" if kind == "splat" else "y"
    outline = fields.get("outline") is None or fields["outline"] is True or fields["outline"] == "true"
    first = pose_at(0)

    def later_poses():
        for index in range(1, count):
            if check_cancel:
                check_cancel()
            yield pose_at(index)

    # Fit all sampled poses together and keep just one additional pose resident.
    grid = _Grid(kind, chain(first, chain.from_iterable(later_poses())), axis, size, layers)

    # Keep integer coordinates in the same JSON spelling as saved JS layouts.
    def json_number(value):
        return int(value) if abs(value) < 1e21 and int(value) == value else value

    layout = {"version": 1, "order": "bottom-to-top", "columns": layers, "rows": 1,
              "tileWidth": size, "tileHeight": size, "upAxis": axis, "source": kind,
              "bounds": {"min": [json_number(x) for x in grid.low], "max": [json_number(x) for x in grid.high]},
              "pixelSize": json_number(grid.unit), "layerHeight": json_number(grid.step)}
    if count > 1:
        layout["animation"] = {"clip": clip, "frames": count, "duration": duration,
                               "times": [start + i * duration / count for i in range(count)]}

    def atlases():
        for primitives in chain((first,), later_poses()):
            yield _rasterize(kind, primitives, grid, palette_image, limit, outline, check_cancel)

    return size * layers, size, atlases(), layout


def slice_model(model_bytes, fields, palette_image, decode_image, *, check_cancel=None):
    """Return the first atlas and layout (static slice helper)."""
    width, height, atlases, layout = slice_sequence(
        model_bytes, fields, palette_image, decode_image, check_cancel=check_cancel)
    return width, height, next(atlases), layout
