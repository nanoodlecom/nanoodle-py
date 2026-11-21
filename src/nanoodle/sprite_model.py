"""Read static glTF scenes and anisotropic Gaussian PLYs for the sprite slicer."""

import io
import json
import math
import re
import struct
from dataclasses import dataclass

from .errors import NanoodleError
from .media import parse_data_url


@dataclass(slots=True)
class Mesh:
    positions: list
    idx: list | None
    mode: int
    length: int
    color: list | None
    uv: list | None
    texture: tuple | None
    factor: tuple
    alpha_mode: str
    cutoff: float


@dataclass(slots=True)
class Splat:
    center: tuple
    scales: tuple
    rotation: tuple
    extent: tuple
    color: tuple
    alpha: float


_IDENTITY = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
_GL_TYPES = {
    5120: ("b", 127), 5121: ("B", 255), 5122: ("h", 32767),
    5123: ("H", 65535), 5125: ("I", 4294967295), 5126: ("f", 1),
}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_PLY_TYPES = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def _trs(t, r, s):
    x, y, z, w = r
    sx, sy, sz = s
    xx, xy, xz = x * (x + x), x * (y + y), x * (z + z)
    yy, yz, zz = y * (y + y), y * (z + z), z * (z + z)
    wx, wy, wz = w * (x + x), w * (y + y), w * (z + z)
    return (
        (1 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0,
        (xy - wz) * sy, (1 - (xx + zz)) * sy, (yz + wx) * sy, 0,
        (xz + wy) * sz, (yz - wx) * sz, (1 - (xx + yy)) * sz, 0,
        t[0], t[1], t[2], 1,
    )


def _multiply(a, b):
    # glTF matrices are column-major.
    return tuple(a[r] * b[c] + a[r+4] * b[c+1] + a[r+8] * b[c+2] + a[r+12] * b[c+3]
                 for c in (0, 4, 8, 12) for r in range(4))


def _vector(value, size):
    if (not isinstance(value, (tuple, list)) or len(value) != size
            or not all(type(v) in (int, float) and math.isfinite(v) for v in value)):
        raise NanoodleError("Invalid glTF scene transform or material factor.")
    return tuple(value)


def _item(items, index, message):
    if type(index) is not int or not 0 <= index < len(items):
        raise NanoodleError(message)
    return items[index]


def _embedded(uri):
    if not isinstance(uri, str) or not uri.startswith("data:"):
        raise NanoodleError("External glTF buffers and images require a self-contained .gltf or .glb export.")
    return parse_data_url(uri)[1]


def _unpack_glb(data):
    if len(data) < 20 or struct.unpack_from("<II", data, 4) != (2, len(data)):
        raise NanoodleError("Invalid GLB v2 header or length.")
    doc = binary = None
    offset = 12
    while offset < len(data):
        if len(data) - offset < 8:
            raise NanoodleError("Truncated GLB chunk header.")
        length, kind = struct.unpack_from("<I4s", data, offset)
        offset += 8
        end = offset + length
        if end > len(data):
            raise NanoodleError("Truncated GLB chunk.")
        if kind == b"JSON":
            doc = json.loads(data[offset:end])
        elif kind == b"BIN\0":
            binary = memoryview(data)[offset:end]
        offset = end
    return doc, binary


class _Scene:
    def __init__(self, doc, binary, decode_image, check_cancel):
        if not isinstance(doc, dict) or doc.get("asset", {}).get("version") != "2.0":
            raise NanoodleError("Expected glTF 2.0.")
        unsupported = [ext for ext in doc.get("extensionsRequired", ()) if ext != "EXT_texture_webp"]
        if unsupported:
            raise NanoodleError("Unsupported required glTF extensions: " + ", ".join(unsupported))
        self.doc = doc
        self.decode_image = decode_image
        self.check_cancel = check_cancel
        self.buffers = []
        self.images = {}
        self.meshes = []
        for i, spec in enumerate(doc.get("buffers", [])):
            data = _embedded(spec["uri"]) if spec.get("uri") else (binary if i == 0 else None)
            size = spec.get("byteLength")
            if data is None or type(size) is not int or not 0 <= size <= len(data):
                raise NanoodleError("Missing or truncated glTF buffer.")
            self.buffers.append(memoryview(data)[:size])

    def buffer_view(self, index):
        msg = "Invalid glTF buffer view."
        bv = _item(self.doc.get("bufferViews", []), index, msg)
        data = _item(self.buffers, bv.get("buffer"), msg)
        start, size = bv.get("byteOffset", 0), bv.get("byteLength")
        if type(start) is not int or type(size) is not int or min(start, size) < 0 or start + size > len(data):
            raise NanoodleError(msg)
        if "EXT_meshopt_compression" in bv.get("extensions", {}):
            raise NanoodleError("Meshopt-compressed buffers are unsupported.")
        return data[start:start + size], bv

    def accessor(self, index, shapes, expected=None):
        msg = "Unsupported glTF accessor type or sparse encoding."
        spec = _item(self.doc.get("accessors", []), index, msg)
        shape, component = spec.get("type"), spec.get("componentType")
        if shape not in shapes or component not in _GL_TYPES or "sparse" in spec:
            raise NanoodleError(msg)
        buf, view = self.buffer_view(spec.get("bufferView"))
        code, maximum = _GL_TYPES[component]
        row = struct.Struct("<" + code * _WIDTHS[shape])
        count, offset, stride = spec.get("count"), spec.get("byteOffset", 0), view.get("byteStride", row.size)
        if (type(count) is not int or count < 1 or (expected is not None and expected != count)
                or type(offset) is not int or offset < 0 or type(stride) is not int or stride < row.size
                or offset + (count - 1) * stride + row.size > len(buf)):
            raise NanoodleError("Invalid glTF accessor bounds or count.")
        if stride == row.size:
            values = list(row.iter_unpack(buf[offset:offset + count * stride]))
        else:
            values = [row.unpack_from(buf, offset + i * stride) for i in range(count)]
        if code == "f" and not all(math.isfinite(v) for point in values for v in point):
            raise NanoodleError("Non-finite glTF attribute.")
        if spec.get("normalized"):
            values = [tuple(max(-1, v / maximum) for v in point) for point in values]
        return values

    def texture(self, info):
        msg = "Unsupported base-color texture encoding or transform."
        if "KHR_texture_transform" in info.get("extensions", {}):
            raise NanoodleError(msg)
        texture = _item(self.doc.get("textures", []), info.get("index"), msg)
        extensions = texture.get("extensions", {})
        if not isinstance(extensions, dict):
            raise NanoodleError(msg)
        use_webp = "EXT_texture_webp" in extensions
        index = extensions["EXT_texture_webp"] if use_webp else texture
        if not isinstance(index, dict):
            raise NanoodleError(msg)
        source = index.get("source")
        image = _item(self.doc.get("images", []), source, msg)
        if use_webp and "bufferView" in image and image.get("mimeType") != "image/webp":
            raise NanoodleError("EXT_texture_webp buffer view must use image/webp.")
        if source not in self.images:
            data = _embedded(image["uri"]) if image.get("uri") else bytes(self.buffer_view(image.get("bufferView"))[0])
            is_webp = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
            self.images[source] = (self.decode_image(data), is_webp)
        pixels, is_webp = self.images[source]
        if use_webp and not is_webp:
            raise NanoodleError("EXT_texture_webp requires WebP image data.")
        sampler = {} if "sampler" not in texture else _item(self.doc.get("samplers", []), texture["sampler"], msg)
        return pixels, sampler

    def emit(self, index, world):
        mesh = _item(self.doc.get("meshes", []), index, "Missing glTF mesh.")
        for primitive in mesh["primitives"]:
            if self.check_cancel:
                self.check_cancel()
            mode = primitive.get("mode", 4)
            if mode not in (4, 5, 6):
                raise NanoodleError("Slice 3D requires triangles. For a point-cloud preview, supply the original Gaussian PLY.")
            if primitive.get("targets") or "KHR_draco_mesh_compression" in primitive.get("extensions", {}):
                raise NanoodleError("Morph targets and Draco-compressed meshes are unsupported.")
            attrs = primitive["attributes"]
            positions = self.accessor(attrs.get("POSITION"), ("VEC3",))
            if world != _IDENTITY:
                positions = [(world[0]*x + world[4]*y + world[8]*z + world[12],
                              world[1]*x + world[5]*y + world[9]*z + world[13],
                              world[2]*x + world[6]*y + world[10]*z + world[14]) for x, y, z in positions]
                if not all(math.isfinite(v) for point in positions for v in point):
                    raise NanoodleError("Invalid glTF scene transform.")
            count = len(positions)
            indices = None
            if "indices" in primitive:
                index_id = primitive["indices"]
                index_spec = _item(self.doc.get("accessors", []), index_id, "Missing triangle indices.")
                if index_spec.get("componentType") not in (5121, 5123, 5125) or index_spec.get("normalized"):
                    raise NanoodleError("glTF triangle indices must be unsigned integers.")
                indices = [i for (i,) in self.accessor(index_id, ("SCALAR",))]
                if max(indices) >= count:
                    raise NanoodleError("Invalid triangle index.")
            length = len(indices) if indices is not None else count
            if length < 3 or (mode == 4 and length % 3):
                raise NanoodleError("Incomplete triangle primitive.")
            mat = {} if "material" not in primitive else _item(
                self.doc.get("materials", []), primitive["material"], "Missing glTF material.")
            alpha = mat.get("alphaMode", "OPAQUE")
            if alpha not in ("OPAQUE", "MASK", "BLEND"):
                raise NanoodleError("Invalid glTF alpha mode.")
            pbr = mat.get("pbrMetallicRoughness", {})
            factor = _vector(pbr.get("baseColorFactor", (1, 1, 1, 1)), 4)
            if not all(0 <= v <= 1 for v in factor):
                raise NanoodleError("Invalid glTF base color factor.")
            color = self.accessor(attrs["COLOR_0"], ("VEC3", "VEC4"), count) if "COLOR_0" in attrs else None
            texture = uv = None
            if "baseColorTexture" in pbr:
                info = pbr["baseColorTexture"]
                uv = self.accessor(attrs.get(f"TEXCOORD_{info.get('texCoord', 0)}"), ("VEC2",), count)
                texture = self.texture(info)
            self.meshes.append(Mesh(positions, indices, mode, length, color, uv, texture,
                                    factor, alpha, mat.get("alphaCutoff", 0.5)))

    def read(self):
        nodes = self.doc.get("nodes", [])
        if nodes:
            scenes = self.doc.get("scenes", [])
            if scenes:
                scene = _item(scenes, self.doc.get("scene", 0), "Invalid glTF scene.")
            else:
                scene = {}
            roots = scene.get("nodes")
            if roots is None:
                children = {i for node in nodes for i in node.get("children", [])}
                roots = [i for i in range(len(nodes)) if i not in children]
            stack = [(i, _IDENTITY) for i in reversed(roots)]
            visited = set()
            while stack:
                index, parent = stack.pop()
                msg = "Invalid or cyclic glTF node tree."
                node = _item(nodes, index, msg)
                if index in visited:
                    raise NanoodleError(msg)
                visited.add(index)
                if "skin" in node:
                    raise NanoodleError("Skinned meshes require a static mesh export.")
                if "matrix" in node:
                    local = _vector(node["matrix"], 16)
                else:
                    local = _trs(_vector(node.get("translation", (0, 0, 0)), 3),
                                 _vector(node.get("rotation", (0, 0, 0, 1)), 4),
                                 _vector(node.get("scale", (1, 1, 1)), 3))
                world = _multiply(parent, local)
                if "mesh" in node:
                    self.emit(node["mesh"], world)
                stack.extend((child, world) for child in reversed(node.get("children", [])))
        else:
            for index in range(len(self.doc.get("meshes", []))):
                self.emit(index, _IDENTITY)
        if not self.meshes:
            raise NanoodleError("No triangles in the selected glTF scene.")
        return self.meshes


def _ply_rows(data, offset, count, encoding, properties, keys):
    indices = [list(properties).index(k) for k in keys]
    if encoding == "ascii":
        stream = io.BytesIO(data)
        stream.seek(offset)
        for _ in range(count):
            fields = stream.readline().split()
            if len(fields) != len(properties):
                raise NanoodleError("Invalid ASCII PLY vertex.")
            yield tuple(float(fields[i]) for i in indices)
        if stream.read().strip():
            raise NanoodleError("Gaussian PLY payload does not match its declared vertex count.")
    else:
        # Skip unused high-order spherical harmonics without allocating Python floats for them.
        used_keys = set(keys)
        used_order = [name for name in properties if name in used_keys]
        order = [used_order.index(name) for name in keys]
        row = struct.Struct("<" + "".join(code if name in used_keys else f"{struct.calcsize('<'+code)}x"
                                          for name, code in properties.items()))
        if count * row.size != len(data) - offset:
            raise NanoodleError("Gaussian PLY payload does not match its declared vertex count.")
        for fields in row.iter_unpack(memoryview(data)[offset:]):
            yield tuple(fields[i] for i in order)


def _scaled_norm(values):
    # Scale first to preserve tiny and very large rotations. Matching the JS
    # operation order also keeps Gaussian bounds stable across executors.
    largest = max(map(abs, values))
    if largest == 0:
        return 0.0
    total = compensation = 0.0
    for v in values:
        ratio = v / largest
        term = ratio * ratio - compensation
        next_total = total + term
        compensation = (next_total - total) - term
        total = next_total
    return math.sqrt(total) * largest


def _splats(data, check_cancel):
    end = re.search(br"^end_header\r?\n", data[:65536], re.MULTILINE)
    if not end:
        raise NanoodleError("Missing PLY header terminator.")
    properties = {}
    element = encoding = count = None
    for line in data[:end.end()].decode("utf-8").splitlines():
        field = line.split()
        if not field:
            continue
        if field[0] == "format":
            if len(field) != 3 or field[2] != "1.0" or field[1] not in ("ascii", "binary_little_endian"):
                raise NanoodleError("Supported Gaussian PLY formats are ASCII and binary little-endian 1.0.")
            encoding = field[1]
        elif field[0] == "element":
            if len(field) != 3:
                raise NanoodleError("Invalid Gaussian PLY element.")
            element, element_count = field[1], int(field[2])
            if element == "vertex":
                if count is not None or element_count < 1:
                    raise NanoodleError("Invalid Gaussian PLY vertex count.")
                count = element_count
            elif element_count != 0:
                raise NanoodleError("Expected an uncompressed vertex-only Gaussian PLY.")
        elif field[0] == "property" and element == "vertex":
            if len(field) != 3 or field[1] not in _PLY_TYPES or field[2] in properties:
                raise NanoodleError("Unsupported Gaussian PLY property encoding.")
            properties[field[2]] = _PLY_TYPES[field[1]]
    keys = ("x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity")
    dc, rgb = ("f_dc_0", "f_dc_1", "f_dc_2"), ("red", "green", "blue")
    has_dc = all(k in properties for k in dc)
    keys += dc if has_dc else rgb
    if not encoding or not count or any(k not in properties for k in keys):
        raise NanoodleError("Expected Gaussian PLY positions, log scales, quaternion, opacity logit, and DC or RGB colors. Ordinary point clouds are unsupported.")
    if not has_dc and any(properties[k] != "B" for k in rgb):
        raise NanoodleError("Gaussian RGB properties must be unsigned bytes, or supply floating-point DC coefficients.")
    splats = []
    for i, values in enumerate(_ply_rows(data, end.end(), count, encoding, properties, keys)):
        if check_cancel and i % 1024 == 0:
            check_cancel()
        if not all(map(math.isfinite, values)):
            raise NanoodleError("Non-finite Gaussian PLY attribute.")
        center = values[:3]
        try:
            scales = tuple(map(math.exp, values[3:6]))
        except OverflowError:
            raise NanoodleError("Invalid Gaussian scale or rotation.") from None
        norm = _scaled_norm(values[6:10])
        if not norm or not all(scales) or not math.isfinite(norm):
            raise NanoodleError("Invalid Gaussian scale or rotation.")
        w, x, y, z = (v / norm for v in values[6:10])
        rotation = _trs((0, 0, 0), (x, y, z, w), (1, 1, 1))
        extent = tuple(3 * _scaled_norm((rotation[k]*scales[0], rotation[k+4]*scales[1],
                                         rotation[k+8]*scales[2])) for k in range(3))
        opacity = values[10]
        alpha = 0.0 if opacity < -709.78 else 1 / (1 + math.exp(-opacity))
        color = tuple(255*(0.5 + 0.2820947918*v) for v in values[11:]) if has_dc else values[11:]
        splats.append(Splat(center, scales, rotation, extent, color, alpha))
    return splats


def read_model(raw_bytes, decode_image, *, check_cancel=None):
    """Return (source kind, primitives). decode_image converts embedded image bytes to (w, h, RGBA)."""
    try:
        if raw_bytes.startswith(b"ply"):
            return "splat", _splats(raw_bytes, check_cancel)
        if raw_bytes.startswith(b"glTF"):
            doc, binary = _unpack_glb(raw_bytes)
        else:
            try:
                doc = json.loads(raw_bytes)
            except (ValueError, UnicodeError):
                raise NanoodleError("Expected a GLB, a self-contained glTF 2 mesh, or a Gaussian PLY.") from None
            binary = None
        return "mesh", _Scene(doc, binary, decode_image, check_cancel).read()
    except (TypeError, KeyError, ValueError, AttributeError, OverflowError, struct.error) as exc:
        raise NanoodleError(f"Invalid sprite model: {exc}") from exc
