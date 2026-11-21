"""Saved graph inputs and Khronos glTF WebP materials through the local executors."""
import json
import struct
import unittest
from pathlib import Path

from nanoodle import Workflow
from nanoodle.errors import NanoodleError
from nanoodle.media import make_data_url
from nanoodle.sprite_codecs import decode_image, encode_png
from nanoodle.sprite_geometry import slice_model
from tests import fixture

OPTIONS = {"slices": 8, "resolution": 16, "palette": 0, "outline": False}


def textured_triangle(webp, fallback, *, required, buffer_mime=None):
    positions = struct.pack("<9f", 0, 0, 0, 0, 1, 2, 2, 2, 0)
    uvs = struct.pack("<6f", 0.05, 0.7, 0.7, 0.7, 0.5, 0.2)
    buf = positions + uvs
    images = [{"uri": make_data_url(fallback, "image/png")}]
    views = [{"buffer": 0, "byteLength": len(positions)},
             {"buffer": 0, "byteOffset": len(positions), "byteLength": len(uvs)}]
    if buffer_mime is None:
        images.append({"uri": make_data_url(webp, "image/webp")})
    else:
        images.append({"bufferView": 2, "mimeType": buffer_mime})
        views.append({"buffer": 0, "byteOffset": len(buf), "byteLength": len(webp)})
        buf += webp
    tex = {"extensions": {"EXT_texture_webp": {"source": 1}}}
    model = {
        "asset": {"version": "2.0"}, "extensionsUsed": ["EXT_texture_webp"],
        "buffers": [{"uri": make_data_url(buf, "application/octet-stream"), "byteLength": len(buf)}],
        "bufferViews": views,
        "accessors": [{"bufferView": view, "componentType": 5126, "count": 3, "type": typ}
                      for view, typ in enumerate(("VEC3", "VEC2"))],
        "images": images, "textures": [tex],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "material": 0}]}],
        "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0,
    }
    if required:
        model["extensionsRequired"] = ["EXT_texture_webp"]
    else:
        tex["source"] = 0
    return model


def slice_sheet(model):
    fields = {"model": make_data_url(json.dumps(model).encode(), "model/gltf+json")}
    graph = {"nodes": [{"id": "m", "type": "mupload", "fields": fields},
                       {"id": "s", "type": "slice-3d", "fields": OPTIONS}],
             "links": [{"from": {"node": "m", "port": "model"}, "to": {"node": "s", "port": "model"}}]}
    return Workflow.from_dict(graph).run().nodes["s"].out["sheet"].bytes()


class WebPSpriteGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sheet = Path(fixture("media/sprite-sheet.webp")).read_bytes()
        cls.fallback = encode_png(1, 1, bytes((255, 0, 255, 255)))

    def test_raw_sheet_and_palette_use_the_same_constructed_graphs_as_png(self):
        png = encode_png(*decode_image(self.sheet))

        def images(data, mime):
            graph = json.loads(Path(fixture("sprite-stack.json")).read_text())
            graph["nodes"][1]["fields"]["palette"] = 8
            graph["nodes"][2]["fields"]["turntable"] = False
            graph["nodes"].append({"id": "picture", "type": "upload", "fields": {"image": make_data_url(data, mime)}})
            graph["links"].append({"from": {"node": "picture", "port": "image"},
                                   "to": {"node": "s", "port": "paletteImage"}})
            graph["links"][1]["from"] = {"node": "picture", "port": "image"}
            result = Workflow.from_dict(graph).run()
            return result.nodes["r"].out["image"].bytes(), result.nodes["s"].out["sheet"].bytes()

        self.assertEqual(images(self.sheet, "image/webp"), images(png, "image/png"))

    def test_gltf_prefers_webp_and_accepts_required_buffer_view(self):
        model = textured_triangle(self.sheet, self.fallback, required=False)
        result = slice_sheet(model)
        reference = textured_triangle(self.sheet, self.fallback, required=False)
        reference["textures"] = [{"source": 0}]
        self.assertNotEqual(slice_sheet(reference), result)
        reference["images"][0]["uri"] = make_data_url(encode_png(*decode_image(self.sheet)), "image/png")
        self.assertEqual(slice_sheet(reference), result)
        self.assertEqual(slice_sheet(textured_triangle(self.sheet, self.fallback, required=True,
                                                      buffer_mime="image/webp")), result)

    def test_gltf_rejects_wrong_mime_content_and_unknown_required_extensions(self):
        model = textured_triangle(self.sheet, self.fallback, required=True, buffer_mime="image/png")
        for expected in ("image/webp", "WebP image data", "UnexpectedExample"):
            with self.subTest(expected=expected), self.assertRaisesRegex(NanoodleError, expected):
                slice_model(json.dumps(model).encode(), OPTIONS, None, decode_image)
            if expected == "image/webp":
                model = textured_triangle(self.fallback, self.fallback, required=True, buffer_mime="image/webp")
            else:
                model["extensionsRequired"].append("UnexpectedExample")
