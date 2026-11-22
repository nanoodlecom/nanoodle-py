"""S1 — labeled-assets surfacing (twin of the JS assets-surfacing tests).

Completed jobs return companion assets next to the primary output (MoGe-3:
depth / normals / mask / point cloud beside the GLB; LTX AlphaGen: alpha
matte + ProRes cutout beside the video). Executors used to keep only the
primary URL and drop the rest. Fixtures are trimmed copies of REAL
2026-11-21 NanoGPT payloads (Edge Styles Lab probes: moge3, alphagen,
convert).
"""

import json
import unittest

from nanoodle.engine import _parse_status_assets, _asset_outputs
from tests._util import FAST, MockedTest

GLB = "https://v3b.fal.media/files/b/0ab36d09/GLB_mesh.glb"
PLY = "https://v3b.fal.media/files/b/0ab36d09/PC_pointcloud.ply"
DEPTH = "https://v3b.fal.media/files/b/0ab36d09/DEPTH.png"
NORMALS = "https://v3b.fal.media/files/b/0ab36d09/NORMALS.png"
MASK = "https://v3b.fal.media/files/b/0ab36d09/MASK.png"
MATTE = "https://v3b.fal.media/files/b/0ab36d6e/MATTE_matte.mp4"
CUTOUT = "https://v3b.fal.media/files/b/0ab36d6e/CUT_cutout.mov"
PROXY = "https://nano-gpt.com/v1/media/content?rid=abc"

MOGE_OUT = {
    "kind": "3d", "format": "glb", "model_url": GLB,
    "model3d_assets": [
        {"label": "Textured GLB", "filename": "mesh.glb", "url": GLB, "contentType": "model/gltf-binary"},
        {"label": "Colored point cloud PLY", "filename": "pointcloud.ply", "url": PLY, "contentType": "application/octet-stream"},
        {"label": "Colorized depth map PNG", "filename": "depth.png", "url": DEPTH, "contentType": "image/png"},
        {"label": "Surface normals PNG", "filename": "normals.png", "url": NORMALS, "contentType": "image/png"},
        {"label": "Validity mask PNG", "filename": "mask.png", "url": MASK, "contentType": "image/png"},
    ],
}

ALPHAGEN_OUT = {
    "video": {"url": PROXY},
    "processing_assets": [
        {"label": "Alpha matte MP4", "filename": "matte.mp4", "url": MATTE, "contentType": "video/mp4"},
        {"label": "Straight-alpha ProRes 4444 MOV", "filename": "cutout.mov", "url": CUTOUT, "contentType": "video/quicktime"},
    ],
}

CONVERT_OUT = {
    "kind": "3d", "format": "fbx", "model_url": "https://cdn.example/converted.fbx",
    "model3d_assets": [
        {"label": "FBX", "filename": "converted.fbx", "url": "https://cdn.example/converted.fbx",
         "contentType": "application/octet-stream"},
    ],
}

CATALOG = {"model3d": [
    {"id": "microsoft/moge-3", "architecture": {"input_modalities": ["image"]}},
    {"id": "tripo3d/convert", "architecture": {"input_modalities": ["3d"]}},
]}


def status(output):
    return {"status": 200, "json": {"data": {"status": "COMPLETED", "output": output}}}


class _FakeEngine:
    def _media_ref(self, url, mime=None):
        return url


class ParseTest(unittest.TestCase):
    def test_parse_normalizes(self):
        assets = _parse_status_assets({"data": {"output": MOGE_OUT}})
        self.assertEqual(len(assets), 5)
        self.assertEqual([a["kind"] for a in assets], ["model3d", "model3d", "image", "image", "image"])
        self.assertEqual(assets[1]["label"], "Colored point cloud PLY")
        self.assertEqual(_parse_status_assets({"data": {"output": {}}}), [])

    def test_role_outputs_moge(self):
        out = _asset_outputs(_FakeEngine(), _parse_status_assets({"data": {"output": MOGE_OUT}}), GLB, "model3d")
        self.assertEqual(out["depth"], DEPTH)
        self.assertEqual(out["normals"], NORMALS)
        self.assertEqual(out["mask"], MASK)
        self.assertEqual(out["points"], PLY)
        self.assertNotIn("model2", out)
        self.assertEqual(len(json.loads(out["assets"])), 5)
        # assets JSON matches the JS stringify shape (no spaces)
        self.assertNotIn('": ', out["assets"])

    def test_role_outputs_alphagen(self):
        out = _asset_outputs(_FakeEngine(), _parse_status_assets({"data": {"output": ALPHAGEN_OUT}}), PROXY, "video")
        self.assertEqual(out["matte"], MATTE)
        self.assertEqual(out["cutout"], CUTOUT)
        self.assertEqual(_asset_outputs(_FakeEngine(), [], "x", "video"), {})


class WorkflowTest(MockedTest):
    def _run(self, nodes, links, output, catalog=None):
        self.mock.script("POST", "/api/generate-video", {"status": 200, "json": {"runId": "r1", "cost": 0.03}})
        self.mock.script("GET", "/api/video/status", status(output))
        opts = dict(FAST)
        if catalog:
            opts["catalog"] = catalog
        return self.wf_dict({"nodes": nodes, "links": links}, **opts).run()

    def test_model3d_moge_surfaces_roles(self):
        nodes = [
            {"id": "up", "type": "upload", "fields": {"image": "data:image/png;base64,AAA="}},
            {"id": "m", "type": "model3d", "fields": {"model": "microsoft/moge-3"}},
        ]
        links = [{"id": "l1", "from": {"node": "up", "port": "image"}, "to": {"node": "m", "port": "image"}}]
        out = self._run(nodes, links, MOGE_OUT, CATALOG).nodes["m"].out
        self.assertEqual(out["model"].url, GLB)
        self.assertEqual(out["depth"].url, DEPTH)
        self.assertEqual(out["normals"].url, NORMALS)
        self.assertEqual(out["mask"].url, MASK)
        self.assertEqual(out["points"].url, PLY)
        self.assertEqual(len(json.loads(out["assets"])), 5)

    def test_vedit_alphagen_surfaces_matte_cutout(self):
        nodes = [
            {"id": "v", "type": "vupload", "fields": {"video": "https://cdn.example/in.mp4"}},
            {"id": "e", "type": "vedit", "fields": {"model": "lightricks/ltx-2.5/alpha-gen", "prompt": "extract alpha"}},
        ]
        links = [{"id": "l1", "from": {"node": "v", "port": "video"}, "to": {"node": "e", "port": "video"}}]
        out = self._run(nodes, links, ALPHAGEN_OUT).nodes["e"].out
        self.assertEqual(out["matte"].url, MATTE)
        self.assertEqual(out["cutout"].url, CUTOUT)
        self.assertEqual(len(json.loads(out["assets"])), 2)

    def test_model3d_convert_single_asset_no_roles(self):
        nodes = [
            {"id": "u", "type": "mupload", "fields": {"model": "https://cdn.example/in.glb"}},
            {"id": "m", "type": "model3d", "fields": {"model": "tripo3d/convert"}},
        ]
        links = [{"id": "l1", "from": {"node": "u", "port": "model"}, "to": {"node": "m", "port": "model"}}]
        out = self._run(nodes, links, CONVERT_OUT, CATALOG).nodes["m"].out
        self.assertEqual(out["model"].url, "https://cdn.example/converted.fbx")
        self.assertEqual(len(json.loads(out["assets"])), 1)
        for port in ("depth", "normals", "mask", "points", "model2"):
            self.assertNotIn(port, out)

    def test_model3d_no_assets_no_new_ports(self):
        nodes = [
            {"id": "up", "type": "upload", "fields": {"image": "data:image/png;base64,AAA="}},
            {"id": "m", "type": "model3d", "fields": {"model": "microsoft/moge-3"}},
        ]
        links = [{"id": "l1", "from": {"node": "up", "port": "image"}, "to": {"node": "m", "port": "image"}}]
        plain = {"kind": "3d", "format": "glb", "model_url": "https://cdn.example/plain.glb"}
        out = self._run(nodes, links, plain, CATALOG).nodes["m"].out
        self.assertEqual(list(out.keys()), ["model"])


if __name__ == "__main__":
    unittest.main()
