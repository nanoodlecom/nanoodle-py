"""Edit nodes surface EVERY image of a multi-image response, and per-item
layer metadata (layer-decomposition models) survives aligned by index —
mirrors the JS edit-layers tests. Before the fix, _gen_image's response map
kept only url/b64 and _kept_first dropped all but the first image, so a
layerize graph received the base plate alone."""

import unittest

from tests._util import MockedTest

L0 = {"z_index": 0, "name": None, "description": None, "bounding_box": None}
L1 = {"z_index": 1, "name": "dark background", "description": "The dark backdrop.",
      "bounding_box": {"absolute": [0, 0, 720, 1304], "normalized": [0, 0, 1000, 906]}}
L2 = {"z_index": 2, "name": "marble bust", "description": "The bust cutout.",
      "bounding_box": {"absolute": [78, 100, 646, 1300], "normalized": [108, 69, 897, 903]}}
LAYER_BODY = {"data": [
    {"url": "https://cdn.example/layer-0.png", "layer": L0},
    {"url": "https://cdn.example/layer-1.png", "layer": L1},
    {"url": "https://cdn.example/layer-2.png", "layer": L2},
]}


def edit_wf(test, model="bytedance/seedream/v5/flash/layerize"):
    return test.wf_dict({"nodes": [
        {"id": "n1", "type": "upload", "fields": {"image": "data:image/png;base64,AAA="}},
        {"id": "n2", "type": "edit", "fields": {"model": model, "prompt": "Separate into layers"}},
    ], "links": [{"id": "l1", "from": {"node": "n1", "port": "image"},
                  "to": {"node": "n2", "port": "image"}}]})


class EditLayersTest(MockedTest):
    def test_layerize_response_surfaces_every_layer_and_metadata(self):
        self.mock.script("POST", "/v1/images/generations", {"status": 200, "json": LAYER_BODY})
        result = edit_wf(self).run()
        out = result.nodes["n2"].out
        self.assertEqual(out["image"].url, "https://cdn.example/layer-0.png")
        self.assertEqual([u.url for u in out["images"]],
                         ["https://cdn.example/layer-0.png",
                          "https://cdn.example/layer-1.png",
                          "https://cdn.example/layer-2.png"])
        self.assertEqual(out["layers"], [L0, L1, L2])

    def test_partial_metadata_stays_index_aligned_with_nones(self):
        body = {"data": [{"url": "https://cdn.example/a.png"},
                         {"url": "https://cdn.example/b.png", "layer": L2}]}
        self.mock.script("POST", "/v1/images/generations", {"status": 200, "json": body})
        out = edit_wf(self).run().nodes["n2"].out
        self.assertEqual(out["layers"], [None, L2])


if __name__ == "__main__":
    unittest.main()
