"""Image-endpoint nodes: image / edit / inpaint — exact payloads, b64 mime
sniffing, multi-reference shapes, and the local media-size guards."""

import base64
import unittest

from tests._util import MockedTest
from tests.harness import image_response

from nanoodle import RunError

BIG_DATA_URL = "data:image/png;base64," + "A" * (4_700_000)   # > 4.4MB inline cap


class ImageNodeTest(MockedTest):
    def test_variations_seed_and_jpeg_sniff(self):
        jpeg_b64 = base64.b64encode(b"\xff\xd8\xffmockjpeg").decode()
        self.mock.script("POST", "/v1/images/generations",
                         image_response(b64_list=[jpeg_b64, jpeg_b64]))
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "image",
             "fields": {"model": "m", "prompt": "p", "variations": "2", "seed": "42"}}]})
        result = wf.run()
        req = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(req.json, {"model": "m", "size": "1024x1024", "n": 2,
                                    "response_format": "b64_json", "prompt": "p",
                                    "seed": 42})
        self.assertTrue(result["Image"].url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(len(result.nodes["n1"].out["images"]), 2)

    def test_b64_sniff_table(self):
        # png / gif / webp prefixes + unknown default png (play.html table)
        for b64, mime in (("iVBORxxxx", "image/png"), ("R0lGxxxx", "image/gif"),
                          ("UklGRxxxx", "image/webp"), ("QUJDRA==", "image/png")):
            self.mock.reset()
            self.mock.script("POST", "/v1/images/generations", image_response(b64_list=[b64]))
            wf = self.wf_dict({"nodes": [
                {"id": "n1", "type": "image", "fields": {"model": "m", "prompt": "p"}}]})
            self.assertEqual(wf.run()["Image"].url, "data:%s;base64,%s" % (mime, b64))

    def test_url_entries_pass_through(self):
        self.mock.script("POST", "/v1/images/generations",
                         image_response(urls=["https://cdn.example/a.png"]))
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "image", "fields": {"model": "m", "prompt": "p"}}]})
        self.assertEqual(wf.run()["Image"].url, "https://cdn.example/a.png")

    def test_empty_data_is_no_image_error(self):
        self.mock.script("POST", "/v1/images/generations", {"status": 200, "json": {"data": []}})
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "image", "fields": {"model": "m", "prompt": "p"}}]})
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("no image in response", str(ctx.exception))

    def test_missing_model_error_before_network(self):
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "image", "fields": {"prompt": "p"}}]})
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("pick a model first", str(ctx.exception))
        self.assertEqual(self.mock.requests, [])

    def test_custom_civitai_air_rides_in_body(self):
        self.mock.script("POST", "/v1/images/generations", image_response(urls=["https://x/y"]))
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "image",
             "fields": {"model": "custom-civitai", "prompt": "p",
                        "customCivitaiAir": "urn:air:sdxl:ckpt@1"}}]})
        wf.run()
        req = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(req.json["customCivitaiAir"], "urn:air:sdxl:ckpt@1")

    def test_non_numeric_seed_omitted(self):
        self.mock.script("POST", "/v1/images/generations", image_response(urls=["https://x/y"]))
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "image",
             "fields": {"model": "m", "prompt": "p", "seed": "random"}}]})
        wf.run()
        self.assertNotIn("seed", self.mock.requests_to("/v1/images/generations")[0].json)


class EditNodeTest(MockedTest):
    def test_multi_image_payload_is_array(self):
        self.mock.script("POST", "/v1/images/generations",
                         image_response(urls=["https://cdn.example/out.png"], cost=0.02))
        wf = self.wf("edit-multi.json")
        result = wf.run()
        req = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(req.json, {
            "model": "nano-banana-2",
            "size": "1k",
            "n": 1,
            "response_format": "b64_json",
            "prompt": "blend them",
            "imageDataUrl": ["data:image/png;base64,AAA=", "data:image/png;base64,BBB="],
        })
        self.assertEqual(result["Edit"].url, "https://cdn.example/out.png")

    def test_single_image_is_string_not_array(self):
        self.mock.script("POST", "/v1/images/generations", image_response(urls=["https://x/y.png"]))
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": "data:image/png;base64,AAA="}},
            {"id": "n2", "type": "edit", "fields": {"model": "m", "prompt": "p"}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "image"},
                      "to": {"node": "n2", "port": "image"}}]})
        wf.run()
        req = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(req.json["imageDataUrl"], "data:image/png;base64,AAA=")

    def test_no_image_input_errors(self):
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "edit", "fields": {"model": "m", "prompt": "p"}}]})
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("no image input", str(ctx.exception))

    def test_empty_prompt_refused_for_normal_model(self):
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": "data:image/png;base64,AAA="}},
            {"id": "n2", "type": "edit", "fields": {"model": "nano-banana-2", "prompt": ""}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "image"},
                      "to": {"node": "n2", "port": "image"}}]})
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("no edit instruction", str(ctx.exception))
        self.assertEqual(self.mock.requests, [])

    def test_upscaler_runs_with_empty_prompt_omitted_from_body(self):
        self.mock.script("POST", "/v1/images/generations", image_response(urls=["https://x/u.png"]))
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": "data:image/png;base64,AAA="}},
            {"id": "n2", "type": "edit", "fields": {"model": "clarity-upscaler", "prompt": ""}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "image"},
                      "to": {"node": "n2", "port": "image"}}]})
        wf.run()
        req = self.mock.requests_to("/v1/images/generations")[0]
        self.assertNotIn("prompt", req.json)
        self.assertEqual(req.json["imageDataUrl"], "data:image/png;base64,AAA=")

    def test_combined_reference_size_guard_no_network(self):
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": BIG_DATA_URL}},
            {"id": "n2", "type": "edit", "fields": {"model": "m", "prompt": "p"}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "image"},
                      "to": {"node": "n2", "port": "image"}}]})
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("too large", str(ctx.exception))
        self.assertEqual(self.mock.requests, [])


class InpaintNodeTest(MockedTest):
    def test_field_source_and_mask_pass_through(self):
        self.mock.script("POST", "/v1/images/generations", image_response(urls=["https://x/o.png"]))
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "inpaint",
             "fields": {"model": "m", "prompt": "a hat",
                        "image": "data:image/png;base64,SRC=",
                        "mask": "data:image/png;base64,MASK="}}]})
        wf.run()
        req = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(req.json["imageDataUrl"], "data:image/png;base64,SRC=")
        self.assertEqual(req.json["maskDataUrl"], "data:image/png;base64,MASK=")
        self.assertEqual(req.json["prompt"], "a hat")

    def test_wired_source_and_mask_win_over_fields(self):
        self.mock.script("POST", "/v1/images/generations", image_response(urls=["https://x/o.png"]))
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": "data:image/png;base64,WSRC"}},
            {"id": "n2", "type": "upload", "fields": {"image": "data:image/png;base64,WMASK"}},
            {"id": "n3", "type": "inpaint",
             "fields": {"model": "m", "prompt": "p",
                        "image": "data:image/png;base64,STALE",
                        "mask": "data:image/png;base64,STALE"}},
        ], "links": [
            {"id": "l1", "from": {"node": "n1", "port": "image"}, "to": {"node": "n3", "port": "image"}},
            {"id": "l2", "from": {"node": "n2", "port": "image"}, "to": {"node": "n3", "port": "mask"}},
        ]})
        wf.run()
        req = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(req.json["imageDataUrl"], "data:image/png;base64,WSRC")
        self.assertEqual(req.json["maskDataUrl"], "data:image/png;base64,WMASK")

    def test_missing_mask_is_upfront_named_error(self):
        # image wired, mask not: the mask is a required derived input -> named upfront
        from nanoodle import NanoodleError
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": "data:image/png;base64,AAA="}},
            {"id": "n2", "type": "inpaint", "fields": {"model": "m", "prompt": "p"}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "image"},
                      "to": {"node": "n2", "port": "image"}}]})
        with self.assertRaises(NanoodleError) as ctx:
            wf.run()
        self.assertIn("missing required input", str(ctx.exception))
        self.assertIn("Mask", str(ctx.exception))
        self.assertEqual(self.mock.requests, [])

    def test_mask_supplied_as_run_input(self):
        self.mock.script("POST", "/v1/images/generations", image_response(urls=["https://x/o.png"]))
        wf = self.wf_dict({"nodes": [
            {"id": "n1", "type": "upload", "fields": {"image": "data:image/png;base64,AAA="}},
            {"id": "n2", "type": "inpaint", "fields": {"model": "m", "prompt": "p"}},
        ], "links": [{"id": "l1", "from": {"node": "n1", "port": "image"},
                      "to": {"node": "n2", "port": "image"}}]})
        wf.run({"Mask (white = repaint)": "data:image/png;base64,RUNMASK"})
        req = self.mock.requests_to("/v1/images/generations")[0]
        self.assertEqual(req.json["maskDataUrl"], "data:image/png;base64,RUNMASK")


if __name__ == "__main__":
    unittest.main()
