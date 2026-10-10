"""🧊 3D model and 📦 3D input. Video-job submit + poll, mocked. No live calls."""

import os
import tempfile
import unittest

from tests._util import FAST, MockedTest

from nanoodle import MediaRef, RunError, estimate_graph_cost
from nanoodle.media import media_from_file, sniff_mime


def _status(output, status="COMPLETED"):
    return {"status": 200, "json": {"data": {"status": status, "output": output}}}


class Model3dRunTest(MockedTest):
    def test_modality_guards_make_no_request(self):
        catalog = {"model3d": [
            {"id": "text-only", "modalities": ["text"]},
            {"id": "both", "architecture": {"input_modalities": ["image", "text"]}},
        ]}
        # image-only default, nothing connected
        wf = self.wf_dict({"nodes": [
            {"id": "m", "type": "model3d", "fields": {"model": "tripo3d/v2.5"}},
        ]}, catalog=catalog, **FAST)
        image = [s for s in wf.inputs if s.field == "image"]
        self.assertEqual(len(image), 1)
        self.assertFalse(image[0].optional)
        self.assertFalse(any(s.field == "prompt" for s in wf.inputs))
        with self.assertRaises(Exception) as ctx:
            wf.run()
        self.assertIn("missing required input", str(ctx.exception))
        self.assertEqual(self.mock.requests, [])

        # text-only, empty prompt wired in (join with nothing to glue yields "")
        wf = self.wf_dict({"nodes": [
            {"id": "j", "type": "join", "fields": {}},
            {"id": "m", "type": "model3d", "fields": {"model": "text-only"}},
        ], "links": [
            {"id": "l", "from": {"node": "j", "port": "text"}, "to": {"node": "m", "port": "prompt"}},
        ]}, catalog=catalog, **FAST)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("Add a prompt first.", str(ctx.exception))
        self.assertEqual(self.mock.requests, [])

        # both modalities, neither supplied
        wf = self.wf_dict({"nodes": [
            {"id": "m", "type": "model3d", "fields": {"model": "both"}},
        ]}, catalog=catalog, **FAST)
        self.assertTrue(all(s.optional for s in wf.inputs))
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("Add a prompt or connect an image first.", str(ctx.exception))
        self.assertEqual(self.mock.requests, [])

    def test_image_only_omits_empty_prompt_and_polls_model_url(self):
        self.mock.script("POST", "/api/generate-video",
                         {"status": 200, "json": {"runId": "3d-1", "cost": 0.04}})
        self.mock.script("GET", "/api/video/status",
                         _status({"kind": "3d", "model_url": "https://cdn/cup.glb"}))
        wf = self.wf_dict({"nodes": [
            {"id": "u", "type": "upload", "fields": {"image": "https://cdn/photo.png"}},
            {"id": "m", "type": "model3d", "name": "Mesh",
             "fields": {"model": "tripo3d/v2.5", "prompt": "", "modelOpts": {"texture": "yes"}}},
        ], "links": [
            {"id": "l", "from": {"node": "u", "port": "image"}, "to": {"node": "m", "port": "image"}},
        ]}, **FAST)
        result = wf.run()
        body = self.mock.requests_to("/api/generate-video")[0].json
        self.assertEqual(body["model"], "tripo3d/v2.5")
        self.assertEqual(body["imageDataUrl"], "https://cdn/photo.png")
        self.assertEqual(body["texture"], "yes")
        self.assertNotIn("prompt", body)
        mesh = result["Mesh"]
        self.assertIsInstance(mesh, MediaRef)
        self.assertEqual(mesh.url, "https://cdn/cup.glb")
        self.assertEqual(mesh.mime, "model/gltf-binary")
        self.assertAlmostEqual(result.cost_usd, 0.04)
        self.assertIn("requestId=3d-1", self.mock.requests_to("/api/video/status")[0].query)

    def test_text_prompt_is_sent_and_format_glb_counts(self):
        self.mock.script("POST", "/api/generate-video",
                         {"status": 200, "json": {"runId": "3d-2"}})
        self.mock.script("GET", "/api/video/status",
                         _status({"format": "glb", "video": {"url": "https://cdn/from-format.glb"}}))
        wf = self.wf_dict({"nodes": [
            {"id": "m", "type": "model3d",
             "fields": {"model": "wavespeed-ai/hunyuan-3d-v3.1-rapid", "prompt": "a ceramic cup"}},
        ]}, **FAST)
        self.assertEqual(wf.run()["3D model"].url, "https://cdn/from-format.glb")
        body = self.mock.requests_to("/api/generate-video")[0].json
        self.assertEqual(body["prompt"], "a ceramic cup")
        self.assertNotIn("imageDataUrl", body)

    def test_bare_video_url_is_not_a_model(self):
        self.mock.script("POST", "/api/generate-video",
                         {"status": 200, "json": {"runId": "3d-3"}})
        self.mock.script("GET", "/api/video/status",
                         _status({"video": {"url": "https://cdn/not-a-model.mp4"}}))
        wf = self.wf_dict({"nodes": [
            {"id": "m", "type": "model3d",
             "fields": {"model": "tripo3d/v2.5", "image": "https://cdn/photo.png"}},
        ]}, **FAST)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("completed but no model url", str(ctx.exception))

    def test_failed_job_and_oversize_photo(self):
        self.mock.script("POST", "/api/generate-video",
                         {"status": 200, "json": {"runId": "3d-4"}})
        self.mock.script("GET", "/api/video/status",
                         {"status": 200, "json": {"data": {"status": "FAILED", "error": "no mesh"}}})
        wf = self.wf_dict({"nodes": [
            {"id": "m", "type": "model3d",
             "fields": {"model": "tripo3d/v2.5", "image": "https://cdn/photo.png"}},
        ]}, **FAST)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("3D failed: no mesh", str(ctx.exception))

        from nanoodle.media import MEDIA_INLINE_MAX
        huge = "data:image/png;base64," + ("A" * (MEDIA_INLINE_MAX + 10))
        wf = self.wf_dict({"nodes": [
            {"id": "m", "type": "model3d",
             "fields": {"model": "tripo3d/v2.5", "image": huge}},
        ]}, **FAST)
        before = len(self.mock.requests)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("3D photo is a bit large", str(ctx.exception))
        self.assertEqual(len(self.mock.requests), before)

    def test_model_id_is_not_scrubbed_and_upload_placeholder_is(self):
        wf = self.wf_dict({"nodes": [
            {"id": "m", "type": "model3d", "fields": {"model": "tripo3d/v2.5", "prompt": "cup"}},
            {"id": "u", "type": "mupload", "fields": {"model": "drop a glb here"}},
        ]})
        self.assertEqual(wf.graph.node("m").fields["model"], "tripo3d/v2.5")
        self.assertEqual(wf.graph.node("u").fields["model"], "")
        self.assertTrue(any("fields.model" in w for w in wf.warnings))

    def test_mupload_passes_the_file_through(self):
        wf = self.wf_dict({"nodes": [
            {"id": "u", "type": "mupload", "name": "Mesh in",
             "fields": {"model": "https://cdn/in.glb"}},
        ]}, **FAST)
        ref = wf.run()["Mesh in"]
        self.assertIsInstance(ref, MediaRef)
        self.assertEqual(ref.url, "https://cdn/in.glb")
        self.assertEqual(ref.mime, "model/gltf-binary")
        self.assertEqual(self.mock.requests, [])

        wf = self.wf_dict({"nodes": [
            {"id": "u", "type": "mupload", "fields": {"optional": True}},
        ]}, **FAST)
        self.assertEqual(wf.run()["3D input"], "")

    def test_per_run_estimate_and_glb_mime(self):
        catalogs = {"model3d": [{"id": "tripo3d/v2.5", "pricing": {
            "per_run": 0.02,
            "per_run_by_variant": {"low": 0.02, "high": 0.08},
        }}]}
        est = estimate_graph_cost({"nodes": [
            {"id": "m", "type": "model3d", "fields": {"model": "tripo3d/v2.5"}},
            {"id": "u", "type": "mupload", "fields": {}},
            {"id": "t", "type": "text", "fields": {"text": "hi"}},
        ]}, catalogs)
        self.assertAlmostEqual(est.usd, 0.02)
        self.assertFalse(est.exact)
        self.assertEqual(est.priced, 1)
        self.assertEqual(est.unpriced, 0)
        self.assertEqual(sniff_mime(b"glTFrest-of-header"), "model/gltf-binary")
        fd, path = tempfile.mkstemp(suffix=".glb")
        os.write(fd, b"glTF\x02\x00\x00\x00")
        os.close(fd)
        try:
            ref = media_from_file(path)
        finally:
            os.remove(path)
        self.assertEqual(ref.mime, "model/gltf-binary")
        self.assertTrue(ref.url.startswith("data:model/gltf-binary;base64,"))


if __name__ == "__main__":
    unittest.main()
