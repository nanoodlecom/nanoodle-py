import base64
import json
import unittest

from nanoodle import RunError, Workflow
from nanoodle.media import b64_image_mime
from nanoodle.transport import HttpResponse

SVG = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>'
SOURCE = "data:image/png;base64,iVBORw0KGgo="
OPTIONS = {"vector_operation": "convert", "mode": "default", "seed": 9,
           "guidance_scale": 3, "model": "wrong", "prompt": "injected", "size": "wrong",
           "n": 99, "response_format": "url", "imageDataUrl": "wrong", "maskDataUrl": "wrong",
           "api_key": "wrong", "Authorization": "wrong", "headers": {"Authorization": "wrong"},
           "endpoint": "https://wrong.example", "__proto__": {"polluted": True}, "constructor": {}}


class VectorImageTest(unittest.TestCase):
    def workflow(self, kind="edit", model="ideogram/vector", options=None, prompt="", source=True):
        self.calls = []

        def http(method, url, headers=None, body=None, timeout=None):
            self.calls.append((method, url, headers, json.loads(body)))
            return HttpResponse(200, {}, json.dumps({"data": [{"b64_json": base64.b64encode(
                SVG.encode()).decode()}], "cost": 0.02}).encode())

        nodes = [{"id": "out", "type": kind, "fields": {"model": model, "prompt": prompt,
                  "seed": 42, "modelOpts": options}}]
        links = []
        if kind == "edit" and source:
            nodes.insert(0, {"id": "source", "type": "upload", "fields": {"image": SOURCE}})
            links = [{"id": "l", "from": {"node": "source", "port": "image"},
                      "to": {"node": "out", "port": "image"}}]
        return Workflow.from_dict({"nodes": nodes, "links": links}, api_key="test-key", http=http)

    def test_image_and_edit_forward_knobs_but_protect_request_fields(self):
        for kind in ("image", "edit"):
            with self.subTest(kind=kind):
                result = self.workflow(kind=kind, options=OPTIONS,
                                       prompt="an icon" if kind == "image" else "").run()
                method, url, headers, body = self.calls[0]
                self.assertEqual((method, url), ("POST", "https://nano-gpt.com/v1/images/generations"))
                self.assertEqual(headers["Authorization"], "Bearer test-key")
                expected = {"model": "ideogram/vector", "size": "1024x1024", "n": 1,
                            "response_format": "b64_json", "vector_operation": "convert",
                            "mode": "default", "seed": 42, "guidance_scale": 3}
                expected.update({"prompt": "an icon"} if kind == "image" else {"imageDataUrl": SOURCE})
                self.assertEqual(body, expected)
                media = result["Image" if kind == "image" else "Edit"]
                self.assertEqual(media.mime, "image/svg+xml")
                self.assertEqual(media.suggested_extension(), "svg")
                self.assertEqual(media.bytes(), SVG.encode())
                self.assertEqual(result.cost_usd, 0.02)

    def test_blank_prompt_exception_is_convert_only(self):
        for model, opts in [("ideogram/vector", {}), ("ideogram/vector", {"vector_operation": "generate"}),
                            ("other/vector", {"vector_operation": "convert"}),
                            ("ideogram/vector-other", {"vector_operation": "convert"})]:
            with self.subTest(model=model, opts=opts):
                wf = self.workflow(model=model, options=opts, prompt=" ")
                with self.assertRaisesRegex(RunError, "no edit instruction"):
                    wf.run()
                self.assertEqual(self.calls, [])
        self.workflow(model="upscaler", options={}).run()
        with self.assertRaisesRegex(RunError, "no image input"):
            self.workflow(options=OPTIONS, source=False).run()
        self.assertEqual(self.calls, [])

    def test_non_object_options_are_ignored(self):
        for opts in (None, [], "bad"):
            self.workflow(kind="image", options=opts, prompt="an icon").run()
            self.assertNotIn("vector_operation", self.calls[0][3])

    def test_svg_sniff(self):
        self.assertEqual(b64_image_mime(base64.b64encode(b"<svg/>").decode()), "image/svg+xml")
        for prefix in ("", " \n", '<?xml version="1.0"?>\n', '\ufeff <?xml version="1.0"?>\n<!-- icon -->\n'):
            self.assertEqual(b64_image_mime(base64.b64encode((prefix + SVG).encode()).decode()), "image/svg+xml")
        for text in ("<html><svg></svg></html>", '<?xml version="1.0"?><other/>', "<svg-not-really/>"):
            self.assertEqual(b64_image_mime(base64.b64encode(text.encode()).decode()), "image/png")
        self.assertEqual(b64_image_mime("%%%invalid%%%"), "image/png")
