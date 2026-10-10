"""Run-cost forecast. Untouched video audio follows the catalog default (nanoodle #712).

The JavaScript library's estimator prices a missing switch as off. This one
does not: NanoGPT bills the catalog default when the key is omitted.
"""

import unittest

from nanoodle import Estimate, Workflow, estimate_graph_cost
from nanoodle.estimate import (apply_video_quote_pricing, audio_unit_usd,
                               video_audio_on, video_price_fields, video_unit_usd)

DUR = {"type": "select", "options": []}
ON = {"params": {"duration": DUR, "generateAudio": {"type": "switch", "default": True,
                                                    "label": "Generate Audio"}},
      "defaults": {"generateAudio": True}}
OFF = {"params": {"duration": DUR, "generateAudio": {"type": "switch", "default": False,
                                                     "label": "Generate Audio"}},
       "defaults": {"generateAudio": False}}
SNAKE = {"params": {"duration": DUR, "generate_audio": {"type": "switch", "default": True}},
         "defaults": {"generate_audio": True}}

PRO_CATALOG = {"currency": "USD", "raw": {
    "type": "resolution-per-second-audio-toggle", "defaultDuration": 5, "defaultResolution": "720p",
    "withoutAudioPricesPerSecond": {"480p": 0.007058823529411765, "720p": 0.015294117647058823,
                                     "1080p": 0.030588235294117645},
    "withAudioPricesPerSecond": {"480p": 0.01411764705882353, "720p": 0.030588235294117645,
                                  "1080p": 0.06117647058823529}}}
FAST_CATALOG = {"currency": "USD", "raw": {
    "type": "resolution-per-second-audio-toggle", "defaultDuration": 5, "defaultResolution": "720p",
    "withoutAudioPricesPerSecond": {"720p": 0.011764705882352941, "1080p": 0.01764705882352941},
    "withAudioPricesPerSecond": {"720p": 0.023529411764705882, "1080p": 0.03529411764705882}}}
VEO = {"currency": "USD", "with_audio": 4.8, "without_audio": 3.2, "fixed_duration_seconds": 8}
VEO_FAST = {"currency": "USD", "with_audio": 1.6, "without_audio": 1.2, "fixed_duration_seconds": 8}
VEO31_FAST = {"currency": "USD", "text_image_with_audio_per_second": 0.15,
              "text_image_without_audio_per_second": 0.1,
              "default_duration": 8, "default_resolution": "720p"}
PIX = {"currency": "USD", "raw": {
    "type": "resolution-per-second-audio-toggle", "defaultDuration": 5, "defaultResolution": "720p",
    "withoutAudioPricesPerSecond": {"720p": 0.029411764705882356},
    "withAudioPricesPerSecond": {"720p": 0.03823529411764706}}}


def _usd(pricing, fields, ctx=None):
    params = (ctx or {}).get("params") or {"duration": DUR}
    return video_unit_usd(pricing, video_price_fields(params, fields), 0, False, None, ctx)


class AudioSwitchTest(unittest.TestCase):
    def test_untouched_switch_uses_the_catalog_default(self):
        pro = apply_video_quote_pricing("bytedance-seedance-v1.5-pro", PRO_CATALOG)
        fast = apply_video_quote_pricing("bytedance-seedance-v1.5-pro-fast", FAST_CATALOG)
        bare = {"duration": "5", "resolution": "720p", "modelOpts": {}}
        cases = [
            (_usd(pro, bare, ON), 0.26, "seedance pro untouched"),
            (_usd(pro, dict(bare, modelOpts={"generateAudio": False}), ON), 0.13, "seedance pro off"),
            (_usd(pro, dict(bare, modelOpts={"generateAudio": True}), ON), 0.26, "seedance pro on"),
            (_usd(pro, bare), 0.13, "seedance pro no descriptor"),
            (_usd(fast, bare, ON), 0.20, "seedance fast untouched"),
            (_usd(fast, dict(bare, modelOpts={"generateAudio": False}), ON), 0.10, "seedance fast off"),
            (_usd(VEO, {"modelOpts": {}}, ON), 4.8, "veo3 untouched"),
            (_usd(VEO, {"modelOpts": {"generateAudio": False}}, ON), 3.2, "veo3 off"),
            (_usd(VEO_FAST, {"modelOpts": {}}, ON), 1.6, "veo3 fast untouched"),
            (_usd(VEO31_FAST, {"duration": "8", "modelOpts": {}}, ON), 1.2, "veo 3.1 fast untouched"),
            (_usd(VEO31_FAST, {"duration": "8", "modelOpts": {"generateAudio": False}}, ON), 0.8,
             "veo 3.1 fast off"),
            (_usd(PIX, bare, OFF), 0.14705882352941178, "pixverse untouched"),
            (_usd(PIX, dict(bare, modelOpts={"generateAudio": True}), OFF), 0.1911764705882353,
             "pixverse on"),
            (_usd({"currency": "USD", "per_duration": {"5": 0.35}, "audio_multiplier": 2},
                  {"duration": "5", "modelOpts": {}}, SNAKE), 0.7, "multiplier default on"),
            (_usd({"currency": "USD", "per_duration": {"5": 0.35}, "audio_multiplier": 2},
                  {"duration": "5", "modelOpts": {}}), 0.35, "multiplier no descriptor"),
        ]
        for got, want, label in cases:
            self.assertIsNotNone(got, label)
            self.assertLess(abs(got - want), 1e-9, "%s got %r want %r" % (label, got, want))
        self.assertIs(video_audio_on({"modelOpts": {}}, ON), True)
        self.assertIs(video_audio_on({"modelOpts": {"generateAudio": False}}, ON), False)
        self.assertIs(video_audio_on({"modelOpts": {}}, OFF), False)

    def test_graph_estimate_prices_an_omitted_seedance_switch_at_the_quote(self):
        catalogs = {"video": [{
            "id": "bytedance-seedance-v1.5-pro",
            "pricing": PRO_CATALOG,
            "supported_parameters": {"parameters": ON["params"], "defaults": ON["defaults"]},
        }]}
        node = {"id": "v", "type": "tvideo", "fields": {
            "model": "bytedance-seedance-v1.5-pro", "prompt": "rain",
            "duration": "5", "resolution": "720p", "modelOpts": {}}}
        est = estimate_graph_cost({"nodes": [node], "links": []}, catalogs)
        self.assertLess(abs(est.usd - 0.26), 1e-9)
        self.assertFalse(est.exact)
        self.assertEqual(est.priced, 1)
        node["fields"] = dict(node["fields"], modelOpts={"generateAudio": False})
        off = estimate_graph_cost({"nodes": [node], "links": []}, catalogs)
        self.assertLess(abs(off.usd - 0.13), 1e-9)
        wf = Workflow.from_dict({"nodes": [node]}, api_key="unused", catalog=catalogs)
        self.assertLess(abs(wf.estimate().usd - 0.13), 1e-9)


class OtherEstimateTest(unittest.TestCase):
    def test_local_nodes_are_free_and_decide_pick_is_asked_twice(self):
        local = estimate_graph_cost({"nodes": [
            {"id": "t", "type": "text", "fields": {"text": "hi"}},
            {"id": "j", "type": "join", "fields": {"sep": " "}},
            {"id": "n", "type": "comment", "fields": {"text": "note"}},
            {"id": "e", "type": "endpoint", "fields": {"url": "http://127.0.0.1:9", "mode": "chat"}},
        ], "links": [
            {"id": "l", "from": {"node": "t", "port": "text"}, "to": {"node": "j", "port": "a"}},
        ]})
        self.assertEqual(local.usd, 0.0)
        self.assertTrue(local.exact)
        self.assertEqual(local.priced, 0)
        self.assertEqual(local.unpriced, 0)
        self.assertIsInstance(local, Estimate)

        pricing = {"prompt": 2.0, "completion": 8.0}
        catalogs = {"chat": [{"id": "decider", "pricing": pricing}]}
        fields = {"model": "decider", "mode": "pick", "question": "which?", "options": "", "levels": ""}
        pick = estimate_graph_cost({"nodes": [
            {"id": "d", "type": "decide", "fields": fields},
        ]}, catalogs)
        yes = estimate_graph_cost({"nodes": [
            {"id": "d", "type": "decide", "fields": dict(fields, mode="yesno")},
        ]}, catalogs)
        self.assertGreater(pick.usd, 0)
        self.assertAlmostEqual(pick.usd, yes.usd * 2)

    def test_image_is_exact_and_bool_is_not_a_price(self):
        catalogs = {"image": [{"id": "img", "pricing": {"per_image": {"1024x1024": 0.04}},
                               "supported_parameters": {"max_output_images": 4}}]}
        est = estimate_graph_cost({"nodes": [
            {"id": "i", "type": "image", "fields": {"model": "img", "size": "1024x1024", "variations": "2"}},
        ]}, catalogs)
        self.assertAlmostEqual(est.usd, 0.08)
        self.assertTrue(est.exact)
        # a boolean must not be read as 1.0 dollars per second
        self.assertIsNone(audio_unit_usd({"per_second": True}))


if __name__ == "__main__":
    unittest.main()
