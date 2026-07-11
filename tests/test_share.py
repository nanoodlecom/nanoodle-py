"""Share-link codec: the golden fixtures in tests/fixtures/share/ are minted by
the REAL editor's encoder and shared byte-identically with nanoodle-js, so this
mirrors nanoodle-js/tests/share-decode.test.mjs. Everything runs offline — the
redirect follower takes an injectable opener that never touches the network."""

import base64
import glob
import json
import os
import unittest
from urllib.parse import urljoin

from tests import fixture

from nanoodle import (NanoodleError, Workflow, decode_share_fragment,
                      decode_share_url, is_share_ref)

SHARE_DIR = fixture("share")
GOLDENS = [json.load(open(p, encoding="utf-8"))
           for p in sorted(glob.glob(os.path.join(SHARE_DIR, "*.json")))]

# fixture app keys are the JS camelCase; the Python codec exposes snake_case.
_APP_KEY = {"hasFiles": "has_files"}


def golden(name):
    return next(g for g in GOLDENS if g["name"] == name)


def _fragment_of(url):
    return url[url.index("#"):]


class GoldenTest(unittest.TestCase):
    def test_goldens_exist_for_every_wire_format(self):
        kinds = sorted({g["name"].split("-")[0] for g in GOLDENS})
        self.assertEqual(kinds, ["a", "g", "j"])
        self.assertGreaterEqual(len(GOLDENS), 6)

    def test_editor_minted_url_decodes_to_editor_graph(self):
        for g in GOLDENS:
            with self.subTest(golden=g["name"]):
                r = decode_share_url(g["url"])
                self.assertEqual(r["graph"], g["graph"])
                for k, v in (g.get("app") or {}).items():
                    self.assertEqual(r["app"][_APP_KEY.get(k, k)], v, "app.%s" % k)

    def test_bare_fragment_and_tail_forms_decode(self):
        for g in GOLDENS:
            with self.subTest(golden=g["name"]):
                frag = _fragment_of(g["url"])
                self.assertEqual(decode_share_fragment(frag)["graph"], g["graph"])       # "#g=…"
                self.assertEqual(decode_share_fragment(frag[1:])["graph"], g["graph"])    # "g=…"


class WorkflowLoadTest(unittest.TestCase):
    def test_load_accepts_share_url_offline_and_derives_inputs(self):
        g = golden("g-starter")
        wf = Workflow.load(g["url"], api_key="unused")
        self.assertGreaterEqual(len(wf.inputs), 1)
        self.assertEqual(len(wf.graph.nodes), len(g["graph"]["nodes"]))

    def test_load_still_loads_plain_file_paths(self):
        wf = Workflow.load(fixture("starter-graph.json"), api_key="unused")
        self.assertGreater(len(wf.graph.nodes), 0)

    def test_load_app_link_yields_graph_only(self):
        g = golden("a-files")
        wf = Workflow.load(g["url"], api_key="unused")
        self.assertGreater(len(wf.graph.nodes), 0)


class IsShareRefTest(unittest.TestCase):
    def test_urls_and_fragments_yes_file_paths_no(self):
        for ok in ("https://nanoodle.com/#g=abc", "http://localhost:8080/play.html#a=abc",
                   "#g=abc", "g=abc", "#j=abc", "a=abc"):
            self.assertTrue(is_share_ref(ok), ok)
        for no in ("noodle-graph.json", "./out/graph.json", "/tmp/g=weird/graph.json"):
            self.assertFalse(is_share_ref(no), no)


class RefusalTest(unittest.TestCase):
    def test_ga_handoff_fragments_refused_with_guidance(self):
        with self.assertRaisesRegex(NanoodleError, "handoff.*internal|internal.*handoff"):
            decode_share_fragment("#ga=H4sIAAAA")

    def test_a_payload_without_graph_refused(self):
        raw = json.dumps({"v": 1, "name": "no graph here"}).encode("utf-8")
        b64u = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        with self.assertRaisesRegex(NanoodleError, "no graph"):
            decode_share_fragment("#a=u" + b64u)

    def test_corrupt_payloads_raise_nanoodle_error(self):
        frag = _fragment_of(golden("g-starter")["url"])
        with self.assertRaises(NanoodleError):
            decode_share_fragment(frag[:40])               # truncated gzip
        with self.assertRaises(NanoodleError):
            decode_share_fragment("#g=!!not-base64!!")     # bad alphabet
        with self.assertRaises(NanoodleError):
            decode_share_fragment("#z=abcd")               # unknown tag
        with self.assertRaises(NanoodleError):
            decode_share_url("#g=")                         # empty payload


# ---- short links: fragments ride the Location header, so redirects are followed by hand ----

def _redirect(loc):
    return (302, loc)


def _opener_over(routes):
    def opener(url):
        if url not in routes:
            raise AssertionError("unexpected fetch: " + url)
        return routes[url]
    return opener


class ShortLinkTest(unittest.TestCase):
    def test_redirect_chain_with_relative_location_decodes(self):
        g = golden("g-starter")
        frag = _fragment_of(g["url"])
        opener = _opener_over({
            "https://da.gd/abc": _redirect("https://hop.example/x"),
            "https://hop.example/x": _redirect("/final" + frag),
        })
        r = decode_share_url("https://da.gd/abc", opener=opener)
        self.assertEqual(r["graph"], g["graph"])
        self.assertTrue(r["url"].startswith("https://hop.example/final#"))

    def test_direct_fragment_urls_never_fetch(self):
        g = golden("g-unicode")

        def tripwire(url):
            raise AssertionError("network touched for a direct link")

        r = decode_share_url(g["url"], opener=tripwire)
        self.assertEqual(r["graph"], g["graph"])

    def test_redirect_without_fragment_ends_with_helpful_error(self):
        opener = _opener_over({"https://short.example/x": (200, None)})
        with self.assertRaisesRegex(NanoodleError, "no redirect|share the long"):
            decode_share_url("https://short.example/x", opener=opener)

    def test_redirect_loops_are_capped(self):
        opener = _opener_over({
            "https://a.example/": _redirect("https://b.example/"),
            "https://b.example/": _redirect("https://a.example/"),
        })
        with self.assertRaisesRegex(NanoodleError, "gave up after"):
            decode_share_url("https://a.example/", opener=opener)


class UnicodeTest(unittest.TestCase):
    def test_unicode_survives_round_trip_byte_for_byte(self):
        r = decode_share_url(golden("g-unicode")["url"])
        texts = [n.get("fields", {}).get("text")
                 for n in r["graph"]["nodes"] if n.get("type") == "text"]
        self.assertTrue(any(isinstance(t, str) and "ラーメン🍜" in t for t in texts))


if __name__ == "__main__":
    unittest.main()
