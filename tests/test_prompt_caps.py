"""Prompt length caps — the wired prompt nobody can shorten.

Many image/video models reject an over-long prompt at NanoGPT's route (400
prompt_too_long). In a graph the prompt is WRITTEN by an upstream LLM, so a headless
caller has nothing to shorten and no hint the limit exists. These tests pin the deal:
trim to fit, always report it, learn a cap we did not know — and never rewrite the
graph's own prompts to avoid any of it. Everything runs against the mock server.
Port of nanoodle-js tests/prompt-caps.test.mjs.
"""

import contextlib
import copy
import hashlib
import io
import unittest
import warnings

from tests._util import MockedTest
from tests.harness import chat_response, image_response

from nanoodle import (PROMPT_CAPS, NanoodleError, RunError, fit_prompt_text,
                      is_prompt_too_long, prompt_cap, prompt_cap_from_error)
from nanoodle.prompt_caps import utf16_len, with_fitted_prompt

PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
           "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

# text -> LLM -> Image(qwen-image-3, cap 800): the reported shape, minimally
GRAPH = {
    "nodes": [
        {"id": "t1", "type": "text", "fields": {"text": "Album: Neon Mirage"}},
        {"id": "m1", "type": "llm",
         "fields": {"model": "gpt-5o", "system": "You are an art director.", "prompt": ""}},
        {"id": "i1", "type": "image",
         "fields": {"model": "qwen-image-3", "prompt": "", "size": "1024x1024"}},
    ],
    "links": [
        {"id": "l1", "from": {"node": "t1", "port": "text"}, "to": {"node": "m1", "port": "prompt"}},
        {"id": "l2", "from": {"node": "m1", "port": "text"}, "to": {"node": "i1", "port": "prompt"}},
    ],
}

TOO_LONG_400 = {
    "status": 400,
    "json": {"error": "Your prompt is too long for Brand New Model. Please shorten it to "
                      "640 characters or less (current: 900 characters).",
             "code": "prompt_too_long"},
}


class CapTableTest(unittest.TestCase):
    def test_table_for_image_and_video_catalog_for_audio_never_for_chat(self):
        self.assertEqual(prompt_cap({"type": "image", "fields": {"model": "qwen-image-3"}}), 800)
        self.assertEqual(prompt_cap({"type": "edit", "fields": {"model": "step-image-edit-2"}}), 512)
        self.assertEqual(prompt_cap({"type": "tvideo", "fields": {"model": "minimax-hailuo-02"}}), 2000)
        # an uncapped (or never-probed) model stays unconstrained — the permissive-gate rule
        self.assertIsNone(prompt_cap({"type": "image", "fields": {"model": "flux-kontext"}}))
        # an LLM's own limit is tokens, and it is never the node being fitted
        self.assertIsNone(prompt_cap({"type": "llm", "fields": {"model": "gpt-5o"}}))
        # audio caps ARE catalog metadata upstream, so they are read live, never listed here
        self.assertNotIn("audio", PROMPT_CAPS)
        catalog = {"audio": [{"id": "tts-1", "supported_parameters": {"max_chars": 4096}}]}
        node = {"type": "tts", "fields": {"model": "tts-1"}}
        self.assertEqual(prompt_cap(node, catalog=catalog), 4096)
        self.assertIsNone(prompt_cap(node), "no catalog -> no cap, never a guess")
        # a learned cap beats the baked table: the table is a snapshot, a live 400 is today
        self.assertEqual(prompt_cap({"type": "image", "fields": {"model": "qwen-image-3"}},
                                    learned={"image:qwen-image-3": 640}), 640)


class FitPromptTextTest(unittest.TestCase):
    def test_boundaries_fallbacks_and_the_one_hard_cut(self):
        s = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."
        cut = fit_prompt_text(s, 30)
        self.assertLessEqual(len(cut), 30)
        self.assertTrue(cut.endswith("."))
        self.assertEqual(fit_prompt_text("short", 100), "short")   # inside the cap -> untouched
        words = fit_prompt_text("aa " * 100, 50)
        self.assertLessEqual(len(words), 50)
        self.assertFalse(words.endswith(" "), "no sentence end -> word boundary, never mid-word")
        self.assertEqual(len(fit_prompt_text("x" * 500, 100)), 100,
                         "one unbroken token -> a hard cut is the last resort")



# The cap is a count of UTF-16 code units, because that is what the model route counts and
# what nanoodle-js's String#length reports. Python's len() counts CODE POINTS, and the two
# disagree for every character outside the Basic Multilingual Plane. The table below is the
# regression: every row was produced by running the REAL nanoodle-js fitPromptText
# (nanoodle-js/src/prompt-caps.mjs) over the exact same input, so it is cross-language
# truth, not a restatement of what Python happens to do.
#
# Row: (prompt, cap, js_units, js_left_an_orphan, sha256 of the expected Python output).
#   js_units          length of the JavaScript result, in UTF-16 code units
#   js_left_an_orphan the JS cut fell between the halves of one character and JS kept the
#                     orphan half. Python drops it (see _from_units), so the expected
#                     Python output is the JS result minus that one code unit. That is the
#                     ONLY place the two may differ, and Python is on the safe side: a lone
#                     surrogate cannot be encoded as UTF-8, so the JS result there is not
#                     even sendable.
#   sha256            of the expected output, UTF-8. A literal would run to thousands of
#                     escaped characters and nobody would read it; the digest is exact.
E = "\U0001F389"             # PARTY POPPER        1 code point,  2 code units
G = "\U0001F600"             # GRINNING FACE
K = "\U00020BB7"             # CJK ext. B          astral, not emoji
F = "\U0001F1FA\U0001F1F8"   # regional indicators 2 code points, 4 code units

JS_TRUTH = [
    # the reviewer's repro: 455 code points, 905 code units
    (E * 450 + " end.", 800, 800, False,
     "5b68e79eb89dd96f0e75f8ceb8dba06d49238b31d5e604a9a6214c24ce56503a"),
    # the same prompt under a cap it really does fit
    (E * 450 + " end.", 3000, 905, False,
     "b89e4540810ebb5b063202b931af4e5aa31b33fd227cf7b719c6ceb1f10ef698"),
    # astral CJK with sentence ends
    ((K * 12 + ". ") * 40, 512, 493, False,
     "65e71801c417b63d24f5200e6d81cb22eea2bd58217d197774b2e4642b7eaff9"),
    # emoji and ascii, a sentence boundary is reachable
    (("A neon city " + E + ". ") * 60, 800, 799, False,
     "faaa16ab438f62fa39dd2c73a5da3c42b596e9043a30a37ee7c492e1671cce3d"),
    # emoji and ascii, only a word boundary is reachable
    ((E + G + " ") * 200, 800, 799, False,
     "ceb7328d20d4aed7441b42fd7d89ac435c005734639036024a64bea100a3eeef"),
    # one unbroken astral run: the hard cut
    (E * 500, 801, 801, True,
     "5b68e79eb89dd96f0e75f8ceb8dba06d49238b31d5e604a9a6214c24ce56503a"),
    # a hard cut on an ODD cap splits a surrogate pair
    (E * 500, 51, 51, True,
     "72670b1549d09a0ac7b112a6848e0bb10867186f29bc412c8efccf98f074f2c3"),
    # a hard cut on an EVEN cap lands clean
    (E * 500, 50, 50, False,
     "72670b1549d09a0ac7b112a6848e0bb10867186f29bc412c8efccf98f074f2c3"),
    # regional-indicator flags: 4 code units each
    (F * 300, 101, 101, True,
     "34a1735539c2213958796a60271d6de5a5f9c53fb56f9e6a1a76884658aceda9"),
    # BMP CJK is 1 unit, so there is no divergence to find
    ("\u4e2d" * 900 + "\u3002", 800, 800, False,
     "7d8cf682fdb924316ddc8bc91b5b15e2061a24fdf55ce1d4be055dbae3102720"),
    # BMP and astral mixed
    (("\u4e2d\u4e2d" + K + ". ") * 80, 300, 299, False,
     "50b24751a4e368e4ad758b0d822ccfb95db088c657944dba3dc3c0f928acefe7"),
    # the sentence end sits just under the 70% floor
    ("x" * 60 + ". " + E * 400, 200, 200, False,
     "3a3ca4398ef35b78154c754158af4eceb896c8bd32b7446b4d73c7ad7b67a971"),
    # trailing spaces after the cut are trimmed
    (("word " + E + "  ") * 120, 400, 394, False,
     "b24b753310c7eecb5ba845e7d0e02391eb6b46a52c9dcab930e52c2c190b3275"),
    # the cut lands right after a newline
    ((E * 10 + "\n") * 60, 300, 293, False,
     "cf59b0aea7a55d98439033e5b70ecb418394749420d1c6f63647d732c1b5d5f2"),
    # ZWNBSP at the cut: JS trims it, Python str.strip() does not
    ((E + "\ufeff") * 300, 61, 61, True,
     "c3be1f0945cbceefa3d4deb52ac884c4dc2c8850213b3501ea53d33744068907"),
    # NBSP at the cut
    ((E + "\u00a0") * 300, 61, 61, True,
     "4daeb787b2af7102abf301de48bc903c30f755f764583254b5638fa564f78c80"),
    # U+0085 at the cut: Python str.strip() takes it, JS does not
    ((E + "\u0085") * 300, 61, 61, True,
     "3684e408f7393ca49d989aa785bd45ca319029d5d337b272917d9e21a8a11a5e"),
    # U+001F at the cut: Python str.strip() takes it, JS does not
    ((E + "\u001f") * 300, 61, 61, True,
     "0d1857f66a8644adc7eecfc7603e5ae239c675015455d15bcd7532cfd221c098"),
    # ideographic space at the cut
    ((E + "\u3000") * 300, 61, 61, True,
     "b445afd3be7673254d8626101277637a3465e8e1d8c21a4abedbfeb1753f4ce2"),
    # astral text with ? and ! sentence ends
    ((E * 8 + "! " + K * 8 + "? ") * 30, 400, 395, False,
     "aabc74e64b6b581832ad4b6c64cc5c47a9eac50c7268ae1d081363e365531429"),
    # a cap of 3 against astral text
    (E * 50, 3, 3, True,
     "6146299cd54818a0e659eb6ac88e80f6f8f70536bbbd962d36973f2d2323f26c"),
    # a cap of 2 against astral text
    (E * 50, 2, 2, False,
     "6146299cd54818a0e659eb6ac88e80f6f8f70536bbbd962d36973f2d2323f26c"),
    # one astral character against a cap of 1
    (E, 1, 1, True,
     "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    # accents and astral together
    (("caf\u00e9 " + E + ". ") * 90, 333, 332, False,
     "e863a85cdda20df657e574a8d45b9ec3cc97a71f1dfbf5183a6d7bb6bff6d328"),
    # a long ascii tail after an astral head
    (E * 100 + " " + "tail " * 200, 700, 695, False,
     "2d3a358afb0aa73ff97233fec1f0c85d368d2e6aee2cd254d613250d670173bf"),
]


class Utf16ParityTest(unittest.TestCase):
    """The same graph must trim the same way in both languages.

    Measuring in code points let an over-cap prompt through untouched (the reviewer's
    repro: 455 to Python, 905 to the API) and cut other prompts to nearly twice the cap.
    """

    def test_the_reviewers_repro(self):
        s = E * 450 + " end."
        self.assertEqual(len(s), 455, "455 code points to Python")
        self.assertEqual(utf16_len(s), 905, "905 code units to the model route")
        out = fit_prompt_text(s, 800)
        self.assertNotEqual(out, s, "code-point length said 455 <= 800 and trimmed nothing")
        self.assertEqual(utf16_len(out), 800, "the cap is a code-unit count")

    def test_every_case_agrees_with_nanoodle_js(self):
        for prompt, cap, js_units, orphan, digest in JS_TRUTH:
            with self.subTest(cap=cap, units=utf16_len(prompt)):
                out = fit_prompt_text(prompt, cap)
                want_units = js_units - 1 if orphan else js_units
                self.assertEqual(hashlib.sha256(out.encode("utf-8")).hexdigest(), digest,
                                 "output differs from nanoodle-js")
                self.assertEqual(utf16_len(out), want_units)
                self.assertLessEqual(utf16_len(out), cap, "never over the cap it was given")

    def test_no_result_ever_carries_half_a_character(self):
        # a lone surrogate is not encodable as UTF-8, so it would break the very request
        # this trim exists to make sendable, and json.dumps of the trim record would raise
        for prompt, cap, _units, _orphan, _digest in JS_TRUTH:
            out = fit_prompt_text(prompt, cap)
            self.assertFalse(any(0xD800 <= ord(c) <= 0xDFFF for c in out),
                             "lone surrogate at cap %d" % cap)
            out.encode("utf-8")            # raises if a half slipped through

    def test_the_disclosed_figures_are_code_units_too(self):
        # from/to are what the caller is TOLD was trimmed. In code points they would
        # understate an emoji prompt by half and disagree with the JS report.
        node = {"type": "image", "fields": {"model": "qwen-image-3"}}
        fields = {"model": "qwen-image-3", "prompt": E * 450 + " end."}
        out, trimmed = with_fitted_prompt(node, fields)
        self.assertEqual(trimmed, {"from": 905, "to": 800, "cap": 800})
        self.assertEqual(utf16_len(out["prompt"]), 800)

    def test_a_bmp_prompt_is_measured_exactly_as_before(self):
        # the fix must move no ascii or BMP case: those are 1 code unit per code point
        self.assertEqual(fit_prompt_text("x" * 500, 100), "x" * 100)
        self.assertEqual(utf16_len("hello"), 5)
        self.assertEqual(utf16_len("中" * 900), 900)


class ErrorShapeTest(unittest.TestCase):
    def test_all_three_live_shapes_parse_and_current_is_never_the_cap(self):
        shapes = [
            ('{"error":"Your prompt is too long for Qwen Image 3. Please shorten it to 800 '
             'characters or less (current: 831 characters).","code":"prompt_too_long"}', 800),
            ('{"error":{"message":"Your prompt is too long for Step Image Edit 2. Please shorten '
             'it to 512 characters or less (current: 5999 characters).","code":"prompt_too_long"}}', 512),
            ('{"error":"Prompt is too long. Please keep it under 3000 characters."}', 3000),
        ]
        for body, want in shapes:
            self.assertTrue(is_prompt_too_long(body), body[:60])
            self.assertEqual(prompt_cap_from_error(body), want)
        # to learn the "current" length would grow the limit with every failure — the model
        # would never be prevented, only re-broken
        self.assertNotEqual(prompt_cap_from_error(shapes[1][0]), 5999)
        self.assertIsNone(prompt_cap_from_error('{"error":"Invalid image input."}'))
        self.assertFalse(is_prompt_too_long('{"error":"Invalid image input."}'))


class TrimAndDiscloseTest(MockedTest):
    def _wf(self, graph=None):
        return self.wf_dict(copy.deepcopy(graph or GRAPH))

    def _script(self, chat_text):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response(chat_text))
        self.mock.script("POST", "/v1/images/generations", image_response(b64_list=[PNG_B64]))

    def test_the_graphs_own_prompts_reach_the_model_exactly_as_written(self):
        self._script("a neon city")
        self._wf().run()
        messages = self.mock.requests_to("/api/v1/chat/completions")[0].json["messages"]
        self.assertEqual(messages[0]["content"], "You are an art director.",
                         "the system prompt is untouched — no length directive is injected")
        self.assertEqual(len(messages), 2, "and nothing extra is appended to the conversation")

    def test_a_downstream_cap_never_leaks_upstream(self):
        self._script("a neon city")
        graph = copy.deepcopy(GRAPH)
        graph["nodes"][2]["fields"]["model"] = "step-image-edit-2"   # tightest cap (512)
        self._wf(graph).run()
        chat = self.mock.requests_to("/api/v1/chat/completions")[0]
        self.assertEqual(chat.json["messages"][0]["content"], "You are an art director.")
        self.assertNotIn("512", chat.body.decode("utf-8"),
                         "the cap is applied to the image call, and nowhere else")

    def test_an_overflowing_prompt_is_fitted_at_a_sentence_boundary_and_reported(self):
        # an LLM that ignores the budget entirely (they do overshoot; this one does it wildly)
        runaway = "A neon city at night. " + "Rain slicks the street. " * 60
        self._script(runaway)
        events = []
        with self.assertWarns(RuntimeWarning):
            result = self._wf().run(on_progress=events.append)

        sent = self.mock.requests_to("/v1/images/generations")[0].json["prompt"]
        self.assertLessEqual(len(sent), 800, "sent %d characters, the cap is 800" % len(sent))
        self.assertTrue(sent.endswith("."), "the cut lands on a sentence boundary")
        self.assertGreaterEqual(len(sent), 560,
                                "a boundary cut never throws away more than the overflow did")

        trims = [e for e in events if e["type"] == "prompt-trimmed"]
        self.assertEqual(len(trims), 1, "a trimmed prompt is announced — never silent")
        self.assertEqual(trims[0]["node_id"], "i1")
        self.assertEqual(trims[0]["cap"], 800)
        self.assertEqual(trims[0]["from"], len(runaway))
        self.assertEqual(trims[0]["to"], len(sent))
        # the result carries the same disclosure for a caller that passed no on_progress
        self.assertEqual(result.prompt_trims,
                         [{"node_id": "i1", "name": "Image", "from": len(runaway),
                           "to": len(sent), "cap": 800}])

    def test_warnings_as_errors_does_not_turn_the_disclosure_into_a_failed_run(self):
        """Telling the caller about the trim must never cost more than the trim saves.

        warnings.warn is not advisory in Python the way process.emitWarning is in
        JavaScript. Under PYTHONWARNINGS=error or warnings.simplefilter("error") — the
        normal setting in a strict CI job or test suite — the warning is RAISED. Raised
        inside a node it became that node's error and escalated to RunError, so a run that
        would have succeeded died of its own disclosure. This is the regression.
        """
        runaway = "A neon city at night. " + "Rain slicks the street. " * 60
        self._script(runaway)
        events = []
        err = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # every warning is now an exception
            with contextlib.redirect_stderr(err):
                result = self._wf().run(on_progress=events.append)   # must NOT raise

        self.assertEqual(result.nodes["i1"].status, "done", "the run completed")
        self.assertIsNotNone(result.outputs.get("Image"))
        # and the trim is still disclosed, by all three channels
        self.assertEqual(len(result.prompt_trims), 1)
        self.assertEqual([e["type"] for e in events if e["type"] == "prompt-trimmed"],
                         ["prompt-trimmed"])
        self.assertIn("prompt trimmed", err.getvalue(),
                      "the warning could not be raised, so the same sentence went to stderr")
        self.assertIn("qwen-image-3", err.getvalue())

    def test_a_prompt_that_fits_is_sent_verbatim_and_nothing_is_announced(self):
        self._script("a neon city at 3am")
        events = []
        result = self._wf().run(on_progress=events.append)
        self.assertEqual(self.mock.requests_to("/v1/images/generations")[0].json["prompt"],
                         "a neon city at 3am")
        self.assertEqual([e for e in events if e["type"] == "prompt-trimmed"], [])
        self.assertEqual(result.prompt_trims, [])


class LearnedCapTest(MockedTest):
    def test_an_unknown_models_cap_is_learned_from_the_live_400(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("a neon city"))
        # a model NOT in the table — exactly the case the table cannot cover
        self.mock.script("POST", "/v1/images/generations", TOO_LONG_400)
        graph = copy.deepcopy(GRAPH)
        graph["nodes"][2]["fields"]["model"] = "brand-new-model"
        wf = self.wf_dict(graph)

        with self.assertRaises(RunError) as ctx:
            wf.run()
        self.assertIn("running again should succeed", str(ctx.exception.result.errors[0]["message"]))
        self.assertIn("640 characters or less", ctx.exception.result.errors[0]["message"],
                      "without hiding what the API actually said")
        self.assertEqual(wf._prompt_caps.get("image:brand-new-model"), 640,
                         "the cap is banked for the retry")

        # second run: the same Workflow now trims to the learned 640 before sending
        self.mock.reset()
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("A neon city. " * 80))
        self.mock.script("POST", "/v1/images/generations", image_response(b64_list=[PNG_B64]))
        events = []
        wf.run(on_progress=events.append)
        sent = self.mock.requests_to("/v1/images/generations")[0].json["prompt"]
        self.assertLessEqual(len(sent), 640, "the learned cap is applied, not rediscovered")
        self.assertEqual([e["cap"] for e in events if e["type"] == "prompt-trimmed"], [640])

    def test_an_unrelated_400_is_relayed_untouched(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("a neon city"))
        self.mock.script("POST", "/v1/images/generations",
                         {"status": 400, "json": {"error": "Invalid image input."}})
        wf = self.wf_dict(copy.deepcopy(GRAPH))
        with self.assertRaises(NanoodleError) as ctx:
            wf.run()
        self.assertNotIn("running again should succeed", str(ctx.exception))
        self.assertEqual(wf._prompt_caps, {})

    def test_a_chat_node_is_never_promised_a_fix_this_library_cannot_deliver(self):
        """"running again should succeed" is a promise, and for llm/vision it is unkeepable.

        prompt_cap returns None for CAP_KIND "chat" before it ever reads the learned table,
        because an LLM's own limit is tokens and that node is never the one being fitted.
        So a cap banked under a chat key can never be applied. Promising the caller a fix
        would send them into a PAID re-run that fails in exactly the same way. Relay what
        the API said and nothing more.
        """
        self.mock.script("POST", "/api/v1/chat/completions", {
            "status": 400,
            "json": {"error": "Your prompt is too long for GPT 5o. Please shorten it to "
                              "4000 characters or less.", "code": "prompt_too_long"}})
        wf = self.wf_dict(copy.deepcopy(GRAPH))
        with self.assertRaises(RunError) as ctx:
            wf.run()
        message = ctx.exception.result.errors[0]["message"]
        self.assertIn("4000 characters or less", message, "the API's own words are relayed")
        self.assertNotIn("running again should succeed", message,
                         "nothing about this run would change on a retry")
        self.assertEqual(wf._prompt_caps, {},
                         "a chat cap is unreadable by prompt_cap, so it is not banked")


if __name__ == "__main__":
    unittest.main()
