"""The seam between two features that both write to a RunResult and to a node error.

Prompt caps (PR #11) add ``result.prompt_trims`` and, on a rejection this library can
prevent next time, a sentence to the node's error. x402 payment tracing (PR #12) adds
``result.payments`` and, on a deposit that never bought its request, ANOTHER sentence to
the same node's error. Neither feature knew about the other, so these tests pin the
combination: one run, one node, both reports, and nothing overwritten.

A paid node is also the node most worth trimming — a request certain to 400 costs real
XNO when it is paid per call — so this is a real combination, not a contrived one.
All offline, against the mock NanoGPT harness.
"""

import contextlib
import copy
import io
import json
import time
import unittest

from tests import fixture
from tests._util import MockedTest
from tests.harness import image_response

from nanoodle import RunError, Workflow
from nanoodle.__main__ import main

PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
           "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")

IMAGES = "/v1/images/generations"
PAY_ID = "pay_66304af72ff034ab673234150b5997ec"
COMPLETE = "/api/x402/complete/" + PAY_ID

# qwen-image-3's cap is 800; this overshoots it about twofold, the way a real LLM
# that ignores a length budget does.
RUNAWAY = "A neon city at night. " + "Rain slicks the street. " * 60

# The 400 a DIFFERENT limit answers with, so the run also has a cap to learn. This is
# the message prompt_caps turns into "running again should succeed".
TOO_LONG_400 = {
    "status": 400,
    "json": {"error": "Your prompt is too long for Qwen Image 3. Please shorten it to "
                      "640 characters or less (current: 800 characters).",
             "code": "prompt_too_long"},
}


def mock_402():
    """The captured 402 fixture, pointed at the mock server and given a live window.

    The fixture's statusUrl/completeUrl are absolute to nano-gpt.com; dropping them
    makes parse_nano_invoice derive "/api/x402/{status,complete}/<id>", which resolves
    against the base_url under test. Its expiresAt is long past, so refresh it too.
    """
    with open(fixture("x402/402.json"), "r", encoding="utf-8") as f:
        body = json.load(f)
    secs = int(time.time()) + 15 * 60
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(secs))
    for option in body.get("accepts", []):
        option["expiresAt"] = secs
        option.pop("statusUrl", None)
        option.pop("completeUrl", None)
    payment = body.get("payment") or {}
    payment["expiresAt"] = stamp
    payment.pop("statusUrl", None)
    payment.pop("completeUrl", None)
    for option in payment.get("accepted", []):
        option["expiresAt"] = stamp
        option.pop("statusUrl", None)
        option.pop("completeUrl", None)
    return {"status": 402, "json": body}


class TrimAndPaymentTest(MockedTest):
    def _wf(self):
        with open(fixture("trim-and-pay.json"), "r", encoding="utf-8") as f:
            graph = json.load(f)
        # keyless + a payment callback = accountless x402. api_key="" must stay
        # keyless even if NANOGPT_API_KEY is set in the environment.
        self.sent = []
        return Workflow.from_dict(copy.deepcopy(graph), api_key="",
                                  base_url=self.mock.base_url,
                                  payment=self.sent.append,
                                  poll_intervals={"x402": 0.001})

    def test_a_paid_node_reports_its_trim_and_its_deposit_in_one_result(self):
        # 402 -> callback sends -> complete replays the image result (redeems the deposit)
        self.mock.script("POST", IMAGES, mock_402())
        self.mock.script("POST", COMPLETE, image_response(b64_list=[PNG_B64]))

        with self.assertWarns(RuntimeWarning):
            result = self._wf().run({"Idea": RUNAWAY})

        # the run succeeded, so both reports come off a HEALTHY result — this is not
        # only an error-path feature
        self.assertEqual(result.nodes["n2"].status, "done")

        sent_prompt = self.mock.requests_to(IMAGES)[0].json["prompt"]
        self.assertLessEqual(len(sent_prompt), 800)
        self.assertEqual(result.prompt_trims,
                         [{"node_id": "n2", "name": "Poster", "from": len(RUNAWAY),
                           "to": len(sent_prompt), "cap": 800}])

        self.assertEqual([p["payment_id"] for p in result.payments], [PAY_ID])
        self.assertEqual(result.payments[0]["node_id"], "n2")
        self.assertEqual(result.payments[0]["status"], "sent")
        self.assertTrue(result.payments[0]["redeemed"],
                        "the deposit bought its request, so it is redeemed")
        self.assertEqual([inv["paymentId"] for inv in self.sent], [PAY_ID])

    def test_the_cap_note_and_the_deposit_sentence_share_one_error_message(self):
        """The collision itself: two features, two sentences, one node.error string.

        The paid request comes back 400 prompt_too_long, so prompt_caps appends "running
        again should succeed" AND the XNO is gone unredeemed, so the payment sweep appends
        the deposit sentence. Either write could have replaced the other.
        """
        self.mock.script("POST", IMAGES, [mock_402(), TOO_LONG_400])
        # complete confirms the deposit but carries no result, so the original request
        # is re-sent with the settled payment id — and THAT is what 400s.
        self.mock.script("POST", COMPLETE, {"status": 200, "json": {"status": "completed"}})

        wf = self._wf()
        with self.assertWarns(RuntimeWarning):
            with self.assertRaises(RunError) as ctx:
                wf.run({"Idea": RUNAWAY})
        result = ctx.exception.result

        message = result.nodes["n2"].error
        # #11's sentence
        self.assertIn("640 characters or less", message)
        self.assertIn("running again should succeed", message)
        # #12's sentence, on the same string, with the money facts intact
        self.assertIn("a Nano deposit was already sent", message)
        self.assertIn(PAY_ID, message)
        # order, so neither write is silently dropped by the other: the API's own words,
        # then the cap note, then the deposit — each joined with " — "
        self.assertLess(message.index("640 characters or less"),
                        message.index("running again should succeed"))
        self.assertLess(message.index("running again should succeed"),
                        message.index("a Nano deposit was already sent"))
        self.assertIn(" — a Nano deposit was already sent", message)
        # the same message reaches result.errors, which is what a caller reads
        self.assertEqual([e["node_id"] for e in result.errors], ["n2"])
        self.assertEqual(result.errors[0]["message"], message)

        # and both structured reports survive the shared error message
        self.assertEqual([t["node_id"] for t in result.prompt_trims], ["n2"])
        self.assertEqual([p["payment_id"] for p in result.payments], [PAY_ID])
        self.assertFalse(result.payments[0]["redeemed"],
                         "the 400 gave nothing back for the XNO")
        # the cap was banked, so the promise the message makes is keepable
        self.assertEqual(wf._prompt_caps.get("image:qwen-image-3"), 640)


class TrimAndPaymentCliTest(MockedTest):
    """--json is the whole interface for an agent caller, so it must carry both."""

    def _run_cli(self, *extra):
        argv = ["run", fixture("trim-and-pay.json"), "--pay",
                "--base-url", self.mock.base_url, "--json",
                "--input", "Idea=" + RUNAWAY] + list(extra)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_json_reports_the_trim_and_the_payment_together(self):
        self.mock.script("POST", IMAGES, mock_402())
        self.mock.script("POST", COMPLETE, image_response(b64_list=[PNG_B64]))

        code, out, err = self._run_cli()
        self.assertEqual(code, 0, err)
        payload = json.loads(out)

        self.assertEqual([t["node_id"] for t in payload["promptTrims"]], ["n2"])
        self.assertEqual(payload["promptTrims"][0]["cap"], 800)
        self.assertEqual([p["payment_id"] for p in payload["payments"]], [PAY_ID])
        self.assertEqual(payload["payments"][0]["status"], "sent")
        self.assertTrue(payload["payments"][0]["redeemed"])
        # the invoice still reaches the human on stderr, and stdout stays pure JSON
        self.assertIn("payment required", err)

    def test_a_keyed_run_reports_an_empty_payments_list(self):
        """The field is always present, so a caller can read it without a guard."""
        self.mock.script("POST", IMAGES, image_response(b64_list=[PNG_B64]))
        argv = ["run", fixture("trim-and-pay.json"), "--api-key", "cli-key",
                "--base-url", self.mock.base_url, "--json", "--input", "Idea=a poster"]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        self.assertEqual(code, 0, err.getvalue())
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["payments"], [])
        self.assertEqual(payload["promptTrims"], [])


if __name__ == "__main__":
    unittest.main()
