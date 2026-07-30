"""x402 accountless payments: invoice parsing pinned to a REAL captured 402
(tests/fixtures/x402/402.json, byte-identical to nanoodle-js's copy) plus the
full settle flow against an injectable transport. Mirrors nanoodle-js
tests/x402.test.mjs behavior for behavior."""

import json
import threading
import time
import unittest

from tests import fixture

from nanoodle import NanoodleError, RunError, Workflow
from nanoodle.engine import Engine
from nanoodle.transport import HttpResponse
from nanoodle.x402 import assert_payment_option, looks_like_result, parse_nano_invoice

with open(fixture("x402/402.json"), "r", encoding="utf-8") as f:
    FIXTURE_402 = json.load(f)

BASE = "https://nano-gpt.com"
CHAT_OK = {"choices": [{"message": {"content": "paid hello"}}], "cost": 0.0001}


def fresh_402(minutes=15):
    """The fixture's real expiresAt is ~15 min after capture — long dead by test
    time. Settle-flow tests need a live window; parser tests use the raw fixture."""
    j = json.loads(json.dumps(FIXTURE_402))
    secs = int(time.time()) + minutes * 60
    for a in j.get("accepts", []):
        a["expiresAt"] = secs
    for a in (j.get("payment") or {}).get("accepted", []):
        a["expiresAt"] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(secs))
    if j.get("payment"):
        j["payment"]["expiresAt"] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(secs))
    return j


def json_resp(status, body):
    return HttpResponse(status, {"Content-Type": "application/json"},
                        json.dumps(body).encode("utf-8"))


def make_engine(http, payment, api_key=None):
    return Engine(api_key, BASE, http, poll_intervals={"x402": 0.001}, payment=payment)


class ParseInvoiceTest(unittest.TestCase):
    def test_real_fixture(self):
        inv = parse_nano_invoice(FIXTURE_402, BASE)
        self.assertEqual(inv["scheme"], "nano")
        self.assertRegex(inv["paymentId"], r"^pay_[0-9a-f]+$")
        self.assertRegex(inv["payTo"], r"^nano_[a-z0-9]+$")
        self.assertRegex(inv["amountRaw"], r"^\d+$")  # integer raw units
        self.assertTrue(inv["amount"].endswith("XNO"))
        self.assertGreater(inv["amountUsd"], 0)
        self.assertEqual(inv["uri"], "nano:%s?amount=%s" % (inv["payTo"], inv["amountRaw"]))
        self.assertTrue(inv["statusUrl"].startswith(BASE + "/api/x402/status/pay_"))
        self.assertTrue(inv["completeUrl"].startswith(BASE + "/api/x402/complete/pay_"))
        self.assertGreater(inv["expiresAt"], 1e12)  # epoch ms
        self.assertTrue(inv["explorerUrl"].startswith("https://"))

    def test_parity_with_js_invoice_fields(self):
        # same fixture, same field names as nanoodle-js's parseNanoInvoice — the
        # invoice a payment callback sees must be interchangeable between libs
        inv = parse_nano_invoice(FIXTURE_402, BASE)
        self.assertEqual(
            sorted(inv.keys()),
            sorted(["scheme", "paymentId", "payTo", "amountRaw", "amount", "amountUsd",
                    "uri", "expiresAt", "statusUrl", "completeUrl", "explorerUrl",
                    "description", "requestHash"]))

    def test_no_nano_option_returns_none(self):
        stripped = json.loads(json.dumps(FIXTURE_402))
        stripped["accepts"] = [a for a in stripped["accepts"] if a["scheme"] != "nano"]
        stripped["payment"]["accepted"] = []
        self.assertIsNone(parse_nano_invoice(stripped, BASE))

    def test_looks_like_result(self):
        self.assertTrue(looks_like_result(CHAT_OK))
        self.assertTrue(looks_like_result({"data": [{"b64_json": "x"}]}))
        self.assertFalse(looks_like_result({"status": "completed", "paymentId": "pay_x"}))
        self.assertFalse(looks_like_result(None))


class SettleFlowTest(unittest.TestCase):
    def chat(self, engine):
        resp = engine._post_json("/api/v1/chat/completions",
                                 {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        return json.loads(resp.text())

    def test_replayed_result_no_repost(self):
        calls = []
        invoices = []

        def http(method, url, headers=None, body=None, timeout=None):
            calls.append((method, url, headers))
            if "/chat/completions" in url:
                return json_resp(402, fresh_402())
            if "/api/x402/complete/" in url:
                n = sum(1 for c in calls if "/complete/" in c[1])
                if n == 1:
                    return json_resp(402, {"error": "Payment not verified", "status": "pending"})
                return json_resp(200, CHAT_OK)
            raise AssertionError("unexpected url " + url)

        eng = make_engine(http, invoices.append)
        self.assertEqual(self.chat(eng), CHAT_OK)
        self.assertEqual(len(invoices), 1, "exactly one payment per request")
        self.assertEqual(invoices[0]["payTo"], parse_nano_invoice(FIXTURE_402, BASE)["payTo"])
        first_headers = calls[0][2]
        self.assertEqual(first_headers.get("x-x402"), "true")
        self.assertNotIn("Authorization", first_headers, "keyless request carries no Authorization")
        self.assertEqual(sum(1 for c in calls if "/chat/completions" in c[1]), 1,
                         "result came from complete, not a re-POST")

    def test_settle_only_reposts_with_payment_id(self):
        calls = []

        def http(method, url, headers=None, body=None, timeout=None):
            calls.append((method, url, headers))
            if "/chat/completions" in url:
                if headers.get("x-x402-payment-id"):
                    return json_resp(200, CHAT_OK)
                return json_resp(402, fresh_402())
            if "/api/x402/complete/" in url:
                return json_resp(200, {"status": "completed", "paymentId": "pay_x"})
            raise AssertionError("unexpected url " + url)

        eng = make_engine(http, lambda inv: None)
        self.assertEqual(self.chat(eng), CHAT_OK)
        reposts = [c for c in calls if "/chat/completions" in c[1] and c[2].get("x-x402-payment-id")]
        self.assertEqual(len(reposts), 1)
        self.assertRegex(reposts[0][2]["x-x402-payment-id"], r"^pay_")

    def test_second_402_after_settle_is_hard_error(self):
        paid = []

        def http(method, url, headers=None, body=None, timeout=None):
            if "/chat/completions" in url:
                return json_resp(402, fresh_402())
            return json_resp(200, {"status": "completed"})

        eng = make_engine(http, lambda inv: paid.append(1))
        with self.assertRaisesRegex(NanoodleError, "still answered 402"):
            self.chat(eng)
        self.assertEqual(len(paid), 1, "never a second payment")

    def test_expired_window(self):
        expired = json.loads(json.dumps(FIXTURE_402))
        for a in expired["accepts"]:
            a["expiresAt"] = int(time.time()) - 60
        expired["payment"]["expiresAt"] = "2020-01-01T00:00:00.000Z"

        def http(method, url, headers=None, body=None, timeout=None):
            if "/chat/completions" in url:
                return json_resp(402, expired)
            return json_resp(402, {"error": "Payment not verified"})

        eng = make_engine(http, lambda inv: None)
        with self.assertRaisesRegex(NanoodleError, "expired.*nano_"):
            self.chat(eng)

    def test_run_deadline_stops_the_settle_poll_and_stays_traceable(self):
        # The payment window is 15 minutes. Without a deadline check this loop
        # polls for all 15, and the non-daemon worker thread holds interpreter
        # shutdown for the same 15 minutes.
        polls = []

        def http(method, url, headers=None, body=None, timeout=None):
            if "/chat/completions" in url:
                return json_resp(402, fresh_402())
            polls.append(url)
            return json_resp(402, {"error": "Payment not verified", "status": "pending"})

        eng = make_engine(http, lambda inv: None)
        eng.set_run_deadline(time.monotonic() + 0.2, 0.2)
        t0 = time.monotonic()
        with self.assertRaises(NanoodleError) as ctx:
            self.chat(eng)
        self.assertLess(time.monotonic() - t0, 5.0)
        # the deposit may already be on-chain: the message must name it
        msg = str(ctx.exception)
        self.assertIn("pay_", msg)
        self.assertIn("nano_", msg)
        self.assertGreater(len(polls), 0, "it did poll before giving up")
        # and the caller can read the deposit back off the engine, because the
        # worker that sent it may be abandoned before anyone sees its exception
        owed = eng.unredeemed_payments()
        self.assertEqual(len(owed), 1)
        self.assertEqual(owed[0]["payment_id"], parse_nano_invoice(FIXTURE_402, BASE)["paymentId"])
        self.assertIn("pay_", owed[0]["trace"])

    def test_a_settled_payment_always_gets_the_request_it_paid_for(self):
        # MONEY: the callback has sent XNO and the complete endpoint confirmed
        # the deposit. If the run deadline passes in that window, the retry must
        # STILL go out. Clamping it to the deadline meant the user paid and no
        # request was ever sent.
        #
        # Its budget is BOUNDED, though: this retry runs on a worker the
        # deadline has already abandoned, and http_timeout (120 s) there put the
        # process hang straight back. It gets redeem_grace instead.
        calls = []

        def http(method, url, headers=None, body=None, timeout=None):
            calls.append((method, url, headers, timeout))
            if "/chat/completions" in url:
                if headers.get("x-x402-payment-id"):
                    return json_resp(200, CHAT_OK)
                return json_resp(402, fresh_402())
            if "/api/x402/complete/" in url:
                time.sleep(0.3)   # the deposit is confirmed AFTER the deadline
                return json_resp(200, {"status": "completed", "paymentId": "pay_x"})
            raise AssertionError("unexpected url " + url)

        eng = make_engine(http, lambda inv: None)
        eng.set_run_deadline(time.monotonic() + 0.1, 0.1)
        self.assertEqual(self.chat(eng), CHAT_OK)
        retries = [c for c in calls
                   if "/chat/completions" in c[1] and c[2].get("x-x402-payment-id")]
        self.assertEqual(len(retries), 1, "a settled payment never got its request")
        budget = retries[0][3]
        self.assertEqual(budget, eng.redeem_grace,
                         "the paid retry needs its own budget, not the run deadline")
        self.assertGreater(budget, 0.1,
                           "the paid retry got the dead run's deadline, not a real budget")
        self.assertLess(budget, eng.http_timeout,
                        "an unbounded paid retry re-creates the process hang")
        self.assertEqual([p["redeemed"] for p in eng.payments()], [True])
        self.assertEqual([p["status"] for p in eng.payments()], ["sent"])

    def _budget_of_the_paid_retry(self, deadline_in=None):
        """Run one paid call and return the socket timeout its retry got."""
        calls = []

        def http(method, url, headers=None, body=None, timeout=None):
            calls.append((method, url, headers, timeout))
            if "/chat/completions" in url:
                if headers.get("x-x402-payment-id"):
                    return json_resp(200, CHAT_OK)
                return json_resp(402, fresh_402())
            if "/api/x402/complete/" in url:
                return json_resp(200, {"status": "completed", "paymentId": "pay_x"})
            raise AssertionError("unexpected url " + url)

        eng = make_engine(http, lambda inv: None)
        if deadline_in is not None:
            eng.set_run_deadline(time.monotonic() + deadline_in, deadline_in)
        self.assertEqual(self.chat(eng), CHAT_OK)
        retry = [c for c in calls
                 if "/chat/completions" in c[1] and c[2].get("x-x402-payment-id")][0]
        return eng, retry[3]

    def test_a_run_with_no_deadline_gives_the_paid_retry_the_full_http_budget(self):
        # The bound above must not shrink the normal path: with no deadline at
        # all the paid retry keeps the whole http_timeout, exactly as before.
        eng, budget = self._budget_of_the_paid_retry()
        self.assertEqual(budget, eng.http_timeout)

    def test_a_run_with_a_deadline_gives_the_paid_retry_what_the_run_has_left(self):
        # Pins what _redeem_timeout ACTUALLY does, which is not "a live run gets
        # the full 120 s". A live run gets the time its own deadline has left,
        # floored at redeem_grace and capped at http_timeout. The floor is the
        # part that matters for money — the request still lands — and the cap is
        # the part that matters for exit.
        eng, budget = self._budget_of_the_paid_retry(deadline_in=300.0)
        self.assertEqual(budget, eng.http_timeout, "capped at http_timeout")

        eng, budget = self._budget_of_the_paid_retry(deadline_in=60.0)
        self.assertLess(budget, eng.http_timeout,
                        "a 60 s run does not get the full 120 s")
        self.assertGreater(budget, 55.0)

        eng, budget = self._budget_of_the_paid_retry(deadline_in=5.0)
        self.assertEqual(budget, eng.redeem_grace,
                         "floored at redeem_grace so the paid request can land")

    def test_a_cancelled_run_starts_no_new_payment(self):
        # The other side of the same rule: no new money may leave the wallet for
        # a run that is already abandoned, because nothing would redeem it.
        paid = []

        def http(method, url, headers=None, body=None, timeout=None):
            return json_resp(402, fresh_402())

        eng = make_engine(http, paid.append)
        eng.cancel()
        with self.assertRaises(NanoodleError):
            self.chat(eng)
        self.assertEqual(paid, [], "a cancelled run sent a deposit nobody can redeem")
        self.assertEqual(eng.payments(), [])

    def test_no_nano_option_is_actionable_and_unpaid(self):
        paid = []
        no_nano = {"accepts": [a for a in FIXTURE_402["accepts"] if a["scheme"] != "nano"]}

        def http(method, url, headers=None, body=None, timeout=None):
            return json_resp(402, no_nano)

        eng = make_engine(http, lambda inv: paid.append(1))
        with self.assertRaisesRegex(NanoodleError, "no usable Nano option"):
            self.chat(eng)
        self.assertEqual(paid, [])

    def test_keyed_mode_ignores_payment(self):
        paid = []

        def http(method, url, headers=None, body=None, timeout=None):
            return json_resp(402, fresh_402())

        eng = make_engine(http, lambda inv: paid.append(1), api_key="k")
        with self.assertRaisesRegex(NanoodleError, "out of balance"):
            self.chat(eng)
        self.assertEqual(paid, [])


class WalletCallbackFailureTest(unittest.TestCase):
    """MONEY: a wallet callback that RAISES sent nothing. The ledger and the
    error message must never say a deposit went out.

    The ledger entry is opened BEFORE the callback, because the worker can be
    abandoned at any point after that. That is why the entry starts at
    "sending" and only turns "sent" when the callback returns. Claiming a
    deposit that never moved is worse than silence: it sends a person hunting
    an explorer for money that is still in their wallet.
    """

    GRAPH = {"v": 1, "links": [], "nodes": [
        {"id": "n1", "type": "llm", "fields": {"model": "m", "prompt": "hi"}}]}

    def http_402(self, method, url, headers=None, body=None, timeout=None):
        if "/chat/completions" in url:
            return json_resp(402, fresh_402())
        raise AssertionError("unexpected url " + url)

    # ---- state 1: attempted and failed -----------------------------------

    def test_a_failed_callback_records_no_sent_deposit(self):
        def wallet(inv):
            raise RuntimeError("wallet offline: no node reachable")

        eng = make_engine(self.http_402, wallet)
        # the wallet's own exception reaches the caller unchanged
        with self.assertRaisesRegex(RuntimeError, "wallet offline"):
            eng._post_json("/api/v1/chat/completions", {"model": "m", "messages": []})
        ledger = eng.payments()
        self.assertEqual(len(ledger), 1, "the attempt is still on the record")
        self.assertEqual(ledger[0]["status"], "failed")
        self.assertIn("wallet offline", ledger[0]["send_error"])
        self.assertFalse(ledger[0]["redeemed"])

    def test_a_failed_callback_never_claims_money_moved(self):
        def wallet(inv):
            raise RuntimeError("wallet offline: no node reachable")

        wf = Workflow.from_dict(self.GRAPH, api_key="", base_url=BASE,
                                http=self.http_402, poll_intervals={"x402": 0.02},
                                payment=wallet)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        message = ctx.exception.result.nodes["n1"].error
        self.assertIn("wallet offline", message)
        self.assertIn("no Nano deposit was sent", message)
        self.assertNotIn("already sent", message,
                         "nanoodle told the user money moved when the send FAILED")
        self.assertEqual([p["status"] for p in ctx.exception.result.payments], ["failed"])

    def test_a_failed_callback_on_a_timed_out_run_still_tells_the_truth(self):
        # The abandonment sweep is the path that produced the false claim: the
        # paid node fails fast, a sibling lane then trips the run deadline, and
        # the result assembly names every unredeemed deposit.
        def wallet(inv):
            raise RuntimeError("wallet offline: no node reachable")

        def http(method, url, headers=None, body=None, timeout=None):
            if "/chat/completions" in url:
                return json_resp(402, fresh_402())
            if "/api/generate-video" in url:
                return json_resp(200, {"runId": "r1"})
            if "/api/video/status" in url:
                return json_resp(200, {"data": {"status": "PENDING"}})
            raise AssertionError("unexpected url " + url)

        graph = {"v": 1, "links": [], "nodes": [
            self.GRAPH["nodes"][0],
            {"id": "n2", "type": "tvideo",
             "fields": {"model": "seedance-2.0", "prompt": "a paper boat"}}]}
        wf = Workflow.from_dict(graph, api_key="", base_url=BASE, http=http,
                                poll_intervals={"x402": 0.02, "video": 0.05},
                                timeouts={"video": 30.0}, payment=wallet)
        with self.assertRaises(RunError) as ctx:
            wf.run(timeout=0.4)
        result = ctx.exception.result
        self.assertEqual(result.nodes["n2"].error, "run timed out after 0.4s")
        message = result.nodes["n1"].error
        self.assertIn("no Nano deposit was sent", message)
        self.assertNotIn("already sent", message)
        self.assertEqual([p["status"] for p in result.payments], ["failed"])

    # ---- state 2: sent and abandoned -------------------------------------

    def test_a_successful_callback_records_a_sent_deposit(self):
        def http(method, url, headers=None, body=None, timeout=None):
            if "/chat/completions" in url:
                return json_resp(402, fresh_402())
            if "/api/x402/complete/" in url:
                return json_resp(200, CHAT_OK)
            raise AssertionError("unexpected url " + url)

        eng = make_engine(http, lambda inv: None)
        eng._post_json("/api/v1/chat/completions", {"model": "m", "messages": []})
        ledger = eng.payments()
        self.assertEqual([p["status"] for p in ledger], ["sent"])
        self.assertIsNone(ledger[0]["send_error"])
        self.assertTrue(ledger[0]["redeemed"])

    # ---- the ledger freezes when the run is cancelled ---------------------

    def test_cancel_freezes_the_ledger_so_no_deposit_can_hide(self):
        # The race FINDING 4 named: a worker that clears check_cancel()
        # microseconds before Workflow calls cancel() must not be able to record
        # (and then send) a deposit that no sweep and no result snapshot sees.
        # _record_payment re-checks the flag under the lock cancel() takes, so
        # once cancel() returns the ledger cannot grow.
        eng = make_engine(self.http_402, lambda inv: None)
        invoice = parse_nano_invoice(FIXTURE_402, BASE)
        refused = []
        stop = threading.Event()

        def hammer():
            while not stop.is_set():
                try:
                    eng._record_payment(invoice, "trace")
                except NanoodleError:
                    refused.append(1)

        threads = [threading.Thread(target=hammer, daemon=True) for _ in range(4)]
        for t in threads:
            t.start()
        try:
            time.sleep(0.05)
            eng.cancel()
            frozen = len(eng.payments())
            time.sleep(0.1)          # the workers keep hammering
            self.assertEqual(len(eng.payments()), frozen,
                             "a deposit was recorded after cancel() returned")
            self.assertGreater(frozen, 0, "the hammer threads did record before cancel")
            self.assertGreater(len(refused), 0, "no attempt was refused after cancel")
        finally:
            stop.set()
            for t in threads:
                t.join(5.0)


class PaidRetryFailsTest(unittest.TestCase):
    """MONEY: the deposit settled, the paid retry then answered an ERROR status.

    The XNO is gone and the caller got nothing for it, so this deposit is NOT
    redeemed. Marking it redeemed dropped the payment id out of the one
    sentence a user reads — the node error — and left it only in
    result.payments, which almost nobody looks at.
    """

    GRAPH = {"v": 1, "links": [], "nodes": [
        {"id": "n1", "type": "llm", "fields": {"model": "m", "prompt": "hi"}}]}

    def http_500_on_retry(self, method, url, headers=None, body=None, timeout=None):
        if "/chat/completions" in url:
            if headers.get("x-x402-payment-id"):
                return json_resp(500, {"error": "model overloaded"})
            return json_resp(402, fresh_402())
        if "/api/x402/complete/" in url:
            return json_resp(200, {"status": "completed", "paymentId": "pay_x"})
        raise AssertionError("unexpected url " + url)

    def test_a_paid_retry_that_errors_leaves_the_deposit_unredeemed(self):
        eng = make_engine(self.http_500_on_retry, lambda inv: None)
        with self.assertRaisesRegex(NanoodleError, "500"):
            eng._post_json("/api/v1/chat/completions", {"model": "m", "messages": []})
        ledger = eng.payments()
        self.assertEqual([p["status"] for p in ledger], ["sent"])
        self.assertFalse(ledger[0]["redeemed"],
                         "the API errored — this deposit bought nothing")
        self.assertEqual(len(eng.unredeemed_payments()), 1)

    def test_the_payment_id_reaches_the_node_error_when_the_paid_call_fails(self):
        wf = Workflow.from_dict(self.GRAPH, api_key="", base_url=BASE,
                                http=self.http_500_on_retry,
                                poll_intervals={"x402": 0.02},
                                payment=lambda inv: None)
        with self.assertRaises(RunError) as ctx:
            wf.run()
        result = ctx.exception.result
        message = result.nodes["n1"].error
        pid = result.payments[0]["payment_id"]
        self.assertIn("model overloaded", message)
        self.assertIn("a Nano deposit was already sent", message)
        self.assertIn(pid, message,
                      "money left the wallet, the API errored, and the error "
                      "the user reads carried no payment id")
        self.assertIn(pid, ctx.exception.result.errors[0]["message"])

    def test_a_paid_retry_that_succeeds_still_redeems(self):
        # the no-regression guard for the bound above
        def http(method, url, headers=None, body=None, timeout=None):
            if "/chat/completions" in url:
                if headers.get("x-x402-payment-id"):
                    return json_resp(200, CHAT_OK)
                return json_resp(402, fresh_402())
            if "/api/x402/complete/" in url:
                return json_resp(200, {"status": "completed", "paymentId": "pay_x"})
            raise AssertionError("unexpected url " + url)

        eng = make_engine(http, lambda inv: None)
        eng._post_json("/api/v1/chat/completions", {"model": "m", "messages": []})
        self.assertEqual([p["redeemed"] for p in eng.payments()], [True])
        self.assertEqual(eng.unredeemed_payments(), [])


class TimedOutRunKeepsThePaymentTraceableTest(unittest.TestCase):
    """MONEY: a deposit sent by a worker that the run then abandons must still
    reach the caller.

    The worker raises NodeCancelled naming the payment id, but its future is
    discarded by the deadline handler and concurrent.futures never logs an
    unretrieved exception. Without the engine-side ledger the user who sent XNO
    gets nothing but "run timed out after 0.3s".
    """

    GRAPH = {"v": 1, "links": [], "nodes": [
        {"id": "n1", "type": "llm", "fields": {"model": "m", "prompt": "hi"}}]}

    def test_the_payment_id_reaches_the_caller_after_a_run_timeout(self):
        sent = []

        def http(method, url, headers=None, body=None, timeout=None):
            if "/chat/completions" in url:
                return json_resp(402, fresh_402())
            if "/api/x402/complete/" in url:
                # slow confirmation: the worker is still INSIDE this call when
                # the deadline fires, so its future is abandoned and its
                # exception is thrown away. That is the case the ledger covers.
                time.sleep(0.6)
                return json_resp(402, {"error": "Payment not verified", "status": "pending"})
            raise AssertionError("unexpected url " + url)

        wf = Workflow.from_dict(self.GRAPH, api_key="", base_url=BASE, http=http,
                                poll_intervals={"x402": 0.02}, payment=sent.append)
        with self.assertRaises(RunError) as ctx:
            wf.run(timeout=0.3)
        result = ctx.exception.result
        self.assertEqual(len(sent), 1, "the callback sent exactly one deposit")
        pay_id = sent[0]["paymentId"]

        message = result.nodes["n1"].error
        self.assertIn("run timed out after 0.3s", message)
        self.assertIn(pay_id, message, "the timed-out node hid the payment id")
        self.assertIn(sent[0]["explorerUrl"], message)
        self.assertIn(pay_id, result.errors[0]["message"])
        # state 2: this deposit REALLY went out, so the message says so plainly
        self.assertIn("a Nano deposit was already sent", message)

        self.assertEqual([p["payment_id"] for p in result.payments], [pay_id])
        self.assertEqual(result.payments[0]["node_id"], "n1")
        self.assertEqual(result.payments[0]["status"], "sent")
        self.assertIsNone(result.payments[0]["send_error"])
        self.assertFalse(result.payments[0]["redeemed"])

    def test_a_keyed_run_reports_no_payments(self):
        def http(method, url, headers=None, body=None, timeout=None):
            return json_resp(200, CHAT_OK)

        wf = Workflow.from_dict(self.GRAPH, api_key="k", base_url=BASE, http=http)
        self.assertEqual(wf.run().payments, [])


class GuardTest(unittest.TestCase):
    def test_seed_string_refused(self):
        with self.assertRaisesRegex(NanoodleError, "never accepts wallet seeds or private keys"):
            assert_payment_option("vault grief snake ... twelve words")
        with self.assertRaisesRegex(NanoodleError, "never accepts wallet seeds"):
            Workflow.from_dict({"nodes": [], "links": []}, payment="seed")

    def test_workflow_keyless_with_payment_passes_guard(self):
        # regression twin of the js env-fallback bug: api_key="" must stay keyless
        # even when NANOGPT_API_KEY is set, and payment= satisfies the network guard
        import os
        prev = os.environ.get("NANOGPT_API_KEY")
        os.environ["NANOGPT_API_KEY"] = "sk-should-never-be-used"
        try:
            wf = Workflow.from_dict(
                {"nodes": [{"id": "n1", "type": "text", "x": 0, "y": 0, "fields": {"text": "hi"}}],
                 "links": []},
                api_key="", payment=lambda inv: None)
            self.assertFalse(wf._api_key)
            self.assertIsNotNone(wf._payment)
        finally:
            if prev is None:
                del os.environ["NANOGPT_API_KEY"]
            else:
                os.environ["NANOGPT_API_KEY"] = prev


if __name__ == "__main__":
    unittest.main()
