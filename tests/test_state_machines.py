"""Offline regressions for lifecycle transitions found in the state-machine audit."""

import threading
import time
import unittest
from typing import ClassVar
from unittest import mock

from tests.test_x402 import BASE, CHAT_OK, fresh_402, json_resp
from nanoodle import NanoodleError, RunError, Workflow
from nanoodle.engine import Engine, NodeCancelled
from nanoodle.transport import HttpResponse


def workflow(kind="llm", **opts):
    fields = {"model": "m", "prompt": "hi"}
    if kind == "decide":
        fields.update(mode="yesno", question="Ready?")
    return Workflow.from_dict({"nodes": [{"id": "n", "type": kind, "fields": fields}]}, api_key="k", **opts)


class InvoiceLifecycleTest(unittest.TestCase):
    def test_expired_invoice_never_enters_wallet(self):
        sent = []
        eng = Engine("", BASE, lambda *a, **kw: json_resp(402, fresh_402(-1)), payment=sent.append)
        with self.assertRaisesRegex(NanoodleError, "expired"):
            eng._post_json("/chat", {})
        self.assertEqual(sent, [])
        self.assertEqual(eng.payments(), [])

    def test_callback_can_broadcast_before_raising(self):
        broadcasts = []

        def wallet(invoice):
            broadcasts.append(invoice["paymentId"])
            raise RuntimeError("confirmation connection lost after broadcast")

        wf = Workflow.from_dict({"nodes": [{"id": "n", "type": "llm",
            "fields": {"model": "m", "prompt": "hi"}}]}, api_key="", payment=wallet,
            http=lambda *a, **kw: json_resp(402, fresh_402()))
        with self.assertRaises(RunError) as ctx:
            wf.run()
        result = ctx.exception.result
        self.assertEqual(len(broadcasts), 1)
        self.assertEqual(result.payments[0]["status"], "failed")
        self.assertIn("may have been sent", result.nodes["n"].error)
        self.assertIn("before paying again", result.nodes["n"].error)
        self.assertNotIn("no Nano deposit was sent", result.nodes["n"].error)
        self.assertIn(broadcasts[0], result.nodes["n"].error)

    def test_timezone_expiry_is_checked_before_wallet(self):
        invoice = fresh_402()
        for option in invoice["accepts"]:
            option["expiresAt"] = "2020-01-01T01:00:00.123+01:00"
        sent = []
        eng = Engine("", BASE, lambda *a, **kw: json_resp(402, invoice), payment=sent.append)
        with self.assertRaisesRegex(NanoodleError, "expired"):
            eng._post_json("/chat", {})
        self.assertEqual(sent, [])

    def test_bad_expiry_or_raw_amount_never_enters_wallet(self):
        for field, value in (("expiresAt", "bad-date"), ("expiresAt", float("inf")),
                             ("maxAmountRequired", "0"), ("maxAmountRequired", "-1"),
                             ("maxAmountRequired", "1.5")):
            with self.subTest(field=field, value=value):
                invoice = fresh_402()
                for option in invoice["accepts"]:
                    option[field] = value
                sent = []
                eng = Engine("", BASE, lambda *a, **kw: json_resp(402, invoice), payment=sent.append)
                with self.assertRaises(NanoodleError):
                    eng._post_json("/chat", {})
                self.assertEqual(sent, [])


class PollLifecycleTest(unittest.TestCase):
    completions: ClassVar = {
        "tts": {"status": "completed", "url": "https://result.test/audio.mp3"},
        "tvideo": {"status": "completed", "output": {"url": "https://result.test/video.mp4"}},
        "model3d": {"status": "completed", "output": {
            "format": "glb", "model_url": "https://result.test/model.glb"}},
    }

    def wf(self, kind, replies):
        self.polls = []
        def http(method, url, **kw):
            if method == "POST":
                return json_resp(200, {"runId": "job", "cost": 0.1})
            self.polls.append(url)
            return replies[min(len(self.polls) - 1, len(replies) - 1)]
        return workflow(kind, http=http, poll_intervals={"video": 0.001, "audio": 0.001},
                        timeouts={"video": 0.04, "audio": 0.04})

    def test_permanent_http_failures_are_terminal_even_with_non_json_body(self):
        for kind in ("tvideo", "tts", "model3d"):
            for status in (401, 403, 404, 410):
                with self.subTest(kind=kind, status=status):
                    wf = self.wf(kind, [HttpResponse(status, {}, b"denied")])
                    with self.assertRaises(RunError) as ctx:
                        wf.run()
                    self.assertNotIn("timed out", str(ctx.exception))
                    self.assertEqual(len(self.polls), 1)

    def test_audio_cannot_accept_success_shaped_error_response(self):
        wf = self.wf("tts", [json_resp(503, {"status": "completed", "url": "https://wrong"}),
                             json_resp(200, {"status": "completed", "url": "https://right"})])
        self.assertEqual(wf.run()["n"].url, "https://right")
        self.assertEqual(len(self.polls), 2)

    def test_cancelled_spellings_are_terminal(self):
        for kind in ("tvideo", "tts"):
            for status in ("canceled", "cancelled"):
                with self.subTest(kind=kind, status=status):
                    wf = self.wf(kind, [json_resp(200, {"status": status})])
                    with self.assertRaises(RunError) as ctx:
                        wf.run()
                    self.assertIn("failed", str(ctx.exception))
                    self.assertEqual(len(self.polls), 1)

    def test_transient_http_and_malformed_json_shapes_recover(self):
        for kind in ("tvideo", "tts"):
            with self.subTest(kind=kind):
                complete = ({"status": "completed", "url": "https://right"} if kind == "tts" else
                            {"status": "completed", "output": {"url": "https://right"}})
                replies = [json_resp(s, {}) for s in (408, 409, 425, 429, 500, 503)]
                replies += [json_resp(200, []), json_resp(200, None), json_resp(200, complete)]
                self.assertEqual(self.wf(kind, replies).run()["n"].url, "https://right")
                self.assertEqual(len(self.polls), len(replies))

    def test_poll_budget_bounds_wait_and_io_and_discards_late_completion(self):
        for kind, completion in self.completions.items():
            for interval in (0.125, 2.0):
                with self.subTest(kind=kind, interval=interval):
                    elapsed = 0.0
                    polls = []

                    def sleep(_engine, seconds):
                        nonlocal elapsed
                        elapsed += seconds

                    def http(method, url, *, timeout, **kw):
                        nonlocal elapsed
                        if method == "POST":
                            return json_resp(200, {"runId": "job", "cost": 0.1})
                        polls.append((elapsed, timeout))
                        elapsed = 1.25  # an adapter that returned despite the timeout
                        return json_resp(200, completion)

                    wf = workflow(kind, http=http, poll_intervals={"video": interval, "audio": interval},
                                  timeouts={"video": 1.0, "audio": 1.0})
                    with mock.patch("nanoodle.engine.time") as clock, mock.patch.object(Engine, "sleep", sleep):
                        clock.monotonic.side_effect = lambda: elapsed
                        with self.assertRaisesRegex(RunError, "timed out") as ctx:
                            wf.run()
                    self.assertEqual(ctx.exception.result.cost_usd, 0.1)
                    if interval > 1.0:
                        self.assertEqual(polls, [])
                        self.assertEqual(elapsed, 1.0)
                    else:
                        self.assertEqual(polls, [(0.125, 0.875)])

    def test_run_deadline_caps_poll_io_and_outranks_late_status(self):
        elapsed = 0.0
        calls = []

        def sleep(_engine, seconds):
            nonlocal elapsed
            elapsed += seconds

        def http(method, url, *, timeout, **kw):
            nonlocal elapsed
            calls.append(timeout)
            elapsed = 0.375
            return json_resp(200, self.completions["tvideo"])

        engine = Engine("k", BASE, http)
        engine.set_run_deadline(0.25, 0.25)
        with mock.patch("nanoodle.engine.time") as clock, mock.patch.object(Engine, "sleep", sleep):
            clock.monotonic.side_effect = lambda: elapsed
            with self.assertRaisesRegex(NodeCancelled, "run timed out"):
                list(engine._poll_statuses("/api/video/status", 0.125, 1.0))
        self.assertEqual(calls, [0.125])


class RunLifecycleTest(unittest.TestCase):
    def test_invalid_timeouts_fail_before_execution(self):
        for timeout in (-1, float("nan"), float("inf"), "10", True):
            with self.subTest(timeout=timeout):
                calls = []
                wf = workflow(http=lambda *a, **kw: calls.append(a))
                with self.assertRaisesRegex(NanoodleError, "timeout"):
                    wf.run(timeout=timeout)
                self.assertEqual(calls, [])

    def test_zero_timeout_does_no_work(self):
        calls = []
        with self.assertRaises(RunError):
            workflow(http=lambda *a, **kw: calls.append(a)).run(timeout=0)
        self.assertEqual(calls, [])

    def test_late_cost_cannot_mutate_returned_node_or_claim_exact_total(self):
        release = threading.Event()
        finished = threading.Event()
        def http(*a, **kw):
            release.wait(2)
            return json_resp(200, dict(CHAT_OK, cost=0.25))
        wf = workflow(http=http)
        original = Engine.run_node
        def run_node(*args):
            try:
                return original(*args)
            finally:
                finished.set()
        with mock.patch.object(Engine, "run_node", run_node):
            try:
                with self.assertRaises(RunError) as ctx:
                    wf.run(timeout=0.1)
                result = ctx.exception.result
                before = (result.nodes["n"].cost_usd, result.cost_usd)
                release.set()
                self.assertTrue(finished.wait(2))
                self.assertEqual((result.nodes["n"].cost_usd, result.cost_usd), before)
                self.assertFalse(result.cost_exact)
            finally:
                release.set()

    def test_interrupt_cancels_inflight_workers_without_joining_them(self):
        entered = threading.Event()
        finished = threading.Event()
        eng = Engine("k", BASE, None)
        def run_node(*args):
            entered.set()
            try:
                eng.sleep(0.5)
                eng.check_cancel()
                return {"text": "late"}
            finally:
                finished.set()
        def interrupt(*args, **kwargs):
            self.assertTrue(entered.wait(1))
            raise KeyboardInterrupt()
        with mock.patch.object(eng, "run_node", run_node), \
             mock.patch.object(Workflow, "_make_engine", return_value=eng), \
             mock.patch("nanoodle.workflow.wait", side_effect=interrupt):
            t0 = time.monotonic()
            with self.assertRaises(KeyboardInterrupt):
                workflow().run()
            self.assertLess(time.monotonic() - t0, 0.3)
            self.assertTrue(eng.cancelled())
            self.assertTrue(finished.wait(1))


class MediaLifecycleTest(unittest.TestCase):
    def test_image_cost_survives_missing_output(self):
        wf = workflow("image", http=lambda *a, **kw: json_resp(200, {"cost": 0.25, "data": []}))
        with self.assertRaisesRegex(RunError, "no image") as ctx:
            wf.run()
        self.assertEqual(ctx.exception.result.cost_usd, 0.25)
        self.assertEqual(ctx.exception.result.nodes["n"].cost_usd, 0.25)

    def test_billed_json_decode_error_records_headers_or_unknown_cost(self):
        for kind in ("llm", "decide", "image", "tvideo", "model3d", "tts"):
            for price in ("0.2", None):
                with self.subTest(kind=kind, price=price):
                    headers = {"content-type": "application/json", "x-remaining-balance": "4.5"}
                    if price is not None:
                        headers["x-cost"] = price
                    wf = workflow(kind, http=lambda *a, **kw: HttpResponse(200, headers, b'{"data":'))
                    with self.assertRaises(RunError) as ctx:
                        wf.run()
                    result = ctx.exception.result
                    self.assertEqual(result.cost_usd, 0.2 if price else 0.0)
                    self.assertEqual(result.nodes["n"].cost_usd, 0.2 if price else None)
                    self.assertEqual(result.cost_exact, price is not None)
                    self.assertEqual(result.remaining_balance, 4.5)

    def test_cleanvoice_probe_tracks_child_and_honors_deadline(self):
        from nanoodle.cleanvoice import clean_voice_media_seconds
        check = mock.Mock()
        with mock.patch("nanoodle.cleanvoice.shutil.which", return_value="/bin/ffprobe"), \
             mock.patch("nanoodle.local_media._run", return_value=(b"6.05", "")) as run:
            self.assertEqual(clean_voice_media_seconds("https://clip", check, 123), 6.05)
            self.assertEqual(run.call_args.kwargs, {"timeout": 3, "cancel_check": check, "deadline": 123})
            check.assert_called_once()

    def test_cleanvoice_probe_cancellation_is_not_an_advisory_failure(self):
        from nanoodle.cleanvoice import clean_voice_media_seconds
        for error in (NodeCancelled("run cancelled"), NanoodleError("probe failed")):
            check = mock.Mock(side_effect=[None, NodeCancelled("run cancelled")])
            with mock.patch("nanoodle.cleanvoice.shutil.which", return_value="/bin/ffprobe"), \
                 mock.patch("nanoodle.local_media._run", side_effect=error):
                with self.assertRaises(NodeCancelled):
                    clean_voice_media_seconds("https://clip", check)
