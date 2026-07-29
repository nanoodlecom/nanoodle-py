"""Regression: after run(timeout=...) fires, the abandoned worker threads must
STOP, not keep polling until the per-node timeout.

Before the fix run() returned on time but the process did not exit. The pool
threads are not daemons and concurrent.futures joins every one of them at
interpreter shutdown, so an abandoned poll loop held the process for the whole
video timeout (default 600 s).
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from tests._util import MockedTest
from tests.harness import video_status

from nanoodle import RunError

# Generous enough for a loaded CI box, tight enough that a 30 s (let alone
# 600 s) poll loop fails it.
SETTLE_BOUND = 10.0
EXIT_BOUND = 25.0

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AbandonedWorkerTest(MockedTest):
    """A video node that never settles, abandoned by a run deadline."""

    def _never_settling_video(self):
        self.mock.script("POST", "/api/generate-video",
                         {"status": 200, "json": {"runId": "r1"}})
        self.mock.script("GET", "/api/video/status", video_status("PENDING"))

    def test_worker_thread_dies_promptly_after_the_deadline(self):
        self._never_settling_video()
        # The node-start event fires ON the pool worker, so this hands us the
        # exact thread object. That is name-independent — no reliance on the
        # executor's thread naming.
        workers = []

        def watch(evt):
            if evt.get("type") == "node-start":
                workers.append(threading.current_thread())

        # Poll interval 5 s: the pre-fix loop slept straight through the
        # deadline and then polled on to timeout_video. 30 s here (not the
        # 600 s default) only bounds the damage a REGRESSION does to this
        # runner, because a stuck non-daemon thread blocks its exit too.
        # ProcessExitTest below carries the full 600 s case, where a hard
        # subprocess timeout contains it.
        wf = self.wf("video-poll.json",
                     poll_intervals={"video": 5.0},
                     timeouts={"video": 30.0})
        t0 = time.monotonic()
        with self.assertRaises(RunError):
            wf.run(timeout=0.2, on_progress=watch)
        self.assertLess(time.monotonic() - t0, 2.0)   # run() still returns fast

        self.assertEqual(len(workers), 1, "expected exactly one in-flight node")
        worker = workers[0]
        # This is WHY a live worker hangs the process: concurrent.futures
        # registers an atexit hook that joins every non-daemon pool thread.
        self.assertFalse(worker.daemon)
        end = time.monotonic() + SETTLE_BOUND
        while worker.is_alive() and time.monotonic() < end:
            time.sleep(0.02)
        settled = time.monotonic() - t0
        self.assertFalse(
            worker.is_alive(),
            "the abandoned worker was still alive %.1fs after the deadline — "
            "the interpreter will block on it at exit" % settled)
        self.assertLess(settled, SETTLE_BOUND)

    def test_poll_loop_stops_calling_the_api_after_the_deadline(self):
        self._never_settling_video()
        wf = self.wf("video-poll.json",
                     poll_intervals={"video": 0.05},
                     timeouts={"video": 30.0})   # see above: bounds a regression
        with self.assertRaises(RunError):
            wf.run(timeout=0.3)
        # give any surviving loop room to prove itself
        time.sleep(0.6)
        polls_a = len(self.mock.requests_to("/api/video/status"))
        time.sleep(0.6)
        polls_b = len(self.mock.requests_to("/api/video/status"))
        self.assertEqual(polls_a, polls_b,
                         "the abandoned node kept polling after the deadline")


class NoDeadlineUnaffectedTest(MockedTest):
    """A run WITHOUT a timeout must behave exactly as before: the poll loop runs
    to the node's own timeout and nothing cancels it early."""

    def test_video_node_still_polls_to_its_own_timeout(self):
        self.mock.script("POST", "/api/generate-video",
                         {"status": 200, "json": {"runId": "r1"}})
        self.mock.script("GET", "/api/video/status", video_status("PENDING"))
        wf = self.wf("video-poll.json",
                     poll_intervals={"video": 0.05},
                     timeouts={"video": 0.4})
        t0 = time.monotonic()
        with self.assertRaises(RunError) as ctx:
            wf.run()
        elapsed = time.monotonic() - t0
        self.assertIn("video timed out", str(ctx.exception))
        self.assertGreaterEqual(elapsed, 0.35)   # it really did poll the 0.4 s
        self.assertGreater(len(self.mock.requests_to("/api/video/status")), 2)


_CHILD = r'''
import sys, time
sys.path.insert(0, {root!r})
sys.path.insert(0, {src!r})

from tests.harness import MockNanoGPT
from nanoodle import Workflow, RunError

GRAPH = {{"v": 1, "nodes": [{{"id": "n1", "type": "tvideo",
         "fields": {{"model": "seedance-2.0", "prompt": "a paper boat"}}}}], "links": []}}

mock = MockNanoGPT().start()
mock.script("POST", "/api/generate-video", {{"status": 200, "json": {{"runId": "r1"}}}})
mock.script("GET", "/api/video/status", {{"status": 200, "json": {{"data": {{"status": "PENDING"}}}}}})

wf = Workflow.from_dict(GRAPH, api_key="k", base_url=mock.base_url,
                        poll_intervals={{"video": 5.0}}, timeouts={{"video": 600.0}})
t0 = time.monotonic()
try:
    wf.run(timeout=0.5)
except RunError:
    pass
print("returned %.3f" % (time.monotonic() - t0), flush=True)
'''


class ProcessExitTest(unittest.TestCase):
    """End-to-end proof: the whole interpreter exits, not just run().

    This is the real-world case — timeout_video at its 600 s default.
    """

    def test_process_exits_promptly_after_a_timed_out_run(self):
        src = _CHILD.format(root=_REPO_ROOT,
                            src=os.path.join(_REPO_ROOT, "src"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "timeout_child.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(src)
            t0 = time.monotonic()
            try:
                p = subprocess.run([sys.executable, path], cwd=_REPO_ROOT,
                                   capture_output=True, text=True,
                                   timeout=EXIT_BOUND)
            except subprocess.TimeoutExpired:
                self.fail("the child process did not exit within %.0fs after a "
                          "0.5s run timeout — the abandoned poll thread is "
                          "holding interpreter shutdown" % EXIT_BOUND)
            elapsed = time.monotonic() - t0
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            self.assertIn("returned", p.stdout)
            self.assertLess(elapsed, EXIT_BOUND)


if __name__ == "__main__":
    unittest.main()
