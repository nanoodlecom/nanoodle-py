#!/usr/bin/env python3
"""Measure the timed-out-run process hang. Offline. Spends nothing.

One `tvideo` node whose status endpoint answers PENDING forever, abandoned by
`run(timeout=0.5)`. A parent process times the child from spawn to REAL exit,
which is what a user sees: `run()` returns on time, the interpreter does not.

The child is the committed harness template `tests.test_timeout_shutdown._CHILD`
itself, so these numbers and the test suite measure the same thing. The poll
interval is the template default of 5.0 s, so exit times land on poll
boundaries, not on `timeout_video` itself.

    python3 scripts/measure-timeout-hang.py                  # this tree
    python3 scripts/measure-timeout-hang.py --tree /path/to/pre-fix-checkout

Make a pre-fix tree with:

    mkdir /tmp/prefix && git archive <commit> | tar -x -C /tmp/prefix
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from tests.test_timeout_shutdown import _CHILD  # noqa: E402


def measure(tree, video_timeout, run_timeout, poll, hard_limit):
    """Spawn the child in `tree` and return (returned_secs, exit_secs)."""
    src = _CHILD.format(root=tree, src=os.path.join(tree, "src"),
                        poll=poll, video_timeout=video_timeout,
                        run_timeout=run_timeout)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "timeout_child.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        t0 = time.monotonic()
        try:
            p = subprocess.run([sys.executable, path], cwd=tree, capture_output=True,
                               text=True, timeout=hard_limit)
        except subprocess.TimeoutExpired:
            return None, None
        exited = time.monotonic() - t0
        returned = None
        for line in p.stdout.splitlines():
            if line.startswith("returned "):
                returned = float(line.split()[1])
        return returned, exited


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=_REPO_ROOT,
                    help="checkout to measure (default: this one)")
    ap.add_argument("--video-timeouts", default="3.0,8.0,16.0",
                    help="comma-separated timeouts={'video': N} values")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--run-timeout", type=float, default=0.5)
    ap.add_argument("--hard-limit", type=float, default=90.0,
                    help="give up on the child after this many seconds")
    args = ap.parse_args(argv)

    tree = os.path.abspath(args.tree)
    print("tree: %s" % tree)
    print("run(timeout=%.1f), poll_intervals={'video': %.1f}" % (args.run_timeout, args.poll))
    print("%-16s %-18s %s" % ("timeouts.video", "run() returned", "process exited"))
    for raw in args.video_timeouts.split(","):
        n = float(raw)
        returned, exited = measure(tree, n, args.run_timeout, args.poll, args.hard_limit)
        if exited is None:
            print("%-16.1f %-18s did not exit within %.0f s" % (n, "-", args.hard_limit))
        else:
            print("%-16.1f %-18.3f %.2f s" % (n, returned, exited))
    return 0


if __name__ == "__main__":
    sys.exit(main())
