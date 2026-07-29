#!/usr/bin/env python3
"""Measure the timed-out-run process hang. Offline. Spends nothing.

One `tvideo` node whose status endpoint answers PENDING forever, abandoned by
`run(timeout=0.5)`. A parent process times the child from spawn to REAL exit,
which is what a user sees: `run()` returns on time, the interpreter does not.

The child is the committed harness template `tests.test_timeout_shutdown._CHILD`
itself, so these numbers and the test suite measure the same thing. The poll
interval is the template default of 5.0 s, so exit times land on poll
boundaries, not on `timeout_video` itself.

Two more cases use the same rule — the committed test harness IS the child:
`_X402_CHILD` (the request a settled deposit paid for is in flight at the
deadline) and `_FFMPEG_CHILD` (a daemon worker is inside `local_media._run`
with an ffprobe child running when the process exits).

    python3 scripts/measure-timeout-hang.py                  # this tree
    python3 scripts/measure-timeout-hang.py --tree /path/to/pre-fix-checkout

    # does an orphaned child stop on its own? (no: ffmpeg ignores SIGPIPE)
    python3 scripts/measure-timeout-hang.py --no-x402 --video-timeouts "" \
        --chatty --no-reaper --ffmpeg-runs 8 --hard-limit 45

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

from tests.test_timeout_shutdown import (CHATTY_FFMPEG_ARGS,  # noqa: E402
                                         QUIET_PROBE_ARGS, _CHILD,
                                         _FFMPEG_CHILD, _X402_CHILD,
                                         _child_alive)


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


def measure_x402(tree, run_timeout, hard_limit):
    """The money path: the request a settled deposit paid for is in flight when
    the deadline fires. Returns (returned_secs, exit_secs, paid_requests)."""
    src = _X402_CHILD.format(root=tree, src=os.path.join(tree, "src"),
                             run_timeout=run_timeout)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x402_child.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        t0 = time.monotonic()
        try:
            p = subprocess.run([sys.executable, path], cwd=tree, capture_output=True,
                               text=True, timeout=hard_limit)
        except subprocess.TimeoutExpired:
            return None, None, None
        exited = time.monotonic() - t0
        returned, paid = None, None
        for line in p.stdout.splitlines():
            if line.startswith("returned "):
                parts = line.split()
                returned, paid = float(parts[1]), int(parts[3])
        return returned, exited, paid


def measure_ffmpeg(tree, hard_limit, chatty=False, reaper=True):
    """Local media: a daemon worker is inside local_media._run with a child
    running when the process exits. Returns (exit_secs, orphan_secs, gave_up),
    where orphan_secs is how long the child outlived its parent and gave_up
    says the child was STILL running when the wait ran out.

    `chatty` swaps the quiet ffprobe for an ffmpeg that writes a progress line
    to stderr about twice a second. `reaper=False` removes the atexit hook, so
    the two together answer: does SIGPIPE bound an orphaned child on its own?
    """
    src = _FFMPEG_CHILD.format(
        root=tree, src=os.path.join(tree, "src"), reaper=reaper,
        bin="ffmpeg" if chatty else "ffprobe",
        args=CHATTY_FFMPEG_ARGS if chatty else QUIET_PROBE_ARGS)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ffmpeg_child.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        t0 = time.monotonic()
        try:
            p = subprocess.run([sys.executable, path], cwd=tree, capture_output=True,
                               text=True, timeout=hard_limit)
        except subprocess.TimeoutExpired:
            return None, None, False
        exited = time.monotonic() - t0
    pids = [int(line.split()[1]) for line in p.stdout.splitlines()
            if line.startswith("child_pid ")]
    if not pids:
        return exited, None, False
    end = time.monotonic() + hard_limit
    while _child_alive(pids[0]) and time.monotonic() < end:
        time.sleep(0.02)
    orphan = time.monotonic() - (t0 + exited)
    gave_up = _child_alive(pids[0])
    if gave_up:
        os.kill(pids[0], 9)      # never leave one behind
    return exited, orphan, gave_up


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
    ap.add_argument("--no-x402", action="store_true",
                    help="skip the paid-retry (money path) measurement")
    ap.add_argument("--no-ffmpeg", action="store_true",
                    help="skip the local-media child-reap measurement")
    ap.add_argument("--chatty", action="store_true",
                    help="local media: use a chatty ffmpeg, not a quiet ffprobe")
    ap.add_argument("--no-reaper", action="store_true",
                    help="local media: remove the atexit child reaper first")
    ap.add_argument("--ffmpeg-runs", type=int, default=1,
                    help="repeat the local-media measurement N times")
    args = ap.parse_args(argv)

    tree = os.path.abspath(args.tree)
    print("tree: %s" % tree)
    print("run(timeout=%.1f), poll_intervals={'video': %.1f}" % (args.run_timeout, args.poll))
    print("%-16s %-18s %s" % ("timeouts.video", "run() returned", "process exited"))
    for raw in args.video_timeouts.split(","):
        if not raw.strip():          # --video-timeouts "" skips the poll-loop case
            continue
        n = float(raw)
        returned, exited = measure(tree, n, args.run_timeout, args.poll, args.hard_limit)
        if exited is None:
            print("%-16.1f %-18s did not exit within %.0f s" % (n, "-", args.hard_limit))
        else:
            print("%-16.1f %-18.3f %.2f s" % (n, returned, exited))
    if not args.no_x402:
        print()
        print("keyless x402 llm node, paid retry in flight at the deadline "
              "(run(timeout=%.1f))" % args.run_timeout)
        returned, exited, paid = measure_x402(tree, args.run_timeout, args.hard_limit)
        if exited is None:
            print("  did not exit within %.0f s" % args.hard_limit)
        else:
            print("  run() returned %.3f s, process exited %.2f s, "
                  "paid requests sent %s" % (returned, exited, paid))
    if not args.no_ffmpeg:
        print()
        print("local media: a daemon worker is inside local_media._run with a "
              "%s child running when the process exits (child reaper %s)"
              % ("chatty ffmpeg" if args.chatty else "quiet ffprobe",
                 "OFF" if args.no_reaper else "ON"))
        for i in range(args.ffmpeg_runs):
            exited, orphan, gave_up = measure_ffmpeg(
                tree, args.hard_limit, chatty=args.chatty,
                reaper=not args.no_reaper)
            if exited is None:
                print("  run %d: did not exit within %.0f s" % (i + 1, args.hard_limit))
            elif orphan is None:
                print("  run %d: process exited %.2f s, no child was started "
                      "(is ffmpeg on PATH?)" % (i + 1, exited))
            else:
                print("  run %d: process exited %.2f s, child outlived the parent "
                      "by %.2f s%s"
                      % (i + 1, exited, orphan,
                         " AND WAS STILL RUNNING (gave up, killed it)" if gave_up else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
