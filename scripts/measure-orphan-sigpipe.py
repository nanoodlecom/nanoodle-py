#!/usr/bin/env python3
"""Does SIGPIPE stop an orphaned ffmpeg? Answer: only sometimes. Offline.

`local_media` starts ffmpeg/ffprobe with pipes. When the parent dies, the read
end of those pipes closes, so the next write of the child gets SIGPIPE. It is
tempting to call that a backstop for an orphaned child. It is not one, and
`local_media._kill_children_at_exit` is the only thing that bounds an orphan.

This script measures the exact fate of the child, not a guess. It makes itself
a child subreaper (`prctl(PR_SET_CHILD_SUBREAPER)`), so the orphan is
reparented to THIS process and `waitpid()` gives the real exit status: killed
by signal 13, or exited with a code, or still alive at the end of the wait.

The result is a race with the start-up of the child:

- `ffmpeg` sets SIGPIPE to `SIG_IGN` a fraction of a second after it starts
  (0.08 s on an idle machine here, 0.61 s on a busy one). Before that point
  SIGPIPE kills it. After that point its writes fail with EPIPE, `av_log`
  discards the error, and it runs on to the end of the work.
- `ffprobe` does not ignore SIGPIPE at all, so a chatty probe dies at once.
- Which side of the race you land on depends on the machine and the load. Do
  not depend on either side.

    # the parent dies inside the start-up window: SIGPIPE kills ffmpeg
    python3 scripts/measure-orphan-sigpipe.py --delay 0.05 --runs 6

    # the parent dies after it: ffmpeg ignores SIGPIPE and runs on
    python3 scripts/measure-orphan-sigpipe.py --delay 1.0 --runs 6

    # when does ffmpeg start to ignore SIGPIPE?
    python3 scripts/measure-orphan-sigpipe.py --when --runs 8

Linux only: it reads /proc and it needs PR_SET_CHILD_SUBREAPER.
"""

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time

PR_SET_CHILD_SUBREAPER = 36
SIGPIPE_BIT = 1 << 12          # signal 13 in the /proc SigIgn mask

# A chatty encode of a long source: it writes a progress line to stderr about
# twice a second, and it needs minutes of work, so it cannot end on its own
# inside the wait.
FFMPEG = ["ffmpeg", "-f", "lavfi",
          "-i", "testsrc=size=1920x1080:rate=30:duration=600", "-f", "null", "-"]
# The same source, probed frame by frame: a CSV line per frame on stdout.
FFPROBE = ["ffprobe", "-show_frames", "-of", "csv",
           "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30:duration=600"]


def sigpipe_ignored(pid):
    """True when the process has SIGPIPE set to SIG_IGN."""
    try:
        with open("/proc/%d/status" % pid, "r") as f:
            mask = [ln for ln in f if ln.startswith("SigIgn")][0].split()[1]
    except (OSError, IndexError):
        return None
    return bool(int(mask, 16) & SIGPIPE_BIT)


def measure_when(cmd, runs):
    """How long after it starts does the child begin to ignore SIGPIPE?"""
    for i in range(runs):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        t0 = time.monotonic()
        seen = None
        while time.monotonic() - t0 < 5.0:
            state = sigpipe_ignored(p.pid)
            if state is None:
                break
            if state:
                seen = time.monotonic() - t0
                break
            time.sleep(0.005)
        print("  run %d: SIGPIPE set to SIG_IGN %s"
              % (i + 1, "%.3f s after the start" % seen if seen
                 else "NEVER inside 5 s"))
        p.kill()
        p.wait()


def measure_fate(cmd, runs, delay, wait):
    """Start the child, kill its parent `delay` s later, report its real fate."""
    for i in range(runs):
        r, w = os.pipe()
        mid = os.fork()
        if mid == 0:                       # the parent that goes away
            os.close(r)
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            os.write(w, b"%d\n" % p.pid)
            os.close(w)
            time.sleep(delay)
            os._exit(0)                    # it does NOT kill its child
        os.close(w)
        with os.fdopen(r) as f:
            pid = int(f.readline())
        os.waitpid(mid, 0)
        gone = time.monotonic()            # the pipes are closed from here
        status = None
        while time.monotonic() - gone < wait:
            try:
                got, st = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                got, st = 0, 0
            if got == pid:
                status = st
                break
            time.sleep(0.02)
        lived = time.monotonic() - gone
        if status is None:
            os.kill(pid, signal.SIGKILL)   # never leave an orphan behind
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            print("  run %d: pid %d STILL RUNNING %.0f s after its parent "
                  "(killed it)" % (i + 1, pid, wait))
        elif os.WIFSIGNALED(status):
            sig = os.WTERMSIG(status)
            print("  run %d: pid %d killed by signal %d (%s) %.2f s after its "
                  "parent" % (i + 1, pid, sig, signal.Signals(sig).name, lived))
        else:
            print("  run %d: pid %d exited with code %d %.2f s after its parent"
                  % (i + 1, pid, os.WEXITSTATUS(status), lived))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="ffmpeg", choices=["ffmpeg", "ffprobe"])
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--delay", type=float, default=0.4,
                    help="seconds from the start of the child to the death of "
                         "its parent")
    ap.add_argument("--wait", type=float, default=12.0,
                    help="how long to wait for the orphan before it is killed")
    ap.add_argument("--when", action="store_true",
                    help="instead: measure when the child starts to ignore "
                         "SIGPIPE")
    args = ap.parse_args(argv)

    cmd = FFMPEG if args.bin == "ffmpeg" else FFPROBE
    print(" ".join(cmd))
    if args.when:
        measure_when(cmd, args.runs)
        return 0
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        print("prctl(PR_SET_CHILD_SUBREAPER) failed — Linux only")
        return 1
    print("parent dies %.2f s after the child starts, wait %.0f s"
          % (args.delay, args.wait))
    measure_fate(cmd, args.runs, args.delay, args.wait)
    return 0


if __name__ == "__main__":
    sys.exit(main())
