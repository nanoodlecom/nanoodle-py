"""Local media ops (resize / vframes / combine / soundtrack / trim / extractaudio).

Headless path shells out to ffmpeg/ffprobe on PATH — a soft dependency (not a
PyPI package), matching nanoodle-js. Behaviour mirrors the browser (resizePlan,
trim defaults, vframes seek math, concat, soundtrack mux). Outputs are data:
URLs for the MediaRef pipeline.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time

from .errors import NanoodleError
from .media import MEDIA_INLINE_MAX, make_data_url, parse_data_url, sniff_mime

MAX_FRAMES = 12


def _effective_timeout(default, deadline=None):
    """Clamp a per-process timeout to the remaining workflow deadline (if any)."""
    if deadline is None:
        return default
    rem = deadline - time.monotonic()
    if rem <= 0:
        return 0
    return min(default, rem)


def _run(bin_name, args, timeout=120, cancel_check=None, deadline=None):
    if cancel_check:
        cancel_check()
    if not shutil.which(bin_name):
        raise NanoodleError(
            "local media nodes need ffmpeg on PATH (not found: %s). "
            "Install ffmpeg, or run this graph in the nanoodle browser app." % bin_name)
    eff = _effective_timeout(timeout, deadline)
    if eff <= 0:
        if cancel_check:
            cancel_check()
        raise NanoodleError("%s timed out after %ss" % (bin_name, timeout))
    try:
        # Use Popen so a mid-run cancel/deadline can kill the child promptly
        # instead of waiting for the full process timeout.
        p = subprocess.Popen(
            [bin_name] + list(args),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise NanoodleError(
            "local media nodes need ffmpeg on PATH (not found: %s). "
            "Install ffmpeg, or run this graph in the nanoodle browser app." % bin_name)
    end = time.monotonic() + eff
    stdout = stderr = None
    try:
        while True:
            try:
                remaining = max(0.05, end - time.monotonic())
                stdout, stderr = p.communicate(timeout=min(0.25, remaining))
                break
            except subprocess.TimeoutExpired:
                if cancel_check:
                    try:
                        cancel_check()
                    except Exception:
                        p.kill()
                        try:
                            p.communicate(timeout=2)
                        except Exception:
                            pass
                        raise
                if time.monotonic() >= end:
                    p.kill()
                    try:
                        p.communicate(timeout=2)
                    except Exception:
                        pass
                    raise NanoodleError("%s timed out after %ss" % (bin_name, timeout))
    finally:
        if p.poll() is None:
            p.kill()
            try:
                p.communicate(timeout=2)
            except Exception:
                pass
    if p.returncode != 0:
        err = (stderr or b"").decode("utf-8", "replace").strip()
        raise NanoodleError("%s failed (exit %s): %s" % (bin_name, p.returncode, err[-400:] or "no stderr"))
    return stdout, (stderr or b"").decode("utf-8", "replace")


def _as_url(url):
    if url is None:
        raise NanoodleError("no media input")
    if hasattr(url, "url"):
        return url.url
    return str(url)


def _url_bytes(url, fetcher=None):
    u = _as_url(url)
    if u.startswith("data:"):
        _mime, data = parse_data_url(u)
        return data
    if re.match(r"^https?://", u, re.I):
        if fetcher is None:
            raise NanoodleError("can't download media: no fetcher available")
        return fetcher(u)
    raise NanoodleError("media must be a data: or http(s) URL")


def _write_input(dirpath, name, url, fetcher=None):
    raw = _url_bytes(url, fetcher)
    mime = sniff_mime(raw)
    if "png" in mime:
        ext = ".png"
    elif "jpeg" in mime:
        ext = ".jpg"
    elif "webp" in mime:
        ext = ".webp"
    elif "gif" in mime:
        ext = ".gif"
    elif "wav" in mime:
        ext = ".wav"
    elif "mpeg" in mime or "mp3" in mime:
        ext = ".mp3"
    elif "mp4" in mime:
        ext = ".mp4"
    elif "webm" in mime:
        ext = ".webm"
    else:
        m = re.search(r"\.([a-z0-9]{2,5})(?:\?|$)", _as_url(url), re.I)
        ext = ("." + m.group(1).lower()) if m else ".bin"
    path = os.path.join(dirpath, name + ext)
    with open(path, "wb") as f:
        f.write(raw)
    return path


def _data_url_from_file(path, mime_hint=None):
    with open(path, "rb") as f:
        raw = f.read()
    mime = mime_hint or sniff_mime(raw)
    return make_data_url(raw, mime)


# ---------- resizePlan (verbatim from index.html) ----------------------------

def resize_plan(sw, sh, mode, tw, th):
    if not (tw > 0) and not (th > 0):
        return None
    if mode == "fit":
        if tw > 0 and th > 0:
            scale = min(tw / sw, th / sh)
        elif tw > 0:
            scale = tw / sw
        else:
            scale = th / sh
        if scale > 1:
            scale = 1
        w = max(1, int(round(sw * scale)))
        h = max(1, int(round(sh * scale)))
        return {"cw": w, "ch": h, "dx": 0, "dy": 0, "dw": w, "dh": h}
    bw = tw if tw > 0 else max(1, int(round(th * sw / sh)))
    bh = th if th > 0 else max(1, int(round(tw * sh / sw)))
    if mode == "exact":
        return {"cw": bw, "ch": bh, "dx": 0, "dy": 0, "dw": bw, "dh": bh}
    scale = max(bw / sw, bh / sh)
    dw, dh = sw * scale, sh * scale
    return {"cw": bw, "ch": bh, "dx": (bw - dw) / 2, "dy": (bh - dh) / 2, "dw": dw, "dh": dh}


def _run_args(cancel_check=None, deadline=None):
    return {"cancel_check": cancel_check, "deadline": deadline}


def resize_crop_image(url, mode, tw, th, fetcher=None, cancel_check=None, deadline=None):
    try:
        w = max(0, int(float(tw))) if tw not in (None, "") else 0
    except (TypeError, ValueError):
        w = 0
    try:
        h = max(0, int(float(th))) if th not in (None, "") else 0
    except (TypeError, ValueError):
        h = 0
    if not w and not h:
        raise NanoodleError("set a width or height to resize to")
    if cancel_check:
        cancel_check()
    m = mode or "fit"
    ra = _run_args(cancel_check, deadline)
    with tempfile.TemporaryDirectory(prefix="nanoodle-media-") as d:
        in_path = _write_input(d, "in", url, fetcher)
        out, _ = _run("ffprobe", [
            "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", in_path], **ra)
        parts = out.decode("utf-8", "replace").strip().split("x")
        try:
            sw, sh = int(parts[0]), int(parts[1])
        except (IndexError, ValueError):
            raise NanoodleError("couldn't read that image to resize")
        plan = resize_plan(sw, sh, m, w, h)
        if not plan:
            raise NanoodleError("set a width or height to resize to")
        src = _as_url(url)
        want_png = src.lower().startswith("data:image/png") or in_path.lower().endswith(".png")
        out_path = os.path.join(d, "out.png" if want_png else "out.jpg")
        if m in ("fit", "exact"):
            vf = "scale=%d:%d" % (plan["cw"], plan["ch"])
        else:
            vf = "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d" % (
                plan["cw"], plan["ch"], plan["cw"], plan["ch"])
        args = ["-y", "-i", in_path, "-vf", vf, "-frames:v", "1"]
        if not want_png:
            args += ["-q:v", "2"]
        args.append(out_path)
        _run("ffmpeg", args, **ra)
        result = _data_url_from_file(out_path, "image/png" if want_png else "image/jpeg")
        if len(result) > MEDIA_INLINE_MAX:
            raise NanoodleError(
                "resized image is still over the ~4 MB inline limit — pick smaller dimensions")
        return result


def trim_audio_to_wav(url, start, length, rate=16000, fetcher=None, whole_if_blank=False,
                      cancel_check=None, deadline=None):
    if cancel_check:
        cancel_check()
    ra = _run_args(cancel_check, deadline)
    with tempfile.TemporaryDirectory(prefix="nanoodle-media-") as d:
        in_path = _write_input(d, "in", url, fetcher)
        out_path = os.path.join(d, "out.wav")
        s = max(0.0, float(start or 0))
        dur = None
        try:
            out, _ = _run("ffprobe", [
                "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", in_path], **ra)
            dur = float(out.decode("utf-8", "replace").strip())
        except (NanoodleError, ValueError):
            pass
        if dur is not None and s >= dur:
            raise NanoodleError(
                "the start point (%ss) is past the end of this clip, which is only %ss long — pick an earlier start"
                % (round(s * 10) / 10, "%.1f" % dur))
        take = None
        if whole_if_blank and not (length and float(length) > 0):
            take = max(0.05, dur - s) if dur is not None else None
        else:
            L = float(length) if length not in (None, "") and float(length) > 0 else 30.0
            take = max(0.05, min(L, dur - s)) if dur is not None else L
        args = ["-y", "-ss", str(s), "-i", in_path]
        if take is not None:
            args += ["-t", str(take)]
        args += ["-vn", "-ac", "1", "-ar", str(rate or 16000), "-f", "wav", out_path]
        try:
            _run("ffmpeg", args, **ra)
        except NanoodleError as e:
            msg = str(e)
            if re.search(r"does not contain any stream|no audio|matches no streams", msg, re.I):
                raise NanoodleError(
                    "this video is silent — generated videos usually have no audio track to extract")
            if re.search(r"Invalid data|could not find codec", msg, re.I):
                raise NanoodleError("couldn't decode that audio for trimming (unsupported format?)")
            raise
        return _data_url_from_file(out_path, "audio/wav")


def extract_audio_to_wav(url, start, length, rate=16000, fetcher=None,
                         cancel_check=None, deadline=None):
    return trim_audio_to_wav(url, start, length, rate, fetcher=fetcher,
                             whole_if_blank=True, cancel_check=cancel_check,
                             deadline=deadline)


def extract_video_frames(url, count=1, gap=0.5, dir="end", fetcher=None,
                         cancel_check=None, deadline=None):
    try:
        n = max(1, min(MAX_FRAMES, int(count or 1)))
    except (TypeError, ValueError):
        n = 1
    try:
        step = max(0.0, float(gap))
    except (TypeError, ValueError):
        step = 0.5
    from_end = (dir or "end") == "end"
    eps = 0.04
    if cancel_check:
        cancel_check()
    ra = _run_args(cancel_check, deadline)
    with tempfile.TemporaryDirectory(prefix="nanoodle-media-") as d:
        in_path = _write_input(d, "in", url, fetcher)
        out, _ = _run("ffprobe", [
            "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", in_path], **ra)
        try:
            dur = float(out.decode("utf-8", "replace").strip())
        except ValueError:
            raise NanoodleError("video has no readable duration")
        if not (dur > 0):
            raise NanoodleError("video has no readable duration")
        result = {}
        for i in range(n):
            # Between frames: honour workflow timeout/cancel so a long vframes
            # loop does not run past the outer deadline after paid upstream work.
            if cancel_check:
                cancel_check()
            t = (dur - eps - i * step) if from_end else (i * step)
            t = max(0.0, min(max(0.0, dur - eps), t))
            frame_path = os.path.join(d, "f%d.jpg" % (i + 1))
            _run("ffmpeg", [
                "-y", "-ss", str(t), "-i", in_path, "-frames:v", "1", "-q:v", "2", frame_path], **ra)
            result["frame%d" % (i + 1)] = _data_url_from_file(frame_path, "image/jpeg")
        return result


def concat_videos(urls, dedup=True, fetcher=None, cancel_check=None, deadline=None):
    if not urls or len(urls) < 2:
        raise NanoodleError("wire at least two clips to combine")
    if cancel_check:
        cancel_check()
    ra = _run_args(cancel_check, deadline)
    with tempfile.TemporaryDirectory(prefix="nanoodle-media-") as d:
        paths = [_write_input(d, "c%d" % i, u, fetcher) for i, u in enumerate(urls)]
        prepared = []
        for i, p in enumerate(paths):
            if cancel_check:
                cancel_check()
            if dedup and i > 0:
                trimmed = os.path.join(d, "t%d.mp4" % i)
                _run("ffmpeg", [
                    "-y", "-ss", "0.033", "-i", p,
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", trimmed], **ra)
                prepared.append(trimmed)
            else:
                prepared.append(p)
        list_path = os.path.join(d, "list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in prepared:
                f.write("file '%s'\n" % p.replace("'", "'\\''"))
        out_path = os.path.join(d, "out.mp4")
        try:
            _run("ffmpeg", [
                "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                "-c", "copy", "-movflags", "+faststart", out_path], **ra)
        except NanoodleError:
            _run("ffmpeg", [
                "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out_path], **ra)
        return _data_url_from_file(out_path, "video/mp4")


def mux_soundtrack(video_url, audio_url, loop=False, fetcher=None,
                   cancel_check=None, deadline=None):
    if cancel_check:
        cancel_check()
    ra = _run_args(cancel_check, deadline)
    with tempfile.TemporaryDirectory(prefix="nanoodle-media-") as d:
        v_path = _write_input(d, "v", video_url, fetcher)
        a_path = _write_input(d, "a", audio_url, fetcher)
        out_path = os.path.join(d, "out.mp4")
        vdur = None
        try:
            out, _ = _run("ffprobe", [
                "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", v_path], **ra)
            vdur = float(out.decode("utf-8", "replace").strip())
        except (NanoodleError, ValueError):
            pass
        args = ["-y", "-i", v_path]
        if loop:
            args += ["-stream_loop", "-1"]
        args += ["-i", a_path, "-map", "0:v:0", "-map", "1:a:0",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "128k"]
        if loop and vdur is not None:
            args += ["-t", str(vdur)]
        else:
            args.append("-shortest")
        args += ["-movflags", "+faststart", out_path]
        try:
            _run("ffmpeg", args, **ra)
        except NanoodleError:
            args2 = ["-y", "-i", v_path]
            if loop:
                args2 += ["-stream_loop", "-1"]
            args2 += ["-i", a_path, "-map", "0:v:0", "-map", "1:a:0",
                      "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                      "-c:a", "aac", "-b:a", "128k"]
            if loop and vdur is not None:
                args2 += ["-t", str(vdur)]
            else:
                args2.append("-shortest")
            args2 += ["-movflags", "+faststart", out_path]
            _run("ffmpeg", args2, **ra)
        return _data_url_from_file(out_path, "video/mp4")
