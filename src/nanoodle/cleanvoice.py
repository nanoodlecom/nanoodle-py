"""🎧 Clean voice — strip background noise and music, keep the voice.

Twin of the editor/play clean-voice block (nanoodle #709). NanoGPT's
noise_reduction models (ElevenLabs Audio Isolation, VEED Clean Audio) take
POST /api/v1/audio/speech with ``{model, input, audio:<public URL>, duration}``
and answer 202 + runId, then /api/tts/status returns a hosted audio file.

The source MUST be a public http(s) URL. data: and blob: clips are refused
before any request: NanoGPT downloads the file itself and rejects inline
bytes, and this library has no file host. ``duration`` is only a quote; the
server measures the file and bills that.
"""

import math
import re
import shutil
import subprocess

from .errors import NanoodleError

CLEANVOICE_DEFAULT_MODEL = "elevenlabs/audio-isolation"
CLEANVOICE_FALLBACK_SECS = 60
# Estimate when the wired source has no duration knob. The run's own quote
# falls back to CLEANVOICE_FALLBACK_SECS when the file's length can't be read.
CLEANVOICE_EST_SECONDS = 30
_VIDEO_EXT = re.compile(r"\.(mp4|m4v|mov|webm|mkv|avi)(\?|#|$)", re.I)
_HTTP = re.compile(r"^https?://", re.I)


def _text(v):
    if v is None:
        return ""
    url = getattr(v, "url", None)
    if isinstance(url, str):
        return url.strip()
    if isinstance(v, str):
        return v.strip()
    return ""


def clean_voice_source(inp, fields):
    """Wired audio, else wired video, else the pasted public link.

    Returns ``{"url", "video"}``. Raises before any request when nothing is
    hosted.
    """
    inp = inp or {}
    audio = _text(inp.get("audio"))
    video = _text(inp.get("video"))
    link = _text((fields or {}).get("url"))
    url = audio or video or link
    is_video = False if audio else (True if video else bool(_VIDEO_EXT.search(link)))
    if not url:
        raise NanoodleError(
            "no audio — wire audio or a video in, or paste a public link to the recording")
    if not _HTTP.match(url):
        raise NanoodleError(
            "Clean voice needs a hosted file — NanoGPT downloads it from a public link, "
            "and a clip uploaded or made in your browser has none. Wire a generated video "
            "or track, or paste a public https link to the recording.")
    return {"url": url, "video": is_video}


def clean_voice_seconds(secs):
    """Duration quote NanoGPT requires (0.6–3600 s). Unknown → 60 s."""
    try:
        s = float(secs)
    except (TypeError, ValueError):
        return CLEANVOICE_FALLBACK_SECS
    if not math.isfinite(s) or s <= 0:
        return CLEANVOICE_FALLBACK_SECS
    rounded = int(math.floor(s * 100 + 0.5)) / 100.0
    return min(3600.0, max(0.6, rounded))


def clean_voice_extra(url, secs):
    return {"audio": url, "duration": clean_voice_seconds(secs)}


def clean_voice_est_seconds(src_fields):
    """Seconds the run-cost forecast assumes: the wired node's duration, else 30."""
    try:
        d = float((src_fields or {}).get("duration"))
    except (TypeError, ValueError):
        return float(CLEANVOICE_EST_SECONDS)
    return d if math.isfinite(d) and d > 0 else float(CLEANVOICE_EST_SECONDS)


def clean_voice_error(exc):
    """Reword NanoGPT's 400s. Anything else, including a cancelled run, passes through."""
    if getattr(exc, "code", None) == "cancelled" or type(exc).__name__ == "NodeCancelled":
        return exc
    m = str(exc or "")
    if re.search(r"verify the source (audio )?duration", m, re.I):
        return NanoodleError(
            "NanoGPT couldn't read this file's length — use MP3, WAV, M4A/AAC or MP4 "
            "(OGG isn't accepted).")
    if re.search(r"unable to download", m, re.I):
        return NanoodleError(
            "NanoGPT couldn't download the file — the link must be public "
            "(no sign-in, not expired).")
    if re.search(r"public http\(?s?\)? source", m, re.I):
        return NanoodleError(
            "Clean voice needs a public https link — clips uploaded in the browser can't reach it.")
    return exc


def clean_voice_media_seconds(url):
    """Best-effort length via ffprobe. None when it can't be read (→ 60 s quote).

    This is not a NanoGPT call. A missing ffprobe, a timeout, or an unreadable
    URL all return None; the server re-measures before billing.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not url:
        return None
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", url],
            capture_output=True, text=True, timeout=3)
        d = float((proc.stdout or "").strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return d if math.isfinite(d) and d > 0 else None
