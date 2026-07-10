"""Media values: MediaRef, data: URL helpers, mime sniffing."""

import base64
import binascii
import os

from .errors import NanoodleError

# NanoGPT's edge (Vercel) rejects request bodies over ~4.5 MB; media rides inline
# as base64 data: URLs (there is no upload endpoint), so guard locally with a
# clear error. Compared against the STRING length of the inlined payload,
# mirroring play.html's MEDIA_INLINE_MAX.
MEDIA_INLINE_MAX = int(4.4 * 1024 * 1024)
# Transcribe uploads raw bytes (multipart) — its cap is on the byte size.
TRANSCRIBE_MAX_BYTES = int(3.5 * 1024 * 1024)

_EXT_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
    ".oga": "audio/ogg", ".opus": "audio/ogg", ".flac": "audio/flac",
    ".aac": "audio/aac", ".m4a": "audio/mp4",
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
    ".txt": "text/plain", ".json": "application/json",
}

_MIME_EXT = {
    "image/png": "png", "image/jpeg": "jpg", "image/gif": "gif",
    "image/webp": "webp", "image/bmp": "bmp",
    "audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/wav": "wav",
    "audio/x-wav": "wav", "audio/ogg": "ogg", "audio/flac": "flac",
    "audio/aac": "aac", "audio/mp4": "m4a",
    "video/mp4": "mp4", "video/webm": "webm", "video/quicktime": "mov",
    "text/plain": "txt", "application/json": "json",
}


def sniff_mime(data, default="application/octet-stream"):
    """Best-effort mime sniff from magic bytes (images + common audio/video)."""
    if not data:
        return default
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if len(data) > 11 and data[4:8] == b"ftyp":
        return "video/mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    return default


def b64_image_mime(b64):
    """Sniff an image mime from base64 magic prefixes (verbatim table from play.html)."""
    if b64.startswith("/9j/"):
        return "image/jpeg"
    if b64.startswith("iVBOR"):
        return "image/png"
    if b64.startswith("R0lG"):
        return "image/gif"
    if b64.startswith("UklGR"):
        return "image/webp"
    return "image/png"


def make_data_url(data, mime=None):
    if mime is None:
        mime = sniff_mime(data)
    return "data:%s;base64,%s" % (mime, base64.b64encode(data).decode("ascii"))


def parse_data_url(url):
    """data:[mime][;base64],payload -> (mime, bytes)."""
    if not url.startswith("data:"):
        raise NanoodleError("not a data: URL")
    head, _, payload = url.partition(",")
    meta = head[5:]
    mime = meta.split(";")[0] or "application/octet-stream"
    if ";base64" in meta:
        try:
            data = base64.b64decode(payload)
        except (binascii.Error, ValueError) as e:
            raise NanoodleError("invalid base64 in data: URL (%s)" % e)
    else:
        from urllib.parse import unquote
        data = unquote(payload).encode("utf-8")
    return mime, data


class MediaRef(object):
    """A media output value: a data: or https URL plus lazy byte access.

    ``str(ref)`` (and ``ref.url``) is the URL; ``ref.bytes()`` decodes/downloads
    the payload; ``ref.save(path)`` writes it to disk.
    """

    def __init__(self, url, mime=None, fetcher=None):
        self.url = url
        self._fetcher = fetcher  # callable(url) -> (bytes, content_type) for https
        if mime is None and url.startswith("data:"):
            head = url[5:].split(",", 1)[0]
            mime = head.split(";")[0] or None
        self.mime = mime

    def bytes(self):
        if self.url.startswith("data:"):
            _, data = parse_data_url(self.url)
            return data
        if self._fetcher is None:
            raise NanoodleError("no fetcher available to download %s" % self.url.split("?")[0])
        data, ctype = self._fetcher(self.url)
        if self.mime is None and ctype:
            self.mime = ctype.split(";")[0].strip() or None
        return data

    def save(self, path):
        data = self.bytes()
        with open(path, "wb") as f:
            f.write(data)
        return path

    def suggested_extension(self):
        mime = self.mime or (sniff_mime(self.bytes()) if self.url.startswith("data:") else None)
        return _MIME_EXT.get((mime or "").split(";")[0].strip().lower(), "bin")

    def __str__(self):
        return self.url

    def __repr__(self):
        u = self.url
        if len(u) > 64:
            u = u[:61] + "..."
        return "MediaRef(url=%r, mime=%r)" % (u, self.mime)

    def __eq__(self, other):
        if isinstance(other, MediaRef):
            return self.url == other.url
        if isinstance(other, str):
            return self.url == other
        return NotImplemented

    def __hash__(self):
        return hash(self.url)


def media_from_file(path, mime=None):
    """Read a local file into a MediaRef with an inline data: URL."""
    with open(path, "rb") as f:
        data = f.read()
    if mime is None:
        ext = os.path.splitext(path)[1].lower()
        mime = _EXT_MIME.get(ext) or sniff_mime(data)
    return MediaRef(make_data_url(data, mime), mime=mime)
