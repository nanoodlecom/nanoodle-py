"""Decode-only codec for nanoodle share links — the editor stays the single
encoder of record; these functions only ever read.

Wire formats (mirrors index.html's loadFromHash / buildShareUrl and the
nanoodle-js src/share.mjs twin, locked by the golden fixtures in
tests/fixtures/share/ — those are minted by a REAL editor and shared
byte-identically between the JS and Python libraries):

    #g=<b64url(gzip(graph JSON))>          workflow link (editor 🔗 Share)
    #j=<b64url(graph JSON)>                uncompressed fallback (no CompressionStream)
    #a=<b64url(gzip(app payload))>         app link (play.html); payload = { v, graph, files?, name?, lang?, ... }
    #a=u<b64url(app payload)>              uncompressed app fallback ('u' tag inside the value)
    #ga=…                                  editor↔play handoff — internal transport, deliberately NOT supported

Stdlib only (zlib for gzip, base64 for base64url, urllib for the by-hand
redirect reads on short links). Direct fragment links decode with zero network
I/O; only fragment-less http(s) URLs (short links) ever touch the network.
"""

import base64
import binascii
import re
import urllib.error
import urllib.request
import zlib
from urllib.parse import urljoin

from .errors import NanoodleError

_URL_RE = re.compile(r"^https?://", re.I)
_FRAG_RE = re.compile(r"^#?(ga|[gja])=")
_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def is_share_ref(s):
    """True when a string is addressable as a share link: an http(s) URL, or a
    bare #g=/#j=/#a= (or tail g=/j=/a=) fragment."""
    return isinstance(s, str) and bool(_URL_RE.match(s) or _FRAG_RE.match(s))


def _b64url_to_bytes(s, what):
    if not _B64URL_RE.match(s):
        raise NanoodleError(
            "share link: %s payload is not base64url data — is the URL complete?" % what)
    # restore the '=' padding urlsafe_b64decode requires (share links drop it);
    # a truncated link can still be an impossible base64 length → friendly error
    try:
        return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    except (binascii.Error, ValueError):
        raise NanoodleError(
            "share link: %s payload is not base64url data — the link may be truncated" % what)


def _parse_json(text, what):
    import json
    try:
        return json.loads(text)
    except ValueError:
        raise NanoodleError(
            "share link: %s payload decoded but is not valid JSON — the link may be truncated"
            % what)


def _gunzip_text(buf, what):
    try:
        # wbits 16 + MAX_WBITS decodes a gzip container (matches the editor's
        # CompressionStream('gzip') and nanoodle-js's gunzipSync)
        return zlib.decompress(buf, 16 + zlib.MAX_WBITS).decode("utf-8")
    except (zlib.error, UnicodeDecodeError):
        raise NanoodleError(
            "share link: %s payload is not valid gzip data — the link may be truncated" % what)


def decode_share_fragment(fragment):
    """Decode a share fragment ("#g=…", "g=…", "#a=…", …) to its graph.

    Returns a dict ``{"graph": dict, "kind": "g"|"j"|"a", "app": dict|None}``.
    ``app`` is present only for #a= links: ``{name?, lang?, has_files: bool}``
    (files/samples/lang are play.html presentation — executors run graphs, not
    apps).
    """
    f = str(fragment)
    if f.startswith("#"):
        f = f[1:]
    if f.startswith("ga="):
        raise NanoodleError(
            "share link: #ga= is the editor↔app-builder handoff — an internal, unstable "
            "format. Open the link in a browser and use 🔗 Share to mint a #g= workflow "
            "link instead.")
    if f.startswith("g="):
        graph = _parse_json(_gunzip_text(_b64url_to_bytes(f[2:], "#g="), "#g="), "#g=")
        return {"graph": graph, "kind": "g", "app": None}
    if f.startswith("j="):
        graph = _parse_json(_b64url_to_bytes(f[2:], "#j=").decode("utf-8"), "#j=")
        return {"graph": graph, "kind": "j", "app": None}
    if f.startswith("a="):
        tag = f[2:]
        if tag[:1] == "u":
            json_text = _b64url_to_bytes(tag[1:], "#a=u").decode("utf-8")
        else:
            json_text = _gunzip_text(_b64url_to_bytes(tag, "#a="), "#a=")
        payload = _parse_json(json_text, "#a=")
        if not isinstance(payload, dict) or not payload.get("graph"):
            raise NanoodleError("share link: #a= app payload has no graph in it")
        app = {"has_files": bool(payload.get("files"))}
        name = payload.get("name")
        if isinstance(name, str) and name:
            app["name"] = name
        lang = payload.get("lang")
        if isinstance(lang, str) and lang:
            app["lang"] = lang
        return {"graph": payload["graph"], "kind": "a", "app": app}
    raise NanoodleError(
        'share link: no #g=/#j=/#a= fragment found in "%s"' % fragment)


def _fragment_of(url):
    i = url.find("#")
    return None if i == -1 else url[i:]


def _default_opener(url):
    """Read one redirect hop without following it (fragments ride the Location
    header, so we must read it by hand) and without attaching any credentials.

    Returns ``(status, location_or_none)``.
    """
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None  # surface the 3xx to us instead of auto-following

    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, method="GET")
    try:
        with opener.open(req, timeout=30) as resp:
            return resp.status, resp.headers.get("location")
    except urllib.error.HTTPError as e:
        return e.code, (e.headers.get("location") if e.headers else None)
    except urllib.error.URLError as e:
        raise NanoodleError("share link: could not resolve %s: %s" % (url, e.reason))
    except OSError as e:
        raise NanoodleError("share link: could not resolve %s: %s" % (url, e))


def decode_share_url(ref, opener=None, max_hops=5):
    """Decode any nanoodle share reference — a full URL, a bare fragment, or a
    shortener link (da.gd/TinyURL/…) whose redirect target carries the fragment.

    Direct fragment links decode with ZERO network calls. Only fragment-less
    http(s) URLs trigger reads, and those are redirect-header reads with no
    credentials attached (the codec never sees an API key by construction).

    ``opener`` is an injectable hook ``opener(url) -> (status, location)`` used
    for the redirect reads (tests stub it; production uses urllib). Returns a
    dict ``{"graph", "kind", "app", "url"}``.
    """
    s = str(ref).strip()
    if not _URL_RE.match(s):
        result = decode_share_fragment(s)
        result["url"] = s
        return result

    url = s
    frag = _fragment_of(url)
    if frag and _FRAG_RE.match(frag):
        result = decode_share_fragment(frag)
        result["url"] = url
        return result

    # No fragment on the URL itself → treat it as a short link and follow
    # redirects by hand: fragments ride in the Location header, which automatic
    # redirect handling would consume before we could read it.
    fetch = opener or _default_opener
    for _ in range(max_hops):
        status, loc = fetch(url)
        loc = loc if (loc and 300 <= status < 400) else None
        if not loc:
            raise NanoodleError(
                "share link: %s answered %s with no #g=/#j=/#a= fragment and no redirect — "
                "open it in a browser and share the long nanoodle.com URL instead"
                % (url, status))
        url = urljoin(url, loc)
        hop_frag = _fragment_of(url)
        if hop_frag and _FRAG_RE.match(hop_frag):
            result = decode_share_fragment(hop_frag)
            result["url"] = url
            return result
    raise NanoodleError(
        "share link: gave up after %d redirects without finding a share fragment" % max_hops)
