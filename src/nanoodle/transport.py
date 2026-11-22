"""HTTP transport: a tiny injectable urllib wrapper + multipart encoder.

The transport is a callable:
    http(method, url, headers=None, body=None, timeout=None) -> HttpResponse
Non-2xx responses are RETURNED (status + body), never raised — the engine owns
error mapping. Only genuine connection failures raise NanoodleError.
"""

import os
import urllib.error
import urllib.request
import uuid
from http.client import IncompleteRead

from .errors import NanoodleError


class HttpBodyReadError(NanoodleError):
    """The response headers arrived, but its body could not be read."""


class HttpResponse(object):
    __slots__ = ("_body", "_body_error", "headers", "status")

    def __init__(self, status, headers, body, *, body_error=None):
        self.status = status
        self.headers = {str(k).lower(): v for k, v in (headers or {}).items()}
        self._body = body if body is not None else b""
        self._body_error = body_error

    @property
    def body(self):
        if self._body_error is not None:
            raise self._body_error
        return self._body

    def header(self, name):
        return self.headers.get(str(name).lower())

    def text(self):
        try:
            return self.body.decode("utf-8")
        except UnicodeDecodeError:
            return self.body.decode("utf-8", "replace")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _read_response(resp, max_bytes):
    headers = dict((resp.headers or {}).items())
    try:
        data = resp.read() if max_bytes is None else resp.read(max_bytes + 1)
    except (IncompleteRead, OSError):
        # A failed read can still carry the provider's charge in the headers.
        # Do not pass incomplete content off as a usable JSON or media body.
        return HttpResponse(resp.code, headers, None, body_error=HttpBodyReadError(
            "could not read HTTP response body"))
    return HttpResponse(resp.code, headers, data)


def default_http(method, url, headers=None, body=None, timeout=None,
                 follow_redirects=True, max_bytes=None):
    if isinstance(body, str):
        body = body.encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        open_url = (urllib.request.urlopen if follow_redirects else
                    urllib.request.build_opener(_NoRedirect()).open)
        with open_url(req, timeout=timeout) as resp:
            return _read_response(resp, max_bytes)
    except urllib.error.HTTPError as e:
        with e:
            return _read_response(e, max_bytes)
    except urllib.error.URLError as e:
        # never echo headers/keys — the reason string is connection-level only
        raise NanoodleError("could not reach %s (%s)" % (url.split("?")[0], e.reason))
    except OSError as e:
        raise NanoodleError("could not reach %s (%s)" % (url.split("?")[0], e))


def encode_multipart(fields, file_field, filename, file_bytes, file_mime):
    """Build a multipart/form-data body. Returns (content_type, body_bytes).

    ``fields`` are plain text form fields; the file part is named ``file_field``
    (the NanoGPT transcription endpoint REQUIRES the name "file")."""
    boundary = "----nanoodle" + uuid.uuid4().hex
    lines = []
    for name, value in fields.items():
        lines.append(b"--" + boundary.encode("ascii"))
        lines.append(('Content-Disposition: form-data; name="%s"' % name).encode("utf-8"))
        lines.append(b"")
        lines.append(str(value).encode("utf-8"))
    lines.append(b"--" + boundary.encode("ascii"))
    lines.append(('Content-Disposition: form-data; name="%s"; filename="%s"'
                  % (file_field, filename)).encode("utf-8"))
    lines.append(("Content-Type: %s" % (file_mime or "application/octet-stream")).encode("utf-8"))
    lines.append(b"")
    lines.append(file_bytes)
    lines.append(b"--" + boundary.encode("ascii") + b"--")
    lines.append(b"")
    body = b"\r\n".join(lines)
    return "multipart/form-data; boundary=%s" % boundary, body


def resolve_api_key(api_key):
    return api_key if api_key is not None else os.environ.get("NANOGPT_API_KEY")
