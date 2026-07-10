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

from .errors import NanoodleError


class HttpResponse(object):
    __slots__ = ("status", "headers", "body")

    def __init__(self, status, headers, body):
        self.status = status
        self.headers = {str(k).lower(): v for k, v in (headers or {}).items()}
        self.body = body if body is not None else b""

    def header(self, name):
        return self.headers.get(str(name).lower())

    def text(self):
        try:
            return self.body.decode("utf-8")
        except UnicodeDecodeError:
            return self.body.decode("utf-8", "replace")


def default_http(method, url, headers=None, body=None, timeout=None):
    if isinstance(body, str):
        body = body.encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return HttpResponse(resp.status, dict(resp.headers.items()), resp.read())
    except urllib.error.HTTPError as e:
        return HttpResponse(e.code, dict((e.headers or {}).items()), e.read())
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
