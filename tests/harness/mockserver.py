"""Mock NanoGPT server for offline tests.

- Runs an http.server.ThreadingHTTPServer on 127.0.0.1:<random port> in a thread.
- Canned per-endpoint responses, scriptable SEQUENCES (e.g. video pending ->
  pending -> completed).
- Records EVERY request (method, path incl. query, headers, raw body, parsed
  JSON) for payload assertions.

Usage:
    mock = MockNanoGPT()
    mock.script("POST", "/api/v1/chat/completions", [chat_response("hello")])
    mock.start()
    wf = Workflow.from_dict(graph, api_key="test-key", base_url=mock.base_url, ...)
    ...
    mock.stop()

A response spec is a dict:
    {"status": 200, "json": {...}}                      # JSON body
    {"status": 200, "body": b"...", "headers": {...}}   # raw body
    {"delay": 0.3, ...}                                 # sleep before replying
When a script queue runs out, its LAST response repeats.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


def chat_response(text, cost_usd=None, balance=None, images=None, reasoning=None,
                  content_parts=None):
    msg = {"role": "assistant", "content": text}
    if content_parts is not None:
        msg["content"] = content_parts
    if images is not None:
        msg["images"] = images
    if reasoning is not None:
        msg["reasoning"] = reasoning
    j = {"id": "chatcmpl-mock", "choices": [{"index": 0, "message": msg,
                                             "finish_reason": "stop"}]}
    pricing = {}
    if cost_usd is not None:
        pricing["costUsd"] = cost_usd
    if balance is not None:
        pricing["remainingBalance"] = balance
    if pricing:
        j["x_nanogpt_pricing"] = pricing
    return {"status": 200, "json": j}


def image_response(b64_list=None, urls=None, cost=None, balance=None):
    data = []
    for b in (b64_list or []):
        data.append({"b64_json": b})
    for u in (urls or []):
        data.append({"url": u})
    j = {"created": 0, "data": data}
    if cost is not None:
        j["cost"] = cost
    if balance is not None:
        j["remainingBalance"] = balance
    return {"status": 200, "json": j}


class _Recorded(object):
    __slots__ = ("method", "path", "query", "headers", "body", "json", "time")

    def __init__(self, method, path, query, headers, body, when):
        self.method = method
        self.path = path            # path WITHOUT query
        self.query = query          # raw query string ("" when none)
        self.headers = headers      # dict, lower-cased keys
        self.body = body            # raw bytes
        self.time = when
        try:
            self.json = json.loads(body.decode("utf-8")) if body else None
        except (ValueError, UnicodeDecodeError):
            self.json = None

    def __repr__(self):
        return "<%s %s%s>" % (self.method, self.path, ("?" + self.query) if self.query else "")


class MockNanoGPT(object):
    def __init__(self):
        self.requests = []
        self._scripts = {}   # (METHOD, path) -> list of response specs
        self._lock = threading.Lock()
        self._server = None
        self._thread = None
        self.max_concurrent = 0
        self._in_flight = 0

    # -- scripting ------------------------------------------------------------

    def script(self, method, path, responses):
        """Queue responses for METHOD path (query ignored for matching)."""
        if isinstance(responses, dict):
            responses = [responses]
        self._scripts[(method.upper(), path)] = list(responses)

    def requests_to(self, path, method=None):
        return [r for r in self.requests
                if r.path == path and (method is None or r.method == method.upper())]

    def reset(self):
        with self._lock:
            self.requests = []
            self.max_concurrent = 0

    # -- lifecycle --------------------------------------------------------------

    @property
    def base_url(self):
        host, port = self._server.server_address[:2]
        return "http://%s:%d" % (host, port)

    def start(self):
        mock = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _handle(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                parsed = urlparse(self.path)
                rec = _Recorded(method, parsed.path, parsed.query,
                                {k.lower(): v for k, v in self.headers.items()},
                                body, time.monotonic())
                with mock._lock:
                    mock.requests.append(rec)
                    mock._in_flight += 1
                    mock.max_concurrent = max(mock.max_concurrent, mock._in_flight)
                    queue = mock._scripts.get((method, parsed.path))
                    if queue:
                        spec = queue.pop(0) if len(queue) > 1 else queue[0]
                    else:
                        spec = None
                try:
                    if spec is None:
                        payload = json.dumps({"error": "no script for %s %s"
                                              % (method, parsed.path)}).encode("utf-8")
                        self._reply(404, {"Content-Type": "application/json"}, payload)
                        return
                    if spec.get("delay"):
                        time.sleep(spec["delay"])
                    headers = dict(spec.get("headers") or {})
                    if "json" in spec:
                        payload = json.dumps(spec["json"]).encode("utf-8")
                        headers.setdefault("Content-Type", "application/json")
                    else:
                        payload = spec.get("body", b"")
                        if isinstance(payload, str):
                            payload = payload.encode("utf-8")
                    self._reply(spec.get("status", 200), headers, payload)
                finally:
                    with mock._lock:
                        mock._in_flight -= 1

            def _reply(self, status, headers, payload):
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=lambda: self._server.serve_forever(poll_interval=0.02), daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # context manager sugar
    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
