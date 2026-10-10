"""🔌 Custom endpoint — POST a NanoGPT-shaped body to a URL the graph owns.

Twin of the editor/play endpoint block. The request goes straight to that URL.
It never uses the NanoGPT API key, never adds x-api-key, and never settles x402.
Custom auth is only ``fields.auth``.

http is allowed for localhost, 127.0.0.1, ::1, RFC1918, link-local, and
``.local``. Everything else must be https. Credentials in the URL are refused;
they belong in the Authorization field.
"""

import json
import re
from urllib.parse import urlparse, urlunparse

from .errors import NanoodleError
from .media import MediaRef, b64_image_mime, make_data_url

ENDPOINT_DEF_URL = "http://127.0.0.1:8787/v1/chat/completions"
ENDPOINT_MODES = ("chat", "image", "video", "audio", "json")


def endpoint_mode(node):
    fields = getattr(node, "fields", None) or {}
    m = str(fields.get("mode") or "chat").lower()
    return m if m in ENDPOINT_MODES else "chat"


def endpoint_out_port(node):
    m = endpoint_mode(node)
    if m == "image":
        return ("image", "image")
    if m == "video":
        return ("video", "video")
    if m == "audio":
        return ("audio", "audio")
    return ("text", "text")


def _loopback(host):
    h = str(host or "").lower().strip("[]")
    return h in ("localhost", "127.0.0.1", "::1")


def _private_ipv4(host):
    parts = str(host or "").split(".")
    if len(parts) != 4:
        return False
    try:
        nums = [int(x) for x in parts]
    except ValueError:
        return False
    if any(n < 0 or n > 255 for n in nums):
        return False
    a, b = nums[0], nums[1]
    if a == 10:
        return True
    if a == 192 and b == 168:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    if a == 169 and b == 254:
        return True
    return False


def endpoint_url_ok(url):
    """True, or an error string. Mirrors play.html endpointUrlOk."""
    s = str(url or "").strip()
    if not s:
        return "URL required — set the custom endpoint URL"
    try:
        u = urlparse(s)
    except ValueError:
        u = None
    if u is None or not u.scheme:
        return ("that URL isn’t allowed — use http://localhost, 127.0.0.1, a LAN host, or https")
    if u.scheme not in ("http", "https"):
        return ("that URL isn’t allowed — use http://localhost, 127.0.0.1, a LAN host, or https")
    if u.username or u.password:
        return ("that URL isn’t allowed — don’t put credentials in the URL; "
                "use the Authorization field")
    if u.scheme == "https":
        return True
    host = u.hostname or ""
    if _loopback(host) or _private_ipv4(host) or host.lower().endswith(".local"):
        return True
    return ("that URL isn’t allowed — http is only for localhost, 127.0.0.1, or a LAN host; "
            "use https for a public host")


def endpoint_url_is_local(url):
    try:
        host = urlparse(str(url or "").strip()).hostname or ""
    except ValueError:
        return False
    return _loopback(host) or _private_ipv4(host) or host.lower().endswith(".local")


def endpoint_headers(auth):
    headers = {"Content-Type": "application/json"}
    a = str(auth or "").strip()
    if a:
        headers["Authorization"] = a if re.search(r"\s", a) else "Bearer " + a
    return headers


def _prompt(node, inp):
    inp = inp or {}
    fields = getattr(node, "fields", None) or {}
    if inp.get("prompt") is not None:
        v = inp.get("prompt")
    elif inp.get("text") is not None:
        v = inp.get("text")
    else:
        v = fields.get("prompt")
    return str("" if v is None else v).strip()


def _is_path(s):
    raw = str("" if s is None else s).strip()
    if not raw or not raw.startswith("/") or len(raw) < 2:
        return False
    if re.search(r"[\s\u00b7|\u2022]", raw):
        return False
    if re.match(r"^[a-z][a-z0-9+.-]*://", raw, re.I):
        return False
    return True


def _join_path(base, path):
    u = urlparse(str(base or "").strip())
    p = str(path or "").strip()
    if not p.startswith("/"):
        p = "/" + p
    q = p.find("?")
    if q >= 0:
        path_part, query = p[:q], p[q + 1:]
    else:
        path_part, query = p, ""
    return urlunparse((u.scheme, u.netloc, path_part, "", query, ""))


def _parse_route(s):
    raw = str("" if s is None else s).strip()
    out = {}
    if not raw:
        return out
    low = raw.lower()
    if low in ENDPOINT_MODES:
        out["mode"] = low
        return out
    mode, rest = "", raw
    for prefix in ENDPOINT_MODES:
        if not low.startswith(prefix):
            continue
        after = raw[len(prefix):]
        ch = after[:1]
        if ch in ("\u00b7", "|", ":", "\u2022", "-") or (ch and ch.isspace()):
            mode = prefix
            rest = re.sub(r"^[\s\u00b7|\u2022:\-]+", "", after)
            break
    rest = re.sub(r"\s+\([^)]*\)\s*$", "", str(rest or "")).strip()
    m = re.search(r"https?://[^\s)]+", rest, re.I)
    url = m.group(0).rstrip(".,;") if m else ""
    if mode:
        out["mode"] = mode
    if url:
        out["url"] = url
    elif _is_path(rest):
        out["path"] = rest
    elif not mode and _is_path(raw):
        out["path"] = raw
    return out


def endpoint_resolve_target(node, inp):
    """Authored URL stays put. A wired url/mode (often a Choice path) rides ``inp``."""
    inp = inp or {}
    fields = getattr(node, "fields", None) or {}
    has_field = "url" in fields
    field_url = fields.get("url") if has_field else ENDPOINT_DEF_URL
    url_raw = inp.get("url") if inp.get("url") not in (None, "") else field_url
    mode_raw = inp.get("mode") if inp.get("mode") not in (None, "") else fields.get("mode")
    parsed_url = _parse_route(url_raw)
    parsed_mode = _parse_route(mode_raw)
    parsed_field = _parse_route(field_url)
    path = parsed_url.get("path") or parsed_mode.get("path")
    if path and not parsed_url.get("url"):
        base = parsed_field.get("url") or str("" if field_url is None else field_url).strip()
        if base and not _is_path(base):
            try:
                parsed_url = {"mode": parsed_url.get("mode"), "url": _join_path(base, path)}
            except ValueError:
                pass
    url = parsed_url.get("url") or str("" if url_raw is None else url_raw).strip()
    if parsed_url.get("url") and parsed_url.get("mode"):
        mode = parsed_url["mode"]
    elif parsed_mode.get("url") and parsed_mode.get("mode"):
        url = parsed_mode["url"]
        mode = parsed_mode["mode"]
    else:
        mode = parsed_mode.get("mode") or parsed_url.get("mode") or fields.get("mode") or "chat"
    mode = str(mode).lower()
    if mode not in ENDPOINT_MODES:
        mode = "chat"
    return {"url": url, "mode": mode}


def _model(node):
    fields = getattr(node, "fields", None) or {}
    m = str(fields.get("model") or "").strip()
    return m or "local"


def _media(v):
    if v is None:
        return None
    url = getattr(v, "url", None)
    return url if isinstance(url, str) else v


def endpoint_request_body(mode, node, inp):
    inp = inp or {}
    fields = getattr(node, "fields", None) or {}
    model = _model(node)
    if mode == "chat":
        prompt = _prompt(node, inp)
        messages = []
        sys = str(fields.get("system") or "").strip()
        if sys:
            messages.append({"role": "system", "content": sys})
        image = _media(inp.get("image"))
        imgs = image if isinstance(image, list) else ([image] if image else [])
        aud = None
        audio = _media(inp.get("audio"))
        if audio:
            data = re.sub(r"^data:[^,]*,", "", str(audio))
            aud = {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}
        if imgs or aud:
            parts = [{"type": "text", "text": prompt or ""}]
            for url in imgs:
                parts.append({"type": "image_url", "image_url": {"url": url}})
            if aud:
                parts.append(aud)
            messages.append({"role": "user", "content": parts})
        else:
            messages.append({"role": "user", "content": prompt})
        return {"model": model, "messages": messages, "temperature": 0.8}
    if mode == "image":
        body = {"model": model, "size": fields.get("size") or "1024x1024",
                "n": 1, "response_format": "b64_json"}
        ip = _prompt(node, inp)
        if ip:
            body["prompt"] = ip
        image = _media(inp.get("image"))
        if image:
            body["imageDataUrl"] = image
        return body
    if mode == "video":
        body = {"model": model, "prompt": _prompt(node, inp)}
        image = _media(inp.get("image"))
        if image:
            body["imageDataUrl"] = image
        video = _media(inp.get("video"))
        if video:
            if re.match(r"^https?:", str(video), re.I):
                body["videoUrl"] = video
            else:
                body["videoDataUrl"] = video
        return body
    if mode == "audio":
        body = {"model": model, "input": _prompt(node, inp)}
        audio = _media(inp.get("audio"))
        if audio:
            if re.match(r"^https?:", str(audio), re.I):
                body["audioUrl"] = audio
            else:
                body["audioDataUrl"] = audio
        return body
    body = {}
    if inp.get("text") not in (None, ""):
        body["text"] = inp.get("text")
    else:
        tp = _prompt(node, inp)
        if tp:
            body["text"] = tp
    for key in ("image", "video", "audio"):
        v = _media(inp.get(key))
        if v:
            body[key] = v
    return body


def _shape_hint(mode):
    return {
        "chat": "OpenAI chat JSON { choices:[{ message:{ content } }] }",
        "image": "{ data:[{ b64_json }] } or { data:[{ url }] }",
        "video": '{ "url" } or NanoGPT { output: { video: { url } } }',
        "audio": '{ "url" } or a binary audio body',
    }.get(mode, '{ "text" } or { "data": ... }')


def _api_message(body):
    try:
        j = json.loads(re.sub(r"^\d{3}:\s*", "", str(body or "")))
    except ValueError:
        return None
    err = j.get("error") if isinstance(j, dict) else None
    msg = None
    if isinstance(err, str) and err:
        msg = err
    elif isinstance(err, dict) and isinstance(err.get("message"), str):
        msg = err["message"]
    elif isinstance(j, dict):
        for k in ("message", "detail", "title"):
            if isinstance(j.get(k), str) and j[k]:
                msg = j[k]
                break
    msg = re.sub(r"\s+", " ", str(msg or "")).strip()
    return msg or None


def endpoint_http_error(status, body):
    extracted = _api_message(body)
    hint = ""
    if status == 404:
        hint = "check the custom endpoint URL"
    elif status in (401, 403):
        hint = "check the Authorization field"
    elif status == 405:
        hint = "this endpoint must accept POST"
    elif status == 413:
        hint = "payload too large; send a smaller body"
    elif status == 415:
        hint = "send JSON (Content-Type: application/json)"
    if extracted:
        return extracted + (" — " + hint if hint else "")
    raw = str(body or "")
    if re.search(r"<html|<body|<!doctype", raw, re.I):
        return "%s — the URL returned a web page, not JSON; check the custom endpoint URL" % status
    if hint:
        return "%s — %s" % (status, hint)
    if not re.sub(r"\s+", "", raw).strip():
        return "%s — the endpoint returned an error with no body; check the URL and mode" % status
    if re.match(r"^\s*[{\[]", raw):
        return "%s — the endpoint rejected the request (check URL, mode, and the posted JSON)" % status
    return "%s: %s" % (status, re.sub(r"\s+", " ", raw).strip()[:160])


def _not_json(mode, ct, raw):
    sample = re.sub(r"\s+", " ", str(raw or "")).strip()[:80]
    if re.search(r"<html|<body|<!doctype", raw or "", re.I) or re.search(r"text/html", ct or "", re.I):
        return ("the endpoint returned a web page, not JSON — check the URL (it must POST "
                + _shape_hint(mode) + ")")
    return "response is not JSON — return " + _shape_hint(mode) + ((" (got: " + sample + ")") if sample else "")


def _parse_chat(j):
    msg = (((j or {}).get("choices") or [{}])[0] or {}).get("message") or {}
    txt = msg.get("content")
    if txt is None:
        raise NanoodleError("no text in response — return " + _shape_hint("chat"))
    if isinstance(txt, str):
        return {"text": txt}
    return {"text": "".join((p or {}).get("text") or "" for p in txt)}


def _parse_image(j, engine):
    urls = []
    for d in ((j or {}).get("data") or []):
        if not isinstance(d, dict):
            continue
        if d.get("b64_json"):
            b64 = d["b64_json"]
            urls.append("data:%s;base64,%s" % (b64_image_mime(b64), b64))
        elif d.get("url"):
            urls.append(d["url"])
    if not urls:
        raise NanoodleError("no image in response — return " + _shape_hint("image"))
    refs = [engine._media_ref(u) for u in urls]
    return {"image": refs[0], "images": refs}


def _parse_video(j, engine):
    data = (j or {}).get("data") if isinstance((j or {}).get("data"), dict) else {}
    out = data.get("output") or (j or {}).get("output") or {}
    url = (j or {}).get("url") or (j or {}).get("videoUrl")
    if not url and isinstance(out, dict):
        video = out.get("video")
        if isinstance(video, dict):
            url = video.get("url")
        url = url or out.get("url")
        if not url and isinstance(video, list) and video:
            url = (video[0] or {}).get("url")
    if not url:
        url = data.get("url")
    if not url:
        raise NanoodleError(
            'no video url in response — return { "url" } or NanoGPT { output: { video: { url } } }')
    return {"video": engine._media_ref(url)}


def _parse_audio_json(j, engine):
    data = (j or {}).get("data") if isinstance((j or {}).get("data"), dict) else {}
    url = (j or {}).get("url") or (j or {}).get("audioUrl") or data.get("url") or data.get("audioUrl")
    if not url:
        raise NanoodleError('no audio url in response — return { "url" } or a binary audio body')
    return {"audio": engine._media_ref(url)}


def _looks_like_echo(j):
    if not isinstance(j, dict):
        return False
    h = j.get("headers")
    if not isinstance(h, dict):
        return False
    if "data" not in j and "json" not in j:
        return False
    return j.get("url") is not None or j.get("origin") is not None or j.get("method") is not None


def _json_mode_text(v):
    if isinstance(v, str):
        s = v.strip()
        if s:
            try:
                p = json.loads(s)
            except ValueError:
                p = None
            if isinstance(p, (dict, list)):
                return _json_mode_text(p)
        return v
    if isinstance(v, dict):
        if _looks_like_echo(v):
            inner = v.get("json") if v.get("json") is not None else v.get("data")
            if inner is not v:
                return _json_mode_text(inner)
        if v.get("text") is not None:
            return str(v["text"])
        try:
            return json.dumps(v, indent=2)
        except (TypeError, ValueError):
            return str(v)
    if isinstance(v, list):
        try:
            return json.dumps(v, indent=2)
        except (TypeError, ValueError):
            return str(v)
    return "" if v is None else str(v)


def _parse_json_mode(j):
    if isinstance(j, dict) and j.get("text") is not None and not _looks_like_echo(j):
        return {"text": str(j["text"])}
    if _looks_like_echo(j):
        inner = j.get("json") if j.get("json") is not None else j.get("data")
        return {"text": _json_mode_text(inner)}
    if isinstance(j, dict) and "data" in j:
        return {"text": _json_mode_text(j.get("data"))}
    raise NanoodleError(
        'json mode expected { "text" } or { "data": ... } — not a chat/completions wrapper')


def endpoint_parse_response(mode, resp, engine):
    ct = resp.header("content-type") or ""
    if mode == "audio" and not re.search(r"json", ct, re.I):
        mime = ct.split(";")[0].strip().lower()
        if not mime or mime in ("application/octet-stream", "binary/octet-stream"):
            mime = "audio/mpeg"
        return {"audio": MediaRef(make_data_url(resp.body, mime), mime=mime,
                                  fetcher=engine.fetch_media)}
    raw = resp.text()
    if not str(raw).strip():
        raise NanoodleError("empty response — the endpoint returned no body; check the URL and mode")
    try:
        j = json.loads(raw)
    except ValueError:
        raise NanoodleError(_not_json(mode, ct, raw))
    if mode == "chat":
        return _parse_chat(j)
    if mode == "image":
        return _parse_image(j, engine)
    if mode == "video":
        return _parse_video(j, engine)
    if mode == "audio":
        return _parse_audio_json(j, engine)
    return _parse_json_mode(j)


def endpoint_fetch_error(exc, url):
    msg = str(exc or "")
    opaque = bool(re.match(r"^(TypeError: )?(Failed to fetch|Load failed|NetworkError)", msg.strip(), re.I))
    opaque = opaque or msg.startswith("could not reach")
    if not opaque:
        return msg
    if not endpoint_url_is_local(url):
        return "blocked by CORS — your server needs Access-Control-Allow-Origin"
    return "blocked — CORS (Access-Control-Allow-Origin + OPTIONS) or the host refused the connection"
