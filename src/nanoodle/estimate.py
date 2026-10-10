"""Up-front run-cost estimate for a graph.

A USD forecast from a caller-supplied NanoGPT catalog, before anything runs.
Nothing here fetches. Image prices are exact; chat, video, audio and 3D are
estimates (``exact`` is false) because the true bill depends on tokens, seconds,
or a variant the catalog only spans.

This is the library twin of the editor/play "~$X to run" chip. Video audio
follows nanoodle #712: an untouched audio switch (the key absent from
``modelOpts``) is priced at the catalog default, which is what NanoGPT bills
when the generate-video body omits the key. An explicit false stays on the
silent tier. A model with no audio-switch descriptor stays silent.
"""

import math
import re

from .graph import IMG_PORT_RE, REF_PORT_RE

# Billable node type → catalog kind. A type absent here is free and local.
PRICE_KIND = {
    "llm": "chat", "vision": "chat", "decide": "chat",
    "image": "image", "edit": "image", "inpaint": "image",
    "tvideo": "video", "ivideo": "video", "vedit": "video", "lipsync": "video",
    "music": "audio", "remix": "audio", "tts": "audio", "transcribe": "audio",
    "cleanvoice": "audio",
    "model3d": "model3d",
}

_VIDEO_FILTER = {"ivideo": "i2v", "vedit": "v2v", "tvideo": "t2v", "lipsync": "avatar"}

EST = {"llmInTokens": 1000, "llmOutTokens": 500, "ttsChars": 600,
       "audioSeconds": 30, "sttMinutes": 1, "videoFps": 30}

# Public catalog rates for these ids are a fraction of the pre-charge 402 quote
# (or only a minimum). The quote table replaces them until the catalog publishes
# a real duration-scaled rate. Twin of index.html VIDEO_QUOTE_PER_SECOND.
VIDEO_QUOTE_PER_SECOND = {
    "minimax-h3-singularity/image-to-video": {
        "default_resolution": "480p", "default_duration": 5,
        "per_second_by_resolution": {"480p": 0.05, "540p": 0.075, "768p": 0.1, "1080p": 0.2}},
    "minimax-h3-singularity/image-to-video-lora": {
        "default_resolution": "480p", "default_duration": 5,
        "per_second_by_resolution": {"480p": 0.0625, "540p": 0.09375, "768p": 0.125, "1080p": 0.25}},
    "minimax-h3/reference-to-video": {
        "default_resolution": "480p", "default_duration": 5,
        "per_second_by_resolution": {"480p": 0.05, "540p": 0.075, "768p": 0.125, "1080p": 0.25}},
    "minimax/h3-max/multi-angle/image-to-video": {
        "default_resolution": "480p", "default_duration": 5,
        "per_second_by_resolution": {"480p": 0.05, "768p": 0.08, "1080p": 0.16}},
    "minimax/h3-max/extend-video": {
        "default_resolution": "768p", "default_duration": 5,
        "per_second_by_resolution": {"480p": 0.05, "768p": 0.08, "1080p": 0.16, "2k": 0.32}},
    "grok-imagine-video-1.5-lite": {
        "default_resolution": "720p", "default_duration": 6,
        "per_second_by_resolution": {"480p": 0.02, "720p": 0.03, "1080p": 0.14}},
    "minimax/h3-max-turbo/extend-video": {
        "default_resolution": "768p", "default_duration": 5,
        "per_second_by_resolution": {"480p": 0.025, "768p": 0.04, "1080p": 0.08, "2k": 0.16}},
    # Catalog raw tables are 1/1.7 of the 402 quote (Seedance 1.5 Pro 720p 5s
    # silent $0.13 / with audio $0.26). Same raw shape, so the audio switch
    # still picks the column.
    "bytedance-seedance-v1.5-pro": {"raw": {
        "type": "resolution-per-second-audio-toggle", "defaultDuration": 5, "defaultResolution": "720p",
        "withoutAudioPricesPerSecond": {"480p": 0.012, "720p": 0.026, "1080p": 0.052},
        "withAudioPricesPerSecond": {"480p": 0.024, "720p": 0.052, "1080p": 0.104}}},
    "bytedance-seedance-v1.5-pro-fast": {"raw": {
        "type": "resolution-per-second-audio-toggle", "defaultDuration": 5, "defaultResolution": "720p",
        "withoutAudioPricesPerSecond": {"720p": 0.02, "1080p": 0.03},
        "withAudioPricesPerSecond": {"720p": 0.04, "1080p": 0.06}}},
}

_AUDIO_KEY = re.compile(r"audio", re.I)
_AUDIO_NEG = re.compile(r"^(no|disable|without|mute)", re.I)
_ON = (True, "true", 1, "1", "on", "yes")

MODEL3D_IMAGE_DEFAULT = "tripo3d/v2.5"
MODEL3D_TEXT_DEFAULT = "wavespeed-ai/hunyuan-3d-v3.1-rapid"


class Estimate(object):
    """Forecast for one run. ``unpriced > 0`` means ``usd`` is a lower bound."""

    __slots__ = ("usd", "exact", "priced", "unpriced")

    def __init__(self, usd, exact, priced, unpriced):
        self.usd = usd
        self.exact = exact
        self.priced = priced
        self.unpriced = unpriced

    def __repr__(self):
        return "Estimate(usd=%r, exact=%r, priced=%r, unpriced=%r)" % (
            self.usd, self.exact, self.priced, self.unpriced)


def _num(x):
    if isinstance(x, bool) or x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _fields(node):
    if node is None:
        return {}
    if isinstance(node, dict):
        return node.get("fields") or {}
    return getattr(node, "fields", None) or {}


def _ntype(node):
    if isinstance(node, dict):
        return node.get("type")
    return getattr(node, "type", None)


def _nid(node):
    if isinstance(node, dict):
        return node.get("id")
    return getattr(node, "id", None)


def _nodes(graph):
    if graph is None:
        return []
    nodes = graph.get("nodes") if isinstance(graph, dict) else getattr(graph, "nodes", None)
    if isinstance(nodes, dict):
        return list(nodes.values())
    return list(nodes or [])


def _links(graph):
    if graph is None:
        return []
    links = graph.get("links") if isinstance(graph, dict) else getattr(graph, "links", None)
    out = []
    for link in links or []:
        if isinstance(link, dict):
            out.append(link)
            continue
        out.append({"from": {"node": link.from_node, "port": link.from_port},
                    "to": {"node": link.to_node, "port": link.to_port}})
    return out


def _to(link):
    to = link.get("to") or {}
    return to.get("node"), to.get("port")


def _node_by_id(graph, nid):
    if isinstance(graph, dict):
        for n in graph.get("nodes") or []:
            if isinstance(n, dict) and n.get("id") == nid:
                return n
        return None
    nodes = getattr(graph, "nodes", None)
    if isinstance(nodes, dict):
        return nodes.get(nid)
    return None


def pick_by_res(map_, req_res, def_res):
    if not isinstance(map_, dict):
        return None
    if req_res is not None and _num(map_.get(req_res)) is not None:
        return _num(map_.get(req_res))
    if def_res is not None and _num(map_.get(def_res)) is not None:
        return _num(map_.get(def_res))
    vals = [v for v in (_num(x) for x in map_.values()) if v is not None]
    return vals[0] if vals else None


def pick_obj_by_res(map_, req_res, def_res):
    if not isinstance(map_, dict):
        return None
    if req_res is not None and isinstance(map_.get(req_res), dict):
        return map_[req_res]
    if def_res is not None and isinstance(map_.get(def_res), dict):
        return map_[def_res]
    objs = [v for v in map_.values() if isinstance(v, dict)]
    return objs[0] if objs else None


def pick_dur(pricing, raw, fields):
    p = pricing or {}
    raw = raw or {}
    try:
        f = float((fields or {}).get("duration"))
    except (TypeError, ValueError):
        f = None
    if f is not None and math.isfinite(f) and f > 0:
        return f
    d = _num(p.get("default_duration"))
    if d is None:
        d = _num(raw.get("defaultDuration"))
    if d is not None and d > 0:
        return d
    per = p.get("per_duration")
    if isinstance(per, dict) and per:
        k = next(iter(per))
        if k is not None and _num(k) is not None:
            return _num(k)
    supported = p.get("supported_durations")
    if isinstance(supported, list) and supported:
        return _num(supported[0]) or 5
    if p.get("fixed_duration_seconds") is not None:
        return _num(p.get("fixed_duration_seconds")) or 5
    return 5


def video_pricing_has_duration_rate(p):
    if not p:
        return False
    return (p.get("per_second_by_resolution") is not None or p.get("per_second") is not None
            or p.get("output_per_second") is not None or p.get("per_second_by_mode") is not None
            or p.get("per_second_by_mode_and_resolution") is not None
            or p.get("base_price_per_second") is not None or p.get("full_per_second") is not None
            or p.get("draft_per_second") is not None
            or p.get("speed_per_second_by_resolution") is not None
            or p.get("quality_per_second_by_resolution") is not None
            or p.get("full_per_second_by_resolution") is not None
            or p.get("draft_per_second_by_resolution") is not None
            or p.get("full_per_second_by_mode_and_resolution") is not None
            or p.get("draft_per_second_by_mode") is not None or p.get("per_duration") is not None
            or (p.get("base_price") is not None and p.get("per_extra_second") is not None)
            or p.get("base_prices_by_resolution") is not None)


def apply_video_quote_pricing(model_id, pricing):
    q = VIDEO_QUOTE_PER_SECOND.get(model_id)
    p = pricing or {}
    if not q or video_pricing_has_duration_rate(p):
        return p
    out = dict(p)
    out.update(q)
    return out


def video_price_fields(params, fields):
    """A model with no duration param bills its own length. Drop a stale knob."""
    pp = params or {}
    if not fields:
        return fields
    dur = fields.get("duration")
    if dur is None or dur == "" or pp.get("duration") or pp.get("seconds"):
        return fields
    out = dict(fields)
    out["duration"] = ""
    return out


def video_lora_on(fields):
    f = fields or {}
    rows = f.get("loras") if isinstance(f.get("loras"), list) else (
        [{"url": f.get("loraUrl")}] if str(f.get("loraUrl") or "").strip() else [])
    return any(r and str((r or {}).get("url") or "").strip() for r in rows)


def _is_on(val):
    return val in _ON


def video_audio_on(fields, audio_ctx=None):
    """Untouched switch → catalog default. Explicit false stays silent.

    No switch descriptor (pricing fixtures, older catalogs) stays off. That is
    the silent tier, and it is what NanoGPT bills only when the model itself
    defaults to off or advertises no switch.
    """
    f = fields or {}
    opts = f.get("modelOpts") if isinstance(f.get("modelOpts"), dict) else {}
    explicit = False
    for key, val in opts.items():
        if _AUDIO_KEY.search(str(key)) and not _AUDIO_NEG.match(str(key)):
            explicit = True
            if _is_on(val):
                return True
    if explicit:
        return False
    ctx = audio_ctx or {}
    pp = ctx.get("params") or {}
    defaults = ctx.get("defaults") or {}
    if not isinstance(pp, dict):
        return False
    for key, desc in pp.items():
        if not isinstance(desc, dict):
            continue
        if desc.get("type") not in ("switch", "boolean"):
            continue
        if not _AUDIO_KEY.search(str(key)) or _AUDIO_NEG.match(str(key)):
            continue
        if key in defaults and defaults.get(key) is not None:
            defv = defaults.get(key)
        else:
            defv = desc.get("default")
        return _is_on(defv)
    return False


def pricing_advertises_refs(pricing):
    if not pricing:
        return False
    if pricing.get("included_reference_images") is not None or pricing.get("extra_reference_image") is not None:
        return True
    mm = pricing.get("per_second_by_mode") or pricing.get("per_second_by_mode_and_resolution")
    return bool(isinstance(mm, dict) and mm.get("reference_to_video") is not None)


def video_unit_usd(pricing, fields, ref_wired=0, video_wired=False, node_filter=None, audio_ctx=None):
    p = pricing or {}
    f = fields or {}
    opts = f.get("modelOpts") if isinstance(f.get("modelOpts"), dict) else {}
    audio_on = video_audio_on(f, audio_ctx)
    if isinstance(ref_wired, bool):
        ref_count = 1 if ref_wired else 0
    else:
        try:
            ref_count = max(0, int(ref_wired))
        except (TypeError, ValueError):
            ref_count = 1 if ref_wired else 0
    ref_on = ref_count > 0 or bool(re.search(r"reference", str(opts.get("mode") or ""), re.I))
    video_on = bool(video_wired) or bool(re.search(r"extend", str(opts.get("mode") or ""), re.I))
    usd = video_base_usd(p, f, audio_on, ref_on, video_on, node_filter)
    if usd is not None and math.isfinite(usd) and audio_on and p.get("audio_multiplier") is not None:
        usd *= _num(p.get("audio_multiplier")) or 1
    if usd is not None and math.isfinite(usd) and ref_count > 0 and video_lora_on(f):
        tier = p.get("lora") if isinstance(p.get("lora"), dict) else None
        per = _num(tier.get("reference_image_or_audio")) if tier else None
        if per is not None:
            usd += per * ref_count
    return usd


def _bill_secs(d, lo, hi):
    lo_n = d if _num(lo) is None else _num(lo)
    hi_n = d if _num(hi) is None else _num(hi)
    return min(hi_n, max(lo_n, d))


def video_base_usd(pricing, fields, audio_on, ref_on, video_on=False, node_filter=None):
    p = pricing or {}
    raw = p.get("raw") if isinstance(p.get("raw"), dict) else {}
    f = fields or {}
    res = f.get("resolution")
    def_res = p.get("default_resolution") if p.get("default_resolution") is not None else raw.get("defaultResolution")
    dur = pick_dur(p, raw, f)

    def rp(m):
        return pick_by_res(m, res, def_res)

    # FLUX.3 nested quality × mode × resolution.
    if p.get("full_per_second_by_mode_and_resolution") or p.get("draft_per_second_by_mode"):
        opts = f.get("modelOpts") if isinstance(f.get("modelOpts"), dict) else {}
        q = str(opts.get("quality") if opts.get("quality") is not None else "full").lower()
        mode_key = "extend" if video_on else "standard"

        def pick_mode(table):
            if not isinstance(table, dict):
                return None
            if table.get(mode_key) is not None:
                return table[mode_key]
            if table.get("standard") is not None:
                return table["standard"]
            keys = list(table)
            return table[keys[0]] if keys else None

        if q == "draft" and p.get("draft_per_second_by_mode"):
            v = _num(pick_mode(p.get("draft_per_second_by_mode")))
            if v is not None:
                return v * dur
        if q == "enhance" and p.get("enhance_per_second") is not None:
            v = _num(p.get("enhance_per_second"))
            if v is not None:
                return v * dur
        if p.get("full_per_second_by_mode_and_resolution"):
            by_res = pick_mode(p.get("full_per_second_by_mode_and_resolution"))
            if isinstance(by_res, dict):
                v = rp(by_res)
                if v is not None:
                    return v * dur
            else:
                v = _num(by_res)
                if v is not None:
                    return v * dur
        if p.get("enhance_per_second") is not None:
            v = _num(p.get("enhance_per_second"))
            if v is not None:
                return v * dur

    if (p.get("speed_per_second_by_resolution") or p.get("quality_per_second_by_resolution")
            or p.get("full_per_second_by_resolution") or p.get("draft_per_second_by_resolution")
            or p.get("full_per_second") is not None or p.get("draft_per_second") is not None):
        o = f.get("modelOpts") if isinstance(f.get("modelOpts"), dict) else {}
        bd = _bill_secs(dur, p.get("minimum_billable_duration"), p.get("maximum_billable_duration"))
        draft = o.get("draft") is True or o.get("draft") == "true"
        mode = str(o.get("mode") if o.get("mode") not in (None, "") else (p.get("default_mode") or "speed")).lower()
        quality = mode == "quality"
        mode_tbl = (p.get("quality_per_second_by_resolution") or p.get("speed_per_second_by_resolution")
                    if quality else p.get("speed_per_second_by_resolution"))
        v = rp(mode_tbl) if mode_tbl else None
        if v is None:
            dt = (p.get("draft_per_second_by_resolution") or p.get("full_per_second_by_resolution")
                  if draft else (p.get("full_per_second_by_resolution") or p.get("draft_per_second_by_resolution")))
            if dt:
                v = rp(dt)
        if v is None:
            v = ((_num(p.get("draft_per_second")) if _num(p.get("draft_per_second")) is not None else _num(p.get("full_per_second")))
                 if draft else
                 (_num(p.get("full_per_second")) if _num(p.get("full_per_second")) is not None else _num(p.get("draft_per_second"))))
        if v is not None:
            return v * bd

    if p.get("output_per_second") is not None:
        L = p.get("lora") if isinstance(p.get("lora"), dict) else None
        if L and video_lora_on(f):
            tbl = L.get("reference_per_second") if ref_on and L.get("reference_per_second") else L.get("text_or_image_per_second")
            v = rp(tbl) if isinstance(tbl, dict) else _num(tbl)
            if v is not None:
                return v * _bill_secs(dur, L.get("min_duration"), L.get("max_duration"))
        v = _num(p.get("output_per_second"))
        if v is not None:
            return v * _bill_secs(dur, p.get("min_duration"), p.get("max_duration"))

    v = rp(p.get("per_second_by_resolution"))
    if v is not None:
        return v * dur
    v = rp(p.get("standard_prices_per_second"))
    if v is not None:
        return v * dur
    if isinstance(p.get("per_second_by_mode"), dict):
        mm = p["per_second_by_mode"]
        v = None
        if ref_on:
            v = _num(mm.get("reference_to_video"))
            if v is None:
                v = _num(mm.get("reference_to_video_image"))
            if v is None:
                v = _num(mm.get("reference_to_video_video"))
        if v is None and video_on:
            v = _num(mm.get("video_edit"))
        if v is None:
            v = _num(mm.get("text_to_video"))
            if v is None:
                v = pick_by_res(mm, res, def_res)
        if v is not None:
            return v * dur
    if isinstance(p.get("per_second_by_mode_and_resolution"), dict):
        mm = p["per_second_by_mode_and_resolution"]

        def pick_mode_res(k):
            t = mm.get(k)
            if t is None:
                return None
            return rp(t) if isinstance(t, dict) else _num(t)

        v = pick_mode_res("reference_to_video") if ref_on else None
        if v is None and video_on:
            v = pick_mode_res("video_edit")
        if v is None:
            v = pick_mode_res("text_to_video")
            if v is None:
                v = pick_mode_res("image_to_video")
        if v is not None:
            return v * dur
    if audio_on:
        for key in ("text_image_with_audio_per_second", "image_to_video_with_audio_per_second",
                    "text_to_video_with_audio_per_second"):
            if p.get(key) is not None:
                v = rp(p[key]) if isinstance(p[key], dict) else _num(p[key])
                if v is not None:
                    return v * dur
    for key in ("text_to_video_per_second", "text_image_without_audio_per_second",
                "image_to_video_without_audio_per_second"):
        if p.get(key) is not None:
            v = rp(p[key]) if isinstance(p[key], dict) else _num(p[key])
            if v is not None:
                return v * dur
    if p.get("base_price_per_second") is not None:
        mult = rp(p.get("resolution_multipliers")) if p.get("resolution_multipliers") else 1
        if mult is None:
            mult = 1
        v = _num(p.get("base_price_per_second"))
        if v is not None:
            return v * dur * mult
    if p.get("per_second") is not None:
        v = _num(p.get("per_second"))
        if v is not None:
            return v * dur
    if isinstance(p.get("per_duration"), dict):
        dk = str(int(round(dur)))
        v = _num(p["per_duration"].get(dk))
        if v is None:
            v = _num(p["per_duration"].get(str(dur)))
        if v is None:
            v = pick_by_res(p["per_duration"], res, def_res)
        if v is not None:
            return v
    if p.get("without_audio") is not None or p.get("with_audio") is not None:
        if audio_on:
            v = _num(p.get("with_audio"))
            if v is None:
                v = _num(p.get("without_audio"))
        else:
            v = _num(p.get("without_audio"))
            if v is None:
                v = _num(p.get("with_audio"))
        if v is not None:
            return v
    if p.get("base_prices_by_resolution"):
        dk = str(int(round(dur)))
        overrides = p.get("duration_overrides") or {}
        if isinstance(overrides, dict) and overrides.get(dk):
            v = rp(overrides[dk])
            if v is not None:
                return v
        base = rp(p.get("base_prices_by_resolution"))
        mult = 1
        if p.get("duration_multiplier") is not None:
            base_d = _num(p.get("base_duration")) or 5
            mult = (_num(p.get("duration_multiplier")) or 1) if dur > base_d else 1
        elif p.get("duration_multipliers"):
            mult = _num((p.get("duration_multipliers") or {}).get(dk))
            if mult is None:
                mult = pick_by_res(p.get("duration_multipliers"), res, def_res)
            if mult is None:
                mult = 1
        if base is not None:
            return base * mult
    v = rp(p.get("per_resolution"))
    if v is not None:
        return v
    if p.get("base_price") is not None and p.get("per_extra_second") is not None:
        bd = _num(p.get("base_duration")) or 0
        return _num(p.get("base_price")) + max(0, dur - bd) * _num(p.get("per_extra_second"))
    if isinstance(p.get("per_video_by_mode"), dict):
        mm = p["per_video_by_mode"]
        want = {"i2v": "image_to_video", "v2v": "video_to_video"}.get(node_filter, "text_to_video")
        v = _num(mm.get(want))
        if v is None:
            vals = [x for x in (_num(x) for x in mm.values()) if x is not None and math.isfinite(x)]
            if vals:
                v = max(vals)
        if v is not None:
            return max(v, _num(p.get("minimum")) or 0)
    if p.get("per_video") is not None:
        v = _num(p.get("per_video"))
        if v is not None:
            return v
    if p.get("per_target_megapixel_second") is not None:
        mp = _num(p.get("default_target_megapixels")) or 1
        v = _num(p.get("per_target_megapixel_second")) * mp * dur
        return max(v, _num(p.get("minimum_price")) or 0)
    if p.get("per_frame_unit") is not None:
        fpu = _num(p.get("frames_per_unit")) or 1
        return (dur * EST["videoFps"] / fpu) * _num(p.get("per_frame_unit"))
    if p.get("base_price") is not None:
        v = _num(p.get("base_price"))
        if v is not None:
            return v
    if raw:
        if ref_on:
            if raw.get("referenceToVideoPrices"):
                inner = pick_obj_by_res(raw["referenceToVideoPrices"], res, def_res)
                if inner:
                    dk = str(int(round(dur)))
                    v = _num(inner.get(dk))
                    if v is None:
                        v = pick_by_res(inner, res, def_res)
                    if v is not None:
                        return v
            for key in ("textToVideoWithReferenceVideoPricesPerSecond",
                        "standardTextToVideoWithReferenceVideoPricesPerSecond",
                        "turboTextToVideoWithReferenceVideoPricesPerSecond",
                        "standardReferenceVideoPricesPerSecond",
                        "fastReferenceVideoPricesPerSecond"):
                if raw.get(key):
                    v = pick_by_res(raw[key], res, def_res)
                    if v is not None:
                        return v * dur
        if audio_on:
            if raw.get("withAudioPricesPerSecond"):
                v = pick_by_res(raw["withAudioPricesPerSecond"], res, def_res)
                if v is not None:
                    return v * dur
            if raw.get("withAudioPrices"):
                inner = pick_obj_by_res(raw["withAudioPrices"], res, def_res)
                if inner:
                    dk = str(int(round(dur)))
                    v = _num(inner.get(dk))
                    if v is None:
                        v = pick_by_res(inner, res, def_res)
                    if v is not None:
                        return v
        for key in ("pricesPerSecond", "textToVideoPricesPerSecond", "withoutAudioPricesPerSecond",
                    "standardPricesPerSecond", "standardTextToVideoPricesPerSecond",
                    "outputPricesPerSecond", "fastPricesPerSecond"):
            if raw.get(key):
                v = pick_by_res(raw[key], res, def_res)
                if v is not None:
                    return v * dur
        for key in ("withoutAudioPrices", "withAudioPrices", "referenceToVideoPrices"):
            if raw.get(key):
                inner = pick_obj_by_res(raw[key], res, def_res)
                if inner:
                    dk = str(int(round(dur)))
                    v = _num(inner.get(dk))
                    if v is None:
                        v = pick_by_res(inner, res, def_res)
                    if v is not None:
                        return v
        if raw.get("pricePerMegapixel") is not None and raw.get("megapixelsByResolution"):
            mp = _num((raw["megapixelsByResolution"] or {}).get(res))
            if mp is None:
                mp = _num((raw["megapixelsByResolution"] or {}).get(def_res))
            if mp is None:
                mp = pick_by_res(raw["megapixelsByResolution"], res, def_res)
            fps_n = _num(f.get("fps")) or _num(raw.get("defaultFramesPerSecond")) or 24
            try:
                chosen = float(f.get("duration"))
            except (TypeError, ValueError):
                chosen = None
            if chosen is not None and math.isfinite(chosen) and chosen > 0:
                nf = round(chosen * fps_n) + 1
            else:
                nf = _num(raw.get("defaultNumFrames")) or (dur * fps_n)
            if mp is not None:
                return _num(raw.get("pricePerMegapixel")) * mp * nf
    return generic_scan_usd(p, dur)


def generic_scan_usd(pricing, dur):
    per_sec = None
    flat = None

    def walk(o, key):
        nonlocal per_sec, flat
        if o is None or isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            if 0 < o < 500:
                if re.search(r"second|persec", str(key), re.I):
                    per_sec = o if per_sec is None else min(per_sec, o)
                else:
                    flat = o if flat is None else min(flat, o)
            return
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)

    walk(pricing, "")
    if per_sec is not None:
        return per_sec * dur
    if flat is not None:
        return flat
    return None


def chat_unit_usd(pricing, in_tokens, out_tokens):
    if not pricing:
        return None
    pin = _num(pricing.get("prompt"))
    if pin is None:
        pin = _num(pricing.get("promptUsd1M"))
    pout = _num(pricing.get("completion"))
    if pout is None:
        pout = _num(pricing.get("completionUsd1M"))
    if pin is None and pout is None:
        return None
    return ((pin or 0) * (EST["llmInTokens"] if in_tokens is None else in_tokens)
            + (pout or 0) * (EST["llmOutTokens"] if out_tokens is None else out_tokens)) / 1e6


def audio_unit_usd(pricing, chars=None, seconds=None):
    p = pricing or {}
    c = EST["ttsChars"] if chars is None else chars
    secs = seconds if (seconds is not None and math.isfinite(seconds)) else EST["audioSeconds"]
    if p.get("per_thousand_chars") is not None:
        cost = _num(p.get("per_thousand_chars")) * (c / 1000.0)
        if p.get("per_generation") is not None:
            cost += _num(p.get("per_generation"))
        return max(cost, _num(p.get("minimum")) or 0)
    if p.get("per_prompt_char_block") is not None:
        bs = _num(p.get("prompt_char_block_size")) or 1
        return max(math.ceil(c / bs) * _num(p.get("per_prompt_char_block")), _num(p.get("minimum")) or 0)
    if p.get("per_generation") is not None:
        return _num(p.get("per_generation"))
    per_sec = _num(p.get("per_second"))
    if per_sec is not None:
        return max(per_sec * secs, _num(p.get("minimum")) or 0)
    if p.get("per_billing_interval") is not None:
        iv = _num(p.get("billing_interval_seconds")) or 60
        return max(math.ceil(secs / iv) * _num(p.get("per_billing_interval")), _num(p.get("minimum")) or 0)
    if p.get("per_minute") is not None:
        return _num(p.get("per_minute")) * EST["sttMinutes"]
    return None


def audio_billed_seconds(params, fields):
    p = params or {}
    if p.get("min_duration") is None or p.get("max_duration") is None:
        return None
    try:
        d = float((fields or {}).get("duration"))
    except (TypeError, ValueError):
        d = None
    lo = _num(p.get("min_duration"))
    hi = _num(p.get("max_duration"))
    if d is None or not math.isfinite(d):
        e = EST["audioSeconds"]
        if (lo is not None and e < lo) or (hi is not None and e > hi):
            return min(hi if hi is not None else e, max(lo if lo is not None else e, e))
        return None
    return min(hi if hi is not None else d, max(lo if lo is not None else d, d))


def image_usd(pricing, size, count):
    per = (pricing or {}).get("per_image") or {}
    if not isinstance(per, dict) or not per:
        return None
    v = per.get(size)
    if v is None:
        v = per.get("square")
    if v is None:
        v = per.get("square_hd")
    if v is None:
        v = next(iter(per.values()))
    n = _num(v)
    return None if n is None else n * (count or 1)


def _cat_maps(catalogs):
    by_kind = {}
    for kind in ("chat", "image", "video", "audio", "model3d"):
        bucket = (catalogs or {}).get(kind)
        mapping = {}
        if isinstance(bucket, dict) and "id" not in bucket and "pricing" not in bucket:
            for k, m in bucket.items():
                if isinstance(m, dict):
                    mapping[str(m.get("id") if m.get("id") is not None else k)] = m
        elif isinstance(bucket, list):
            for m in bucket:
                if isinstance(m, dict) and m.get("id") is not None:
                    mapping[str(m["id"])] = m
        elif isinstance(bucket, dict) and bucket.get("id") is not None:
            mapping[str(bucket["id"])] = bucket
        by_kind[kind] = mapping
    return by_kind


def cat_item(catalogs, kind, model_id):
    if not model_id:
        return None
    return _cat_maps(catalogs).get(kind, {}).get(str(model_id))


def model3d_input_mods(model_id, catalogs=None):
    """Which of image / text this 3D model accepts.

    Unknown ids accept both, matching the editor when the catalog hasn't loaded.
    ``tripo3d/v2.5`` is image-only; the text default accepts both.
    """
    it = cat_item(catalogs, "model3d", model_id)
    mods = None
    if it:
        mods = it.get("modalities")
        if not isinstance(mods, list):
            arch = it.get("architecture") or {}
            mods = arch.get("input_modalities") if isinstance(arch, dict) else None
    if isinstance(mods, list):
        return {"image": "image" in mods, "text": "text" in mods}
    if model_id == MODEL3D_IMAGE_DEFAULT:
        return {"image": True, "text": False}
    if model_id == MODEL3D_TEXT_DEFAULT:
        return {"image": True, "text": True}
    return {"image": True, "text": True}


def model3d_has_span(pricing, fields, defaults):
    """True when per_run_by_variant spans and a knob has left the catalog default."""
    by = (pricing or {}).get("per_run_by_variant")
    if not isinstance(by, dict):
        return False
    vals = [v for v in (_num(x) for x in by.values()) if v is not None]
    if len(vals) < 2 or min(vals) == max(vals):
        return False
    opts = (fields or {}).get("modelOpts")
    defs = defaults or {}
    if not isinstance(opts, dict):
        return False
    for k, v in opts.items():
        if v is None or v == "":
            continue
        if defs.get(k) is None or str(v) != str(defs.get(k)):
            return True
    return False


def _param_ctx(model):
    sp = (model or {}).get("supported_parameters") or {}
    if not isinstance(sp, dict):
        sp = {}
    nested = sp.get("parameters")
    params = nested if isinstance(nested, dict) else sp
    defaults = sp.get("defaults") if isinstance(sp.get("defaults"), dict) else {}
    # Normalized editor rows keep descriptors on ``params`` and defaults beside them.
    if (model or {}).get("params") and isinstance(model.get("params"), dict) and not nested:
        params = model["params"]
    if (model or {}).get("defaults") and isinstance(model.get("defaults"), dict) and not defaults:
        defaults = model["defaults"]
    return params, defaults


def _ref_count(node, graph, params, pricing):
    ntype = _ntype(node)
    if ntype not in ("tvideo", "vedit"):
        return 0
    pp = params or {}
    has = any(k in pp for k in ("reference_images", "reference_image_urls", "referenceImages"))
    has = has or pricing_advertises_refs(pricing)
    if not has:
        return 0
    nid = _nid(node)
    n = 0
    for link in _links(graph):
        to_node, to_port = _to(link)
        if to_node == nid and to_port and REF_PORT_RE.match(str(to_port)):
            n += 1
    return n


def node_unit_usd(node, catalogs=None, graph=None):
    """USD for one execution, or None when the node is free or unpriceable."""
    kind = PRICE_KIND.get(_ntype(node))
    if not kind:
        return None
    fields = _fields(node)
    model_id = fields.get("model")
    if not model_id:
        return None
    model = cat_item(catalogs, kind, model_id)
    if not model:
        return None
    pricing = model.get("pricing") or {}
    ntype = _ntype(node)
    if kind == "image":
        sp = model.get("supported_parameters") or {}
        max_out = sp.get("max_output_images") or model.get("maxOut") or 1
        fixed = sp.get("fixed_image_count") or model.get("fixedCount") or 0
        try:
            variations = int(fields.get("variations") or 1)
        except (TypeError, ValueError):
            variations = 1
        count = fixed if fixed > 1 else min(max_out, max(1, variations))
        return image_usd(pricing, fields.get("size"), count)
    if kind == "video":
        params, defaults = _param_ctx(model)
        pricing = apply_video_quote_pricing(model_id, pricing)
        ref_count = _ref_count(node, graph, params, pricing)
        nid = _nid(node)
        video_wired = any(_to(l) == (nid, "video") for l in _links(graph))
        priced_fields = video_price_fields(params, fields)
        u = video_unit_usd(pricing, priced_fields, ref_count, video_wired,
                           _VIDEO_FILTER.get(ntype), {"params": params, "defaults": defaults})
        return u if u is not None and math.isfinite(u) else None
    if ntype == "decide":
        from .decide import decide_mode
        imgs = 0
        nid = _nid(node)
        for link in _links(graph):
            to_node, to_port = _to(link)
            if to_node == nid and to_port and IMG_PORT_RE.match(str(to_port)):
                imgs += 1
        in_tok = (120 + int(round((len(fields.get("question") or "") + len(fields.get("options") or "")
                                   + len(fields.get("levels") or "")) / 4))
                  + 250 + imgs * 400)
        u = chat_unit_usd(pricing, in_tok, 0)
        if u is None:
            return None
        return u * (2 if decide_mode(fields) == "pick" else 1)
    if kind == "chat":
        in_tok = max(200, int(round((len(fields.get("system") or "") + len(fields.get("prompt") or "")) / 4)) + 200)
        out_tok = _num(fields.get("maxTokens")) if fields.get("maxTokens") else EST["llmOutTokens"]
        return chat_unit_usd(pricing, in_tok, out_tok)
    if ntype == "cleanvoice":
        from .cleanvoice import clean_voice_est_seconds
        nid = _nid(node)
        src = None
        # The run cleans wired audio before wired video. Price that same source.
        for want in ("audio", "video"):
            for link in _links(graph):
                to_node, to_port = _to(link)
                if to_node == nid and to_port == want:
                    src = _node_by_id(graph, (link.get("from") or {}).get("node"))
                    break
            if src is not None:
                break
        return audio_unit_usd(pricing, None, clean_voice_est_seconds(_fields(src)))
    if kind == "audio":
        chars = None if ntype == "transcribe" else (len(fields.get("prompt") or "") or EST["ttsChars"])
        sp = model.get("supported_parameters") if isinstance(model.get("supported_parameters"), dict) else {}
        secs = audio_billed_seconds(sp, fields)
        u = audio_unit_usd(pricing, chars, secs)
        return u if u is not None and math.isfinite(u) else None
    if kind == "model3d":
        return _num(pricing.get("per_run"))
    return None


def estimate_graph_cost(graph, catalogs=None):
    """Sum a graph's per-run forecast.

    ``catalogs`` is ``{"chat": [...], "image": [...], "video": [...], "audio": [...],
    "model3d": [...]}`` — the same arrays ``/api/v1/*-models`` return, or the
    opt-in ``Workflow(catalog=...)`` blob. Never fetched here.
    """
    catalogs = catalogs or {}
    usd = 0.0
    priced = 0
    unpriced = 0
    exact = True
    for node in _nodes(graph):
        kind = PRICE_KIND.get(_ntype(node))
        if not kind:
            continue
        u = node_unit_usd(node, catalogs, graph)
        if u is None or not math.isfinite(u):
            unpriced += 1
            continue
        usd += u
        priced += 1
        if kind != "image":
            exact = False
        if kind == "model3d":
            model = cat_item(catalogs, "model3d", _fields(node).get("model"))
            _params, defaults = _param_ctx(model or {})
            pricing = (model or {}).get("pricing") or {}
            by = pricing.get("per_run_by_variant")
            if isinstance(by, dict):
                vals = [v for v in (_num(x) for x in by.values()) if v is not None]
                if len(vals) >= 2 and min(vals) != max(vals):
                    exact = False
    return Estimate(usd, exact if priced else True, priced, unpriced)
