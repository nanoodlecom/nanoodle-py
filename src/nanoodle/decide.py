"""⚖️ Decide — typed judgments from NanoGPT decision models (POST /api/v1/decisions).

Port of nanoodle-js src/decide.mjs (itself the twin of the editor/play "⚖️ DECIDE" block):
same question shapes, same image state parts, same in-order + reversed pick debias, same
outputs. Only the image shrink differs from the browser: ffmpeg (fit_image_jpeg) instead
of canvas — text-only decisions need no ffmpeg at all.
"""

import re

from .errors import NanoodleError

DECIDE_ENDPOINT = "/api/v1/decisions"
DECIDE_DEFAULT_MODEL = "perplexity/pplx-decider-v1.1-27b"
DECIDE_SCALE_DEFAULT = "poor\nokay\ngood\ngreat"
# The live limits of every image-capable decision model at launch (catalog decision_input.image_limits).
DECIDE_IMG_FALLBACK = {"maxImages": 4, "maxDimension": 512, "maxEncodedBytes": 240000}


class DecideGateClosed(NanoodleError):
    """A closed yes/no gate: a deliberate stop, not a failure.

    The runner raises it (``code == "decide-gate"``, ``gate is True``); Workflow.run()
    settles it as node status ``"gated"`` — the decision ran and billed, ``out`` holds its
    answer, everything downstream is ``"skipped"`` unbilled, and the run succeeds.
    """

    code = "decide-gate"
    gate = True

    def __init__(self, decision, out=None):
        super().__init__(
            "gate closed — the answer was no (yes %d%%), so nothing downstream ran"
            % int(round(decision["yes"] * 100)))
        self.decision = decision
        self.out = out


def _js_round(x):
    """Math.round: half rounds up (Python's round() is banker's)."""
    import math
    return int(math.floor(x + 0.5))


def _num(v):
    """JS +v || 0 for probabilities/scores."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f else 0.0


def decide_mode(f):
    m = (f or {}).get("mode")
    return m if m in ("choose", "score", "yesno") else "pick"


def decide_lines(s):
    return [x.strip() for x in str("" if s is None else s).split("\n") if x.strip()]


def decide_default_question(mode):
    return {
        "pick": "Which image best matches the brief?",
        "choose": "Which label fits best?",
        "score": "How good is it?",
    }.get(mode, "Is it good enough to use?")


def decide_image_limits(di, known):
    """Catalog decision_input -> limits. known=False (absent/offline) -> permissive launch
    limits; a known text-only model -> None."""
    if not known:
        return dict(DECIDE_IMG_FALLBACK)
    if not di or not di.get("image_input"):
        return None
    lim = di.get("image_limits") or {}

    def pos(k):
        v = lim.get(k)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else DECIDE_IMG_FALLBACK[k]
    return {k: pos(k) for k in ("maxImages", "maxDimension", "maxEncodedBytes")}


def decide_question_for(mode, f, n_imgs):
    """The one typed question this node asks. Raises (before any request) on an unanswerable setup."""
    f = f or {}
    q = str(f.get("question") or "").strip() or decide_default_question(mode)
    if mode == "pick":
        if n_imgs < 2:
            raise NanoodleError("pick needs at least two images — wire 2 or more into the image ports")
        return {"type": "choice", "instructions": q,
                "criteria": {"image_%d" % i: "image %d" % i for i in range(1, n_imgs + 1)}}
    if mode == "choose":
        labels = []
        for x in decide_lines(f.get("options")):
            if x not in labels:
                labels.append(x)
        if len(labels) < 2:
            raise NanoodleError("add at least two labels to choose from (one per line)")
        if len(labels) > 255:
            raise NanoodleError("too many labels — 255 at most")
        return {"type": "choice", "instructions": q, "criteria": {x: None for x in labels}}
    if mode == "score":
        raw = f.get("levels")
        lv = decide_lines(raw if (raw is not None and str(raw).strip()) else DECIDE_SCALE_DEFAULT)
        if len(lv) < 2 or len(lv) > 10:
            raise NanoodleError("the scale needs 2 to 10 levels, worst first (one per line)")
        return {"type": "score", "instructions": q, "criteria": lv}
    return {"type": "noul", "instructions": q}


def decide_state(text, imgs):
    """state: plain text, or (with images) text + labelled inline image_url parts."""
    if not imgs:
        return text
    s = []
    if text:
        s.append(text)
    for i, u in enumerate(imgs):
        if len(imgs) > 1:
            s.append("image_%d:" % (i + 1))
        s.append({"type": "image_url", "image_url": {"url": u}})
    return s


def _fmt_score(x):
    r = _js_round(x * 100) / 100.0
    return str(int(r)) if r == int(r) else repr(r)


def decide_outputs(mode, ans, q, imgs, usage, model):
    """API answer -> node outputs: text, image (pick: the winner; else the first image), decision."""
    if not isinstance(ans, dict) or not ans.get("type"):
        raise NanoodleError("the decision model returned no answer")
    cost = (usage or {}).get("cost")
    d = {"mode": mode, "model": model,
         "cost": cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
         "rows": []}
    probs = ans.get("probabilities") or {}
    first = imgs[0] if imgs else ""
    if mode == "pick":
        k = str(ans.get("choice") or "")
        try:
            idx = int(re.sub(r"^image_", "", k)) or 1
        except ValueError:
            idx = 1
        d["pick"] = idx
        d["confidence"] = ans.get("confidence")
        d["rows"] = [{"label": "image %d" % (i + 1), "p": _num(probs.get(key)), "win": key == k}
                     for i, key in enumerate(q["criteria"])]
        return {"text": "image %d" % idx, "image": imgs[idx - 1] if 0 < idx <= len(imgs) else "",
                "decision": d}
    if mode == "choose":
        d["confidence"] = ans.get("confidence")
        d["rows"] = [{"label": key, "p": _num(probs.get(key)), "win": key == ans.get("choice")}
                     for key in q["criteria"]]
        return {"text": str(ans.get("choice") or ""), "image": first, "decision": d}
    if mode == "score":
        sc = _num(ans.get("score")) + 1  # 1 = the first (worst) level
        d["score"] = sc
        d["levels"] = len(q["criteria"])
        d["confidence"] = ans.get("confidence")
        top = 0
        for i in range(len(q["criteria"])):
            if _num(probs.get(str(i))) > _num(probs.get(str(top))):
                top = i
        d["rows"] = [{"label": "%d · %s" % (i + 1, x), "p": _num(probs.get(str(i))), "win": i == top}
                     for i, x in enumerate(q["criteria"])]
        return {"text": _fmt_score(sc), "image": first, "decision": d}
    py = max(0.0, min(1.0, _num(ans.get("noul"))))
    d["yes"] = py
    d["rows"] = [{"label": "yes", "p": py, "win": py >= 0.5},
                 {"label": "no", "p": 1 - py, "win": py < 0.5}]
    return {"text": "yes" if py >= 0.5 else "no", "image": first, "decision": d}


def decide_merge_orders(js, orders):
    """Average pick probabilities across runs that saw the candidates in different orders; costs add up."""
    n = len(orders[0])
    total = [0.0] * n
    cost, cost_known = 0.0, True
    for r, j in enumerate(js):
        a = ((j or {}).get("answers") or {}).get("answer")
        if not isinstance(a, dict) or not a.get("probabilities"):
            raise NanoodleError("the decision model returned no answer")
        for k, orig in enumerate(orders[r]):
            total[orig] += _num(a["probabilities"].get("image_%d" % (k + 1))) / len(js)
        c = ((j or {}).get("usage") or {}).get("cost")
        if isinstance(c, (int, float)) and not isinstance(c, bool):
            cost += c
        else:
            cost_known = False
    best = 0
    probs = {}
    for i, p in enumerate(total):
        probs["image_%d" % (i + 1)] = p
        if p > total[best]:
            best = i
    return {"answers": {"answer": {"type": "choice", "choice": "image_%d" % (best + 1),
                                   "confidence": total[best], "probabilities": probs}},
            "usage": {"cost": cost if cost_known else None}, "runs": len(js)}


def decide_run(f, model, text, imgs, lim, send, fit):
    """The whole run. send(body) POSTs /api/v1/decisions and returns the JSON;
    fit(url, max_dim, budget) returns a data: URL that fits (inlining remote URLs)."""
    f = f or {}
    mode = decide_mode(f)
    text = str("" if text is None else text).strip()
    imgs = [u for u in (imgs or []) if u]
    if imgs and not lim:
        raise NanoodleError("this decision model can’t see images — pick one that can (e.g. PPLX Decider or Clef)")
    if lim and len(imgs) > lim["maxImages"]:
        raise NanoodleError("this decision model takes at most %d images — unwire the extras" % lim["maxImages"])
    q = decide_question_for(mode, f, len(imgs))
    if not text and not imgs and not str(f.get("question") or "").strip():
        raise NanoodleError("nothing to judge — wire text or an image into Decide")
    sent = []
    if imgs:
        budget = int(lim["maxEncodedBytes"] * 0.95 // len(imgs))
        for u in imgs:
            sent.append(fit(u, lim["maxDimension"], budget))
    if mode == "pick":
        # decision models lean toward the first image they see: ask in order and reversed, average
        fwd = list(range(len(sent)))
        rev = fwd[::-1]
        both = [send({"model": model, "state": decide_state(text, [sent[i] for i in order]),
                      "questions": {"answer": q}}) for order in (fwd, rev)]
        j = decide_merge_orders(both, [fwd, rev])
    else:
        j = send({"model": model, "state": decide_state(text, sent), "questions": {"answer": q}})
    j = j or {}
    out = decide_outputs(mode, (j.get("answers") or {}).get("answer"), q, imgs, j.get("usage"), model)
    if mode == "yesno" and f.get("gate") in (True, "true") and out["decision"]["yes"] < 0.5:
        raise DecideGateClosed(out["decision"], out)
    return out
