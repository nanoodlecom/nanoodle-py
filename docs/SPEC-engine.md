# Nanoodle Run Engine — server-side re-implementation spec

Blueprint = play.html RUNTIME_JS (the exported-app runtime). Line refs are play.html.

## Endpoints & auth
```
NANOGPT = https://nano-gpt.com
IMG_ENDPOINT   POST {NANOGPT}/v1/images/generations      (note: NOT /api/v1)
CHAT_ENDPOINT  POST {NANOGPT}/api/v1/chat/completions
VIDEO submit   POST {NANOGPT}/api/generate-video
VIDEO poll     GET  {NANOGPT}/api/video/status?requestId=<id>
AUDIO speech   POST {NANOGPT}/api/v1/audio/speech        (music + tts + remix)
AUDIO poll     GET  {NANOGPT}/api/tts/status?<qs>
TRANSCRIBE     POST {NANOGPT}/api/v1/audio/transcriptions   (multipart)
Catalogs (public, no key): /api/v1/models?detailed=true, /api/v1/image-models,
                           /api/v1/video-models, /api/v1/audio-models
```
Every JSON call: `Content-Type: application/json` + BOTH `Authorization: Bearer <key>` and `x-api-key: <key>`.
Multipart transcribe: the two auth headers, NO explicit Content-Type (let the http lib set boundary). File form field MUST be named "file"; also fields `model`, optional `language`.

All media is inlined as base64 `data:` URLs in JSON bodies (no upload endpoint). MEDIA_INLINE_MAX = 4.4MB — guard locally and raise a clear error above it **when the graph has network nodes** (NanoGPT request bodies). Local-only graphs (resize / vframes / combine / soundtrack / trim / extractaudio) may accept larger `data:` inputs for on-device ffmpeg. Chat runs NON-STREAMING (no `stream` key); parse `r.json()` once.

Model strings pass through VERBATIM (`body.model = node.fields.model`). Missing model → error "pick a model first". Endpoint choice is by node TYPE, never by model lookup. The catalog is optional/best-effort (capability gating + clamps); the library must run fine with no catalog fetch.

## Per-node payloads

### llm → CHAT_ENDPOINT (genChat 1481-1504, run 2574-2603)
```
body = { model, messages, temperature: 0.8 }
if maxTokens          body.max_tokens = +maxTokens
if format == "JSON"   body.response_format = { type: "json_object" }
if reasoningEffort set and != "default"  body.reasoning_effort = value
```
messages: optional {role:"system", content: fields.system} then user message.
User content = plain string prompt, UNLESS wired images (img1,img2,... sorted by index) or wired audio:
then array `[{type:"text",text:prompt}, {type:"image_url",image_url:{url:<value verbatim>}}..., audioPart?]`.
audioPart = `{type:"input_audio", input_audio:{data:<base64 body, no data: prefix>, format:"mp3"|"wav"|...}}`.
prompt = wired `prompt` port value ?? fields.prompt; error "no prompt" if empty.
Parse: `j.choices[0].message.content` (if array, join `.map(p=>p.text)`); throw "no text in response" if null/empty.
(withLocale non-English system suffix: SKIP in the library / make opt-in.)

### vision → same as llm: one user message [{text: q||"Describe this image."}, {image_url: inp.image}]

### image / edit / inpaint → IMG_ENDPOINT (genImage 1465-1480)
```
body = { model, size: fields.size || "1024x1024", n: variations||1, response_format: "b64_json" }
if prompt        body.prompt = prompt
if source image  body.imageDataUrl = <string OR array of strings (edit multi-ref)>
if mask          body.maskDataUrl = <mask data URL; white = repaint>
+ seed (when numeric), + customCivitaiAir (model "custom-civitai"), + LoRA params
```
- image node: `variations` → n (multi output = j.data list).
- edit: sources from wired `image, image2, ...` ports; single → string, multiple → array. Prompt may be empty for upscaler models — do not hard-require.
- inpaint: source+mask from ports or fields. (Browser composites mask onto black at source size via canvas; library v1: pass mask through verbatim and document the caveat.)
Parse: `j.data[]` → `d.b64_json ? "data:<sniffed mime>;base64,"+b64 : d.url`. Sniff mime from magic bytes (PNG \x89PNG, JPEG \xFF\xD8, GIF, WEBP RIFF....WEBP; default image/png). Throw "no image in response" if empty.

### tvideo / ivideo / vedit / lipsync → submit + poll (genVideo 1527-1588)
```
body = { model, prompt }
+ dims from node fields: aspect → aspect_ratio, duration → duration, resolution → resolution
  (catalog can rename e.g. aspect→orientation / duration→seconds; WITHOUT catalog use the standard names)
+ ivideo/lipsync source image → body.imageDataUrl (data: or https, verbatim)
+ ivideo wired endframe → body.last_image
+ vedit source: https URL → body.videoUrl ; local data → body.videoDataUrl
+ lipsync audio: https → body.audioUrl ; local → body.audioDataUrl
+ Object.assign(body, fields.modelOpts || {})   (per-model knobs incl. seed)
+ tvideo wired ref1.. → body.reference_images (array) — catalog may rename key; default "reference_images"
```
Submit response: `runId = j.runId || j.id`. Then poll every 5s: GET /api/video/status?requestId=<runId>.
status = (s.data?.status || s.status).toUpperCase().
COMPLETED|SUCCEEDED → url = s.data?.output?.video?.url || out.url || out.video?.[0]?.url (out = s.data?.output or s).
FAILED|ERROR|CANCELED → raise "video failed: " + (s.data?.error || status). Timeout 600s.

### music / tts / remix → AUDIO speech endpoint (genAudio 1591-1637)
```
body = { model, input: <wired text ?? fields.prompt> } + params
music params: lyrics, instrumental(bool), duration(number), negative_prompt, seed, response_format(default "mp3")
tts params:   voice, speed(omit when 1), instructions, response_format(default "mp3")
remix params: lyrics, duration, response_format + body.audio = <source: https as-is | local data URL>
+ merge fields.extraJson (parsed object) verbatim last
Omit empty params. (Catalog gating of voice/duration: skip in v1 — send what the node has.)
```
Response handling:
- content-type JSON → url = j.url||j.audioUrl||j.data?.url||j.data?.audioUrl; if none but j.runId||j.id → poll
  GET /api/tts/status?runId=..&model=..&cost=..&paymentSource=..&isApiRequest=true every 3s;
  status lowercase: completed|succeeded → s.audioUrl||s.url||s.data?.audioUrl||s.data?.url;
  error|failed|content_policy_violation → raise. Timeout 300s.
- else BINARY body → the audio bytes; mime from response content-type (pin from requested format if generic).
  Library returns bytes+mime (a data: URL or MediaRef), not an object URL.

### transcribe → multipart (1641-1666)
FormData: file=<audio blob> (field name "file"), model, language?. Local guard: >3.5MB raise.
Parse: `j.transcription ?? j.text ?? j.data?.transcription ?? j.data?.text`.

## Cost extraction (costFromJson 998-1013)
USD priority: j.cost (if >0) → j.x_nanogpt_pricing.(costUsd|cost|amount) → j.metadata?.cost → header x-cost / x-nano-cost.
Balance: header x-remaining-balance (wins) → j.remainingBalance → x_nanogpt_pricing.remainingBalance.
Present-but-zero = known-included (subscription), keep 0. Absent → cost unknown (mark total inexact).

## HTTP errors (922-935)
- 401/403 → auth error ("API key rejected").
- 402 OR body matching /insufficient|balance|funds|not enough|payment required/i → out-of-funds error.
- else → error "<status>: <body first 160 chars>".
No streaming retries needed (engine is non-streaming). Poll GET failures: silently continue the loop until timeout.

## Run deadline (`run(timeout=…)`)
A run-level timeout sets an absolute deadline on the engine. The deadline outranks every per-node timeout:
- The video and audio status-poll loops check the deadline before and after each sleep, and stop at once when it passes. `timeout_video` (600 s) and `timeout_audio` (300 s) only apply while the deadline is in the future.
- The sleep between poll attempts is interruptible. It waits on a cancel event and never sleeps past the deadline.
- Per-request socket timeouts are capped at the time left, so one read cannot outlive the deadline by up to `http_timeout` (120 s).
- The x402 settle poll stops at the deadline too, and starts no NEW deposit once the run is cancelled.
This matters because a live worker can block interpreter exit, not just `run()`. `ThreadPoolExecutor` workers are non-daemon, and both the `concurrent.futures` atexit hook and `threading._shutdown()` join every non-daemon thread, so a thread still polling an abandoned run holds the whole process. The run pool is therefore `workflow._DaemonPool`, whose workers are daemon threads: deadline-aware loops shorten that window, and daemon workers close it. With no `timeout=`, no deadline exists and every loop behaves exactly as it did before.

A daemon thread is frozen at interpreter finalization, which is right for a poll loop and wrong in the middle of a wallet callback. Those spans mark themselves money-critical (`engine._money_critical`) and an atexit hook waits up to `EXIT_MONEY_GRACE` (5 s) for them, then exits anyway.

A frozen daemon also never reaches the `finally` that kills its ffmpeg child. Local media nodes shell out to ffmpeg/ffprobe, so every child `local_media._run` starts is registered and a second atexit hook kills whatever is still alive (`local_media._kill_children_at_exit`, hard-bounded by `EXIT_CHILD_GRACE`, 2 s).

**That hook is the only thing that bounds an orphaned child. Do not remove it. SIGPIPE is not a backstop, because it is a race that the orphan usually wins.**

- A quiet child — `ffprobe -v error`, the flag every `local_media` probe passes — writes nothing until it is done. It never touches the closed pipe, so nothing tells it to stop, ever.
- A chatty `ffmpeg` writes to stderr about twice a second, so it does touch the closed pipe. Whether that kills it depends on WHEN the parent died: `ffmpeg` sets SIGPIPE to `SIG_IGN` itself (the SIGPIPE bit of `SigIgn` in `/proc/PID/status`, mask `…1000`) a fraction of a second after it starts. Measured with `--when`: 0.08 to 0.11 s on an idle machine, 0.39 to 0.61 s on a busy one, 19 runs in total. Before that point the next write kills the child. After it, the writes fail with `EPIPE`, `av_log` discards the error, and the child runs on to the end of its work.

`scripts/measure-orphan-sigpipe.py` reads the REAL fate of the orphan: it makes itself a child subreaper, so `waitpid()` reports the exit status of a process whose parent is gone. On Linux 6.14, ffmpeg 7.1.1, Python 3.11.10. **Every count below moves with the machine and its load** — the start-up of ffmpeg is what varies, and this box was shared with other work:

| the parent dies … | the orphaned chatty ffmpeg | runs |
|---|---|---|
| 0.05 s after the child starts | killed by SIGPIPE, 11 of 11 | 11 |
| 0.4 s after | killed by SIGPIPE 11 times, alive at the end of the 12 s wait 13 times | 24 |
| 1.0 s after | alive at the end of the wait 7 times, killed once | 8 |
| 3.0 s after | alive at the end of the wait, 6 of 6 | 6 |

The 0.4 s row is the one that matters, because that is where an abandoned worker of this library lands, and there the answer is a coin flip. Through the library harness itself (`scripts/measure-timeout-hang.py --chatty --no-reaper`), a chatty ffmpeg was still running 45 s after its parent in 14 of 16 runs, and died 0.33 s and 1.74 s after it in the other 2. Only children that had NOT yet reached their `SIG_IGN` call died; every child that had reached it survived (6 runs, each correlated against `/proc/PID/status`). With the hook, the same chatty child was dead at the first check after the parent exited (0.00 s, 8 runs out of 8).

Two earlier revisions of this document each wrote down one side of that race as a fact — first "a chatty ffmpeg dies on its own about 1.5 to 2.0 s later", then "still running 45 s after its parent, 8 runs out of 8". Both are retracted. The orphan usually survives, sometimes dies, and nothing in this library decides which.

`ffprobe` is the one predictable case: it does not ignore SIGPIPE at all. A chatty probe (`ffprobe -show_frames -of csv` on the same source) died 0.02 to 0.04 s after its parent in 8 runs of `measure-orphan-sigpipe.py --bin ffprobe`, and 0.00 to 0.02 s after it in 5 runs of `measure-timeout-hang.py --chatty-probe --no-reaper`. No `local_media` ffprobe call is chatty, though — they all pass `-v error`. Do not depend on SIGPIPE anywhere.

### What the deadline must NOT bound
The deadline governs work the run is still doing. Two things have a different lifetime and stay outside it:
- **Media of a value the run already returned.** Every `MediaRef` carries `engine.fetch_media` as its lazy fetcher, and the caller may call it any time after `run()` returns — the CLI does exactly that in `_save_outputs`. That download uses the full `http_timeout` and never checks the cancel flag, so a successful run keeps its output and a lane that finished keeps its media after a sibling lane timed out. Fetches the run itself makes (`local_fetcher`, inlining hosted audio, transcribe input) pass `run_bound=True` and do stop with the run.
- **The request a settled x402 deposit paid for.** Once `_settle_402` returns, real XNO has left the wallet. The retry carrying `x-x402-payment-id` is not cancelled by the deadline: a settled payment with no request ever sent is a money bug. Its budget is bounded, though, because it runs on a worker the run may already have abandoned. `_redeem_timeout()` gives it whatever the run has left, never less than `redeem_grace` (15 s) and never more than `http_timeout` (120 s). Exactly:

| run state | socket budget for the paid retry |
|---|---|
| no `timeout=` (no deadline) | `http_timeout` — 120 s, unchanged |
| live, more than 120 s of deadline left | 120 s |
| live, 60 s of deadline left | 60 s |
| live, less than 15 s left | 15 s (`redeem_grace`) |
| cancelled or past the deadline | 15 s (`redeem_grace`) |

So a run WITH a deadline can give the paid retry less than `http_timeout`. That is intentional and it is not a loss: the run dies at its deadline whatever this call does, and `redeem_grace` is the floor that keeps the request itself alive. The full 120 s on an abandoned worker re-created the process hang at 120 s.

### Traceability of a sent deposit
The engine keeps a ledger of every deposit it asked the callback to send: `node_id`, `payment_id`, `amount`, `pay_to`, `explorer_url`, `trace`, `status`, `send_error`, `redeemed`. `redeemed` turns True only when the request the deposit paid for answered 2xx; a settled deposit whose paid retry answered 500 stays unredeemed, so the payment id reaches the node's error message next to the API's own message.

The record is written BEFORE the callback fires, because the worker thread can be abandoned at any point after that and an abandoned future's exception surfaces nowhere. It therefore opens at `status="sending"`, which claims nothing. It becomes `"sent"` only when the callback RETURNS, and `"failed"` with a `send_error` when the callback raises. A ledger that said "sent" for a wallet callback that failed would send a person hunting an explorer for money that never moved.

`_record_payment` re-checks the cancel flag under the same lock `cancel()` latches it with. So once `cancel()` returns the ledger cannot grow, and every deposit that will ever exist is already visible to the caller that cancelled.

`Workflow` copies the ledger to `result.payments` and names every unredeemed deposit in the error message of the node that sent it, with one sentence per state: "a Nano deposit was already sent …", "a Nano deposit may have been sent …" (callback still in flight), or "no Nano deposit was sent (payment …): the wallet callback failed — …". So a user who sent XNO gets the payment id and explorer URL even when the run timed out and the worker's `NodeCancelled` went nowhere, and a user whose wallet failed is never told money moved.

## Execution (runGraph 3000-3133)
1. Alias/filter nodes (materialize): audio→tts, drop unknown types + orphaned links, migrate music/tts inbound "text" port → "prompt".
2. Kahn topological order; cyclic → error naming the cyclic nodes.
3. Concurrency: node starts when ITS deps finish (siblings run concurrently). Library: same semantics (asyncio / Promise per node).
4. Input resolution per node: for declared ports, value = srcNode.out[from.port]. Dynamic families: img\d+ / image\d*, vid\d+, clip\d+, audio, endframe, ref\d+, frame\d+ (vframes outputs). ANY other inbound link = FIELD OVERRIDE: run with fields = {...fields, [port]: value} (that's how wired prompt/system/lyrics/q override typed values).
5. Node failure: record error for that node, continue independent lanes (library default: collect; raise at end if a sink failed — see DESIGN).
6. comment nodes never run. Fixed-seed skip-cache: optional for a library (stateless one-shot runs don't need it) — SKIP in v1.

## Local nodes to IMPLEMENT (pure logic)
- text: out.text = fields.text
- upload/aupload/vupload: out = the stored/provided data URL. Empty + author-marked optional
  (`fields.optional` true or "true") = out is "" and the run continues; consumers drop empty
  media. Empty and NOT optional = error.
- choice: options = fields.options.split("\n") non-empty trimmed; out = fields.selected if in options else first; error if no options
- join: [a,b].filter(non-empty).join(sep) where sep = fields.sep ?? " ", literal "\\n" in sep means newline
- comment: skip

## Local media nodes (on-device; require ffmpeg on PATH — soft dependency, not an npm package)
Implemented in `src/local-media.mjs` (JS) / `nanoodle/local_media.py` (Py). Behaviour mirrors the browser:

- **resize** — `mode` fit|fill|exact, width/height; fit never upscales; PNG stays PNG else JPEG.
- **vframes** — `frames` 1–12, `gap` seconds, `dir` end|start; emits `frame1..frameN` JPEG data URLs.
  - **wiredFramesFloor**: before run (and in `derive_settings` min), raise `fields.frames` up to the highest outbound `frameK` wire (clamped 1..12). Graphs saved with `frames=1` but a wire from `frame3` would otherwise starve the consumer after paid upstream steps.
- **combine** — clip1../vid1.. inputs (≥2), sorted by port number then name, de-duped; `dedup` drops ~1 frame from subsequent clips; concat → mp4.
- **soundtrack** — video+audio; `loop` loops audio to fill video length; mux → mp4.
- **trim** — audio → mono WAV @ 16 kHz; `start` + `length` (default 30s when blank).
- **extractaudio** — video → mono WAV @ 16 kHz; blank `length` = whole clip after start.

Missing ffmpeg → clear error naming the binary. No `UnsupportedNodeError` for these types. Local media subprocesses honour the workflow deadline (cancel check before each op / between vframes frames; remaining time caps ffmpeg timeouts).
