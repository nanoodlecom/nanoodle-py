# nanoodle (Python)

Re-execute [nanoodle](https://nanoodle.io) AI workflows server-side against the
[NanoGPT](https://nano-gpt.com) API. **Zero runtime dependencies** — Python >= 3.9, stdlib only.

Build a workflow visually in the nanoodle editor, hit **💾 Save** to download
`noodle-graph.json`, then run it anywhere Python runs:

```python
from nanoodle import Workflow

wf = Workflow.load("noodle-graph.json", api_key="...")   # or env NANOGPT_API_KEY
result = wf.run({"Text": "a cozy ramen shop"})
result["Image"].save("out.png")
```

## Inspecting a workflow

```python
wf.inputs     # [InputSpec(key="Text", node_id="n1", field="text", kind="textarea", ...)]
wf.outputs    # [OutputSpec(key="Image", node_id="n3", type="image", ports=["image"])]
wf.settings   # [SettingSpec(key="n3.size", kind="select", default="1k", ...)]
```

Input names resolve flexibly (case-insensitive): the node's custom name, `nodeId.field`
(e.g. `"n2.prompt"`), or the input label when unique. A workflow with exactly one
required input also accepts a bare value: `wf.run("hello")`.

Media inputs accept a `data:`/`https` URL string, raw `bytes`,
`{"data": bytes, "mime": "image/png"}`, or `media_from_file("photo.jpg")`.

## Results

```python
result = wf.run({"Text": "..."}, settings={"n3.size": "1k"}, timeout=600,
                on_progress=lambda evt: print(evt))
result["Image"]            # MediaRef (media) or str (text); also keyed by node id
result.outputs             # {key: value}
result.cost_usd            # total spend reported by NanoGPT (result.cost_exact = False
                           #  when any call omitted a price — treat the total as a floor)
result.remaining_balance   # latest balance the API reported
result.nodes               # per-node {status, out, error, cost_usd, ms}
```

`run()` raises `RunError` when an output node failed — partial results stay on
`error.result`. `UnsupportedNodeError` is raised *before any paid call* when the
graph contains a node this library can't execute.

## CLI

```
python -m nanoodle inspect graph.json
python -m nanoodle run graph.json --input Text="a cozy ramen shop" \
    --input n2.system=@system.txt --set n3.size=1k --out ./out [--json]
```

`--out DIR` saves media outputs to files; `--json` prints a machine-readable result.

## Supported nodes

| Runs | Nodes |
|---|---|
| Local (free) | text, upload, aupload, vupload, choice, join, comment |
| NanoGPT API | llm, vision, image, edit, inpaint, draw, tvideo, ivideo, vedit, lipsync, music, remix, tts (speech), transcribe |
| **Not supported (v1)** | resize, vframes, combine, soundtrack, trim, extractaudio — these do local media processing that requires the nanoodle browser app; `run()` fails fast with `UnsupportedNodeError` |

Notes:
- Model ids pass through verbatim; no catalog fetch is needed to run.
- Media is inlined as base64 (no upload endpoint) with a ~4.4 MB per-request cap
  (~3.5 MB for transcription).
- Inpaint masks are passed through verbatim (white = repaint); the browser app
  additionally composites the mask at source size.
- No telemetry, no analytics, and your API key is never logged.

## Cost

Every generation bills your NanoGPT balance. `result.cost_usd` totals what the
API reported back; the image endpoint reports no price, so runs that include it
show `cost_exact=False`. `scripts/live-spot-check.py` is an opt-in, hand-run
smoke test against the live API (cheap llm-only by default, `--image` adds one
image generation). It is never run by CI.

## Development

```
python -m unittest discover -s tests -t . -v
```

All tests are offline — they run against a local mock NanoGPT server
(`tests/harness/`). Never point tests at nano-gpt.com.

Build workflows at [nanoodle.io](https://nanoodle.io). MIT licensed.
