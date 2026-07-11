---
name: poster-generator
description: Generate a poster image from a short idea — an LLM expands the idea into a detailed image prompt, then an image model renders it. Use when the user asks for a poster, illustration, or promo image from a one-line concept.
---

# Poster generator

Runs the bundled nanoodle workflow `workflows/poster.noodle-graph.json` against the NanoGPT
API: a text input (`Idea`) feeds an LLM that writes a vivid image prompt, which feeds an
image model that renders the poster (`Poster`). Requires Python >= 3.9 and the `nanoodle`
PyPI package (stdlib-only, `pip install nanoodle`).

## API key

The run needs a NanoGPT API key. Use whichever is available:

- `NANOGPT_API_KEY` already set in the environment — just run the command.
- A `.env` file (e.g. this skill's directory or the project root) containing
  `NANOGPT_API_KEY=...` — add `--env-file <path-to-.env>` to the command.

Never print the key.

## Run

```sh
python -m nanoodle run workflows/poster.noodle-graph.json \
  --input "Idea=<the user's poster idea, e.g. a cozy ramen shop on a rainy night>" \
  --env-file .env \
  --out ./poster-out
```

(With the package installed, the `nanoodle-py` console script is equivalent:
`nanoodle-py run …`. Paths are relative to this skill's directory; prefix them if running
from elsewhere. Drop `--env-file .env` when `NANOGPT_API_KEY` is already in the environment.)

Inspect the interface anytime with:

```sh
python -m nanoodle inspect workflows/poster.noodle-graph.json
```

## Outputs

- `./poster-out/Poster.png` — the rendered poster image. Hand this file to the user.
- The CLI also prints the run's total cost to stderr when it finishes.

## Cost

Each run costs about **$0.04** in NanoGPT credit (the image-generation step dominates; the
LLM prompt-writing step is a fraction of a cent).
