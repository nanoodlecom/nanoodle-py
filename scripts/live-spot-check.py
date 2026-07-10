#!/usr/bin/env python3
"""OPT-IN live spot check against the real NanoGPT API. Spends real money (a
fraction of a cent for the default text run; more with --image).

NEVER run by CI or pre-commit hooks. Run it by hand:

    NANOGPT_API_KEY=... python3 scripts/live-spot-check.py
    python3 scripts/live-spot-check.py --env-file ../nanoodle/.env
    python3 scripts/live-spot-check.py --image   # also runs the starter graph image step

Default run: text -> llm (zai-org/glm-5.2, maxTokens 60) and prints the text + cost.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from nanoodle import Workflow  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "tests", "fixtures", "starter-graph.json")

TEXT_ONLY_GRAPH = {
    "v": 1,
    "nodes": [
        {"id": "n1", "type": "text", "fields": {"text": "a cozy ramen shop on a rainy night"}},
        {"id": "n2", "type": "llm",
         "fields": {"model": "zai-org/glm-5.2", "maxTokens": "60",
                    "system": "Reply with one short vivid sentence."}},
    ],
    "links": [
        {"id": "l1", "from": {"node": "n1", "port": "text"}, "to": {"node": "n2", "port": "prompt"}},
    ],
}


def load_env_file(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="read NANOGPT_API_KEY from this .env file")
    parser.add_argument("--image", action="store_true",
                        help="ALSO run the starter graph's image step (costs more)")
    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file)
    if not os.environ.get("NANOGPT_API_KEY"):
        print("error: set NANOGPT_API_KEY (or pass --env-file)", file=sys.stderr)
        return 1

    def progress(evt):
        if evt["type"] == "node-start":
            print("  ▶ %s" % evt["name"], file=sys.stderr)

    print("== live text run (zai-org/glm-5.2, maxTokens 60) ==")
    wf = Workflow.from_dict(TEXT_ONLY_GRAPH)
    result = wf.run(on_progress=progress)
    print(result["LLM"])
    print("cost: %s$%.5f  balance: %s" % ("" if result.cost_exact else "~",
                                          result.cost_usd, result.remaining_balance))

    if args.image:
        print("\n== live starter-graph run (llm + image) ==")
        wf2 = Workflow.load(FIXTURE)
        result2 = wf2.run(settings={"n2.maxTokens": "60"}, on_progress=progress)
        out = os.path.join(os.getcwd(), "live-spot-check.png")
        result2["Image"].save(out)
        print("image saved: %s" % out)
        print("cost: %s$%.5f  balance: %s" % ("" if result2.cost_exact else "~",
                                              result2.cost_usd, result2.remaining_balance))
    return 0


if __name__ == "__main__":
    sys.exit(main())
