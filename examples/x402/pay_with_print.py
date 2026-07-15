#!/usr/bin/env python3
"""x402 example — accountless workflow run with a printed Nano invoice.

Mirrors ``python -m nanoodle run … --pay``: each paid NanoGPT call prints the
address + nano: URI on stderr and waits for the deposit. No API key, no account.

Usage:
  python examples/x402/pay_with_print.py [graph.json|share-url] [prompt]

Defaults: ./noodle-graph.json (download from nanoodle.com or copy from the JS
``nanoodle init`` scaffold).
"""

from __future__ import annotations

import os
import re
import sys

from nanoodle import MediaRef, Workflow


def pay_printer(inv: dict) -> None:
    # accountless x402: print the Nano invoice on stderr and let the engine wait
    # for the deposit. The send happens in the user's own wallet — never here.
    usd = " (~$%s)" % inv["amountUsd"] if inv.get("amountUsd") is not None else ""
    amount = inv.get("amount") or (str(inv.get("amountRaw")) + " raw")
    print("", file=sys.stderr)
    print("⚡ payment required: %s%s" % (amount, usd), file=sys.stderr)
    print("send with your Nano wallet:", file=sys.stderr)
    print("  %s" % inv.get("payTo"), file=sys.stderr)
    print("  %s" % inv.get("uri"), file=sys.stderr)
    if inv.get("explorerUrl"):
        print("  explorer: %s" % inv["explorerUrl"], file=sys.stderr)
    print("waiting for the deposit… (Ctrl-C aborts)\n", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    graph = argv[0] if argv else "noodle-graph.json"
    prompt = argv[1] if len(argv) > 1 else None

    # api_key="" stays explicitly keyless — None would fall back to $NANOGPT_API_KEY.
    wf = Workflow.load(graph, api_key="", payment=pay_printer)

    inputs = {}
    if prompt is not None:
        inputs["Text"] = prompt

    def progress(evt):
        if evt["type"] == "node-start":
            print("▶ %s (%s)" % (evt["name"], evt["node_id"]), file=sys.stderr)
        elif evt["type"] == "node-done":
            print("✓ %s — %d ms" % (evt["name"], evt.get("ms") or 0), file=sys.stderr)
        elif evt["type"] == "node-error":
            print("✗ %s — %s" % (evt["name"], evt.get("error")), file=sys.stderr)

    result = wf.run(inputs, on_progress=progress)

    out_dir = "noodle-out"
    for o in wf.outputs:
        value = result.outputs.get(o.key)
        if isinstance(value, MediaRef):
            os.makedirs(out_dir, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", o.key).strip("_") or "output"
            path = os.path.join(out_dir, "%s.%s" % (safe, value.suggested_extension()))
            value.save(path)
            print("%s: %s" % (o.key, path))
        else:
            print("%s:\n%s" % (o.key, value))

    approx = "" if result.cost_exact else "≥ "
    cost_line = "cost: %s$%s" % (approx, result.cost_usd)
    if result.remaining_balance is not None:
        cost_line += " · balance: $%s" % result.remaining_balance
    print(cost_line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("aborted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as e:
        print("error: %s" % e, file=sys.stderr)
        raise SystemExit(1)
