#!/usr/bin/env python3
"""x402 example — accountless run with a programmatic payment callback.

The callback receives the invoice dict. Plug your own Nano signer in the marked
section. This script does NOT broadcast a transaction — it prints the fields you
would send, then returns so the engine can wait for a deposit (which will time
out unless you send from a real wallet).

Usage:
  python examples/x402/pay_with_wallet.py [graph.json|share-url] [prompt]
"""

from __future__ import annotations

import json
import sys
import time

from nanoodle import Workflow


def pay_with_my_wallet(inv: dict) -> None:
    # ── 1. Inspect before sending ───────────────────────────────────────────
    summary = {
        "paymentId": inv.get("paymentId"),
        "payTo": inv.get("payTo"),
        "amountRaw": inv.get("amountRaw"),  # integer string; 1 XNO = 10**30 raw
        "amount": inv.get("amount"),
        "amountUsd": inv.get("amountUsd"),
        "uri": inv.get("uri"),              # nano:ADDRESS?amount=RAW
        "expiresAt": inv.get("expiresAt"),  # epoch ms
        "explorerUrl": inv.get("explorerUrl"),
    }
    print("invoice received:", file=sys.stderr)
    print(json.dumps(summary, indent=2), file=sys.stderr)

    expires = inv.get("expiresAt")
    if expires is not None and time.time() * 1000 > expires:
        raise RuntimeError("invoice already expired — refuse to send")

    # ── 2. YOUR signer does the send (never pass a seed into nanoodle) ──────
    #
    #   my_wallet.send(to=inv["payTo"], amount_raw=inv["amountRaw"])
    #
    # Keep amountRaw as a string — raw is a 30+ digit integer.
    # Or hand the URI to any Nano wallet that understands nano: deep links:
    #   open_external(inv["uri"])
    #
    # Until a real deposit lands, the engine will poll completeUrl and
    # eventually raise on expiry. That is expected for this demo stub.

    print(
        "\n(stub) no send performed — open a real wallet and pay:\n"
        "  %s\n"
        "then wait; nanoodle will resume when the deposit is detected.\n"
        % inv.get("uri"),
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    graph = argv[0] if argv else "noodle-graph.json"
    prompt = argv[1] if len(argv) > 1 else "hello from x402 wallet example"

    wf = Workflow.load(graph, api_key="", payment=pay_with_my_wallet)

    def progress(evt):
        if evt["type"] == "node-start":
            print("▶ %s" % evt["name"], file=sys.stderr)
        elif evt["type"] == "node-done":
            print("✓ %s" % evt["name"], file=sys.stderr)

    result = wf.run({"Text": prompt}, on_progress=progress)

    for o in wf.outputs:
        value = result.outputs.get(o.key)
        if hasattr(value, "mime"):
            print("%s: (media %s)" % (o.key, value.mime or "binary"))
        else:
            print("%s: %s" % (o.key, value))
    print("cost: $%s" % result.cost_usd, file=sys.stderr)
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
