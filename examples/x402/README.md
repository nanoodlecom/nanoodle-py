# x402 usage examples — pay per run in Nano (no account)

NanoGPT supports **x402** accountless payments: call the API with no key, get an
HTTP 402 invoice, settle it in **Nano (XNO)** — instant, feeless — and the call
completes. nanoodle wires that up end to end.

The library **never holds funds or keys**. You supply a `payment` callback; the
send happens in *your* wallet/signer (or a human paying the printed invoice).
Passing a seed or private key raises.

## How the flow works

```
1. Request with x-x402: true  (no Authorization)
2. API answers HTTP 402 with payment options (Nano among them)
3. nanoodle parses the Nano invoice → calls your payment(invoice)
4. You send amountRaw raw units of XNO to payTo (or show invoice["uri"])
5. nanoodle polls the complete URL until the deposit is seen
6. Result is returned (replayed body, or re-POST with x-x402-payment-id)
```

Each API call pays **at most once**. Graphs with several paid nodes produce one
small invoice per node. The invoice dict is field-identical to nanoodle-js's, so
payment callbacks port between the two libraries unchanged.

## Prerequisites

- Python ≥ 3.9
- A Nano wallet with a tiny amount of XNO (cents)
- This package installed (`pip install nanoodle` or `pip install -e .` from a checkout)

No NanoGPT account. No API key.

## 1. CLI — print the invoice and wait

Fastest path. `--pay` ignores any configured key, prints a Nano address +
`nano:` URI on stderr for each paid call, and resumes when the deposit lands:

```bash
# scaffold the starter graph if you don't have one (JS CLI), or download from nanoodle.com
python -m nanoodle run noodle-graph.json \
  --input Text="a cozy ramen shop on a rainy night" \
  --pay --out ./noodle-out

# or any share link
python -m nanoodle run "https://nanoodle.com/#g=..." --input Text="hello" --pay
```

## 2. Library — human pays (print invoice)

Same behavior as the CLI, as a script:

```bash
# from a checkout with `pip install -e .`
python examples/x402/pay_with_print.py noodle-graph.json "a cozy ramen shop on a rainy night"
```

See [`pay_with_print.py`](./pay_with_print.py).

## 3. Library — programmatic wallet / signer

Your `payment` callback receives the invoice dict and must send XNO itself.
This example logs the invoice and shows where to plug a signer — it does **not**
broadcast a transaction:

```bash
python examples/x402/pay_with_wallet.py noodle-graph.json "hello from x402"
```

See [`pay_with_wallet.py`](./pay_with_wallet.py).

## Invoice shape

The dict your `payment` callback receives (field-identical in nanoodle-js):

| Field | Meaning |
|---|---|
| `scheme` | always `"nano"` |
| `paymentId` | e.g. `pay_…` |
| `payTo` | `nano_…` destination address |
| `amountRaw` | integer raw units as a **string** (1 XNO = 10³⁰ raw) |
| `amount` | human string, e.g. `"0.00018406 XNO"` |
| `amountUsd` | USD estimate (float) or `None` |
| `uri` | ready-to-scan/click `nano:ADDRESS?amount=RAW` |
| `expiresAt` | epoch **ms** (or `None`) |
| `statusUrl` | poll deposit status |
| `completeUrl` | settle / fetch the stored result |
| `explorerUrl` | block explorer link when present |
| `description` | optional |
| `requestHash` | optional request binding |

Helper: `parse_nano_invoice(body, base_url)` from `nanoodle`.

## Critical: stay keyless

When using `payment`, pass **`api_key=""`** (empty string), not `None`.
`None` falls back to `$NANOGPT_API_KEY`, which **disables** the x402 settle
path and charges the account instead. The CLI `--pay` flag does this for you.

```python
wf = Workflow.load(graph, api_key="", payment=my_pay_callback)
```

## Safety

- Never put a seed or private key in `payment` — it must be a callable.
- Confirm `amount` / `amountUsd` before sending in automated wallets.
- Respect `expiresAt`; expired invoices fail cleanly.
- A second 402 after a successful settle is an error, never a second send.

## Sibling package

Same examples for JavaScript (with terminal QR): [nanoodle-js/examples/x402](https://github.com/nanoodlecom/nanoodle-js/tree/main/examples/x402).
