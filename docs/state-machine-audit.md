# Implicit state-machine audit — 2026-11-06

Scope: all `src/nanoodle` modules, CLI, local-media subprocesses, offline tests,
examples and release workflow. Baseline: `0ce5cf4` (`main`, package 0.7.0).
No open pull requests existed when the audit started. No live generation,
wallet callback, production data change or package upload was performed.

## Inventory

| Machine / ownership | States and transitions | Guards, recovery and evidence |
|---|---|---|
| Share loading (`share.py`, `Workflow.load`) | path or direct fragment → decode; short URL → bounded redirects → decode; strict decode failure → salvage → warning or error | Direct fragments are offline; redirect reads carry no API key. `test_share.py`, `test_graph_load.py`. Salvage changes only the recovered graph and surfaces a warning. |
| Graph materialization (`graph.py`) | persisted nodes/links → legacy aliases and port migration → scrubbed fields → executable graph; unknown nodes retained with warning → preflight error | Orphaned links dropped; duplicate dependency edges counted once; Kahn cycle detection precedes execution. `test_graph_load.py`, `test_media_placeholders.py`. |
| Interface derivation (`iodef.py`, `Workflow` caches) | unset cache → derived inputs/settings/outputs; name → exact key / alias / unique label → resolved or ambiguous error | Wired fields hidden; author-optional inputs may be empty; explicit empty optional values clear defaults. Derivation is cached per Workflow; direct graph mutation after deriving interface is outside the intended load/run API. `test_io_interface.py`, `test_optional_nodes.py`, `test_model3d.py`. |
| Run preparation (`Workflow.run`) | graph copy → inputs/settings/coercion/defaults → required/type/key/cycle checks → frame-count floor → scheduler | Each run copies graph fields; input or configuration failures do no paid work. Finite nonnegative timeout validation now precedes execution; zero is an immediate deadline. `test_state_machines.RunLifecycleTest`, existing IO/local-node tests. |
| Scheduler and pool (`workflow.py`) | pending → queued → future running → done/error/gated; failed dependency → error; closed gate → skipped; comment → skipped | Maximum eight daemon workers. Only scheduler writes terminal output status. Sinks determine RunError; partial outputs remain accessible. Ctrl-C/SystemExit now cancel and abandon outstanding work rather than joining an unlimited poll. `test_run_semantics.py`, `test_timeout_shutdown.py`, new interrupt regression. |
| Decide/gate (`decide.py`, engine) | validate → fit images → one request, or forward + reverse pick → merge → result; yes/no false + gate → gated → transitive skips | Failure takes precedence over a gate in a shared descendant. Each decision request records cost independently, including partial two-call failures. `test_decide.py`; fitting uses tracked subprocess lifecycle. |
| Cancellation (`Engine`, scheduler) | live → deadline/explicit cancellation latched → stop queued work and poll waits → result returned → background workers drain | Cancellation and opening payment records share a lock. Sleeps now always wait on the cancel event, including runs without deadlines. Returned costs and trim records are sealed against late worker writes; abandoned runs report an inexact cost floor. `test_state_machines.RunLifecycleTest`, `test_timeout_shutdown.py`, `test_x402.py`. |
| Transport/auth (`transport.py`, engine, `endpoint.py`) | request construction → response or connection failure → HTTP classification → parse | No automatic retry of paid generation POSTs. NanoGPT credentials stay on NanoGPT API requests; custom endpoints use their authored auth. Content downloads use separate credential/redirect rules. Existing `test_cost_errors.py`, `test_endpoint.py`, `test_content_url.py`. |
| Video/3D jobs (`_gen_video`) | submit once → run ID → pending/processing polls → completed URL, failed/canceled, permanent HTTP error, deadline or cap | Video intentionally has no default cap; 3D caps at 25 minutes. Status GETs retry transport failures, 408/409/425/429 and 5xx, and malformed JSON. Permanent other 4xx terminate; both cancellation spellings terminate. Missing output after completion is an error. `test_video_nodes.py`, `test_model3d.py`, `test_state_machines.PollLifecycleTest`. |
| Audio jobs (`_gen_audio`, `_poll_audio`) | submit → binary or direct URL, else run ID → polls → complete/error/cancellation/300-second cap | Same HTTP retry classification as video. An HTTP error body cannot masquerade as successful output. Queries preserve billing metadata. `test_audio_nodes.py`, `test_cleanvoice.py`, new poll regressions. |
| x402 (`x402.py`, engine, workflow) | initial 402 → validate invoice → ledger `sending` → wallet returns `sent` or raises `failed` → deposit detection → replayed result or one paid-ID repost → redeemed only on 2xx | Expired invoices, malformed expiry and invalid raw amount are refused before wallet invocation. ISO timezone offsets and fractional seconds are preserved. Callback failure means send outcome unknown; retain ID/address/explorer and advise verification before another payment. Once settled, the paid repost keeps its bounded budget even after deadline. No automatic second payment. `test_x402.py`, `test_state_machines.InvoiceLifecycleTest`. |
| Process/funds shutdown (`engine.py`, `local_media.py`) | money-critical entry → exit; active child → reaped or canceled/killed; interpreter exit → bounded money grace + child cleanup | Wallet/redeem sections get up to five seconds at exit; children are tracked and killed/reaped with a bounded grace. Clean voice's duration probe now joins this common lifecycle instead of untracked `subprocess.run`. Existing subprocess-exit tests plus new media lifecycle tests. |
| Local media (`local_media.py`) | resolve/download input → temporary files → probe/render → encode data URL → cleanup | Deadline/cancellation reaches shared child runner; each render has a process timeout. Temporary directories clean on normal completion/error. Missing ffmpeg is a local error. `test_local_media.py`, `test_local_nodes.py`, `test_trim_and_payment.py`, `test_timeout_shutdown.py`. |
| Media values/content URLs (`media.py`, engine) | inline data → decode; hosted URL → lazy download → optional MIME enrichment → save | Output download lifetime is independent of run deadline. Chained content URL resolution is bounded and protects auth across redirects. No persistent cache or automatic paid rerun. `test_media.py`, `test_content_url.py`, timeout media tests. |
| Prompt caps (`prompt_caps.py`, workflow) | known/catalog/learned limit → fit + disclosure; unknown rejection → learn for next run | Learning is in-memory, per Workflow; never resubmits the rejected call automatically. Chat token limits do not promise a character-cap fix. Warnings cannot fail execution. `test_prompt_caps.py`, prompt-trim coverage in existing run/CLI suites. |
| Cost ledger/estimate (`engine.py`, workflow, `estimate.py`) | unknown → response/header cost → per-node/total aggregation → snapshot; estimate → priced/estimated/unpriced | Known zero differs from unknown. Image response cost is now recorded before validating image output, preserving charges on malformed completion. Late workers cannot mutate the returned per-node cost while its total remains frozen. Estimation is pure catalog-data computation. `test_cost_errors.py`, `test_estimate.py`, new image/late-cost tests. |
| CLI (`__main__.py`) | parse/load → inspect or run → success/partial failure/preflight failure → output saving + JSON/text → exit | Failure JSON preserves accrued cost and payment trace; gated outputs are successful skips. `--pay` prints invoices; library never holds wallet keys. `test_cli.py`, existing failure/payment tests. |
| Distribution (`pyproject.toml`, workflow) | source commit → offline CI → optional manual build → separate package upload | Only workflow is offline unittest on push/PR, Python 3.11/3.12. No publish workflow, deployment configuration, or GitHub deployments. Public PyPI remains 0.7.0; a merge does not update installed packages. |

## Confirmed defects repaired

1. **High — cancellation could hang forever:** scheduler cleanup joined an unlimited
   video poll after KeyboardInterrupt/SystemExit. Cancel before nonblocking shutdown;
   make every poll sleep interruptible.
2. **High — wallet callback exception falsely proved no transfer:** a callback can
   broadcast and then lose confirmation. Preserve the `failed` callback status for
   compatibility, but report unknown money movement and the verification reference.
3. **High — invoice validation happened too late:** expired invoices entered the
   wallet callback; offset timestamps could turn into an absent expiry. Reject stale,
   malformed-expiry and invalid-amount invoices before opening the ledger/callback.
4. **Medium — polling lost terminal transitions:** permanent 4xx were retried, video
   potentially forever; canceled audio and British-spelled cancelled jobs also polled
   to timeout. Shared classification now terminates these and retains transient retries.
5. **Medium — HTTP error could produce false successful audio:** success-shaped JSON
   in a 503 body was accepted. HTTP classification now precedes result parsing.
6. **Medium — returned accounting changed asynchronously:** abandoned workers mutated
   `NodeRun.cost_usd` after return without updating `RunResult.cost_usd`; exactness could
   be falsely true. Seal reporting and mark abandoned-run totals as a floor.
7. **Medium — zero/invalid timeouts:** zero disabled the deadline and could start paid
   work; NaN, infinity and negative values were accepted. Validate early; honor zero.
8. **Medium — untracked Clean voice probe:** its subprocess bypassed cancellation and
   exit cleanup. Route it through the tracked child runner, preserving advisory probe
   fallback while allowing cancellation to propagate.
9. **Medium — failed image output hid reported charges:** empty/malformed image output
   raised before recording the response's cost. Record cost before output validation.

## Verification

- Baseline: 413 tests, passed on Python 3.14; 12 optional ffmpeg-dependent tests skipped.
- New regression file initially failed on baseline (25 failing subcases and 3 errors
  across the first 10 tests), including real asynchronous late-cost reproduction.
- After fixes: full offline suite passes on Python 3.11 and 3.12, 428 tests on each;
  12 ffmpeg-dependent tests skipped on this host. Focused suite: 42 tests passed.
- Wheel and sdist built using the declared patched build-tool constraints; wheel
  installed in an isolated environment and passed a local text workflow / zero-deadline
  smoke test. Artifacts retain version 0.7.0 for local verification only, not upload.
- Tests use injected transports/local mock HTTP servers only. CI has no API secrets
  or paid calls. Existing timeout tests exercise child interpreter exit and payment
  redemption lifetimes with synthetic wallet callbacks.

## Remaining limits / recovery boundaries

- The engine is in-memory, with no durable job journal, payment reconciliation API,
  crash resume or cross-process exactly-once guarantee. Persist returned payment IDs
  in the calling application and verify an uncertain wallet outcome before a new run.
  A whole-workflow rerun may generate and charge again; it is not a resume operation.
- Local timeout/cancellation stops observation, not an already submitted remote job.
  Remote work may complete and charge later. Costs returned at abandonment are a floor.
- Bounded exit grace cannot guarantee completion of an arbitrarily blocked external
  wallet callback; SIGKILL cannot run cleanup. The library cannot atomically combine an
  arbitrary external wallet and its memory ledger without a new persistence contract.
- Transient failures and unknown job statuses can continue indefinitely for video
  without a caller-supplied timeout, intentionally retaining long-render support.
- Media save is ordinary filesystem output, not a transactional persistent workflow.
  Optional ffmpeg behavior needs its existing CI/integration coverage when available.
- Interface caches assume the loaded graph remains unchanged through the public API.
  Independent run copies are safe; direct concurrent mutation of public graph/catalog
  objects or user callbacks is not a transactional operation.
- Package publication is a separate irreversible release. This audit prepares verified
  source/builds; it does not select a new version or upload to PyPI.
