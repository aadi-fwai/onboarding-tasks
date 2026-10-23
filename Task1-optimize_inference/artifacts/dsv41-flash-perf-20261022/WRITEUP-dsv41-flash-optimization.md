# Tuning DeepSeek V4.1 Flash to a real workload

Onboarding exercise: take a real sizing request, size it, stand it up, and move one lever
at a time until the targets hold. Four live deployment experiments and one on-demand sweep.
Everything below is measured unless marked otherwise.

## The workload

An agentic coding workload from a recent `#perf-agent` sizing request — long prompts with a
shared prefix across turns, modest output, TTFT and throughput as the priority metrics.

| Input | Value |
|---|---|
| Prompt / cached / output | 55,000 / 49,500 (90% hit) / 1,000 tokens |
| Fleet load | 3 QPS |
| Targets | decode >= 400 tok/s, TTFT p50 <= 600 ms, e2e mean <= 3.1 s |
| Model | `accounts/fireworks/models/deepseek-v4p1-flash` (`deepseek_v41`, 552B, FP4) |

The request as written asked for e2e <= 1.5 s, which is unreachable by construction: 1,000
output tokens at the 400 tok/s floor is 2.5 s of decode before TTFT. Reconciled to 3.1 s
(0.6 s TTFT + 2.5 s decode) and flagged back to the requester. **Catching that before
sizing was the single highest-value thing in the whole exercise** — sizing against a
self-contradictory budget produces a config that fails validation by construction.

## What was measured

| Experiment | Config | Cache | QPS/GPU | Decode | TTFT mean | TTFT p95 | E2E | Fleet | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `bmag9uba` | tp2, 2xB300 inline, 6.176.1 | 90% | 0.4109 | 408.3 | 204 ms | 292 ms | 2,657 ms | 8 GPU | **PASS** |
| `xkpw07es` | tp2, 2xB300 inline, 6.176.1 | 50% | 0.2181 | 469.7 | 419 ms | 491 ms | 2,552 ms | 14 GPU | **PASS** |
| `dicchhaw` | tp2, 2+2 disagg, 6.176.1 | 90% | 0.2932 | 398.7 | 263 ms | 295 ms | 2,775 ms | 12 GPU | **FAIL** |
| `xpok1fmt` | tp4-inline, 4xB300, 7.21.0 | 90% | 0.4152 | 409.8 | 189 ms | 292 ms | 2,634 ms | 8 GPU | **PASS** |

Each ran 1,110 s (30 s + 180 s ramp + 900 s steady) in `pyroworks` at the config's own
per-replica rate, judged over the steady window against the customer's targets rather than
against the estimate. `dicchhaw` failed on decode by 0.3% — thin, but it missed while also
costing 50% more GPUs, so the conclusion is unambiguous.

## Findings

**Prompt caching dominates everything else.** 90% to 50% hit rate costs 8 -> 14 GPUs and
*flips the topology ranking*: at 90% inline wins, at 50% the cheapest row becomes 2+2
disaggregated. Session affinity across four replicas is therefore load-bearing, and it is
the one input a deployment experiment cannot verify — the experiment asserts the hit rate
via `--prompt-cache-max-len` rather than earning it through routing.

**Tensor-parallel width scales throughput linearly, so it trades latency and never cost.**
Going 2 -> 4 ranks multiplied per-replica QPS by 2.019 — it doubled the decode batch from
2.0 to 4.0 while holding per-request speed. QPS per GPU moved +1.1%. You get 15 ms of TTFT
and 23 ms of e2e for free, and zero GPUs back.

**Disaggregated prefill lost twice.** Measured 6% below inline at 2+2 with worse TTFT
(263 ms vs 204 ms), and estimated 6.3% below at 4+4. At a 90% cache hit there are only
~5,500 uncached tokens per request — not enough prefill work to justify dedicated prefill
hardware. Note the 2+2 row was a *modeled* tiered row (`prefill_source=inline`) and had the
worst prediction error of any run at +26% TTFT, which is the characteristic failure mode of
a generator/prefiller pair that was never deployed together.

**Draft depth was already optimal.** The periodic sweep measures the tp2 cell at `[0, 5]`
and interpolates, so the estimator had already searched the full range and chosen 4. Worth
knowing *why* deeper is not better here: the config delivers 408 tok/s against a 400 floor,
so only 2% of headroom exists, and every token/s above the target is batch capacity spent
for nothing. The unvalidated 5-draft rows show the failure mode — 465 tok/s per request,
16% of pure waste, paid for by halving the batch and doubling the fleet.

**Precision is not a lever for this model.** Every measured cell on every hardware type is
FP4. There is no FP8 or BF16 sweep to compare against.

**Image version is not a lever either.** Re-measuring the same tp2 cell on 7.21.0 instead of
6.176.1 moved QPS/GPU by +0.15%, inside the noise. This had previously been entangled with
width and could not be tested separately.

**8 GPUs is the floor.** Reaching 6 requires 0.5000 QPS/GPU, or +22% over measured. The best
lever found delivered +1.1%. For reference the largest single improvement available anywhere
— moving the acceptance assumption from `code-30` (3.15 tokens/forward) to `coding-long-20`
(3.44, +9.2%) — is worth +7.6%, so you would need roughly three times the best acceptance
gain that exists for this model.

## On estimator accuracy

Throughput modelling was essentially exact: **QPS/GPU within 0.1% on all four runs, and
every predicted fleet count reproduced on the hardware.** Decode within 1.3%, e2e within
2.5%.

TTFT is the exception, and the error is structured rather than random:

| Run | TTFT error |
|---|---|
| tp2 @ 90% cache | +14.6% |
| tp4-inline @ 90% cache | +14.5% |
| tp2 @ 50% cache | +0.1% |
| 2+2 disagg (modeled row) | +26.2% |

Two independent configs, same magnitude, confined to the high-cache regime. Meanwhile the
*relative* comparison held precisely — the estimator predicted tp4-inline would beat tp2 by
13 ms of TTFT and it delivered 15 ms. **Absolute TTFT is optimistic when most of the prompt
is cached; TTFT rankings stay trustworthy.** I had predicted the optimism would compound at
low cache and it did not, so do not carry a correction factor forward.

## The economics worth repeating to a customer

At 1,000 output tokens, **92% of end-to-end latency is decode** (204 ms TTFT + 1,000 x 2.449 ms).
A latency target is therefore a decode-speed target, decode speed is bought with smaller
batches, and batch size is throughput. Concretely, tightening e2e from the measured 2.657 s
to 2.5 s — a 5.9% gain — costs **40 GPUs instead of 8**. And a TTFT p95 of 190 ms is below
what any measured config reaches at any load (best mean TTFT anywhere is 162-172 ms at 90%
cache, 149 ms at 95%).

Meanwhile, trimming average output from 1,000 to 600 tokens — inside the request's own
stated 600-1,200 range — gets e2e to **1.67 s on the same 8 GPUs**. When a customer wants
lower latency on a shape like this, the conversation is about output length, not hardware.

## Recommendation

**Config A** — 2xB300 FP4, inline `tp2`, 4 draft tokens, release image 6.176.1, 4 replicas,
8 GPUs. Validated, backed by release checkpoint data, and its GPU count is the one that does
not move with the acceptance assumption (the disagg rows swing 12 -> 8 and 16 -> 8 between
the two acceptance datasets; config A stays at 8 under both).

**`tp4-inline`** — 4xB300, 2 replicas, same 8 GPUs — is a validated alternative worth 15 ms
of TTFT and 23 ms of e2e. Caveat: it rests on unvalidated on-demand sweep data at image
7.21.0, not the release checkpoint.

## What the exercise could not settle

- **Real acceptance.** Experiments force it via `--enable-acceptance-override`, so speculation
  quality is asserted, not tested. Needs production dashboards or a run without the override.
- **Whether affinity earns 90%.** Needs a multi-replica deployment behind a router with
  affinity on, checking `tokens_cached_prompt_total / tokens_prompt_total`.
- **The token-length distribution.** Traffic was fixed at 55,000/1,000, so the 50-90K input
  spread and the 8K output tail went unexercised. Needs `load_test.yml` against a pinned
  deployment.
- **Fleet behaviour.** Every experiment ran one replica; the 4-replica figure is arithmetic
  on top of a per-replica measurement.

## Side effect worth knowing about

The periodic `deepseek_v41-550b` config pinned `tp4` to `deploy_mode: disagg`, so the
estimator structurally could not offer a 4-rank TP generator as an inline row. An on-demand
sweep (4 cells, ~5h40m, no reruns) added `tp4-inline` plus a tp2 control on the same image.
That row is now permanent coverage for every future sizing of this model, and the tp2
control is what retired the image-version hypothesis.
