# Task 1 — Chat notes (Perf Agent skill study)

Source: local agent chat "Understanding the perfagent skill" (Oct 21, 2026).
The hands-on work (sizing, 4 deployment experiments, sweep, write-up) was done in cloud agent
run `bc-9d4c477f-6e6c-45d2-8b3d-b3721bd037a7`; its results are in
`artifacts/dsv41-flash-perf-20261022/` (see `WRITEUP-dsv41-flash-optimization.md`).

## Status

- Studied the `perfagent` skill (`fw-ai/fireworks/.cursor/skills/perfagent/SKILL.md`; private repo,
  text pasted into chat).
- Executed on a different `#perf-agent` request than the Readme example: DeepSeek V4.1 Flash,
  55k/90% cached/1k out, 3 QPS -> Config A tp2 x4 replicas = 8 GPUs (see Readme).

## Example workload from Readme (not the one executed)

| | p50 | p90 | p95 |
|---|---|---|---|
| Input tokens | 10k | 13k | 24k |
| Output tokens | 40 | 70 | 90 |
| Prompt cache hit | 60% | 75% | — |

Traffic: median 5 RPS, peak 40 RPS. TTFT p50 500 ms / p95 900 ms / hard ceiling 5 s.
Starting config: TP4, FP4 weights. Priority: TTFT and throughput.

## What Perf Agent is

Assistant for **sizing, validating, optimizing, and pricing** Fireworks LLM deployments
(FireAttention engine). Two workflows, picked up front:

1. **Creating a deployment** — from workload requirements.
2. **Optimizing an existing deployment** — given `accounts/<acct>/deployments/<id>`;
   requirements come from Prometheus instead of the customer.

### Tooling

- `firectl-admin` — `model list/get`, `deployment get`, `deployment-experiment create/get/end`
  (`--api-key`, `-s gateway-preview.fireworks.ai:443`). Experiments need a superuser key
  (`PERFAGENT_API_KEY="${E2E_MODEL_TEST_SECRET:-$FIREWORKS_API_KEY}"`).
- `perfagent_mcp` scripts: `estimator.py`, `data_sources.py`, `acceptance_probs.py`,
  `create_deployment.py`, `create_deployment_experiment.py`, `metrics.py`. They read the
  `fw-ai/configs` repo (`~/configs` or `PERFAGENT_CONFIGS_ROOT`).
- Prometheus metrics for live deployments (throughput, TTFT, queueing, KV-cache pressure,
  speculation acceptance).

### Sizing workflow

1. **Gather requirements** — model, avg prompt/gen length, gen tok/s *or* E2E latency
   (TTFT-only is rejected), cache hit rate. Missing = hard stop, ask, end turn. Ranges use the
   midpoint labeled `Assumed`.
2. **Resolve model** — `model list --no-paginate --kind HF_BASE_MODEL` + exact client-side match.
3. **Measured data** — `data_sources.py` → `MEASURED` (go) / `NOT MEASURED` (stop, no proxies).
4. **Acceptance probs** — `acceptance_probs.py`, pick `chat-30` vs `code-30` to match traffic.
5. **Estimate** — `estimator.py --all-sources [--qps N] --avg-cached-len <prompt*hit>`; ranked
   table, interpolates only within measured points.
6. **Present** — configs A/B/C with source, QPS/GPU, predicted TTFT/TPS/latency; user chooses.
7. **Validate** — deployment experiment (synthetic `traffic_generator` or mirrored
   `traffic_forker`), `CREATING → ACTIVE → COMPLETED`, compare measured vs predicted.

### Guardrails

- Never executes `deployment create`; only renders it via `create_deployment.py`.
- Experiment create/end need user confirmation; max 1 + 1 create attempts.
- Flags (`extra_args`, `image_tag`, `precision`, `num_draft_tokens`) only from the chosen
  estimator row — never copied from live deployments.
- Every recommendation names its data source; modeled tiered rows labeled as modeled.
- p95 targets sized as `target / 1.35` (predictions are means).
- Cost-first: assume fewest GPUs.
- Temp files in `$PERFAGENT_TMPDIR`; `exclude_prefiller=true` on request counters.

## Q&A from the chat

**Where are the Prometheus metrics hosted?**
Google Managed Prometheus (Cloud Monitoring). `metrics.py` queries GMP and is being ported to
Chronosphere; `build_dashboard_url` renders a Cloud Monitoring dashboard (pyroworks/fireworks only).

**Worked example — "size DeepSeek for 200 QPS":**
ask for missing prompt/gen/latency/cache → resolve exact model id → `data_sources.py` →
`acceptance_probs.py` (`chat-30`) → `estimator.py --qps 200 --avg-cached-len 2454`
(4096 × 0.6) → rank by `gpus_needed` → experiment with `--max-context-length 5120`
→ compare against customer targets (not the estimate).

**Experiment validation details**
- `--max-context-length` is mandatory (longest prompt+output); otherwise KV cache is sized for
  the full window → OOM or shrunken batch.
- Poll with the full name `accounts/<acct>/deploymentExperiments/<id>`.
- Analyze the steady window (skip first 30 s + 180 s ramp).
- Experiments force the dataset's acceptance rates, so they **cannot catch an acceptance gap**.

**Why not pass `gen_batch_size` as `--concurrency`?**
Batch size counts only requests decoding; concurrency counts everything in flight (prefilling +
decoding). Using batch size under-models load → deployment under-runs the estimate (higher
latency / lower QPS). Pass the client's real in-flight count.

**Flag-registry tuning** (`serving/perfagent_mcp/fireworks/perfagent/configs/flag_registry.yaml`)
- Extract signals (context length, TTFT vs ITL pain, concurrency, prefix reuse, spikiness).
- Match levers' `helps_when`; flag spelling comes from `surfaces` per engine (no cross-engine copy).
- Maturity: `validated` → may recommend (as diff + confirm if not on the row);
  `exploring`/`hypothesized` → experiment only; `model-config-candidate` → default will handle it.
- Relevant lever for this task: **prefill CUDA graph** (`deployment.extra_args.enable_prefill_cudagraph`)
  for prompts ≲8K when TTFT hurts.

**Existing-deployment variant**
Requirements from Prometheus (`tokens_prompt_per_request`, `tokens_generated_per_request`,
`latency_to_first_token_ms`, `latency_generation_per_token_ms` (mean only),
`tokens_cached_prompt_total / tokens_prompt_total`). `deployment get` is context only —
show a delta table vs the estimator row.

## How the Readme example would map (superseded — see artifacts for the executed run)

1. Inputs for estimator: avg prompt ≈ 10k (size for p95 24k context), gen ≈ 40,
   cache hit 60% → `--avg-cached-len 6000`, `--qps 40` (peak), `--max-context-length ~24.2k`.
2. TTFT p95 900 ms → size as 900 / 1.35 ≈ 667 ms mean.
3. Baseline TP4/FP4, then vary one lever at a time: TP, replicas, precision, spec decoding,
   prompt caching, prefill CUDA graph. Record TTFT vs throughput movement.
4. Write up results in `findings.md`.
