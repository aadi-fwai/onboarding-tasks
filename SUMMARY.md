# Onboarding exercises — summary for mentor & manager

Three exercises, all in the `pyroworks` account, each ending in a working
deployment tested with a real inference call. Detailed artifacts live in each
task folder; this page is the overview.

| # | Exercise | One-line outcome | Status |
|---|---|---|---|
| 1 | Optimize a deployment | Sized + validated DeepSeek V4.1 Flash for a real #perf-agent request; 8-GPU config passes all targets | Done |
| 2 | Train a model | Full fine-tuning loop on a 5-class query classifier; tuned beats base 98% vs 96% and eliminates out-of-label errors | Done (firectl pass; SDK pass scripted, not yet run) |
| 3 | Enable a new model | GLM Edge 1.5B uploaded, serving, and called; 23 paper cuts documented in an open PR | Done |

---

## Task 1 — Optimize a deployment for a real use case

**What I did.** Studied the `perfagent` skill (workflow, guardrails, tooling),
then took a live sizing request from `#perf-agent` — DeepSeek V4.1 Flash for
agentic coding: 55k prompt / 90% cached / 1k output, 3 QPS; targets decode
>= 400 tok/s, TTFT p50 <= 600 ms, e2e <= 3.1 s (reconciled from an impossible
1.5 s with the requester). Ran the estimator, stood up 4 deployment
experiments plus an on-demand sweep in `pyroworks`, and compared measured
results against predictions and customer targets.

**Outcome.** Recommendation: Config A — 2xB300 FP4, inline tp2, 4 draft
tokens, image 6.176.1, 4 replicas = 8 GPUs. Measured: 408 tok/s decode,
TTFT 204 ms mean / 292 ms p95, e2e 2.66 s — all targets pass. Key levers
found: prompt-cache hit rate dominates cost (90% -> 50% cache = 8 -> 14
GPUs); TP width trades latency, not cost; disaggregated prefill lost on this
workload; draft depth / precision / image version were not levers. 8 GPUs is
the floor.

**What I learned.** The full sizing workflow (requirements -> resolve model ->
measured-data gate -> acceptance probabilities -> estimate -> present configs
-> experiment validation), plus the operational details that only show up in
validation: `--max-context-length` is mandatory or KV cache mis-sizes;
concurrency (not batch size) must be passed to the load generator; analyze
the steady window, not the ramp; experiments force acceptance rates so they
cannot catch acceptance gaps; p95 targets are sized as target/1.35 because
predictions are means.

**Deliverables** (in `Task1-optimize_inference/`):
`artifacts/dsv41-flash-perf-20261022/WRITEUP-dsv41-flash-optimization.md`
(the shareable write-up), `ALL-EXPERIMENTS-SUMMARY.txt`, per-experiment
validation metrics, sweep comparison + estimator CSVs, and `CHAT_NOTES.md`
(perfagent skill study).

---

## Task 2 — Train a model with the training API

**What I did.** Picked a concrete use case: a query-domain classifier
(user query -> strict JSON `{"domain": "math|code|history|finance|other"}`).
Built a tiny dataset by hand (200 train rows, 40/class; 50 held-out eval
rows; 40 DPO pairs), uploaded via `firectl`, ran a managed LoRA SFT job
(rank 8, 1 epoch) on `llama-v3p1-8b-instruct` — $0.024, 6 minutes. Deployed
the tuned model, ran a base-vs-tuned eval on the held-out set through two
same-shape deployments, then tore everything down. Also designed a
customer-support **tone** DPO task (40 preference pairs) as the case where
preference optimization is genuinely the right tool — and consciously
*dropped* DPO for the classifier itself after reasoning it through (gold
labels make SFT the correct tool; DPO adds nothing there). Wrote
`sdk_pass.py` to repeat the loop through the Python SDK (not yet run).

**Outcome.** Base 96.0% -> tuned 98.0% exact-match, 100% valid JSON on both.
The real win: the base model invented a label outside the class set
(`"health"`); the tuned model made **zero out-of-label answers** across all
50 rows. On a task the base already handles well, the fine-tune's value is
label-space discipline — the guarantee a production router needs.

**What I learned.** How to choose SFT vs DPO vs GRPO by task shape (gold
labels -> SFT; preferences -> DPO; verifiable reward -> GRPO). The managed
training flow: datasets are inert resources; jobs pin the output-model name
at creation (cancel does not free it); loss trains assistant turns only.
The deployment side was the deepest lesson: deployment shapes are
pre-validated configs whose capabilities (`MULTI_LORA`) and serving-image
floors are invisible until create time; an undeployed, non-serverless model
404s by name — you route by deployment name; and a **shapeless live-merge
deploy was the only thing that worked** when every validated shape was
pinned to a stale image.

**Deliverables** (in `Task2-model_finetuning/`): `findings.md` (results +
10-item friction log), `COMMANDS.md` (working command history),
`make_dataset.py`, `eval_compare.py`, `sdk_pass.py`, datasets in `data/`
(including the tone-DPO set for a future run).

**Friction highlights** (full log in `findings.md`): output-model name
pinning after job cancel; docs-vs-CLI drift (`--loss-method`,
`load-lora` vs `model load-lora`); BF16 necessary-but-not-sufficient for
addons; image-version floors invisible at planning time — and images before
6.6.0 *silently corrupt* Q/K LoRA deltas instead of erroring; shapeless
deploys labeled "advanced only" yet were the only working path.

---

## Task 3 — Enable a new model and hunt paper cuts

**What I did.** Picked `zai-org/glm-edge-1.5b-chat` (not previously on
Fireworks). Checked vLLM image compatibility, attempted the
`enable_vllm_model.yml` workflow (died in preflight — missing Actions
secret, unrelated to the model), uploaded weights via `model_download.yml`,
deployed shapeless on 1x H200 with `--engine VLLM`, made real chat calls
against it, documented everything, and tore the deployment down.

**Outcome.** Model `accounts/pyroworks/models/glm-edge-1p5b-chat` is READY
and public; deployment `kor9fm3p` served coherent chat responses (quality
weak — it's a 1.5B model); deployment deleted after testing. The real
deliverable — **23 paper cuts** across model choice, vLLM image checking,
upload/validation, deploy/call/catalog — is in
[fw-ai/fireworks#66204](https://github.com/fw-ai/fireworks/pull/66204).

**What I learned.** "Model" is three different products on this platform
(text/VLM checkpoints, a closed image-gen registry, native FireAttention
enablement) and the onboarding paths don't branch on that. The upload and
enable workflows are separate things that fail separately. A freshly
uploaded model has no validated shape, so the documented
`--deployment-shape default` deploy line cannot succeed — shapeless with
`--accept-shapeless-risk` plus `--engine VLLM` is the working path, none of
which the public page mentions. Routing uses `<model>#<deployment>` or the
deployment name.

**Deliverables** (in `Task3-deply_model/`):
`model-enablement-onboarding-findings.md` (the paper-cut list, also in PR
#66204), `STEPS.md` (reproducible bring-up steps), `test_model.py`.

---

## Cross-task themes

1. **Shapeless deploys rescued Tasks 2 and 3.** In both, every validated
   shape failed (stale serving image in Task 2; no shape exists for a new
   model in Task 3) and the "advanced users only" `--accept-shapeless-risk`
   path was the only thing that worked. The shape system's pre-validation
   value inverted into a stale-pinning liability in both cases.
2. **Docs-vs-CLI drift is systemic.** `--loss-method`, `load-lora` naming,
   `--deployment-shape default` for new models, missing `--engine` and
   `--accept-shapeless-risk` on the upload page — each task found examples.
3. **Routing semantics are undocumented at the point of failure.** Undeployed
   non-serverless models 404 by name with an error that doesn't hint the fix
   is a deployment name or `model#deployment`.
4. **Silent failures are the worst class.** Pre-6.6.0 images silently
   corrupting Q/K LoRA deltas (Task 2) and shapes silently dropped to the
   default image (per the deploy docs) are correctness hazards, not just
   blockers.

## Open follow-ups

- Task 2: run `sdk_pass.py` (SDK pass) and optionally the tone DPO on
  muse-glimmer-30b (~$0.05) — both scripted and documented.
- Task 3: PR #66204 review/merge.
- GRPO on the classifier (verifiable exact-match reward) via the serverless
  Training API — documented as the natural next exercise.