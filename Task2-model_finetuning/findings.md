# Task 2 — Model Fine-Tuning: Findings

Domain-classifier fine-tune on Fireworks: user query -> strict JSON
`{"domain": "math|code|history|finance|other"}`. Full loop completed:
dataset in -> job running -> checkpoint out -> deployment -> base-vs-tuned eval.

## Results (50 held-out eval rows, exact-match, temperature 0)

| model | accuracy | valid JSON |
|---|---|---|
| base (`llama-v3p1-8b-instruct`) | 96.0% (48/50) | 100% |
| tuned (SFT LoRA rank 8, 1 epoch, 200 rows) | **98.0%** (49/50) | 100% |

The base's one real failure: it invented `{"domain": "health"}` — a label
*outside* the 5-class set. The tuned model made zero out-of-label answers
across all 50 rows. The single remaining miss (base and tuned both) is
`"How do I improve my credit card rewards strategy?"` -> `finance` vs gold
`other` — genuinely ambiguous; arguably a gold-label bug in our dataset.

**Takeaway:** on a task the base already handles well, the fine-tune's value
was *label-space discipline* (never emits an out-of-set label), not raw
accuracy. For a production classifier routing queries, that's exactly the
guarantee you want.

## Job details

- firectl pass: job `x9ildknj`, completed in ~6 min, LoRA rank 8,
  estimated cost $0.024 (up-to-16B tier, $0.50/1M LoRA SFT training tokens).
- (An earlier Muse Glimmer job `phg27joz` was cancelled — see friction #3.)

## Dataset

- `make_dataset.py` -> `data/train.jsonl` (200 rows, 40/class),
  `eval.jsonl` (50 rows, 10/class, held out), `dpo_train.jsonl` (40 pairs).
- Managed SFT trains loss on assistant turns only (system/user masked).
- Datasets uploaded as `onboarding-domain-train` / `onboarding-domain-eval`.

## Deployment story (the interesting part)

Every shape-based deploy failed; only a **shapeless live-merge deploy** worked:

1. `--deployment-shape default` -> rejected: PEFT addon requires serving
   capability `peft.qk-lora-layout.hf-checkpoint.v1`, but the resolved image
   was `4.763.1` (that capability ships in text-completion >= 6.6.0).
2. `--deployment-shape wtkld7yr --enable-addons` -> rejected: shape has no
   `MULTI_LORA` capability (its `Capabilities` list is empty).
3. Admin-CLI shape `rft-llama-v3p1-8b-instruct` -> rejected: "the latest
   validated deployment shape version is private".
4. **Worked:** deploy the fine-tuned model directly, shapeless, with explicit
   opt-in: `firectl deployment create <tuned-model> --accelerator-type
   NVIDIA_H100_80GB --accelerator-count 1 --precision BF16
   --accept-shapeless-risk`. Shapeless create resolves a *current* image, so
   the capability floor was met; the LoRA live-merges at deploy time.

The base model for a fair comparison needed its own deployment —
`llama-v3p1-8b-instruct` has **no serverless route** anymore
(SupportsServerless off, deprecation date 2025-11-26), so requesting the base
by model name 404s ("Model not found, inaccessible, and/or not deployed").
Routing lesson: pass the *deployment* name as the `model` field —
`accounts/pyroworks/deployments/<id>` — for both base and tuned.

Both deployments deleted right after eval (`--ignore-checks` needed).

## Friction log (fresh-eyes feedback)

1. **AI-agent guardrail by account**: mutating firectl commands are blocked
   for AI agents on `pyroworks` ("BLOCKED: mutating command ... cannot run
   inside an AI agent"). Reasonable guardrail, but it surfaced only at
   execution time, not from any doc. Also hit a transient
   `inferenceuser cannot perform Create operations` on the first
   `dataset create`, which succeeded on retry with no explanation.
2. **Output-model name is pinned at job creation**: cancelling a job does
   *not* free the output-model name. Relaunching with the same
   `--output-model` fails with "output model ... already exists"; you must
   `firectl model delete` the placeholder first (it exists in `UPLOADING`
   state, tied to the cancelled job). Nothing in the docs warns about this.
3. **Training-slot queue message was good**: "waiting for a serverless
   training slot: this account is at its limit of 1 concurrent training runs
   on <model>" — clear and actionable. The *limit itself* (per-account,
   per-base-model concurrency caps) isn't documented in the fine-tuning guide.
4. **Docs vs CLI drift**: docs show `--loss-method DPO` for
   `dpo-job create` (and ORPO via `--loss-method ORPO`), but local firectl
   `--help` doesn't list it. Docs use `firectl load-lora`; the CLI has
   `firectl model load-lora`. Which surface is authoritative is unclear.
5. **Shape capability model is confusing**: BF16 is *necessary* but not
   *sufficient* for `--enable-addons` — the `MULTI_LORA` capability is
   validated per shape version, and most BF16 shapes on this account lack
   it. Meanwhile the one shape with `MULTI_LORA` is FP4, which the docs'
   precision table says cannot support addons at all. The table (BF16 yes /
   FP8,FP4 no) oversimplifies; `deployment-shape-version get` +
   `Capabilities` is the ground truth.
6. **Image-version floors are invisible**: shape versions don't expose their
   serving image tag; it's resolved at create time. The failure only surfaces
   as an error at deploy. Worse, images before 6.6.0 silently corrupt Q/K
   LoRA deltas instead of erroring (fw_qk_lora_layout ignored) — a silent
   correctness hazard, not just a deploy blocker.
7. **Silent shape drop**: per the Deploying Trained Models doc, requesting a
   shape not validated for your *exact* model version is silently dropped to
   the default serving image (no error at create), then addon loads fail
   later. `deployment-shape-version match` doesn't show which model version
   each shape binds — you need `list` + `latest_validated=true`.
8. **Shapeless worked where shapes failed** — but the docs label shapeless as
   "advanced users only" and "most common cause of failed deployments". For
   us it was the *only* path that worked; the validated shapes were pinned to
   a stale image (4.763.1, pre-6.6.0) from July. The shape system's value
   prop (pre-validated) inverted into a stale-pinning liability here.
9. **Model-name 404 for undeployed non-serverless models** is technically
   correct but opaque — "Model not found, inaccessible, and/or not deployed"
   doesn't hint that the fix is deploying the base or using a deployment name
   as the model. A pointer in the error text would save a debugging round.
10. **Deprecated models still accept training jobs**: `llama-v3p1-8b-instruct`
    shows a deprecation date in the past (2025-11-26) yet remains
    `Tunable: true` and trainable/servable. Confusing for a newcomer deciding
    which base to pick — the model card should state the actual behavior.

## Later / not done

- **SDK pass**: `sdk_pass.py` written (same loop via the Python SDK;
    note the `lora_rank=0`-means-full-parameter gotcha). Not yet run.
- **Tone DPO**: task designed (customer-support tone, `data/dpo_tone/`,
    40 input/preferred_output/non_preferred_output pairs + 12 eval prompts)
    — the case where preference data is genuinely the right signal, unlike
    the gold-label classifier where SFT suffices. Not run: the only
    deployable base on this account is muse-glimmer-30b (~$0.05-0.10 DPO
    cost), and llama-3b/8b lack addon-capable or live-merge-deployable
    shapes here (see friction #5/#6).
- **GRPO**: the same classifier is a good GRPO candidate later (verifiable
    exact-match reward) via the serverless Training API
    (`base_url .../training/v1/serverless`) — a different surface entirely.
- **Serverless Training API SFT** (cookbook `muse_glimmer_starter`): the
    documented "free first fine-tune" path; skipped because Muse Glimmer
    training slots on this account were saturated.
