# Domain classifier fine-tune — commands for you to run

AI-agent guardrail on `pyroworks` blocks mutating firectl calls from this chat,
so these run in YOUR terminal. Cwd: `Task2-model_finetuning/data`.

Plan shape: SFT via firectl → eval → SFT via Python SDK → eval → then decide
whether to do a DPO learning exercise (on real failure outputs) or design a new
task where preference data actually fits.

## Step 1 — upload datasets (train + eval)

```bash
cd /Users/aadesh/Desktop/Onboarding_tasks/Task2-model_finetuning/data
firectl dataset create -a pyroworks onboarding-domain-train train.jsonl
firectl dataset create -a pyroworks onboarding-domain-eval  eval.jsonl
```

(dpo_train.jsonl upload deferred until we decide on the DPO stage)

## Step 2 — launch the SFT job (firectl pass)

CANCEL the pending Muse Glimmer job first, then relaunch on llama-v3p1-8b-instruct:

```bash
# kill the queued Muse Glimmer job (stuck waiting on the busy training slot)
firectl sftj cancel -a pyroworks phg27joz
firectl model delete -a pyroworks domain-classifier-sft

# relaunch on the 8B model (LoRA SFT $0.50/1M training tokens, up-to-16B tier)
firectl sftj create -a pyroworks \
  --base-model accounts/fireworks/models/llama-v3p1-8b-instruct \
  --dataset onboarding-domain-train \
  --evaluation-dataset onboarding-domain-eval \
  --output-model domain-classifier-sft
```

Monitor with (note the job id it prints): 

Example - Name: accounts/pyroworks/supervisedFineTuningJobs/```<JOB_ID>```

```bash
firectl sftj get -a pyroworks <JOB_ID>
```

## Step 3 — deploy + eval (after job completes)

### WORKING DEPLOY COMMAND (shapeless, live-merge — succeeded after shapes failed)

All shape-based attempts failed (details in findings.md):
- `--deployment-shape default` → resolved image 4.763.1 lacks `peft.qk-lora-layout.hf-checkpoint.v1`
- `--deployment-shape wtkld7yr --enable-addons` → shape has no MULTI_LORA capability
- `--deployment-shape accounts/fireworks/deploymentShapes/rft-llama-v3p1-8b-instruct` (via firectl-admin) → "the latest validated deployment shape version is private"
- `--accelerator-type/count --precision` WITHOUT `--accept-shapeless-risk` → now rejected; shapeless needs explicit opt-in

What WORKED — deploy the fine-tuned model directly (live merge) with raw
hardware specs, explicitly opting out of shapes:

```bash
firectl deployment create -a pyroworks accounts/pyroworks/models/domain-classifier-sft \
  --deployment-id domain-classifier-eval \
  --accelerator-type NVIDIA_H100_80GB \
  --accelerator-count 1 \
  --precision BF16 \
  --accept-shapeless-risk
```

Check readiness (wait for `State` gone / `Status: OK` + `Ready Replica Count: 1`):

```bash
firectl deployment get -a pyroworks domain-classifier-eval
```

### Eval (deployment is LIVE — run now, then tear down)

Routing: pass the DEPLOYMENT name as the model — the gateway routes to the
deployment. Model-name routing 404s for undeployed models (llama-8b base has
no serverless route anymore: SupportsServerless off, deprecated 2025-11-26).

First, create a base deployment for a fair comparison (a second GPU, up only
for the eval window):

```bash
firectl deployment create -a pyroworks accounts/fireworks/models/llama-v3p1-8b-instruct \
  --deployment-id llama-v3p1-8b-base-eval \
  --accelerator-type NVIDIA_H100_80GB \
  --accelerator-count 1 \
  --precision BF16 \
  --accept-shapeless-risk
```

Wait for READY (`firectl deployment get -a pyroworks llama-v3p1-8b-base-eval`
-> Ready Replica Count: 1), then run the eval with both deployment names:

```bash
pip3 install openai
python3 eval_compare.py \
  --base-model accounts/pyroworks/deployments/llama-v3p1-8b-base-eval \
  --tuned-model accounts/pyroworks/deployments/domain-classifier-eval \
  --out eval_results.json
```

(Ask me to run the eval — inference calls are not agent-blocked.)

### Teardown (do this right after eval — both deployments bill hourly)

If a plain delete warns about checks (e.g. recently-created deployments), use
`--ignore-checks`:

```bash
firectl deployment delete -a pyroworks domain-classifier-eval --ignore-checks
firectl deployment delete -a pyroworks llama-v3p1-8b-base-eval --ignore-checks
```

## Step 4 — SDK pass (same loop via Python)

Handled via `sdk_pass.py` at the task root (I write it; you run it with your
FIREWORKS_API_KEY). Mirrors Steps 1–3 through the Fireworks Python SDK.
Gotcha: pass `lora_rank=8` explicitly — an omitted rank is sent as 0
(full-parameter training), not LoRA.

## Step 5 — DPO pass: tone task on the cheap 3B (runs while/after SFT)

Task: make `llama-v3p2-3b-instruct` reply like warm customer support instead of
curt/robotic. Dataset: `data/dpo_tone/tone_dpo_train.jsonl` (40 pairs,
input/preferred_output/non_preferred_output, one-turn — managed DPO format).

Upload (from `data/dpo_tone/`):

```bash
cd /Users/aadesh/Desktop/Onboarding_tasks/Task2-model_finetuning/data/dpo_tone
firectl dataset create -a pyroworks onboarding-tone-dpo tone_dpo_train.jsonl
```

Launch (cost: LoRA DPO $1/1M training tokens on the 3B — pennies):

```bash
firectl dpo-job create -a pyroworks \
  --base-model accounts/fireworks/models/llama-v3p2-3b-instruct \
  --dataset onboarding-tone-dpo \
  --output-model tone-classifier-dpo
```

Note: docs show a `--loss-method DPO` flag; local firectl --help doesn't list
it (docs-vs-CLI drift — friction note). If your firectl accepts it, add it.

Monitor:

```bash
firectl dpo-job get -a pyroworks <DPO_JOB_ID>
```

Deploy + subjective compare on `tone_eval_prompts.jsonl` (12 prompts) vs base:

```bash
firectl deployment create -a pyroworks accounts/pyroworks/models/tone-classifier-dpo \
  --deployment-shape default
```

Ask me when it's up — I'll run both models on the 12 prompts side by side and
we compare the replies. Tear down after.

## Step 6 — SFT eval analysis + findings.md
Handled by me: eval results tables, UX friction notes (fresh-eyes feedback),
plus a "later" section on serverless Training API / cookbook starter / GRPO.


Paste output / job ids back here as you go and I'll keep explaining + evaluating.
