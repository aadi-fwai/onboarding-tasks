# Task 3 — Steps taken (bring up a new model in `pyroworks`)

Date: Oct 22, 2026 (PT). Detailed paper cuts are in
`model-enablement-onboarding-findings.md` (verbatim copy of
`docs/model-enablement-onboarding-findings.md` from
[fw-ai/fireworks#66204](https://github.com/fw-ai/fireworks/pull/66204)).
Work was split between cloud agent run `bc-c4f50c52-345b-5f2b-ac67-1938f98d1ed0`
and local terminals.

## 1. Pick a model not already on Fireworks

- Considered `black-forest-labs/flux-3-image-klein` → rejected: image-gen is a closed
  registry (`py/fireworks/serving/image_generation/registry.py`), not the upload path.
- Tried `Spark-X2.6-4B` (`trust_remote_code`, hybrid sliding/full attention) via the vLLM
  enable path → enable workflow died in preflight (unrelated to the model).
- Settled on **`zai-org/glm-edge-1.5b-chat`** @ `7b201d3c160c25beda4cf0d107617ad975cd1ca8`
  (`GlmForCausalLM`, 1.59B BF16, ctx 8192). Use the safetensors repo, not `-gguf`.

## 2. Check vLLM image compatibility (`pick-vllm-image` skill)

```bash
aiconfigurator cli check-compat ...   # note: `cli` subcommand is required
```

Image chosen: upstream mirror `vllm/vllm-openai:v0.32.0` (`isVllmRelease=true`).

## 3. Attempt `enable_vllm_model.yml` — FAILED

Run [39843565770](https://github.com/fw-ai/fireworks/actions/runs/39843565770) died in
preflight (12:51 PM): `pip install` of private `fw-ai/aiconfigurator` needs
`VLLM_IMAGE_PICKER_PAT_AI_CONFIGURATOR`, which is empty. No model created. No skip input.

## 4. Upload weights with `model_download.yml` — WORKED

Run [39845135642](https://github.com/fw-ai/fireworks/actions/runs/39845135642) on `main`
(`stream`, pinned revision, account `pyroworks`, model id `glm-edge-1p5b-chat`),
finished 1:05 PM. Skill: `.cursor/skills/upload-hf-model/SKILL.md`.

Verify:

```bash
firectl model get -a pyroworks glm-edge-1p5b-chat
firectl model list -a pyroworks --search glm-edge-1p5b   # -a is required, else empty table
```

Result: `State: READY`, `Model Type: glm`, `Context Length: 8192`. Later set `Public: true`.

## 5. Create the deployment (vLLM, shapeless)

```bash
firectl-admin deployment create -a pyroworks \
  accounts/pyroworks/models/glm-edge-1p5b-chat \
  --engine VLLM \
  --accelerator-type NVIDIA_H200_141GB \
  --accelerator-count 1 \
  --precision BF16 \
  --min-replica-count 1 \
  --max-replica-count 1 \
  --accept-shapeless-risk \
  --extra-args='--trust-remote-code' \
  --extra-values isVllmRelease=true,vllmImageTag=v0.32.0 \
  --wait
```

→ `accounts/pyroworks/deployments/kor9fm3p` (created 13:14, spot, expires 2026-10-29 13:14).

Why these flags: the docs' `--deployment-shape default` fails for a new model (no validated
shape) → `--accept-shapeless-risk`; default engine is FireAttention, which can't serve this
→ `--engine VLLM`.

## 6. Call it

`test_model.py` posts to `https://api.fireworks.ai/inference/v1/chat/completions` with
`model: accounts/pyroworks/deployments/kor9fm3p` (the PR doc also documents the
`<model>#<deployment>` form). Setup:

```bash
pip install requests
echo 'export FIREWORKS_API_KEY=...' >> ~/.zshrc && source ~/.zshrc
cd Task3-deply_model && python3 test_model.py
```

Both calls returned coherent chat responses (quality is weak for factual questions — 1.5B model).

## 7. Document findings

Opened [fw-ai/fireworks#66204](https://github.com/fw-ai/fireworks/pull/66204)
("docs: record new-model onboarding paper cuts") adding
`docs/model-enablement-onboarding-findings.md` + a link in `docs/README.md`.
23 paper cuts across model choice, vLLM image check, upload/validation, deploy/call/catalog.

## 8. Teardown — DONE (Oct 22, 2026)

```bash
firectl deployment delete -a pyroworks kor9fm3p --wait   # stops replica, keeps the model
```

Deployment `kor9fm3p` deleted. Model `glm-edge-1p5b-chat` kept (READY, public).
