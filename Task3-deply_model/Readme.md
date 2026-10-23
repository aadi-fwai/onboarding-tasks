# Task 3 — Deploy a new model

## Status: DONE (deployment torn down)

Uploaded `zai-org/glm-edge-1.5b-chat` into `pyroworks`, served it on vLLM (1x H200, BF16),
called it, and recorded paper cuts in
[fw-ai/fireworks#66204](https://github.com/fw-ai/fireworks/pull/66204).

## Files

| file | what |
|---|---|
| `STEPS.md` | Step-by-step of what was done, with exact commands and run links |
| `model-enablement-onboarding-findings.md` | Verbatim copy of `docs/model-enablement-onboarding-findings.md` from PR #66204 (do not edit) |
| `test_model.py` | Chat-completion call against the deployment |

## Resource IDs (account `pyroworks`)

- Model: `accounts/pyroworks/models/glm-edge-1p5b-chat` (READY, public)
- Deployment: `accounts/pyroworks/deployments/kor9fm3p` (deleted Oct 22, 2026)
- Upload run: [model_download.yml 39845135642](https://github.com/fw-ai/fireworks/actions/runs/39845135642)
- Failed enable run: [enable_vllm_model.yml 39843565770](https://github.com/fw-ai/fireworks/actions/runs/39843565770)
- Cloud agent: `bc-c4f50c52-345b-5f2b-ac67-1938f98d1ed0`
