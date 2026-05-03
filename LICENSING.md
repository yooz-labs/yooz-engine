# Yooz Licensing Strategy

> Decision date: 2026-05-03. Engine is the heart of the Yooz ecosystem; this
> document is the operational policy for every Yooz repo and every Yooz model
> release. The strategic version of this document also lives in the `yooz`
> coordination repo at `legal/LICENSING.md`.

## TL;DR

| What | License | Where |
|---|---|---|
| Product code (engine, whisper, notes, keyboard, vault, messenger, remi) | **PolyForm Shield 1.0.0** (source-available) | github.com/yooz-labs |
| Model weights (ASR, touchup LLMs, distilled students, LoRA adapters) | **Apache 2.0** (open source) | huggingface.co/YoozLabs |

We split the license so the **product** (orchestration, platform, UX, integration) is protected from competing managed services, while the **weights** stay open for the research community.

## Code: PolyForm Shield 1.0.0

PolyForm Shield is a source-available license that:

- **Allows**: read, fork, modify, and use for any purpose **except** building a competing product.
- **Forbids**: offering a "managed Yooz", a hosted clone, or a re-skinned commercial fork.
- **Does not auto-convert**: unlike BSL, there is no built-in expiry to a permissive license. Each repo can choose to relicense later, but it isn't automatic.

Spec: https://polyformproject.org/licenses/shield/1.0.0/

### Why not Apache / MIT?

We considered Apache and MIT. The risk: someone (or a hyperscaler) wraps Yooz Engine in a managed offering, monetizes it, and we lose the moat without contributing back. PolyForm Shield prevents that without harming honest use, education, research, or contribution.

### Why not AGPL?

AGPL is open source but viral — every consuming app would have to publish their changes under AGPL. That's incompatible with the Yooz ecosystem itself: we want consumer apps to be able to embed the engine without AGPL contagion to their UI code.

### Why not BSL or FSL?

Both auto-convert to permissive licenses (4 years for BSL, 2 years for FSL). For an early-stage project where the moat is still forming, an automatic clock is risky. PolyForm Shield holds the line indefinitely; we can choose to relicense voluntarily once the moat is durable.

## Model weights: Apache 2.0

Every model checkpoint we publish on HuggingFace is **Apache 2.0**. Concretely:

- **Base model weights** (Qwen3-ASR Swift port, Parakeet variants, etc.): Apache 2.0, lineage credit to upstream (Qwen team / NVIDIA / etc.).
- **Fine-tuned checkpoints** (Yooz-Quality LoRA, distilled students, etc.): Apache 2.0, training data documented, base model linked.
- **Adapters only** (LoRA `.safetensors`): Apache 2.0, must be loaded onto the documented base.

### Why open weights?

The competitive moat for privacy-first AI lives in the **product**, not the weights:

- Multi-device elastic orchestration (phone → PC → Vault) over WireGuard mesh.
- Universal AI platform layer across Apple / Android / Windows AI APIs.
- Private AI memory: encrypted, local, with permissioned cross-app context.
- Beautiful, consumer-grade UX.

Open weights buy us:

- **Audit credibility**. Privacy claims that can't be reproduced are marketing. Open weights let researchers verify them.
- **Recruiting + research signal**. Top ML engineers want to ship work that the community can actually use.
- **Ecosystem flywheel**. Other tools (mlx-audio, mlx-lm, llama.cpp, ONNX exporters) can integrate Yooz checkpoints without licensing friction.

If we ever ship a checkpoint we can't open (e.g., something built on a non-permissive base), we'll publish it with the most permissive license the lineage allows and document the reason.

## Distribution channels

**HuggingFace is primary.** All model weights live at https://huggingface.co/YoozLabs.

**GHCR is being archived.** The historical OCI artifacts at `ghcr.io/yooz-labs/yooz-models/*` (smollm2-135m, qwen2-0.5b-4bit, gemma-3-1b-4bit, parakeet-tdt-0.6b-v3, yooz-light v1/v2/v3, yooz-quality v1/v2/v3) will be migrated to HuggingFace and the GHCR packages will be either archived or deleted. Migration is tracked as its own issue under `yooz-labs/yooz-engine`.

**GitHub releases** stay as the artifact channel for built binaries (Yooz Engine.app, Whisper helper, etc.) but not for raw model weights.

## Per-artifact policy

| Artifact type | License | Channel |
|---|---|---|
| Engine source code | PolyForm Shield 1.0.0 | github.com/yooz-labs/yooz-engine |
| Engine compiled `.app` binary | PolyForm Shield 1.0.0 | GitHub Releases |
| Engine SDK (`YoozEngineClient` Swift Package) | PolyForm Shield 1.0.0 | GitHub Releases / SwiftPM |
| Model weights (any) | Apache 2.0 | HuggingFace |
| Training data we generated | CDLA-Permissive 2.0 (or Apache 2.0 where applicable) | HuggingFace Datasets |
| Eval suites (yooz-benchmark, gold_standard.jsonl) | Apache 2.0 | GitHub or HuggingFace Datasets |
| Documentation | CC-BY-4.0 | In-repo `.md` files |
| Internal financial / corporate docs | Proprietary, not distributed | private only |

## Migration plan for existing repos

Engine, whisper, notes, etc. are currently MIT (legacy). Conversion plan:

1. Open a tracking issue per repo: "Re-license to PolyForm Shield 1.0.0".
2. Run a contributor audit (git log) to confirm no third-party contributors who'd need to consent.
3. Land a commit replacing `LICENSE` with `LICENSE.md` (PolyForm Shield), updating `README.md` license section, and adding the SPDX identifier `LicenseRef-PolyForm-Shield-1.0.0` to package manifests.
4. Tag the last MIT-licensed commit so existing forks can pin it cleanly.
5. Update CONTRIBUTING.md to reference the new license + a CLA / DCO if needed.

## Contact

Licensing questions, dual-license requests, or commercial-use inquiries: **dev@yooz.info**.

---

*This document is the canonical statement of the Yooz licensing strategy. If
anything in another repo conflicts with this document, this document wins.*
