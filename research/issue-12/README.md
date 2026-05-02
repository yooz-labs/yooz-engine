# Issue 12 — Qwen3-ASR vs Parakeet TDT benchmark

Status: Phase 1 + Phase 2 + Phase 3 complete; Phase 4 (fine-tuning)
not started, gated on team-lead approval after reviewing Phase 2
quality numbers.

## Headline answers

- **Phase 1 (latency):** Qwen3-ASR-1.7B is ~4.6x slower than Parakeet
  TDT 0.6B per utterance, but still 16x faster than real-time on
  Apple Silicon. Both 0.6B and 1.7B Qwen3-ASR models load and run
  cleanly via `mlx-audio`. Numbers in `results/phase1_feasibility/RESULTS.md`.

- **Phase 2 (quality):** On the yooz 25-utterance sets, Qwen3-ASR-1.7B
  matches or beats Parakeet TDT on English (6.3% vs 6.9% WER) and
  delivers production-quality Arabic (6.7% WER, identical with or
  without language hint). Persian is usable (28%); Hebrew is not
  supported. Numbers in `results/phase2_quality/COMPARISON.md`.

- **Phase 3 (integration):** mlx-swift-lm has no audio module and
  cannot load Qwen3-ASR today; a native Swift port is 3-6 weeks of
  work. Recommended path is a Python sidecar (`mlx-audio`) running
  inside YoozEngine, similar to the existing Parakeet worker.
  Detail in `results/INTEGRATION_DESIGN.md` and
  `results/MLX_SWIFT_COMPAT.md`.

## Single-model verdict (the issue's headline question)

**Ship Qwen3-ASR-1.7B as an opt-in `qwen3_asr_preview` backend, not
as the default `auto` path.** Default stays Parakeet TDT for English
and other Parakeet-supported languages; Arabic flips its language
default to Qwen3-ASR-1.7B (the one language where it clearly wins
at comparable latency to FastConformer-ar); Persian stays on
FastConformer-fa pending a head-to-head; Hebrew stays on
FastConformer-he indefinitely (Qwen3-ASR does not support Hebrew).

The 4.6x latency multiplier on a 5 s clip is too risky to drop on
top of Whisper / Notes voice-keyboard UX as a default. RTFX stays at
16x or better, which is fine for dictation, but streaming-mode WER
is unmeasured. The preview earns a default flip only after telemetry
clears the graduation criteria in `results/INTEGRATION_DESIGN.md`
(P50 RTFX, WER parity, no regression complaints, streaming WER
measured). Telemetry is local-only, opt-in, performance signals only;
no transcript content.

## Layout

```
research/issue-12/
  README.md                                  -- this file
  MLX_SWIFT_COMPAT.md                        -- definitive no-go for native Swift today
  scripts/
    bootstrap_env.sh                         -- one-shot S1 venv + model download
    bench_phase1_latency.py                  -- 2/5/15 s clip latency for all 4 models
    bench_phase2_quality.py                  -- WER/CER on yooz multilingual sets
  results/
    MLX_SWIFT_COMPAT.md                      -- copy of the compat doc
    INTEGRATION_DESIGN.md                    -- Phase 3 single-model verdict + API
    phase1_feasibility/
      RESULTS.md                             -- Phase 1 narrative
      summary.csv                            -- machine-readable per-model x per-clip
      *.json                                 -- per-model raw samples
      clips/                                 -- the calibrated 2/5/15 s clips
    phase2_quality/
      COMPARISON.md                          -- Phase 2 narrative
      yooz_english/{summary.json, *.jsonl}
      yooz_persian/{summary.json, *.jsonl}
      yooz_arabic/{summary.json, *.jsonl}
      yooz_hebrew/{summary.json, *.jsonl}
    phase2_quality_autolid/                  -- same datasets, auto-LID (no language hint)
      yooz_persian/{summary.json, *.jsonl}
      yooz_arabic/{summary.json, *.jsonl}
```

Models, audio inputs, and the venv all live on `/Volumes/S1/yooz/research/issue-12/`.
Code in this directory is what gets committed; everything heavy stays on S1.

## Reproducing

```bash
# One-shot bootstrap (creates the S1 venv, downloads the four MLX models).
bash research/issue-12/scripts/bootstrap_env.sh

# Phase 1 latency
env HF_HOME=/Volumes/S1/yooz/research/issue-12/models/hf_cache \
    /Volumes/S1/yooz/research/issue-12/.venv/bin/python \
    research/issue-12/scripts/bench_phase1_latency.py --warm-iters 5

# Phase 2 quality, English
env HF_HOME=/Volumes/S1/yooz/research/issue-12/models/hf_cache \
    /Volumes/S1/yooz/research/issue-12/.venv/bin/python \
    research/issue-12/scripts/bench_phase2_quality.py \
        --dataset yooz_english \
        --models parakeet_0_6b qwen3_asr_0_6b_mlx_4bit qwen3_asr_1_7b_mlx_8bit

# Persian / Arabic / Hebrew (`--language-hint` to pin language; omit for auto-LID)
env HF_HOME=/Volumes/S1/yooz/research/issue-12/models/hf_cache \
    /Volumes/S1/yooz/research/issue-12/.venv/bin/python \
    research/issue-12/scripts/bench_phase2_quality.py \
        --dataset yooz_persian --models qwen3_asr_1_7b_mlx_8bit --language-hint
```

## Known packaging gotcha

`aufklarer/Qwen3-ASR-0.6B-MLX-4bit` (the high-download community
variant) ships without `preprocessor_config.json`,
`chat_template.json`, and `generation_config.json`. mlx-audio raises
on `load()`. Fix: copy those three files from
`mlx-community/Qwen3-ASR-0.6B-4bit` into the snapshot directory.
**Recommendation:** don't ship the aufklarer variant; pin the
mlx-community releases (which include all required configs and
have effectively identical latency / output text).
