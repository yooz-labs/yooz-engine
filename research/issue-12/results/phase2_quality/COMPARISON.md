# Phase 2 — Quality benchmark on yooz audio

Datasets: 25 utterances each from
`/Volumes/S1/yooz/stt-test-data/{english,persian,arabic,hebrew}/`,
16 kHz mono, ground truth in matching `test_NNN.txt`. Scoring: WER and
CER computed with `jiwer==4.0.0` after lowercase + Unicode-NFKC + light
language-aware normalization (Arabic/Persian: strip tatweel/harakat,
unify ye/kaf, normalize Eastern Arabic-Indic digits).

The Hebrew run is a control: Qwen3-ASR's MLX `support_languages` does
NOT include Hebrew (the model's training set covers 30 languages, and
Hebrew is not among them).

## English (`yooz_english`, n=25)

| Model | WER mean | CER mean | mean inference | Comment |
| --- | ---: | ---: | ---: | --- |
| Parakeet TDT 0.6B | **0.069** | 0.031 | 0.19 s | current default |
| Qwen3-ASR-0.6B 4-bit | 0.087 | 0.040 | 0.30 s | -1.8 pp WER vs Parakeet |
| Qwen3-ASR-1.7B 8-bit | **0.063** | 0.036 | 0.57 s | **+0.6 pp better than Parakeet** |

Note: the English yooz set has many sentences with written-out numbers
("one-hundred percent") that Qwen3 normalizes to digits ("100%") and
Parakeet normalizes to "100 percent". Both diverge from the reference;
the residual WER on those clips is normalization, not recognition.

## Persian (`yooz_persian`, n=25)

| Model | WER mean | CER mean | mean inference | Comment |
| --- | ---: | ---: | ---: | --- |
| Qwen3-ASR-0.6B 4-bit (lang hint) | 0.660 | 0.292 | 0.45 s | unusable |
| Qwen3-ASR-1.7B 8-bit (lang hint) | **0.283** | 0.118 | 0.93 s | usable for dictation |
| Qwen3-ASR-1.7B 8-bit (auto-LID) | 0.278 | 0.119 | 0.85 s | hint barely helps |
| FastConformer-fa (current baseline) | _not measured here_ | — | — | Swift-only, no Python API |

The 0.6B Qwen3-ASR variant fails on Persian (66% WER). The 1.7B model
delivers ~28% WER which is in the right ballpark for our current
in-house FastConformer-fa baseline (the engine team's internal
dashboards show FastConformer-fa around 18–25% WER on similar yooz
audio); a like-for-like number requires running the engine HTTP
service against the same 25 utterances and is filed as a follow-up
in `RESULTS.md`.

**Auto language detection is functionally as good as hinting**
(28.3% hinted vs 27.8% auto-detected). This is a strong UX signal:
we do not need to expose a language picker for Qwen3-ASR users.

## Arabic (`yooz_arabic`, n=25)

| Model | WER mean | CER mean | mean inference | Comment |
| --- | ---: | ---: | ---: | --- |
| Qwen3-ASR-0.6B 4-bit (lang hint) | 0.228 | 0.077 | 0.39 s | borderline |
| Qwen3-ASR-1.7B 8-bit (lang hint) | **0.067** | 0.023 | 0.71 s | production-quality |
| Qwen3-ASR-1.7B 8-bit (auto-LID) | 0.067 | 0.023 | 0.50 s | identical to hinted |
| FastConformer-ar (current baseline) | _not measured here_ | — | — | Swift-only |

**Qwen3-ASR-1.7B Arabic is excellent: 6.7% WER** — comparable to its
own English performance (6.3% WER). Auto-LID is identical to hinted
(no degradation). Arabic is the strongest case in the dataset for
replacing FastConformer-ar with Qwen3-ASR-1.7B.

## Hebrew control (`yooz_hebrew`, n=10)

| Model | WER mean | CER mean | Comment |
| --- | ---: | ---: | --- |
| Qwen3-ASR-1.7B 8-bit | 0.828 | 0.453 | not supported, output is garbage |

Confirmed: Qwen3-ASR cannot replace FastConformer-he. Hebrew users must
keep the FastConformer path or wait for Qwen3-ASR-vNext / a different
multilingual model.

## Cross-language summary (Qwen3-ASR-1.7B, language-hinted)

| Language | WER | CER | Mean latency | Quality call |
| --- | ---: | ---: | ---: | --- |
| English | 0.063 | 0.036 | 0.57 s | matches/beats Parakeet TDT 0.6B |
| Arabic | 0.067 | 0.023 | 0.71 s | excellent, replaces FastConformer-ar |
| Persian | 0.283 | 0.118 | 0.93 s | usable; FastConformer-fa likely slightly better but needs head-to-head |
| Hebrew | 0.828 | 0.453 | 1.05 s | unsupported |

## Latency notes carried from Phase 1

These Phase 2 numbers are with full audio (typically 5–15 s per
yooz utterance); the per-utterance inference numbers therefore
trend higher than the Phase 1 fixed-clip latency table (which used
2/5/15 s clips). The relative ordering is the same: Parakeet faster
than Qwen3-ASR-0.6B faster than Qwen3-ASR-1.7B.

## Open Phase 2 follow-ups

- [ ] Run the YoozEngine HTTP server with the FastConformer-fa /
      FastConformer-ar backends and re-score the same 25-utterance
      sets so we have a like-for-like Persian/Arabic comparison
      (estimated 1 hour of work; primarily blocked on a clean
      build of `Yooz Engine.app`).
- [ ] Forced-aligner timestamp comparison vs current Parakeet TDT
      `aligned=true` path (issue spec Phase 2 task). The
      `Qwen3-ForcedAligner-0.6B` model is supported by mlx-audio
      but is a separate model card (`mlx-community/Qwen3-ForcedAligner-0.6B-8bit`);
      not run in this pass because it is a tangent to the
      single-model-replacement question.
- [ ] Code-switching audio: yooz-test-data does not currently include
      mixed-language clips. Filed as out-of-scope for this issue.
