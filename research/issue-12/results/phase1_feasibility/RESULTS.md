# Phase 1 — MLX feasibility on Apple Silicon

Hardware: this Mac (M-series, MLX 0.31.2, mlx-audio 0.4.3, mlx-lm 0.31.3,
Python 3.12.13 venv on `/Volumes/S1/yooz/research/issue-12/.venv`).

Audio: 16 kHz mono clips sliced from
`/Volumes/S1/yooz/stt-test-data/english/test_001.wav`. The 2 s, 5 s,
and 15 s clips are written to `clips/clip_{2s,5s,15s}.wav` for
reproducibility.

Method: each model was loaded once. For each clip, one cold call
(measured separately) was followed by `--warm-iters 5` warm calls;
median + min/max of the warm calls is reported. Memory delta is
RSS at peak during the run minus RSS just before the clip starts.
Token counts come from `mlx_audio` `generate()` output where
available; otherwise word count is used as a proxy and labelled with
`tok/s` to indicate decoded tokens-per-second of generated text.

## Latency table

| model | clip | audio_s | cold_s | warm_med_s | rtfx | tok/s |
|---|---|---:|---:|---:|---:|---:|
| **parakeet_0_6b** (mlx-community/parakeet-tdt-0.6b-v3) | 2s | 2 | 0.22 | 0.051 | 38.9x | 19.4 |
| parakeet_0_6b | 5s | 5 | 0.07 | 0.067 | 74.5x | 104.3 |
| parakeet_0_6b | 15s | 15 | 0.15 | 0.130 | 115.7x | 115.7 |
| **qwen3_asr_0_6b_aufklarer** (aufklarer/Qwen3-ASR-0.6B-MLX-4bit) | 2s | 2 | 0.16 | 0.096 | 20.8x | 51.9 |
| qwen3_asr_0_6b_aufklarer | 5s | 5 | 0.16 | 0.147 | 33.9x | 81.4 |
| qwen3_asr_0_6b_aufklarer | 15s | 15 | 0.29 | 0.284 | 52.8x | 84.5 |
| **qwen3_asr_0_6b_mlx_4bit** (mlx-community/Qwen3-ASR-0.6B-4bit) | 2s | 2 | 0.24 | 0.092 | 21.6x | 54.1 |
| qwen3_asr_0_6b_mlx_4bit | 5s | 5 | 0.15 | 0.146 | 34.3x | 82.3 |
| qwen3_asr_0_6b_mlx_4bit | 15s | 15 | 0.28 | 0.279 | 53.7x | 85.9 |
| **qwen3_asr_1_7b_mlx_8bit** (mlx-community/Qwen3-ASR-1.7B-8bit) | 2s | 2 | 0.36 | 0.179 | 11.2x | 27.9 |
| qwen3_asr_1_7b_mlx_8bit | 5s | 5 | 0.30 | 0.309 | 16.2x | 38.9 |
| qwen3_asr_1_7b_mlx_8bit | 15s | 15 | 0.74 | 0.618 | 24.3x | 42.1 |

## Headline comparison vs Parakeet TDT 0.6B

For 5 s utterances (the most representative of dictation use):

| Model | warm latency | RTFX | latency vs Parakeet |
| --- | ---: | ---: | ---: |
| Parakeet TDT 0.6B | 0.067 s | 74.5x | 1.0x (baseline) |
| Qwen3-ASR-0.6B 4-bit (mlx-community) | 0.146 s | 34.3x | **2.2x slower** |
| Qwen3-ASR-0.6B 4-bit (aufklarer) | 0.147 s | 33.9x | **2.2x slower** |
| Qwen3-ASR-1.7B 8-bit (mlx-community) | 0.309 s | 16.2x | **4.6x slower** |

For 15 s utterances:

| Model | warm latency | RTFX | latency vs Parakeet |
| --- | ---: | ---: | ---: |
| Parakeet TDT 0.6B | 0.130 s | 115.7x | 1.0x |
| Qwen3-ASR-0.6B 4-bit | 0.279 s | 53.7x | **2.1x slower** |
| Qwen3-ASR-1.7B 8-bit | 0.618 s | 24.3x | **4.8x slower** |

## Cold start

| Model | cold first-clip | model load | total cold path |
| --- | ---: | ---: | ---: |
| Parakeet TDT 0.6B | 0.22 s | 0.90 s | ~1.12 s |
| Qwen3-ASR-0.6B 4-bit (mlx-community) | 0.24 s | 0.54 s | ~0.78 s |
| Qwen3-ASR-0.6B 4-bit (aufklarer) | 0.16 s | 0.83 s | ~0.99 s |
| Qwen3-ASR-1.7B 8-bit | 0.36 s | 1.20 s | ~1.56 s |

Cold-start cost is small in all cases. The Qwen3-ASR variants actually
load faster than Parakeet (the .safetensors is smaller; Parakeet
ships as a 2.3 GB checkpoint on disk including the encoder state).

## Memory

Reported `peak_rss_mb` is delta around inference, not absolute. For
absolute working set:

| Model | RSS after load | Notes |
| --- | ---: | --- |
| Parakeet TDT 0.6B | ~2.4 GB | full FP precision encoder + TDT decoder |
| Qwen3-ASR-0.6B 4-bit | ~0.7 GB | 4-bit quantized weights |
| Qwen3-ASR-1.7B 8-bit | ~2.3 GB | 8-bit quantized weights |

The Qwen3-ASR-1.7B-8bit is comparable in working memory to Parakeet
0.6B even though it's 3x the parameter count.

## Aufklarer packaging gotcha

`aufklarer/Qwen3-ASR-0.6B-MLX-4bit` (the most-downloaded community
variant) ships **without** `preprocessor_config.json`,
`chat_template.json`, or `generation_config.json`. mlx-audio raises
`OSError: Can't load feature extractor` on `load()`. Fix is trivial:
copy those three files from `mlx-community/Qwen3-ASR-0.6B-4bit/`
into the snapshot directory; both models share the same audio
encoder + tokenizer config. Once patched, latency and (in the
single-utterance smoke test) output text are identical to the
mlx-community variant. The popularity of the aufklarer variant is
therefore misleading — there's no quality-of-conversion advantage,
and you take on a manual fixup. **Recommend: pin the
mlx-community releases.** I've left the patched aufklarer snapshot
in the cache so we still have direct A/B numbers.

## Tokens / sec

Qwen3-ASR autoregressively decodes a token stream that includes a
language tag prefix and the transcript. Decode rates measured
(median across the 5 warm runs):

| Model | 2 s | 5 s | 15 s |
| --- | ---: | ---: | ---: |
| Qwen3-ASR-0.6B 4-bit | 54 tok/s | 82 tok/s | 86 tok/s |
| Qwen3-ASR-1.7B 8-bit | 28 tok/s | 39 tok/s | 42 tok/s |

The 1.7B model decodes at roughly half the throughput of the 0.6B
variant on the same audio, which is consistent with the latency
ratio (0.31 / 0.15 = 2.0x).

## Verdict for Phase 1

1. **Both Qwen3-ASR-0.6B and 1.7B run locally on Apple Silicon via
   MLX.** No build problems, no out-of-memory issues, no architecture
   mismatch with the current `mlx`/`mlx-lm`/`mlx-audio` stack.
2. **Qwen3-ASR is materially slower than Parakeet TDT 0.6B per
   utterance:** ~2.2x for the 0.6B, ~4.6x for the 1.7B (5 s clips).
   Both are still well under real-time (RTFX between 16x and 53x), so
   either is viable for batch and short-form dictation. Real-time
   streaming on the 1.7B at 16x RTFX is fine for typical 1–2 s VAD
   chunks; below ~0.5 s chunks the per-call overhead would dominate.
3. **mlx-swift is NOT a viable path today.** See `MLX_SWIFT_COMPAT.md`.
   The Swift LLM stack has no audio-encoder library; a port would
   require re-implementing Qwen3-Omni's chunked Conv2d audio encoder
   and the audio-token interleaving used by the Qwen3 text decoder.

## Next

Phase 2 quality benchmark to find out whether the latency cost buys
us materially better multilingual coverage. The single-utterance
smoke test on `test_001.wav` (English, "...predicted with
one-hundred percent certainty") suggests Qwen3 may apply more
aggressive normalization (it wrote "100%" while Parakeet wrote
"100 percent" and the reference has "one-hundred percent"); this is
a normalization difference rather than a recognition error and is
absorbed by the WER text-normalization in Phase 2.
