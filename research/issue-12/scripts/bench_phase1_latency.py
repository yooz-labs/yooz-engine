"""Phase 1 — Qwen3-ASR vs Parakeet TDT latency on Apple Silicon (MLX).

Measures, per backend and per audio clip:
    - cold-start (first call after model load)
    - warm latency (median of N=5 repeated calls)
    - peak resident memory delta around inference
    - tokens/sec (decoder side)
    - real-time factor (audio_duration / wall_time)

Models compared:
    parakeet            mlx-community/parakeet-tdt-0.6b-v3
    qwen3_asr_0_6b_4    mlx-community/Qwen3-ASR-0.6B-4bit
    qwen3_asr_0_6b_au   aufklarer/Qwen3-ASR-0.6B-MLX-4bit
    qwen3_asr_1_7b_8    mlx-community/Qwen3-ASR-1.7B-8bit

Audio: three calibrated clips at 2 s, 5 s, 15 s. The script slices them
from /Volumes/S1/yooz/stt-test-data/english/test_001.wav (which is ~12 s)
and a longer clip; if the longer clip is missing, it pads/loops the
short one. For real benchmarks supply --audio-dir to point at any
prepared 16 kHz mono wav directory.

Outputs JSON per (model, clip) and a CSV summary into
/Volumes/S1/yooz/research/issue-12/results/phase1_feasibility/.

Usage:
    source /Volumes/S1/yooz/research/issue-12/.venv/bin/activate
    python research/issue-12/scripts/bench_phase1_latency.py \\
        --warm-iters 5 --output-root /Volumes/S1/yooz/research/issue-12/results/phase1_feasibility
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

# mlx_audio handles both Parakeet and Qwen3-ASR via its STT registry.
from mlx_audio.stt import load as load_stt

DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/S1/yooz/research/issue-12/results/phase1_feasibility"
)
DEFAULT_AUDIO_SOURCE = Path("/Volumes/S1/yooz/stt-test-data/english/test_001.wav")
TARGET_SR = 16000
CLIPS = [("2s", 2.0), ("5s", 5.0), ("15s", 15.0)]

MODELS = {
    "parakeet_0_6b": "mlx-community/parakeet-tdt-0.6b-v3",
    "qwen3_asr_0_6b_mlx_4bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
    "qwen3_asr_0_6b_aufklarer": "aufklarer/Qwen3-ASR-0.6B-MLX-4bit",
    "qwen3_asr_1_7b_mlx_8bit": "mlx-community/Qwen3-ASR-1.7B-8bit",
}


@dataclass
class Sample:
    model: str
    repo_id: str
    clip: str
    audio_seconds: float
    cold_seconds: float
    warm_median_seconds: float
    warm_min_seconds: float
    warm_max_seconds: float
    peak_rss_mb: float
    rtfx: float                    # audio_seconds / warm_median_seconds
    decoder_tokens: int
    decoder_tok_per_sec: float
    text: str


def _prepare_clips(source_wav: Path, output_dir: Path) -> dict[str, Path]:
    """Slice the source wav into 2 s, 5 s, 15 s mono 16 kHz clips."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(source_wav), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        # Avoid a librosa import for this preprocessing-light step
        from scipy.signal import resample_poly

        audio = resample_poly(audio, TARGET_SR, sr).astype(np.float32)
        sr = TARGET_SR
    out: dict[str, Path] = {}
    for name, secs in CLIPS:
        n = int(secs * sr)
        if len(audio) >= n:
            clip = audio[:n]
        else:
            reps = int(np.ceil(n / len(audio)))
            clip = np.tile(audio, reps)[:n]
        path = output_dir / f"clip_{name}.wav"
        sf.write(str(path), clip, sr)
        out[name] = path
    return out


def _peak_rss_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024**2)


def _generate_once(model, audio_path: Path) -> tuple[str, int]:
    """Run a single transcription. Returns (text, decoder_token_count)."""
    out = model.generate(str(audio_path), verbose=False)
    text = getattr(out, "text", str(out)).strip()
    # Try to read decoder tokens from common attribute names.
    n_tok = (
        getattr(out, "generation_tokens", None)
        or getattr(out, "n_tokens", None)
        or len(text.split())  # fallback approximation
    )
    return text, int(n_tok)


def _bench_model(name: str, repo_id: str, clips: dict[str, Path], warm_iters: int):
    print(f"\n=== {name} ({repo_id}) ===")
    gc.collect()
    rss_before_load = _peak_rss_mb()
    t0 = time.perf_counter()
    model = load_stt(repo_id)
    load_seconds = time.perf_counter() - t0
    rss_after_load = _peak_rss_mb()
    print(f"  load: {load_seconds:.2f}s   RSS delta: {rss_after_load - rss_before_load:.0f} MB")

    samples: list[Sample] = []
    for clip_name, clip_path in clips.items():
        # Cold = first inference for this clip (on first model use overall it's
        # also the first kernel-compile cost; subsequent clips reuse it).
        gc.collect()
        rss_before = _peak_rss_mb()
        t0 = time.perf_counter()
        text, tok = _generate_once(model, clip_path)
        cold = time.perf_counter() - t0
        rss_peak = _peak_rss_mb()

        warms = []
        last_text, last_tok = text, tok
        for _ in range(warm_iters):
            t0 = time.perf_counter()
            last_text, last_tok = _generate_once(model, clip_path)
            warms.append(time.perf_counter() - t0)
            rss_peak = max(rss_peak, _peak_rss_mb())

        median = statistics.median(warms)
        sample = Sample(
            model=name,
            repo_id=repo_id,
            clip=clip_name,
            audio_seconds=dict(CLIPS)[clip_name],
            cold_seconds=cold,
            warm_median_seconds=median,
            warm_min_seconds=min(warms),
            warm_max_seconds=max(warms),
            peak_rss_mb=rss_peak - rss_before,
            rtfx=dict(CLIPS)[clip_name] / median if median > 0 else float("nan"),
            decoder_tokens=last_tok,
            decoder_tok_per_sec=last_tok / median if median > 0 else 0.0,
            text=last_text,
        )
        samples.append(sample)
        print(
            f"  {clip_name}: cold {cold:.2f}s  warm median {median:.2f}s "
            f"(n={warm_iters})  rtfx {sample.rtfx:.2f}x  "
            f"~{sample.decoder_tok_per_sec:.1f} tok/s"
        )

    del model
    gc.collect()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-iters", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--audio-source", type=Path, default=DEFAULT_AUDIO_SOURCE)
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODELS.keys()),
        help="Subset of model keys to run (default: all).",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    clip_dir = args.output_root / "clips"
    clips = _prepare_clips(args.audio_source, clip_dir)

    all_samples: list[Sample] = []
    for key in args.models:
        if key not in MODELS:
            raise SystemExit(f"unknown model key: {key}")
        repo = MODELS[key]
        try:
            samples = _bench_model(key, repo, clips, args.warm_iters)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {key}: {type(e).__name__}: {e}")
            (args.output_root / f"{key}.error.txt").write_text(f"{type(e).__name__}: {e}\n")
            continue
        all_samples.extend(samples)
        with (args.output_root / f"{key}.json").open("w") as f:
            json.dump([asdict(s) for s in samples], f, indent=2)

    csv_path = args.output_root / "summary.csv"
    fields = list(Sample.__annotations__.keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in all_samples:
            writer.writerow(asdict(s))
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
