"""Phase 2 — Qwen3-ASR vs Parakeet TDT vs FastConformer on real audio.

Reads ground-truth from one of three sources (selectable via --dataset):

    yooz_english       : /Volumes/S1/yooz/stt-test-data/english/{test_NNN.wav,.txt}
    yooz_persian       : /Volumes/S1/yooz/stt-test-data/persian/{test_NNN.wav,.txt}
    yooz_arabic        : /Volumes/S1/yooz/stt-test-data/arabic/{test_NNN.wav,.txt}
    yooz_benchmark_en  : pairs from yooz-benchmark gold_standard.jsonl
    librispeech_clean  : segments under /Volumes/S1/yooz/external/common-voice/segments
                         (LibriSpeech-derived; refs in ../transcriptions/*.json)

Runs the requested model(s) on every utterance, records hypothesis text,
computes WER + CER with `jiwer` (with per-language text normalization),
and writes both per-utterance JSON and a per-(dataset, model) summary.

FastConformer-ar / FastConformer-fa keys exist in the MODELS table but
were NOT run in this PR's benchmark pass: the production FastConformer
checkpoints we ship are Swift-only (NeMo .nemo packaging consumed by
mlx-audio's Swift port, not the Python API). A like-for-like Persian /
Arabic comparison vs FastConformer requires running YoozEngine's HTTP
service against the same 25-utterance sets and is filed as a follow-up
in `results/phase2_quality/COMPARISON.md`.

Usage:
    source /Volumes/S1/yooz/research/issue-12/.venv/bin/activate
    python research/issue-12/scripts/bench_phase2_quality.py \\
        --dataset yooz_english --models parakeet_0_6b qwen3_asr_0_6b_mlx_4bit qwen3_asr_1_7b_mlx_8bit
    python research/issue-12/scripts/bench_phase2_quality.py \\
        --dataset yooz_persian --models qwen3_asr_1_7b_mlx_8bit fastconformer_fa
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import jiwer

from mlx_audio.stt import load as load_stt

DEFAULT_OUTPUT_ROOT = Path("/Volumes/S1/yooz/research/issue-12/results/phase2_quality")

DATASETS = {
    "yooz_english": ("/Volumes/S1/yooz/stt-test-data/english", "English"),
    "yooz_persian": ("/Volumes/S1/yooz/stt-test-data/persian", "Persian"),
    "yooz_arabic": ("/Volumes/S1/yooz/stt-test-data/arabic", "Arabic"),
    "yooz_hebrew": ("/Volumes/S1/yooz/stt-test-data/hebrew", "Hebrew"),
    "yooz_japanese": ("/Volumes/S1/yooz/stt-test-data/japanese", "Japanese"),
    "yooz_korean": ("/Volumes/S1/yooz/stt-test-data/korean", "Korean"),
    "yooz_chinese": ("/Volumes/S1/yooz/stt-test-data/chinese", "Chinese"),
}

MODELS = {
    "parakeet_0_6b": "mlx-community/parakeet-tdt-0.6b-v3",
    "qwen3_asr_0_6b_mlx_4bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
    "qwen3_asr_0_6b_aufklarer": "aufklarer/Qwen3-ASR-0.6B-MLX-4bit",
    "qwen3_asr_1_7b_mlx_8bit": "mlx-community/Qwen3-ASR-1.7B-8bit",
    "fastconformer_ar": "/Volumes/S1/yooz/stt-models/fastconformer-ar",
    "fastconformer_fa": "/Volumes/S1/yooz/stt-models/fastconformer-fa",
}


@dataclass
class Utterance:
    audio_path: Path
    reference: str


@dataclass
class Hypothesis:
    audio: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    inference_seconds: float


# ---------- text normalization ----------
_PUNCT_RE = re.compile(r"[\.\,\!\?\;\:\-\_\"\'\`\(\)\[\]\{\}\|\/\\<>“”‘’،؛؟٪٫٬۔]")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str, language: str) -> str:
    """Lightweight, language-aware normalization for fair WER comparison."""
    s = unicodedata.normalize("NFKC", text or "").strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    if language in {"Persian", "Arabic", "Hebrew"}:
        # Map Arabic-Indic + Eastern Arabic-Indic digits to ASCII.
        s = s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩"
                                       "۰۱۲۳۴۵۶۷۸۹",
                                       "01234567890123456789"))
        # Persian-specific: unify ye / kaf
        s = s.replace("ي", "ی").replace("ك", "ک")
        # Strip tatweel and harakat
        s = re.sub(r"[ً-ْـ]", "", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _wer_cer(reference: str, hypothesis: str, language: str) -> tuple[float, float]:
    ref = _normalize(reference, language)
    hyp = _normalize(hypothesis, language)
    if not ref:
        return float("nan"), float("nan")
    wer = jiwer.wer(ref, hyp)
    cer = jiwer.cer(ref, hyp)
    return wer, cer


# ---------- dataset loaders ----------
def _load_yooz_dataset(root: Path) -> list[Utterance]:
    out: list[Utterance] = []
    for wav in sorted(root.glob("test_*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        out.append(Utterance(wav, txt.read_text(encoding="utf-8").strip()))
    return out


# ---------- model wrapper ----------
def _generate_text(model, audio_path: Path, language: str | None) -> tuple[str, float]:
    t0 = time.perf_counter()
    kwargs: dict = {}
    if language is not None:
        kwargs["language"] = language
    out = model.generate(str(audio_path), verbose=False, **kwargs)
    elapsed = time.perf_counter() - t0
    return getattr(out, "text", str(out)).strip(), elapsed


# ---------- main ----------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--language-hint",
        action="store_true",
        help="Pass language to the Qwen3-ASR generate() call. Default: omit so we measure the model's auto-LID accuracy.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root, language = DATASETS[args.dataset]
    out_root = args.output_root / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    utterances = _load_yooz_dataset(Path(root))
    if args.limit:
        utterances = utterances[: args.limit]
    if not utterances:
        raise SystemExit(f"no audio found in {root}")
    print(f"{args.dataset}: {len(utterances)} utterances ({language})")

    summary: list[dict] = []
    for model_key in args.models:
        if model_key not in MODELS:
            raise SystemExit(f"unknown model key: {model_key}")
        repo = MODELS[model_key]
        print(f"\n=== {model_key} ({repo}) ===")
        try:
            model = load_stt(repo)
        except Exception as e:  # noqa: BLE001
            print(f"  load FAILED: {e}")
            continue

        is_qwen = "qwen3_asr" in model_key
        hint = language if (is_qwen and args.language_hint) else None
        records: list[Hypothesis] = []
        failures: list[dict] = []
        for utt in utterances:
            try:
                hyp_text, secs = _generate_text(model, utt.audio_path, hint)
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                print(f"  {utt.audio_path.name} FAILED: {msg}")
                failures.append({"audio": utt.audio_path.name, "error": msg})
                continue
            wer, cer = _wer_cer(utt.reference, hyp_text, language)
            records.append(
                Hypothesis(
                    audio=utt.audio_path.name,
                    reference=utt.reference,
                    hypothesis=hyp_text,
                    wer=wer,
                    cer=cer,
                    inference_seconds=secs,
                )
            )
            print(f"  {utt.audio_path.name}  WER {wer:.3f}  CER {cer:.3f}  {secs:.2f}s")

        per_utt_path = out_root / f"{model_key}.jsonl"
        with per_utt_path.open("w") as f:
            for r in records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        if failures:
            err_path = out_root / f"{model_key}.errors.jsonl"
            with err_path.open("w") as f:
                for entry in failures:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(
                f"  WARN: {len(failures)}/{len(utterances)} utterances FAILED "
                f"(see {err_path.name}); aggregates below cover "
                f"{len(records)} succeeding utterances only."
            )

        if records:
            agg = {
                "dataset": args.dataset,
                "language": language,
                "model": model_key,
                "repo": repo,
                "n": len(records),
                "n_failed": len(failures),
                "n_total": len(utterances),
                "wer_mean": sum(r.wer for r in records) / len(records),
                "cer_mean": sum(r.cer for r in records) / len(records),
                "rtf_mean": sum(r.inference_seconds for r in records) / len(records),
                "language_hint": bool(hint),
            }
            summary.append(agg)
            print(f"  AGG: WER {agg['wer_mean']:.3f}  CER {agg['cer_mean']:.3f}")
        del model

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
