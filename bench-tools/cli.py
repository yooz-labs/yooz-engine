"""CLI for the benchmark pipeline."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import click

DATA_DIR = Path(__file__).parent.parent / "data"

# Source registry: name -> "module.path:ClassName"
# Uses lazy imports so missing source files don't break the CLI.
SOURCE_REGISTRY: dict[str, str] = {
    "debug-logs": "benchmarks.sources.debug_logs:DebugLogSource",
    "ami": "benchmarks.sources.ami:AMISource",
    "peoples-speech": "benchmarks.sources.peoples_speech:PeoplesSpeechSource",
    "switchboard": "benchmarks.sources.switchboard:SwitchboardSource",
    "common-voice": "benchmarks.sources.common_voice:CommonVoiceSource",
    "santa-barbara": "benchmarks.sources.santa_barbara:SantaBarbaraSource",
}

EXTERNAL_SOURCES = [k for k in SOURCE_REGISTRY if k != "debug-logs"]


def _load_source_class(name: str) -> type:
    """Lazily import a source class from the registry."""
    if name not in SOURCE_REGISTRY:
        available = ", ".join(SOURCE_REGISTRY)
        raise click.BadParameter(f"Unknown source: {name}. Available: {available}")

    module_path, class_name = SOURCE_REGISTRY[name].rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise click.ClickException(
            f"Source '{name}' is registered but its module could not be imported: {e}"
        ) from e
    return getattr(module, class_name)


@click.group()
def cli() -> None:
    """STT Benchmark Pipeline - extract, annotate, and evaluate."""


@cli.command()
@click.option(
    "--source",
    default="debug-logs",
    type=click.Choice(list(SOURCE_REGISTRY.keys())),
    help="Data source to extract from.",
)
@click.option(
    "--output",
    default=None,
    help="Output JSONL path (default: data/<source>_raw_entries.jsonl).",
)
@click.option(
    "--logs-dir",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Override debug logs directory (debug-logs source only).",
)
@click.option(
    "--data-dir",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Override data directory for external sources.",
)
@click.option("--dry-run", is_flag=True, help="Show stats without writing.")
def extract(
    source: str,
    output: str | None,
    logs_dir: Path | None,
    data_dir: Path | None,
    dry_run: bool,
) -> None:
    """Step 1: Extract raw transcriptions from a source."""
    from benchmarks.extract import extract as do_extract

    source_class = _load_source_class(source)

    if source == "debug-logs":
        src = source_class(log_dir=logs_dir)
    else:
        src = source_class(data_dir=data_dir) if data_dir else source_class()

    if output:
        out = Path(output)
    elif source == "debug-logs":
        out = DATA_DIR / "raw_entries.jsonl"
    else:
        out = DATA_DIR / f"{source}_raw_entries.jsonl"

    do_extract(src, out, dry_run=dry_run)


@cli.command()
@click.option(
    "--source",
    required=True,
    type=click.Choice(EXTERNAL_SOURCES),
    help="External dataset source.",
)
@click.option(
    "--data-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Override data directory for external sources.",
)
@click.option(
    "--max-entries",
    default=None,
    type=int,
    help="Maximum entries to download (sources that support it).",
)
def download(
    source: str,
    data_dir: Path | None,
    max_entries: int | None,
) -> None:
    """Download audio data for an external dataset."""
    source_class = _load_source_class(source)
    src = source_class(data_dir=data_dir) if data_dir else source_class()

    kwargs: dict[str, object] = {}
    if max_entries is not None:
        kwargs["max_entries"] = max_entries

    if not hasattr(src, "download"):
        raise click.ClickException(f"Source '{source}' does not support download.")

    src.download(**kwargs)
    print(f"Download complete for {source}. Data in {src.base_dir}")


@cli.command()
@click.option(
    "--source",
    required=True,
    type=click.Choice(EXTERNAL_SOURCES),
    help="External dataset source.",
)
@click.option(
    "--data-dir",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Override data directory for external sources.",
)
def segment(
    source: str,
    data_dir: Path | None,
) -> None:
    """Segment downloaded audio into utterances (AMI, Santa Barbara)."""
    source_class = _load_source_class(source)
    src = source_class(data_dir=data_dir) if data_dir else source_class()

    if not hasattr(src, "segment"):
        raise click.ClickException(
            f"Source '{source}' does not need segmentation "
            "(audio is pre-segmented during download)."
        )

    src.segment()
    print(f"Segmentation complete for {source}. Segments in {src.segments_dir}")


@cli.command()
@click.option(
    "--source",
    required=True,
    type=click.Choice(EXTERNAL_SOURCES),
    help="External dataset source.",
)
@click.option(
    "--engine-dir",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to yooz-stt-engine repo.",
)
@click.option(
    "--model",
    default=None,
    help="Model path or HuggingFace ID (default: parakeet-tdt-0.6b-v3).",
)
@click.option(
    "--data-dir",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Override data directory for external sources.",
)
def transcribe(
    source: str,
    engine_dir: Path | None,
    model: str | None,
    data_dir: Path | None,
) -> None:
    """Run STT engine on segmented audio for an external dataset."""
    from benchmarks.stt import DEFAULT_MODEL, transcribe_batch

    source_class = _load_source_class(source)
    src = source_class(data_dir=data_dir) if data_dir else source_class()

    segments_dir = src.segments_dir
    if not segments_dir.exists() or not any(segments_dir.iterdir()):
        raise click.ClickException(
            f"No segments found in {segments_dir}. Download and segment audio first."
        )

    audio_files = sorted(
        p
        for p in segments_dir.iterdir()
        if p.suffix.lower() in {".wav", ".flac", ".mp3"}
    )

    engine = Path(engine_dir) if engine_dir else None
    progress_file = src.base_dir / "stt_progress.jsonl"

    results = transcribe_batch(
        audio_files=audio_files,
        output_dir=src.transcriptions_dir,
        engine_dir=engine,
        progress_file=progress_file,
        model=model or DEFAULT_MODEL,
    )
    print(f"Transcribed {len(results)} files. Output in {src.transcriptions_dir}")


@cli.command()
@click.option(
    "--input",
    "input_path",
    default=None,
    help="Input JSONL path (default: data/raw_entries.jsonl).",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    help="Output JSONL path (default: data/gold_standard.jsonl).",
)
@click.option("--batch-size", default=20, help="Entries per API call.")
@click.option("--concurrency", default=5, help="Max concurrent API calls.")
@click.option(
    "--model",
    default="claude-sonnet-4-6",
    help="Claude model ID.",
)
@click.option(
    "--sample",
    default=None,
    type=int,
    help="Process only N entries (for testing).",
)
@click.option("--dry-run", is_flag=True, help="Show what would be processed.")
@click.option("--api-key", default=None, help="Anthropic API key.")
def annotate(
    input_path: str | None,
    output_path: str | None,
    batch_size: int,
    concurrency: int,
    model: str,
    sample: int | None,
    dry_run: bool,
    api_key: str | None,
) -> None:
    """Step 2: Generate gold-standard annotations with Claude."""
    from benchmarks.annotate import annotate as do_annotate

    inp = Path(input_path) if input_path else DATA_DIR / "raw_entries.jsonl"
    out = Path(output_path) if output_path else DATA_DIR / "gold_standard.jsonl"
    progress = out.parent / f"progress_{out.stem}.jsonl"

    if not inp.exists():
        raise click.ClickException(
            f"Input file not found: {inp}\nRun 'extract' first to generate it."
        )

    _load_env(out.parent.parent)

    asyncio.run(
        do_annotate(
            input_path=inp,
            output_path=out,
            progress_path=progress,
            batch_size=batch_size,
            concurrency=concurrency,
            model=model,
            sample=sample,
            dry_run=dry_run,
            api_key=api_key,
        )
    )


@cli.command()
@click.option(
    "--input",
    "input_path",
    default=None,
    help="Gold standard JSONL path (default: data/gold_standard.jsonl).",
)
def stats(input_path: str | None) -> None:
    """Show statistics for the gold-standard dataset."""
    from benchmarks.models import GoldEntry

    path = Path(input_path) if input_path else DATA_DIR / "gold_standard.jsonl"

    if not path.exists():
        raise click.ClickException(f"File not found: {path}")

    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(GoldEntry.model_validate_json(line))

    if not entries:
        print("No entries found.")
        return

    word_counts = [e.word_count for e in entries]
    difficulties: dict[str, int] = {}
    domains: dict[str, int] = {}
    error_types: dict[str, int] = {}
    sources: dict[str, int] = {}

    for e in entries:
        d = e.metadata.get("difficulty", "unknown")
        difficulties[d] = difficulties.get(d, 0) + 1
        dom = e.metadata.get("domain", "unknown")
        domains[dom] = domains.get(dom, 0) + 1
        src = e.source
        sources[src] = sources.get(src, 0) + 1
        for et in e.metadata.get("error_types", []):
            error_types[et] = error_types.get(et, 0) + 1

    print(f"\n{'=' * 50}")
    print("Gold Standard Dataset Statistics")
    print(f"{'=' * 50}")
    print(f"Total entries:  {len(entries)}")
    print(
        f"Word count:     min={min(word_counts)}, max={max(word_counts)}, "
        f"avg={sum(word_counts) / len(word_counts):.1f}"
    )
    print(f"\nSources:        {sources}")
    print(f"Difficulties:   {difficulties}")
    print(f"Domains:        {domains}")
    print("\nError types:")
    for et, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"  {et}: {count}")
    print(f"{'=' * 50}")

    print("\nSample entries:")
    for e in entries[:3]:
        print(
            f"\n  [{e.id[:8]}] {e.word_count} words, "
            f"{e.metadata.get('difficulty', '?')}, {e.metadata.get('domain', '?')}"
        )
        print(f"    raw:       {e.raw_transcription[:80]}...")
        print(f"    proofread: {e.proofread[:80]}...")
        print(f"    rewrite:   {e.rewrite[:80]}...")


@cli.command()
@click.option(
    "--repo",
    required=True,
    help="MLX-Swift HuggingFace model id (e.g. mlx-community/Qwen3.5-0.8B-MLX-4bit).",
)
@click.option(
    "--family",
    type=click.Choice(["qwen2", "qwen3", "qwen35", "gemma4"]),
    required=True,
    help="Chat-template family (controls /no_think, channel forcing, etc.).",
)
@click.option(
    "--gold",
    default=None,
    type=click.Path(path_type=Path),
    help="Gold standard JSONL (default: bench-tools/gold_standard.jsonl in this repo).",
)
@click.option(
    "--limit",
    default=0,
    type=int,
    help="Cap test samples (0 = all).",
)
@click.option(
    "--max-tokens",
    default=320,
    type=int,
)
@click.option(
    "--results-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Output dir for {slug}.json (default: results/eval/).",
)
@click.option("--slug", default=None, help="Filename slug (default: derived from repo).")
def evaluate(
    repo: str,
    family: str,
    gold: Path | None,
    limit: int,
    max_tokens: int,
    results_dir: Path | None,
    slug: str | None,
) -> None:
    """Evaluate a single MLX model against the gold-standard touchup set.

    Runs proofread + rewrite for every sample, computes exact-match,
    character-error-rate (jiwer), and semantic similarity (MiniLM).
    Output schema matches scripts/run_phase1_quality.py so results compose.
    """
    # Lazy import — heavy deps (mlx_lm, sentence-transformers) only on use.
    import sys
    import importlib.util

    here = Path(__file__).resolve().parent.parent
    spec_path = here / "scripts" / "run_phase1_quality.py"
    if not spec_path.exists():
        raise click.ClickException(
            f"Phase 1 harness missing at {spec_path}. Run from the engine-issue-9-llm "
            "worktree, or upstream the script first."
        )
    spec = importlib.util.spec_from_file_location("run_phase1_quality", spec_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_phase1_quality"] = mod
    spec.loader.exec_module(mod)

    gold_path = gold or (here / "bench-tools" / "gold_standard.jsonl")
    samples = mod.load_test_set(gold_path, limit or None)
    print(f"Loaded {len(samples)} samples from {gold_path}")

    from sentence_transformers import SentenceTransformer
    import jiwer

    sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    derived_slug = slug or repo.split("/")[-1].lower()
    spec_obj = mod.ModelSpec(
        slug=derived_slug, repo=repo, family=family, role="ad-hoc",
    )

    # Override RESULTS_DIR if requested.
    if results_dir is not None:
        mod.RESULTS_DIR = Path(results_dir)
        mod.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = mod.evaluate_model(spec_obj, samples, sim_model, jiwer, max_tokens)
    if "error" in summary:
        raise click.ClickException(f"Evaluation failed: {summary['error']}")

    t = summary["total"]
    print(
        f"\n{spec_obj.slug}  EM={t['exact_match'] * 100:.1f}%  "
        f"JSON={t['json_ok'] * 100:.1f}%  CER={t['avg_cer']:.4f}  "
        f"Sim={t['avg_semantic_similarity']:.4f}  Lat={t['avg_latency_ms']:.0f}ms"
    )


def _load_env(base_dir: Path) -> None:
    """Load .env file if it exists."""
    import os

    env_file = base_dir / ".env"
    if not env_file.exists():
        return

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
