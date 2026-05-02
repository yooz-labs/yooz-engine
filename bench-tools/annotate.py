"""Async annotation engine using Claude API."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import anthropic
from tqdm import tqdm

from benchmarks.extract import load_raw_entries
from benchmarks.models import Annotation, GoldEntry, RawEntry
from benchmarks.progress import ProgressTracker
from benchmarks.prompts import SYSTEM_PROMPT, build_batch_prompt

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_BATCH_SIZE = 20
DEFAULT_CONCURRENCY = 5


async def annotate(
    input_path: Path,
    output_path: Path,
    progress_path: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
    model: str = DEFAULT_MODEL,
    sample: int | None = None,
    dry_run: bool = False,
    api_key: str | None = None,
) -> None:
    """Run the annotation pipeline.

    Args:
        input_path: Path to raw_entries.jsonl.
        output_path: Path to write gold_standard.jsonl.
        progress_path: Path to progress.jsonl.
        batch_size: Entries per API call.
        concurrency: Max concurrent API calls.
        model: Claude model ID.
        sample: Process only this many entries (for testing).
        dry_run: Show what would be processed without calling the API.
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
    """
    entries = load_raw_entries(input_path)
    tracker = ProgressTracker(progress_path)

    remaining_entries = [e for e in entries if not tracker.is_done(e.id)]

    if sample is not None:
        remaining_entries = remaining_entries[:sample]

    print(f"Total entries:     {len(entries)}")
    print(f"Already done:      {tracker.done_count}")
    print(f"Errors (retrying): {tracker.error_count}")
    print(f"To process:        {len(remaining_entries)}")

    if dry_run or not remaining_entries:
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    batches = _make_batches(remaining_entries, batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=len(remaining_entries), desc="Annotating", unit="entry")

    tasks = [
        _process_batch(client, model, batch, output_path, tracker, semaphore, pbar)
        for batch in batches
    ]
    await asyncio.gather(*tasks)

    pbar.close()
    print(f"\nDone. {tracker.done_count} entries in {output_path}")


async def _process_batch(
    client: anthropic.AsyncAnthropic,
    model: str,
    batch: list[RawEntry],
    output_path: Path,
    tracker: ProgressTracker,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
) -> None:
    """Process a single batch of entries."""
    async with semaphore:
        annotations = await _call_api(client, model, batch)

        if annotations is None:
            # Retry with smaller batches
            if len(batch) > 1:
                mid = len(batch) // 2
                await _process_batch(
                    client,
                    model,
                    batch[:mid],
                    output_path,
                    tracker,
                    semaphore,
                    pbar,
                )
                await _process_batch(
                    client,
                    model,
                    batch[mid:],
                    output_path,
                    tracker,
                    semaphore,
                    pbar,
                )
                return

            # Single entry also failed
            for entry in batch:
                tracker.mark_error(entry.id, "api_failure")
                pbar.update(1)
            return

        _write_results(batch, annotations, output_path, tracker)
        pbar.update(len(batch))


async def _call_api(
    client: anthropic.AsyncAnthropic,
    model: str,
    batch: list[RawEntry],
    max_retries: int = 3,
) -> list[Annotation] | None:
    """Call Claude API for a batch, return parsed annotations or None on failure."""
    texts = [{"idx": i, "text": e.raw_transcription} for i, e in enumerate(batch)]
    user_prompt = build_batch_prompt(texts)

    # Estimate output tokens: each entry produces ~200 tokens of JSON output
    # (proofread + rewrite + metadata). Add buffer for longer entries.
    avg_input_words = sum(e.word_count for e in batch) / max(len(batch), 1)
    tokens_per_entry = max(200, int(avg_input_words * 3))
    max_tokens = min(max(len(batch) * tokens_per_entry, 4096), 16384)

    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            if response.stop_reason == "max_tokens":
                logger.warning(
                    f"Response truncated (max_tokens={max_tokens}, "
                    f"batch={len(batch)} entries). Will retry with smaller batch."
                )
                return None

            content = response.content[0].text
            return _parse_response(content, len(batch))

        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt + 1})")
            await asyncio.sleep(wait)
        except anthropic.APIError as e:
            logger.warning(f"API error: {e} (attempt {attempt + 1})")
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    return None


def _parse_response(content: str, expected_count: int) -> list[Annotation] | None:
    """Parse Claude's JSON array response into Annotation objects."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON response: {content[:200]}...")
        return None

    if not isinstance(data, list):
        logger.error(f"Expected JSON array, got {type(data).__name__}")
        return None

    if len(data) != expected_count:
        logger.warning(
            f"Expected {expected_count} items, got {len(data)}. Using what we got."
        )

    annotations = []
    for item in data:
        try:
            annotations.append(Annotation.model_validate(item))
        except Exception as e:
            logger.warning(f"Failed to parse annotation item: {e}")
            annotations.append(
                Annotation(
                    proofread=item.get("proofread", ""),
                    rewrite=item.get("rewrite", ""),
                    difficulty=item.get("difficulty", "medium"),
                    error_types=item.get("error_types", []),
                    domain=item.get("domain", "casual"),
                )
            )

    return annotations


def _write_results(
    batch: list[RawEntry],
    annotations: list[Annotation],
    output_path: Path,
    tracker: ProgressTracker,
) -> None:
    """Write annotated entries to output file and update progress."""
    with open(output_path, "a") as f:
        for entry, ann in zip(batch, annotations, strict=False):
            gold = GoldEntry(
                id=entry.id,
                source=entry.source,
                raw_transcription=entry.raw_transcription,
                proofread=ann.proofread,
                rewrite=ann.rewrite,
                word_count=entry.word_count,
                metadata={
                    **entry.metadata,
                    "difficulty": ann.difficulty,
                    "error_types": ann.error_types,
                    "domain": ann.domain,
                },
            )
            f.write(gold.model_dump_json() + "\n")
            tracker.mark_done(entry.id)


def _make_batches(entries: list[RawEntry], batch_size: int) -> list[list[RawEntry]]:
    """Split entries into batches, reducing size for long entries."""
    batches = []
    current_batch: list[RawEntry] = []
    current_words = 0
    # Cap total words per batch to avoid output token overflow.
    # 20 entries * 50 avg words = 1000 words is the baseline.
    max_words_per_batch = 1500

    for entry in entries:
        if current_batch and (
            len(current_batch) >= batch_size
            or current_words + entry.word_count > max_words_per_batch
        ):
            batches.append(current_batch)
            current_batch = []
            current_words = 0
        current_batch.append(entry)
        current_words += entry.word_count

    if current_batch:
        batches.append(current_batch)

    return batches
