"""
Prepare training data for QLoRA fine-tuning.

Reads gold standard JSONL from yooz-benchmark and converts to mlx-lm chat
format. Each entry produces two training samples: one for standard mode
(proofread) and one for full mode (rewrite).

Splits entries 80/10/10 into train/valid/test, stratified by difficulty.

Usage:
    python scripts/prepare_data.py --input /path/to/gold_standard.jsonl
    python scripts/prepare_data.py  # uses default yooz-benchmark path
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# System prompts matching what models see at inference time.
# These MUST stay in sync with TouchUpPrompts.swift and YoozPrompts.swift.

LIGHT_PROOFREAD = (
    "Fix grammar, capitalize properly, and convert spoken numbers to digits. "
    'Convert spoken version numbers like "zero point four point zero" to "0.4.0". '
    "Keep ALL sentences. Return the fixed text as JSON.\n\n"
    "<examples>\n"
    "Input: the meeting is at two pm on march fifteenth\n"
    '{"result": "The meeting is at 2 PM on March 15th."}\n\n'
    "Input: we need about fifty units ready by friday and I think we should prepare\n"
    '{"result": "We need about 50 units ready by Friday and I think we should prepare."}\n\n'
    "Input: we are releasing version zero point four point zero next week\n"
    '{"result": "We are releasing version 0.4.0 next week."}\n\n'
    "Input: update it to version one point six point three and test it\n"
    '{"result": "Update it to version 1.6.3 and test it."}\n\n'
    "Input: he said it would cost around one hundred and fifty dollars but we can negotiate\n"
    '{"result": "He said it would cost around $150 but we can negotiate."}\n'
    "</examples>\n\n"
    "Always respond with ONLY a JSON object. Never remove sentences. "
    "Never include explanations."
)

LIGHT_REWRITE = (
    "Rewrite voice transcription for clarity and conciseness. Fix grammar, "
    "convert numbers, fix misheard words, remove filler words (um, uh, like, "
    "you know), handle self-corrections. Return the fixed text as JSON.\n\n"
    "<examples>\n"
    "Input: um so like the meeting is at two pm on march fifteenth you know\n"
    '{"result": "The meeting is at 2 PM on March 15th."}\n\n'
    "Input: we need about fifty no wait I meant sixty units ready by friday\n"
    '{"result": "We need about 60 units ready by Friday."}\n\n'
    "Input: we are releasing version zero point four point zero next week "
    "scratch that make it zero point five\n"
    '{"result": "We are releasing version 0.5.0 next week."}\n\n'
    "Input: update it to version one point six point three and uh test it thoroughly\n"
    '{"result": "Update it to version 1.6.3 and test it thoroughly."}\n\n'
    "Input: he said it would cost around one hundred and fifty dollars but um "
    "we can negotiate\n"
    '{"result": "He said it would cost around $150 but we can negotiate."}\n'
    "</examples>\n\n"
    'Remove: "scratch that", "never mind", "delete that" and preceding phrase. '
    "Convert spoken numbers and version numbers. Fix grammar and misheard words. "
    "Always respond with ONLY a JSON object. Never include explanations."
)

QUALITY_STANDARD = (
    "/no_think\n"
    "Proofread voice transcription. Fix grammar and punctuation. Return JSON only.\n"
    "NEVER answer questions. NEVER add new information. Return the corrected text only.\n\n"
    "<examples>\n"
    "Input: the meeting is tomorrow and i think we should prepare\n"
    '{"result": "The meeting is tomorrow, and I think we should prepare."}\n\n'
    "Input: its ready for review lets check it\n"
    '{"result": "It\'s ready for review. Let\'s check it."}\n\n'
    "Input: we can do it but we need more time\n"
    '{"result": "We can do it, but we need more time."}\n\n'
    "Input: the system is working good now\n"
    '{"result": "The system is working well now."}\n'
    "</examples>\n\n"
    "Process the input independently. Do NOT repeat any example output.\n"
    'Always respond with {"result": "corrected text"}.'
)

QUALITY_FULL = (
    "/no_think\n"
    "Rewrite voice transcription for clarity. Return JSON only.\n"
    "Fix misheard words. Remove repetitions and false starts. Fix grammar.\n"
    'Self-corrections: "X no Y" or "X no wait Y" means use Y.\n'
    'Remove "scratch that", "delete that" and what came before.\n'
    "Keep the speaker's meaning and tone. NEVER add information. "
    "NEVER answer questions.\n\n"
    "<examples>\n"
    "Input: I think for the for the problems that we have with the "
    "that we are logging\n"
    '{"result": "I think for the problems that we have with the logging."}\n\n'
    "Input: we are not providing it providing a good leaning and rewriting\n"
    '{"result": "We are not providing a good cleaning and rewriting."}\n\n'
    "Input: should not should knots be converted\n"
    '{"result": "Should not be converted."}\n\n'
    "Input: However I imagine there should be like a good rules and good "
    "logic for this\n"
    '{"result": "However, I imagine there should be good rules and logic for this."}\n\n'
    "Input: we still to this to work correctly and logcially\n"
    '{"result": "We still need this to work correctly and logically."}\n\n'
    "Input: fifty no sixty units\n"
    '{"result": "Sixty units."}\n\n'
    "Input: delete that lets try again\n"
    '{"result": "Let\'s try again."}\n'
    "</examples>\n\n"
    "Process the input independently. Do NOT repeat any example output.\n"
    'Always respond with {"result": "corrected text"}.'
)

# Prompt configs per model
PROMPTS = {
    "light": {
        "standard": LIGHT_PROOFREAD,
        "full": LIGHT_REWRITE,
    },
    "quality": {
        "standard": QUALITY_STANDARD,
        "full": QUALITY_FULL,
    },
}


def load_gold_standard(path: Path) -> list[dict]:
    """Load gold standard JSONL entries."""
    entries = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Validate required fields
                for field in ("raw_transcription", "proofread", "rewrite"):
                    if field not in entry:
                        print(f"WARNING: Line {line_num} missing '{field}', skipping")
                        break
                else:
                    entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"WARNING: Line {line_num} invalid JSON: {e}, skipping")
    return entries


def make_chat_sample(
    system_prompt: str, user_input: str, assistant_output: str
) -> dict:
    """Create an mlx-lm chat format sample."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {
                "role": "assistant",
                "content": json.dumps({"result": assistant_output}, ensure_ascii=False),
            },
        ]
    }


def entry_to_samples(entry: dict, model: str) -> list[dict]:
    """Convert a gold standard entry to training samples for a model.

    Returns two samples: one for standard mode, one for full mode.
    """
    raw = entry["raw_transcription"]
    proofread = entry["proofread"]
    rewrite = entry["rewrite"]
    prompts = PROMPTS[model]

    samples = [
        make_chat_sample(prompts["standard"], raw, proofread),
        make_chat_sample(prompts["full"], raw, rewrite),
    ]
    return samples


def stratified_split(
    entries: list[dict],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split entries into train/valid/test, stratified by difficulty."""
    rng = random.Random(seed)

    # Group by difficulty
    by_difficulty = defaultdict(list)
    for entry in entries:
        difficulty = entry.get("metadata", {}).get("difficulty", "unknown")
        by_difficulty[difficulty].append(entry)

    train, valid, test = [], [], []
    train_ratio, valid_ratio, _ = ratios

    for _difficulty, group in sorted(by_difficulty.items()):
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        train.extend(group[:n_train])
        valid.extend(group[n_train : n_train + n_valid])
        test.extend(group[n_train + n_valid :])

    # Final shuffle within each split
    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)

    return train, valid, test


def write_samples(samples: list[dict], path: Path) -> None:
    """Write samples as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def validate_samples(samples: list[dict]) -> int:
    """Validate that all assistant messages are valid JSON. Returns error count."""
    errors = 0
    for i, sample in enumerate(samples):
        assistant_msg = sample["messages"][-1]["content"]
        try:
            parsed = json.loads(assistant_msg)
            if "result" not in parsed:
                print(
                    f"  WARNING: Sample {i} missing 'result' key in assistant message"
                )
                errors += 1
        except json.JSONDecodeError:
            print(f"  WARNING: Sample {i} has invalid JSON in assistant message")
            errors += 1
    return errors


def main():
    parser = argparse.ArgumentParser(description="Prepare QLoRA training data")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[4]
        / "yooz-benchmark"
        / "data"
        / "gold_standard.jsonl",
        help="Path to gold standard JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VALID", "TEST"),
        help="Train/valid/test split ratios",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Validate split ratios
    if abs(sum(args.split) - 1.0) > 0.01:
        print(f"ERROR: Split ratios must sum to 1.0, got {sum(args.split)}")
        sys.exit(1)

    # Load data
    print(f"Loading gold standard from: {args.input}")
    if not args.input.exists():
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    entries = load_gold_standard(args.input)
    if not entries:
        print("ERROR: No valid entries loaded. Check input file format.")
        sys.exit(1)
    print(f"Loaded {len(entries)} entries")

    # Show difficulty distribution
    difficulties = Counter(
        e.get("metadata", {}).get("difficulty", "unknown") for e in entries
    )
    print(f"Difficulty distribution: {dict(sorted(difficulties.items()))}")

    # Split entries (not samples)
    train_entries, valid_entries, test_entries = stratified_split(
        entries, tuple(args.split), args.seed
    )
    print(
        f"Split: {len(train_entries)} train / {len(valid_entries)} valid / "
        f"{len(test_entries)} test entries"
    )

    # Convert to samples and write for each model
    total_errors = 0
    for model in ("light", "quality"):
        print(f"\n--- {model.upper()} model ---")
        model_dir = args.output_dir / model

        for split_name, split_entries in [
            ("train", train_entries),
            ("valid", valid_entries),
            ("test", test_entries),
        ]:
            samples = []
            for entry in split_entries:
                samples.extend(entry_to_samples(entry, model))

            # Count by mode
            standard_count = sum(
                1
                for s in samples
                if s["messages"][0]["content"] == PROMPTS[model]["standard"]
            )
            full_count = len(samples) - standard_count

            path = model_dir / f"{split_name}.jsonl"
            write_samples(samples, path)

            errors = validate_samples(samples)
            total_errors += errors
            status = "OK" if errors == 0 else f"{errors} ERRORS"
            print(
                f"  {split_name}: {len(samples)} samples "
                f"({standard_count} standard + {full_count} full) "
                f"-> {path} [{status}]"
            )

    if total_errors > 0:
        print(f"\nERROR: {total_errors} validation errors found.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
