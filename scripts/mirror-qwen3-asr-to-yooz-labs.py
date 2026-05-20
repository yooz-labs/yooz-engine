#!/usr/bin/env python3
# mirror-qwen3-asr-to-yooz-labs.py
#
# Mirror the qwen3 ASR 1.7B 8-bit model to YoozLabs/Qwen3-ASR-1.7B-8bit
# on HuggingFace with a generated tokenizer.json bundled in. The
# canonical mlx-community drop ships only legacy BPE artifacts
# (vocab.json + merges.txt + tokenizer_config.json) and omits the
# unified tokenizer.json that swift-transformers' AutoTokenizer
# requires. Without the mirror, the engine's first-run fetch
# downloads all files cleanly but then fails tokenizer prep with
# `tokenizer_validation_failed: configurationMissing("tokenizer.json")`.
#
# Usage:
#   uv tool install --with transformers --with tokenizers \
#                   --with huggingface_hub transformers
#   export HF_TOKEN=hf_yourtoken
#   python3 scripts/mirror-qwen3-asr-to-yooz-labs.py
#
# Steps:
#   1. Snapshot-download the canonical mlx-community/Qwen3-ASR-1.7B-8bit
#      into a temp dir (so we have known-good originals, not the
#      potentially-corrupted local cache).
#   2. Generate tokenizer.json via AutoTokenizer.from_pretrained +
#      save_pretrained into a SEPARATE dir so save_pretrained's
#      legacy-config rewrite doesn't clobber the original
#      tokenizer_config.json (verified bug: corrupts the JSON).
#   3. Copy ONLY the generated tokenizer.json back into the snapshot.
#   4. Upload the snapshot to YoozLabs/Qwen3-ASR-1.7B-8bit.
#
# After upload, bump engine's `Qwen3ASRModelFetcher.canonicalRepo`
# constant to point at the mirror in a follow-up engine PR.

from pathlib import Path
import os
import shutil
import sys
import tempfile

UPSTREAM_REPO = "mlx-community/Qwen3-ASR-1.7B-8bit"
MIRROR_REPO = "YoozLabs/Qwen3-ASR-1.7B-8bit"


def main() -> int:
    try:
        from huggingface_hub import snapshot_download, create_repo, upload_folder
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(f"missing dep: {exc}", file=sys.stderr)
        print("install with: uv tool install --with transformers --with tokenizers --with huggingface_hub transformers", file=sys.stderr)
        return 1

    if "HF_TOKEN" not in os.environ:
        print("HF_TOKEN env var required (HuggingFace personal access token with write to YoozLabs)", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="qwen3-mirror-") as tmp:
        tmp_dir = Path(tmp)
        snapshot = tmp_dir / "snapshot"
        tok_dump = tmp_dir / "tok-dump"

        print(f"[1/4] snapshot-download {UPSTREAM_REPO} -> {snapshot}")
        snapshot_download(
            repo_id=UPSTREAM_REPO,
            local_dir=str(snapshot),
            local_dir_use_symlinks=False,
        )

        print(f"[2/4] generate tokenizer.json via AutoTokenizer -> {tok_dump}")
        tok = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=False)
        tok.save_pretrained(str(tok_dump))
        if not (tok_dump / "tokenizer.json").exists():
            print("tokenizer.json was not generated", file=sys.stderr)
            return 1

        print("[3/4] graft generated tokenizer.json into snapshot")
        # Only copy tokenizer.json — save_pretrained rewrites
        # tokenizer_config.json in an incompatible format that yyjson
        # (swift-transformers' parser) rejects with "unexpected content
        # after document". Keep the canonical config from upstream.
        shutil.copy2(tok_dump / "tokenizer.json", snapshot / "tokenizer.json")

        print(f"[4/4] upload to {MIRROR_REPO}")
        create_repo(repo_id=MIRROR_REPO, exist_ok=True, repo_type="model")
        upload_folder(
            folder_path=str(snapshot),
            repo_id=MIRROR_REPO,
            commit_message=(
                "Mirror of mlx-community/Qwen3-ASR-1.7B-8bit with generated "
                "tokenizer.json so swift-transformers AutoTokenizer can load"
            ),
        )

    print(f"\nDone. Update engine's Qwen3ASRModelFetcher.canonicalRepo to:")
    print(f'    public static let canonicalRepo = "{MIRROR_REPO}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
