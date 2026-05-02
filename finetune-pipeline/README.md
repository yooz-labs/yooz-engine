# QLoRA Fine-Tuning Pipeline

Fine-tune Yooz LLM models for voice transcription proofreading and rewriting.

## Prerequisites

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate yooz
uv pip install "mlx-lm[train]" pyyaml
```

## Quick Start

```bash
cd ai-touchup/finetune

# 1. Prepare training data from yooz-benchmark gold standard
python scripts/prepare_data.py \
  --input /Users/yahya/Documents/git/yooz/yooz-benchmark/data/gold_standard.jsonl

# 2. Train Light model (Qwen2.5-0.5B, ~10 min)
python scripts/train.py --config configs/light.yaml

# 3. Train Quality model (Qwen3-1.7B, ~20 min)
python scripts/train.py --config configs/quality.yaml

# 4. Evaluate against baseline
python scripts/evaluate.py \
  --config configs/light.yaml \
  --test-set data/light/test.jsonl

python scripts/evaluate.py \
  --config configs/quality.yaml \
  --test-set data/quality/test.jsonl

# 5. Fuse adapters into base model (if eval looks good)
python scripts/fuse_model.py \
  --config configs/light.yaml \
  --save-path outputs/light/fused

python scripts/fuse_model.py \
  --config configs/quality.yaml \
  --save-path outputs/quality/fused
```

## Models

| Model | HuggingFace ID | Size | Use Case |
|-------|----------------|------|----------|
| Yooz-Light | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | 278 MB | Fast proofreading (standard mode) |
| Yooz-Quality | `mlx-community/Qwen3-1.7B-Instruct-4bit` | 1.0 GB | Full rewriting (full mode) |

## Training Data

Source: `yooz-benchmark` repo (`data/gold_standard.jsonl`)

Each entry has:
- `raw_transcription`: original STT output
- `proofread`: minimal correction (standard mode target)
- `rewrite`: full editorial rewrite (full mode target)

Each entry produces two training samples (one per mode), so ~3,000 entries yield ~6,000 samples.

## Directory Layout

```
finetune/
  configs/          # Training configs (tracked)
  scripts/          # Pipeline scripts (tracked)
  data/             # Processed training data (gitignored)
  outputs/          # Adapters, fused models, eval results (gitignored)
```
