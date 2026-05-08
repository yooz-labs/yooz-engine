#!/usr/bin/env bash
# Sequential driver for issue #9 phases. Avoids GPU contention.
# Logs go to /Volumes/S1/yooz/research/issue-9/logs/.

set -euo pipefail

ROOT="/Users/yahya/Documents/git/yooz/engine-issue-9-llm"
LOGDIR="/Volumes/S1/yooz/research/issue-9/logs"
mkdir -p "$LOGDIR"

# shellcheck disable=SC1091
source "$ROOT/.envrc"
cd "$ROOT"

phase="${1:-all}"

run_phase1() {
    echo "=== phase 1: quality ==="
    PYTHONUNBUFFERED=1 python -u scripts/run_phase1_quality.py \
        2>&1 | tee "$LOGDIR/phase1_full.log"
}

run_phase2() {
    echo "=== phase 2: speed ==="
    PYTHONUNBUFFERED=1 python -u scripts/run_phase2_speed.py \
        2>&1 | tee "$LOGDIR/phase2_full.log"
}

run_phase3() {
    echo "=== phase 3: fine-tune ==="
    cd "$ROOT/finetune-pipeline"
    for cfg in configs/qwen35-light.yaml configs/qwen35-quality.yaml; do
        name="$(basename "$cfg" .yaml)"
        echo "--- training $name ---"
        PYTHONUNBUFFERED=1 python -u scripts/train.py --config "$cfg" \
            2>&1 | tee "$LOGDIR/phase3_${name}_train.log"
        echo "--- evaluating $name ---"
        PYTHONUNBUFFERED=1 python -u scripts/evaluate.py \
            --config "$cfg" \
            --test-set "data/$(echo "$name" | sed 's/qwen35-//')/test.jsonl" \
            --output "/Volumes/S1/yooz/research/issue-9/results/phase3_finetune/$name/eval.json" \
            2>&1 | tee "$LOGDIR/phase3_${name}_eval.log"
    done
}

case "$phase" in
    1|phase1) run_phase1 ;;
    2|phase2) run_phase2 ;;
    3|phase3) run_phase3 ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        ;;
    *)
        echo "Usage: $0 [1|2|3|all]"
        exit 1
        ;;
esac

echo "=== done ==="
