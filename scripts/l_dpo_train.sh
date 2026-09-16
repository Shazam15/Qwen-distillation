#!/usr/bin/env bash
set -euo pipefail

# Optional Phase 11 - only run this after 06_evaluate.py's gate passed on the SFT model
# and you want a further push. Needs data/dpo_ready/ built first: chosen = teacher
# completions, rejected = the PRE-DISTILLATION baseline model's completions on the same
# prompts (generate that "rejected" column with a small script calling common.call_ollama_chat
# against qwen3:14b-q4_K_M over the eval-excluded prompts).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-$REPO_ROOT/models/merged}"
DPO_DATA_PATH="${DPO_DATA_PATH:-$REPO_ROOT/data/dpo_ready}"
PROJECT_NAME="${PROJECT_NAME:-atlas-distill-v1-dpo}"
OUTPUT_DIR="$REPO_ROOT/models/dpo_adapter"

LEARNING_RATE="${LEARNING_RATE:-5e-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-1}"

if [ ! -d "$MERGED_MODEL_DIR" ]; then
  echo "ERROR: $MERGED_MODEL_DIR not found - run 07_merge_lora.py first." >&2
  exit 1
fi
if [ ! -d "$DPO_DATA_PATH" ]; then
  echo "ERROR: $DPO_DATA_PATH not found - build the chosen/rejected pairs first (see the note above)." >&2
  exit 1
fi

echo "==> Running optional DPO refinement on $MERGED_MODEL_DIR"
autotrain llm \
  --train \
  --trainer dpo \
  --model "$MERGED_MODEL_DIR" \
  --project-name "$PROJECT_NAME" \
  --data-path "$DPO_DATA_PATH" \
  --peft \
  --quantization int4 \
  --lr "$LEARNING_RATE" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation "$GRAD_ACCUM" \
  --epochs "$EPOCHS"

mkdir -p "$(dirname "$OUTPUT_DIR")"
if [ -d "$PROJECT_NAME" ]; then
  # Same fix as 05_train_qlora.sh: always take the freshest run's output rather than
  # silently keeping a stale adapter when OUTPUT_DIR already exists from a prior run.
  if [ -d "$OUTPUT_DIR" ]; then
    echo "    Replacing existing adapter at $OUTPUT_DIR with this run's output."
    rm -rf "$OUTPUT_DIR"
  fi
  mv "$PROJECT_NAME" "$OUTPUT_DIR"
fi

echo "==> DPO training complete. Adapter: $OUTPUT_DIR"
echo "    Merge it on top of $MERGED_MODEL_DIR the same way 07_merge_lora.py did (point"
echo "    --base-model at $MERGED_MODEL_DIR and --adapter-path at $OUTPUT_DIR), then re-run 08/09/10."
