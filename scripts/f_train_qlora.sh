#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-14B}"
PROJECT_NAME="${PROJECT_NAME:-atlas-distill-v1}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/autotrain_ready}"
OUTPUT_DIR="$REPO_ROOT/models/lora_adapter"

LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-3}"

echo "==> Training QLoRA adapter: $PROJECT_NAME"
echo "    base model:   $BASE_MODEL"
echo "    data path:    $DATA_PATH"
echo "    lora r/alpha: $LORA_R / $LORA_ALPHA"
echo "    epochs:       $EPOCHS"

autotrain llm \
  --train \
  --model "$BASE_MODEL" \
  --project-name "$PROJECT_NAME" \
  --data-path "$DATA_PATH" \
  --text-column messages \
  --chat-template tokenizer \
  --trainer sft \
  --peft \
  --quantization int4 \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lr "$LEARNING_RATE" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation "$GRAD_ACCUM" \
  --epochs "$EPOCHS" \
  --mixed-precision fp16

mkdir -p "$(dirname "$OUTPUT_DIR")"
if [ -d "$PROJECT_NAME" ] && [ ! -d "$OUTPUT_DIR" ]; then
  mv "$PROJECT_NAME" "$OUTPUT_DIR"
fi

echo "==> Training complete. Adapter should be under: $OUTPUT_DIR"
echo "    Next: python 06_evaluate.py --adapter-path $OUTPUT_DIR"
