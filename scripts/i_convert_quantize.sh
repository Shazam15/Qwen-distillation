#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-$REPO_ROOT/models/merged}"
GGUF_DIR="$REPO_ROOT/models/gguf"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$REPO_ROOT/../llama.cpp}"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"
MODEL_NAME="${MODEL_NAME:-atlas-distill-v1}"

if [ ! -d "$MERGED_MODEL_DIR" ]; then
  echo "ERROR: $MERGED_MODEL_DIR not found - run 07_merge_lora.py first." >&2
  exit 1
fi

mkdir -p "$GGUF_DIR"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo "==> llama.cpp not found at $LLAMA_CPP_DIR - cloning it"
  git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
fi

if [ ! -x "$LLAMA_CPP_DIR/llama-quantize" ]; then
  echo "==> Building llama.cpp (llama-quantize binary not found)"
  cmake -B "$LLAMA_CPP_DIR/build" -S "$LLAMA_CPP_DIR"
  cmake --build "$LLAMA_CPP_DIR/build" --config Release -j "$(nproc)"
  cp "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$LLAMA_CPP_DIR/llama-quantize"
fi

echo "==> Converting $MERGED_MODEL_DIR to f16 GGUF"
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$MERGED_MODEL_DIR" \
  --outfile "$GGUF_DIR/${MODEL_NAME}-f16.gguf" \
  --outtype f16

echo "==> Quantizing to $QUANT_TYPE (matches ATLAS's current qwen3:14b-q4_K_M scheme)"
"$LLAMA_CPP_DIR/llama-quantize" \
  "$GGUF_DIR/${MODEL_NAME}-f16.gguf" \
  "$GGUF_DIR/${MODEL_NAME}-${QUANT_TYPE}.gguf" \
  "$QUANT_TYPE"

echo "==> Done: $GGUF_DIR/${MODEL_NAME}-${QUANT_TYPE}.gguf"
echo "    Optional cleanup (frees disk space, no longer needed once quantized):"
echo "      rm $GGUF_DIR/${MODEL_NAME}-f16.gguf"
