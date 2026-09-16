#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_NAME="${MODEL_NAME:-atlas-distill-v1}"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"
GGUF_PATH="$REPO_ROOT/models/gguf/${MODEL_NAME}-${QUANT_TYPE}.gguf"
OLLAMA_TAG="${OLLAMA_TAG:-atlas-qwen3-14b-distilled}"
NUM_CTX="${NUM_CTX:-8192}"
MODELFILE_PATH="$REPO_ROOT/models/gguf/Modelfile"

if [ ! -f "$GGUF_PATH" ]; then
  echo "ERROR: GGUF not found at $GGUF_PATH - run 08_convert_quantize.sh first." >&2
  exit 1
fi

echo "==> Writing Modelfile to $MODELFILE_PATH"
cat > "$MODELFILE_PATH" <<EOF
FROM $GGUF_PATH
PARAMETER num_ctx $NUM_CTX
EOF

echo "==> Creating Ollama model: $OLLAMA_TAG"
ollama create "$OLLAMA_TAG" -f "$MODELFILE_PATH"

echo "==> Sanity check - sending one test prompt"
ollama run "$OLLAMA_TAG" "Responde en una frase: ¿qué eres?"

echo
echo "==> Model created: $OLLAMA_TAG"
echo "    Inspect the response above for template/tag leakage before running 10_swap_and_smoketest.sh."
