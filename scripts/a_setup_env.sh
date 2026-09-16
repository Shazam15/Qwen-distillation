#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TEACHER_MODEL="${TEACHER_MODEL:-hf.co/bartowski/Qwen3.8-Flash-Next-GGUF:Q4_K_M}"
STUDENT_MODEL="${STUDENT_MODEL:-qwen3:14b-q4_K_M}"

echo "==> Installing Python dependencies from requirements.txt"
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

echo "==> Checking Ollama is installed and reachable"
if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama not found on PATH. Install it from https://ollama.com first." >&2
  exit 1
fi

if ! ollama list >/dev/null 2>&1; then
  echo "ERROR: could not reach the Ollama daemon. Is it running?" >&2
  exit 1
fi

echo "==> Pulling teacher model: $TEACHER_MODEL"
ollama pull "$TEACHER_MODEL"

echo "==> Confirming student/baseline model is present: $STUDENT_MODEL"
if ! ollama list | grep -q "$STUDENT_MODEL"; then
  echo "    Not found locally - pulling it now."
  ollama pull "$STUDENT_MODEL"
fi

echo "==> Confirming the autotrain CLI installed correctly"
if ! command -v autotrain >/dev/null 2>&1; then
  echo "ERROR: 'autotrain' not found after pip install - check requirements.txt installed cleanly." >&2
  exit 1
fi
autotrain llm --help > /tmp/autotrain_llm_help.txt
echo "    Saved 'autotrain llm --help' to /tmp/autotrain_llm_help.txt"
echo "    Review it before running 05_train_qlora.sh - flag names drift between autotrain-advanced releases."

echo "==> Setup complete."
