#!/usr/bin/env bash
set -euo pipefail

# Points at the ATLAS (PDF-Assistant-RAG) checkout - a DIFFERENT repo from this one.
ATLAS_REPO_PATH="${ATLAS_REPO_PATH:?Set ATLAS_REPO_PATH to the ATLAS checkout path on this machine}"
ENV_FILE="${ENV_FILE:-$ATLAS_REPO_PATH/backend/.env}"
NEW_MODEL_TAG="${NEW_MODEL_TAG:-atlas-qwen3-14b-distilled}"
RESTART_CMD="${RESTART_CMD:-}"   # e.g. "systemctl restart atlas-backend" or "docker compose restart backend"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: env file not found at $ENV_FILE - set ENV_FILE explicitly." >&2
  exit 1
fi

echo "==> Backing up $ENV_FILE"
BACKUP_PATH="${ENV_FILE}.bak.$(date +%Y%m%d%H%M%S)"
cp "$ENV_FILE" "$BACKUP_PATH"
echo "    Backup: $BACKUP_PATH"

CURRENT_MODEL="$(grep '^LLM_MODEL=' "$ENV_FILE" | cut -d= -f2- || true)"
echo "==> Current LLM_MODEL: ${CURRENT_MODEL:-<not set>}"
echo "==> Setting LLM_MODEL=$NEW_MODEL_TAG"

if grep -q '^LLM_MODEL=' "$ENV_FILE"; then
  sed -i.tmp "s/^LLM_MODEL=.*/LLM_MODEL=$NEW_MODEL_TAG/" "$ENV_FILE"
  rm -f "${ENV_FILE}.tmp"
else
  echo "LLM_MODEL=$NEW_MODEL_TAG" >> "$ENV_FILE"
fi

if [ -n "$RESTART_CMD" ]; then
  echo "==> Restarting backend: $RESTART_CMD"
  eval "$RESTART_CMD"
else
  echo "==> No RESTART_CMD set - restart the ATLAS backend manually now."
  read -rp "Press Enter once it has restarted with the new model... " _
fi

echo
echo "==> Smoke test"
echo "Exercise POST /chat/ask (JWT-authenticated) with a handful of real queries - including"
echo "one against a selected document, to hit the citation-revision path in agent.py - and"
echo "compare the answers against what '$CURRENT_MODEL' produced before the swap."
echo
echo "Example (fill in a real bearer token from a logged-in session):"
echo '  curl -s -X POST "http://localhost:7860/chat/ask" \'
echo '    -H "Authorization: Bearer <your JWT>" \'
echo '    -H "Content-Type: application/json" \'
echo "    -d '{\"question\": \"Resume el documento en una frase.\"}' | jq ."
echo
echo "If anything regresses, roll back with:"
echo "  cp $BACKUP_PATH $ENV_FILE   # then restart the backend again"
