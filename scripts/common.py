"""Shared helpers for the distillation pipeline scripts (b_ through l_).

Kept dependency-light: only NLIScorer touches torch/transformers, and it imports
them lazily so scripts that don't need a live model (b_, c_) don't require a GPU
stack just to import this module.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

import httpx

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_ids(path) -> set:
    """Reads an in-progress output file and returns the ids already written,
    so a batch job (c_generate_teacher_data.py, l_build_dpo_dataset.py) can be
    interrupted and resumed without redoing finished work."""
    return {r["id"] for r in read_jsonl(path)}


def call_ollama_chat(
    model: str,
    system: str,
    user: str,
    num_ctx: int = 8192,
    num_predict: int = 2048,
    base_url: Optional[str] = None,
    timeout: float = 600.0,
) -> str:
    """Calls Ollama's /api/chat (non-streaming) and returns the assistant's text."""
    url = f"{(base_url or OLLAMA_BASE_URL).rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict},
    }
    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["message"]["content"]


# Reasoning-trace delimiters seen across the teacher model's own output and the
# public CoT datasets pulled in by b_build_prompts.py - they don't all agree on
# one tag style, so split_thinking tries each in turn.
_THINK_TAG = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL)
_BESPOKE_TAGS = re.compile(
    r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>\s*"
    r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>",
    re.DOTALL,
)


def split_thinking(raw_response: str) -> tuple:
    """Splits a raw completion into (thinking, answer).

    Returns ("", raw_response) when no known reasoning-trace delimiter is found -
    not every teacher call reasons out loud, and some datasets ship answer-only text.
    """
    raw_response = raw_response.strip()

    match = _BESPOKE_TAGS.search(raw_response)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    match = _THINK_TAG.match(raw_response)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    return "", raw_response


_REFUSAL_MARKERS = [
    "no tengo informaci",
    "no se menciona",
    "no encuentro informaci",
    "no cuento con informaci",
    "no hay informaci",
    "i don't have enough information",
    "i cannot find",
    "the provided context does not",
    "el contexto proporcionado no",
    "no context provided",
]


def is_refusal(answer: str) -> bool:
    """Heuristic: did the model correctly decline to answer for lack of evidence?

    Tune _REFUSAL_MARKERS against your own data/filtered/rejected.jsonl - exact
    phrasing drifts by teacher model and by language (EN/ES).
    """
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


# Matches the D1/D2/... markers b_build_prompts.py assigns to context_docs (see
# make_record). Update this pattern if RAG_PROMPT_TEMPLATE's citation style differs
# from what's copied into prompts_template.py.
_CITATION_PATTERN = re.compile(r"\[?D\d+\]?", re.IGNORECASE)


def has_citation(answer: str) -> bool:
    return bool(_CITATION_PATTERN.search(answer))


class NLIScorer:
    """Multilingual (EN/ES) NLI entailment scorer.

    Used by d_filter_dataset.py (Phase 3) and g_evaluate.py (Phase 6) to check
    whether an answer is actually supported by ("entailed by") its cited context,
    rather than trusting the teacher's or student's citations at face value.
    """

    MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

    def __init__(self, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID)
        self.model.to(self.device)
        self.model.eval()
        self.entailment_index = self.model.config.label2id.get("entailment", 0)

    def entailment_prob(self, premise: str, hypothesis: str, max_length: int = 1024) -> float:
        import torch

        inputs = self.tokenizer(
            premise, hypothesis, truncation=True, max_length=max_length, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        return probs[self.entailment_index].item()
