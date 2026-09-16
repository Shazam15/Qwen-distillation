"""
Phase 1 - Build the prompt set

Combines (a) an export of real ATLAS traffic and (b) public datasets from the
plan's Section 5 (general reasoning, grounded/citation QA, Spanish coverage),
reformats them into ATLAS's own RAG_PROMPT_TEMPLATE, and writes one JSONL file
the teacher will be run over in the next script.

Output schema (data/raw_prompts/prompts.jsonl), one record per line:

{
    "id": "hotpotqa_00001",
    "question": "...",
    "context_docs": ["D1: <passage text>", "D2: <passage text>"],
    "prompt_rendered": "<full RAG_PROMPT_TEMPLATE.format(...) string>",
    "source": "hotpotqa",
    "lang": "en"
}

General-reasoning sources (openr1_math, nemotron, bespoke_stratos, s1k) have no
retrieval context - they train the underlying reasoning ability, not the RAG task
itself (see the plan, Section 5). For those, context_docs is empty and
prompt_rendered is just the raw question: forcing them through RAG_PROMPT_TEMPLATE
with an empty context section would teach the model to associate "no documents"
with a normal question, which is the opposite of what Phase 3's filtering wants.

Any single loader failing (renamed dataset, missing HF_TOKEN for a gated one, no
network) prints a warning and is skipped rather than aborting the whole build -
useful since this script combines a dozen+ independent remote sources.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from prompts_template import RAG_PROMPT_TEMPLATE

def render_prompt(question: str, context_docs: list, style_reference: str = "") -> str:
    """Renders ATLAS's exact production prompt so the student trains on the task shape
    it will actually see - not a generic Q&A format."""
    context = "\n\n".join(context_docs)
    return RAG_PROMPT_TEMPLATE.format(
        context=context, question=question, style_reference=style_reference
    )


def make_record(idx: int, source: str, question: str, passages: list, lang: str, doc_id: str = None) -> dict:
    """doc_id groups records that share an underlying source document/passage, so
    e_format_for_autotrain.py's train/eval split can keep a whole document on one
    side of the split. Defaults to this record's own id (i.e. "no sharing") - pass
    an explicit doc_id when one loader emits multiple rows over the same document
    (e.g. load_qasper's several questions per paper)."""
    record_id = f"{source}_{idx:05d}"
    context_docs = [f"D{i + 1}: {p.strip()}" for i, p in enumerate(passages) if p and p.strip()]
    return {
        "id": record_id,
        "doc_id": doc_id or record_id,
        "question": question,
        "context_docs": context_docs,
        "prompt_rendered": render_prompt(question, context_docs),
        "source": source,
        "lang": lang,
    }


def make_reasoning_record(idx: int, source: str, question: str, lang: str, doc_id: str = None) -> dict:
    """Like make_record, but for context-free general-reasoning sources (see module
    docstring) - prompt_rendered is the bare question, not the RAG template."""
    record_id = f"{source}_{idx:05d}"
    return {
        "id": record_id,
        "doc_id": doc_id or record_id,
        "question": question,
        "context_docs": [],
        "prompt_rendered": question,
        "source": source,
        "lang": lang,
    }


# ------------------------------------------------------------------
# Grounded / citation QA loaders (plan Section 5, "Grounded / citation QA")
# ------------------------------------------------------------------

def load_squad(limit: int) -> list:
    ds = load_dataset("squad", split="train").shuffle(seed=42)
    ds = ds.select(range(min(limit, len(ds))))
    return [
        make_record(i, "squad", row["question"], [row["context"]], lang="en")
        for i, row in enumerate(ds)
    ]


def load_hotpotqa(limit: int) -> list:
    ds = load_dataset("hotpot_qa", "distractor", split="train").shuffle(seed=42)
    ds = ds.select(range(min(limit, len(ds))))
    records = []

    for i, row in enumerate(ds):
        sentence_groups = row["context"]["sentences"]
        passages = [" ".join(sentences) for sentences in sentence_groups]
        records.append(make_record(i, "hotpotqa", row["question"], passages, lang="en"))

    return records


def load_asqa(limit: int) -> list:
    """din0s/asqa - long-form QA over multiple Wikipedia passages.

    Passages live under qa_pairs[i]["context"]; the gold long-form answer under
    annotations[0]["long_answer"] is not used here (Phase 2 asks the teacher to
    generate its own answer over these passages, it doesn't copy ASQA's).
    """
    ds = load_dataset("din0s/asqa", split="train").shuffle(seed=42)
    ds = ds.select(range(min(limit, len(ds))))
    records = []

    for i, row in enumerate(ds):
        passages = [qa.get("context", "") for qa in row.get("qa_pairs", []) if qa.get("context")]
        if not passages:
            continue
        records.append(make_record(i, "asqa", row["ambiguous_question"], passages, lang="en"))

    return records


def load_multihop_rag(limit: int) -> list:
    """yixuantt/MultiHopRAG - multi-hop QA with evidence spread across 2-4 documents."""
    ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    ds = ds.shuffle(seed=42).select(range(min(limit, len(ds))))
    records = []

    for i, row in enumerate(ds):
        passages = [ev.get("fact", "") for ev in row.get("evidence_list", []) if ev.get("fact")]
        if not passages:
            continue
        records.append(make_record(i, "multihop_rag", row["query"], passages, lang="en"))

    return records


def load_qasper(limit: int) -> list:
    """allenai/qasper - questions over full scientific papers.

    Only the paragraphs from the paper's sections are used as context (not the
    figures/tables); one prompt is emitted per (paper, question) pair, capped at
    `limit` total.
    """
    ds = load_dataset("allenai/qasper", split="train", trust_remote_code=True)
    records = []

    for paper_idx, row in enumerate(ds):
        if len(records) >= limit:
            break
        paragraphs = [p for group in row["full_text"]["paragraphs"] for p in group if p.strip()]
        if not paragraphs:
            continue
        doc_id = f"qasper_paper_{paper_idx:05d}"
        for question in row["qas"]["question"]:
            if len(records) >= limit:
                break
            records.append(
                make_record(len(records), "qasper", question, paragraphs[:10], lang="en", doc_id=doc_id)
            )

    return records


def load_ragtruth(limit: int) -> list:
    """TODO before enabling: wandb/RAGTruth-processed's exact column names were not
    verified when this loader was written - confirm source/response field names on
    huggingface.co/datasets/wandb/RAGTruth-processed before use."""
    raise NotImplementedError(
        "load_ragtruth: confirm wandb/RAGTruth-processed's schema (source passages / "
        "response columns) on the HF dataset page, then fill this in."
    )


def load_crag(limit: int) -> list:
    """TODO before enabling: Quivr/CRAG's exact column names were not verified when
    this loader was written - confirm on huggingface.co/datasets/Quivr/CRAG."""
    raise NotImplementedError(
        "load_crag: confirm Quivr/CRAG's schema (query / search-result columns) on "
        "the HF dataset page, then fill this in."
    )


# ------------------------------------------------------------------
# Spanish coverage loaders (plan Section 5, "Spanish coverage") - all three are
# SQuAD-format (context, question, answers) so they reuse load_squad's shape.
# ------------------------------------------------------------------

def load_sqac(limit: int) -> list:
    ds = load_dataset("PlanTL-GOB-ES/SQAC", split="train").shuffle(seed=42)
    ds = ds.select(range(min(limit, len(ds))))
    return [
        make_record(i, "sqac", row["question"], [row["context"]], lang="es")
        for i, row in enumerate(ds)
    ]


def load_mlqa_es(limit: int) -> list:
    """facebook/mlqa, es/es config - MLQA is an eval benchmark (no train split)."""
    ds = load_dataset("facebook/mlqa", "mlqa.es.es", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(min(limit, len(ds))))
    return [
        make_record(i, "mlqa_es", row["question"], [row["context"]], lang="es")
        for i, row in enumerate(ds)
    ]


def load_xquad_es(limit: int) -> list:
    """google/xquad, xquad.es config - XQuAD is an eval benchmark (validation only)."""
    ds = load_dataset("google/xquad", "xquad.es", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(limit, len(ds))))
    return [
        make_record(i, "xquad_es", row["question"], [row["context"]], lang="es")
        for i, row in enumerate(ds)
    ]


# ------------------------------------------------------------------
# General reasoning loaders (plan Section 5, "General reasoning") - no retrieval
# context; see make_reasoning_record and the module docstring.
# ------------------------------------------------------------------

def load_openr1_math(limit: int) -> list:
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    ds = ds.shuffle(seed=42).select(range(min(limit, len(ds))))
    return [make_reasoning_record(i, "openr1_math", row["problem"], lang="en") for i, row in enumerate(ds)]


def load_nemotron(limit: int) -> list:
    """nvidia/Llama-Nemotron-Post-Training-Dataset - 33M+ rows, so this streams
    instead of downloading + shuffling the whole thing. `input` is a chat-formatted
    list of turns; the last user turn's content is used as the question."""
    ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split="code", streaming=True)
    records = []

    for row in ds:
        if len(records) >= limit:
            break
        turns = row.get("input") or []
        user_turns = [t["content"] for t in turns if isinstance(t, dict) and t.get("role") == "user"]
        if not user_turns:
            continue
        records.append(make_reasoning_record(len(records), "nemotron", user_turns[-1], lang="en"))

    return records


def load_bespoke_stratos(limit: int) -> list:
    ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
    ds = ds.shuffle(seed=42).select(range(min(limit, len(ds))))
    records = []

    for i, row in enumerate(ds):
        turns = row.get("conversations", [])
        user_turns = [t["value"] for t in turns if t.get("from") == "user"]
        if not user_turns:
            continue
        records.append(make_reasoning_record(i, "bespoke_stratos", user_turns[0], lang="en"))

    return records


def load_s1k(limit: int) -> list:
    ds = load_dataset("simplescaling/s1K-1.1", split="train")
    ds = ds.shuffle(seed=42).select(range(min(limit, len(ds))))
    return [make_reasoning_record(i, "s1k", row["question"], lang="en") for i, row in enumerate(ds)]


def load_openthoughts3(limit: int) -> list:
    """TODO before enabling: open-thoughts/OpenThoughts3-1.2M's exact column names
    were not verified when this loader was written - confirm on
    huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M (likely a ShareGPT-style
    'conversations' column similar to Bespoke-Stratos, but check before trusting it
    for a 1.2M-row source)."""
    raise NotImplementedError(
        "load_openthoughts3: confirm open-thoughts/OpenThoughts3-1.2M's schema on "
        "the HF dataset page, then fill this in."
    )


def load_own_traffic(path: str) -> list:
    """Loads a JSON export of real ATLAS queries: a list of
    {"question": ..., "context_docs": [...], "lang": "es", "document_id": "..."}
    objects. "document_id" is optional - pass it through as doc_id when present so
    several questions against the same uploaded PDF stay on one side of the
    train/eval split; each row is its own group otherwise."""
    p = Path(path)
    if not p.exists():
        print(f"[own_traffic] {path} not found - skipping (this source is optional)")
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    records = []

    for i, row in enumerate(raw):
        record_id = f"own_traffic_{i:05d}"
        doc_id = f"own_traffic_doc_{row['document_id']}" if row.get("document_id") else record_id
        records.append({
            "id": record_id,
            "doc_id": doc_id,
            "question": row["question"],
            "context_docs": row["context_docs"],
            "prompt_rendered": render_prompt(row["question"], row["context_docs"]),
            "source": "own_traffic",
            "lang": row.get("lang", "es"),
        })

    return records


# Registry of public loaders. "default" is what runs with no --datasets flag: a
# broad, schema-confirmed mix biased toward the plan's few-thousand-prompt target.
# The three NotImplementedError stubs above are intentionally excluded from
# "default" and "all" until their schemas are confirmed - pass them by name
# explicitly once you've filled in the loader.
LOADERS = {
    "squad": load_squad,
    "hotpotqa": load_hotpotqa,
    "asqa": load_asqa,
    "multihop_rag": load_multihop_rag,
    "qasper": load_qasper,
    "sqac": load_sqac,
    "mlqa_es": load_mlqa_es,
    "xquad_es": load_xquad_es,
    "openr1_math": load_openr1_math,
    "nemotron": load_nemotron,
    "bespoke_stratos": load_bespoke_stratos,
    "s1k": load_s1k,
    "ragtruth": load_ragtruth,
    "crag": load_crag,
    "openthoughts3": load_openthoughts3,
}

DEFAULT_DATASETS = [
    "squad", "hotpotqa", "asqa", "multihop_rag", "qasper",
    "sqac", "mlqa_es", "xquad_es",
    "openr1_math", "bespoke_stratos", "s1k",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../data/raw_prompts/prompts.jsonl")
    parser.add_argument("--per-dataset-limit", type=int, default=500)
    parser.add_argument("--own-traffic-path", default="../data/own_traffic_export.json")
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help=f"comma-separated subset of: {','.join(LOADERS)} (or 'all')",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    requested = list(LOADERS) if args.datasets == "all" else [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_records: list = []

    for name in requested:
        loader = LOADERS.get(name)
        if loader is None:
            print(f"[{name}] unknown dataset - skipping. Known: {list(LOADERS)}")
            continue
        try:
            records = loader(args.per_dataset_limit)
        except Exception as exc:
            print(f"[{name}] FAILED to load ({exc}) - skipping this source")
            continue
        print(f"[{name}] loaded {len(records)} prompts")
        all_records.extend(records)

    own = load_own_traffic(args.own_traffic_path)
    print(f"[own_traffic] loaded {len(own)} prompts")
    all_records.extend(own)

    random.seed(args.seed)
    random.shuffle(all_records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_records)} prompts to {out_path}")
    by_source: dict = {}
    for r in all_records:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print("Mix by source:", by_source)


if __name__ == "__main__":
    main()
