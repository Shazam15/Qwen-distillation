"""

Phase 1 - Build the prompt set

Combines (a) an export of real ATLAS traffic and (b) public grounded-QA datasheets, 
reformats bith into ATLAS's own RAG_PROMPT_TEMPLATE, and writes one JSONL file the
teacher will be run over in the next script.

Output schema (data/raw_prompts/prompts.jsonl), one record per line

{
    "id": "hotpotqa_0001",
    "question": "...",
    "context_docs":  ["D1: <passage text>", "D2: <passage text>"],
    "prompt_rendered": "<full RAG_PROMPT_TEMPLATE.format(...) string>",
    "source": "hotpotqa",
    "lang": "en"
}

"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

from prompts_template import RAG_PROMPT_TEMPLATE

def render_prompt(question: str, context_docs: list[str], style_reference: str = "") -> str:
    """Renders ATLAS's exact production prompt so the student trains on the task shape it will actually see - not a 
    generic Q&A format"""
    context = "\n\n".join(context_docs)
    return RAG_PROMPT_TEMPLATE.format(
        context=context, question=question, style_reference=style_reference
    )

def make_record(idx: int, source: str, question: str, passages: list[str], lang: str) -> dict:
    context_docs = [f"D{i+1}: {p.strip()}" for  i, p in enumerate(passages) if p and p.strip()]
    return{
        "id": f"{source}_{idx:05d}",
        "question": question,
        "context_docs": context_docs,
        "prompt_rendered": render_prompt(question, context_docs),
        "source": source,
        "lang": lang,
    }


#------------------------------------------
# Public dataset loaders
#------------------------------------------

def load_squad(limit: int) -> list[dict]:
    ds = load_dataset("squad", split="train").shuffle(seed=42).select(range(limit))
    return [
        make_record(i, "squad", row["question"], [row["context"]], lang="en")
        for i, row in enumerate(ds)
    ]


def load_hotpotqa(limit: int) -> list[dict]:
    ds = load_dataset("hotpot_qa", "distractor", split="train").shuffle(seed=42).select(range(limit))
    records = []

    for i, row in enumerate(ds):
        titles = row["context"]["title"]
        sentence_groups = row["context"]["sentence"]
        passages = [" ".join(sentences) for sentences in sentence_groups]
        records.append(make_record(i, "hotpotqa", row["question"], passages, lang="en"))
    
    return records


def load_public_dataset_stub(name: str, limit: int) -> list[dict]:
    """TODO before use: confirm the exact HF id and colunnames for `name` on huggingface.co/datasets"""

    raise NotImplementedError(
        f"load_public_dataset_stub: fill in the loader for '{name}' - see the TODO in this "
        "function's docstring."
    )


def load_own_traffic(path: str) -> list[dict]:
    """Loads a JSON export of real ATLAS queries"""

    p = Path(path)
    if not p.exists():
        print(f"[own_traffic] {path} not found - skipping (this source is optional)")
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    records = []

    for i, row in enumerate(raw):
        records.append({
            "id": f"own_traffic_{i:05d}",
            "question": row["question"],
            "context_docs": row["context_docs"],
            "prompt_rendered": render_prompt(row["question"], row["context_docs"]),
            "source": "own_traffic",
            "lang": row.get("lang", "es")
        })

    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../data/raw_prompts/prompts.jsonl")
    parser.add_argument("--per-dataset-limit", type=int, default=500)
    parser.add_argument("--own-traffic-path", default="../data/own_traffic_export.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_records: list[dict] = []

    for loader, label in [(load_squad, "squad"), (load_hotpotqa, "hotpotqa")]:
        records = loader(args.per_dataset_limit)
        print(f"[{label}] loaded {len(records)} prompts")
        all_records.extend(records)
    
    own = load_own_traffic(args.own_traffic_path)
    print(f"[own_traffic] loaded {len(own)} prompts")
    all_records.extend(own)

    random.seed(args.seed)
    random.shuffle(all_records)

    out_path = Path(args.out)
    out_path_parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_records)} prompts to {out_path}")
    by_source: dict[str, int] = {}
    for r in all_records:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print("Mix by source:", by_source)

if __name__ == "__main__":
    main()

