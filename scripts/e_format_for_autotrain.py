"""Format the filtered dataset for AutoTrain's chat-SFT trainer.

Builds a messages column per row and splits into train/eval - critically,
splitting by SOURCE DOCUMENT rather than by row, so no document's passages
appear in both splits. A row-level random splits would let the model "pass"
evail by having memorized the exact passage from a near-duplicate training
row, which overstates how well it generalizes.

Output schema:

{"messages": [
    {"role": "system", "content": "..."},
    {"role": "system", "content": "<rendered RAG_PROMPT_TEMPLATE prompt>"},
    {"role": "assistant", "content": "<thinking>...<thinking>\\n<answer>"},
]}

"""


import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import read_jsonl
from prompts_template import SYSTEM_PROMPT

def source_group_key(record: dict) -> str:
    """Groups by dataset source + question id prefix"""

    return f"{record['source']}::{record['id'].rsplit('_', 1)[0]}"

def build_messages(record: dict) -> dict:
    thinking = record.get("teacher_thinking", "")
    answer = record["teacher_answer"]
    assistant_content = f"<thinking>{thinking}</thinking>\n{answer}" if thinking else answer
    return{
        "messages": [
            {"role": "system","content": SYSTEM_PROMPT},
            {"role": "user", "content": record["prompt_rendered"]},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in", dest="input_path", default="../data/filtered/passed.jsonl")
    parser.add_argument("--train-out", default="../data/autotrain_ready/train.jsonl")
    parser.add_argument("--eval-out", default="../data/autotrain_ready/eval.jsonl")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    records = read_jsonl(args.input_path)

    groups: dict[str, list[dict]] = defaultdict(list)

    for r in records:
        groups[source_group_key(r)].append(r)

    group_keys = list(groups.keys())
    random.seed(args.seed)
    random.shuffle(group_keys)

    n_eval_groups = max(1, int(len(group_keys) * args.eval_fraction))
    eval_keys = set(group_keys[:n_eval_groups])

    train_records, eval_records = [], []
    for key, items in groups.items():
        target = eval_records if key in eval_keys else train_records
        target.extend(items)

    for path, items in [(args.train_out, train_records), (args.eval_out, eval_records)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(build_messages(r), ensure_ascii=False) + "\n")

    
    print(f"Groups: {len(group_keys)} total, {len(eval_keys)} held out for eval")
    print(f"Train rows: {len(train_records)} -> {args.train_out}")
    print(f"Eval rows: {len(eval_records)} -> {args.eval_out}")

    by_source: dict[str, int] = defaultdict(int)
    for r in train_records:
        by_source[r["source"]] += 1

    print["Train mix by source:", dict(by_source)]


if __name__ == "__main__":
    main()

