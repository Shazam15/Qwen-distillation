"""Phase 3 - Filter teacher completions before training on them.

Scores each (context, answer) pair with an NLI entailment check plus a citation-
format sanity check, per the plan's "rejection sampling / judged filtering" step.
The optional Claude Sonnet 5 judging pass from the plan is intentionally not
implemented here - wire it in as an extra automated_verdict branch if you want it,
since it costs API calls this script otherwise avoids entirely.

Output: two JSONL files - passed.jsonl (train on these) and rejected.jsonl (keep for
analysis). A high rejection rate on one dataset source usually means that source's
prompts need reformatting, not that the teacher is bad.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from common import NLIScorer, has_citation, is_refusal, read_jsonl


def automated_verdict(record: dict, nli: NLIScorer, threshold: float) -> tuple:
    answer = record["teacher_answer"]

    if is_refusal(answer):
        return True, "correct_refusal"

    if not has_citation(answer):
        return False, "no_citation_marker"

    premise = "\n\n".join(record["context_docs"])
    prob = nli.entailment_prob(premise, answer)
    if prob >= threshold:
        return True, f"entailed ({prob:.2f})"
    return False, f"not_entailed ({prob:.2f})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", default="../data/teacher_completions/completions.jsonl")
    parser.add_argument("--passed", default="../data/filtered/passed.jsonl")
    parser.add_argument("--rejected", default="../data/filtered/rejected.jsonl")
    parser.add_argument("--entailment-threshold", type=float, default=0.5)

    args = parser.parse_args()

    records = read_jsonl(args.input_path)
    print(f"Scoring {len(records)} completions..")
    nli = NLIScorer()

    passed, rejected = [], []

    for record in tqdm(records, desc="Filtering"):
        ok, reason = automated_verdict(record, nli, args.entailment_threshold)

        record["filter_verdict"] = "pass" if ok else "fail"
        record["filter_reason"] = reason
        (passed if ok else rejected).append(record)

    for path, items in [(args.passed, passed), (args.rejected, rejected)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(records) or 1
    print(f"\nPassed: {len(passed)} ({len(passed) / total:.1%})")
    print(f"Rejected: {len(rejected)} ({len(rejected) / total:.1%})")


if __name__ == "__main__":
    main()
