"""Phase 11 (optional) - build the chosen/rejected pairs for l_dpo_train.sh.

chosen   = the teacher completion already used for SFT (from data/autotrain_ready/train.jsonl)
rejected = the PRE-DISTILLATION baseline model's own completion on that same prompt,
           generated fresh here by calling the baseline through Ollama.

Only run this after e_format_for_autotrain.py, and only if g_evaluate.py's gate
passed and you want the optional DPO refinement pass. Resumable like
c_generate_teacher_data.py: rows already written to --out are skipped on rerun.

Row ids here are positional (str(i) into train.jsonl), since that file has no
per-row id column - don't regenerate train.jsonl between a partial run and its resume,
or the ids will no longer line up with the same rows.
"""

import argparse

from tqdm import tqdm

from common import append_jsonl, call_ollama_chat, load_done_ids, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="../data/autotrain_ready/train.jsonl")
    parser.add_argument(
        "--out",
        default="../data/dpo_ready/train.jsonl",
        help="filename matters - autotrain's --data-path expects a train.jsonl inside the directory",
    )
    parser.add_argument(
        "--baseline-model",
        default="qwen3:14b-q4_K_M",
        help="the PRE-distillation model - must be the same tag g_evaluate.py compared against",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap rows for a quick DPO pass")
    args = parser.parse_args()

    rows = read_jsonl(args.train)
    if args.limit:
        rows = rows[: args.limit]

    done_ids = load_done_ids(args.out)

    for i, row in enumerate(tqdm(rows, desc="Building DPO pairs")):
        row_id = str(i)
        if row_id in done_ids:
            continue

        system_msg = row["messages"][0]["content"]
        user_msg = row["messages"][1]["content"]
        chosen = row["messages"][2]["content"]

        try:
            rejected = call_ollama_chat(args.baseline_model, system_msg, user_msg)
        except Exception as exc:
            print(f"\n[ERROR] row {row_id}: {exc}")
            continue

        append_jsonl(args.out, {
            "id": row_id,
            "prompt": user_msg,
            "chosen": chosen,
            "rejected": rejected,
        })

    print(f"Wrote DPO pairs to {args.out}")
    print("Check autotrain's expected DPO column names ('autotrain llm --help') before")
    print("running l_dpo_train.sh - this writes prompt/chosen/rejected, which may need")
    print("renaming depending on the installed autotrain-advanced version.")


if __name__ == "__main__":
    main()
