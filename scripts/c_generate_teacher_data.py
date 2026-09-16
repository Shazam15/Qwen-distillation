"""Generate teacher completions


Runs every prompt from a_build_prompts.py through the teacher model and records
both its reasoning trace and final answer. This is the slowest step inthe pipeline (hours) - it's
built to be interrupted and resumed freely

Usage:
    python c_generate_teacher_data.py \\
        --prompt ../data/raw_prompts/prompts.jsonl \\
        --out ../data/teacher_completions/completions.jsonl\\
        --model qwen3.8-flash-next \\
        --workers 3

"""


import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from common import append_jsonl, call_ollama_chat, load_done_ids, read_jsonl, split_thinking
from prompts_template import SYSTEM_PROMPT

def process_one(record: dict, model: str, num_ctx: int, num_predict: int) -> dict:
    
    raw_response = call_ollama_chat(
        model=model, 
        system=SYSTEM_PROMPT,
        user=record["prompt_rendered"],
        num_ctx=num_ctx,
        num_predict=num_predict,
    )

    thinking, answer = split_thinking(raw_response)
    return {
        "id": record["id"],
        "question": record["question"],
        "context_docs": record["context_docs"],
        "prompt_rendered": record["prompt_rendered"],
        "source": record["source"],
        "lang": record["lang"],
        "teacher_thinking": thinking,
        "teacher_answer": answer
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="../data/raw_prompts/prompts.jsonl")
    parser.add_argument("--out", default="../data/teacher_completions/completions.jsonl")
    parser.add_argument(
        "--model",
        default="hf.co/bartowski/Qwen3.8-Flash-Next-GGUF:Q4_K_M",
        help="Ollama tag for the teacher - must match TEACHER_MODEL pulled by a_setup_env.sh",
    )
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=3, help=(
        "Concurrent requests to Ollama. Keep this modest (2-4) - the teacher is CPU bound "
        "on this machine, so more workers than the model can actually parallelize just "
        "queues requests without anything up. "
    ))

    args = parser.parse_args()

    prompts = read_jsonl(args.prompts)
    done_ids = load_done_ids(args.out)
    pending = [p for p in prompts if p["id"] not in done_ids]
    print(f"{len(prompts)} total prompts, {len(done_ids)} already done, {len(pending)} pending")
    
    if not pending:
        print("Nothing to do.")
        return
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, rec, args.model, args.num_ctx, args.num_predict): rec
            for rec in pending
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            record = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"\n[ERROR] {record['id']}: {exc}")
                continue
            append_jsonl(args.out, result)


if __name__ == "__main__":
    main()
