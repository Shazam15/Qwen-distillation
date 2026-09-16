"""Phase 6 - Evaluate the LoRA checkpoint before trusting it.

Compares two models on the held-out eval split:

- "baseline": the current production model, called through Ollama (qwen3:14b)
- "distilled": the freshly trained LoRA, loaded directly on top of the base HF model in 4-bit

Gate: only proceed to h_merge_lora.py if the distilled model measurably beats the
baseline on the held-out split, per the plan's Phase 6.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import NLIScorer, call_ollama_chat, has_citation, is_refusal, read_jsonl


def load_distilled_model(base_model: str, adapter_path: str):
    """Loads the base model in 4-bit with the LoRA adapter applied, for generation
    only (no gradient updates here). bnb_4bit_compute_dtype is float16, not bfloat16 -
    the Tesla T4 (Turing) has no native bf16 support."""

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def generate_distilled(model, tokenizer, system: str, user: str, max_new_tokens: int = 1024) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False
        )
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def score_answer(nli: NLIScorer, context_docs: list, answer: str, threshold: float) -> bool:
    if is_refusal(answer):
        return True
    if not has_citation(answer):
        return False
    premise = "\n\n".join(context_docs)
    return nli.entailment_prob(premise, answer) >= threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="../data/autotrain_ready/eval.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--adapter-path", default="../models/lora_adapter")
    parser.add_argument("--baseline-ollama-model", default="qwen3:14b-q4_K_M")
    parser.add_argument("--entailment-threshold", type=float, default=0.5)
    parser.add_argument("--report-out", default="../data/eval_report.json")
    parser.add_argument("--sample-dump", default="../data/eval_samples.txt", help=(
        "where a handful of side-by-side answers get written for a manual skim - numbers "
        "alone can hide a model that games the automated check"
    ))

    args = parser.parse_args()

    eval_rows = read_jsonl(args.eval)
    nli = NLIScorer()
    print(f"Loading distilled model ({args.base_model} + {args.adapter_path})...")
    model, tokenizer = load_distilled_model(args.base_model, args.adapter_path)

    baseline_pass, distilled_pass = 0, 0
    samples = []

    for row in eval_rows:
        system_msg = row["messages"][0]["content"]
        user_msg = row["messages"][1]["content"]

        context_block = user_msg.split("## Contexto del Documento")[1].split(
            "## Solicitud del Investigador"
        )[0]
        context_docs = [line for line in context_block.strip().splitlines() if line.strip()]

        baseline_answer = call_ollama_chat(args.baseline_ollama_model, system_msg, user_msg)
        distilled_answer = generate_distilled(model, tokenizer, system_msg, user_msg)

        b_ok = score_answer(nli, context_docs, baseline_answer, args.entailment_threshold)
        d_ok = score_answer(nli, context_docs, distilled_answer, args.entailment_threshold)
        baseline_pass += b_ok
        distilled_pass += d_ok

        if len(samples) < 30:
            samples.append(
                f"--- {user_msg[:80]}...\n"
                f"[baseline pass={b_ok}] {baseline_answer[:300]}\n"
                f"[distilled pass={d_ok}] {distilled_answer[:300]}\n"
            )

    total = len(eval_rows) or 1
    report = {
        "n_eval": total,
        "baseline_pass_rate": baseline_pass / total,
        "distilled_pass_rate": distilled_pass / total,
        "delta": (distilled_pass - baseline_pass) / total,
    }

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.sample_dump).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(args.sample_dump, "w", encoding="utf-8") as f:
        f.write("\n".join(samples))

    print(json.dumps(report, indent=2))
    if report["delta"] <= 0:
        print(
            "\nGATE FAILED: distilled model did not beat baseline. Do not proceed to "
            "h_merge_lora.py yet - revisit data filtering (d) or training epochs (f)."
        )
    else:
        print(
            f"\nGATE PASSED: distilled model beat baseline by {report['delta']:.1%}. "
            f"Read {args.sample_dump} before proceeding to h_merge_lora.py."
        )


if __name__ == "__main__":
    main()
