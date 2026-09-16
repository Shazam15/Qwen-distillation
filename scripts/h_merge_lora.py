

import argparse

import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--adapter-path", default="../models/lora_adapter")
    parser.add_argument("--out", default="../models/merged")
    args = parser.parse_args()

    print(f"Loading base model {args.base_model} in fp16 on CPU (uses system RAM, not VRAM)")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("Merging adapter into base weights")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.out}")
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("Done. Next: i_convert_quantize.sh to produce the GGUF file")

if __name__ == "__main__":
    main()