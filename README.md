# What is this for?

This repo distills Qwen3.8-Flash-Next (teacher) into Qwen3-14B (student), to
supercharge the smaller Qwen3 model served locally by the academic RAG agent ATLAS
(PDF-Assistant-RAG) - via Ollama, with no code changes to ATLAS and no recurring API
costs.

**Scope**: this repo is separate from ATLAS. It produces a `.gguf` file and an Ollama
model tag; `k_swap_and_smoketest.sh` is the only script that touches an ATLAS checkout,
and only to flip one `.env` line.

# What's distillation?

Distillation takes a large "teacher" LLM and uses its outputs to train a smaller
"student" to approximate its behavior. Concretely, here that means **response-based
SFT**: the teacher generates full completions (reasoning trace + answer) for a set of
prompts, and the student is fine-tuned with cross-entropy loss against that text - not
logit-level (Hinton-style) distillation, which needs white-box teacher logits and a
matching tokenizer that AutoTrain doesn't provide. See the plan, Section 4, for why.

# Hardware & constraints

| Component | Spec | Role in this plan |
|---|---|---|
| GPU | Tesla T4, 16GB VRAM (Turing, **fp16 only**, no bf16) | QLoRA fine-tune of the student (4-bit fits comfortably in 16GB) |
| CPU | Intel Xeon Gold 6258R | Runs the teacher model for offline batch generation |
| RAM | ~400GB | Holds the quantized teacher (~65-75GB in Q4) plus generation overhead |
| Serving layer | Ollama (unchanged) | Every LLM call site in ATLAS reads `settings.LLM_MODEL` as a plain string passed to `ChatOllama` - the distilled model is a drop-in tag swap, no ATLAS code change needed |

**This machine is Windows.** Run the whole pipeline inside WSL2, not native Windows
Python - see [`docs/WINDOWS_SETUP.md`](docs/WINDOWS_SETUP.md) for why and exactly how
to set it up (WSL2 install, GPU passthrough, `.wslconfig` RAM limits, disk budget).
Start with `python scripts/00_preflight_check.py` to confirm the machine is actually
ready (GPU visible, right compute capability, enough RAM/disk, Ollama reachable)
before running anything else.

# Pipeline (Phases 0-11)

Each phase from the plan maps to one script, in alphabetical/phase order:

| Phase | Script | What it does |
|---|---|---|
| 0 | `00_preflight_check.py` | Verify GPU/CUDA, RAM, disk, Ollama, native-Windows-vs-WSL2 |
| 0 | `a_setup_env.sh` | Install Python deps, pull the teacher + baseline student via Ollama |
| 1 | `b_build_prompts.py` | Build the prompt set from public datasets + your own ATLAS traffic, rendered through `RAG_PROMPT_TEMPLATE` |
| 2 | `c_generate_teacher_data.py` | Run every prompt through the teacher, capturing reasoning trace + answer (resumable; the slowest phase, ~1-3 days) |
| 3 | `d_filter_dataset.py` | NLI-entailment + citation-marker filtering; reject unsupported/uncited answers before training on them |
| 4 | `e_format_for_autotrain.py` | Build chat-formatted JSONL, split train/eval by source document (not by row) |
| 5 | `f_train_qlora.sh` | 4-bit QLoRA SFT of Qwen3-14B via `autotrain llm` (~3-8h) |
| 6 | `g_evaluate.py` | Gate: baseline vs. distilled on the held-out split - only proceed if distilled measurably wins |
| 7 | `h_merge_lora.py` | Merge the LoRA adapter into the base weights (runs in fp16 on CPU RAM) |
| 8 | `i_convert_quantize.sh` | Convert to GGUF, quantize to Q4_K_M via llama.cpp |
| 9 | `j_create_ollama_model.sh` | Package the GGUF into an Ollama model via a `Modelfile` |
| 10 | `k_swap_and_smoketest.sh` | Flip `LLM_MODEL` in the ATLAS `.env` to the new tag, with backup + rollback instructions |
| 11 (optional) | `l_build_dpo_dataset.py` + `l_dpo_train.sh` | Build chosen/rejected pairs (teacher vs. pre-distillation baseline) and run a DPO refinement pass on top of the merged SFT checkpoint |

Run them in order. `b`/`c`/`l_build_dpo_dataset.py` are resumable (interrupt and rerun
freely - they skip ids already written to their output file).

`scripts/common.py` and `scripts/prompts_template.py` are shared modules the other
scripts import from - not standalone steps. **`prompts_template.py` is a copy of
ATLAS's production `RAG_PROMPT_TEMPLATE`/`SYSTEM_PROMPT`** (from
`backend/app/rag/prompts.py` in the ATLAS checkout); diff it against the real thing
before running Phase 1 if ATLAS's prompt has changed since this was written.

# Setup

```bash
cp .env.example .env
# edit .env: at minimum set ATLAS_REPO_PATH once you reach Phase 10
set -a && source .env && set +a

python scripts/00_preflight_check.py
bash scripts/a_setup_env.sh
```

See [`docs/WINDOWS_SETUP.md`](docs/WINDOWS_SETUP.md) first if this is the Windows/T4
machine - it covers WSL2 install, GPU passthrough, and RAM/disk configuration that
`a_setup_env.sh` assumes are already in place.

# Training data sources (Phase 1)

`b_build_prompts.py --datasets ...` selects which public sources to pull, on top of
your own ATLAS traffic export (`--own-traffic-path`, optional). Per the plan's Section
5:

| Layer | Datasets wired up | Purpose |
|---|---|---|
| General reasoning | OpenR1-Math-220k, Nemotron (`nemotron`, off by default - 33M-row source, streamed), Bespoke-Stratos-17k, s1K-1.1 | Boosts reasoning ability, independent of the RAG task |
| Grounded / citation QA | SQuAD, HotpotQA, ASQA, MultiHop-RAG, QASPER | Faithfulness to retrieved evidence, multi-hop synthesis, citation discipline |
| Spanish coverage | SQAC, MLQA (es/es), XQuAD (es) | Keeps citation discipline and quality from degrading in Spanish |

RAGTruth, CRAG, and OpenThoughts3-1.2M are listed in `LOADERS` as stubs that raise
`NotImplementedError` - their exact HF schemas weren't confirmed when this was written.
Confirm the columns on the dataset's HF page and fill in the loader before enabling
them by name.

# Notes

- **No logit-matching KD.** AutoTrain does supervised fine-tuning on text, not
  KL-divergence over logits - see the plan, Section 4, for why that's the right
  tradeoff here.
- **Gate before merging.** Don't skip `g_evaluate.py` - Phase 6 exists specifically so
  you don't ship a distilled model that's actually worse than the current
  `qwen3:14b-q4_K_M` baseline.
- **Rollback is one file.** `k_swap_and_smoketest.sh` backs up ATLAS's `.env` before
  editing it and prints the exact command to restore it.
