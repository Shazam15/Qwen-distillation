# Running this pipeline on Windows (Xeon Gold 6258R / 400GB RAM / Tesla T4)

## The short version

**Run everything inside WSL2 (Ubuntu), not native Windows Python or PowerShell.**

Every script in this repo (`.sh` files, `nproc`, `cmake`, heredocs, `sed -i`) already
assumes a POSIX shell. That's not an oversight to route around - it's the right call,
because the training stack itself doesn't work well on native Windows:

- `autotrain-advanced` depends on `deepspeed` and `triton`. Both have a long history of
  broken or unofficial Windows builds (missing `lscpu`, no reliable wheels, partial
  Triton support). Getting them to work natively is a time sink with no payoff, since
  WSL2 gives you a real Linux userspace with full CUDA passthrough to the T4 anyway.
- `bitsandbytes` *does* have official Windows wheels since v0.43.0, so 4-bit QLoRA
  itself isn't the blocker - `autotrain-advanced`'s other dependencies are.
- `llama.cpp`'s build step uses `cmake` + `nproc`; trivial in WSL2, awkward on native
  Windows without MSVC + a separate parallelism flag.

Ollama has a native Windows installer, but for this pipeline, run it inside WSL2 too
(Ollama supports Linux + NVIDIA GPU passthrough under WSL2 natively) so every phase -
Ollama, autotrain, llama.cpp, the HF/torch stack - shares one filesystem and one CUDA
context. Mixing native-Windows Ollama with WSL2-side Python scripts just adds a network
hop and two different model-cache locations for no benefit.

## 1. Install / verify WSL2

From an elevated PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
wsl --update
```

If WSL2 is already installed, confirm the distro is actually WSL **2** (not 1):

```powershell
wsl -l -v
```

## 2. NVIDIA driver (Windows side only)

Install the latest **Windows** NVIDIA driver (Data Center / Tesla driver branch, since
this is a T4) from NVIDIA directly - not a driver inside WSL2. WSL2 uses the Windows
host driver through a passthrough layer; do **not** install a separate Linux NVIDIA
driver inside the WSL2 distro, and do not install `nvidia-cuda-toolkit`'s driver
package there either - only the CUDA toolkit/libraries, if anything beyond what pip
pulls in.

Verify from inside WSL2:

```bash
nvidia-smi
```

You should see the Tesla T4 listed. If this fails, the fix is almost always a stale
Windows-side driver or a WSL2 version that predates GPU passthrough (`wsl --update`).

## 3. Give WSL2 enough RAM

By default, WSL2 caps itself at roughly half the host's RAM. On a 400GB box, that
default (~200GB) is close to what the plan needs just for the quantized teacher
(~65-75GB) plus generation overhead - too tight. Raise the cap explicitly.

Create/edit `%UserProfile%\.wslconfig` on the **Windows** side (e.g.
`C:\Users\<you>\.wslconfig`):

```ini
[wsl2]
memory=350GB
processors=<leave unset to use all cores, or cap explicitly>
swap=32GB
```

Then from PowerShell:

```powershell
wsl --shutdown
```

and reopen your WSL2 terminal. Confirm inside WSL2:

```bash
free -h
```

## 4. Base packages inside WSL2

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3.11 python3.11-venv python3-pip
```

`build-essential` + `cmake` are needed for `i_convert_quantize.sh`'s `llama-quantize`
build (CPU-only build, no CUDA flags needed - quantization doesn't touch the GPU).

## 5. Ollama inside WSL2

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama list   # confirms the daemon is up
```

Ollama auto-detects the GPU through the same passthrough `nvidia-smi` uses; no extra
config needed on a working WSL2 + driver setup from step 2.

## 6. Python environment

```bash
cd /path/to/Qwen-distillation
python3.11 -m venv .venv
source .venv/bin/activate
python scripts/00_preflight_check.py     # will warn about missing torch/psutil - expected here
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/00_preflight_check.py     # re-run - should be clean now
```

`scripts/00_preflight_check.py` checks: native-Windows vs. Linux, Python version,
CUDA visibility + GPU name/compute-capability (expects a T4, Turing, fp16-only),
system RAM, free disk space, and whether Ollama is reachable. Fix anything it reports
as `[FAIL]` before continuing; `[WARN]` lines are worth reading but not blockers.

## 7. Filesystem location matters

Keep this repo, your HF cache, and all model/data output **inside the WSL2 filesystem**
(e.g. `~/Qwen-distillation`), not under `/mnt/c/...`. Cross-filesystem I/O from WSL2
into the Windows NTFS mount is dramatically slower and will make the teacher-generation
step (already the slowest phase, ~1-3 days) worse for no reason.

## 8. Disk space budget

Roughly, across the full pipeline:

| Item | Approx. size |
|---|---|
| Teacher GGUF (Ollama-managed) | ~65-75GB |
| Qwen3-14B base checkpoint (HF cache, fp16) | ~30GB |
| Merged fp16 checkpoint (Phase 7 output) | ~30GB |
| f16 GGUF before quantization (Phase 8, deletable after) | ~30GB |
| Quantized Q4_K_M GGUF (final artifact) | ~9GB |
| Public datasets (HF cache, Section 5 mix) | ~10-30GB depending on which sources you enable |
| llama.cpp checkout + build | ~2GB |

Plan for 150-250GB free on the WSL2 filesystem. `00_preflight_check.py` warns if free
space looks low, but the exact number depends on which optional dataset sources
(`--datasets` in `b_build_prompts.py`) and DPO refinement you actually run.

## 9. Moving the finished model to wherever ATLAS runs

If ATLAS itself also runs inside a WSL2 distro on this same machine, `k_swap_and_smoketest.sh`
just needs `ATLAS_REPO_PATH` pointed at that distro's filesystem path. If ATLAS runs
elsewhere, see the plan's Section 9 (USB/SSD, LAN transfer, or a private HF Hub repo) -
and `sha256sum` the GGUF on both ends after any multi-gigabyte transfer before trusting it.
