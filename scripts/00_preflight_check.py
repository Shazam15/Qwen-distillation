"""Phase 0 (run this first) - sanity-check the machine before touching any data.

Pure Python, no repo-specific imports, so it can run before `pip install -r
requirements.txt` even finishes (it only needs whatever's already on the box) -
run it again after installing requirements to also check torch/CUDA.

Checks, in order:
  1. Whether this is native Windows or a Linux environment (WSL2 or otherwise) -
     autotrain-advanced's dependencies (deepspeed, triton) do not build cleanly on
     native Windows; see docs/WINDOWS_SETUP.md.
  2. Python version.
  3. torch + CUDA visibility, GPU name and compute capability (expects a Tesla T4 /
     Turing, cc 7.5 - fp16 only, no bf16).
  4. System RAM (the plan assumes ~400GB for holding the quantized teacher).
  5. Free disk space in the repo (models/ + data/ need on the order of 150-200GB
     across HF caches, the merged fp16 checkpoint, and GGUF outputs).
  6. Whether the `ollama` CLI is on PATH and the daemon is reachable.

Exits non-zero if a hard blocker is found; prints warnings (without failing) for
soft issues like low disk space, since exact needs depend on which phases you run.
"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ok = True


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    global ok
    ok = False
    print(f"  [FAIL] {msg}")


def info(msg: str) -> None:
    print(f"  {msg}")


section("1. Host OS")
system = platform.system()
if system == "Windows":
    fail(
        "Running on native Windows Python. autotrain-advanced's dependencies "
        "(deepspeed, triton) do not build reliably on native Windows. Run this "
        "pipeline inside WSL2 instead - see docs/WINDOWS_SETUP.md."
    )
elif system == "Linux":
    is_wsl = "microsoft" in platform.uname().release.lower()
    info(f"Linux ({'WSL2' if is_wsl else 'bare metal / other VM'}) - {platform.uname().release}")
    if not is_wsl:
        info("Not detected as WSL - fine if this IS the target Linux box (not the Windows host).")
else:
    warn(f"Unrecognized platform.system() == {system!r} - this pipeline is only tested on Linux/WSL2.")

section("2. Python version")
info(f"Python {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    warn("Python < 3.10 - autotrain-advanced and recent transformers expect 3.10+.")

section("3. GPU / CUDA")
try:
    import torch

    info(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False - no GPU visible. Check NVIDIA drivers "
             "(host) and, under WSL2, that `nvidia-smi` works inside the WSL2 shell too.")
    else:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        info(f"GPU 0: {name} (compute capability {major}.{minor})")
        if "T4" not in name:
            warn(f"Expected a Tesla T4 per the plan - found {name!r}. Bandwidth/VRAM assumptions may not hold.")
        if major < 7:
            fail(f"Compute capability {major}.{minor} is too old for bitsandbytes 4-bit (needs >=7.0).")
        bf16 = torch.cuda.is_bf16_supported()
        info(f"bf16 supported: {bf16} (Turing/T4 should report False - fp16 only)")
        if bf16 is True and major == 7 and minor == 5:
            warn("bf16 reported supported on what looks like Turing - double check the driver/torch build.")
except ImportError:
    warn("torch not installed yet - run `pip install -r requirements.txt` first, then re-run this check.")

section("4. System RAM")
try:
    import psutil

    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    info(f"Total RAM: {total_gb:.0f} GB")
    if total_gb < 350:
        warn(f"Plan assumes ~400GB RAM to hold the quantized teacher (~65-75GB) plus overhead; "
             f"this machine reports {total_gb:.0f}GB.")
except ImportError:
    warn("psutil not installed - run `pip install -r requirements.txt` first, then re-run this check.")

section("5. Disk space")
usage = shutil.disk_usage(REPO_ROOT)
free_gb = usage.free / (1024 ** 3)
info(f"Free space at {REPO_ROOT}: {free_gb:.0f} GB")
if free_gb < 150:
    warn(
        f"Only {free_gb:.0f}GB free. Rough budget: base Qwen3-14B fp16 checkpoint "
        "(~30GB) + merged output (~30GB) + f16 GGUF (~30GB) + quantized GGUF (~9GB) "
        "+ HF datasets cache + llama.cpp build - plan for 150-200GB free."
    )

section("6. Ollama")
ollama_path = shutil.which("ollama")
if not ollama_path:
    fail("`ollama` not found on PATH. Install it (inside WSL2, if that's where you're running "
         "this) from https://ollama.com before running a_setup_env.sh.")
else:
    info(f"Found at {ollama_path}")
    try:
        subprocess.run(["ollama", "list"], check=True, capture_output=True, timeout=10)
        info("Ollama daemon is reachable.")
    except Exception as exc:
        fail(f"`ollama list` failed ({exc}) - is the daemon running?")

print()
if ok:
    print("All hard checks passed. Review any [WARN] lines above, then run scripts/a_setup_env.sh.")
    sys.exit(0)
else:
    print("One or more hard checks FAILED - fix these before running the pipeline.")
    sys.exit(1)
