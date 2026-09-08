%%writefile /content/mindeye2-lora/setup/colab_setup.sh
#!/usr/bin/env bash
# Idempotent Colab environment setup.
#
# Run this once per runtime. Packages install to local disk (importing site-packages
# from Drive is painfully slow), but the pip *wheel cache* lives on Drive, so every run
# after the first resolves from cache and finishes in well under a minute.
#
#   !bash setup/colab_setup.sh
#
# Deliberately does NOT touch torch: Colab ships a build matched to its CUDA driver and
# replacing it is the single most common way to break a runtime.

set -euo pipefail

ROOT="${MINDEYE_LORA_ROOT:-/content/drive/MyDrive/mindeye2_lora}"
export PIP_CACHE_DIR="${ROOT}/cache/pip"
mkdir -p "${PIP_CACHE_DIR}"

echo "-> pip cache: ${PIP_CACHE_DIR}"
PIP="python -m pip install -q --cache-dir ${PIP_CACHE_DIR}"

echo "-> core dependencies"
${PIP} \
  "numpy>=1.26" "scipy>=1.11" "h5py>=3.10" "fsspec[http]>=2024.2.0" "aiohttp>=3.9" \
  "huggingface_hub>=0.23" "pyyaml>=6.0" "matplotlib>=3.8" "scikit-image>=0.22" \
  "tqdm>=4.66" "open_clip_torch>=2.24" "omegaconf>=2.3"

echo "-> dalle2-pytorch runtime imports (no torch changes)"
# dalle2_pytorch/__init__.py imports its trainer, which pulls in pytorch-warmup,
# accelerate and embedding-reader. Because we install dalle2 with --no-deps to protect
# Colab's torch, every transitive import has to be listed here by hand.
${PIP} \
  "einops>=0.7" "einops-exts>=0.0.4" "rotary-embedding-torch>=0.3" "ema-pytorch>=0.2" \
  "vector-quantize-pytorch>=1.9" "x-clip>=0.12" "coca-pytorch>=0.1" \
  "clip-anytorch>=2.5" "resize-right>=0.0.2" "kornia>=0.7" "webdataset" "click" \
  "pytorch-warmup" "accelerate" "embedding-reader"

echo "-> dalle2-pytorch (--no-deps, so torch is left alone)"
${PIP} --no-deps "dalle2-pytorch==1.15.6"

echo "-> this package"
${PIP} -e .

python - <<'PY'
import importlib, subprocess, sys
import torch

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu   {p.name}  {p.total_memory/1024**3:.1f} GB")

# Self-healing import check. The --no-deps install of dalle2-pytorch means its
# transitive imports are our responsibility, and the exact set drifts with the
# dalle2 version and the Colab base image. Rather than hard-coding a list that goes
# stale, resolve whatever is missing by following the ModuleNotFoundError chain.
# Everything installs --no-deps, so nothing here can swap out Colab's torch.
PKG = {
    "pytorch_warmup": "pytorch-warmup", "embedding_reader": "embedding-reader",
    "ema_pytorch": "ema-pytorch", "einops_exts": "einops-exts",
    "rotary_embedding_torch": "rotary-embedding-torch", "x_clip": "x-clip",
    "coca_pytorch": "coca-pytorch", "clip": "clip-anytorch",
    "resize_right": "resize-right", "vector_quantize_pytorch": "vector-quantize-pytorch",
    "PIL": "pillow", "sklearn": "scikit-learn", "cv2": "opencv-python-headless",
}

def ensure(module: str, label: str, max_installs: int = 15) -> None:
    for attempt in range(max_installs):
        try:
            importlib.invalidate_caches()
            importlib.import_module(module)
            extra = f" (resolved {attempt} missing import{'s' if attempt != 1 else ''})"
            print(f"{label} OK{extra if attempt else ''}")
            return
        except ModuleNotFoundError as exc:
            name = exc.name or ""
            if not name or name == module:
                print(f"{label} FAILED: {exc}")
                sys.exit(1)
            pkg = PKG.get(name, name.replace("_", "-"))
            print(f"  {label}: missing {name!r} -> installing {pkg}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-deps", pkg],
                check=False,
            )
        except Exception as exc:
            print(f"{label} FAILED: {exc}")
            sys.exit(1)
    print(f"{label} FAILED: unresolved after {max_installs} installs")
    sys.exit(1)

ensure("dalle2_pytorch", "dalle2_pytorch")
ensure("open_clip", "open_clip")
ensure("h5py", "h5py")
ensure("fsspec", "fsspec")
PY

echo "environment ready"