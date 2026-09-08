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

echo "→ pip cache: ${PIP_CACHE_DIR}"
PIP="python -m pip install -q --cache-dir ${PIP_CACHE_DIR}"

echo "→ core dependencies"
${PIP} \
  "numpy>=1.26" "scipy>=1.11" "h5py>=3.10" "fsspec[http]>=2024.2.0" "aiohttp>=3.9" \
  "huggingface_hub>=0.23" "pyyaml>=6.0" "matplotlib>=3.8" "scikit-image>=0.22" \
  "tqdm>=4.66" "open_clip_torch>=2.24" "omegaconf>=2.3"

echo "→ dalle2-pytorch runtime imports (no torch changes)"
${PIP} \
  "einops>=0.7" "einops-exts>=0.0.4" "rotary-embedding-torch>=0.3" "ema-pytorch>=0.2" \
  "vector-quantize-pytorch>=1.9" "x-clip>=0.12" "coca-pytorch>=0.1" \
  "clip-anytorch>=2.5" "resize-right>=0.0.2" "kornia>=0.7" "webdataset" "click"

echo "→ dalle2-pytorch (--no-deps, so torch is left alone)"
${PIP} --no-deps "dalle2-pytorch==1.15.6"

echo "→ this package"
${PIP} -e .

python - <<'PY'
import torch, sys
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu   {p.name}  {p.total_memory/1024**3:.1f} GB")
try:
    import dalle2_pytorch  # noqa: F401
    print("dalle2_pytorch OK")
except Exception as e:
    print("dalle2_pytorch FAILED:", e); sys.exit(1)
try:
    import open_clip  # noqa: F401
    print("open_clip OK")
except Exception as e:
    print("open_clip FAILED:", e); sys.exit(1)
PY

echo "✔ environment ready"
