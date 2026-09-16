"""Generate notebooks/MindEye2_LoRA_Colab.ipynb — the deliverable notebook."""
import json
from pathlib import Path


def _lines(src: str) -> list[str]:
    parts = src.strip().split("\n")
    return [ln + "\n" for ln in parts[:-1]] + parts[-1:]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(src)}


cells = [
md("""
# LoRA vs. full fine-tuning for MindEye2 subject adaptation

Adapts a shared-subject MindEye2 model to a held-out subject using ~1 hour of fMRI, six
ways, and compares them statistically.

| arm | trainable | what it is |
|---|---|---|
| `frozen` | 16.1M | subject-specific ridge layer only |
| `bitfit` | 16.6M | + biases and LayerNorms |
| `lora_r4` | 18.8M | + rank-4 adapters |
| `lora_r16` | 26.7M | + rank-16 adapters |
| `lora_r64` | 58.6M | + rank-64 adapters |
| `full` | 729M | everything |

Requires a GPU runtime. Cells 1–2 run at the start of every session; all stages
checkpoint to Drive and resume.
"""),

md("### 1. Setup"),
code("""
import os, sys

GITHUB_USER = "<your-username>"
REPO = "/content/mindeye2-lora"

from google.colab import drive
drive.mount('/content/drive')

if os.path.exists(REPO):
    !git -C {REPO} pull --ff-only
else:
    !git clone https://github.com/{GITHUB_USER}/mindeye2-lora.git {REPO}
%cd {REPO}
"""),

code("""
!bash setup/colab_setup.sh
!pip install -q bitsandbytes
"""),

md("Restart the runtime after editing anything under `src/` — Python caches imports."),

md("### 2. Attach"),
code("""
import os, sys

REPO = "/content/mindeye2-lora"
os.chdir(REPO)
os.environ["MINDEYE_LORA_ROOT"] = "/content/drive/MyDrive/mindeye2_lora"
if f"{REPO}/src" not in sys.path:
    sys.path.insert(0, f"{REPO}/src")

from mindeye_lora.cli import main

CONFIG = "configs/main.yaml"     # smoke.yaml to validate · paper_scale.yaml for 4096

main(["setup", "--config", CONFIG])
"""),

code("""
import torch
p = torch.cuda.get_device_properties(0)
print(f"{p.name} — {p.total_memory/1024**3:.1f} GB")
"""),

md("""
| config | needs | runs |
|---|---|---|
| `configs/smoke.yaml` | any GPU, ~30 min | 3 arms × 10 epochs |
| `configs/main.yaml` | ~12 GB VRAM | 6 arms × 3 seeds × 150 epochs |
| `configs/paper_scale.yaml` | ~24 GB VRAM | same at `hidden_dim=4096` |
"""),

md("### 3. Run the experiment"),
code("""
main(["run-all", "--config", CONFIG])
"""),

code("""
main(["status", "--config", CONFIG])
"""),

md("""
`run-all` covers assets → precompute → train → predict → recon → evaluate → compare →
report, skipping finished work. Re-run the same command after a disconnect.

Individual stages, if preferred:

```python
main(["assets",     "--config", CONFIG])   # download and slice data
main(["verify",     "--config", CONFIG])   # check pretrained weights load
main(["precompute", "--config", CONFIG])   # cache CLIP embeddings
main(["train",      "--config", CONFIG])   # ~150 min/arm on a 16 GB GPU
main(["predict",    "--config", CONFIG])   # ~13 min/arm
main(["evaluate",   "--config", CONFIG])
main(["compare",    "--config", CONFIG])
main(["report",     "--config", CONFIG])
```
"""),

md("### 4. Generated reconstructions"),
code("""
# SDXL unCLIP decoder. 18 GB download once, then ~45 s/image.
main(["recon", "--config", CONFIG, "--decoder", "sdxl_unclip",
      "--arm", "frozen", "bitfit", "lora_r4", "lora_r16", "lora_r64", "full",
      "--n_images", "16"])

main(["evaluate", "--config", CONFIG, "--force"])
main(["report", "--config", CONFIG])
"""),

md("""
To show all six arms in the figures, set in `configs/main.yaml`:

```yaml
qualitative_arms: 6
retrieval_k: 1
```
"""),

md("### 5. Report"),
code("""
from IPython.display import HTML, display
from pathlib import Path

report = Path(os.environ["MINDEYE_LORA_ROOT"]) / "results/reports/REPORT.html"
display(HTML(report.read_text()))
"""),

code("""
from IPython.display import Image, display
from pathlib import Path

for p in sorted((Path(os.environ["MINDEYE_LORA_ROOT"]) / "results/figures").glob("*.png")):
    print(p.name)
    display(Image(str(p)))
"""),

md("""
### Reading the report

`two_way_clip` is the primary metric: per test image, the fraction of the other 999 test
images that the correct target outranks in CLIP space. Chance is 0.5.

Check the frozen→reference headroom before interpreting retention ratios — when it is
small, or when another arm beats the reference, the ratio divides by a near-zero
denominator and the report says so.

Any between-arm difference smaller than the between-seed standard deviation is not
worth interpreting.

### Troubleshooting

| symptom | cause |
|---|---|
| `NameError: main is not defined` | kernel restarted — re-run cell 2 |
| session crashed, all RAM used | system RAM, not GPU; try `--num_workers 0` |
| CUDA OOM | apply the `batch_size`/`grad_accum` pair in the message, then restart |
| `needs roughly 24 GB` | use `configs/main.yaml` |
| missing module | re-run `setup/colab_setup.sh` |
| `Refusing to run: baseline would be fake` | upstream changed; pin `--upstream_ref=<sha>` |

Checkpoints are fsynced, verified on read, and kept in two generations — nothing needs
deleting after a crash.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "MindEye2_LoRA_Colab.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(cells)} cells)")
