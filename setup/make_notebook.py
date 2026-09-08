"""Generate notebooks/MindEye2_LoRA_Colab.ipynb."""
import json
from pathlib import Path

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.strip().split("\n")}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": src.strip().split("\n")}

cells = [
md("""
# MindEye2 + LoRA — Colab driver

Runs the whole experiment: adapt the pretrained shared-subject MindEye2 model to a
held-out subject with ~1 hour of fMRI, using LoRA vs. full fine-tuning vs. a frozen
baseline, then compare them statistically.

**Every cell is safe to re-run.** If the runtime disconnects, reconnect and run the
cells again from the top — each stage checks the workspace manifest on Drive and skips
work that's already done.

Set the runtime to **GPU** (Runtime → Change runtime type → T4 is fine).
"""),

md("## 1. Environment\n\nMounts Drive, installs dependencies from the Drive-backed pip cache, and points every library cache at Drive. Fast after the first run."),
code("""
import os, sys, subprocess
REPO = "/content/mindeye2-lora"

if not os.path.exists(REPO):
    # replace with your fork if you have one
    !git clone https://github.com/<you>/mindeye2-lora.git {REPO}
%cd {REPO}

from google.colab import drive
drive.mount('/content/drive')

# everything persistent lives here — change it if you keep Drive organised differently
os.environ["MINDEYE_LORA_ROOT"] = "/content/drive/MyDrive/mindeye2_lora"
"""),

code("""
!bash setup/colab_setup.sh
"""),

code("""
# Colab sometimes needs a restart after installs. If the next cell errors on imports,
# run this, then continue from cell 4 (you will NOT lose any downloaded data).
# import IPython; IPython.Application.instance().kernel.do_shutdown(True)
"""),

md("## 2. Workspace\n\nCreates the Drive directory tree and redirects `HF_HOME`, `TORCH_HOME`, and the pip cache into it."),
code("""
sys.path.insert(0, f"{REPO}/src")
from mindeye_lora.cli import main

main(["setup", "--config", "configs/colab_t4.yaml"])
"""),

md("""
## 3. Choose a config

| config | runtime | time |
|---|---|---|
| `configs/smoke.yaml` | any GPU | ~10 min — **start here** |
| `configs/colab_t4.yaml` | T4 (free) | several hours across sessions |
| `configs/a100_paper_scale.yaml` | A100/L4 (Pro) | longer; `hidden_dim=4096` |

The smoke config runs 3 arms for 10 epochs and exercises every stage. Run it first so
configuration problems surface in minutes rather than hours.
"""),
code("""
CONFIG = "configs/smoke.yaml"      # switch to configs/colab_t4.yaml for the real run
"""),

md("""
## 4. Assets

Downloads what the experiment needs and nothing else. The 22 GB COCO image file is
*sliced remotely over HTTPS* — only the ~1,750 rows this experiment touches are pulled.
The 2.86 GB pretrained checkpoint is slimmed to weights-only and the original deleted.

First run: 10–20 minutes. Afterwards: instant.
"""),
code("""
main(["assets", "--config", CONFIG])
"""),

md("## 5. Verify the pretrained weights load\n\nThis is the guard rail. It builds the model, loads the shared-subject checkpoint, and refuses to continue if any non-ridge parameter is missing — which would silently turn 'fine-tuning' into 'training from scratch'. It also prints which layers LoRA will wrap."),
code("""
main(["verify", "--config", CONFIG])
"""),

md("## 6. Precompute CLIP embeddings\n\nEmbeds each stimulus once with OpenCLIP ViT-bigG/14 and caches the 256x1664 token embeddings in fp16. This is what keeps the 2.5 GB vision tower out of memory during training."),
code("""
main(["precompute", "--config", CONFIG])
"""),

md("""
## 7. Train

One run per (arm, seed). All arms share data order, schedule, and starting weights;
only the trainable parameter set differs.

State is written to Drive every 10 minutes and at every epoch boundary. `time_budget_min`
in the config stops training cleanly before a session is likely to be reclaimed — just
re-run this cell in the next session to continue.
""".strip()),
code("""
main(["train", "--config", CONFIG])
"""),

code("""
# where things stand — run this any time, especially after a disconnect
main(["status", "--config", CONFIG])
"""),

md("## 8. Predict\n\nRuns each trained model over the test set and caches the predicted CLIP embeddings (sampled through the diffusion prior)."),
code("""
main(["predict", "--config", CONFIG])
"""),

md("""
## 9. Images

Two paths, and this cell always produces something.

**SDXL unCLIP decoder** (`--decoder sdxl_unclip`): the paper's decoder. 18 GB download,
needs Stability's `sgm`, ~3-5 s/image. Frozen and identical across arms, so it adds no
between-arm variance — every statistic works without it. Worth it on an A100.

**Retrieval fallback** (automatic): if the decoder is missing or fails, you get a
nearest-neighbour panel instead — the closest test-set images to each predicted
embedding, top-3, with correct hits outlined. Costs seconds and no downloads.

Those are **retrieved photographs, not generated images**. A correct top-1 is
pixel-identical to the stimulus, which is a retrieval hit, not a reconstruction. It is
also a coarse discriminator: arms a few percent apart often give identical rows, so read
the statistics tables for the size of any difference.
"""),
code("""
# retrieval fallback only (fast, always works)
main(["recon", "--config", CONFIG])

# real generation — uncomment on an A100 with disk to spare
# main(["recon", "--config", CONFIG, "--decoder", "sdxl_unclip", "--n_images", "64"])
"""),

md("## 10. Evaluate\n\nPer-image metrics. CLIP-space metrics always; the eight image metrics if reconstructions exist."),
code("""
main(["evaluate", "--config", CONFIG])
"""),

md("""
## 11. Compare

Paired statistics against the full fine-tune: bootstrap CIs, Wilcoxon with Holm
correction, Cohen's d_z, TOST equivalence, and the retention ratio.

Expect to see a difference that is *statistically detectable but practically negligible*
for a well-chosen rank. Both facts are reported because both are true.
"""),
code("""
main(["compare", "--config", CONFIG])
"""),

md("## 12. Report\n\nWrites figures and `REPORT.md` into the Drive workspace under `results/reports/`."),
code("""
main(["report", "--config", CONFIG])
"""),

code("""
from IPython.display import Markdown, display
from pathlib import Path
report = Path(os.environ["MINDEYE_LORA_ROOT"]) / "results/reports/REPORT.md"
display(Markdown(report.read_text()))
"""),

code("""
# figures inline
from IPython.display import Image
figdir = Path(os.environ["MINDEYE_LORA_ROOT"]) / "results/figures"
for p in sorted(figdir.glob("*.png")):
    print(p.name)
    display(Image(str(p)))
"""),

md("""
---

## One-shot alternative

Everything above in a single resumable command:

```python
main(["run-all", "--config", "configs/colab_t4.yaml"])
```

Re-run it verbatim after any disconnect.

## Poking at things

```python
main(["train", "--config", CONFIG, "--arm", "lora_r16", "--seed", "0"])   # one run
main(["verify", "--config", CONFIG, "--tree"])                            # every Linear layer
main(["compare", "--config", CONFIG, "--equivalence_fraction", "0.1"])    # stricter margin
```
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
print("wrote", out)
