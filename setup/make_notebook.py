"""Generate notebooks/MindEye2_LoRA_Colab.ipynb."""
import json
from pathlib import Path


def _lines(src: str) -> list[str]:
    """nbformat wants each source entry to keep its trailing newline (last one bare)."""
    parts = src.strip().split("\n")
    return [ln + "\n" for ln in parts[:-1]] + parts[-1:]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(src)}


cells = [
md("""
# MindEye2 + LoRA — Colab driver

Adapts the pretrained shared-subject MindEye2 model to a held-out subject with ~1 hour
of fMRI, six different ways (frozen / BitFit / LoRA r=4,16,64 / full fine-tune), then
compares them statistically and shows images.

**Every cell is safe to re-run.** If the runtime disconnects, reconnect, run cells 1-2
again, and continue — each stage checks the manifest on Drive and skips finished work.

Runtime → Change runtime type → **GPU** (T4 is fine to start).
"""),

md("""
## 1. Clone / update the repo

Mounts Drive and clones the repo, or pulls if it is already there.
"""),
code("""
import os, sys

GITHUB_USER = "<your-username>"      # <-- change this
REPO_NAME   = "mindeye2-lora"
REPO        = f"/content/{REPO_NAME}"

from google.colab import drive
drive.mount('/content/drive')

if os.path.exists(REPO):
    !git -C {REPO} pull --ff-only
else:
    !git clone https://github.com/{GITHUB_USER}/{REPO_NAME}.git {REPO}

os.chdir(REPO)

# Fail here with a clear message rather than three cells later on a cryptic ImportError
assert os.path.isdir(f"{REPO}/src/mindeye_lora"), (
    f"{REPO}/src/mindeye_lora not found. If GitHub shows a nested "
    f"{REPO_NAME}/{REPO_NAME}/ folder, the contents were uploaded one level too deep."
)
print("repo ready:", os.getcwd())
"""),

code("""
!bash setup/colab_setup.sh
# bitsandbytes needs its real dependency chain to find CUDA, so it is installed
# separately rather than through the --no-deps path the setup script uses.
!pip install -q bitsandbytes
"""),

md("""
Installs come from the Drive-backed pip cache: 2-4 minutes the first time, under a
minute afterwards. The script resolves `dalle2-pytorch`'s transitive imports
automatically and ends with `dalle2_pytorch OK` / `environment ready`.

Colab sometimes needs a kernel restart after installs. If cell 2 fails on imports, run
this, then continue from cell 2 — nothing downloaded is lost.
"""),
code("""
# import IPython; IPython.Application.instance().kernel.do_shutdown(True)
"""),

md("""
## 2. Attach to the workspace

**Run this first after any kernel restart or reconnect.** `main`, the Drive root and the
working directory are all lost on restart; the usual symptom is
`NameError: name 'main' is not defined`.

Setting `MINDEYE_LORA_ROOT` matters: without it the workspace silently falls back to
local disk, and every cached download disappears when the session ends.
"""),
code("""
import os, sys

REPO = "/content/mindeye2-lora"
os.chdir(REPO)
os.environ["MINDEYE_LORA_ROOT"] = "/content/drive/MyDrive/mindeye2_lora"
if f"{REPO}/src" not in sys.path:
    sys.path.insert(0, f"{REPO}/src")

from mindeye_lora.cli import main

CONFIG = "configs/smoke.yaml"     # switch to configs/colab_t4.yaml for the real run

main(["setup", "--config", CONFIG])
"""),

code("""
# what you actually got — Pro gives priority, not a guarantee
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"{p.name}  {total/1024**3:.1f} GB total, {free/1024**3:.1f} GB free")
else:
    print("NO GPU — Runtime > Change runtime type > GPU")
"""),

md("""
## 3. Which config

| config | runtime | time |
|---|---|---|
| `configs/smoke.yaml` | any GPU | ~30 min — **start here** |
| `configs/colab_t4.yaml` | T4 (free) | several hours across sessions |
| `configs/a100_paper_scale.yaml` | A100 (Pro) | also `!pip install bitsandbytes` |

The smoke config runs 3 arms for 10 epochs and exercises every stage, so problems
surface in minutes rather than hours. Nothing it downloads is wasted — assets and CLIP
embeddings are cached on Drive and reused by the full run.

To switch, edit `CONFIG` in the cell above and re-run it.

Global flags work on either side of the subcommand, so `main(["train", "--config",
CONFIG])` and `main(["--config", CONFIG, "train"])` are equivalent.
"""),

md("""
## 4. Everything, in one resumable command

Runs assets → precompute → train → predict → recon → evaluate → compare → report,
skipping anything already finished. Re-run it verbatim after any disconnect.

Prefer to watch stage by stage? Skip this cell and use sections 5-13 instead.
"""),
code("""
main(["run-all", "--config", CONFIG])
"""),

md("""
## 5. Assets

Downloads only what is needed. The 22 GB COCO image file is **sliced remotely over
HTTPS** — just the ~1,750 rows this experiment touches. The 2.86 GB pretrained
checkpoint is slimmed to weights-only and the original deleted.

First run: 10-20 minutes. Afterwards: instant. Watch for repeated `fetched N/M rows`.
"""),
code("""
main(["assets", "--config", CONFIG])
"""),

md("""
## 6. Verify the pretrained weights

**The stage most likely to fail, and the cheapest place to find out.** It builds the
model, loads the shared-subject checkpoint, and refuses to continue if any non-ridge
parameter is missing — which would quietly turn "fine-tuning" into "training from
scratch" and invalidate the whole comparison.

Look for `skipped 7 pretrained ridge tensors`: those are subjects 2-8's subject-specific
layers, correctly left behind so subject 1 gets a fresh one.
"""),
code("""
main(["verify", "--config", CONFIG])
"""),

md("""
## 7. Precompute CLIP embeddings

Embeds each stimulus once with OpenCLIP ViT-bigG/14 and caches the 256x1664 token
embeddings in fp16. This keeps the 2.5 GB vision tower out of memory during training,
which is what makes the T4 config fit.
"""),
code("""
main(["precompute", "--config", CONFIG])
"""),

md("""
## 8. Train

One run per (arm, seed). All arms share data order, schedule and starting weights; only
the trainable parameter set differs. Watch the `trainable` counts differ by orders of
magnitude — that line is the experiment in miniature.

State is saved to Drive every 10 minutes and at each epoch boundary, and
`time_budget_min` stops cleanly before a session is likely to be reclaimed. Re-run this
cell next session to continue.

Progress reports at three levels — stage, run within the sweep, and batch within the
run — so a long stage is never indistinguishable from a hang:

```
╔═ training runs 5/18 · subj01_1sess_lora_r16_seed0 ═══
║  ~74 min left (5.3 min per training run so far, 14 remaining)
lora_r16 seed0 · epoch 3/150: 41%|████  | 47/114 [01:52<02:39, loss=10.264]
```

A memory preflight runs before each arm and refuses runs it predicts will not fit,
naming a `batch_size` / `grad_accum` pair that should. The shipped configs use
`batch_size: 6, grad_accum: 4` — an effective batch of 24, matching the paper, split
because the diffusion prior's attention allocates a `[batch, 32 heads, 257, 257]`
similarity matrix per layer and OOMs a T4 at batch 24.
"""),
code("""
main(["train", "--config", CONFIG])
"""),

code("""
# where things stand — worth running after any disconnect
main(["status", "--config", CONFIG])
"""),

md("""
## 9. Predict

Runs each trained model over the test set and caches its predicted CLIP embeddings,
sampled through the diffusion prior. Roughly 12 minutes per arm.

Partial results are written every 10 batches to `predictions.partial.pt` and resumed
automatically, so a disconnect costs a minute or two rather than the whole arm. Running
one arm per cell narrows the blast radius further.
"""),
code("""
main(["predict", "--config", CONFIG])

# or one arm at a time, if the connection is unreliable:
# main(["predict", "--config", CONFIG, "--arm", "frozen"])
# main(["predict", "--config", CONFIG, "--arm", "lora_r16"])
# main(["predict", "--config", CONFIG, "--arm", "full"])
"""),

md("""
## 10. Images

Two paths, and this cell always produces something.

**Retrieval fallback** (default): the nearest test-set images to each predicted
embedding, top-3, with correct hits outlined and the true rank annotated. Seconds, no
downloads. These are **retrieved photographs, not generated images** — a correct top-1
is pixel-identical to the stimulus, which is a retrieval hit rather than a
reconstruction. It is also a coarse discriminator: arms a few percent apart often give
identical rows, so read the statistics for the size of any difference.

**SDXL unCLIP decoder** (`--decoder sdxl_unclip`): the paper's decoder, and real
generation. 18 GB download, needs Stability's `sgm`, ~3-5 s/image. Frozen and identical
across arms, so it adds no between-arm variance and every statistic works without it.
Worth it on an A100; skip on a free T4.
"""),
code("""
main(["recon", "--config", CONFIG])

# real generated images — uncomment on an A100 with disk to spare
# main(["recon", "--config", CONFIG, "--decoder", "sdxl_unclip",
#       "--arm", "frozen", "lora_r16", "full", "--n_images", "32"])
"""),

md("""
## 11. Evaluate

Per-image metrics. CLIP-space metrics always (cosine, two-way identification, retrieval
percentile); the eight MindEye image metrics as well if reconstructions exist.
"""),
code("""
main(["evaluate", "--config", CONFIG])
"""),

md("""
## 12. Compare

Paired statistics against the full fine-tune: BCa bootstrap intervals, Wilcoxon with
Holm correction, Cohen's d_z, TOST equivalence, and the retention ratio.

Expect a difference that is *statistically detectable but practically negligible* for a
well-chosen rank. Both facts get reported, because with ~1,000 paired test images you
can detect gaps far below anything that matters.
"""),
code("""
main(["compare", "--config", CONFIG])
"""),

md("""
## 13. Report
"""),
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
from IPython.display import Image
figdir = Path(os.environ["MINDEYE_LORA_ROOT"]) / "results/figures"
for p in sorted(figdir.glob("*.png")):
    print(p.name)
    display(Image(str(p)))
"""),

md("""
---

## If something breaks

**`NameError: name 'main' is not defined`** — the kernel restarted. Re-run cell 2.

**CUDA out of memory.** The error names a `batch_size` / `grad_accum` pair that should
fit; apply it in the config, keeping the product at 24. Then **restart the runtime**
before retrying: a CUDA OOM in a notebook is sticky, because the traceback holds every
local in every frame including the model that just failed, so a retry often OOMs before
training even starts. Nothing on Drive is lost.

**Is it stuck?** Check the GPU from a second cell:
`!nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader`.
A few MiB of memory means no process holds the GPU — the run died rather than stalled,
so reconnect and re-run. Several GB held at a sustained 0% means it is blocked on data
loading, not computing. Data caches are mirrored to local disk automatically because
Drive is slow at the random-access reads the loaders do; `--no-local-cache` disables it.

**Anything about a missing module.** `colab_setup.sh` resolves missing packages
automatically. If it reports the *same* module twice, that is a moved import path rather
than an absent package and needs a code fix, not another install.

**`Refusing to run: the 'pretrained' baseline would be fake`.** Upstream's architecture
no longer matches the checkpoint. Do not work around this — it would mean training from
scratch while calling it fine-tuning. Pin a revision with `--upstream_ref=<sha>`.

## Reading the report

Check the **frozen→full headroom** column first. If it is tiny, the shared-subject model
was already nearly sufficient, neither method had room to differ, and nothing else in
the report means much.

Then the **retention ratio** — the fraction of the achievable gain each arm recovered —
and the **equivalence** column, which is the positive claim. A non-significant p-value
alone never establishes equivalence.

## Poking at things

```python
main(["train", "--config", CONFIG, "--arm", "lora_r16", "--seed", "0"])  # one run
main(["verify", "--config", CONFIG, "--tree"])                           # every Linear layer
main(["compare", "--config", CONFIG, "--equivalence_fraction", "0.1"])   # stricter margin
main(["train", "--config", CONFIG, "--ignore-memory-check"])             # override preflight
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
print("wrote", out, f"({len(cells)} cells)")
