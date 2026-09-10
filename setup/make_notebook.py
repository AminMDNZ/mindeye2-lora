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
# MindEye2 + LoRA

Adapts the pretrained shared-subject MindEye2 model to a held-out subject with ~1 hour
of fMRI, six different ways, then compares them statistically and reconstructs images.

**Run cells 1 and 2 at the start of every session.** After that, jump to whichever stage
you need — every stage checkpoints to Drive and skips work already done, so a disconnect
costs minutes rather than hours.

Runtime → Change runtime type → **GPU**.
"""),

md("""
## 1. Environment

Clones or updates the repo, mounts Drive, installs dependencies from the Drive-backed
pip cache. 2–4 minutes the first time in a fresh runtime, under a minute afterwards.
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
assert os.path.isdir(f"{REPO}/src/mindeye_lora"), (
    f"{REPO}/src/mindeye_lora not found — check the repo layout on GitHub."
)
print("repo ready:", os.getcwd())
"""),

code("""
!bash setup/colab_setup.sh
!pip install -q bitsandbytes
"""),

md("""
**If you edit any file under `src/`, restart the runtime before re-running.** Python
caches imported modules, so copying a file into place does not change what is running —
you get errors whose line numbers point at the new file while the old code executes.
This cost several debugging rounds during development.
"""),

md("""
## 2. Attach

Re-imports `main`, sets the Drive root, and enters the repo directory. All three are
lost on a kernel restart; the usual symptom is
`NameError: name 'main' is not defined`.
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
# What GPU did Colab give you? It assigns, you cannot choose.
import torch

if not torch.cuda.is_available():
    print("NO GPU — Runtime > Change runtime type > GPU")
else:
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    gb = total / 1024**3
    print(f"{p.name} — {gb:.1f} GB total, {free/1024**3:.1f} GB free")
    print("  -> configs/colab_t4.yaml" if gb < 24
          else "  -> configs/a100_paper_scale.yaml will also fit")
"""),

md("""
## 3. Which config

| config | needs | gives you |
|---|---|---|
| `configs/smoke.yaml` | any GPU, ~30 min | proof the pipeline runs — **start here** |
| `configs/colab_t4.yaml` | 15 GB, hours | **the experiment**: 6 arms x 3 seeds x 150 epochs |
| `configs/a100_paper_scale.yaml` | 24 GB+ | efficiency numbers at 4096 (2.1B params) |

The T4 config is the experiment, not a fallback — it produces the retention ratio,
equivalence test and full report. The A100 config answers a different question: at
`hidden_dim=4096` the gap between LoRA and full fine-tuning becomes ~23 GB vs ~11 GB of
VRAM and ~8 GB vs a few MB per subject. A supplement, not the finding.

Edit `CONFIG` in cell 2 and re-run it to switch.

Global flags work on either side of the subcommand: `main(["train", "--config", CONFIG])`
and `main(["--config", CONFIG, "train"])` are equivalent.
"""),

md("""
## 4. Everything, in one resumable command

Runs assets → precompute → train → predict → recon → evaluate → compare → report,
skipping whatever is already done. Re-run verbatim after any disconnect.

For the image-heavy stages you may prefer the individual cells below.
"""),
code("""
main(["run-all", "--config", CONFIG])
"""),

code("""
# Where things stand. Safe to run in a second cell while something else is working.
main(["status", "--config", CONFIG])
"""),

md("""
## 5. Assets

Downloads only what is needed. The 22 GB COCO image file is **sliced remotely over
HTTPS** — just the ~1,750 rows this experiment touches — and the fetch resumes per
256-row chunk if the CDN returns a 503. The 2.86 GB pretrained checkpoint is slimmed to
weights-only.

First run: 10–20 minutes. Afterwards: instant.
"""),
code("""
main(["assets", "--config", CONFIG])
"""),

md("""
## 6. Verify

The cheapest place to catch the worst failure. It builds the model, loads the
shared-subject checkpoint, and refuses to continue if any non-ridge parameter is
missing — which would quietly turn "fine-tuning" into "training from scratch".

Look for `skipped 14 pretrained ridge tensors`: 7 subjects x (weight + bias), correctly
left behind so subject 1 gets a fresh ridge layer that every arm trains.
"""),
code("""
main(["verify", "--config", CONFIG])
"""),

md("""
## 7. Precompute CLIP embeddings

Embeds each stimulus once with OpenCLIP ViT-bigG/14 and caches the 256x1664 token
embeddings in fp16, so the 2.5 GB vision tower is never resident during training.
"""),
code("""
main(["precompute", "--config", CONFIG])
"""),

md("""
## 8. Train

One run per (arm, seed). All arms share data order, schedule and starting weights; only
the trainable parameter set differs. Watch the `trainable` counts differ by orders of
magnitude — that line is the experiment in miniature.

Progress reports at three levels, so a long stage is never mistaken for a hang:

```
╔═ training runs 5/18 · subj01_1sess_lora_r16_seed0 ═══
║  ~74 min left (5.3 min per training run so far, 14 remaining)
lora_r16 seed0 · epoch 3/150: 41%|████  | 47/114 [01:52<02:39, loss=10.264]
```

State is saved every 10 minutes and at each epoch boundary; `time_budget_min` stops
cleanly before a session is reclaimed. Re-run this cell next session to continue.
"""),
code("""
main(["train", "--config", CONFIG])
"""),

md("""
## 9. Predict

Runs each trained model over the 1,000 test images and streams predicted CLIP
embeddings to disk as fp16 memmaps. ~13 min per arm.

Streaming matters: holding these in RAM is ~1.7 GB per tensor, and concatenating several
exceeds a 12 GB Colab VM — which appears as "your session crashed after using all
available RAM" with no traceback.
"""),
code("""
main(["predict", "--config", CONFIG])
"""),

md("""
## 10. Images

Two paths.

**Retrieval panel** (default, seconds): the nearest test-set images to each predicted
embedding, top-3, correct hits outlined. These are **retrieved photographs, not
generated** — a correct top-1 is pixel-identical to the stimulus. It is also a coarse
discriminator, so read the statistics for the size of any difference.

**SDXL unCLIP decoder** (real generation): 18 GB download once, then split into fp16
shards that are reused. **~45 s/image on a T4**, so budget accordingly — 32 images x 3
arms is about 40 minutes. It runs fine on a T4: the encoder is not loaded here, the 1.9B
CLIP image embedder is dropped from the conditioner, and xformers is replaced by
PyTorch's native attention, leaving the decoder at ~5 GB of VRAM.

The report's grid shows at most three arms, so decoding more than three is wasted time.
"""),
code("""
# retrieval panel only — fast, no downloads
main(["recon", "--config", CONFIG])
"""),

code("""
# real generated images (slow). 8 for a smoke test, 32 for the real run.
main(["recon", "--config", CONFIG, "--decoder", "sdxl_unclip",
      "--arm", "frozen", "lora_r16", "full", "--n_images", "8"])
"""),

md("""
## 11. Evaluate

Per-image metrics on all 1,000 test images. Two CLIP-space measures always:

- `two_way_clip` — probability the true image outranks a random distractor (chance 0.5)
- `cosine` — similarity between predicted and true CLIP token embeddings

They measure different things and can disagree: a fine-tune can shift embedding geometry
(hurting cosine) while separating images better (helping two-way). A third measure,
`retrieval_percentile`, is excluded because it is algebraically identical to
`two_way_clip` — reporting both would make one result look like two.

The eight MindEye image metrics are added where reconstructions exist. Note they only
cover the images you decoded, so their confidence intervals are much wider than the
CLIP-space ones.

First run downloads AlexNet, Inception, EfficientNet and SwAV (~500 MB total).
"""),
code("""
main(["evaluate", "--config", CONFIG, "--force"])
"""),

md("""
## 12. Compare

Paired statistics against the full fine-tune: BCa bootstrap intervals, Wilcoxon with
Holm correction, Cohen's d_z, TOST equivalence, and the retention ratio.

Expect a difference that is *statistically detectable but practically negligible* for a
well-chosen rank. Both facts get reported: with ~1,000 paired images you can detect gaps
far below anything that matters, which is exactly why the equivalence test is there.
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
from IPython.display import HTML, display
from pathlib import Path

# The HTML twin embeds its figures. The markdown version uses relative paths, which a
# notebook resolves against its own working directory — hence blank images.
report = Path(os.environ["MINDEYE_LORA_ROOT"]) / "results/reports/REPORT.html"
display(HTML(report.read_text()))
"""),

code("""
# Individual figures at full size.
from IPython.display import Image, display
from pathlib import Path

figdir = Path(os.environ["MINDEYE_LORA_ROOT"]) / "results/figures"
for p in sorted(figdir.glob("*.png")):
    print(p.name)
    display(Image(str(p)))
"""),

md("""
---

## Reading the report

**1. frozen→full headroom, first.** If it is tiny, the shared-subject model was already
nearly sufficient and neither method had room to differ. If it is *negative*, full
fine-tuning did worse than not adapting at all, the retention ratio divides by a negative
number, and the finding changes shape entirely.

**2. Retention ratio** — the fraction of the achievable gain each arm recovered, with a
bootstrap interval. An interval spanning zero establishes nothing.

**3. Equivalence column** — the positive claim: the gap is *bounded* below the margin. A
non-significant p-value alone never establishes equivalence.

**4. Seed variability** — any between-arm difference smaller than the between-seed sd is
not worth interpreting.

The bar chart's y-axis is zoomed to the data, so bar height is not proportional to score:
the arms differ by ~0.02 on a metric whose floor is 0.5. The chance line is marked.

## If something breaks

**`NameError: name 'main' is not defined`** — the kernel restarted. Re-run cell 2.

**"Your session crashed after using all available RAM"** — the VM's 12 GB system memory,
not the GPU. `predict` streams to disk so this should not recur; if it does, try
`--num_workers 0`.

**CUDA out of memory** — the error names a `batch_size` / `grad_accum` pair that fits;
apply it in the config, keeping the product at 24. Then **restart the runtime** before
retrying: a CUDA OOM is sticky, because the traceback holds the model that just failed,
so a retry often OOMs before it starts.

**"needs roughly 24 GB of VRAM but this GPU has 15.0 GB"** — you selected the paper-scale
config on a T4 or L4. Switch `CONFIG` to `configs/colab_t4.yaml`.

**Missing module** — run `bash setup/colab_setup.sh`. It resolves dependencies
automatically. If it reports the *same* module twice, that is a moved import path rather
than an absent package and needs a code fix.

**`Refusing to run: the 'pretrained' baseline would be fake`** — upstream's architecture
no longer matches the checkpoint. Do not work around this; pin a revision with
`--upstream_ref=<sha>`.

**Nothing needs deleting after a crash.** Checkpoints are fsynced, verified on read, and
kept in two generations. A corrupt one falls back to its backup; if both are bad they are
removed and that stage restarts. Re-run the same command.

## Poking at things

```python
main(["train", "--config", CONFIG, "--arm", "lora_r16", "--seed", "0"])  # one run
main(["verify", "--config", CONFIG, "--tree"])                           # every Linear layer
main(["compare", "--config", CONFIG, "--equivalence_fraction", "0.1"])   # stricter margin
main(["recon", "--config", CONFIG, "--strict-decoder"])                  # full decoder traceback
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
