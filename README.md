# MindEye2 + LoRA

**Question:** when you adapt a pretrained shared-subject MindEye2 model to a new person
with one hour of fMRI, does the pretrained backbone actually need a full fine-tune — or
is a low-rank update enough?

This repo runs that experiment end to end on a single Colab GPU: it downloads the
published MindEye2 checkpoints, fine-tunes the same model six different ways, decodes
images, and produces a statistical report that says how much difference LoRA makes and
whether the gap matters.

---

## The setup in one picture

```
voxels ──► ridge (subject-specific, ALWAYS fully trained)
             │
             ▼
        backbone MLP  ┐
             │        ├── pretrained on subjects 2–8 ── this is what LoRA adapts
        diffusion prior ┘
             │
             ▼
     CLIP ViT-bigG/14 token embeddings ──► SDXL unCLIP (frozen) ──► image
```

The subject-1 ridge layer does not exist in the pretrained checkpoint — it is randomly
initialised, sized to that subject's voxel count. So **it is fully trained in every arm,
including the LoRA and frozen arms.** LoRA is a claim about how far *pretrained* weights
need to move; it says nothing about a layer that has no pretrained value. Getting this
wrong is the easiest way to produce a meaningless comparison, so `model_factory.py`
refuses to run if any non-ridge parameter turns out to be missing from the checkpoint.

## Arms

| arm | what is trainable | why it's here |
|---|---|---|
| `frozen` | ridge only | Defines the headroom. If this already matches `full`, nothing else matters. |
| `bitfit` | ridge + biases + LayerNorms | Cheap PEFT baseline that isn't low-rank. |
| `lora_r4` | ridge + rank-4 adapters | Is even a tiny update enough? |
| `lora_r16` | ridge + rank-16 adapters | The default. |
| `lora_r64` | ridge + rank-64 adapters | Does more rank keep helping? |
| `full` | everything | The paper's recipe; the reference arm. |

Data, schedule, batch size, seed, and loss are identical across arms. The only thing
that varies is which parameters carry gradients.

---

## Quickstart (Colab)

```python
# cell 1 — clone and set up (once per runtime, ~60s after the first time)
!git clone https://github.com/<you>/mindeye2-lora.git
%cd mindeye2-lora
!bash setup/colab_setup.sh
```

```python
# cell 2 — smoke test the whole pipeline in ~10 minutes before committing hours
!python -m mindeye_lora.cli run-all --config configs/smoke.yaml
```

```python
# cell 3 — the real run
!python -m mindeye_lora.cli run-all --config configs/colab_t4.yaml
```

If the runtime dies, **re-run the exact same command.** Every stage checks the workspace
manifest and picks up where it stopped.

Or use `notebooks/MindEye2_LoRA_Colab.ipynb`, which walks through the stages one cell at
a time with checkpoints in between.

### Stages individually

Global options (`--config`, `--epochs`, `--seeds`, ...) work on either side of the
subcommand, so `--config c.yaml train` and `train --config c.yaml` are equivalent.

```bash
python -m mindeye_lora.cli setup       # create the Drive workspace, print GPU info
python -m mindeye_lora.cli assets      # download / slice the data (~3 GB, once)
python -m mindeye_lora.cli verify      # confirm the pretrained weights load cleanly
python -m mindeye_lora.cli precompute  # cache CLIP bigG embeddings (~1.5 GB, once)
python -m mindeye_lora.cli train --arm lora_r16 --seed 0
python -m mindeye_lora.cli predict     # test-set embeddings for each trained model
python -m mindeye_lora.cli recon --decoder sdxl_unclip   # optional, heavy
python -m mindeye_lora.cli evaluate    # per-image metrics
python -m mindeye_lora.cli compare     # paired statistics
python -m mindeye_lora.cli report      # figures + REPORT.md
python -m mindeye_lora.cli status      # what's done, what isn't
```

---

## Not re-downloading everything every session

Colab wipes `/content` when the runtime recycles. Everything expensive therefore lives
under one Drive root (`/content/drive/MyDrive/mindeye2_lora` by default), and
`env.setup_environment()` redirects the cache environment variables **before** torch or
huggingface_hub are imported:

```
mindeye2_lora/
  cache/huggingface/   HF_HOME          – hub downloads
  cache/torch/         TORCH_HOME       – AlexNet, Inception, EfficientNet, SwAV
  cache/pip/           PIP_CACHE_DIR    – wheels, so reinstalls are near-instant
  upstream/            MindEyeV2 clone (pinned by SHA in the manifest)
  assets/              slimmed checkpoints, webdataset shards
  data/                voxel + image + CLIP-embedding subsets
  runs/<run>/          state.pt, adapter.pt, train_log.csv, predictions.pt, metrics
  results/             statistics.json, figures, REPORT.md
  MANIFEST.json        which stages have completed
```

Python packages install to *local* disk on purpose — importing site-packages from Drive
is painfully slow — but the wheel cache is on Drive, so `colab_setup.sh` is fast and
mostly offline after the first run.

### Keeping it under a free 15 GB Drive

The full HF dataset is 215 GB. Two tricks get the working set to about 3.5 GB:

1. **Remote HDF5 row slicing.** A 1-session experiment touches fewer than 2,000 distinct
   images, so instead of downloading the 22 GB `coco_images_224_float16.hdf5` we open it
   over HTTPS with fsspec and pull only the rows we need. Same for the 1.89 GB betas
   file. Pass `--full-hdf5` if you'd rather have local copies.

   The fetch takes ~15 minutes and runs against a CDN that intermittently returns 503
   or 429 under load, so transient failures are retried with exponential backoff, the
   HDF5 handle is rebuilt if a connection drops, and rows are checkpointed after every
   chunk. An interrupted fetch resumes where it stopped rather than starting over.
2. **Checkpoint slimming.** The published `last.pth` is 2.86 GB because it carries
   DeepSpeed optimiser state. We strip it once and delete the original.

| artifact | size |
|---|---|
| voxel subset (1 session + test) | ~250 MB |
| image subset (~1,750 images) | ~530 MB |
| CLIP bigG embeddings, fp16 | ~1.5 GB |
| slimmed pretrained checkpoint | ~1 GB |
| all six arms × 3 seeds, LoRA adapters | a few hundred MB |
| **full fine-tune checkpoints** | ~1 GB **each** — this is the point |

The SDXL unCLIP decoder is a separate 18 GB download; see below.

---

## Reconstructions

`--decoder sdxl_unclip` runs the paper's decoder. It is optional and heavy:

- 18 GB checkpoint, downloaded to *local* disk (`/content/unclip_cache`), then split
  once into ~1.5 GB fp16 shards kept on Drive (~9 GB total). The raw file is deleted
  afterwards and later sessions reuse the shards.
- **Never loaded whole.** `torch.load` on 18 GB needs 18 GB of RAM and kills a standard
  12 GB Colab VM outright. Sharding reads the source with `mmap=True`, casts tensor by
  tensor, and streams shards into a GPU-resident engine, so peak RAM is about one shard.
- Needs Stability's `sgm` package, which ships inside the MindEyeV2 clone.
- Roughly 11 s/image on a T4 at 38 diffusion steps. The encoder is not loaded during
  this stage, and the 1.9B-parameter CLIP image embedder is dropped from the conditioner
  (the embedding is supplied directly), so the decoder needs about 5 GB of VRAM.
- xformers is not required: a shim routes sgm's `memory_efficient_attention` to
  PyTorch's native SDPA, which computes the same thing without needing a build matched
  to your torch version.

**Every statistic in this project works with `--decoder none.`** The decoder is frozen
and identical across arms, so it contributes no between-arm variance. The primary
outcome measures live in CLIP embedding space, which is the only thing LoRA touches:

- `cosine` — similarity between predicted and true CLIP token embeddings
- `two_way_clip` — per-image probability the true image outranks a random distractor

(`retrieval_percentile` exists but is excluded by default: it is algebraically the same
quantity as `two_way_clip`, so reporting both would double-count one result.)

With reconstructions you additionally get the eight metrics from the MindEye papers
(PixCorr, SSIM, AlexNet-2/5, Inception, CLIP, EffNet-B, SwAV). Note these are computed
on the first `recon_n_images` (default 100), so their confidence intervals are wider
than the CLIP-space ones, which use all ~1,000 test images.

### Retrieval panel (the fallback)

If the decoder is unavailable — the sgm dependency is the most fragile part of the
project — `cli recon` falls back to a nearest-neighbour panel instead of failing the
run. For each test scan it shows the top-3 closest test-set images to the model's
predicted embedding, outlines correct hits, and annotates where the true image ranked.

**These are retrieved photographs, not generated images.** A correct top-1 is
pixel-identical to the stimulus above it, which is a retrieval hit and not a
reconstruction; the figure and the report both say so in as many words. Retrieval is
also a hit-or-miss threshold, so two arms a few percent apart often produce identical
rows — read the statistics tables for the size of the difference, not this panel. It
renders what the existing `retrieval_percentile` column already measures.

Disable with `retrieval_fallback: false`.

---

## How the comparison is made

Averages alone can't answer "can LoRA be used here." Three separate questions get three
separate tools (`src/mindeye_lora/stats.py`):

**Is there a difference?** Arms are scored on the same 1,000 shared NSD test images, so
the tests are paired (Wilcoxon + paired *t*), which is far more powerful than comparing
two group means. p-values are Holm-corrected across arms within each metric.

**How big, and how sure?** BCa bootstrap confidence intervals on the paired mean
difference, plus Cohen's *d_z*.

**Is LoRA good enough?** A non-significant difference is *not* evidence of equivalence.
So we also run TOST (two one-sided tests) against a margin derived from the data: 20% of
the headroom that full fine-tuning creates over the frozen baseline. And we report the
headline number — **retention ratio**:

```
retention = (LoRA − frozen) / (full − frozen)
```

1.0 means LoRA matched full fine-tuning; 0.0 means it did nothing beyond training the
ridge layer. Reported with its own bootstrap interval, because a ratio of two noisy
differences has more uncertainty than people expect.

Three seeds per arm, with between-seed standard deviation reported so you can see the
noise floor. Any between-arm difference smaller than that isn't worth interpreting.

A typical outcome looks like this — and the report is designed to make it legible rather
than hide it:

> `lora_r16` differs from `full` significantly (p_holm = 4e-06) **and** is practically
> equivalent (TOST p = 6e-24), retaining 94% [91%, 96%] of the achievable gain, with
> 0.4% of the trainable parameters.

Both statements are true at once. With 1,000 paired samples you can detect differences
far smaller than anyone cares about, which is exactly why the equivalence test and the
retention ratio are there.

---

## Sizing

| runtime | config | notes |
|---|---|---|
| T4 (free) | `configs/colab_t4.yaml` | `hidden_dim=1024`, batch 24, ~10 GB |
| A100 40GB | `configs/a100_paper_scale.yaml` | `hidden_dim=4096` (~2.1B params), 8-bit AdamW, adds targeting ablations |
| anything | `configs/smoke.yaml` | 3 arms × 10 epochs, validates the pipeline |

### Memory at paper scale

The backbone is lopsided: at `hidden_dim=4096` the single layer mapping 4096 → 256×1664
is ~1.75B of the ~2.1B total. Full fine-tuning is therefore **optimiser-state bound**,
not activation bound — smaller batches barely help.

| arm | optimiser | estimated |
|---|---|---|
| `full` | fp32 AdamW | ~37 GB — will not fit a 40GB A100 |
| `full` | 8-bit AdamW | ~23 GB |
| `lora_r16` | either | ~11 GB |

The same applies one scale down: on a 16 GB T4 at `hidden_dim=1024`, the full arm needs
8-bit AdamW *and* `batch_size` ≤ 8, while LoRA and frozen fit at 12. The shipped configs
use `batch_size: 6, grad_accum: 4` so every arm runs identically at an effective batch of
24.

`capacity.estimate_memory` models the prior's attention explicitly, but its activation
term carries a 5x empirical correction calibrated against a single observed T4 OOM — it
is a guard rail, not a measurement. Real `peak_memory_bytes` per arm goes in the report.

That gap *is* the result. It is also why the paper used an 8×A100-80GB node with
DeepSpeed ZeRO-2, which shards optimiser state across GPUs.

`capacity.py` runs a preflight check before every arm and refuses to start a run it
predicts will OOM (`--ignore-memory-check` to override). These are estimates; the
trainer records real `peak_memory_bytes` per arm, and that is what the report uses.

**`optimizer: adamw8bit` applies to every arm, deliberately.** Using it for only the
full arm would confound the LoRA comparison with an optimiser change.

Memory tricks that make the T4 config fit: CLIP embeddings are precomputed so the 2.5 GB
bigG tower is never resident during training; the low-level VAE branch is disabled; fp16
autocast throughout.

The `a100_paper_scale.yaml` config also adds two targeting ablations — LoRA on the
backbone only vs. the diffusion prior only — which answer a question the headline number
can't: *where* does the adaptation actually need to happen?

---

## Troubleshooting

**`Refusing to run: the 'pretrained' baseline would be fake.`**
Non-ridge parameters had no pretrained value, meaning the architecture we built doesn't
match the checkpoint. Run `cli.py verify --tree` to see the module tree, and pin a known
upstream revision with `--upstream_ref=<sha>`.

**`LoRA targeting matched zero Linear layers.`**
Upstream attribute names changed. `cli.py verify` prints every Linear layer; adjust
`lora_targets` in your config.

**`dalle2_pytorch` import fails.**
It's installed `--no-deps` so it can't downgrade Colab's torch. If the fallback still
fails, `!pip install dalle2-pytorch==1.15.6` in a fresh runtime and restart.

**OOM, or "Arm 'X' probably will not fit".**
The error names a `batch_size` / `grad_accum` pair that should fit — apply it in the
config, keeping the product at 24. Which term dominates depends on the arm: for LoRA and
frozen it is activations (the prior's `[batch, 32 heads, 257, 257]` attention matrices),
so a smaller batch helps a lot; for the full fine-tune it is optimiser state, so
`optimizer: adamw8bit` matters more. On a 16 GB T4 the full arm needs both.

**A CUDA OOM is sticky in notebooks.** The traceback holds every local in every frame,
including the model that just failed, so a retry often OOMs before training starts.
`cmd_train` frees what it can, but if the next attempt fails at `model.to(device)`,
restart the runtime — nothing on Drive is lost.

**"Your session crashed after using all available RAM."**
That is the VM's 12 GB system memory, not the GPU. `predict` streams to disk so this
should not recur; if it does, `--num_workers 0` removes the dataloader worker copies,
and `--batch_size 8` shrinks the per-batch buffers.

**"No data left in file" / EOFError reading a .npy.**
A prediction array was truncated: Drive uploads asynchronously, so the small
`meta.json` can land while the 852 MB arrays beside it do not. `PredictionStore` now
opens every array before declaring a store complete, so a truncated one is detected and
recomputed automatically rather than being skipped forever on the strength of its
metadata. Re-run `predict`.

**Session died mid-training.**
Re-run the same command; nothing needs deleting by hand.

Checkpoints are written to survive a killed VM. Google Drive's mount uploads
asynchronously, so a file that finished writing locally can still be truncated in the
cloud if the session is reclaimed a moment later. Every checkpoint is therefore
fsynced, read back to prove it deserialises, and rotated so the previous good copy
survives as `.bak`. On load, a corrupt primary falls back to the backup; if both are
bad they are removed and the stage restarts cleanly instead of crashing.

Worst case is one checkpoint interval: an epoch for training, five batches for
`predict`. Set `time_budget_min` below your typical session length so runs stop at a
clean boundary rather than being killed mid-write.

**Every slow stage reports progress and checkpoints its work.**

| stage | progress | resume granularity |
|---|---|---|
| `assets` (remote HDF5 fetch) | rows fetched | 256-row chunk |
| `precompute` (CLIP embeddings) | images embedded | ~10 batches |
| `train` | batch within epoch, run within sweep | one epoch |
| `predict` | batch within arm, run within sweep | 5 batches (memmap on disk) |
| `recon` (SDXL decode) | images decoded | ~3 batches |
| `evaluate` | images per metric | whole arm (fast) |
| `compare` | metrics processed | whole stage (fast) |

Nothing needs deleting after a crash: re-run the same command and each stage picks up
from its last checkpoint.

**Memory.** `predict` streams to float16 memmaps under `runs/<run>/predictions/`
rather than accumulating in RAM. Holding 1,000 x 256 x 1664 predictions in memory is
~1.7 GB per tensor in float32, and concatenating several of them exceeds a standard
12 GB Colab VM — which appears as "your session crashed after using all available RAM"
with no traceback. Downstream stages map the store read-only, and metrics normalise in
chunks, so peak memory is independent of test-set size.

**Progress reporting.** Three levels, so you always know where you are:

```
═══ stage 3/8: train ═══ (12.4 min elapsed)
╔═ training runs 5/18 · subj01_1sess_lora_r16_seed0 ═══
║  ~74 min left (5.3 min per training run so far, 14 remaining)
lora_r16 seed0 · epoch 3/150: 41%|████  | 47/114 [01:52<02:39, loss=10.264, lr=8.3e-04]
```

Stage, then run within the sweep, then batch within the run. Runs skipped as already
finished are counted but excluded from the rate, so the ETA reflects real work.

**Is it working, or hung?**
Training and `predict` both show a progress bar with a live ETA. If you need to check
from another cell, `!nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` —
a few MiB of memory means no process holds the GPU at all, i.e. the run died rather than
stalled. Sustained 0% utilisation *with* several GB held means it is blocked on data
loading, not computing.

**`Upstream utils.unclip_recon is unavailable.`**
The decoder path needs upstream's helper. Use `--decoder none` — all statistics still
work.

---

## Layout

```
src/mindeye_lora/
  env.py            Drive workspace + cache redirection
  upstream.py       runtime clone/import of MedARC-AI/MindEyeV2
  assets.py         HF downloads, remote HDF5 slicing, checkpoint slimming
  data.py           datasets, test-repeat averaging, voxel z-scoring
  embeddings.py     CLIP ViT-bigG/14 token embedding cache
  lora.py           LoRALinear, regex targeting, merging, adapter checkpoints
  model_factory.py  architecture inference + strict pretrained loading
  losses.py         BiMixCo / SoftCLIP
  train.py          resumable trainer, one arm per run
  evaluate.py       per-image metrics (CLIP-space + the eight image metrics)
  recon.py          SDXL unCLIP decoding (optional)
  retrieval.py      nearest-neighbour fallback when the decoder is unavailable
  capacity.py       pre-flight memory estimation
  stats.py          bootstrap, paired tests, Holm, TOST, retention ratio
  report.py         figures + REPORT.md
  cli.py            stage orchestration
configs/            colab_t4 · a100_paper_scale · smoke
tests/              test_lora.py · test_stats.py · test_retrieval.py
docs/               EXPERIMENT_DESIGN.md
```

```bash
pytest tests/          # test_stats.py runs without torch
```

---

## Credit and licensing

This is an analysis harness around other people's models. The architecture, checkpoints,
and decoder are from **MindEye2** (Scotti et al., ICML 2024, arXiv:2403.11207) and are
imported from `MedARC-AI/MindEyeV2` at runtime rather than copied. The underlying data is
the **Natural Scenes Dataset** (Allen et al., Nature Neuroscience 2021) — using it
requires agreeing to the [NSD terms and
conditions](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions). Cite MindEye2,
MindEye1, and NSD in anything you publish from this.

Harness code: MIT.
