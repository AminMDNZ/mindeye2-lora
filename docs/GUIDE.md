# Implementation guide

How to run this experiment from nothing, and how to change it. Written for Google Colab,
but nothing here is Colab-specific except the Drive paths.

- [1. What you need](#1-what-you-need)
- [2. First run](#2-first-run)
- [3. What each stage does](#3-what-each-stage-does)
- [4. Changing the experiment](#4-changing-the-experiment)
- [5. Adding a new adaptation method](#5-adding-a-new-adaptation-method)
- [6. Running on your own hardware](#6-running-on-your-own-hardware)
- [7. When something breaks](#7-when-something-breaks)

---

## 1. What you need

**A GPU with at least 12 GB.** A T4 is enough for the main experiment. 24 GB is needed
for `configs/paper_scale.yaml`; `cli assets` refuses to start if the GPU is too small,
before wasting a ~10 GB download.

**About 12 GB of persistent storage** for the data caches, the pretrained checkpoint and
the run outputs. Add ~9 GB if you want the SDXL decoder.

**Agreement to the NSD terms.** The data is distributed under the
[Natural Scenes Dataset terms and conditions](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions);
read and accept them before downloading anything.

No account or token is required — everything is fetched from public HuggingFace repos.

---

## 2. First run

```bash
git clone https://github.com/<you>/mindeye2-lora.git
cd mindeye2-lora
bash setup/colab_setup.sh
pip install bitsandbytes            # needed for optimizer: adamw8bit
```

Then, from Python or the shell:

```bash
python -m mindeye_lora.cli run-all --config configs/smoke.yaml
```

That runs the whole pipeline with three arms for ten epochs — about 30 minutes — and
exists to surface configuration problems before you commit hours. Nothing it downloads is
wasted; the assets and CLIP cache are reused by the real run.

When it finishes and `results/reports/REPORT.html` renders, switch to the experiment:

```bash
python -m mindeye_lora.cli run-all --config configs/main.yaml
```

6 arms × 3 seeds × 150 epochs. Roughly 44 hours on a T4, 10 on an A100. Every stage
checkpoints, so re-running the same command after an interruption resumes rather than
restarting. `cli status` shows where you are.

Generated images are a separate step, because the decoder is an 18 GB download and
~45 s/image:

```bash
python -m mindeye_lora.cli recon --config configs/main.yaml \
    --decoder sdxl_unclip --arm frozen lora_r16 full --n_images 32
python -m mindeye_lora.cli evaluate --config configs/main.yaml --force
python -m mindeye_lora.cli report   --config configs/main.yaml
```

---

## 3. What each stage does

| stage | what it produces | cost |
|---|---|---|
| `assets` | voxel, image and behaviour caches; slimmed pretrained checkpoint | ~20 min once |
| `verify` | confirms the pretrained weights load onto the constructed model | seconds |
| `precompute` | CLIP ViT-bigG/14 token embeddings for every stimulus | ~5 min once |
| `train` | one adapted model per (arm, seed) | the bulk |
| `predict` | test-set CLIP embeddings per run, streamed to disk | ~13 min/arm |
| `recon` | retrieval panel, or generated images with `--decoder sdxl_unclip` | seconds, or ~45 s/image |
| `evaluate` | per-image metrics | ~1 min/arm |
| `compare` | paired statistics | seconds |
| `report` | figures, `REPORT.md`, `REPORT.html` | seconds |

Two of these are worth understanding before you change anything.

**`assets` never downloads the full dataset.** The HuggingFace repo is ~215 GB. A
one-session experiment touches under 2,000 images, so `RemoteH5` opens the 22 GB image
file over HTTPS with fsspec and pulls only the rows it needs — a few hundred MB. Same for
the betas. Transient CDN failures are retried with backoff and the fetch resumes per
256-row chunk.

**`verify` is the guard rail.** It builds the model, loads the shared-subject checkpoint,
and raises if any non-ridge parameter has no pretrained value. Without that check, an
architecture mismatch would silently train from scratch while the report called it
fine-tuning. Run it whenever you change the model code.

---

## 4. Changing the experiment

Almost everything lives in the config. Copy `configs/main.yaml` and edit.

**Different subject or more data.** `subj: 2` and `num_sessions: 4`. The asset stage
re-slices automatically; the pretrained checkpoint is per-held-out-subject, so
`pretrain: multisubject_4096` resolves to `final_multisubject_subj02` for subject 2. Note
`multisubject_1024` exists only for subject 1.

**Different ranks.** Add or edit arms:

```yaml
  - name: lora_r8
    mode: lora
    lora_r: 8
    lora_alpha: 16
    lr: 1.0e-3
    label: "LoRA r=8"
```

Keep `lora_alpha = 2 × lora_r` unless you are deliberately testing the scaling, since
the effective update is scaled by `alpha / r`.

**Where LoRA is applied.** `lora_targets` is a list of regexes matched against the
fully-qualified module path of every `nn.Linear`:

```yaml
    lora_targets: ["^backbone\\."]              # backbone only, prior frozen
    lora_targets: ["^diffusion_prior\\."]       # prior only, backbone frozen
```

`cli verify --tree` prints every Linear layer so you can target precisely. `apply_lora`
raises if a regex matches nothing, rather than silently adapting an empty set.

**Fewer runs.** `seeds: [0, 1]` and a shorter `arms` list. Seeds are the outer loop, so
all arms complete at seed 0 before seed 1 begins — a truncated sweep still gives you a
complete, interpretable result at one seed.

**Memory.** Keep `batch_size × grad_accum = 24` so every arm trains on the same effective
batch. If an arm OOMs, the preflight error names a pair that fits.

---

## 5. Adding a new adaptation method

The comparison is designed so that arms differ **only** in which parameters carry
gradients. To add a method, that is the only thing you should change.

1. Add a branch to `lora.set_trainable`:

   ```python
   elif mode == "my_method":
       unfreeze_named(lambda n, p: <your predicate>)
   ```

2. If it needs new modules, inject them in `train.build_arm_model` before
   `set_trainable` is called, the way `apply_lora` does.

3. Add the arm to your config with `mode: my_method`.

Two invariants to preserve. The **ridge layer is always fully trained** — it is randomly
initialised for a held-out subject, so there is no pretrained weight for a constrained
update to be constrained relative to; `always_trainable` handles this and you should not
override it. And any new module must be **zero-initialised at the output**, so the model
at step 0 is numerically identical to the pretrained one. `tests/test_lora.py::
test_zero_init_is_identity` checks this for LoRA; write the equivalent for your method.

---

## 6. Running on your own hardware

Nothing depends on Colab except the default paths. Set the workspace explicitly:

```bash
export MINDEYE_LORA_ROOT=/data/mindeye2_lora
python -m mindeye_lora.cli run-all --config configs/main.yaml --no-drive
```

`--no-drive` skips the Drive mount. The local data mirror (which exists because Drive is
slow at random reads) is a no-op off Colab.

On a machine with 24 GB or more, `configs/paper_scale.yaml` runs the same comparison at
`hidden_dim=4096` (~2.1B parameters), where the memory and checkpoint-size differences
between LoRA and full fine-tuning are far larger.

---

## 7. When something breaks

| symptom | cause and fix |
|---|---|
| `Refusing to run: the 'pretrained' baseline would be fake` | upstream's architecture no longer matches the checkpoint. Do not work around it; pin a revision with `--upstream_ref=<sha>` |
| `LoRA targeting matched zero Linear layers` | upstream attribute names changed; `cli verify --tree` lists them |
| "session crashed after using all available RAM" | system RAM, not GPU. Try `--num_workers 0` |
| CUDA OOM | apply the `batch_size`/`grad_accum` pair from the error, then **restart the process** — a CUDA OOM is sticky, the traceback pins the failed model |
| `needs roughly 24 GB of VRAM` | use `configs/main.yaml` |
| missing module | re-run `setup/colab_setup.sh`; it resolves dependencies by following the import chain |
| a metric is `nan` | check `seed_variability` in the report — an arm whose seeds disagree wildly usually has a corrupt prediction store |

**Nothing needs deleting after a crash.** Checkpoints are fsynced, verified on read and
kept in two generations; a corrupt one falls back to its backup, and if both are bad they
are removed and that stage restarts. Prediction stores are shape-checked before being
trusted as complete — a truncated store is detected and recomputed rather than silently
reused. That last check exists because it happened: a truncated store scored an arm below
chance and, because that arm is the denominator of every retention ratio, inflated the
apparent headroom thirteen-fold before it was caught.

If you edit anything under `src/`, **restart the Python process**. Copying a file into
place does not reload an already-imported module, and the resulting errors point at line
numbers in the new file while the old code executes.
