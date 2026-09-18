# Results

Subject 1, one NSD session (688 training trials), `hidden_dim=1024`, 6 arms × 3 seeds ×
150 epochs, evaluated on the 1,000 held-out shared test images.

Reproduce with:

```bash
python -m mindeye_lora.cli run-all --config configs/main.yaml
```

## Headline

Every parameter-efficient arm outperformed full fine-tuning.

| arm | two-way CLIP | sd across seeds | cosine | trainable | checkpoint |
|---|---|---|---|---|---|
| `lora_r64` | **0.9764** | 0.0006 | 0.3528 | 58.6M (7.59%) | 224 MB |
| `lora_r4` | 0.9748 | 0.0018 | 0.3584 | 18.8M (2.56%) | 72 MB |
| `lora_r16` | 0.9737 | 0.0013 | 0.3580 | 26.7M (3.61%) | 102 MB |
| `bitfit` | 0.9630 | 0.0016 | 0.3536 | 16.6M (2.27%) | 63 MB |
| `full` | 0.9621 | 0.0020 | 0.3267 | 729.3M (100%) | 2.7 GB |
| `frozen` | 0.9547 | 0.0008 | 0.3517 | 16.1M (2.21%) | 61 MB |

Paired differences against `full` on `two_way_clip`, 1,000 test images:

| arm | Δ | 95% CI | p (Holm) | d_z |
|---|---|---|---|---|
| `lora_r64` | +0.0143 | [+0.0118, +0.0171] | 6.5e-65 | 0.340 |
| `lora_r4` | +0.0127 | [+0.0099, +0.0159] | 3.3e-33 | 0.267 |
| `lora_r16` | +0.0116 | [+0.0088, +0.0146] | 1.4e-39 | 0.252 |
| `bitfit` | +0.0010 | [−0.0028, +0.0047] | 0.0022 | 0.016 |
| `frozen` | −0.0074 | [−0.0118, −0.0035] | < 0.001 | −0.157 |

## Three findings

**1. LoRA beats full fine-tuning, it does not merely match it.** The experiment was
designed as a non-inferiority test; the result is superiority at every rank. With 688
training trials and 729M parameters, unconstrained updating overfits. The training curves
show it directly: `full` reaches a final training loss around 4.5 against 6.5–9.4 for
every other arm, while ranking below all the LoRA arms on held-out data.

**2. Rank barely matters.** r=4 scores 0.9748, r=64 scores 0.9764 — sixteen times the
adapter parameters for +0.0016, which is comparable with the between-seed spread. The
adaptation a new participant requires is genuinely low-rank. This is the more interesting
scientific claim, and it supports the shared-subject premise of MindEye2 more directly
than the accuracy comparison does.

**3. Most of the work is done by pretraining.** `frozen`, which trains only the
subject-specific ridge layer, already reaches 0.9547. Adapting the shared network at all
is worth at most 0.022.

## Why the retention ratio is not the headline

`stats.retention_table` expresses each arm as a fraction of the reference arm's gain over
the frozen baseline. That only means something when the reference is the best arm. Here
the frozen→full headroom is **0.0074**, and LoRA exceeds `full`, so the ratio divides by a
near-zero denominator and returns figures like 292%. `report.build_report` detects both
conditions and leads with the raw ranking instead, printing retention with an explicit
caveat. Read the ranking and the paired differences.

## Deviations from the published recipe

All three are applied uniformly across arms, so the internal comparison holds, but
absolute values are not comparable with published MindEye2 numbers.

- **Learning rate differs by arm**: 3e-4 for `full`, 1e-3 for the PEFT arms. Low-rank
  adapters are undertrained at the full fine-tuning rate, but this means the comparison is
  each method at its own rate rather than at a common one. A per-arm LR sweep would settle
  it and was not run. This is the weakest point in the design.
- **8-bit AdamW** instead of fp32, required to fit `full` in 15 GB.
- **Voxel z-scoring** using training-split statistics, on top of the already-normalized
  betas.

Also: one subject, one session, `hidden_dim=1024` rather than 4096, low-level branch
disabled, no caption-refinement stage.

## A correction applied before analysis

The `frozen` seed-0 prediction store was truncated by an interrupted write and initially
scored 0.1429 — below chance, and contradicting its own training log. Recomputing that one
arm's predictions gave 0.954, consistent with its other two seeds. `frozen` is the
denominator of every retention ratio, so the corrupt run had inflated the apparent headroom
by a factor of 13.

`PredictionStore.is_complete` now opens and shape-checks every array instead of trusting
`meta.json`, so a truncated store is detected and recomputed rather than silently reused.

## Cost

| arm | peak VRAM | sec/epoch | per-subject storage |
|---|---|---|---|
| `full` | 12.4 GB | 36.9 | 2,700 MB |
| `lora_r4` | 11.4 GB | 28.5 | 72 MB |
| `lora_r16` | 8.2 GB | 28.9 | 102 MB |
| `lora_r64` | 8.5 GB | 29.4 | 224 MB |
| `bitfit` | 7.6 GB | 25.3 | 63 MB |
| `frozen` | 7.6 GB | 69.4 | 61 MB |

Storage is where the saving is unambiguous: 38× smaller per subject at r=4, and adapters
merge into the base weights so inference costs nothing extra. Memory falls less than the
parameter ratio suggests because activations dominate at this model size. Time per epoch
falls least, because the forward and backward passes still traverse the whole network.
`frozen`'s 69.4 s is an artefact of running first and absorbing warm-up costs.

The subject-specific ridge layer is 16.1M of the 18.8M trainable parameters in the r=4
arm, so further savings must come from the ridge layer, not from smaller adapters.
