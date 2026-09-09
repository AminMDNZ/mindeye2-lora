# Experimental design

This document records *why* the experiment is set up the way it is, including the
choices that would be easy to get wrong and the ways the result could still mislead you.

---

## 1. The question

MindEye2's central claim is that pretraining on seven subjects lets you adapt to an
eighth with about one hour of scanning. That adaptation is currently a full fine-tune of
the entire shared model.

Two things make that expensive in practice:

- **Storage per subject.** A clinical or consumer deployment would need one full
  checkpoint per person. At `hidden_dim=4096` that is a multi-gigabyte artifact each.
- **Compute per subject.** Full fine-tuning holds optimiser state for every parameter,
  which is what pushes the paper's recipe onto A100-class hardware.

If a low-rank update recovers the same performance, both costs collapse: subjects become
a few megabytes of adapter, and the training fits on consumer hardware. That is the
practical payoff, and it is what makes the *equivalence* question more interesting than
the *difference* question.

---

## 2. What is adapted, and what isn't

| component | pretrained? | trainable in LoRA arm | why |
|---|---|---|---|
| `ridge` (subject-specific linear) | **no** — new subject, new voxel count | **fully** | Randomly initialised. There is no pretrained weight for a low-rank update to be low-rank *relative to*. |
| `backbone` (residual MLP) | yes | LoRA adapters | Shared across subjects; the main target. |
| `diffusion_prior` (DALLE-2 style) | yes | LoRA adapters | Shared; maps backbone output to CLIP space. |
| SDXL unCLIP decoder | yes | **never** | Frozen in every arm, including `full`. Contributes no between-arm variance. |
| CLIP ViT-bigG/14 tower | yes | **never** | Provides the regression target; precomputed. |

This split is the single most important design decision. If you applied LoRA to the
ridge layer too, the LoRA arm would be crippled for reasons that have nothing to do with
low-rank adaptation, and you would wrongly conclude LoRA doesn't work here.

Enforcement: `model_factory.load_pretrained_weights` raises if any non-ridge parameter
is missing from the checkpoint, so a silent architecture mismatch cannot masquerade as a
"pretrained" baseline.

---

## 3. Controls

Everything except the trainable-parameter set is held fixed across arms:

- same 1-session training split and same 1,000-image test split
- same data order (seeded `DataLoader` generator and worker seeds)
- same epochs, batch size, gradient accumulation, warmup, and OneCycle schedule shape
- same loss: BiMixCo for the first 33% of epochs, then SoftCLIP with the annealed
  temperature from the paper
- same starting weights — LoRA's `B` matrix is zero-initialised, so an injected model is
  *numerically identical* to the pretrained one at step 0 (verified by
  `tests/test_lora.py::test_zero_init_is_identity`)

**One deliberate exception: learning rate.** PEFT arms use `1e-3` rather than the
paper's `3e-4`. Low-rank adapters have far fewer effective degrees of freedom and are
routinely undertrained at full-fine-tuning learning rates; holding LR identical would
stack the deck against LoRA. This is a judgement call and it is a real confound — if you
want a strictly LR-matched comparison, set every arm's `lr` to the same value in the
config. Better still, run a small LR sweep per arm and report each at its own best
setting; the harness supports this by simply adding more arms.

---

## 4. Metrics

**Primary (no decoder needed).** These measure the brain→CLIP mapping, which is the only
thing any arm modifies:

- `cosine` — similarity between predicted and true CLIP token embeddings
- `two_way_clip` — per-image probability the true image outranks a random distractor

An earlier version also reported `retrieval_percentile` (the rank of the correct image
among all test images). It is **algebraically identical** to `two_way_clip`: both equal
the fraction of distractors the true target outranks, and they agree to floating-point
precision on every sample. Reporting both made a single result look like two independent
confirmations, so the default metric set now excludes it and `compare_against_reference`
drops it if present. The lesson generalises: check that your metrics are actually
measuring different things before treating their agreement as evidence.

**Secondary (needs reconstructions).** The eight metrics from MindEye1/2: PixCorr, SSIM,
AlexNet-2, AlexNet-5, Inception, CLIP, EffNet-B, SwAV. Two-way metrics use the published
protocol — compare the reconstruction's features to the true stimulus versus every other
test stimulus.

Every metric returns a **per-image vector**, not a mean. That is what makes the paired
analysis possible.

Test-set betas are averaged over the three repetitions of each shared image, matching
the paper's evaluation protocol.

---

## 5. Statistics

### Pairing
Arms are evaluated on identical images, so differences are computed per image. Paired
tests remove between-image variance, which dominates: some stimuli are simply easier to
decode than others, for every arm.

### Difference testing
Wilcoxon signed-rank (primary, no normality assumption) and paired *t* (reported for
comparability). Holm–Bonferroni correction across the arms compared within each metric —
step-down, so it controls family-wise error without Bonferroni's conservatism.

### Interval estimation
BCa bootstrap on the paired mean difference, resampling image pairs. Bias-corrected and
accelerated rather than plain percentile, because paired-difference distributions on
bounded metrics (two-way accuracy lives in [0, 1]) are frequently skewed.

### Equivalence
This is the part most published comparisons skip. **A non-significant difference is not
evidence of equivalence** — it is compatible with "the effect is zero" and with "the
study was underpowered." TOST tests the opposite null: that the difference is *at least*
as large as a margin δ. Rejecting it is positive evidence that the gap is bounded.

Margin choice: δ = 20% of the frozen→full headroom, computed from the data. Declaring
equivalence then means "LoRA is within 20% of whatever gain full fine-tuning was able to
buy," which is a statement about practical relevance rather than about arbitrary metric
units. Change it with `--equivalence_fraction`. **Pre-register this number before you
look at the results**; picking it afterward is the equivalence-testing equivalent of
p-hacking.

### Retention ratio
```
retention = (arm − frozen) / (full − frozen)
```
The headline number, with a bootstrap CI. It answers "how much of the achievable
adaptation did this method capture," which is what you actually want to know, and it
normalises away the metric's arbitrary scale.

### Seeds
Three per arm. Per-image scores are averaged across seeds for the primary analysis
(reducing seed noise while preserving pairing across images), and between-seed standard
deviation is reported separately as the noise floor.

---

## 6. Expected outcomes and how to read them

**If retention ≈ 1.0 and TOST says equivalent:** LoRA works here. Report the parameter
and storage savings; that is the contribution.

**If retention is high but rank barely matters (r=4 ≈ r=64):** the subject-adaptation
signal is genuinely low-rank. This is the most interesting scientific outcome — it says
something about what changes between brains, not just about training efficiency.

**If `frozen` already scores close to `full`:** the headroom is small and *neither*
method has room to differ. Check the `frozen→full headroom` column before interpreting
anything else. A retention ratio computed over a tiny denominator is noise. This is a
real possibility given that the shared-subject model was pretrained on seven people.

**If LoRA underperforms at every rank:** check the LR confound first (§3), then the
targeting ablations — it may be that the diffusion prior needs full adaptation while the
backbone doesn't, which the `a100_paper_scale.yaml` config tests directly.

---

## 7. Threats to validity

| threat | mitigation | residual risk |
|---|---|---|
| Learning rate confound between arms | documented; configurable; sweep supported | real — the honest fix is a per-arm sweep |
| Single subject (subj01) | subject is configurable | conclusions may not transfer across subjects; run subj02/05/07 to check |
| Single data budget (1 session) | `num_sessions` configurable | LoRA may behave differently with 40 sessions, where there's more to learn |
| `hidden_dim=1024` instead of 4096 | upstream reports minimal impact; 4096 config provided | LoRA's advantage should *grow* at 4096, so the T4 result is conservative |
| CLIP embedding convention | `verify` stage + published reference checkpoint as a sanity check | if the convention were wrong, all arms would fail together and be obviously broken |
| Multiple comparisons | Holm within metric family | comparing across many metrics still invites cherry-picking — declare your primary metric first |
| Reconstruction metrics on a prefix | image metrics computed on the first N images, padded with NaN and masked before pooling | reduced N means wider CIs on image metrics than on CLIP-space metrics |
| Retrieval panel read as generation | figure title, row labels and report caption all state it explicitly; written to `retrieval.pt` so `evaluate.py` cannot score it as a reconstruction | a screenshot separated from its caption still misleads |
| Retrieval panel read as evidence of no difference | top-3 shown with true-rank annotations; caption points readers to the statistics tables | retrieval is a hit-or-miss threshold and cannot resolve a few-percent gap |
| Optimiser choice at paper scale | `optimizer` is a global setting applied to all arms | 8-bit moments add quantisation noise to every arm equally, but that is still a deviation from the paper's fp32 AdamW |

---

## 8. Reproducing

```bash
python -m mindeye_lora.cli run-all --config configs/colab_t4.yaml
```

Recorded for reproducibility in each run directory: the resolved config, the upstream
git SHA, the inferred architecture, the checkpoint load report, per-arm trainable
parameter counts, and the full training log.

---

## References

- Scotti et al. (2024). *MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour
  of Data.* ICML. arXiv:2403.11207
- Scotti et al. (2023). *Reconstructing the Mind's Eye.* NeurIPS 36. arXiv:2305.18274
- Allen et al. (2021). *A massive 7T fMRI dataset...* Nature Neuroscience.
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Liu et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation.* arXiv:2402.09353
- Lakens (2017). *Equivalence tests.* Social Psychological and Personality Science.
