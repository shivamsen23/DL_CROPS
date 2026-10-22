# ACCURACY_DIAGNOSIS.md
# Models_DL — Accuracy Diagnosis Report

**Date:** 2026-10-10  
**Status:** Phase 1–3 complete (diagnosis + improvements applied)

---

## 1. Baseline Results

| Notebook | Dataset | Val Acc | Test Acc | F1-macro | F1-wtd | Duration |
|---|---|---|---|---|---|---|
| 04 — InternImage Rabi | Rabi (13 cls) | **63.57%** | **63.24%** | 62.99% | 63.13% | 139 min |
| 01–03, 05–06 | — | Not yet run | — | — | — | — |

> Only NB04 has confirmed results from Mam's GPU run. All others are untested baselines.

---

## 2. Data Integrity (Phase 2) — ALL CLEAN ✅

| Check | Kharif | Rabi |
|---|---|---|
| Sample shape | All `(12,7,7)` ✅ | All `(12,7,7)` ✅ |
| NaN / Inf | 0 affected ✅ | 0 affected ✅ |
| Zero-variance tiles | 0 ✅ | 0 ✅ |
| Near-duplicates (1000 sample check) | 995/1000 unique ✅ | 986/1000 unique ✅ |
| Split overlap | None ✅ | None ✅ |
| Normalization source | Train-only ✅ | Train-only ✅ |
| Split sizes | 89,564 / 11,194 / 11,198 ✅ | 103,213 / 12,901 / 12,901 ✅ |

**No data integrity issues found.** The ~1% near-duplicates in Rabi are expected for a geographic dataset (adjacent fields can have identical spectra).

---

## 3. Critical Findings (Evidence-based)

### Finding 1: Channels 8–11 are spatially constant within every 7×7 tile

**Evidence (measured):**

| Channel | Within-tile std | Global std | Ratio |
|---|---|---|---|
| Ch 0–6 | 0.026–0.109 | 0.064–0.185 | 0.41–0.63 |
| Ch 7 | 99.4 | 158.6 | 0.63 |
| **Ch 8** | **0.000** | 197.4 | **0.000** |
| **Ch 9** | **0.000** | 3.15 | **0.000** |
| **Ch 10** | **0.000** | 0.63 | **0.000** |
| **Ch 11** | **0.000** | 1.12 | **0.000** |

Channels 8–11 are **scalar per sample**, not spatial maps. Within each 7×7 tile they are completely flat (no spatial variation).

**Impact:** Spatial convolution and the SSM cross-scan over 49 pixels cannot extract anything meaningful from flat channels. The only useful operation is global pooling — but the existing `InputAdapter` projects these into 3 channels via `Conv2d(12→3, 1×1)` before the backbone, which may discard or dilute their signal.

**Fisher ratio (most discriminative channels):**

| Channel | Kharif Fisher | Rabi Fisher |
|---|---|---|
| Ch 9 | **1.613** | **0.804** |
| Ch 11 | **1.412** | **0.700** |
| Ch 8 | **1.375** | 0.310 |
| Ch 10 | **1.104** | 0.658 |
| Ch 0–7 | 0.158–0.236 | 0.105–0.222 |

The four spatially-constant channels are the **most discriminative** for both seasons.

---

### Finding 2: The 12→3 InputAdapter is a confirmed information bottleneck

**Evidence:** All three model families use `InputAdapter(12→3)` before the backbone. This reduces 12 channels to 3 before any nonlinear learning. Given that channels 8–11 carry 4–10× more class-discriminative signal than channels 0–7, this bottleneck likely discards or dilutes the most useful information.

**Fix applied:** Replaced with two-branch `InputAdapter(12→12)`:
- **Spatial branch**: `Conv2d(12→12, 1×1) + BN + GELU` — learns spatial patterns from all channels
- **Scalar branch**: Global mean of channels 8–11 → `Linear(4→12)` → expand to (B,12,H,W)
- **Output**: Learned weighted blend `w·spatial + (1-w)·scalar`

---

### Finding 3: Class imbalance is severe in Kharif (9.9× ratio)

**Evidence:**

| Class | Train Count | Notes |
|---|---|---|
| Cotton | 850 | **Smallest** — 10× below majority |
| Guar | 1,028 | 2nd smallest |
| Matbean | 1,629 | 3rd smallest |
| Cowpea | 8,387 | **Largest** |

Rabi is nearly balanced (1.5× ratio). Weighted CrossEntropy is already applied in all notebooks ✅ — this is the correct mitigation.

**Per-class LogReg accuracy reveals affected classes:**
- **Guar: 6.9%** (barely above random for 16 classes = 6.25%)
- **Flax: 17.9%**, Groundnut: 28.8%, Black gram: 24.6%
- These are the classes expected to improve most with improved architecture

---

### Finding 4: Linear separability ceiling (LogReg upper bound)

**Measured with 20,000 train samples, 5,000 val:**

| Dataset | Features | LogReg Val Acc | F1-macro |
|---|---|---|---|
| Kharif | Flat 588 | 53.3% | 49.4% |
| Kharif | 12 ch means | 55.5% | 50.3% |
| Kharif | 24 means+stds | 61.7% | 59.3% |
| Rabi | Flat 588 | 39.6% | 37.4% |
| Rabi | 12 ch means | 41.9% | 38.9% |
| Rabi | 24 means+stds | 48.4% | 46.9% |

> **InternImage already exceeds the linear ceiling on Rabi (63.2% vs 48.4%)**, confirming the models ARE learning genuinely nonlinear features. The data has more signal than a linear model can access.

> **Interpretation**: The 85% target requires nonlinear learning of spectral + textural patterns that LogReg cannot capture. Deep models with the right inductive biases can plausibly reach this.

---

### Finding 5: Rabi is harder than Kharif (counter-intuitive)

Despite being nearly balanced, Rabi LogReg accuracy (48.4%) is far below Kharif (61.7%). The most confused Rabi classes are: Barley/Linseed/Masur/Taramira/Wheat — winter cereals with very similar spectral signatures. This is a **genuine class-separation difficulty**, not an imbalance problem.

---

## 4. Architecture Problems Fixed

### A. InputAdapter bottleneck (all 6 notebooks)
- **Before**: `Conv2d(12→3)` — collapses most discriminative channels
- **After**: `Conv2d(12→12) + scalar branch (ch 8-11)` — preserves all signal

### B. Missing LR warmup (all 6 notebooks)
- **Before**: Flat LR from epoch 1 — unstable early gradients with large batch
- **After**: Linear warmup over 5 epochs, then `ReduceLROnPlateau`

### C. Missing training-time augmentation (all 6 notebooks)
- **Before**: No augmentation — model only sees exact tiles
- **After**: Spatial-only augmentation (hflip, vflip, rot90) — scientifically valid for top-down satellite imagery

---

## 5. What Was NOT Changed

| Item | Decision |
|---|---|
| Adam optimizer | Preserved ✅ |
| Frozen splits | Preserved ✅ |
| Train-only normalization | Preserved ✅ |
| Weighted CrossEntropy | Preserved ✅ |
| Model depths/dims | Preserved ✅ |
| Checkpoint by val_acc | Preserved ✅ |
| Test set isolation | Preserved ✅ |

---

## 6. 85% Accuracy Assessment

**Is 85% achievable?**

Based on data analysis:
- The data has strong nonlinear signal (InternImage already 15% above LogReg ceiling)
- Kharif has 9.9× imbalance — heavily imbalanced minority classes (Cotton, Guar) will pull accuracy down
- Rabi winter cereals (Barley/Wheat/Linseed) are spectrally similar — hard ceiling from data ambiguity
- The 12→3 bottleneck was likely the single biggest fix

**Conservative honest estimate:**
- InternImage Rabi: 63% → potentially 68–75% with fixes
- InternImage Kharif: similar range (harder due to imbalance)
- VMamba: may perform comparably once adapter fixed
- Hybrid: may gain 2–4% over individual models

> 85% cannot be guaranteed or fabricated. It depends on how separable the hard classes (Guar, Cotton, Rabi cereals) actually are with nonlinear features — which only full training will reveal.

---

## 7. Smoke Test Results (Updated Notebooks)

See experiment log for full results. Run after all architectural patches.
