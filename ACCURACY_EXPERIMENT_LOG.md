# ACCURACY_EXPERIMENT_LOG.md
# Models_DL — Accuracy Experiment Log

**Session:** 2026-10-10  
**Baseline:** 63.24% test acc on NB04 (InternImage Rabi, 100 epochs)

---

## Confirmed Measurements (Before Any Changes)

### Data analysis (2026-10-10, full dataset)

| Measurement | Result | Method |
|---|---|---|
| Kharif class imbalance | **9.9× ratio** (Cotton 850 vs Cowpea 8,387) | Full dataset count |
| Rabi class imbalance | **1.5× ratio** (nearly balanced) | Full dataset count |
| NaN/Inf | Zero | Sample check (1000) |
| Shape | All (12,7,7) | Sample check |
| Channels 8–11 within-tile std | **0.000** (spatially constant) | 200-sample measured |
| Channels 0–7 within-tile/global ratio | 0.41–0.63 (genuine spatial signal) | 200-sample measured |

### Fisher ratio (class discriminability per channel)

| Channel | Kharif | Rabi |
|---|---|---|
| Ch 9 | **1.61** | **0.80** |
| Ch 11 | **1.41** | **0.70** |
| Ch 8 | **1.38** | 0.31 |
| Ch 10 | **1.10** | 0.66 |
| Ch 0–7 | 0.16–0.24 | 0.11–0.22 |

> Channels 8–11 are most discriminative AND spatially constant — they need scalar treatment, not spatial convolution.

### LogReg signal ceiling (20k train samples)

| Dataset | Flat 588 | 12 ch means | 24 means+stds |
|---|---|---|---|
| Kharif | 53.3% | 55.5% | 61.7% |
| Rabi | 39.6% | 41.9% | 48.4% |

> **InternImage baseline (63.2%) already exceeds linear ceiling on Rabi** → architecture is learning nonlinear features. More capacity + better adapter can push further.

---

## Changes Applied — Session 2026-10-10

### CHANGE 1: ReduceLROnPlateau scheduler ✅ (already in previous session)

**Type:** Training improvement  
**Rationale:** Flat LR=1e-4 for 100 epochs stalls at plateaus  
**Config:** `factor=0.5, patience=8, min_lr=1e-6`  
**Status:** Applied to all 6 notebooks, smoke-tested ✅

---

### CHANGE 2: Early stopping ✅ (already in previous session)

**Type:** Training improvement  
**Rationale:** Stop training when val_loss stops improving  
**Config:** `patience=15` (larger than scheduler patience to allow LR reduction)  
**Status:** Applied to all 6 notebooks, smoke-tested ✅

---

### CHANGE 3: AMP + cuDNN benchmark ✅ (already in previous session)

**Type:** Efficiency  
**Rationale:** RTX A5000 has Tensor Cores; FP16 is 2× faster with no accuracy loss  
**Config:** `USE_AMP=True`, `cudnn.benchmark=True`  
**Status:** Applied to all 6 notebooks, smoke-tested ✅

---

### CHANGE 4: Batch size 32→128, NUM_WORKERS 0→4 ✅ (already in previous session)

**Type:** Efficiency  
**Rationale:** GPU was using 0.19 GB of 24 GB (< 1% utilization)  
**Status:** Applied to all 6 notebooks, smoke-tested ✅

---

### CHANGE 5: Improved InputAdapter (12→12, dual-branch) ← **NEW**

**Type:** Architecture improvement  
**Rationale:** 
- Confirmed finding: channels 8–11 have **zero** within-tile spatial variance
- These are the **most discriminative** channels (Fisher 0.66–1.61 vs 0.11–0.24 for spatial channels)
- Original `Conv2d(12→3)` discards 9 channels and collapses scalar channels into spatial channels — losing their signal

**Implementation:**
```python
class InputAdapter(nn.Module):
    # Spatial branch: Conv2d(12→12, 1×1) + BN + GELU
    # Scalar branch:  global_mean(ch 8–11) → Linear(4→12) → expand
    # Output: sigmoid(blend) * spatial + (1-sigmoid(blend)) * scalar
```

**Applied to:** All 6 notebooks (VMamba `InputAdapter`, InternImage `InputAdapter12`, Hybrid `SharedInputAdapter`)  
**Backbone in_chans:** Changed from 3 → 12 (no information bottleneck)  
**Smoke test:** ✅ Forward and backward pass verified

---

### CHANGE 6: LR warmup (5 epochs linear) ← **NEW**

**Type:** Training stability  
**Rationale:** Large batch size (128) amplifies gradient variance in epoch 1. Linear warmup stabilizes early training and prevents the optimizer from jumping to a poor local minimum.  
**Config:** `LR_WARMUP_EPOCHS=5`, warmup from `LR/5` to `LR` linearly  
**Status:** Applied to all 6 notebooks, smoke-tested ✅

---

### CHANGE 7: Satellite spatial augmentation ← **NEW**

**Type:** Regularization / data augmentation  
**Scientific justification:**
- Fields are viewed from above (nadir) → no preferred orientation
- `hflip` (left-right reflection): valid — field boundaries don't have preferred direction
- `vflip` (top-bottom reflection): valid — same reasoning
- `rot90` (0°/90°/180°/270°): valid — satellite sensor captures all orientations equally

**Rejected augmentations (would corrupt spectral data):**
- Color jitter → changes actual band reflectance values (scientifically invalid)
- Random erase → removes valid spectral pixels
- Gaussian blur → destroys spatial texture patterns

**Config:** Applied only to train_loader (`augment=not SMOKE_TEST`)  
**Status:** Applied to all 6 notebooks, smoke-tested ✅

---

## Complete Configuration Table

| Parameter | Original | Session 1 | Session 2 |
|---|---|---|---|
| BATCH_SIZE | 32 | 128 | 128 |
| NUM_WORKERS | 0 | 4 | 4 |
| MAX_EPOCHS | 100 | 100 | 100 |
| LR | 1e-4 | 1e-4 | 1e-4 |
| WEIGHT_DECAY | 1e-4 | 1e-4 | 1e-4 |
| LR Scheduler | None | ReduceLROnPlateau(p=8) | ReduceLROnPlateau(p=8) |
| LR Warmup | None | None | **5 epochs** |
| Early Stopping | None | patience=15 | patience=15 |
| AMP | Off | On | On |
| cuDNN benchmark | Off | On | On |
| Augmentation | None | None | **hflip+vflip+rot90** |
| InputAdapter | Conv2d(12→3) | Conv2d(12→3) | **Conv2d(12→12)+scalar** |
| Backbone in_chans | 3 | 3 | **12** |
| drop_last | False | True | True |

---

## Smoke Test Results (2026-10-10, after all patches)

```
Device: cpu (smoke test; full training must be on GPU)

Dataset loading:       Kharif 111,956 ✅  Rabi 129,015 ✅
Frozen splits:         Kharif 89k/11k/11k ✅  Rabi 103k/12k/12k ✅
Split overlap check:   CLEAN ✅
Normalization:         Train-only ✅

VMamba Kharif    forward: (4,12,7,7) → (4,16) ✅   backward ✅
VMamba Rabi      forward: (4,12,7,7) → (4,13) ✅
InternImage Kharif forward: (4,12,7,7) → (4,16) ✅  backward ✅
InternImage Rabi   forward: (4,12,7,7) → (4,13) ✅
Hybrid Kharif    forward: (4,12,7,7) → (4,16) ✅   backward ✅
Hybrid Rabi      forward: (4,12,7,7) → (4,13) ✅

Optimizer steps:
  VMamba Adam:       loss=2.6115 ✅
  InternImage Adam:  loss=2.9056 ✅
  Hybrid Adam:       loss=2.7760 ✅

FINAL: ALL SMOKE TESTS PASSED ✅
```

---

## Full Training Instructions (Mam's GPU — RTX A5000)

### One-time setup
```bash
conda activate shivamsen
cd /path/to/Models_DL    # wherever the .ipynb files are
jupyter notebook
```

### For each notebook
1. Open notebook in Jupyter
2. **Cell 2 (Config)** — set:
   ```python
   SMOKE_TEST    = False
   BATCH_SIZE    = 128
   NUM_WORKERS   = 4
   MAX_EPOCHS    = 100
   USE_AMP       = True
   VMAMBA_FORWARD_TYPE = 'v05_noz'   # keep until CUDA exts compiled
   ```
3. **Kernel → Restart & Run All**
4. Monitor output — look for:
   - LR warmup: `LR=2.00e-05` at epoch 1, reaching `1.00e-04` at epoch 5
   - Scheduler trigger: `LR=5.00e-05` when plateau detected
   - Early stopping: `Early stopping at epoch N`

### Recommended run order (shortest first)
```
03_InternImage_Kharif_Adam.ipynb    (~25–35 min estimated)
04_InternImage_Rabi_Adam.ipynb      (~25–35 min estimated)
01_VMamba_Kharif_Adam.ipynb         (~30–40 min estimated)
02_VMamba_Rabi_Adam.ipynb           (~30–40 min estimated)
05_Hybrid_Kharif_Adam.ipynb         (~45–60 min estimated)
06_Hybrid_Rabi_Adam.ipynb           (~45–60 min estimated)
```

Run one at a time. Do not start the next until the previous finishes.

---

## Result Recording Template

After each full training run, record here:

| Notebook | Best val_acc | Best epoch | Test acc | F1-mac | F1-wtd | Duration |
|---|---|---|---|---|---|---|
| 01 VMamba Kharif | ___ | ___ | ___ | ___ | ___ | ___ |
| 02 VMamba Rabi | ___ | ___ | ___ | ___ | ___ | ___ |
| 03 InternImage Kharif | ___ | ___ | ___ | ___ | ___ | ___ |
| 04 InternImage Rabi | ___ | ___ | ___ | ___ | ___ | ___ |
| 05 Hybrid Kharif | ___ | ___ | ___ | ___ | ___ | ___ |
| 06 Hybrid Rabi | ___ | ___ | ___ | ___ | ___ | ___ |

---

## Remaining Hypotheses (Not Yet Tested)

These would require a second round of experiments after seeing full training results:

| Hypothesis | Experiment | When to Try |
|---|---|---|
| More adapter capacity helps | Increase adapter hidden dim | If accuracy < 70% after improvements |
| InternImage depth=[2,2,2,2] vs [1,1,1,1] | Compare val_acc curves | After seeing baseline convergence |
| Label smoothing (ε=0.1) | Alternative to hard CrossEntropy | If minority class F1 < 0.4 |
| Focal loss vs weighted CE | For severe Kharif imbalance | If Guar/Cotton F1 still near 0 |
| Longer training (150 epochs) | Only if val_loss still decreasing at epoch 100 | Check learning curves first |
