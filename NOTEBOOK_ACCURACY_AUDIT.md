# NOTEBOOK_ACCURACY_AUDIT.md
# Models_DL — Accuracy Improvement Audit Report

**Date:** 2026-10-10 | **Auditor:** Antigravity  
**Baseline backed up to:** `Models_DL/BACKUP_baseline/`

---

## 1. Confirmed Findings from Code Inspection

### A. Correctness — NO critical bugs found ✅

| Check | Finding |
|---|---|
| Normalization from train-only | ✅ Confirmed correct (`norm_mean/std` from `train_idx` only) |
| Split overlap | ✅ `assert len(set(train_idx) & set(val_idx)) == 0` present |
| Checkpoint selection criterion | ✅ Best checkpoint selected by `val_acc` (not test) |
| Test set never used in training | ✅ Confirmed — test loop only in Eval cell |
| Label-to-class mapping | ✅ `classes = sorted(raw.keys())` → deterministic ordering |
| Class weights | ✅ `compute_class_weight('balanced', ...)` on train labels only |
| DataLoader normalization consistency | ✅ Same `norm_mean/std` applied in all three splits |
| Hybrid gradient flow | ✅ No detach anywhere in forward pass |
| Hybrid fusion | ✅ Feature-level (pre-classifier) concat, not logit averaging |

### B. Performance Gaps — CONFIRMED

| Issue | Evidence | Impact |
|---|---|---|
| **No LR scheduler** | `ABSENT` in all 6 config cells | Learning stalls; final plateau is suboptimal |
| **No early stopping** | `ABSENT` in all 6 training loops | Wastes compute; can't protect best checkpoint |
| **Batch size = 32** | `BATCH_SIZE = 32` in all configs | GPU severely underutilized: only 0.19 GB / 24 GB used |
| **NUM_WORKERS = 0** | `NUM_WORKERS = 0` in all configs | CPU blocks GPU every batch (sequential data loading) |
| **No AMP** | `ABSENT` in all loops | Tensor Cores on RTX A5000 completely idle |
| **No cuDNN benchmark** | `ABSENT` | cuDNN uses generic convolution kernels instead of optimal |
| **No LR logging** | `history` only has 4 keys, no `lr` | Cannot diagnose scheduler behavior |
| **No `drop_last`** | `drop_last` absent in train DataLoader | Last partial batch at large batch sizes can cause instability |
| **LR curve plot absent** | Plot cell has 3 plots, no LR subplot | Cannot visualize LR decay |
| **`cudnn.deterministic = True`** | Conflicts with `benchmark = True` | Cannot use cuDNN benchmark without disabling strict determinism |

---

## 2. Architecture Summary (Unchanged)

### VMamba (NB 01, 02)
- Input: `(B, 12, 7, 7)` → `InputAdapter(1×1 Conv, BN, GELU)` → `(B, 3, 7, 7)`
- Backbone: `VSSM(patch_size=1, dims=48, depths=[2,2,4,2], forward_type='v05_noz')`
- Spatial: 7→7→4→2→1 → `Linear(384, K)`
- Params: ~2.1 M

### InternImage (NB 03, 04)
- Input: `(B, 12, 7, 7)` → `InputAdapter12` → `MicroStemLayer(stride=1)` → `(B, 7, 7, 64)` channels-last
- Levels: 64→128→256→512 with DCNv3_pytorch (pure PyTorch `F.grid_sample`)
- Head: `Conv2d(512→768) + AvgPool + Linear(768, K)`
- Params: ~5.6 M

### Hybrid (NB 05, 06)
- Shared adapter: `Conv2d(12→3, 1×1, BN, GELU)`
- VMamba branch → `(B, 384)` (classifier[:-1])
- InternImage branch → `(B, 768)` (probed dynamically at `__init__`)
- Fusion: `concat(B, 1152)` → `Linear(1152, 576)` → GELU → `Linear(576, 576)` → GELU → `Dropout(0.3)` → `Linear(576, K)`

---

## 3. Baseline Results (from user-reported run)

| Notebook | Dataset | Test Acc | Test F1 (macro) | Duration |
|---|---|---|---|---|
| 04_InternImage_Rabi_Adam | Rabi (13 cls) | **63.24%** | 62.99% | 139.3 min |
| Others | Not yet run | — | — | — |

---

## 4. Changes Applied (All 6 Notebooks)

See `ACCURACY_IMPROVEMENTS_CHANGELOG.md` for full details.

| Change | Setting |
|---|---|
| Batch size | 32 → **128** |
| NUM_WORKERS | 0 → **4** |
| LR scheduler | Added `ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6)` |
| Early stopping | Added (patience=**15**, monitored: val_loss) |
| AMP mixed precision | Added (`USE_AMP = True`) |
| cuDNN benchmark | Added (`torch.backends.cudnn.benchmark = True`) |
| deterministic | `True` → `False` (required for benchmark mode) |
| History | Added `lr` key |
| Checkpoint | Added `scheduler_state_dict` |
| DataLoader | Added `drop_last=True` to train_loader |
| Plots | Added LR curve (`lr_curve.png`) |

---

## 5. Expected Impact

| Change | Expected Speedup | Expected Accuracy Effect |
|---|---|---|
| Batch size 32→128 | **3–5×** fewer iterations per epoch | Similar or slightly better (larger effective batch) |
| NUM_WORKERS 0→4 | **2–3×** data loading speed | No direct accuracy effect |
| AMP | **1.5–2×** GPU throughput | No direct accuracy effect |
| cuDNN benchmark | **5–15%** kernel speed | No direct accuracy effect |
| LR scheduler | — | **+2–5%** accuracy (avoids plateau) |
| Early stopping | — | Prevents overfitting; saves best model more reliably |

**Combined: training time expected to drop from ~140 min → ~20–40 min on RTX A5000.**

---

## 6. Preserved Constraints

- ✅ Adam optimizer (unchanged)
- ✅ Frozen splits (unchanged)
- ✅ Train-only normalization (unchanged)
- ✅ Class-weighted CrossEntropy (unchanged)
- ✅ All 6 experiments (unchanged)
- ✅ All architectures (unchanged)
- ✅ Checkpoint by val_acc (unchanged)
- ✅ Test set never used for selection (unchanged)
- ✅ No external project imports (unchanged)
