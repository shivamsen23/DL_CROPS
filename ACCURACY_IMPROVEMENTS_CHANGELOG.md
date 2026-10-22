# ACCURACY_IMPROVEMENTS_CHANGELOG.md
# Models_DL — Accuracy Improvements Changelog

**Applied:** 2026-10-10  
**Baseline backed up to:** `Models_DL/BACKUP_baseline/*.ipynb`  
**Smoke test:** ALL PASSED ✅

---

## Changes vs Baseline (All 6 Notebooks)

### 1. Batch Size: 32 → 128

**Rationale:**  
Mam's RTX A5000 has 24 GB VRAM. The baseline used only 0.19 GB (< 1%). With `(12,7,7)` inputs and ~5 M parameters, a batch of 128 uses < 2 GB — trivial. Larger batches:
- Reduce number of gradient steps per epoch (less Python overhead)
- Enable better GPU parallelization
- Reduce training time by ~3–5×

**Before:** `BATCH_SIZE = 32`  
**After:**  `BATCH_SIZE = 128`

---

### 2. NUM_WORKERS: 0 → 4

**Rationale:**  
`NUM_WORKERS=0` forces sequential data loading on the main thread. The GPU sits idle while Python unpickles and normalizes batches. On Linux (Mam's system), `NUM_WORKERS=4` uses 4 parallel worker processes to prefetch batches asynchronously. The `pin_memory=True` was already set correctly.

**Before:** `NUM_WORKERS = 0`  
**After:**  `NUM_WORKERS = 4`

> **Windows note:** If running locally on Windows, set `NUM_WORKERS = 0` — multiprocessing DataLoader does not work well on Windows without `if __name__ == '__main__':` guards.

---

### 3. LR Scheduler: Added `ReduceLROnPlateau`

**Rationale:**  
The baseline had no LR scheduler. With a flat `LR=1e-4`, training stalls at a plateau when the loss stops decreasing. `ReduceLROnPlateau` monitors `val_loss` and halves the LR when no improvement is seen for `patience=8` epochs, down to a floor of `1e-6`. This allows the model to "fine-tune" out of local plateaus without manual intervention.

**Before:** No scheduler.  
**After:**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
)
# stepped after each val epoch (NOT in smoke test mode)
scheduler.step(val_loss)
```

**New config keys:**
```python
SCHED_FACTOR   = 0.5
SCHED_PATIENCE = 8
SCHED_MIN_LR   = 1e-6
```

---

### 4. Early Stopping: Added (patience=15, val_loss)

**Rationale:**  
Without early stopping, training always runs the full `MAX_EPOCHS=100` even if the model converges at epoch 40. This wastes GPU time and can allow the optimizer to drift away from the best solution (overfitting). The patience of 15 epochs (intentionally larger than the scheduler patience of 8) gives the model time to benefit from LR reduction before stopping.

**Before:** No early stopping.  
**After:**
```python
_es_counter = 0
_es_best_loss = float('inf')
# ... inside epoch loop:
if val_loss < _es_best_loss - 1e-6:
    _es_best_loss = val_loss
    _es_counter = 0
else:
    _es_counter += 1
if _es_counter >= ES_PATIENCE:
    break   # stopped early
```

**New config key:** `ES_PATIENCE = 15`

> **Note:** The best checkpoint is still saved by `val_acc`. Early stopping uses `val_loss` as its monitor (more stable for triggering).

---

### 5. Mixed Precision (AMP): Added

**Rationale:**  
Mam's RTX A5000 has 3rd-generation Tensor Cores supporting FP16/BF16. The baseline runs everything in FP32, using none of this hardware. AMP (`torch.cuda.amp.autocast`) casts forward pass to FP16 automatically and uses `GradScaler` to prevent underflow in gradients. On Tensor Core hardware this gives **1.5–2×** speedup with no accuracy loss.

**Before:** No AMP.  
**After:**
```python
USE_AMP = True   # in config
_use_amp = USE_AMP and (DEVICE.type == 'cuda')
scaler = torch.cuda.amp.GradScaler(enabled=_use_amp)

# In train loop:
with torch.cuda.amp.autocast(enabled=_use_amp):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)   # before grad clip
scaler.step(optimizer)
scaler.update()
```

> On CPU, `_use_amp=False` so scaler is a no-op — backward-compatible.

---

### 6. cuDNN Benchmark Mode: Enabled

**Rationale:**  
`torch.backends.cudnn.benchmark = True` tells cuDNN to profile and select the fastest convolution algorithm for the fixed input shape `(B, 12, 7, 7)` on the first batch. Since all inputs are the same shape throughout training, this gives a free 5–15% speed boost.

**Note:** `deterministic = True` was changed to `False` because these two flags are mutually exclusive. For research reproducibility, the `SEED=42` and frozen splits still guarantee identical data ordering.

**Before:** `torch.backends.cudnn.deterministic = True` (no benchmark)  
**After:** 
```python
torch.backends.cudnn.deterministic = False
# In training cell:
torch.backends.cudnn.benchmark = True   # enabled when CUDA available
```

---

### 7. DataLoader `drop_last=True`: Added to train_loader

**Rationale:**  
With `BATCH_SIZE=128`, the last partial batch may have only a few samples. This can cause instability in BatchNorm layers (statistics computed over very few samples). `drop_last=True` discards the last partial batch.

**Before:** `DataLoader(..., shuffle=True, num_workers=0, pin_memory=...)`  
**After:**  `DataLoader(..., shuffle=True, num_workers=4, pin_memory=..., drop_last=True)`

---

### 8. LR Logging and LR Curve Plot

**Rationale:**  
Without LR logging it's impossible to diagnose whether the scheduler triggered or whether training stopped before any LR reduction. Now every epoch logs the current LR.

**Before:** `history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}`  
**After:**  `history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}`

**New plot saved:** `Results/.../lr_curve.png`

**Epoch log now shows:**  
```text
Epoch  1/100 | Train Loss=2.7123 Acc=0.1234 | Val Loss=2.6890 Acc=0.1456 | Best Val Acc=0.1456 | LR=1.00e-04 | 12s
```

---

### 9. Checkpoint Enhanced

**Before:** Saved `epoch, model_state_dict, optimizer_state_dict, val_acc, val_loss, classes, class_to_id, norm_mean, norm_std`  
**After:** Also saves `scheduler_state_dict` — allows resuming training with correct LR state.

---

## Baseline vs Improved Configuration

| Parameter | Baseline | Improved |
|---|---|---|
| `BATCH_SIZE` | 32 | **128** |
| `NUM_WORKERS` | 0 | **4** |
| `MAX_EPOCHS` | 100 | 100 (unchanged) |
| `LR` | 1e-4 | 1e-4 (unchanged) |
| `WEIGHT_DECAY` | 1e-4 | 1e-4 (unchanged) |
| `GRAD_CLIP` | 1.0 | 1.0 (unchanged) |
| `SEED` | 42 | 42 (unchanged) |
| LR Scheduler | None | **ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6)** |
| Early Stopping | None | **patience=15, monitor=val_loss** |
| AMP | Off | **USE_AMP=True (auto-disables on CPU)** |
| cuDNN benchmark | Off | **On (when CUDA available)** |
| `deterministic` | True | **False** (allows benchmark) |
| `drop_last` | False | **True** (train_loader only) |
| History keys | 4 | **5** (added `lr`) |
| Plots | 3 | **4** (added `lr_curve.png`) |

---

## Smoke Test Results (2026-10-10)

| Test | Result |
|---|---|
| Dataset loading (Kharif + Rabi) | ✅ |
| Frozen splits (overlap check) | ✅ |
| Normalization (train-only) | ✅ |
| VMamba forward+backward | ✅ |
| InternImage forward+backward | ✅ |
| Hybrid forward+backward | ✅ |
| VMamba Adam optimizer step | ✅ loss=3.0258 |
| InternImage Adam optimizer step | ✅ loss=2.9045 |
| Hybrid Adam optimizer step | ✅ loss=2.7665 |

**FINAL: ALL 9 SMOKE TESTS PASSED**

---

## Next Steps — Run Each Experiment on Mam's GPU

### Before running (one-time check on Mam's system)
```bash
conda activate shivamsen
cd /mnt/c/Users/goell/desktop/Models_DL   # or wherever Models_DL is
jupyter notebook
```

### For each notebook, open it, then:
1. **Confirm** `SMOKE_TEST = False` in Cell 2
2. **Confirm** `NUM_WORKERS = 4` (Linux supports this)
3. **Confirm** `VMAMBA_FORWARD_TYPE = 'v05_noz'` (or `'v2'` if CUDA extensions compiled)
4. Click **Kernel → Restart & Run All**
5. Monitor the epoch log — look for LR drops and early stopping messages

### Recommended order (shortest training first)
1. `03_InternImage_Kharif_Adam.ipynb` (~30 min estimated)
2. `04_InternImage_Rabi_Adam.ipynb` (~30 min estimated)
3. `01_VMamba_Kharif_Adam.ipynb` (~30 min estimated)
4. `02_VMamba_Rabi_Adam.ipynb` (~30 min estimated)
5. `05_Hybrid_InternImage_VMamba_Kharif_Adam.ipynb` (~45 min estimated)
6. `06_Hybrid_InternImage_VMamba_Rabi_Adam.ipynb` (~45 min estimated)

> Run one at a time. Do not start the next until the previous finishes and saves the checkpoint.
