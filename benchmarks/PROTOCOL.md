# Classification benchmark protocol

Implemented by `mmu/scripts/benchmark_unlearning.py`; one cell-seed is launched by
`run_cell.sh`, the whole grid by `run_all.sh`. Every setting below is written into the
`protocol` block of each `final.json`.

## Shared across datasets

- **Source ("Original") training.** Adam, lr 1e-3 (betas 0.9/0.999, weight decay 0),
  batch 64, ReduceLROnPlateau on validation loss (factor 0.5, patience 5), early
  stopping with the best-validation checkpoint restored. Train augmentation
  RandomCrop(32, pad 4) + horizontal flip; validation/test normalise only.
- **Splits.** The 50,000 training images are split 90/10 stratified into 45,000 train /
  5,000 validation. The 10,000-image official test set is evaluated **once per method,
  after selection**.
- **Retrain** uses the identical recipe on D_r only (same optimiser, scheduler,
  validation and early stopping). Every method in a cell starts from one byte-identical
  source checkpoint (SHA-256 recorded per result).
- **Access.** Post-hoc methods see D_f plus 10% of D_r (`--retain_ratio 0.1`);
  Fine-tuning sees all of D_r. Amnesiac additionally needs the per-batch update ledger
  recorded during source training, so it runs for class deletion only.
- **Selection (`--selection utility`).** Among grid candidates whose retain **and**
  test accuracy on validation data stay within 2.0 points of Retrain, pick the one whose
  forget accuracy is closest to Retrain's (ties: higher retain accuracy). If nothing is
  feasible, the least damaging candidate is reported and flagged `feasible: false`.
  The reported |ΔD_f| is therefore not the tuned quantity.
- **SCRUB and MMU** use max-step gradient-norm clipping 5.0 (without it SCRUB diverged at
  every lr ≥ 5e-4 on the CNN backbone).
- **Relearn probe.** After the test evaluation, 5 Adam epochs at lr 1e-4 on D_f; forget
  accuracy after each epoch is stored as `relearn_forget_acc`.
- **Seeds** 42, 43, 44. **U-LiRA** (`mmu/scripts/ulira.py`): seed 42, 16 shadow models
  per cell, per-example Gaussian likelihood ratio with shared variance; positives are the
  method's final model, negatives the Retrain model on the same D_f examples.

## Per-dataset settings

| | CIFAR-10 | CIFAR-100 | RTI |
|---|---|---|---|
| label space | 10 classes | 100 classes | 20 CIFAR-100 superclasses |
| epoch cap / patience / min | 100 / 10 / 20 | 150 / 15 / 30 | 150 / 15 / 30 |
| CNN (FPECNN) head width | 64 | 512 | 256 |
| MMU nesting-width floor | 8 | 100 | 20 |
| class deletion | class 0 (airplane) | class 42 | superclass 0 (2,250 images) |
| subclass deletion | fine class 0 inside the 5×2 grouping of `core/datasets/cifar10_coarse.py` | fine class 42 inside its superclass; model trains on the 20 superclasses | fine class 0 inside its superclass |
| instance deletion | 4,500 images (10%), class-balanced | same | same |

Backbones: FPECNN ("CNN", ~2M parameters) and ResNet-18 (small-input stem, ~11M).

## Metrics (`core/evaluation/metrics.py`, `UnlearningEvaluator`)

| key in `final.json` | paper column |
|---|---|
| `forget_acc` | D_f accuracy; the tables report \|ΔD_f\| = \|D_f(method) − D_f(Retrain)\| per seed |
| `retain_acc`, `test_acc` | D_r and test accuracy |
| `mia`, `mia_auc` | membership-inference attack (threshold and score direction learned on independent calibration subsets) |
| `output_kl_divergence` | predictive KL to Retrain |
| `kl_divergence` | parameter-distribution KL to Retrain |
| `unlearn_time` | wall-clock seconds, method preparation included (e.g. SSD's Fisher); Retrain = its training time |
| `relearn_forget_acc` | D_f accuracy after each relearn epoch |
| `ulira.json` | U-LiRA AUC, TPR@1%FPR, TPR@0.1%FPR, balanced accuracy |

Timing caveat: the campaign ran three cells in parallel on one GPU, so unlearning times
include contention from unrelated cells.
