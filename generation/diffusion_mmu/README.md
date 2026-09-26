# Revised diffusion experiment

`mmu_gen` now defaults to **retained-content redirection**, with a full-width
student. Retained images are presented under both their original label and the
forgotten label. The frozen full-width teacher under the original label supplies
the replacement target. This is an experimental surrogate, not exact unlearning.

- One joint update combines retention, redirection, and null-branch preservation.
- Nested student losses are averaged across widths and target the full teacher.
- `--objective repulsion` retains the older sequential algorithm for comparison.
  Its width sums and per-width teachers are deliberately unchanged; partial
  final batches are now included for all methods, so runs are not bitwise legacy
  reproductions. `--delta` applies only to repulsion. `--final_repair` is optional.
- `salun` uses detached current-model predictions under `(c_f+1)%10`, as in the
  [released DDPM runner](https://github.com/OPTML-Group/Unlearn-Saliency/blob/master/DDPM/runners/diffusion.py).
  It remains a local adaptation: mask estimation, architecture, loss scaling and
  training protocol have not been validated against published results.
- Each method resets its training seed; evaluation uses a separate shared seed.
  Runs save configuration, checkpoints, diagnostics, metrics, and grids for
  both retained and forgotten conditions. Existing run directories are refused.
- Guidance defaults to 1.0. Compare guidance 2.0 by reloading the same checkpoint.
  Retained FID is not a measure of forgotten-condition image quality. Inspect
  both grids; high UA is not sufficient evidence of unlearning.

From `generation/`:

```bash
python diffusion_mmu/run_experiment.py --methods mmu_fullwidth,mmu --tag redirect_v1 --seed 42
python diffusion_mmu/run_experiment.py --methods mmu_fullwidth --eval_checkpoint diffusion_mmu/results/redirect_v1_c0_seed42/mmu_fullwidth.pt --guidance 2 --tag redirect_cfg2 --seed 42
python -m unittest discover -s diffusion_mmu -p 'test_*.py'
```

These commands train new experiments; previous results and figures describe the
old objective. Run multiple seeds/classes before drawing performance conclusions.

## Improvement study

`--warmup_epochs 2` adds two retained-only epochs before joint redirection,
using the same optimizer and full-width teacher targets. Apply the same warm-up
to the full-width control when measuring the benefit of nested prefix training.
The default remains zero; repulsion rejects this option.

`run_improvement_study.py` runs the fixed seed-42 beta sweep (0.5, 2, 4; reuses
the existing beta-1 reference) and matched two-epoch warm-up at guidance 2.
Its plan and code snapshot are saved under `results/improvement_20260907`.
Outputs are separate from the original experiment. Further seeds validate
unlearning/sampling variability conditional on the same source checkpoint;
they are not independently trained source models.
