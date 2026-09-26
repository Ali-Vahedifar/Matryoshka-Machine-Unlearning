# MMU variants: three-seed CIFAR-10 classification

Status: complete. ResNet-18; seeds 42, 43, 44. Matched cached source and retrain checkpoints passed protocol and content checks; all unlearning variants and SalUn were rerun. No new hyperparameter selection on test results.

Class deletion removes airplane from the 10-way task. Instance deletion removes 4,500 class-balanced training examples (10% of the training split). Subclass deletion uses the existing custom five-superclass CIFAR-10 grouping: airplane/ship, automobile/truck, bird/frog, cat/dog, deer/horse. It removes airplane while its sibling ship remains. CIFAR-10 has no official subclass hierarchy.

D_f and D_r below are accuracies on the complete forgotten and retained training subsets with augmentation disabled. Test accuracy uses the official test set (5-way labels for subclass deletion). MIA is the existing independently calibrated confidence-attack accuracy, not the SalUn-protocol efficacy metric. Gap averages absolute deviations from paired Retrain across D_f, D_r, Test and MIA. Raw metrics target retrain; lower raw D_f is not universally better for instance/subclass deletion.

The original validation-selected MMU lr/epochs/forget-pass count is shared within each cell by source-role variants. The margin is fixed at 0.05 in temperature-4 KL units; it is not the diffusion MSE margin. Retain-FT adaptations use five joint epochs, lr .001, temperature 1, and margin .002 where present. All post-hoc students use the same 10% retained-data subset. Their retain-FT attraction teacher also uses that subset for five epochs, and its cost is included. SalUn uses its existing validation-selected settings; all per-cell settings are saved. Thus schedules are controlled within families, not identical across every method.

## Variant key

| ID | Name | Teacher target | Forgotten attraction |
|---|---|---|---|
| V01 | MMU unbounded / full | width-aligned source | no |
| V02 | MMU unbounded / nested MRL / sum | width-aligned source | no |
| V03 | MMU unbounded / nested MRL / weighted mean | width-aligned source | no |
| V04 | MMU unbounded / nested MRL-E / sum | width-aligned source | no |
| V05 | MMU unbounded / nested MRL-E / weighted mean | width-aligned source | no |
| V06 | MMU margin / full | width-aligned source | no |
| V07 | MMU margin / nested MRL / sum | width-aligned source | no |
| V08 | MMU margin / nested MRL / weighted mean | width-aligned source | no |
| V09 | MMU margin / nested MRL-E / sum | width-aligned source | no |
| V10 | MMU margin / nested MRL-E / weighted mean | width-aligned source | no |
| V11 | MMU margin / nested MRL / full privileged teacher | full source | no |
| V12 | MMU retain-FT two-network / full / attraction only | full retain-FT + full source | yes |
| V13 | MMU retain-FT two-network / full / attraction + margin | full retain-FT + full source | yes |
| V14 | MMU retain-FT two-network / nested MRL / attraction only | full retain-FT + full source | yes |
| V15 | MMU retain-FT two-network / nested MRL / attraction + margin | full retain-FT + full source | yes |

MRL = separate prefix heads; MRL-E = one sliced shared head. Sum = unnormalised width losses. Weighted mean = normalised original MMU weights (uniform forget weights and inverse-width retained weights). Source-role variants preserve the frozen source on D_r and repel it on D_f; this is one frozen network with two roles. Retain-FT adaptations use two distinct frozen networks and should not be confused with the diffusion method’s two conditioned predictions. No variant uses a moving full-width student as its teacher.

## Class deletion

| ID | D_f | D_r | Test | MIA | Gap ↓ | Seconds ↓ | Complete seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source | 99.99 ± 0.01 | 99.98 ± 0.02 | 92.86 ± 0.23 | 57.07 ± 0.53 | 28.96 ± 0.33 | — | 3/3 |
| Retrain | 0.00 ± 0.00 | 99.95 ± 0.08 | 83.92 ± 0.36 | 50.23 ± 1.59 | 0.00 ± 0.00 | 621.22 ± 135.22 | 3/3 |
| SalUn | 51.54 ± 12.58 | 99.54 ± 0.11 | 87.49 ± 1.06 | 57.18 ± 0.33 | 15.62 ± 3.04 | 9.70 ± 4.84 | 3/3 |
| V01 | 0.36 ± 0.63 | 99.97 ± 0.01 | 83.97 ± 0.19 | 54.44 ± 2.78 | 1.26 ± 1.15 | 7.60 ± 1.80 | 3/3 |
| V02 | 15.66 ± 27.12 | 99.98 ± 0.01 | 85.39 ± 2.21 | 55.88 ± 2.13 | 5.76 ± 8.18 | 8.15 ± 2.02 | 3/3 |
| V03 | 0.00 ± 0.00 | 99.73 ± 0.22 | 83.31 ± 0.50 | 54.91 ± 2.18 | 1.38 ± 0.75 | 8.13 ± 2.00 | 3/3 |
| V04 | 34.51 ± 55.67 | 99.98 ± 0.01 | 86.96 ± 4.61 | 55.99 ± 1.21 | 10.85 ± 15.70 | 8.16 ± 1.93 | 3/3 |
| V05 | 0.01 ± 0.01 | 99.91 ± 0.05 | 83.67 ± 0.37 | 54.61 ± 2.22 | 1.23 ± 0.83 | 8.17 ± 1.99 | 3/3 |
| V06 | 99.94 ± 0.05 | 99.98 ± 0.02 | 92.88 ± 0.20 | 56.71 ± 0.41 | 28.86 ± 0.45 | 7.65 ± 1.85 | 3/3 |
| V07 | 99.96 ± 0.03 | 99.98 ± 0.01 | 92.87 ± 0.24 | 57.06 ± 0.20 | 28.95 ± 0.40 | 8.10 ± 1.95 | 3/3 |
| V08 | 99.99 ± 0.01 | 99.97 ± 0.01 | 92.75 ± 0.33 | 56.90 ± 0.30 | 28.89 ± 0.53 | 8.06 ± 1.97 | 3/3 |
| V09 | 99.96 ± 0.04 | 99.98 ± 0.02 | 92.87 ± 0.24 | 56.96 ± 0.01 | 28.93 ± 0.42 | 8.15 ± 1.97 | 3/3 |
| V10 | 99.98 ± 0.02 | 99.98 ± 0.02 | 92.83 ± 0.28 | 56.91 ± 0.10 | 28.91 ± 0.42 | 8.16 ± 2.03 | 3/3 |
| V11 | 99.90 ± 0.09 | 99.94 ± 0.04 | 92.78 ± 0.16 | 56.63 ± 0.35 | 28.81 ± 0.41 | 8.02 ± 1.89 | 3/3 |
| V12 | 99.99 ± 0.01 | 99.65 ± 0.04 | 91.81 ± 0.22 | 56.82 ± 0.13 | 28.69 ± 0.49 | 9.95 ± 0.11 | 3/3 |
| V13 | 99.99 ± 0.01 | 99.65 ± 0.04 | 91.81 ± 0.22 | 56.82 ± 0.13 | 28.69 ± 0.49 | 10.00 ± 0.08 | 3/3 |
| V14 | 99.98 ± 0.02 | 99.76 ± 0.03 | 92.06 ± 0.14 | 56.71 ± 0.74 | 28.70 ± 0.54 | 10.32 ± 0.06 | 3/3 |
| V15 | 99.98 ± 0.02 | 99.76 ± 0.03 | 92.06 ± 0.14 | 56.71 ± 0.74 | 28.70 ± 0.54 | 10.37 ± 0.10 | 3/3 |
## Instance deletion

| ID | D_f | D_r | Test | MIA | Gap ↓ | Seconds ↓ | Complete seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source | 99.99 ± 0.01 | 99.98 ± 0.02 | 92.86 ± 0.23 | 57.60 ± 0.64 | 4.05 ± 0.23 | — | 3/3 |
| Retrain | 92.44 ± 0.25 | 99.82 ± 0.25 | 92.18 ± 0.30 | 49.76 ± 0.52 | 0.00 ± 0.00 | 511.12 ± 118.94 | 3/3 |
| SalUn | 96.18 ± 2.68 | 96.15 ± 2.45 | 88.43 ± 2.41 | 53.78 ± 1.42 | 3.79 ± 0.23 | 7.17 ± 4.88 | 3/3 |
| V01 | 93.98 ± 5.77 | 94.52 ± 5.31 | 87.78 ± 3.95 | 53.60 ± 1.25 | 4.67 ± 2.15 | 5.77 ± 0.07 | 3/3 |
| V02 | 97.05 ± 2.11 | 97.34 ± 1.88 | 90.23 ± 1.42 | 53.76 ± 0.58 | 3.26 ± 0.23 | 6.13 ± 0.05 | 3/3 |
| V03 | 58.02 ± 19.03 | 58.85 ± 18.93 | 56.85 ± 17.44 | 50.49 ± 0.73 | 27.90 ± 13.94 | 6.13 ± 0.03 | 3/3 |
| V04 | 97.51 ± 0.66 | 97.42 ± 0.74 | 90.26 ± 0.72 | 53.97 ± 0.69 | 3.40 ± 0.40 | 6.14 ± 0.02 | 3/3 |
| V05 | 50.04 ± 28.47 | 50.63 ± 27.92 | 48.41 ± 24.88 | 50.00 ± 1.99 | 34.13 ± 20.18 | 6.17 ± 0.07 | 3/3 |
| V06 | 99.97 ± 0.01 | 99.98 ± 0.02 | 92.72 ± 0.26 | 57.51 ± 0.74 | 3.99 ± 0.18 | 5.77 ± 0.06 | 3/3 |
| V07 | 99.98 ± 0.02 | 99.97 ± 0.01 | 92.87 ± 0.29 | 57.51 ± 0.57 | 4.03 ± 0.15 | 6.17 ± 0.04 | 3/3 |
| V08 | 99.99 ± 0.01 | 99.98 ± 0.01 | 92.99 ± 0.18 | 57.55 ± 0.51 | 4.08 ± 0.16 | 6.20 ± 0.11 | 3/3 |
| V09 | 99.99 ± 0.01 | 99.98 ± 0.02 | 92.85 ± 0.13 | 57.57 ± 0.52 | 4.04 ± 0.23 | 6.12 ± 0.02 | 3/3 |
| V10 | 99.99 ± 0.01 | 99.98 ± 0.01 | 92.87 ± 0.22 | 57.46 ± 0.59 | 4.02 ± 0.22 | 6.16 ± 0.04 | 3/3 |
| V11 | 99.90 ± 0.06 | 99.91 ± 0.04 | 92.81 ± 0.34 | 56.56 ± 0.42 | 3.76 ± 0.14 | 6.05 ± 0.03 | 3/3 |
| V12 | 99.99 ± 0.01 | 99.98 ± 0.02 | 92.85 ± 0.20 | 57.68 ± 0.73 | 4.07 ± 0.11 | 10.21 ± 0.08 | 3/3 |
| V13 | 99.99 ± 0.01 | 99.98 ± 0.01 | 92.85 ± 0.25 | 57.68 ± 0.56 | 4.07 ± 0.14 | 10.24 ± 0.08 | 3/3 |
| V14 | 99.99 ± 0.01 | 99.98 ± 0.02 | 92.91 ± 0.15 | 57.67 ± 0.63 | 4.09 ± 0.13 | 10.52 ± 0.07 | 3/3 |
| V15 | 99.99 ± 0.01 | 99.98 ± 0.02 | 92.91 ± 0.15 | 57.46 ± 0.84 | 4.03 ± 0.23 | 10.82 ± 0.37 | 3/3 |
## Subclass deletion

| ID | D_f | D_r | Test | MIA | Gap ↓ | Seconds ↓ | Complete seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source | 99.92 ± 0.05 | 99.98 ± 0.01 | 95.18 ± 0.03 | 55.57 ± 0.05 | 21.94 ± 0.21 | — | 3/3 |
| Retrain | 25.14 ± 0.88 | 99.88 ± 0.14 | 88.30 ± 0.17 | 49.56 ± 1.53 | 0.00 ± 0.00 | 479.47 ± 78.13 | 3/3 |
| SalUn | 68.32 ± 0.99 | 97.39 ± 0.69 | 89.50 ± 0.91 | 55.63 ± 0.81 | 13.23 ± 0.69 | 4.41 ± 0.06 | 3/3 |
| V01 | 99.21 ± 0.48 | 99.95 ± 0.02 | 94.90 ± 0.15 | 54.88 ± 1.34 | 21.52 ± 0.63 | 6.27 ± 2.67 | 3/3 |
| V02 | 99.57 ± 0.22 | 99.96 ± 0.01 | 95.03 ± 0.09 | 55.61 ± 0.82 | 21.82 ± 0.34 | 6.64 ± 2.80 | 3/3 |
| V03 | 37.80 ± 22.78 | 95.20 ± 2.33 | 85.10 ± 3.74 | 53.95 ± 2.53 | 7.70 ± 3.34 | 6.69 ± 2.87 | 3/3 |
| V04 | 99.73 ± 0.18 | 99.98 ± 0.01 | 95.09 ± 0.05 | 55.47 ± 0.08 | 21.85 ± 0.17 | 6.71 ± 2.81 | 3/3 |
| V05 | 68.72 ± 15.52 | 98.44 ± 0.94 | 90.49 ± 2.00 | 53.75 ± 2.97 | 12.85 ± 4.68 | 6.68 ± 2.86 | 3/3 |
| V06 | 99.84 ± 0.14 | 99.97 ± 0.02 | 95.08 ± 0.08 | 54.64 ± 0.36 | 21.66 ± 0.29 | 6.28 ± 2.68 | 3/3 |
| V07 | 99.89 ± 0.09 | 99.98 ± 0.01 | 95.10 ± 0.01 | 55.54 ± 0.35 | 21.90 ± 0.29 | 6.61 ± 2.81 | 3/3 |
| V08 | 99.93 ± 0.04 | 99.97 ± 0.01 | 95.03 ± 0.10 | 54.95 ± 0.51 | 21.75 ± 0.23 | 6.61 ± 2.87 | 3/3 |
| V09 | 99.87 ± 0.11 | 99.98 ± 0.01 | 95.11 ± 0.11 | 54.89 ± 0.22 | 21.74 ± 0.20 | 6.94 ± 2.68 | 3/3 |
| V10 | 99.91 ± 0.07 | 99.97 ± 0.01 | 95.09 ± 0.19 | 54.91 ± 0.15 | 21.75 ± 0.19 | 6.67 ± 2.89 | 3/3 |
| V11 | 99.92 ± 0.06 | 99.94 ± 0.02 | 94.79 ± 0.07 | 55.03 ± 0.87 | 21.70 ± 0.40 | 6.60 ± 2.94 | 3/3 |
| V12 | 99.88 ± 0.05 | 82.33 ± 0.31 | 78.85 ± 0.48 | 54.90 ± 0.65 | 26.77 ± 0.49 | 10.33 ± 0.15 | 3/3 |
| V13 | 99.88 ± 0.05 | 82.33 ± 0.31 | 78.85 ± 0.48 | 54.90 ± 0.65 | 26.77 ± 0.49 | 10.36 ± 0.13 | 3/3 |
| V14 | 99.87 ± 0.06 | 83.12 ± 1.34 | 79.53 ± 1.43 | 55.02 ± 0.83 | 26.43 ± 1.09 | 10.58 ± 0.16 | 3/3 |
| V15 | 99.87 ± 0.06 | 83.12 ± 1.34 | 79.53 ± 1.43 | 55.02 ± 0.83 | 26.43 ± 1.09 | 10.73 ± 0.16 | 3/3 |

Times include teacher construction for retain-FT variants. Source training is excluded, while retrain time is the cached full-retraining measurement. Three training seeds provide a variability estimate, not proof of superiority. Class and subclass each test one fixed deletion target. Underlying per-seed results, checkpoints and auxiliary heads are preserved under $MMU_WORK/mmu_variants_3seed.