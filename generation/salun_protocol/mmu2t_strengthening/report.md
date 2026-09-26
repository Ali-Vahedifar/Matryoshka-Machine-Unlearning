# Two-teacher MMU: strengthening study


Reference: latest full-width two-teacher MMU, fixed margin 0.002. Generation training seed 42; classification seed 2 reuses existing paired source/retrain models. No seed sweep or test-based hyperparameter selection. These are exploratory results, including negative outcomes.

## 1. Component ablation

Three fixed deletion tasks: airplane, automobile, bird. Full-width versus nested student prefixes [0.5, 0.75, 1.0], crossed with attraction only versus attraction plus repulsion. Nested variants share full-width frozen-source targets; they are ablations, not the reference MMU. Baselines use the existing local masked objective. Each run has 1,000 updates and 128 retained + 128 forgotten examples per full batch; partial batches are preserved. Nested gradients average across widths.

| Method | D_f condition accuracy (%) ↓ | D_r condition accuracy (%) ↑ | Mean training seconds ↓ |
|---|---:|---:|---:|
| MMU | 0.00 | 96.76 | 106.4 |
| Attraction only | 4.17 | 98.84 | 107.8 |
| Nested MMU | 6.25 | 77.08 | 240.2 |
| Nested attraction | 10.42 | 75.69 | 240.2 |
| Random mask | 6.25 | 93.52 | 80.2 |
| SalUn | 0.00 | 97.92 | 80.2 |

The CIFAR masked baselines were rerun in this suite’s training loop. Their stochastic draw order differs from the earlier ten-class driver, despite the same numeric seed and loss settings. Treat these as new trajectories, not exact reproductions of the earlier baseline checkpoints; do not pool the two tables as independent seeds.

Training time includes checkpoint writes, excludes evaluation and mask construction. Baseline saliency computation is additional overhead. Matched update counts are not equal FLOPs; nested variants use three student widths. These are local baseline implementations.

## 2. Update-budget curves

All checkpoints at 100, 250, 500, and 1,000 updates are reported; no best checkpoint is selected using these evaluation samples. Curves average counts over three deletion tasks (48 forgotten and 432 retained generated images per checkpoint). No statistical superiority is inferred from task averages.

[Budget and trade-off curves](budget_curves.pdf)

## 3. Sequential versus joint deletion

Sequential order is airplane → automobile → bird, 1,000 updates per stage. Each stage receives only the newly forgotten class; retained batches exclude all classes removed so far. There is no forgotten-data rehearsal. All deleted conditions redirect toward truck, which remains retained. The frozen original source supplies MMU targets throughout. Joint deletion uses all three forgotten classes together for 3,000 updates. Both have the same total optimizer budget, but data exposure necessarily differs. A single fixed order is tested.

| Method | Route | Airplane still predicted /16 ↓ | Automobile /16 ↓ | Bird /16 ↓ | D_r accuracy (%) ↑ |
|---|---|---:|---:|---:|---:|
| MMU | sequential/stage3 | 3 | 0 | 0 | 89.29 |
| MMU | joint | 0 | 0 | 0 | 97.32 |
| Random mask | sequential/stage3 | 0 | 0 | 0 | 97.32 |
| Random mask | joint | 0 | 0 | 0 | 97.32 |
| SalUn | sequential/stage3 | 0 | 1 | 0 | 98.21 |
| SalUn | joint | 0 | 1 | 0 | 98.21 |

[Reappearance by deletion stage](sequential_reappearance.pdf)

## 4. Held-out generation

CIFAR: sampling seeds 60042 and 60043, fixed before evaluation; same three deletion tasks and full-width final models. No image selection. Each method has 96 forgotten and 864 retained outputs over tasks and draws.

| Method | D_f accuracy (%) ↓ | D_r accuracy (%) ↑ | Forgotten-class predictions under retained conditions /864 ↓ |
|---|---:|---:|---:|
| MMU | 0.00 | 96.99 | 3 |
| Random mask | 2.08 | 93.06 | 16 |
| SalUn | 1.04 | 96.99 | 2 |

Stable Diffusion: 20 new nonsexual adult-art prompts (paraphrases/compositions), plus 20 new retained prompts, fixed seed-42 checkpoints. Source, MMU, Random mask, and SalUn share sampling noise. Detector thresholds remain 0.6 for scores and 0.2 for censored previews. CLIP is alignment, not comprehensive quality.

| Method | Forgotten detections /20 ↓ | Retained CLIP alignment ↑ |
|---|---:|---:|
| source | 7 | 0.3319 |
| mmu_2t_hi | 0 | 0.3335 |
| random_mask | 0 | 0.3265 |
| salun_port | 0 | 0.3298 |

[Held-out SD details and censored images](sd_heldout/report.md)

## 5. Retrain-calibrated classification

This is an explicit classification adaptation: a retain-only fine-tuned source supplies attraction targets on forgotten examples; the original frozen source supplies repulsion and retained preservation targets. The good teacher is trained for five epochs without forgotten data, but is initialized from a source that saw it. It is not the retrain oracle. MMU and attraction-only use five joint epochs, lr 0.001, KL temperature 1, margin 0.002 for MMU. Teacher construction cost is included separately. The full-width SCRUB-style control uses the prior local max/min recipe and temperature 4. Protocols are not interchangeable.

D_f = forgotten-set accuracy; D_r = retained-set accuracy. Average gap = mean absolute percentage-point deviation from paired retrain over D_f, D_r, Test accuracy and released confidence-SVC MIA efficacy. For random forgetting, closer to retrain is preferred; D_f should not be driven blindly to zero.

| Forget ratio | Method | D_f | D_r | Test accuracy | MIA efficacy | Avg. gap ↓ | Training minutes ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| 10% | Source | 100.00 | 100.00 | 94.62 | 0.02 | 4.84 | — |
| 10% | Retrain | 94.27 | 100.00 | 94.45 | 13.49 | 0.00 | 15.57 |
| 10% | SalUn | 99.42 | 99.99 | 94.42 | 12.38 | 1.58 | 1.04 |
| 10% | Retain FT teacher | 100.00 | 100.00 | 94.74 | 0.16 | 4.84 | 0.31 |
| 10% | MMU | 100.00 | 100.00 | 94.63 | 0.04 | 4.84 | 1.46 |
| 10% | Attraction only | 100.00 | 100.00 | 94.62 | 0.04 | 4.84 | 1.47 |
| 10% | Full-width SCRUB control | 100.00 | 100.00 | 94.64 | 0.04 | 4.84 | 0.55 |
| 50% | Source | 100.00 | 100.00 | 94.62 | 0.07 | 7.13 | — |
| 50% | Retrain | 92.58 | 100.00 | 91.84 | 18.39 | 0.00 | 8.81 |
| 50% | SalUn | 99.35 | 99.83 | 93.45 | 10.35 | 4.15 | 1.09 |
| 50% | Retain FT teacher | 99.99 | 100.00 | 94.62 | 0.19 | 7.10 | 0.17 |
| 50% | MMU | 100.00 | 100.00 | 94.66 | 0.07 | 7.14 | 0.81 |
| 50% | Attraction only | 100.00 | 100.00 | 94.72 | 0.07 | 7.15 | 0.81 |
| 50% | Full-width SCRUB control | 31.79 | 32.43 | 32.71 | 46.85 | 53.99 | 0.40 |

## Scope and reproducibility

Classification times include good-teacher construction for the two-teacher adaptation and attraction-only control; reused SalUn and retrain timings are historical. Sequential masks use the fixed original source on each new forgotten subset. Nested history `loss` records the mean scaled width contribution; its full averaged objective is the sum of recorded attraction, hinge, and retain terms (logging convention only). No additional training seeds, test-set tuning, or automatic winner selection. Sampling seeds are not independent training trials. All methods, checkpoints, and failure cases remain reported. Generation uses a fixed local CIFAR DDPM and local SD implementations. This study does not establish certified removal, comprehensive perceptual quality, or broad superiority.

Scripts and fixed protocols are beside this report. Raw checkpoints, sampled tensors, and per-sample predictions are in `$MMU_WORK/mmu2t_strengthening`. All original experiment results remain available.
