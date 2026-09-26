# MMU improvement study

CIFAR-10 airplane forgetting; guidance 2; 100 DDIM steps; 500 forgotten-condition and 5,000 retained samples per evaluation. Three joint epochs; warm-up adds two retained-only epochs with the same optimizer. The beta=1 and nested no-warm-up references reuse the completed experiment.

Arrows: ↑ higher, ↓ lower. UA ↑ denotes stronger class suppression; it must be interpreted alongside generation quality.

| Seed-42 configuration | UA (%) ↑ | Retained FID ↓ | Retained accuracy (%) ↑ |
|---|---:|---:|---:|
| Full width beta=0.5 | 83.8 | 17.43 | 97.76 |
| Full width beta=1 | 86.2 | 17.66 | 97.56 |
| Full width beta=2 | 88.6 | 18.00 | 97.08 |
| Full width beta=4 | 90.4 | 17.01 | 97.14 |
| Full width warm-up=2 | 85.2 | 17.05 | 96.94 |
| Nested warm-up=0 | 49.0 | 25.72 | 93.68 |
| Nested warm-up=2 | 72.4 | 19.47 | 95.90 |

Seed 42 is the tuning run. Additional seeds share the same pretrained source checkpoint; they measure variation in retained subset selection, unlearning and sampling, not source training. Higher UA alone does not establish improved unlearning.

[Forgotten-condition grid](samples_forget.png)

[Retained-condition grid](samples_retain.png)

## Paired-seed comparison

| Configuration | Seed | UA (%) ↑ | Retained FID ↓ | Retained accuracy (%) ↑ |
|---|---:|---:|---:|---:|
| Baseline beta=1 | 42 | 86.2 | 17.66 | 97.56 |
| Baseline beta=1 | 43 | 92.2 | 17.01 | 97.70 |
| Baseline beta=1 | 44 | 90.0 | 17.31 | 97.68 |
| Baseline beta=1 | mean ± sample SD | 89.47 ± 3.04 | 17.33 ± 0.33 | 97.65 ± 0.08 |
| Selected candidate | 42 | 90.4 | 17.01 | 97.14 |
| Selected candidate | 43 | 91.2 | 19.60 | 97.02 |
| Selected candidate | 44 | 93.2 | 17.01 | 97.02 |
| Selected candidate | mean ± sample SD | 91.60 ± 1.44 | 17.87 ± 1.50 | 97.06 ± 0.07 |

The three-seed summary includes the tuning seed. The seed-43/44 rows are the additional checks.

## Conclusion

The beta sweep gives a tradeoff, not a clear overall improvement. Across seeds 42/43/44, beta=4 increases mean UA by 2.13 percentage points, but retained FID worsens by 0.54 and retained accuracy falls by 0.59 points. On the two additional seeds alone, its mean UA advantage is 1.10 points, with retained FID worse by 1.14 and accuracy lower by 0.67 points. Seed 43 does not reproduce the tuning seed's FID gain. Keep beta=1 as the default for retained utility; beta=4 is an optional stronger-suppression setting, not a validated overall winner.

Two-epoch prefix warm-up substantially improves the nested model on seed 42 (UA 49.0 to 72.4, FID 25.72 to 19.47, retained accuracy 93.68 to 95.90). However, it still loses on all three metrics to the equally warmed full-width control (85.2, 17.05, 96.94). This does not establish a benefit from nesting. The warm-up comparison is single-seed.

The saved grids show structured outputs, including recognizable alternative objects, rather than the earlier noise-only failure. They do not establish distributional equivalence to retraining or removal of learned information.

No new SalUn runs were included in this improvement study; it does not establish superiority over SalUn. The source checkpoint is shared across seeds. No defaults or earlier result tables were overwritten.

[Additional-seed forgotten-condition grid](validation_forget.png) · [Additional-seed retained-condition grid](validation_retain.png)
