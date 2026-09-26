# One-seed gradient/pass diagnostic

Seed 2, CIFAR-10, 10% forgetting. One source checkpoint, two matched controls, five epochs, two forget passes, learning rate 0.001. No search or additional seeds. Full-width SCRUB is the existing SCRUB-style objective restricted to width 512; it is not an independent released SCRUB reproduction. Both controls use identical training batch/augmentation seeds. Instrumentation preserves training RNG and uses eval-mode probes.

Notation: D_f denotes forget-set accuracy, D_r denotes retain-set accuracy, and Test denotes test accuracy. D_f accuracy = 100 − UA; absolute gaps to Retrain are unchanged.

All 4,500 forgotten examples are evaluated; retained/test accuracies below use fixed 2,048-example probes.

| Control | Stage | D_f accuracy | D_r probe accuracy | Test probe accuracy | Forget teacher KL |
|---|---|---:|---:|---:|---:|
| fullwidth_scrub | source | 100.00 | 100.00 | 93.85 | -0.000000 |
| fullwidth_scrub | epoch0_forget | 100.00 | 100.00 | 93.85 | 0.028704 |
| fullwidth_scrub | epoch0_retain | 100.00 | 100.00 | 93.90 | 0.005397 |
| fullwidth_scrub | epoch1_forget | 100.00 | 100.00 | 94.09 | 0.648809 |
| fullwidth_scrub | epoch1_retain | 100.00 | 100.00 | 93.90 | 0.004570 |
| fullwidth_scrub | epoch2_retain | 100.00 | 100.00 | 94.04 | 0.006015 |
| fullwidth_scrub | epoch3_retain | 100.00 | 100.00 | 93.85 | 0.004614 |
| fullwidth_scrub | epoch4_retain | 100.00 | 100.00 | 93.95 | 0.005682 |
| nested_mmu | source | 100.00 | 100.00 | 93.85 | -0.000000 |
| nested_mmu | epoch0_forget | 100.00 | 100.00 | 93.85 | 0.091467 |
| nested_mmu | epoch0_retain | 100.00 | 100.00 | 94.14 | 0.018349 |
| nested_mmu | epoch1_forget | 81.42 | 82.23 | 75.78 | 4.758034 |
| nested_mmu | epoch1_retain | 100.00 | 100.00 | 94.29 | 0.024816 |
| nested_mmu | epoch2_retain | 100.00 | 100.00 | 94.14 | 0.023995 |
| nested_mmu | epoch3_retain | 100.00 | 100.00 | 93.85 | 0.024160 |
| nested_mmu | epoch4_retain | 100.00 | 100.00 | 93.95 | 0.019008 |

## Findings

1. Identical eval-mode student/teacher predictions yield essentially zero initial KL gradient (about 1.5e-6, numerical residual). In the actual train-student/eval-teacher mode, the initial gradient norms are 5.61 (full width) and 6.91 (nested). The train/eval mismatch breaks prediction equality. A zero-gradient start is therefore not the observed cause of failure.
2. The second nested forget pass lowers forgotten accuracy from 100% to 81.42%, but also lowers retained probe accuracy to 82.23% and test probe accuracy to 75.78%. This is broad damage, not evidence of selective forgetting.
3. The immediately following retain pass restores forgotten and retained probe accuracy to 100%, and test probe accuracy to 94.29%. Forget teacher KL falls from 4.7580 to 0.02482, a roughly 99.5% reduction.
4. Full-width SCRUB changes teacher disagreement but retains 100% forgotten accuracy throughout. Both controls finish with D_f accuracy=100% (equivalently UA=0).

The supported mechanism is nonselective repulsion followed by restoration, rather than effective selective forgetting. Simply ending after the forget pass would report higher UA on a broadly damaged classifier. This one-seed diagnostic does not establish a universal limitation of MMU or SCRUB.

Existing classification and generation results are preserved. No superiority over SalUn is claimed. Any follow-up should first test whether an update separates forget effects from retained damage; it should remain one seed unless explicitly expanded.

[Pass plot](pass_diagnostic.pdf)

Follow-up: [One-seed revision results, in D_f/D_r/Test notation](../joint_target_one_seed/report.md). Generation: [expanded comparison](../generation/expanded/comparison.pdf) and [all samples](../generation/expanded/all_samples.pdf).
