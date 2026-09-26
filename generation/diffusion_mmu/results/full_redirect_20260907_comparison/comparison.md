# Revised MMU-gen: complete single-seed experiment

CIFAR-10, forgotten class airplane (0), seed 42. Existing source and class-0 retrain checkpoints; 3 unlearning epochs, 13,500 retained training images, 5,000 forgotten training images, batch 128, learning rate 1e-4. Redirection uses retained content and the forgotten label, not the forgotten images. All methods use the same learning rate in this run.

Each evaluation generates 500 forgotten-condition and 5,000 retained-condition images with 100 DDIM steps. Guidance 1 and 2 use identical saved model weights and evaluation seeds.

| Method | UA g=1 | FID g=1 | Retain acc g=1 | UA g=2 | FID g=2 | Retain acc g=2 | Training/load seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| source | 12.8 | 22.05 | 81.5 | 1.4 | 16.02 | 98.2 | 0.1 |
| retrain | 93.6 | 21.67 | 82.4 | 84.0 | 15.91 | 97.9 | 0.1 |
| finetune | 10.8 | 27.39 | 78.7 | 0.6 | 17.58 | 97.3 | 10.5 |
| neggrad | 100.0 | 304.34 | 15.3 | 100.0 | 304.80 | 5.6 | 20.7 |
| salun | 97.6 | 31.11 | 76.8 | 98.4 | 18.59 | 96.4 | 26.4 |
| mmu_fullwidth | 90.8 | 24.40 | 79.1 | 86.2 | 17.66 | 97.6 | 40.6 |
| mmu | 81.4 | 34.97 | 69.0 | 49.0 | 25.72 | 93.7 | 83.8 |
| mmu_noskip | 91.8 | 23.14 | 80.6 | 85.2 | 17.39 | 97.7 | 83.4 |
| mmu_retainonly | 30.4 | 34.54 | 72.1 | 4.4 | 24.95 | 94.8 | 52.1 |

Source and retrain times measure loading, not their original training cost. Model evaluation time is excluded from unlearning times.

The SalUn row is a local adaptation with the released prediction-matching direction, not a validated reproduction of published results. NegGrad uses 1e-4 here; the older experiment used 5e-5, so its rows are not directly interchangeable.

The retained-only control uses nested widths with skip truncation. No parameter sweep or checkpoint selection was performed. One seed and one forgotten class do not establish general superiority. UA measures classifier rejection of the forgotten class, not removal of training information. Retained FID does not assess image quality under the forgotten condition.

[Guidance-1 report and grids](diffusion_mmu/results/full_redirect_20260907_c0_seed42/summary.md)

[Guidance-2 forgotten-condition grid](samples_forget.png)

[Guidance-2 retained-condition grid](samples_retain.png)

## Interpretation

Full-width MMU redirection produces recognizable alternative objects in the saved forgotten-condition grids at both guidance scales, avoiding the noise-only behavior reported for the earlier repulsion objective. This is qualitative evidence from the saved samples, not a measurement of information removal.

At guidance 2, full-width MMU has UA 86.2%, retained FID 17.66, and retained accuracy 97.56%, compared with the retrain reference's 84.0%, 15.91, and 97.86%. The local SalUn adaptation suppresses airplanes more strongly (UA 98.4%) but has somewhat worse retained quality (FID 18.59, accuracy 96.36%). These are tradeoffs, not evidence of general superiority.

Nested MMU with skip truncation performs worse than full width on all three metrics at both guidance scales. Leaving skips untruncated brings its performance close to full width, at approximately twice the training cost. The nested retain-only control also degrades quality, indicating that prefix adaptation contributes to the utility loss independently of redirection. There is no demonstrated benefit of the Matryoshka construction in this run.


