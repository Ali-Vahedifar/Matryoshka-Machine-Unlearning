# CIFAR-10 two-teacher transfer: seed 42

Same source, forgotten airplane class, two evaluation-noise seeds and 1,000-step sampler as the existing expanded figure. One fixed training seed; no hyperparameter search. Full-width model only.

| Method | D_f: airplane / 32 | D_f: airplane rate ↓ | D_r: condition accuracy ↑ |
|---|---:|---:|---:|
| Source | 31/32 | 96.88 | 99.65 |
| Retrain reference | 1/32 | 3.12 | 99.65 |
| SalUn (local) | 0/32 | 0.00 | 97.22 |
| Random mask (local) | 0/32 | 0.00 | 97.57 |
| MMU full width, beta=1 | 2/32 | 6.25 | 97.92 |
| MMU full width, beta=4 | 3/32 | 9.38 | 98.26 |
| MMU nested, warm-up=2 | 10/32 | 31.25 | 44.10 |
| frozen_redirect | 0/32 | 0.00 | 98.96 |
| two_teacher | 0/32 | 0.00 | 98.96 |

The two new runs differ only by the repulsion term. Both use frozen-teacher attraction on noisy forgotten images, retained denoising and teacher preservation on retained data. The teacher uses automobile conditioning as the acceptable target and airplane conditioning as the unwanted reference. The margin is the fixed transferred SD value 0.002; its suitability for this model is not established. The hinge acts on batch-mean per-coordinate MSE.

Each new run has 1,000 Adam updates, matching the existing SalUn/random update budget, but adds teacher preservation. Earlier MMU checkpoints used different training recipes. Lower airplane rate alone can reflect destroyed generation; inspect retained accuracy and images. The report uses only 32 forgotten and 288 retained generated images per method, so it is an exploratory comparison, not a reliable FID benchmark or a general superiority claim. Raw checkpoints and all sampled tensors are preserved in $MMU_WORK/mmu_cifar_two_teacher.

[Comparison](comparison.pdf) · [Airplane](airplane.pdf) · [Settings](config.json) · [Raw metrics](metrics.json)
