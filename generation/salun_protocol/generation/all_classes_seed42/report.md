# One class removed per experiment: CIFAR-10, seed 42

Status: complete; 10/10 forgotten classes evaluated.

Each page is a separately unlearned model, initialized from the same source. Rows: Source, MMU, Random mask, SalUn. Columns show all ten generation conditions. Red outlines identify the forgotten condition, not detector results. The same initial noise is used across methods and experiments. The displayed sample is fixed in advance (index 0); no selection by appearance.

One training seed (42), 1,000 updates per method and deletion class. Class-0 checkpoints and matched samples are reused; the other nine deletions are trained independently from the source. MMU uses full-width frozen-source attraction and margin repulsion; its alternative target condition is (forgotten class + 1) modulo 10. SalUn and Random mask use the previous local recipes with class-specific 50% masks. This is not an official SalUn reproduction.

Exploratory classifier metrics use 16 generated samples per condition (16 forgotten and 144 retained). D_f is the fraction classified as the forgotten class under its condition; lower alone does not establish successful unlearning. D_r is retained-condition accuracy. No retrain reference, FID, or multiple-seed significance is established here. Native images are 32×32; nearest-neighbor rendering preserves their actual detail.

| Forgotten class | Method | D_f condition accuracy (%) | D_r condition accuracy (%) |
|---|---|---:|---:|
| airplane | Source | 100.00 | 100.00 |
| airplane | MMU | 0.00 | 97.92 |
| airplane | Random mask | 0.00 | 100.00 |
| airplane | SalUn | 0.00 | 97.22 |
| automobile | Source | 100.00 | 100.00 |
| automobile | MMU | 0.00 | 95.83 |
| automobile | Random mask | 0.00 | 99.31 |
| automobile | SalUn | 0.00 | 96.53 |
| bird | Source | 100.00 | 100.00 |
| bird | MMU | 0.00 | 95.83 |
| bird | Random mask | 0.00 | 97.92 |
| bird | SalUn | 0.00 | 97.92 |
| cat | Source | 100.00 | 100.00 |
| cat | MMU | 0.00 | 98.61 |
| cat | Random mask | 0.00 | 100.00 |
| cat | SalUn | 0.00 | 92.36 |
| deer | Source | 100.00 | 100.00 |
| deer | MMU | 0.00 | 99.31 |
| deer | Random mask | 0.00 | 98.61 |
| deer | SalUn | 0.00 | 97.22 |
| dog | Source | 100.00 | 100.00 |
| dog | MMU | 0.00 | 95.83 |
| dog | Random mask | 0.00 | 99.31 |
| dog | SalUn | 0.00 | 95.83 |
| frog | Source | 100.00 | 100.00 |
| frog | MMU | 0.00 | 98.61 |
| frog | Random mask | 0.00 | 96.53 |
| frog | SalUn | 0.00 | 93.06 |
| horse | Source | 100.00 | 100.00 |
| horse | MMU | 0.00 | 99.31 |
| horse | Random mask | 0.00 | 97.92 |
| horse | SalUn | 0.00 | 92.36 |
| ship | Source | 100.00 | 100.00 |
| ship | MMU | 0.00 | 98.61 |
| ship | Random mask | 0.00 | 98.61 |
| ship | SalUn | 0.00 | 86.81 |
| truck | Source | 100.00 | 100.00 |
| truck | MMU | 0.00 | 99.31 |
| truck | Random mask | 0.00 | 99.31 |
| truck | SalUn | 0.00 | 90.97 |

[Ten-page comparison](comparison.pdf) · [Configuration](config.json) · [Counts](metrics.json)

Checkpoints and raw samples: `$MMU_WORK/mmu_cifar_all_classes`.
