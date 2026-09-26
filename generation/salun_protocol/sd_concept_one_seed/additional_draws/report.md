# Additional matched SD draws

160 new images: the same 40 prompts, new per-prompt initial-noise seeds 400000–400039, four frozen trained models. Training seed remains 42. This tests new noise, not new prompts or new training seeds. The previously selected higher-margin MMU variant was fixed before sampling. All samples are included.

| Method | Detected on D_f / 20 ↓ | D_r CLIP cosine ↑ |
|---|---:|---:|
| Source SD | 15/20 | 0.3097 |
| MMU | 0/20 | 0.3025 |
| Random mask | 0/20 | 0.2966 |
| SalUn | 0/20 | 0.3013 |

NudeNet 3.4.2, fixed exposed categories, detection threshold 0.6. Black boxes use threshold 0.2 for presentation only. Retained CLIP is alignment, not comprehensive quality. These small counts do not establish complete removal. All original experiment artifacts are unchanged.

[Figure 1](figure_1.pdf) · [Figures 2–4 combined](figures_2_3_4.pdf) · [Both pages](comparison.pdf)
