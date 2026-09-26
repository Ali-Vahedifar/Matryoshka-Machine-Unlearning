# Two-teacher comparison: seed 42 only

Re-scored cached images from the completed runs. No new training, sampling or extra seeds were launched. Same 20 forgotten prompts and 20 retained prompts, with the same initial-noise seeds. All detection IDs and retained CLIP means reproduce the prior seed-42 records.

| Method | D_f: detected / 20 ↓ | D_r: mean CLIP cosine ↑ |
|---|---:|---:|
| Source SD | 15/20 | 0.3078 |
| SalUn | 0/20 | 0.2993 |
| Random mask | 0/20 | 0.3001 |
| MMU previous (weight 10) | 1/20 | 0.3044 |
| MMU two roles (margin 0.0005) | 1/20 | 0.3059 |
| MMU two roles (margin 0.002) | 0/20 | 0.2984 |

The higher-margin variant matches SalUn at 0/20 detected images on this development set. Its retained CLIP is 0.2984 versus SalUn 0.2993: no demonstrated retained-quality advantage. The lower-margin variant has 1/20 detections and higher retained alignment (0.3059). Both margins are shown; the successful one is not presented as the only attempted configuration.

These are two teacher roles from one frozen checkpoint, using retained versus forgotten conditioning. They are full-width variants, not evidence that nesting works. The training loss combines good-target attraction and batch-mean hinge repulsion, with retained denoising/preservation. Both use 100 updates and learning rate 1e-5. The main table refers only to training seed 42.

NudeNet 3.4.2, specified exposed categories and threshold 0.6. Preview censoring uses 0.2 and is separate from scoring. Red borders mark detections. CLIP cosine measures alignment, not comprehensive image quality. Zero detected images does not establish zero nudity or general equivalence; the 95% Wilson interval for 0/20 is approximately 0–16.1%. These prompts have already been used for development, so a new held-out set is needed for a generalization claim.

[Selected four-row comparison, two pages](comparison.pdf) · [First forgotten page](comparison_00.pdf) · [Second forgotten page](comparison_10.pdf) · [Raw re-scoring](results.json)

Figure exports: [Figure 1](figure_1.pdf) and [Figures 2–4 combined](figures_2_3_4.pdf). Rows are Source SD, MMU, Random mask, and SalUn. The displayed MMU is mmu_2t_hi (margin 0.002, seed 42); the complete six-row figures are preserved in archive_six_rows/. Figure labels omit titles and variant details; the full results above are unchanged.
