# Figure-4-style generation comparison

[PDF figure](figure4_comparison.pdf) · [PNG figure](figure4_comparison.png)

I1–I4 use the airplane condition. C1–C9 use the nine retained conditions. All rows share initial sampling noise. Source and retrain references, local SalUn, its random-mask control, full-width MMU (beta 1 and 4), and warmed nested MMU are shown. The first samples were used without selection.

SalUn and random have exactly matched 50% global mask cardinality, identical 1,000-update objectives, and identical training seeds; their only training difference is the mask. Frozen coordinates were verified unchanged. `samples.pt` stores all 16 images per class, not only the figure subset.

Sampling uses 1,000 DDIM steps at guidance 2, unlike the earlier 100-step figures. The warmed nested MMU checkpoint shows severe visual deterioration at this sampling setting, and the retrain reference also shows distorted forgotten-condition outputs. Full-width MMU, local SalUn, and random produce recognizable images. This is visual evidence, not a full quantitative image-quality comparison.

The existing local architecture and source training differ from the released DDPM. MMU rows reuse checkpoints with their earlier training budgets; this figure is not an equal-compute comparison across MMU and SalUn. No published SalUn performance claim is made.

## Expanded samples

[Expanded comparison](expanded/comparison.pdf) · [Complete sample atlas](expanded/all_samples.pdf) · [Details](expanded/README.md)
