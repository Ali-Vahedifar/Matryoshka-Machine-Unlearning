# Two-teacher proposal: interpretation and checks

The workspace already contains `mmu_2t_lo` and `mmu_2t_hi` in run.py. An existing
seeds.py process is evaluating this family; this review does not launch or alter it.

The implementation uses two predictions from one frozen source checkpoint:
the forgotten condition supplies the reference to repel, and the retained
condition supplies the reference to approach. These are two teacher roles,
not two independently trained teachers. The latter prediction is not an oracle
trained without the forgotten concept.

The current forget loss is attraction to the retained-condition prediction
plus margin-limited squared-distance repulsion from the forgotten-condition
prediction, evaluated on noisy forgotten latents. Retained denoising and
teacher preservation are separate terms on noisy retained latents.

Important mathematical correction: at exact student/bad-teacher agreement,
the squared-distance repulsion gradient is zero. The code comment claiming
that agreement itself supplies a repulsion gradient is incorrect. Attraction
to a distinct good-teacher prediction supplies the initial direction. While
the hinge is active, equally weighted attraction and repulsion have prediction
gradient 2(bad-good)/d, where d is the number of averaged coordinates. They do
not create an independent nonzero repulsion gradient at initialization.

The hinge currently operates on batch-mean MSE. It does not enforce a margin
for each example. A future comparison should make that reduction explicit;
changing it during the active run would change the experiment being measured.

Additional limits on the pasted interpretation:

- Small loss values do not establish zero gradients, and a term's fraction of
  total loss is not its fraction of the gradient. Measure gradients to support
  those claims.
- Noisy forgotten latents and noisy retained latents have overlapping support.
  It is too strong to say a sampler never visits the retained-latent region.
  A better result after changing the training distribution supports that
  revision empirically, not that absolute trajectory claim.
- Overlapping confidence intervals do not demonstrate equivalence. Paired
  comparisons should use the matched prompts; zero detections on the same
  twenty prompts across training seeds does not establish general equivalence
  of random and saliency masks.
- The repeatedly inspected twenty prompts are now development evidence. A
  fixed, independently held-out prompt set is needed before claiming that a
  selected MMU variant matches SalUn.

The two-teacher idea is plausible, but its success must be established with
suppression and retained-quality measurements together. It does not by itself
resolve the independently observed degradation from nesting.
