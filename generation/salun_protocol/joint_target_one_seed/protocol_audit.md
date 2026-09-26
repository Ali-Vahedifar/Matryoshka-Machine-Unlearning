# Why the earlier ranking did not transfer

The earlier positive results and the SalUn-protocol results evaluate different
training recipes and use different selection criteria. They do not establish a
contradiction or universal superiority of either method.

| Item | Earlier CIFAR-10 protocol v3 | Current SalUn protocol |
|---|---|---|
| Scenarios | Class, subclass and random-instance deletion | 10% and 50% random deletion |
| Random-deletion sampling | Balanced across classes; final forget accuracy uses a reserved member subset | Uniform random subset; final forget accuracy uses the whole forgotten set |
| Source training | Adam, validation early stopping, cap 100 epochs | SGD, cosine schedule, 182 epochs |
| MMU retain access per pass | Fixed 10% subset of retained training data | All retained training data |
| Batch size | 64 | 256 |
| MMU settings searched | Epochs, forget passes, learning rate, head mode, width weighting; gradient clipping at 5 | Three learning rates; five epochs, two forget passes, independent heads, summed widths, no clipping |
| Main ranking | Forget accuracy gap subject to a two-point utility constraint; source, retrain and fine-tuning excluded from medals | Average absolute Retrain gap across UA, RA, TA and confidence-SVC MIA |
| Retain accuracy | Held-out retained evaluation split | Retained training set |
| Membership evaluation | Population confidence attack and a separate U-LiRA experiment | Released confidence-SVC nonmembership statistic |
| SalUn implementation | Earlier local implementation | Adapter around pinned released classification code, with disclosed reproduction limitations |

The current 10% split contains 4,500 forgotten and 40,500 retained images.
At batch size 256, each forget pass has 18 batches and each retain pass has
159 batches. Five retain passes and two forget passes therefore give 795
retain updates versus 36 forget updates. The earlier retain subset contained
4,050 examples; at batch size 64, a pass has 64 retain versus 71 forget batches.
This changes the relative repair budget substantially. It is a plausible
contributor to the observed restoration, not an isolated causal result.

The archived ResNet-18 instance-deletion seed-42 choice was learning rate
0.005, three forget passes, five epochs, tied heads, summed width losses, and
max-step gradient clipping at 5. The newer grid did not include that learning
rate or head mode. Calling the newer three-point search a sufficient test of
the earlier MMU recipe would be too strong.

The old evidence also had limits. On ResNet-18 random-instance deletion, MMU's
forget-accuracy gap was 6.2 points against the source's 7.9: partial progress,
not oracle matching. SalUn's gap was smaller (4.3), but it missed the utility
constraint and was excluded from medal eligibility. MMU's U-LiRA AUC was 69.0
against a 50 reference, so the gold accuracy-gap cell did not imply the best
privacy result. The class-deletion gold cell also tied other methods at a
rounded zero forgetting gap, with utility breaking ties.

## This follow-up

Use seed 2 and the existing 10% split only. Preserve the earlier tables and
diagnostic. Two fixed new controls minimize a finite-target forget loss and
retained CE/distillation in the same optimizer step. All widths use the
full-width retained teacher target, and losses are averaged across widths.
Only nesting differs between these two controls. The uniform forgotten target
is an exploratory surrogate: a retrained model need not predict uniformly on
randomly forgotten examples. A high MIA score or low confidence alone would
not prove unlearning.

The final fifth epoch is specified in advance. No checkpoint is selected using
test metrics. The new recipe changes multiple ingredients, so its comparison
to the archived sequential method cannot attribute effects to one ingredient.
A separate transfer check uses the archived seed-42 MMU recipe on the current
seed-2 source and split, without searching parameters. That check also changes
multiple recipe ingredients together and is not a causal ablation.

Sources within the workspace:

- `cifar10_protocol_v3/run_cell.sh`
- `cifar10_protocol_v3/mu/resnet18_adam_instance/seed42/cache/manifest.json`
- `cifar10_protocol_v3/mu/resnet18_adam_instance/seed42/final.json`
- `cifar10_protocol_v3/tables/results_tables.tex`
- `mmu/scripts/benchmark_unlearning.py`
- `salun_protocol/classification.py`
- `salun_protocol/diagnostic_one_seed/report.md`
