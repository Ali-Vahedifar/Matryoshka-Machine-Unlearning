# One-seed MMU improvement attempt

Completed: seed 2, CIFAR-10, 10% random forgetting. Three new fixed training runs: two matched joint-target controls and one transfer of the archived MMU recipe. No sweep, extra seeds or source retraining. Existing baseline rows are reused from this same seed and split.

All four accuracy-related metrics should approach Retrain; D_f is forget accuracy (100 minus UA), D_r is retain accuracy, and Test is test accuracy. Parentheses show absolute gaps to Retrain in percentage points. Avg. gap and runtime should decrease.

| Method | D_f accuracy → Retrain | D_r accuracy → Retrain | Test accuracy → Retrain | MIA → Retrain | Avg. gap ↓ | Train min ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Retrain | 94.27 (0.00) | 100.00 (0.00) | 94.45 (0.00) | 13.49 (0.00) | 0.00 | 15.57 |
| SalUn (released-code adapter) | 99.42 (5.16) | 99.99 (0.01) | 94.42 (0.03) | 12.38 (1.11) | 1.58 | 1.04 |
| MMU (previous adapter) | 100.00 (5.73) | 100.00 (0.00) | 94.87 (0.42) | 0.09 (13.40) | 4.89 | 0.45 |
| MMU (transferred old recipe) | 21.96 (72.31) | 21.29 (78.71) | 21.62 (72.83) | 31.38 (17.89) | 60.44 | 0.07 |
| Joint uniform target, full width | 100.00 (5.73) | 100.00 (0.00) | 94.44 (0.01) | 2.33 (11.16) | 4.22 | 0.99 |
| Joint uniform target, nested | 99.80 (5.53) | 100.00 (0.00) | 94.49 (0.04) | 5.53 (7.96) | 3.38 | 1.01 |

## Interpretation

The joint nested variant changes the average gap from 4.89 to 3.38. Its D_f accuracy is 99.80% against Retrain 94.27%, and its MIA is 5.53% against 13.49%. These values must be considered together: preserving accuracy does not establish removal of data influence.

The full-width joint control has average gap 4.22; the nested version has 3.38. This is a single-seed comparison, not statistical evidence of a general nesting advantage. The SalUn adapter has average gap 1.58.

In this pilot, the nested revision improves the aggregate gap while preserving utility, but still underforgets and remains behind SalUn. This is partial progress, not a solved method. The archived recipe transfer severely damages the current model (retained accuracy about 21%). Restoring old settings therefore does not recover old performance on this source. The schedule mismatch is real, but it is not the whole explanation.

The transferred recipe is reported even if it damages utility. It uses the old seed-42 instance recipe unchanged: lr 0.005, five epochs, three forget passes, tied heads, summed width losses, max-step gradient clipping at 5, batch size 64 and a fixed 10% retained subset. Only the source/split/evaluation come from the current seed-2 protocol. It is a recipe-transfer check, not an ablation isolating the retain-pass ratio.

## What changed in the experimental revision

Each update jointly minimizes retained cross-entropy, retained full-width-teacher distillation (temperature 4), and KL from a uniform class target to the student on forgotten images (temperature 1). The forget coefficient is |Df|/|Dr| = 1/9, fixed before training. Losses are averaged across widths. The full-width teacher supervises every retained prefix; no assumption is made that frozen source prefixes are competent classifiers.

The uniform target gives a finite optimum, unlike unrestricted repulsion. The KL loss itself is not bounded above. Uniform predictions are also not the true retraining target for random deletion; this is an exploratory confidence-reduction surrogate, not a certified unlearning objective. It changes the original MMU formulation and must be named separately in a paper.

Both groups share each forward batch and optimizer step. There are 795 joint steps, with forgotten batches recycled. This increases forget exposure relative to the archived 36 forget steps, so the comparison is not compute matched. Only width differs between the two new joint controls. Final epoch five is used without test-based checkpoint selection. Runtime excludes diagnostic probes and final evaluation.

## Verification and limits

Checked finite losses and parameters, immutability of the frozen teacher including buffers, absence of teacher gradients, and the forget objective gradient at a confident prediction and its uniform optimum. Evaluated all 4,500 forgotten, 40,500 retained and 10,000 test images with the existing confidence-SVC evaluator. Epoch diagnostics reuse the prior fixed 2,048 retained/test probes. Original table and diagnostic contents are preserved and hash-verified in notation_archive_ua; current displays use D_f, D_r and Test accuracy. Auxiliary heads and final model weights are saved. No paper superiority claim follows from this pilot.

[Why the earlier ranking changed](protocol_audit.md) · [Raw comparison](comparison.json) · [Fixed configuration](config.json)
