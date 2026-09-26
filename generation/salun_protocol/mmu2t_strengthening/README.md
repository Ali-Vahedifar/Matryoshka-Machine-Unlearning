# Two-teacher MMU strengthening study

Running a fixed one-seed suite. Reference is the latest full-width two-teacher MMU, not the older nested model. No ten-seed expansion or tuning sweep.

1. CIFAR generation: 3 fixed deleted classes × 6 methods (four component ablations plus Random mask and SalUn), checkpoints at 100/250/500/1000 updates.
2. Those checkpoints supply forgetting–retention curves without separate training.
3. Sequential airplane→automobile→bird deletion versus matched 3000-update joint deletion, with truck as a retained redirect target. No forgotten-data rehearsal; check earlier deletions after every stage.
4. Held-out CIFAR sampling noise and new SD prompt paraphrases/compositions, using fixed checkpoints.
5. Paired retrain-calibrated classification at 10% and 50% random forgetting, reusing seed-2 sources/retraining. Explicit two-network classification adaptation, not the old hinge-only implementation named MMU2T.

Generation runs first; classification and SD evaluation are queued on the same GPU. Raw checkpoints/samples live under $MMU_WORK/mmu2t_strengthening. Logs are generation.log, classification_study.log, sd_heldout.log, queue.log. Final report.md and analysis.pdf are produced only after all groups finish.

MMU nested ablations share full-width frozen targets; the three student prefix losses are averaged. Equal optimizer budgets do not equal compute. Classification attraction teacher construction is charged as method overhead. Small samples and single training seeds support exploratory findings, not significance or certified unlearning claims. No prior negative results are overwritten.

Sequential baseline masks are computed on each newly forgotten subset using the fixed original source. Random masks match their global 50% density. This is a fixed-source masking adaptation, not a claim of an official sequential SalUn protocol. For nested histories, the recorded `loss` is the mean per-width scaled contribution; the full averaged objective is `attraction + hinge + retain`. This affects logging only, not gradients.
