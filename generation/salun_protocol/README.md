# CIFAR-10 SalUn/MMU study

- [Generation comparison](generation/README.md): complete, including matched random-mask control.
- [Classification status](classification_status.json): live queue status and current log.
- [Classification results](classification_summary.md): updated after each trial/ratio; incomplete rows show trial counts.
- [Protocol and deviations](protocol.json).

Classification first tunes on pilot seed 1 and the 5,000-image validation set.
It then freezes choices and runs independent source/retrain/unlearning trials
for seeds 2–11 at both 10% and 50% random forgetting. Source and retrain use
182 epochs of cosine-scheduled SGD. Final MIA uses the released confidence SVC
with test nonmembers and retained members; UA/RA/TA/MIA gaps compare to paired
retraining references. The explicit search is a coarse grid in published ranges.

The released SalUn-soft entry point is used with its required mask_ratio set.
Its implementation differs from the paper's idealized proximal algorithm;
this is disclosed in the protocol, not silently presented as an exact reproduction.

The queue runs independently after launch. It updates classification_status.json
on stage transitions, including failures. Rerunning the queue resumes completed
source/retrain checkpoints and skips completed result files. Model storage is
limited to source/retrain/MMU/SalUn; all candidate settings and metrics are kept.

Logs are under `logs/`. No original experiment results were overwritten.

Display notation: `D_f` is forget-set accuracy (`100 - UA`), `D_r` is retained accuracy, and Test is held-out test accuracy. Raw JSON keys remain UA/RA/TA for compatibility; reported absolute gaps are unchanged. Pre-conversion reports are preserved in `notation_archive_ua/`.
