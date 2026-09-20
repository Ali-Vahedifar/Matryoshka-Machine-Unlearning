# MMU: Matryoshka Machine Unlearning

A linear-head classifier already contains a nested family of readouts: the first `m`
feature dimensions read through the matching head columns,
`z_m = W[:, :m] f(x)[:m] + b`. Standard unlearning objectives act only on the
full-width readout, so the narrower readouts can keep the forgotten information.
MMU applies SCRUB's two losses at every nesting width `m ∈ M`:

```
max-step (D_f)   L_f = − Σ_m c_m · KL(teacher_m ‖ student_m)
min-step (D_r)   L_r =   Σ_m c_m · [ γ·CE(z_m, y) + α·KL(teacher_m ‖ student_m) ]
```

`M = {d/16, d/8, d/4, d/2, d}`, derived from the head's `in_features` and floored at
the number of classes (a narrower prefix is rank-deficient).

- `head_mode='independent'`: a separate classifier per width, warm-started from the
  head's first `m` columns and discarded after unlearning. `head_mode='tied'` slices
  the source head instead and adds no parameters.
- `weighting='mrl'`: `c_m = 1`. `weighting='normalised'`: the weights sum to 1.
- `bad_margin=None`: unbounded max-step. Passing a float hinges it at that margin.
- `granularities=(d,)` reduces exactly to SCRUB (covered by the tests).

The formal statement is in `method.tex`.

## Files

| File | Purpose |
|---|---|
| `mmu.py` | `MSCRUB`, `nested_logits`, `prefix_accuracy`, width weights |
| `prefix_leakage.py` | D_f / D_r accuracy at every nesting width for Original, Retrain, Fine-tune, SCRUB and MMU |
| `probe.py` | frozen-backbone linear probe per width: how much of D_f a fresh width-`m` readout recovers |

MMU is registered as `mmu` in `mmu/scripts/benchmark_unlearning.py`. Its grid is
SCRUB's grid times `head_mode × weighting`, with max-step gradient clipping at 5.0.
