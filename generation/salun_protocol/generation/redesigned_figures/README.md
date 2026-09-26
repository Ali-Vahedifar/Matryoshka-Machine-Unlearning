# Redesigned behavioral analysis

Three descriptive figures from existing samples and logs. No new training or sample selection.

1. **Collateral damage:** retained accuracy loss from source, by deleted class. Rows sorted by SalUn loss; small vertical offsets distinguish ties, without changing x-values. Each method has 144 retained samples per task.
2. **Redirection versus disruption:** success means the classifier predicts the prescribed replacement class, not simply any non-forgotten class. Identical coordinates are grouped, with all class names shown. Each task has 16 forgotten samples. The desired direction is upper left; redirection is not proof of data removal.
3. **Repulsion control:** airplane only. Teacher discrepancy from sparse batch-loss logs alongside final generated-sample counts for MMU and the matched attraction-only control. Equal counts are not a statistical equivalence result. This control uses 32 forgotten and 288 retained samples, spanning two sampling seeds.

All training uses seed 42 and local implementations. Ten deletion classes are separate tasks, not independent training seeds. No confidence intervals or significance claims. Baseline and MMU source data remain unchanged. PDF and SVG are vector exports; PNG is provided for preview. Exact plotted data are in data.json.
