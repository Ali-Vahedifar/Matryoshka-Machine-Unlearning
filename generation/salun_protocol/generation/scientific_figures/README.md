# Scientific figures: one-seed CIFAR-10 analysis

Generated entirely from saved samples, classifier evaluations, and training logs. No additional training or image generation. PDF, SVG, and PNG versions are provided for each figure.

## Figures

- [Retained-accuracy heatmap](retained_accuracy_heatmap.pdf): each cell evaluates 144 retained-condition samples for one deleted class. Fixed color scale 80–100%; values are printed.
- [Class-transition matrices](class_transition_matrices.pdf): each row evaluates 16 samples under the corresponding forgotten condition from a separately unlearned model. Source is the unchanged reference. All matrices share 0–100%; displayed percentages are rounded to the nearest integer, with exact values saved in plot_data.json.
- [Preservation–forgetting scatter](preservation_forgetting_scatter.pdf): each point is one deletion task. Coordinates are mean paired RGB MSE relative to source, after scaling to [0,1]. Horizontal uses 16 forgotten-condition samples; vertical uses 144 retained-condition samples. Larger horizontal change and smaller vertical change indicate selectivity, but neither proves semantic forgetting or image quality.
- [Three-panel comparison](comparison_analysis.pdf): compact combination of the preceding analyses.
- [MMU mechanism and attraction-only control](mmu_mechanism_control.pdf): airplane only, training seed 42, 1,000 updates. Full-width MMU and an otherwise matched attraction-only control. Lines connect sparse recorded losses; they are not full trajectories. The zero control repulsion curve is disabled by construction, not evidence that the margin was satisfied. Loss magnitude is not gradient magnitude.
- [All figures in one PDF](all_scientific_figures.pdf).

## Interpretation and limitations

All three unlearning methods have zero forgotten-class detections across 160 evaluated samples. Average retained-condition accuracy is MMU 97.92%, Random mask 98.75%, SalUn 94.03%. These are local implementations, not an official SalUn reproduction. Ten deletion classes are tasks, not independent training seeds; no confidence interval or superiority test is claimed.
All methods deliberately redirect forgotten class c toward class (c+1) modulo 10. Transition matrices show whether generation follows that training target; they do not independently demonstrate data removal. Source samples and initial noise are reused across tasks and methods.
The existing airplane control and MMU both produced 0/32 forgotten-class detections and 285/288 retained correct predictions in the earlier two-noise-seed evaluation. Those equal endpoints do not establish a benefit from repulsion. Mechanism curves use that original airplane training log, while the ten-class plots use the single matched sampling seed 10043 (16 samples per condition).

Raw plotted values: [JSON](plot_data.json), [paired errors CSV](paired_changes.csv), [classifier predictions](predictions.json). Reproduce with `python salun_protocol/generation/scientific_figures.py` from generation/.
