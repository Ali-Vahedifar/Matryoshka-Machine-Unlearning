# Figures

Figures are not committed. Each script below writes its figures as PDF under `figures/pdf/`
(git-ignored); the table gives the script that draws each one and the data it reads.

## Regenerated from the committed results (no GPU needed)

```bash
python figures/make_classification_figures.py   # -> pdf/classification/
python figures/make_paper_figures.py            # -> pdf/paper/
```

| PDF | Script | Data |
|---|---|---|
| `pdf/classification/fig1_ulira_roc_{cifar10,cifar100,rti}.pdf` | `make_classification_figures.py` (`fig1`) | `results/classification_benchmark/*/*/seed42/{ulira.json,ulira_scores.npz}` |
| `pdf/classification/fig5_privacy_combined_{…}.pdf` | `make_classification_figures.py` (`fig5`) | seed-42 `final.json` + `ulira.json` |
| `pdf/classification/fig4_unlearning_time.pdf` | `make_classification_figures.py` (`fig4`) | every `final.json` (18 cell-seeds per dataset) |
| `pdf/classification/fig6_recovery_{…}.pdf` | `make_classification_figures.py` (`fig6`) | `relearn_forget_acc` in every `final.json` |
| `pdf/paper/fig1_prefix_depth.pdf` | `make_paper_figures.py` | `results/mmu_prefix_leakage/` |
| `pdf/paper/fig2_ulira_roc.pdf` | `make_paper_figures.py` | CIFAR-10 ResNet-18 instance, seed 42 |
| `pdf/paper/fig3_saliency_null.pdf` | `make_paper_figures.py` | `generation/salun_protocol/sd_concept_one_seed/seed_study.json` |
| `pdf/paper/fig4_margin_dial.pdf` | `make_paper_figures.py` | `generation/salun_protocol/classification/seed1/forget10/tuning/` |
| `pdf/paper/fig5_cross_domain.pdf` | `make_paper_figures.py` | SalUn-protocol classification + SD seed study |

The numbers each figure draws are written to `results/figure_data/`.

## Generation figures (need the sampled images / checkpoints, i.e. a GPU re-run)

Paths are relative to `pdf/generation/` and mirror the source tree under `generation/`.

| PDF | Script |
|---|---|
| `diffusion_mmu/results/<run>/samples_{forget,retain}.pdf` | `generation/diffusion_mmu/summarize_run.py` (single runs), `summarize_improvements.py` (`improvement_20260907`); samples from `run_experiment.py` |
| `salun_protocol/generation/figure4_comparison.pdf` | `generation/salun_protocol/generation_figure.py` |
| `salun_protocol/generation/expanded/all_samples.pdf` | `generation/salun_protocol/generation_expanded.py` |
| `salun_protocol/generation/all_classes_seed42/*.pdf` | `generation/salun_protocol/generation/all_classes.py` |
| `salun_protocol/generation/two_teacher_seed42/*.pdf` | `generation/salun_protocol/generation/two_teacher_trial.py` |
| `salun_protocol/generation/redesigned_figures/*.pdf` | `generation/salun_protocol/generation/redesigned_figures.py` |
| `salun_protocol/generation/scientific_figures/*.pdf` | `generation/salun_protocol/generation/scientific_figures.py` |
| `salun_protocol/mmu2t_strengthening/{analysis,budget_curves,sequential_reappearance}.pdf` | `generation/salun_protocol/mmu2t_strengthening/summarize.py` |
| `salun_protocol/sd_concept_one_seed/fig_{forget,retain}_{1,2}.pdf`, `fig_forget_all.pdf` | `generation/salun_protocol/sd_concept_one_seed/figures.py` |
| `salun_protocol/sd_concept_one_seed/comparison.pdf` | `generation/salun_protocol/sd_concept_one_seed/summarize.py` / `more_figures.py` |
| `salun_protocol/sd_concept_one_seed/two_teacher_seed42/*.pdf` | `generation/salun_protocol/sd_concept_one_seed/two_teacher_seed42/figures.py` |

**Content note:** the Stable Diffusion forget-set figures show adult artistic nudity
from the source model, censored with black boxes (the detector scores were computed
on the uncensored images before censoring).
