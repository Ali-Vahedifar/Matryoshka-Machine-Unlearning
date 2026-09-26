# Matryoshka Machine Unlearning (MMU)

Code, final results and figures for **Matryoshka Machine Unlearning**. MMU applies SCRUB's
teacher–student max/min steps at every Matryoshka prefix of the feature vector, so
forgetting is enforced across the whole family of nested readouts rather than only the
full-width classifier. The method is in `mmu/MMU/mmu.py`.

Model checkpoints are not included. Every final number is included, in
`results/MMU_results.xlsx` and in the per-run JSON and log files it is built from.

## Experimental setup → code

| Paper item | Where |
|---|---|
| Class-, subclass- and instance-level deletion | `mmu/scripts/benchmark_unlearning.py`, `benchmarks/run_cell.sh`, protocol in `benchmarks/PROTOCOL.md` |
| Backbones: CNN (FPECNN) and ResNet-18 | `mmu/core/models/backbones.py` |
| Data splits (disjoint train/val/test and MIA calibration/selection/final subsets) | `mmu/scripts/train_unlearning.py`, `mmu/core/datasets/` |
| RTI (20 classes) | `--dataset rti`, loader in `mmu/core/datasets/cifar.py` |
| *Original* (joint training) and *Retrain* (same recipe on D_r only) | `Benchmark._source` / `Benchmark._reference` |
| **MMU** | `mmu/MMU/` |
| *Fine-tuning*, Bad Teacher, Amnesiac (class deletion only), UNSIR, SSD, SalUn, UniCLUN†, SCRUB | `mmu/Machine_Unlearning_baselines/<Method>/` |
| RL, GA, IU, ℓ1-sparse, Boundary Shrink (BS), Boundary Expanding (BE) | official SalUn release (`salun_official/Classification/unlearn/`), driven by `generation/salun_protocol/classification.py` |
| Validation-only selection under a utility constraint vs Retrain; one official-test evaluation | `Benchmark.select` (`--selection utility --utility_tolerance 2.0`) |
| \|ΔD_f\|, retain/test accuracy, MIA, predictive and parameter KL, time | `mmu/core/evaluation/metrics.py` |
| Post-unlearning relearning accuracy | `Benchmark.relearn_probe` (5 Adam epochs on D_f) |
| U-LiRA per-example membership attack (16 shadow models) | `mmu/scripts/ulira.py` |
| Image generation: CIFAR-10 class-conditional DDPM class forgetting | `generation/diffusion_mmu/` |
| Nudity removal: Stable Diffusion v1.4 concept erasure (NudeNet 3.4.2) | `generation/salun_protocol/sd_concept_one_seed/` |
| MMU ablations (15 variants × 3 modes × 3 seeds; margin at lr 5e-3; two-teacher MMU; prefix-depth leakage) | `generation/salun_protocol/{mmu_variants_3seed,mmu_margin_lr005,mmu2t_strengthening}/`, `mmu/MMU/{prefix_leakage,probe}.py` |

## Layout

```
mmu/                             Python root (the scripts put it on sys.path themselves)
  MMU/                           the method
  Machine_Unlearning_baselines/  one folder per baseline
  core/                          datasets, backbones, metrics, training helpers
  scripts/                       benchmark_unlearning.py, ulira.py, train_unlearning.py
  tests/                         unit tests
benchmarks/                      run_cell.sh, run_all.sh, PROTOCOL.md
salun_official/                  official SalUn release (MIT), one import fix (generation/salun_protocol/upstream.patch)
generation/
  diffusion_mmu/                 CIFAR-10 DDPM: model, MMU / SalUn / NegGrad / fine-tune, UA + FID evaluation
  salun_protocol/                SalUn-protocol classification study, DDPM figures, SD nudity study, MMU ablations
figures/                         figure scripts (figures are not committed; see figures/README.md)
results/
  MMU_results.xlsx               all final numbers (tools/build_results_workbook.py)
  classification_benchmark/      <dataset>/<backbone>_adam_<mode>/seed<k>/: final.json, search.json.gz,
                                 manifest.json, ulira.json + ulira_scores.npz (seed 42); <dataset>/logs/
  tables/<dataset>/              summary.md, per-seed and 3-seed CSVs
  mmu_prefix_leakage/            per-width accuracy of Original, Retrain, Fine-tune, SCRUB, MMU
  prefix_depth_runs/             selected configurations read by mmu/MMU/{prefix_leakage,probe}.py
  figure_data/                   the numbers each figure draws
tools/                           build_results_workbook.py, make_tables.py
```

## Setup

```bash
pip install -r requirements.txt
export MMU_DATA=/path/to/data      # cifar-10-batches-py/ and cifar-100-python/
export MMU_WORK=/path/to/outputs   # checkpoints, shadow models, samples (default: ./work)
python -m pytest -q mmu/tests
```

## Reproducing

**Classification benchmark** (datasets × 2 backbones × 3 modes × 3 seeds, then U-LiRA on seed 42):

```bash
bash benchmarks/run_cell.sh cifar100 resnet18 class 42          # one cell-seed
bash benchmarks/run_cell.sh cifar100 resnet18 class 42 ulira    # its U-LiRA attack
PARALLEL=3 bash benchmarks/run_all.sh                          # everything
```

Each cell writes `final.json` (one test evaluation per method, with the selected
configuration and whether it met the utility budget) and `search.json` (every validation
candidate) under `$MMU_WORK/<dataset>/mu/<backbone>_adam_<mode>/seed<k>/`. The Original
model and the Retrain oracle are cached there and shared by every method in the cell.

**RL / GA / IU / ℓ1-sparse / BS / BE** (CIFAR-10 ResNet-18, 10% and 50% random forgetting,
pilot seed 1 for selection, test seeds 2–11):

```bash
cd generation/salun_protocol
python run_classification_study.py
python summarize_classification.py        # -> classification_summary.md
```

**CIFAR-10 DDPM class forgetting.** The UA judge is the benchmark's CIFAR-10 ResNet-18
Original model (`MMU_JUDGE_CKPT`).

```bash
cd generation
python diffusion_mmu/train_source.py
python diffusion_mmu/train_source.py --exclude_class 0 --tag retrain_c0
python diffusion_mmu/run_experiment.py --methods source,retrain,finetune,neggrad,salun,mmu_fullwidth,mmu,mmu_noskip,mmu_retainonly --tag full_redirect --seed 42
python diffusion_mmu/run_experiment.py --methods mmu_fullwidth --eval_checkpoint diffusion_mmu/results/full_redirect_c0_seed42/mmu_fullwidth.pt --guidance 2 --tag full_redirect_cfg2 --seed 42
```

**Stable Diffusion nudity removal:**

```bash
cd generation/salun_protocol/sd_concept_one_seed
python run.py        # training data, SalUn, random mask, MMU variants, NudeNet + CLIP evaluation
python seeds.py      # 3-seed re-run (seed_study.json)
python figures.py    # fig_forget_* / fig_retain_* (censored)
```

**Tables, workbook and figures** (from the committed results, no GPU):

```bash
python tools/make_tables.py
python tools/build_results_workbook.py
python figures/make_classification_figures.py
python figures/make_paper_figures.py
```

## Notes

- Three cells shared the GPU during the campaign, so `unlearn_time` includes contention;
  use `PARALLEL=1` for contention-free timings.
- Amnesiac needs the update ledger recorded during training, so it runs for class
  deletion only and has no U-LiRA row.
- The Stable Diffusion forget-set figures contain censored adult artistic nudity.
