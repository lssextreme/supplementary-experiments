# HARNet-Experiments

Supplementary code and raw results for the revision experiments of the paper

> **HARNet: A Hierarchical Attention-enhanced Residual Network for Fine-grained Boundary-aware Segmentation in High-resolution Remote Sensing Imagery** (*Applied Soft Computing*, revised manuscript)

This repository makes every experiment added during the revision fully reproducible and traceable to the reviewer comments. It ships the **complete code and raw results only — no trained weights or checkpoints are distributed**. Each comment below maps to the exact scripts that produced the numbers reported in the response letter, together with the raw per-seed result files and the exact training configuration, so that every reported figure can be re-derived from scratch or independently checked.

This repository: **https://github.com/lssextreme/supplementary-experiments**. The model implementation and the training pipeline live in the main repository:
**https://github.com/lssextreme/HARNet** — this repository is a companion that contains the *revision* experiments, which are non-invasive with respect to the main model code (see `harness/harnet_model.py`).

---

## Repository layout

| Directory | Content |
|---|---|
| `harness/` | Training/evaluation harness for the revision experiments (multi-seed runs, loss-weight sensitivity, mask-only ablation, failure-case visualization, FLOPs/parameter accounting) |
| `analysis/` | Per-comment analysis scripts (coverage stratification, patch-overlap verification, boundary-head diagnostics, figure generation) |
| `results/` | Raw per-seed evaluation JSONs for the Shandong and Netherlands multi-seed runs and the loss-weight sensitivity runs |
| `figures/` | The three figures added in the response letter (per-seed IoU, coverage stratification, failure cases) |

---

## Mapping from reviewer comments to experiments

| Comment | Added experiment / claim | Scripts | Raw results |
|---|---|---|---|
| **R1** — single-run estimates; report mean ± std over ≥3 seeds | Re-ran HARNet and every ablation configuration over seeds 42/123/2024; per-seed IoU figure | `harness/run_multiseed.py`, `harness/stats.py`, `analysis/make_figs12.py` | `results/full_{shandong,netherlands}_seed{42,123,2024}.json`, `figures/fig_per_seed.pdf` |
| **R2** — justify the 1:1:1 loss weighting; isolate the auxiliary heads | Loss-weight sensitivity (four configurations, Table 1 of the revision) and a mask-only single-task variant (Case H of the ablation) | `harness/run_sensitivity.py`, `harness/run_mask_only.py` | `results/full_shandong_seed42_w1.0_*.json`, `results/sensitivity_summary.json` |
| **R3** — trainable vs total parameters; FLOPs convention | Itemized parameter accounting (frozen FastSAM-x encoder 72.2341M; trainable 1.7808M; total 74.0149M) and FLOPs at 512×512 including the frozen encoder | `harness/compute_flops.py`, `analysis/flops_thop.py`, `analysis/flops_tfnet.py`, `analysis/profile_modules.py` | printed to stdout; Table 4 of the revision |
| **R4** — baseline reproduction; TFNet Netherlands outlier | All baselines re-trained in-house with the same split/preprocessing/hyperparameter budget; per-seed baseline results recorded | baseline codebases + `harness/evaluate.py` (shared metric implementation) | baseline means and standard deviations reported in the response letter (Table 2 of the revision), Netherlands diagnostic paragraph in the manuscript |
| **R5** — 40% coverage filter bias | Coverage-stratified evaluation of the *complete, unfiltered* Shandong and Netherlands test sets with the seed-42 checkpoint (no retraining) | `analysis/band_split_eval.py`, `analysis/coverage_stats.py`, `analysis/make_figs12.py` | `analysis/band_split_results.json`, `figures/fig_coverage.pdf` |
| **R6** — 50% overlap between training/validation patches | Verified on all 656 patches that the sliding-window stride is 512 px (no overlap), so no pixel is shared between any two patches | `analysis/check_overlap.py` | printed to stdout |
| **R7** — batch size 2, 150 epochs, gradient accumulation | Confirmed and stated in the manuscript: batch size 2, no gradient accumulation, single GPU | `harness/config.py` (`EPOCHS=150`, `BATCH_SIZE=2`) | — |
| **R8** — failure-case analysis; frozen encoder in the low-label regime | Failure-case quadruplets (input/GT/prediction/overlay) for the three lowest-IoU Sichuan patches (Fig. 10) | `harness/failure_cases.py`, `analysis/make_fig3.py` | `figures/fig_failure.pdf`; per-patch IoU in the figure |
| **R9** — Limitations subsection | Text addition (no experiment) | — | Section 5.1 of the manuscript |
| **R10** — code and data availability | This repository, together with the main repository | — | — |

---

## Environment and prerequisites

The harness runs on top of the main repository: clone **https://github.com/lssextreme/HARNet**, place `harness/` inside it (as `revise_exp/`, the layout used for the revision), and configure `harness/config.py` (dataset root, FastSAM weight path, seeds). The experiments were run with PyTorch (single NVIDIA GPU). `thop` is optional (FLOPs accounting); `scipy` is optional (paired tests in `stats.py`).

Training hyperparameters follow the paper exactly: Adam lr = 0.001, 150 epochs, batch size 2, no gradient accumulation, equal loss weights, 512×512 inputs, 10% of the training patches for validation, confidence threshold 0.4 / IoU threshold 0.9 at inference.

## Model weights

**No trained weights are included in this repository.** All reported numbers are reproducible by retraining from scratch with the commands below; the seeds, data splits, and hyperparameters are fixed in `harness/config.py` and in the `run_*.py` scripts, so a re-run regenerates the checkpoints that produced the reported results.

The only pretrained component is the frozen FastSAM-x encoder, a third-party public release: obtain the official `FastSAM-x.pt` checkpoint from the [FastSAM repository](https://github.com/CASIA-IVA-Lab/FastSAM) and point `FASTSAM_WEIGHT` in `harness/config.py` to it — it is not included here. No other weights are required. The raw per-seed outputs of the Shandong and Netherlands multi-seed runs and the loss-weight sensitivity runs are preserved in `results/`, so their reported means and standard deviations can be checked without retraining.

## Reproduction commands (summary)

```bash
# R1/R4: multi-seed training + evaluation (HARNet and ablations)
python harness/run_multiseed.py --models full no_semfe no_spafe no_eca no_aru no_spafe_no_eca only_aru \
    --datasets shandong netherlands sichuan --seeds 42 123 2024
python harness/stats.py

# R2: loss-weight sensitivity and mask-only variant
python harness/run_sensitivity.py --dataset shandong --seeds 42 123 2024
python harness/run_mask_only.py --datasets shandong --seeds 42 123 2024

# R3: parameters and FLOPs
python harness/compute_flops.py --model full --size 512

# R5: coverage-stratified evaluation on the unfiltered test sets (seed-42 checkpoint)
python analysis/band_split_eval.py
python analysis/make_figs12.py

# R6: patch-overlap verification (656 patches)
python analysis/check_overlap.py

# R8: failure-case visualization (Fig. 10)
python harness/failure_cases.py --dataset sichuan --checkpoint <best.pth> --topk 20
python analysis/make_fig3.py
```

Each `analysis/` script keeps its local paths (dataset directories, checkpoint locations) at the top of the file; adjust them to your machine.

## Data

The Netherlands cadastral parcel vectors are publicly available from the Dutch national geo-information portal (https://www.pdok.nl/). The GF-2 imagery and the Chinese parcel annotations are subject to data-sharing agreements and can be made available on reasonable request from the corresponding author.

## License

[to be completed]

---

*Prepared to accompany the point-by-point response letter of the HARNet revision. All results in `results/` were produced by the scripts in this repository on the configurations stated in the letter.*
