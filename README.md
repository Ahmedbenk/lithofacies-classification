# Lithofacies classification: a diagnostic framework across three benchmarks

Code for the paper *A Diagnostic Framework for Minority-Class Lithofacies Classification: Three-Dataset Benchmark* (Benkhayal, Mohamed, Farag).

The paper tests a calibrated single-model XGBoost baseline, four hybrid mechanisms, and a four-way diagnosis of underperforming classes on three public well-log datasets: SEG (Hugoton-Panoma), FORCE 2020, and Geolink.

## What is where

| Script | What it does | Paper |
|---|---|---|
| `code/seg_final_pipeline.py` | SEG baseline: leave-one-well-out CV and blind-well scoring (STUART, CRAWFORD) | Sec. 4.1 |
| `code/lithology_hybrid_pipeline.py` | XGBoost + neural-network stacking hybrid | Sec. 2.4 |
| `revalidation/revalidate_hybrids_xgboost.py` | HMM/Viterbi, cascade specialist and depth-window features, tested with XGBoost (output in `revalidation/`) | Sec. 2.4 |
| `code/force_final_pipeline.py` | FORCE 2020 final model (PEF included, NPHI excluded), 10-fold grouped CV, per-class report with fold presence | Sec. 4.2 |
| `code/geolink_baseline_kaggle.py` | Geolink: parse 223 LAS files, 10-fold grouped CV baseline | Sec. 4.3 |
| `code/geolink_pef_test_kaggle.py` | Geolink PEF-inclusion test on the wells that have PEF | Table 4 |
| `code/generate_logplot_comparison.py` | Actual-vs-predicted well-log plots (run after the SEG pipeline) | Fig. 2 |
| `code/seg_perwell_analysis.py` | SEG per-well comparison of baseline vs HMM/Viterbi (per-well accuracy, balanced accuracy, macro-F1) | Sec. 2.4 |
| `code/geolink_heldout_and_signatures_kaggle.py` | Geolink: 20% held-out-well validation (seed 42) and Chalk/Dolomite vs Limestone log-signature comparison | Sec. 4.3, Tables 5-6 |
| `code/make_paper_figures.py` | Figures 1, 3, 4, 5, 6 | Figs. 1, 3-6 |
| `geolink/` | Notes and results from the Geolink survey, baseline, diagnosis and PEF test | n/a |

Note on FORCE: `force_final_pipeline.py` ends at the cell that inspects the leaderboard blind-set files. The cell that scores that blind set is not included in the script, so the FORCE blind-set numbers in the paper (Fig. 4, Fig. 5) come from a separate run of that step.

## Data (not included)

The datasets are not redistributed here. Get them from the original sources:

- **SEG facies contest**: the SEG machine learning contest repository (`facies_data.csv`, `validation_data_nofacies.csv`, and the core-derived blind labels).
- **FORCE 2020**: https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition
- **Geolink**: Data Underground, https://dataunderground.org/dataset/geolink (223 LAS files plus a lithology code table).

Check each dataset's own license. For Geolink the archive's two license files disagree (NLOD 2.0 in one, CC-BY-4.0 in the other). We asked GEOLINK which applies and had not received a reply at the time of writing, so confirm the terms before reusing the data.

## Running the code

Python 3.10+ with `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `openpyxl`, and `lasio`. The stacking pipeline also needs `tensorflow`/`keras`.

- The SEG, FORCE and stacking scripts are written as Colab cells (`# %% CELL n`). The data paths sit at the top of the first cell, so edit them to point at your copies of the files. Some of these scripts mount Google Drive and need that block removed to run elsewhere.
- The Geolink scripts were run on Kaggle with internet enabled. They download the archive with `gdown` from a public Drive link, which Google sometimes blocks when too many people have used it. If that happens, download the archive from Data Underground and point the script at the extracted folder.
- All random seeds are fixed (`random_state=42`). Different XGBoost versions or hardware can still shift results slightly in the third decimal place.

## License

The code is released under the MIT License (see `LICENSE`). The datasets keep their own licenses.
