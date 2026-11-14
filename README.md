# Lithofacies classification: a diagnostic framework across three benchmarks

Code for the paper *A Diagnostic Framework for Minority-Class Lithofacies Classification: Three-Dataset Benchmark* (Benkhayal, Nicholson, Mohamed, Farag).

The paper tests a class-weighted single-model XGBoost baseline, four hybrid mechanisms, and a four-way diagnosis of underperforming classes on three public well-log datasets: SEG (Hugoton-Panoma), FORCE 2020, and Geolink.

## What is where

| Script | What it does | Paper |
|---|---|---|
| `code/seg_final_pipeline.py` | SEG baseline: leave-one-well-out CV and blind-well scoring (STUART, CRAWFORD) | Sec. 4.1 |
| `code/lithology_hybrid_pipeline.py` | XGBoost + neural-network stacking hybrid | Sec. 2.4 |
| `revalidation/revalidate_hybrids_xgboost.py` | HMM/Viterbi, cascade specialist and depth-window features, tested with XGBoost (output in `revalidation/`) | Sec. 2.4 |
| `code/force_final_pipeline.py` | FORCE 2020 final model (PEF included, NPHI excluded), 10-fold grouped CV, per-class report with fold presence | Sec. 4.2 |
| `code/geolink_baseline_kaggle.py` | Geolink: parse 223 LAS files, 10-fold grouped CV baseline | Sec. 4.3 |
| `code/geolink_pef_test_kaggle.py` | Geolink PEF-inclusion test on the wells that have PEF | Table 7 |
| `code/generate_logplot_comparison.py` | Actual-vs-predicted well-log plots (run after the SEG pipeline) | Fig. 2 |
| `code/seg_perwell_analysis.py` | SEG per-well comparison of baseline vs HMM/Viterbi (per-well accuracy, balanced accuracy, macro-F1) | Sec. 2.4 |
| `code/geolink_heldout_and_signatures_kaggle.py` | Geolink: 20% held-out-well validation (seed 42) and Chalk/Dolomite vs Limestone log-signature comparison | Sec. 4.3, Tables 8-9 |
| `code/geolink_pef_ci_kaggle.py` | Geolink PEF-inclusion test with paired well-level bootstrap intervals (same 85-well subset and folds) | Sec. 4.3, Table 7 |
| `code/force_pef_ci_kaggle.py` | FORCE PEF-inclusion test with paired well-level bootstrap intervals (same folds, with vs without PEF) | Sec. 4.2, Table 6 |
| `code/geolink_rules_kaggle.py` | Numeric diagnostic rules applied to Geolink 10-fold out-of-fold predictions | Sec. 2.3, Table 4 |
| `code/force_rules_kaggle.py` | Numeric diagnostic rules applied to FORCE 10-fold out-of-fold predictions | Sec. 2.3, Table 4 |
| `code/geolink_uncertainty_kaggle.py`, `code/force_uncertainty_kaggle.py` | Well-level bootstrap intervals, class-by-fold coverage and calibration (Geolink, FORCE) | Sec. 4.5, Tables 10-11 |
| `code/seg_calibration_spacing.py` | SEG calibration (leave-one-well-out) and HMM gain after thinning the sample spacing | Sec. 4.5 |
| `code/make_paper_figures.py` | Figures 1, 3, 4, 5, 6 | Figs. 1, 3-6 |
| `geolink/` | Notes and results from the Geolink survey, baseline, diagnosis and PEF test | n/a |

Note on the rules scripts: the `mutual` column printed by `*_rules_kaggle.py` uses the first, fraction-based criterion (each class sends at least 20% of its samples to the other). The paper judges symmetry from counts instead (this was changed after the first result, as Section 2.3 states), so FORCE Sandstone/Shale is flagged there from the counts and not from the printed column. Logs of all runs are in `results/`. The `*_uncertainty_*` and `*_rules_*` scripts read the LAS files in sorted order and group wells by name, so their cross-validation folds and row order differ from `geolink_baseline_kaggle.py`; their point estimates therefore differ slightly from the main baseline (the paper reports both and says which table uses which). The 36-well hold-out is the same set of wells in both.

Note on FORCE: `force_final_pipeline.py` ends at the cell that inspects the leaderboard blind-set files. The cell that scores that blind set is not included in the script, so the FORCE blind-set numbers in the paper (Fig. 4, Fig. 5) come from a separate run of that step.

## Data (not included)

The datasets are not redistributed here. Get them from the original sources:

- **SEG facies contest**: the SEG machine learning contest repository (`facies_data.csv`, `validation_data_nofacies.csv`, and the core-derived blind labels).
- **FORCE 2020**: https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition
- **Geolink**: Data Underground, https://dataunderground.org/dataset/geolink (223 LAS files plus a lithology code table).

Check each dataset's own license. For Geolink the archive's two license files disagree (NLOD 2.0 in one, CC-BY-4.0 in the other). We asked GEOLINK which applies and had not received a reply at the time of writing. The paper treats the data as CC-BY-4.0; confirm the terms before reusing the data.

## Running the code

Python 3.10+ with `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `openpyxl`, and `lasio`. The stacking pipeline also needs `tensorflow`/`keras`.

- The SEG, FORCE and stacking scripts are written as Colab cells (`# %% CELL n`). The data paths sit at the top of the first cell, so edit them to point at your copies of the files. Some of these scripts mount Google Drive and need that block removed to run elsewhere.
- The Geolink scripts were run on Kaggle with internet enabled. They download the archive with `gdown` from a public Drive link, which Google sometimes blocks when too many people have used it. If that happens, download the archive from Data Underground and point the script at the extracted folder.
- All random seeds are fixed (`random_state=42`). Different XGBoost versions or hardware can still shift results slightly in the third decimal place.

## License

The code is released under the MIT License (see `LICENSE`). The datasets keep their own licenses.


## Final unified runs (v1.1)

All FORCE and Geolink results in the paper (Tables 4 and 6-11, Figures 3-5) come from one run of one script per dataset, using a single fold assignment:

- `code/force_unified_kaggle.py` - FORCE 10-fold CV, rules, bootstrap intervals, calibration, official leaderboard blind-set scoring, PEF and NPHI paired tests. Log: `results/force_unified_log.txt`; tables: `results/force_blind_*.csv`.
- `code/geolink_unified_kaggle.py` - Geolink 10-fold CV, rules, intervals, 36-well hold-out, PEF paired test, log signatures. Log: `results/geolink_unified_log.txt`; table: `results/geolink_heldout_results.csv`.

The earlier per-analysis scripts remain for history; where they differ from the unified runs (Geolink fold/row order), the unified runs are the ones reported.
