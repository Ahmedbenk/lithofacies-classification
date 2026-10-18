# Geolink PEF-inclusion test: targeted remedy for the Dolomite and Calcareous
# Cement missing-feature hypothesis (Table 4 of the paper).
#
# Run in a Kaggle notebook with Internet turned ON. Same data source and
# fallback as geolink_baseline_kaggle.py.
#
# Design: keep only wells that have PEF as well as the five baseline curves,
# then compare a model without PEF to a model with PEF on that SAME well subset,
# so the only thing that changes between the two runs is the feature.

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown", "lasio"], check=True)

import gdown, lasio, glob, gc, zipfile
import pandas as pd
import numpy as np
import openpyxl
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import xgboost as xgb

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
BASE_CURVES = ['GR', 'RDEP', 'RHOB', 'NPHI', 'DTC']
ALL_CURVES = BASE_CURVES + ['PEF']

gdown.download("https://drive.google.com/uc?id=1ZvLG1SRBQB4mDUmPHSc_6kP-ubS5uNCv", "geolink.zip", quiet=False)
with zipfile.ZipFile("geolink.zip") as z:
    z.extractall("las")
files = glob.glob("las/*.las")
print("LAS files found:", len(files))

# Parse, keeping only wells that have every curve including PEF
frames = []
for f in files:
    try:
        las = lasio.read(f, engine='normal')
    except Exception:
        continue
    curves = {c.mnemonic.upper(): c.mnemonic for c in las.curves}
    lith_col = next((c.mnemonic for c in las.curves if 'LITH' in c.mnemonic.upper()), None)
    if lith_col is None or any(c not in curves for c in ALL_CURVES):
        continue
    well_name = las.well['WELL'].value if 'WELL' in las.well else f.split('/')[-1]
    d = {c: np.asarray(las[curves[c]], dtype=np.float32) for c in ALL_CURVES}
    d['LITH_CODE'] = np.asarray(las[lith_col], dtype=np.float32)
    df = pd.DataFrame(d)
    df = df.dropna()
    df = df[df['RDEP'] > 0]
    if len(df) == 0:
        continue
    df['WELL'] = well_name
    frames.append(df)
    del las

combined = pd.concat(frames, ignore_index=True)
combined['LITH_CODE'] = combined['LITH_CODE'].round(1)
combined['RDEP_log10'] = np.log10(combined['RDEP']).astype(np.float32)
del frames
gc.collect()
print("PEF-coverage subset:", combined.shape, "| wells:", combined['WELL'].nunique())

vc = combined['LITH_CODE'].value_counts()
combined = combined[combined['LITH_CODE'].isin(vc[vc >= 50].index)].reset_index(drop=True)
print("After min-50 filter:", combined.shape, "| classes:", combined['LITH_CODE'].nunique())

lith_xlsx = glob.glob("las/**/*.xlsx", recursive=True)[0]
ws = openpyxl.load_workbook(lith_xlsx).active
code_to_name = {}
for name, color, attr, code in ws.iter_rows(values_only=True, min_row=2):
    if name and code and name != 'NULL':
        try:
            code_to_name[round(float(code), 1)] = name
        except Exception:
            pass

classes_sorted = sorted(combined['LITH_CODE'].unique())
code_to_idx = {c: i for i, c in enumerate(classes_sorted)}
groups = combined['WELL'].values
y = combined['LITH_CODE'].map(code_to_idx).values.astype(np.int32)


def run_model(feature_cols, label):
    X = combined[feature_cols].values.astype(np.float32)
    n_splits = min(10, combined['WELL'].nunique())
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(y), dtype=np.int32)
    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        sw = np.sqrt(compute_sample_weight('balanced', y[train_idx])).astype(np.float32)
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                random_state=RANDOM_STATE, eval_metric='mlogloss',
                                tree_method='hist')
        clf.fit(X[train_idx], y[train_idx], sample_weight=sw)
        oof_pred[test_idx] = clf.predict(X[test_idx])
        del clf, sw
        gc.collect()
        print(f"[{label}] fold {fold_i + 1}/{n_splits} done", flush=True)
    acc = accuracy_score(y, oof_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y, oof_pred, labels=range(len(classes_sorted)), zero_division=0)
    return acc, pd.DataFrame({
        'lithology': [code_to_name.get(c, f"CODE_{c}") for c in classes_sorted],
        'code': classes_sorted, 'precision': prec, 'recall': rec, 'f1': f1, 'support': support,
    })


acc_no_pef, res_no_pef = run_model(BASE_CURVES + ['RDEP_log10'], "no-PEF")
acc_pef, res_pef = run_model(BASE_CURVES + ['RDEP_log10', 'PEF'], "with-PEF")

comparison = res_no_pef[['lithology', 'code', 'support', 'f1']].rename(columns={'f1': 'f1_no_pef'})
comparison['f1_with_pef'] = res_pef['f1'].values
comparison['f1_delta'] = comparison['f1_with_pef'] - comparison['f1_no_pef']
comparison = comparison.sort_values('support', ascending=False)

print(f"\nPEF-inclusion test, accuracy without PEF: {acc_no_pef:.4f} | with PEF: {acc_pef:.4f}")
print(comparison.to_string(index=False))
comparison.to_csv("pef_inclusion_test_results.csv", index=False)
