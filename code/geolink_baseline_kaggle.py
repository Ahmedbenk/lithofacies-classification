# Geolink baseline: calibrated single XGBoost, full dataset, 10-fold grouped CV.
#
# Run in a Kaggle notebook with Internet turned ON (Settings -> Internet).
# Downloads the Geolink archive with gdown from a public Drive link. If Google
# blocks the download ("too many users have viewed or downloaded this file"),
# download the archive from Data Underground instead and unzip it into ./las.
#
# Same method as the SEG and FORCE pipelines: dampened (sqrt) class weighting,
# well-grouped cross-validation, per-class metrics from pooled out-of-fold
# predictions. Used for Section 4.3 of the paper.

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown", "lasio"], check=True)

import gdown, lasio, glob, gc, zipfile
import pandas as pd
import numpy as np
import openpyxl
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
BASELINE_CURVES = ['GR', 'RDEP', 'RHOB', 'NPHI', 'DTC']
FEATURES = ['GR', 'RDEP_log10', 'RHOB', 'NPHI', 'DTC']

# Step 1: data. The lithology code table (xlsx) is inside the same zip.
gdown.download("https://drive.google.com/uc?id=1ZvLG1SRBQB4mDUmPHSc_6kP-ubS5uNCv", "geolink.zip", quiet=False)
with zipfile.ZipFile("geolink.zip") as z:
    z.extractall("las")
files = glob.glob("las/*.las")
print("LAS files found:", len(files))

# Step 2: parse each well, keep rows with all five curves and a label
frames = []
for f in files:
    try:
        las = lasio.read(f, engine='normal')
    except Exception:
        continue
    curves = {c.mnemonic.upper(): c.mnemonic for c in las.curves}
    lith_col = next((c.mnemonic for c in las.curves if 'LITH' in c.mnemonic.upper()), None)
    if lith_col is None or any(c not in curves for c in BASELINE_CURVES):
        continue
    well_name = las.well['WELL'].value if 'WELL' in las.well else f.split('/')[-1]
    d = {c: np.asarray(las[curves[c]], dtype=np.float32) for c in BASELINE_CURVES}
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
print("Cleaned dataset:", combined.shape, "| wells:", combined['WELL'].nunique())

# Drop classes with fewer than 50 samples
vc = combined['LITH_CODE'].value_counts()
combined = combined[combined['LITH_CODE'].isin(vc[vc >= 50].index)].reset_index(drop=True)
print("After min-50 filter:", combined.shape, "| classes:", combined['LITH_CODE'].nunique())

# Step 3: lithology code -> name
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
X = combined[FEATURES].values.astype(np.float32)
y = combined['LITH_CODE'].map(code_to_idx).values.astype(np.int32)
groups = combined['WELL'].values
del combined
gc.collect()

# Step 4: 10-fold grouped CV, dampened class weighting
n_splits = 10
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
    print(f"Fold {fold_i + 1}/{n_splits} done", flush=True)

acc = accuracy_score(y, oof_pred)
prec, rec, f1, support = precision_recall_fscore_support(
    y, oof_pred, labels=range(len(classes_sorted)), zero_division=0)
results = pd.DataFrame({
    'lithology': [code_to_name.get(c, f"CODE_{c}") for c in classes_sorted],
    'code': classes_sorted, 'precision': prec, 'recall': rec, 'f1': f1, 'support': support,
}).sort_values('support', ascending=False)

print(f"\nGeolink baseline, full dataset, accuracy: {acc:.4f}")
print(results.to_string(index=False))
results.to_csv("phase1_full_results.csv", index=False)
