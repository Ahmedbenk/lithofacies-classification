# Geolink: (1) log-signature comparison for Chalk vs Limestone and Dolomite vs Limestone,
# (2) held-out-wells validation (20% of wells, seed 42).
# Run on Kaggle with Internet ON (pulls the archive with gdown). ~10-15 min.
# Part 2 is an ADDITIONAL validation, not an untouched blind test: model settings were
# chosen earlier from cross-validation over all wells.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown", "lasio"], check=True)
import gdown, lasio, glob, zipfile
import numpy as np, pandas as pd, openpyxl
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

RS = 42
B = ['GR', 'RDEP', 'RHOB', 'NPHI', 'DTC']
FE = ['GR', 'RDEP_log10', 'RHOB', 'NPHI', 'DTC']
gdown.download("https://drive.google.com/uc?id=1ZvLG1SRBQB4mDUmPHSc_6kP-ubS5uNCv", "geolink.zip", quiet=False)
with zipfile.ZipFile("geolink.zip") as z:
    z.extractall("las")
frames = []
for wi, f in enumerate(sorted(glob.glob("las/*.las"))):
    try:
        las = lasio.read(f, engine='normal')
    except Exception:
        continue
    cu = {c.mnemonic.upper(): c.mnemonic for c in las.curves}
    lc = next((c.mnemonic for c in las.curves if 'LITH' in c.mnemonic.upper()), None)
    if lc is None or any(b not in cu for b in B):
        continue
    d = pd.DataFrame({b: np.asarray(las[cu[b]], dtype=np.float32) for b in B})
    d['L'] = np.round(np.asarray(las[lc], dtype=np.float32), 1)
    d = d.dropna()
    d = d[d['RDEP'] > 0]
    if len(d) == 0:
        continue
    d['W'] = wi
    frames.append(d)
D = pd.concat(frames, ignore_index=True)
D['RDEP_log10'] = np.log10(D['RDEP']).astype(np.float32)
D = D.drop(columns=['RDEP'])
vc = D['L'].value_counts()
D = D[D['L'].isin(vc[vc >= 50].index)].reset_index(drop=True)
print("Data:", D.shape, "| wells:", D['W'].nunique(), "| classes:", D['L'].nunique())
names = {}
xl = glob.glob("las/**/*.xlsx", recursive=True)[0]
for r in openpyxl.load_workbook(xl).active.iter_rows(values_only=True, min_row=2):
    if r[0] and r[3] is not None and r[0] != 'NULL':
        try:
            names[round(float(r[3]), 1)] = r[0]
        except Exception:
            pass

def auc(a, b):
    a = pd.Series(np.asarray(a)); b = pd.Series(np.asarray(b))
    ranks = pd.concat([a, b], ignore_index=True).rank()
    return (ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))

def sample(code, per_well_cap=None, n=30000):
    x = D[D['L'] == code]
    if per_well_cap:
        x = x.groupby('W', group_keys=False).apply(lambda g: g.sample(min(len(g), per_well_cap), random_state=RS))
    return x.sample(min(n, len(x)), random_state=RS)

print("\n=== PART 1: LOG SIGNATURES (median [25th, 75th percentile]) ===")
for label, cap in [("all samples", None), ("capped at 2,000 samples per well", 2000)]:
    print(f"\n-- {label} --")
    S = {c: sample(c, cap) for c in (9, 11, 15)}
    for c, nm in ((11, 'Limestone'), (9, 'Chalk'), (15, 'Dolomite')):
        print(f"{nm} ({D[D['L'] == c]['W'].nunique()} wells):",
              {k: f"{S[c][k].median():.2f} [{S[c][k].quantile(.25):.2f}, {S[c][k].quantile(.75):.2f}]" for k in FE})
    for c, nm in ((9, 'Chalk'), (15, 'Dolomite')):
        print(f"AUC {nm} vs Limestone (0.5 = indistinguishable):", {k: round(float(auc(S[c][k], S[11][k])), 2) for k in FE})

wells = np.array(sorted(D['W'].unique()))
test_wells = set(np.random.default_rng(RS).choice(wells, size=int(round(0.2 * len(wells))), replace=False))
te = D['W'].isin(test_wells).values
tr = ~te
print(f"\n=== PART 2: HELD-OUT WELLS: {len(test_wells)} of {len(wells)} wells held out; train rows {tr.sum():,}, test rows {te.sum():,} ===")
uc, yi = np.unique(D['L'].values[tr], return_inverse=True)
sw = np.sqrt(compute_sample_weight('balanced', yi)).astype(np.float32)
model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RS,
                          eval_metric='mlogloss', tree_method='hist')
model.fit(D[FE].values[tr], yi, sample_weight=sw)
pred = uc[model.predict(D[FE].values[te])]
yt = D['L'].values[te]
cls = sorted(set(yt))
p, r, f1, s = precision_recall_fscore_support(yt, pred, labels=cls, zero_division=0)
print(f"Held-out accuracy: {accuracy_score(yt, pred):.4f}")
print(f"Macro-F1: {np.mean(f1):.3f} | balanced accuracy: {np.mean(r):.3f}")
# Cross-validated recall per class from the main Geolink baseline (geolink_baseline_kaggle.py), keyed by lithology code
CV_RECALL = {5: .349397, 6: .466466, 13: .489843, 7: .469095, 2: .339968, 3: .378631, 4: .057082,
             11: .525050, 12: .280873, 1: .307141, 16: .056365, 19: .405893, 9: .231338,
             10: .438323, 18: .660473, 15: .013659, 22: .747677, 21: .223629, 17: 0.0,
             26: .008333, 33: 0.0, 34: 0.0}
rows = []
for i, c in enumerate(cls):
    rows.append({'class': names.get(c, c), 'code': c,
                 'train_wells': D[(D['L'] == c) & tr]['W'].nunique(),
                 'test_wells': D[(D['L'] == c) & te]['W'].nunique(),
                 'test_samples': int(s[i]), 'heldout_recall': round(r[i], 3),
                 'cv_recall': CV_RECALL.get(c, np.nan)})
T = pd.DataFrame(rows).sort_values('train_wells')
T['heldout_minus_cv'] = (T['heldout_recall'] - T['cv_recall']).round(3)
print(T.to_string(index=False))
big = T[T['test_samples'] >= 50]
rho = np.corrcoef(big['train_wells'].rank(), big['heldout_recall'].rank())[0, 1]
print(f"\nSpearman rho, training wells vs held-out recall (n={len(big)}): {rho:.2f}")
T.to_csv("geolink_heldout_results.csv", index=False)
