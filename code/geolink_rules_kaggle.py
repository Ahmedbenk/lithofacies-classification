# Geolink: numeric diagnostic rules (paper Section 2.3, Table 4) applied to pooled 10-fold out-of-fold predictions.
# Same data cleaning, folds (StratifiedGroupKFold, seed 42, groups = well name) and model as geolink_baseline_kaggle.py.
# The printed 'mutual' column is the first, fraction-based criterion; the paper judges symmetry from counts (see Section 2.3).
# Kaggle, Internet ON.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown", "lasio"], check=True)
import gdown, lasio, glob, zipfile, gc, json
import numpy as np, pandas as pd, openpyxl
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

RS = 42
B = ['GR', 'RDEP', 'RHOB', 'NPHI', 'DTC']
FE = ['GR', 'RDEP_log10', 'RHOB', 'NPHI', 'DTC']
gdown.download("https://drive.google.com/uc?id=1ZvLG1SRBQB4mDUmPHSc_6kP-ubS5uNCv", "geolink.zip", quiet=False)
with zipfile.ZipFile("geolink.zip") as z:
    z.extractall("las")
files = glob.glob("las/*.las")
sidx = {f: i for i, f in enumerate(sorted(files))}
frames = []
for f in files:
    try:
        las = lasio.read(f, engine='normal')
    except Exception:
        continue
    cu = {c.mnemonic.upper(): c.mnemonic for c in las.curves}
    lc = next((c.mnemonic for c in las.curves if 'LITH' in c.mnemonic.upper()), None)
    if lc is None or any(b not in cu for b in B):
        continue
    wn = las.well['WELL'].value if 'WELL' in las.well else f.split('/')[-1]
    d = pd.DataFrame({b: np.asarray(las[cu[b]], dtype=np.float32) for b in B})
    d['L'] = np.round(np.asarray(las[lc], dtype=np.float32), 1)
    d = d.dropna(); d = d[d['RDEP'] > 0]
    if len(d) == 0:
        continue
    d['WN'] = wn; d['W'] = sidx[f]
    frames.append(d)
    del las
D = pd.concat(frames, ignore_index=True); del frames; gc.collect()
D['RDEP_log10'] = np.log10(D['RDEP']).astype(np.float32); D = D.drop(columns=['RDEP'])
vc = D['L'].value_counts()
D = D[D['L'].isin(vc[vc >= 50].index)].reset_index(drop=True)
print("Data:", D.shape, "| wells:", D['WN'].nunique(), "| classes:", D['L'].nunique())
names = {}
for r in openpyxl.load_workbook(glob.glob("las/**/*.xlsx", recursive=True)[0]).active.iter_rows(values_only=True, min_row=2):
    if r[0] and r[3] is not None and r[0] != 'NULL':
        try: names[round(float(r[3]), 1)] = r[0]
        except Exception: pass
classes = sorted(D['L'].unique()); K = len(classes)
cidx = {c: i for i, c in enumerate(classes)}
y = D['L'].map(cidx).values.astype(np.int32)
X = D[FE].values.astype(np.float32)
gn = D['WN'].values

def fit(Xtr, ytr, nc):
    uc, yi = np.unique(ytr, return_inverse=True)
    sw = np.sqrt(compute_sample_weight('balanced', yi)).astype(np.float32)
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RS,
                          eval_metric='mlogloss', tree_method='hist')
    m.fit(Xtr, yi, sample_weight=sw)
    return m, uc

def predict_full(m, uc, Xte, nc):
    p = m.predict_proba(Xte)
    P = np.zeros((len(Xte), nc), dtype=np.float32); P[:, uc] = p
    return P

# ---------- CV out-of-fold ----------
oofP = np.zeros((len(y), K), dtype=np.float32); fold = np.zeros(len(y), dtype=np.int32)
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RS)
for fi, (tr, te) in enumerate(sgkf.split(X, y, groups=gn)):
    m, uc = fit(X[tr], y[tr], K)
    oofP[te] = predict_full(m, uc, X[te], K); fold[te] = fi
    del m; gc.collect(); print("fold", fi + 1, flush=True)
oof = oofP.argmax(1)

# ---------- explicit diagnostic rules applied to pooled out-of-fold confusion ----------
def apply_rules(tag, C, label, fam, wells_per_class, folds_present):
    sup = C.sum(1); pr = C.sum(0); tp = np.diag(C)
    print(f"\n### RULES: {tag}  (underperforming = CV F1 < 0.30; scarcity = <=10 wells or <=7 of 10 folds; adjacency = top wrong class in same family and >=30% of the class's errors; mutual = top partner takes >=20% of the class and the class takes >=20% of the partner)")
    rows = []
    for k in range(len(label)):
        if sup[k] == 0: continue
        rec = tp[k] / sup[k]; prc = tp[k] / pr[k] if pr[k] else 0.0; f1 = 2 * prc * rec / (prc + rec) if prc + rec else 0.0
        err = C[k].copy(); err[k] = 0; ne = err.sum(); j = int(err.argmax()) if ne else k
        share = err[j] / ne if ne else 0.0; frac = err[j] / sup[k]; rev = C[j, k] / sup[j] if sup[j] else 0.0
        same = fam.get(label[k]) is not None and fam.get(label[k]) == fam.get(label[j])
        s = (wells_per_class[k] <= 10) or (folds_present[k] <= 7)
        a = same and share >= 0.30
        m = same and frac >= 0.20 and rev >= 0.20
        rows.append({'class': label[k], 'wells': int(wells_per_class[k]), 'folds': int(folds_present[k]), 'F1': round(f1, 3), 'under': f1 < 0.30,
                     'top_wrong': label[j], 'share_of_errors': round(share, 2), 'frac_of_class': round(frac, 2), 'reverse_frac': round(rev, 2),
                     'scarce': s, 'adjacent': a, 'mutual': m})
    print(pd.DataFrame(rows).sort_values('wells').to_string(index=False))

FAM = {}
for n in ['Limestone','Porous Limestone','Argillaceous Limestone','Chalk','Marlstone','Dolomite','Calcareous Cement']: FAM[n] = 'carb'
for n in ['Sandstone','Cross Bedded Sst','Silty Sand','Sandy Silt','Silt','Shaly Silt','Silty Shale','Conglomerate']: FAM[n] = 'silic'
for n in ['Halite','Anhydrite','Carnallite','Kaïnite']: FAM[n] = 'evap'

C = np.zeros((K, K), dtype=np.int64); np.add.at(C, (y, oof), 1)
label = [names.get(c, str(c)) for c in classes]
wpc = [len(set(gn[y == k])) for k in range(K)]
fp = [len(set(fold[y == k])) for k in range(K)]
apply_rules("GEOLINK", C, label, FAM, wpc, fp)
print("DONE")
