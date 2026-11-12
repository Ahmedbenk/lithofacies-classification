# FORCE 2020: numeric diagnostic rules (paper Section 2.3, Table 4) applied to pooled 10-fold out-of-fold predictions.
# Same features, imputation, folds (StratifiedGroupKFold, seed 42) and XGBoost settings as force_final_pipeline.py.
# The printed 'mutual' column is the first, fraction-based criterion; the paper judges symmetry from counts (see Section 2.3).
# Kaggle, Internet ON.
import subprocess, sys, os, glob, zipfile, gc
subprocess.run("git clone --depth 1 https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition force", shell=True, check=True)
DD = "force/lithology_competition/data"
zipfile.ZipFile(f"{DD}/train.zip").extractall("tr")
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight

RS = 42
FEATURES = ['GR', 'RDEP', 'RMED', 'RHOB', 'DRHO', 'DTC', 'CALI', 'PEF']
LN = {30000: 'Sandstone', 65030: 'Sandstone/Shale', 65000: 'Shale', 80000: 'Marl', 74000: 'Dolomite', 70000: 'Limestone',
      70032: 'Chalk', 88000: 'Halite', 86000: 'Anhydrite', 99000: 'Tuff', 90000: 'Coal', 93000: 'Basement'}
CN = list(LN.values()); K = len(CN); C2I = {n: i for i, n in enumerate(CN)}
df = pd.read_csv(glob.glob("tr/*.csv")[0], sep=';', usecols=['WELL', 'FORCE_2020_LITHOFACIES_LITHOLOGY'] + FEATURES)
y = df['FORCE_2020_LITHOFACIES_LITHOLOGY'].map(LN).map(C2I).values.astype(int)
X = df[FEATURES].values.astype(np.float32); g = df['WELL'].values
print("train", df.shape, "wells", len(set(g)))

def fit(Xtr, ytr):
    imp = SimpleImputer(strategy='median'); Xi = imp.fit_transform(Xtr)
    sw = np.sqrt(compute_sample_weight('balanced', ytr))
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RS, eval_metric='mlogloss', tree_method='hist')
    uc, yi = np.unique(ytr, return_inverse=True)
    m.fit(Xi, yi, sample_weight=sw)
    return imp, m, uc
def predict(imp, m, uc, Xte):
    p = m.predict_proba(imp.transform(Xte)); P = np.zeros((len(Xte), K), dtype=np.float32); P[:, uc] = p; return P

oofP = np.zeros((len(y), K), dtype=np.float32); fold = np.zeros(len(y), dtype=int)
for fi, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RS).split(X, y, groups=g)):
    imp, m, uc = fit(X[tr], y[tr]); oofP[te] = predict(imp, m, uc, X[te]); fold[te] = fi; del m; gc.collect(); print("fold", fi + 1, flush=True)
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

FAM = {'Sandstone':'silic','Sandstone/Shale':'silic','Shale':'silic','Limestone':'carb','Chalk':'carb','Dolomite':'carb','Marl':'carb','Halite':'evap','Anhydrite':'evap'}
C = np.zeros((K, K), dtype=np.int64); np.add.at(C, (y, oof), 1)
wpc = [len(set(g[y == k])) for k in range(K)]; fp = [len(set(fold[y == k])) for k in range(K)]
apply_rules("FORCE", C, CN, FAM, wpc, fp)
print("DONE")
