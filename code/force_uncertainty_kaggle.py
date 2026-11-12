# Well-level bootstrap intervals, class-by-fold coverage and calibration (paper Section 4.5). Run on Kaggle with Internet ON.
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

def mfc(C):
    tp = np.diag(C).astype(float); sup = C.sum(1).astype(float); pr = C.sum(0).astype(float)
    rec = np.divide(tp, sup, out=np.zeros(K), where=sup > 0); prec = np.divide(tp, pr, out=np.zeros(K), where=pr > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0); pres = sup > 0
    return tp.sum() / C.sum(), f1[pres].mean(), rec[pres].mean(), rec, pres
def pwc(yt, yp, wells):
    uw = np.unique(wells); wi = {w: i for i, w in enumerate(uw)}; C = np.zeros((len(uw), K, K), dtype=np.int64)
    np.add.at(C, (np.array([wi[w] for w in wells]), yt, yp), 1); return uw, C
def boot(C, nb=1000):
    rng = np.random.default_rng(RS); n = C.shape[0]; out = []; recs = []
    for _ in range(nb):
        s = rng.integers(0, n, n); a, mf, ba, rec, pres = mfc(C[s].sum(0)); out.append((a, mf, ba)); r = rec.copy(); r[~pres] = np.nan; recs.append(r)
    return np.array(out), np.array(recs)
ci = lambda v: tuple(round(float(x), 4) for x in (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)))
def report(tag, yt, yp, wells):
    uw, C = pwc(yt, yp, wells); a, mf, ba, rec, pres = mfc(C.sum(0)); out, recs = boot(C)
    print(f"\n### {tag}: wells={len(uw)} accuracy {a:.4f} {ci(out[:,0])} | macro-F1 {mf:.4f} {ci(out[:,1])} | balanced acc {ba:.4f} {ci(out[:,2])}")
    rows = []
    for k in range(K):
        if C.sum(0)[k].sum() == 0: continue
        lo, hi = ci(recs[:, k]) if np.isfinite(recs[:, k]).any() else (np.nan, np.nan)
        rows.append({'class': CN[k], 'wells_with_class': int((C[:, k, :].sum(1) > 0).sum()), 'support': int(C[:, k, :].sum()), 'recall': round(rec[k], 3), 'ci_lo': lo, 'ci_hi': hi})
    print(pd.DataFrame(rows).sort_values('wells_with_class').to_string(index=False)); return C, uw

report("FORCE CROSS-VALIDATION (10-fold, well-grouped)", y, oof, g)
print("\n### CLASS-BY-FOLD (distinct wells containing class in each test fold)")
cf = pd.DataFrame({CN[k]: [len(set(g[(fold == f) & (y == k)])) for f in range(10)] for k in range(K)}).T
cf.columns = [f"f{f+1}" for f in range(10)]; cf['folds_present'] = (cf > 0).sum(1); print(cf.to_string())
conf = oofP.max(1); corr = (oof == y).astype(float); bins = np.linspace(0, 1, 16); ece = 0.0; rows = []
for i in range(15):
    mk = (conf > bins[i]) & (conf <= bins[i + 1])
    if mk.sum(): ece += mk.mean() * abs(corr[mk].mean() - conf[mk].mean()); rows.append((round(bins[i], 2), int(mk.sum()), round(conf[mk].mean(), 3), round(corr[mk].mean(), 3)))
brier = float(np.mean(np.sum((oofP - np.eye(K, dtype=np.float32)[y]) ** 2, 1)))
print(f"\n### CALIBRATION (OOF top-label): ECE={ece:.4f} | Brier={brier:.4f} | mean confidence={conf.mean():.3f} vs accuracy={corr.mean():.3f}")
for r in rows: print(r)
print("class, mean predicted probability of true class, recall")
for k in range(K):
    mk = y == k
    if mk.sum(): print(CN[k], round(float(oofP[mk, k].mean()), 3), round(float((oof[mk] == k).mean()), 3))
del oofP; gc.collect()

imp, m, uc = fit(X, y)
tf = pd.read_csv(f"{DD}/leaderboard_test_features.csv", sep=';'); tt = pd.read_csv(f"{DD}/leaderboard_test_target.csv", sep=';')
print("\ntest features", tf.shape, "target", tt.shape, list(tt.columns))
tcol = 'FORCE_2020_LITHOFACIES_LITHOLOGY'
assert len(tf) == len(tt)
yt = tt[tcol].map(LN).map(C2I).values
ok = ~np.isnan(yt.astype(float)); yt = yt[ok].astype(int)
P = predict(imp, m, uc, tf[FEATURES].values.astype(np.float32)[ok]); pr = P.argmax(1)
print("blind wells:", tf['WELL'].nunique(), "rows", ok.sum(), "train-test well overlap:", len(set(tf['WELL']) & set(g)))
C, uw = report("FORCE OFFICIAL BLIND WELLS", yt, pr, tf['WELL'].values[ok])
for cname in ('Chalk', 'Anhydrite'):
    k = C2I[cname]; print(f"\nPer-well blind recall, {cname}:")
    for wi_, w in enumerate(uw):
        s = C[wi_, k, :].sum()
        if s: print(f"  {w}: n={int(s)}, recall={C[wi_, k, k] / s:.3f}")
print("Chalk row of blind confusion (true Chalk -> predicted):", {CN[j]: int(C.sum(0)[C2I['Chalk'], j]) for j in range(K) if C.sum(0)[C2I['Chalk'], j]})
print("DONE")
