# Well-level bootstrap intervals, class-by-fold coverage and calibration (paper Section 4.5). Run on Kaggle with Internet ON.
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

oofP = np.zeros((len(y), K), dtype=np.float32); fold = np.zeros(len(y), dtype=np.int32)
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RS)
for fi, (tr, te) in enumerate(sgkf.split(X, y, groups=gn)):
    m, uc = fit(X[tr], y[tr], K)
    oofP[te] = predict_full(m, uc, X[te], K); fold[te] = fi
    del m; gc.collect(); print("fold", fi + 1, flush=True)
oof = oofP.argmax(1)

def metrics_from_conf(C):
    tp = np.diag(C).astype(float); sup = C.sum(1).astype(float); pr = C.sum(0).astype(float)
    rec = np.divide(tp, sup, out=np.zeros(K), where=sup > 0)
    prec = np.divide(tp, pr, out=np.zeros(K), where=pr > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    pres = sup > 0
    return tp.sum() / C.sum(), f1[pres].mean(), rec[pres].mean(), rec, pres

def per_well_conf(yt, yp, wells):
    uw = np.unique(wells); wi = {w: i for i, w in enumerate(uw)}
    C = np.zeros((len(uw), K, K), dtype=np.int64)
    np.add.at(C, (np.array([wi[w] for w in wells]), yt, yp), 1)
    return uw, C

def boot(C, nb=1000, seed=RS):
    rng = np.random.default_rng(seed); n = C.shape[0]
    out = []; recs = []
    for _ in range(nb):
        s = rng.integers(0, n, n); Cs = C[s].sum(0)
        a, mf, ba, rec, pres = metrics_from_conf(Cs)
        out.append((a, mf, ba)); r = rec.copy(); r[~pres] = np.nan; recs.append(r)
    out = np.array(out); recs = np.array(recs)
    ci = lambda v: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))
    return out, recs, ci

def report(tag, yt, yp, wells):
    uw, C = per_well_conf(yt, yp, wells)
    a, mf, ba, rec, pres = metrics_from_conf(C.sum(0))
    out, recs, ci = boot(C)
    print(f"\n### {tag}: wells={len(uw)}  accuracy {a:.4f} {tuple(round(float(x),4) for x in ci(out[:,0]))} | macro-F1 {mf:.4f} {tuple(round(float(x),4) for x in ci(out[:,1]))} | balanced acc {ba:.4f} {tuple(round(float(x),4) for x in ci(out[:,2]))}")
    rows = []
    for k in range(K):
        if C.sum(0)[k].sum() == 0: continue
        lo, hi = ci(recs[:, k]) if np.isfinite(recs[:, k]).any() else (np.nan, np.nan)
        rows.append({'class': names.get(classes[k], classes[k]), 'wells_with_class': int((C[:, k, :].sum(1) > 0).sum()),
                     'support': int(C[:, k, :].sum()), 'recall': round(rec[k], 3), 'ci_lo': round(lo, 3), 'ci_hi': round(hi, 3)})
    T = pd.DataFrame(rows).sort_values('wells_with_class'); print(T.to_string(index=False)); return T, C, uw

T_cv, C_cv, uw_cv = report("CROSS-VALIDATION (10-fold, well-grouped)", y, oof, gn)
T_cv.to_csv("geolink_cv_ci.csv", index=False)

print("\n### CLASS-BY-FOLD (number of distinct wells containing the class in each test fold)")
cf = pd.DataFrame({names.get(c, c): [len(set(gn[(fold == f) & (y == cidx[c])])) for f in range(10)] for c in classes}).T
cf.columns = [f"f{f+1}" for f in range(10)]; cf['folds_present'] = (cf > 0).sum(1); print(cf.to_string())
cf.to_csv("geolink_class_by_fold.csv")

conf = oofP.max(1); corr = (oof == y).astype(float)
bins = np.linspace(0, 1, 16); ece = 0.0; rows = []
for i in range(15):
    mk = (conf > bins[i]) & (conf <= bins[i + 1])
    if mk.sum():
        ece += mk.mean() * abs(corr[mk].mean() - conf[mk].mean()); rows.append((round(bins[i], 2), int(mk.sum()), round(conf[mk].mean(), 3), round(corr[mk].mean(), 3)))
brier = float(np.mean(np.sum((oofP - np.eye(K, dtype=np.float32)[y]) ** 2, 1)))
print(f"\n### CALIBRATION (OOF, top-label): ECE={ece:.4f} | multiclass Brier={brier:.4f} | mean confidence={conf.mean():.3f} vs accuracy={corr.mean():.3f}")
print("bin_lo, n, mean_conf, accuracy:"); [print(r) for r in rows]
print("class, mean predicted probability of true class, recall")
for k in range(K):
    mk = y == k
    if mk.sum(): print(names.get(classes[k], classes[k]), round(float(oofP[mk, k].mean()), 3), round(float((oof[mk] == k).mean()), 3))
del oofP; gc.collect()

wl = np.array(sorted(D['W'].unique()))
tw = set(np.random.default_rng(RS).choice(wl, size=int(round(0.2 * len(wl))), replace=False))
te = D['W'].isin(tw).values; tr = ~te
m, uc = fit(X[tr], y[tr], K)
P = predict_full(m, uc, X[te], K); ho = P.argmax(1)
T_ho, C_ho, uw_ho = report("HELD-OUT 36 WELLS", y[te], ho, D['W'].values[te])
T_ho.to_csv("geolink_heldout_ci.csv", index=False)
for cname in ('Chalk', 'Halite'):
    k = [i for i, c in enumerate(classes) if names.get(c) == cname][0]
    print(f"\nPer-well held-out recall, {cname}:")
    for wi_, w in enumerate(uw_ho):
        s = C_ho[wi_, k, :].sum()
        if s: print(f"  well {w}: n={int(s)}, recall={C_ho[wi_, k, k] / s:.3f}")
print("DONE")
