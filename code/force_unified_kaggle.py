# ======================= shared analysis helpers (used by both unified scripts) =======================
import numpy as np, pandas as pd

def conf_from(y, p, K):
    C = np.zeros((K, K), dtype=np.int64); np.add.at(C, (y, p), 1); return C

def prf_from(C):
    K = C.shape[0]; tp = np.diag(C).astype(float); sup = C.sum(1).astype(float); pr = C.sum(0).astype(float)
    rec = np.divide(tp, sup, out=np.zeros(K), where=sup > 0)
    prec = np.divide(tp, pr, out=np.zeros(K), where=pr > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    return prec, rec, f1, sup

def agg_from(C):
    """accuracy, macro-F1 and balanced accuracy over classes with support in C."""
    prec, rec, f1, sup = prf_from(C); m = sup > 0
    return C.trace() / C.sum(), f1[m].mean(), rec[m].mean()

def per_well_conf(y, p, wells, K):
    uw = np.unique(wells); wi = {w: i for i, w in enumerate(uw)}
    Cw = np.zeros((len(uw), K, K), dtype=np.int64)
    np.add.at(Cw, (np.array([wi[w] for w in wells]), y, p), 1)
    return uw, Cw

def boot_aggregate(Cw, nb=1000, seed=42):
    """95% well-level bootstrap (wells resampled with replacement) for accuracy, macro-F1, balanced accuracy,
    and per-class recall. Resamples in which a class has no samples give NaN recall and are left out of that class's interval."""
    rng = np.random.default_rng(seed); n = Cw.shape[0]; K = Cw.shape[1]
    A = np.zeros((nb, 3)); R = np.full((nb, K), np.nan)
    for b in range(nb):
        C = Cw[rng.integers(0, n, n)].sum(0); A[b] = agg_from(C)
        _, rec, _, sup = prf_from(C); R[b] = np.where(sup > 0, rec, np.nan)
    return A, R

def ci(v):
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)

def paired_boot(Cw0, Cw1, labels, nb=1000, seed=42):
    """Paired well-level bootstrap of the F1 change (variant 1 minus variant 0), same wells in each resample."""
    rng = np.random.default_rng(seed); n = Cw0.shape[0]; K = Cw0.shape[1]
    C0, C1 = Cw0.sum(0), Cw1.sum(0); f0 = prf_from(C0)[2]; f1 = prf_from(C1)[2]; sup = C0.sum(1)
    D = np.full((nb, K), np.nan); M = np.zeros((nb, 3))
    for b in range(nb):
        s = rng.integers(0, n, n); a = Cw0[s].sum(0); c = Cw1[s].sum(0)
        fa, fc = prf_from(a)[2], prf_from(c)[2]; sp = a.sum(1)
        D[b] = np.where(sp > 0, fc - fa, np.nan); M[b] = (agg_from(c)[1] - agg_from(a)[1], 0, 0)
    print("class | wells | support | F1 without | F1 with | delta | 95% paired well-level interval | share of resamples with delta>0")
    for k in range(K):
        if sup[k] == 0: continue
        lo, hi = ci(D[:, k]); wk = int((Cw0[:, k, :].sum(1) > 0).sum())
        print(f"{labels[k]} | {wk} | {int(sup[k])} | {f0[k]:.3f} | {f1[k]:.3f} | {f1[k]-f0[k]:+.3f} | ({lo:+.3f}, {hi:+.3f}) | {np.nanmean(D[:, k] > 0):.2f}")
    m0 = prf_from(C0)[2][sup > 0].mean(); m1 = prf_from(C1)[2][sup > 0].mean(); lo, hi = ci(M[:, 0])
    print(f"macro-F1 without {m0:.4f} with {m1:.4f} delta {m1-m0:+.4f} ({lo:+.4f}, {hi:+.4f})")

def calibration(P, y, label):
    pred = P.argmax(1); conf = P.max(1); ok = (pred == y).astype(float)
    edges = np.linspace(0, 1, 11); ece = 0.0; rows = []
    for i in range(10):
        mk = (conf > edges[i]) & (conf <= edges[i + 1])
        if mk.sum(): ece += mk.mean() * abs(ok[mk].mean() - conf[mk].mean()); rows.append((f"({edges[i]:.1f},{edges[i+1]:.1f}]", int(mk.sum()), round(conf[mk].mean(), 3), round(ok[mk].mean(), 3)))
    brier = float(np.mean(np.sum((P - np.eye(P.shape[1])[y]) ** 2, 1)))
    print(f"CALIBRATION {label}: accuracy {ok.mean():.4f}, mean confidence {conf.mean():.4f}, ECE {ece:.4f}, Brier {brier:.4f}")
    for r in rows: print("   bin", r)

def fold_table(y, fold, K, labels, wells_of_class):
    rows = []
    for k in range(K):
        if (y == k).sum() == 0: continue
        per = [int(((y == k) & (fold == f)).sum()) for f in range(int(fold.max()) + 1)]
        rows.append({'class': labels[k], 'wells': int(wells_of_class[k]), 'folds_present': int(sum(v > 0 for v in per)), 'samples_per_fold': per})
    return pd.DataFrame(rows).sort_values('wells')

def rules_table(tag, C, labels, fam, wells_of_class, folds_present):
    """Numeric diagnostic rules (paper Section 2.3). Flags are evaluated for every class.
    underperforming: F1 < 0.30 (a label only).  scarce: <=10 wells or <=7 of 10 folds.
    adjacent: most frequent wrong class is in the same family and holds >=30% of the class's errors.
    mutual (exploratory): same family, the two classes are each other's most frequent wrong class, and the two
    directional error counts differ by a factor below 1.25."""
    K = C.shape[0]; prec, rec, f1, sup = prf_from(C); top = {}; rows = []
    for k in range(K):
        e = C[k].copy(); e[k] = 0; top[k] = int(e.argmax()) if e.sum() else k
    for k in range(K):
        if sup[k] == 0: continue
        j = top[k]; e = C[k].copy(); e[k] = 0; ne = e.sum()
        share = e[j] / ne if ne else 0.0; frac = C[k, j] / sup[k]; rev = C[j, k] / sup[j] if sup[j] else 0.0
        same = fam.get(labels[k]) is not None and fam.get(labels[k]) == fam.get(labels[j])
        n1, n2 = C[k, j], C[j, k]; ratio = (max(n1, n2) / min(n1, n2)) if min(n1, n2) > 0 else np.inf
        rows.append({'class': labels[k], 'wells': int(wells_of_class[k]), 'folds': int(folds_present[k]), 'F1': round(f1[k], 3),
                     'under': bool(f1[k] < 0.30), 'top_wrong': labels[j], 'share_of_errors': round(share, 2),
                     'n_k_to_j': int(n1), 'n_j_to_k': int(n2), 'pct_of_class': round(frac, 2), 'pct_of_partner': round(rev, 2),
                     'scarce': bool(wells_of_class[k] <= 10 or folds_present[k] <= 7), 'adjacent': bool(same and share >= 0.30),
                     'mutual': bool(same and top[j] == k and ratio < 1.25), 'count_ratio': round(float(ratio), 2) if np.isfinite(ratio) else 'inf'})
    print(f"\n### RULES: {tag}"); print(pd.DataFrame(rows).sort_values('wells').to_string(index=False))

def cv_analysis(tag, y, P, fold, wells, labels, fam, K):
    """Everything derived from one set of out-of-fold probabilities (so every table comes from the same predictions)."""
    pred = P.argmax(1); C = conf_from(y, pred, K); prec, rec, f1, sup = prf_from(C)
    acc, mf, ba = agg_from(C)
    wpc = [len(set(wells[y == k])) for k in range(K)]; fp = [len(set(fold[y == k])) for k in range(K)]
    print(f"\n##### {tag}: CROSS-VALIDATION  accuracy {acc:.4f} macro-F1 {mf:.4f} balanced accuracy {ba:.4f} (n={len(y)}, wells={len(set(wells))})")
    print(pd.DataFrame({'class': labels, 'wells': wpc, 'folds_present': fp, 'precision': prec.round(3), 'recall': rec.round(3), 'f1': f1.round(3), 'support': sup.astype(int)}).sort_values('wells').to_string(index=False))
    rules_table(tag, C, labels, fam, wpc, fp)
    uw, Cw = per_well_conf(y, pred, wells, K); A, R = boot_aggregate(Cw)
    print(f"\n95% WELL-LEVEL BOOTSTRAP ({tag} CV, 1000 resamples of {len(uw)} wells): accuracy {ci(A[:,0])} macro-F1 {ci(A[:,1])} balanced accuracy {ci(A[:,2])}")
    print("class | wells | CV recall | 95% interval | share of resamples with class present")
    for k in np.argsort(wpc):
        if sup[k] > 0: lo, hi = ci(R[:, k]); print(f"{labels[k]} | {wpc[k]} | {rec[k]:.3f} | ({lo:.3f}, {hi:.3f}) | {np.mean(~np.isnan(R[:, k])):.2f}")
    print("\nCLASS-BY-FOLD SAMPLE COUNTS"); print(fold_table(y, fold, K, labels, wpc).to_string(index=False))
    calibration(P, y, tag + " CV")
    return pred, Cw

def save_preds(path, **arrs):
    np.savez_compressed(path, **arrs)

# ======================= FORCE 2020: one fold setup, every table from the same predictions, plus the blind-set scoring =======================
# Kaggle script kernel, Internet ON. Same features, per-fold median imputation, folds (StratifiedGroupKFold by well, shuffled, seed 42)
# and XGBoost settings as the original pipeline. XGBoost uses 4 threads.
import subprocess, sys, time, glob, zipfile, gc, os
subprocess.run("git clone --depth 1 https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition force", shell=True, check=True)
DD = "force/lithology_competition/data"; print(sorted(os.listdir(DD)))
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
RS, NJ = 42, 4
BASE = ['GR', 'RDEP', 'RMED', 'RHOB', 'DRHO', 'DTC', 'CALI', 'PEF']
NOPEF = [c for c in BASE if c != 'PEF']; PLUS_NPHI = BASE + ['NPHI']
T0 = time.time()
def tick(m): print(f"[{(time.time()-T0)/60:.1f} min] {m}", flush=True)
LN = {30000: 'Sandstone', 65030: 'Sandstone/Shale', 65000: 'Shale', 80000: 'Marl', 74000: 'Dolomite', 70000: 'Limestone',
      70032: 'Chalk', 88000: 'Halite', 86000: 'Anhydrite', 99000: 'Tuff', 90000: 'Coal', 93000: 'Basement'}
labels = list(LN.values()); K = len(labels); C2I = {n: i for i, n in enumerate(labels)}
FAM = {'Sandstone': 'silic', 'Sandstone/Shale': 'silic', 'Shale': 'silic', 'Limestone': 'carb', 'Chalk': 'carb', 'Dolomite': 'carb', 'Marl': 'carb', 'Halite': 'evap', 'Anhydrite': 'evap'}
LAB = 'FORCE_2020_LITHOFACIES_LITHOLOGY'

zipfile.ZipFile(f"{DD}/train.zip").extractall("tr")
df = pd.read_csv(glob.glob("tr/*.csv")[0], sep=';', usecols=['WELL', LAB] + sorted(set(PLUS_NPHI)))
y = df[LAB].map(LN).map(C2I).values.astype(np.int32); g = df['WELL'].values
tick(f"train {df.shape}, wells {len(set(g))}")

def fit_probs(Xtr, ytr, Xte):
    imp = SimpleImputer(strategy='median'); uc, yi = np.unique(ytr, return_inverse=True)
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RS, eval_metric='mlogloss', tree_method='hist', n_jobs=NJ)
    m.fit(imp.fit_transform(Xtr), yi, sample_weight=np.sqrt(compute_sample_weight('balanced', yi)))
    P = np.zeros((len(Xte), K), dtype=np.float32); P[:, uc] = m.predict_proba(imp.transform(Xte)); del m; gc.collect(); return P, imp

def cv_probs(cols):
    X = df[cols].values.astype(np.float32); P = np.zeros((len(y), K), dtype=np.float32); fold = np.zeros(len(y), dtype=np.int32)
    for fi, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RS).split(X, y, groups=g)):
        P[te] = fit_probs(X[tr], y[tr], X[te])[0]; fold[te] = fi; tick(f"{len(cols)} curves, fold {fi+1}/10")
    return P, fold

# ---------------- PART A: baseline cross-validation ----------------
P, fold = cv_probs(BASE)
pred, Cw = cv_analysis("FORCE", y, P, fold, g, labels, FAM, K)
np.savez_compressed("force_cv_predictions.npz", y=y, pred=pred, fold=fold, wells=g)
rec_cv = prf_from(conf_from(y, pred, K))[1]

# ---------------- PART B: blind scoring on the official leaderboard wells ----------------
print("\n##### FORCE BLIND SET (official leaderboard wells), model trained once on all 98 training wells")
Xall = df[BASE].values.astype(np.float32)
uc, yi = np.unique(y, return_inverse=True)
final = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RS, eval_metric='mlogloss', tree_method='hist', n_jobs=NJ)
imp_f = SimpleImputer(strategy='median'); final.fit(imp_f.fit_transform(Xall), yi, sample_weight=np.sqrt(compute_sample_weight('balanced', yi))); tick("final model fitted")
def read_any(pattern):
    f = sorted(glob.glob(pattern))[0]; print("reading", f); return pd.read_csv(f, sep=';')
lf = read_any(f"{DD}/leaderboard_test_features*"); lt = read_any(f"{DD}/leaderboard_test_target*")
print("features:", lf.shape, list(lf.columns)); print("target:", lt.shape, list(lt.columns))
if 'WELL' in lt.columns and 'DEPTH_MD' in lt.columns:
    keys = ['WELL', 'DEPTH_MD']; lb = lf.merge(lt[keys + [LAB]], on=keys, how='inner'); assert len(lb) == len(lf), "alignment lost rows"
else:
    assert len(lf) == len(lt), "features and target differ in length and share no keys"; lb = lf.copy(); lb[LAB] = lt[LAB].values
lb = lb[lb[LAB].isin(LN)].reset_index(drop=True)
yb = lb[LAB].map(LN).map(C2I).values.astype(np.int32); wb = lb['WELL'].values
Pb = np.zeros((len(lb), K), dtype=np.float32); Pb[:, uc] = final.predict_proba(imp_f.transform(lb[BASE].values.astype(np.float32))); pb = Pb.argmax(1)
Cb = conf_from(yb, pb, K); prec, rec, f1, sup = prf_from(Cb); acc, mf, ba = agg_from(Cb)
print(f"Blind set: {len(yb):,} samples, {len(set(wb))} wells. Accuracy {acc:.4f}, macro-F1 {mf:.4f}, balanced accuracy {ba:.4f}")
uwb, Cwb = per_well_conf(yb, pb, wb, K); Ab, Rb = boot_aggregate(Cwb)
print(f"95% well-level bootstrap, blind ({len(uwb)} wells): accuracy {ci(Ab[:,0])} macro-F1 {ci(Ab[:,1])} balanced accuracy {ci(Ab[:,2])}")
wpc = [len(set(g[y == k])) for k in range(K)]
T = pd.DataFrame({'class': labels, 'train_wells': wpc, 'blind_wells': [len(set(wb[yb == k])) for k in range(K)], 'blind_samples': sup.astype(int),
                  'blind_precision': prec.round(3), 'blind_recall': np.where(sup > 0, rec.round(3), np.nan), 'blind_f1': f1.round(3), 'cv_recall': rec_cv.round(3)}).sort_values('train_wells')
print(T.to_string(index=False)); T.to_csv("force_blind_vs_cv.csv", index=False)
rown = np.divide(Cb, Cb.sum(1, keepdims=True), out=np.full(Cb.shape, np.nan), where=Cb.sum(1, keepdims=True) > 0)
pd.DataFrame(rown, index=labels, columns=labels).round(3).to_csv("force_blind_confusion_rownorm.csv"); pd.DataFrame(Cb, index=labels, columns=labels).to_csv("force_blind_confusion_counts.csv")
print("\nPER-WELL BLIND RECALL for Chalk, Anhydrite, Dolomite")
for nm in ['Chalk', 'Anhydrite', 'Dolomite']:
    k = labels.index(nm)
    for w in np.unique(wb[yb == k]):
        m = (wb == w) & (yb == k); print(f"  {nm} | well {w} | samples {int(m.sum())} | recall {np.mean(pb[m] == k):.3f}")
calibration(Pb, yb, "FORCE blind")
np.savez_compressed("force_blind_predictions.npz", y=yb, pred=pb, wells=wb)
del final, Pb; gc.collect()

# ---------------- PART C: PEF and NPHI inclusion tests on the same folds ----------------
Pn, _ = cv_probs(NOPEF); tick("no-PEF CV done")
print("\n##### FORCE PEF TEST (paired, well-level): without PEF (7 curves) vs baseline with PEF (8 curves)")
_, Cw0 = per_well_conf(y, Pn.argmax(1), g, K); paired_boot(Cw0, Cw, labels); del Pn
Pq, _ = cv_probs(PLUS_NPHI); tick("+NPHI CV done")
print("\n##### FORCE NPHI TEST (paired, well-level): baseline (8 curves) vs baseline plus NPHI (9 curves)")
_, Cw2 = per_well_conf(y, Pq.argmax(1), g, K); paired_boot(Cw, Cw2, labels)
print("DONE")
