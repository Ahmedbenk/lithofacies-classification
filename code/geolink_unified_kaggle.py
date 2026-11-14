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

# ======================= GEOLINK: one data pass, one fold setup, every table from the same predictions =======================
# Kaggle script kernel, Internet ON. Wells are read in sorted file order and grouped by well name, so the cross-validation
# folds are the same in every part below. Seeds are 42 and XGBoost uses 4 threads.
import subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown", "lasio"], check=True)
import gdown, lasio, glob, zipfile, gc, openpyxl
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
RS, NJ = 42, 4
BASE = ['GR', 'RDEP', 'RHOB', 'NPHI', 'DTC']
FE = ['GR', 'RDEP_log10', 'RHOB', 'NPHI', 'DTC']
T0 = time.time()
def tick(m): print(f"[{(time.time()-T0)/60:.1f} min] {m}", flush=True)

gdown.download("https://drive.google.com/uc?id=1ZvLG1SRBQB4mDUmPHSc_6kP-ubS5uNCv", "geolink.zip", quiet=False)
with zipfile.ZipFile("geolink.zip") as z: z.extractall("las")
FILES = sorted(glob.glob("las/*.las")); SIDX = {f: i for i, f in enumerate(FILES)}
print("LAS files:", len(FILES))

def load_wells(curves):
    frames = []
    for f in FILES:
        try: las = lasio.read(f, engine='normal')
        except Exception: continue
        cu = {c.mnemonic.upper(): c.mnemonic for c in las.curves}
        lc = next((c.mnemonic for c in las.curves if 'LITH' in c.mnemonic.upper()), None)
        if lc is None or any(c not in cu for c in curves): continue
        wn = las.well['WELL'].value if 'WELL' in las.well else f.split('/')[-1]
        d = pd.DataFrame({c: np.asarray(las[cu[c]], dtype=np.float32) for c in curves})
        d['L'] = np.round(np.asarray(las[lc], dtype=np.float32), 1)
        d = d.dropna(); d = d[d['RDEP'] > 0]
        if len(d) == 0: continue
        d['WN'] = wn; d['W'] = SIDX[f]; frames.append(d); del las
    D = pd.concat(frames, ignore_index=True); del frames
    D['RDEP_log10'] = np.log10(D['RDEP']).astype(np.float32); D = D.drop(columns=['RDEP'])
    vc = D['L'].value_counts(); D = D[D['L'].isin(vc[vc >= 50].index)].reset_index(drop=True); gc.collect()
    return D

names = {}
for r in openpyxl.load_workbook(glob.glob("las/**/*.xlsx", recursive=True)[0]).active.iter_rows(values_only=True, min_row=2):
    if r[0] and r[3] is not None and r[0] != 'NULL':
        try: names[round(float(r[3]), 1)] = r[0]
        except Exception: pass

FAM = {}
for n in ['Limestone','Porous Limestone','Argillaceous Limestone','Chalk','Marlstone','Dolomite','Calcareous Cement']: FAM[n] = 'carb'
for n in ['Sandstone','Cross Bedded Sst','Silty Sand','Sandy Silt','Silt','Shaly Silt','Silty Shale','Conglomerate']: FAM[n] = 'silic'
for n in ['Halite','Anhydrite','Carnallite','Kaïnite']: FAM[n] = 'evap'

def fit_probs(Xtr, ytr, Xte, K):
    uc, yi = np.unique(ytr, return_inverse=True)
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RS, eval_metric='mlogloss', tree_method='hist', n_jobs=NJ)
    m.fit(Xtr, yi, sample_weight=np.sqrt(compute_sample_weight('balanced', yi)).astype(np.float32))
    P = np.zeros((len(Xte), K), dtype=np.float32); P[:, uc] = m.predict_proba(Xte); del m; gc.collect(); return P

def cv_probs(X, y, groups, K, n_splits=10):
    P = np.zeros((len(y), K), dtype=np.float32); fold = np.zeros(len(y), dtype=np.int32)
    for fi, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RS).split(X, y, groups=groups)):
        P[te] = fit_probs(X[tr], y[tr], X[te], K); fold[te] = fi; tick(f"fold {fi+1}/{n_splits}")
    return P, fold

# ---------------- PART A: cross-validation on all wells ----------------
D = load_wells(BASE); tick(f"data {D.shape}, wells {D['WN'].nunique()}, classes {D['L'].nunique()}")
classes = sorted(D['L'].unique()); K = len(classes); labels = [names.get(c, str(c)) for c in classes]
y = D['L'].map({c: i for i, c in enumerate(classes)}).values.astype(np.int32)
X = D[FE].values.astype(np.float32); WN = D['WN'].values; WI = D['W'].values
P, fold = cv_probs(X, y, WN, K)
pred, Cw = cv_analysis("GEOLINK", y, P, fold, WN, labels, FAM, K)
np.savez_compressed("geolink_cv_predictions.npz", y=y, pred=pred, fold=fold, wells=WN, classes=np.array(classes))
rec_cv = prf_from(conf_from(y, pred, K))[1]

# ---------------- PART B: 20% of wells held out (same wells as before: seed 42 over sorted usable-well indices) ----------------
wl = np.array(sorted(np.unique(WI))); tw = set(np.random.default_rng(RS).choice(wl, size=int(round(0.2 * len(wl))), replace=False))
te = np.isin(WI, list(tw)); tr = ~te
print(f"\n##### GEOLINK HELD-OUT: {len(tw)} of {len(wl)} wells held out; train rows {tr.sum():,}, test rows {te.sum():,}")
Ph = fit_probs(X[tr], y[tr], X[te], K); ph = Ph.argmax(1); yt = y[te]; wt = WN[te]
Ch = conf_from(yt, ph, K); prec, rec, f1, sup = prf_from(Ch); acc, mf, ba = agg_from(Ch)
print(f"Held-out accuracy {acc:.4f}, macro-F1 {mf:.4f}, balanced accuracy {ba:.4f}")
uwh, Cwh = per_well_conf(yt, ph, wt, K); Ah, Rh = boot_aggregate(Cwh)
print(f"95% well-level bootstrap, held-out ({len(uwh)} wells): accuracy {ci(Ah[:,0])} macro-F1 {ci(Ah[:,1])} balanced accuracy {ci(Ah[:,2])}")
rows = []
for k in range(K):
    if sup[k] == 0: continue
    rows.append({'class': labels[k], 'train_wells': len(set(WN[tr][y[tr] == k])), 'held_out_wells': len(set(wt[yt == k])), 'held_out_samples': int(sup[k]),
                 'held_out_recall': round(rec[k], 3), 'cv_recall': round(rec_cv[k], 3), 'diff': round(rec[k] - rec_cv[k], 3)})
T9 = pd.DataFrame(rows).sort_values('train_wells'); print(T9.to_string(index=False)); T9.to_csv("geolink_heldout_results.csv", index=False)
big = T9[T9['held_out_samples'] >= 50]
print(f"Spearman rho, training wells vs held-out recall (classes with >= 50 held-out samples, n={len(big)}): {big['train_wells'].rank().corr(big['held_out_recall'].rank()):.2f}")
bigcv = T9[T9['held_out_samples'] >= 50]
print(f"Spearman rho, training wells vs CV recall, same classes: {bigcv['train_wells'].rank().corr(bigcv['cv_recall'].rank()):.2f}")
print("\nPER-WELL HELD-OUT RECALL for Chalk, Halite, Dolomite, Anhydrite")
for nm in ['Chalk', 'Halite', 'Dolomite', 'Anhydrite']:
    if nm not in labels: continue
    k = labels.index(nm)
    for w in np.unique(wt[yt == k]):
        m = (wt == w) & (yt == k); print(f"  {nm} | well {w} | samples {int(m.sum())} | recall {np.mean(ph[m] == k):.3f}")
calibration(Ph, yt, "GEOLINK held-out")
np.savez_compressed("geolink_heldout_predictions.npz", y=yt, pred=ph, wells=wt)
del Ph; gc.collect()

# ---------------- PART C: PEF-inclusion test on the wells that have PEF (same folds for both variants) ----------------
D2 = load_wells(BASE + ['PEF']); tick(f"PEF subset {D2.shape}, wells {D2['WN'].nunique()}, classes {D2['L'].nunique()}")
c2 = sorted(D2['L'].unique()); K2 = len(c2); lab2 = [names.get(c, str(c)) for c in c2]
y2 = D2['L'].map({c: i for i, c in enumerate(c2)}).values.astype(np.int32); g2 = D2['WN'].values; ns = min(10, D2['WN'].nunique())
P0, _ = cv_probs(D2[FE].values.astype(np.float32), y2, g2, K2, ns); tick("no-PEF done")
P1, _ = cv_probs(D2[FE + ['PEF']].values.astype(np.float32), y2, g2, K2, ns); tick("PEF done")
print("\n##### GEOLINK PEF TEST (paired, well-level)")
_, Cw0 = per_well_conf(y2, P0.argmax(1), g2, K2); _, Cw1 = per_well_conf(y2, P1.argmax(1), g2, K2)
paired_boot(Cw0, Cw1, lab2); del P0, P1, D2; gc.collect()

# ---------------- PART D: log signatures, Chalk and Dolomite against Limestone ----------------
print("\n##### GEOLINK LOG SIGNATURES (median [25th, 75th percentile])")
def auc(a, b):
    r = pd.concat([pd.Series(np.asarray(a)), pd.Series(np.asarray(b))], ignore_index=True).rank()
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
def sample(code, cap=None, n=30000):
    x = D[D['L'] == code]
    if cap: x = x.groupby('W', group_keys=False).apply(lambda g: g.sample(min(len(g), cap), random_state=RS))
    return x.sample(min(n, len(x)), random_state=RS)
for lab, cap in [("all samples", None), ("capped at 2,000 samples per well", 2000)]:
    print(f"\n-- {lab} --"); S = {c: sample(c, cap) for c in (9, 11, 15)}
    for c, nm in ((11, 'Limestone'), (9, 'Chalk'), (15, 'Dolomite')):
        print(f"{nm} ({D[D['L']==c]['W'].nunique()} wells):", {k: f"{S[c][k].median():.2f} [{S[c][k].quantile(.25):.2f}, {S[c][k].quantile(.75):.2f}]" for k in FE})
    for c, nm in ((9, 'Chalk'), (15, 'Dolomite')):
        print(f"AUC {nm} vs Limestone:", {k: round(auc(S[c][k], S[11][k]), 2) for k in FE})
print("DONE")
