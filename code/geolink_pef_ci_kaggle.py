# Geolink PEF-inclusion test with paired well-level bootstrap intervals (same 85-well subset and folds as the original test).
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown", "lasio"], check=True)
import gdown, lasio, glob, zipfile, gc
import numpy as np, pandas as pd, openpyxl, xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
RS = 42; BASE = ['GR', 'RDEP', 'RHOB', 'NPHI', 'DTC']; ALL = BASE + ['PEF']
gdown.download("https://drive.google.com/uc?id=1ZvLG1SRBQB4mDUmPHSc_6kP-ubS5uNCv", "geolink.zip", quiet=False)
zipfile.ZipFile("geolink.zip").extractall("las")
frames = []
for f in glob.glob("las/*.las"):
    try: las = lasio.read(f, engine='normal')
    except Exception: continue
    cu = {c.mnemonic.upper(): c.mnemonic for c in las.curves}
    lc = next((c.mnemonic for c in las.curves if 'LITH' in c.mnemonic.upper()), None)
    if lc is None or any(c not in cu for c in ALL): continue
    wn = las.well['WELL'].value if 'WELL' in las.well else f.split('/')[-1]
    d = pd.DataFrame({c: np.asarray(las[cu[c]], dtype=np.float32) for c in ALL}); d['L'] = np.asarray(las[lc], dtype=np.float32)
    d = d.dropna(); d = d[d['RDEP'] > 0]
    if len(d) == 0: continue
    d['W'] = wn; frames.append(d); del las
D = pd.concat(frames, ignore_index=True); D['L'] = D['L'].round(1); D['RDEP_log10'] = np.log10(D['RDEP']).astype(np.float32)
vc = D['L'].value_counts(); D = D[D['L'].isin(vc[vc >= 50].index)].reset_index(drop=True)
print("PEF subset:", D.shape, "wells", D['W'].nunique(), "classes", D['L'].nunique())
names = {}
for r in openpyxl.load_workbook(glob.glob("las/**/*.xlsx", recursive=True)[0]).active.iter_rows(values_only=True, min_row=2):
    if r[0] and r[3] is not None and r[0] != 'NULL':
        try: names[round(float(r[3]), 1)] = r[0]
        except Exception: pass
classes = sorted(D['L'].unique()); K = len(classes); y = D['L'].map({c: i for i, c in enumerate(classes)}).values.astype(int); g = D['W'].values
labels = [names.get(c, str(c)) for c in classes]
def oof(cols):
    X = D[cols].values.astype(np.float32); P = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedGroupKFold(n_splits=min(10, D['W'].nunique()), shuffle=True, random_state=RS).split(X, y, groups=g):
        uc, yi = np.unique(y[tr], return_inverse=True)
        m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RS, eval_metric='mlogloss', tree_method='hist')
        m.fit(X[tr], yi, sample_weight=np.sqrt(compute_sample_weight('balanced', yi)).astype(np.float32)); P[te] = uc[m.predict(X[te])]; del m; gc.collect()
    return P
p0 = oof(BASE[:1] + ['RDEP_log10'] + BASE[2:]); print("no-PEF done", flush=True)
p1 = oof(BASE[:1] + ['RDEP_log10'] + BASE[2:] + ['PEF']); print("PEF done", flush=True)

def f1_vec(C):
    K = C.shape[0]; tp = np.diag(C).astype(float); sup = C.sum(1).astype(float); pr = C.sum(0).astype(float)
    rec = np.divide(tp, sup, out=np.zeros(K), where=sup > 0); prec = np.divide(tp, pr, out=np.zeros(K), where=pr > 0)
    return np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0), sup
def pwc(yt, yp, wells, K):
    uw = np.unique(wells); wi = {w: i for i, w in enumerate(uw)}; C = np.zeros((len(uw), K, K), dtype=np.int64)
    np.add.at(C, (np.array([wi[w] for w in wells]), yt, yp), 1); return uw, C
def paired(yt, p0, p1, wells, K, labels, focus, nb=1000):
    uw, C0 = pwc(yt, p0, wells, K); _, C1 = pwc(yt, p1, wells, K)
    f0, sup = f1_vec(C0.sum(0)); f1, _ = f1_vec(C1.sum(0))
    rng = np.random.default_rng(42); n = len(uw); D = []
    for _ in range(nb):
        s = rng.integers(0, n, n); a, _ = f1_vec(C0[s].sum(0)); b, sp = f1_vec(C1[s].sum(0)); d = b - a; d[sp == 0] = np.nan; D.append(d)
    D = np.array(D); print("class | wells | support | F1 without | F1 with | delta | 95% well-level paired interval | share of resamples with delta>0")
    for k in range(K):
        if labels[k] in focus or True:
            if sup[k] == 0: continue
            col = D[:, k]; lo, hi = np.nanpercentile(col, [2.5, 97.5]); wk = int((C0[:, k, :].sum(1) > 0).sum())
            print(f"{labels[k]} | {wk} | {int(sup[k])} | {f0[k]:.3f} | {f1[k]:.3f} | {f1[k]-f0[k]:+.3f} | ({lo:+.3f}, {hi:+.3f}) | {np.nanmean(col > 0):.2f}")
    print("macro-F1 without", round(f0[sup > 0].mean(), 4), "with", round(f1[sup > 0].mean(), 4))
paired(y, p0, p1, g, K, labels, ['Dolomite','Calcareous Cement','Limestone','Chalk']); print('DONE')
