# FORCE PEF-inclusion test with paired well-level bootstrap intervals (same folds; with vs without PEF).
import subprocess, glob, zipfile, gc
subprocess.run("git clone --depth 1 https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition force", shell=True, check=True)
zipfile.ZipFile("force/lithology_competition/data/train.zip").extractall("tr")
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
RS = 42; WITH = ['GR', 'RDEP', 'RMED', 'RHOB', 'DRHO', 'DTC', 'CALI', 'PEF']; WITHOUT = [c for c in WITH if c != 'PEF']
LN = {30000: 'Sandstone', 65030: 'Sandstone/Shale', 65000: 'Shale', 80000: 'Marl', 74000: 'Dolomite', 70000: 'Limestone', 70032: 'Chalk', 88000: 'Halite', 86000: 'Anhydrite', 99000: 'Tuff', 90000: 'Coal', 93000: 'Basement'}
CN = list(LN.values()); K = len(CN); C2I = {n: i for i, n in enumerate(CN)}
df = pd.read_csv(glob.glob("tr/*.csv")[0], sep=';', usecols=['WELL', 'FORCE_2020_LITHOFACIES_LITHOLOGY'] + WITH)
y = df['FORCE_2020_LITHOFACIES_LITHOLOGY'].map(LN).map(C2I).values.astype(int); g = df['WELL'].values
def oof(cols):
    X = df[cols].values.astype(np.float32); P = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RS).split(X, y, groups=g):
        imp = SimpleImputer(strategy='median'); uc, yi = np.unique(y[tr], return_inverse=True)
        m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RS, eval_metric='mlogloss', tree_method='hist')
        m.fit(imp.fit_transform(X[tr]), yi, sample_weight=np.sqrt(compute_sample_weight('balanced', yi))); P[te] = uc[m.predict(imp.transform(X[te]))]; del m; gc.collect()
    return P
p0 = oof(WITHOUT); print("no-PEF done", flush=True); p1 = oof(WITH); print("PEF done", flush=True)

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
paired(y, p0, p1, g, K, CN, ['Chalk','Anhydrite','Limestone','Dolomite']); print('DONE')
