# SEG: per-well comparison of the baseline and HMM/Viterbi smoothing (paper Section 2.4).
#
# Leave-one-well-out XGBoost with dampened class weighting, then Viterbi smoothing
# of the predicted probabilities. Reports pooled accuracy, balanced accuracy and
# macro-F1, per-well accuracy and macro-F1 for both methods, and a bootstrap
# interval over wells for the per-well differences.
# Needs facies_data.csv from the SEG contest in the working folder.

import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
RS = 42; np.random.seed(RS)
F = ['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS']
df = pd.read_csv('facies_data.csv'); df.columns = [c.strip() for c in df.columns]
X = df[F]; y = df['Facies'] - 1; g = df['Well Name']; K = 9
def mk(): return xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=RS, eval_metric='mlogloss')
def trans(tr):
    C = np.ones((K,K)); I = np.ones(K)
    for w, gg in tr.groupby('Well Name'):
        s = gg.sort_values('Depth')['Facies'].values - 1; I[s[0]] += 1
        for a, b in zip(s[:-1], s[1:]): C[a,b] += 1
    return C / C.sum(1, keepdims=True), I / I.sum()
def viterbi(E, T, I, eps=1e-12):
    n = len(E); lT = np.log(T+eps); d = np.zeros((n,K)); bp = np.zeros((n,K), int)
    d[0] = np.log(I+eps) + np.log(E[0]+eps)
    for t in range(1, n):
        s = d[t-1][:,None] + lT; bp[t] = s.argmax(0); d[t] = s.max(0) + np.log(E[t]+eps)
    p = np.zeros(n, int); p[-1] = d[-1].argmax()
    for t in range(n-2, -1, -1): p[t] = bp[t+1, p[t+1]]
    return p
base = np.zeros(len(y), int); hmm = np.zeros(len(y), int)
for tr, te in LeaveOneGroupOut().split(X, y, g):
    sw = np.sqrt(compute_sample_weight('balanced', y.iloc[tr]))
    m = mk(); m.fit(X.iloc[tr], y.iloc[tr], sample_weight=sw)
    base[te] = m.predict(X.iloc[te])
    ted = df.iloc[te].sort_values('Depth'); E = m.predict_proba(X.loc[ted.index])
    T, I = trans(df.iloc[tr]); hmm[df.index.get_indexer(ted.index)] = viterbi(E, T, I)
print("POOLED (leave-one-well-out, 8 wells)")
for name, p in [('baseline', base), ('HMM', hmm)]:
    print(f"{name:9s} acc={accuracy_score(y,p):.4f} balanced_acc={balanced_accuracy_score(y,p):.4f} macroF1={f1_score(y,p,average='macro'):.4f}")
rows = []
for w in sorted(g.unique()):
    i = (g == w).values
    r = {'well': w, 'n': int(i.sum())}
    for name, p in [('base', base), ('hmm', hmm)]:
        r[f'acc_{name}'] = accuracy_score(y[i], p[i]); r[f'mf1_{name}'] = f1_score(y[i], p[i], average='macro')
    rows.append(r)
R = pd.DataFrame(rows); R['d_acc'] = R.acc_hmm - R.acc_base; R['d_mf1'] = R.mf1_hmm - R.mf1_base
print(R.round(4).to_string(index=False))
rng = np.random.default_rng(RS)
for col in ['d_acc', 'd_mf1']:
    v = R[col].values; bs = [rng.choice(v, len(v)).mean() for _ in range(10000)]
    print(f"{col}: mean={v.mean():+.4f} wells improved={(v>0).sum()}/8 bootstrap95%CI over wells=[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]")
