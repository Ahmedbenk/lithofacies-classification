"""
Re-validation of three preliminary hybrid mechanisms with the REAL
production model (XGBoost), replacing the RandomForest stand-in used
during initial exploratory testing.

This directly addresses the "Document status" caveat in the AIG-format
manuscript draft: NN-stacking was already validated with the real
pipeline; HMM smoothing, the cascade specialist, and depth-window
features were not. This script fixes that gap.

Methodology matches the settled SEG pipeline exactly: leave-one-well-out
CV, dampened (sqrt) class weighting, same 7 base features, same
adjacent-facies reporting. Only the mechanism under test changes.

Run locally (or paste each # %% CELL block into its own Colab cell).
"""

# %% CELL 1: Load data (run locally instead of Colab -- same file, same content)
import pandas as pd

TRAIN_PATH = 'facies_data.csv'
df = pd.read_csv(TRAIN_PATH)
df.columns = [c.strip() for c in df.columns]
print(f"Rows: {len(df)}, Wells: {df['Well Name'].nunique()}")


# %% CELL 2: Imports
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
import xgboost as xgb

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FEATURES = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
TARGET = 'Facies'
GROUP_COL = 'Well Name'

X = df[FEATURES]
y = df[TARGET] - 1  # zero-index
groups = df[GROUP_COL]
N_CLASSES = y.nunique()


# %% CELL 3: Shared reporting helper (same as the final SEG pipeline)
ADJACENT_FACIES = {
    1: [2], 2: [1, 3], 3: [2], 4: [5], 5: [4, 6],
    6: [5, 7], 7: [6, 8], 8: [6, 7, 9], 9: [7, 8],
}


def adjacent_accuracy(y_true_1idx, y_pred_1idx):
    correct = 0
    for t, p in zip(y_true_1idx, y_pred_1idx):
        if p == t or p in ADJACENT_FACIES.get(t, []):
            correct += 1
    return correct / len(y_true_1idx)


def per_facies_report(y_true_0idx, y_pred_0idx, label=""):
    y_true_1 = np.asarray(y_true_0idx) + 1
    y_pred_1 = np.asarray(y_pred_0idx) + 1
    acc = accuracy_score(y_true_1, y_pred_1)
    adj_acc = adjacent_accuracy(y_true_1, y_pred_1)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_1, y_pred_1, labels=sorted(np.unique(y_true_1)), zero_division=0
    )
    print(f"\n--- {label} ---")
    print(f"Accuracy: {acc:.4f}   Adjacent-facies accuracy: {adj_acc:.4f}")
    report_df = pd.DataFrame({
        'facies': sorted(np.unique(y_true_1)),
        'precision': prec, 'recall': rec, 'f1': f1, 'support': support,
    })
    print(report_df.to_string(index=False))
    return {'accuracy': acc, 'adjacent_accuracy': adj_acc, 'report': report_df}


def make_xgb(random_state=RANDOM_STATE):
    """Same XGBoost configuration used throughout the settled pipeline."""
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        random_state=random_state, eval_metric='mlogloss'
    )


# %% CELL 4: Baseline (for reference -- re-run here so all comparisons
# are from the exact same session/environment)
logo = LeaveOneGroupOut()
oof_baseline_pred = np.zeros(len(y), dtype=int)

for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr = y.iloc[train_idx]
    sw = np.sqrt(compute_sample_weight('balanced', y_tr))
    clf = make_xgb()
    clf.fit(X_tr, y_tr, sample_weight=sw)
    oof_baseline_pred[test_idx] = clf.predict(X_te)

baseline_results = per_facies_report(y, oof_baseline_pred, label="BASELINE (calibrated single XGBoost)")


# %% CELL 5: RE-VALIDATION 1 -- HMM / Viterbi sequence smoothing (real XGBoost)
def estimate_transition_matrix(train_df, facies_col='Facies', well_col='Well Name',
                                 depth_col='Depth', n_classes=9, smoothing=1.0):
    counts = np.full((n_classes, n_classes), smoothing)
    initial_counts = np.full(n_classes, smoothing)
    for well, g in train_df.groupby(well_col):
        g = g.sort_values(depth_col)
        facies_seq = (g[facies_col].values - 1)
        initial_counts[facies_seq[0]] += 1
        for a, b in zip(facies_seq[:-1], facies_seq[1:]):
            counts[a, b] += 1
    transition_matrix = counts / counts.sum(axis=1, keepdims=True)
    initial_probs = initial_counts / initial_counts.sum()
    return transition_matrix, initial_probs


def viterbi_decode(emission_probs, transition_matrix, initial_probs, eps=1e-12):
    T, n_classes = emission_probs.shape
    log_trans = np.log(transition_matrix + eps)
    log_init = np.log(initial_probs + eps)
    log_emit = np.log(emission_probs + eps)
    log_delta = np.zeros((T, n_classes))
    backpointer = np.zeros((T, n_classes), dtype=int)
    log_delta[0] = log_init + log_emit[0]
    for t in range(1, T):
        for j in range(n_classes):
            scores = log_delta[t - 1] + log_trans[:, j]
            backpointer[t, j] = np.argmax(scores)
            log_delta[t, j] = np.max(scores) + log_emit[t, j]
    path = np.zeros(T, dtype=int)
    path[T - 1] = np.argmax(log_delta[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]
    return path


oof_hmm_pred = np.zeros(len(y), dtype=int)

for train_idx, test_idx in logo.split(X, y, groups=groups):
    train_df_fold = df.iloc[train_idx]
    test_df_fold = df.iloc[test_idx].sort_values('Depth')  # must sort by depth for Viterbi
    test_idx_sorted = test_df_fold.index

    X_tr = X.iloc[train_idx]
    y_tr = y.iloc[train_idx]
    X_te = X.loc[test_idx_sorted]

    sw = np.sqrt(compute_sample_weight('balanced', y_tr))
    clf = make_xgb()
    clf.fit(X_tr, y_tr, sample_weight=sw)
    emission_probs = clf.predict_proba(X_te)

    trans, init = estimate_transition_matrix(train_df_fold, n_classes=N_CLASSES)
    hmm_pred = viterbi_decode(emission_probs, trans, init)

    pos = df.index.get_indexer(test_idx_sorted)
    oof_hmm_pred[pos] = hmm_pred

hmm_results = per_facies_report(y, oof_hmm_pred, label="RE-VALIDATED: HMM/Viterbi smoothing (real XGBoost)")


# %% CELL 6: RE-VALIDATION 2 -- Cascade binary specialist (real XGBoost)
CONFUSED_PAIR = (7, 8)  # 0-indexed facies 8, 9
oof_cascade_pred = np.zeros(len(y), dtype=int)
n_triggered_total = 0

for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr = y.iloc[train_idx]

    sw = np.sqrt(compute_sample_weight('balanced', y_tr))
    base = make_xgb()
    base.fit(X_tr, y_tr, sample_weight=sw)
    base_probs = base.predict_proba(X_te)
    base_pred = np.argmax(base_probs, axis=1)

    triggered = np.isin(base_pred, CONFUSED_PAIR)
    n_triggered_total += triggered.sum()

    train_pair_mask = y_tr.isin(CONFUSED_PAIR)
    cascade_pred = base_pred.copy()

    if train_pair_mask.sum() > 10 and triggered.sum() > 0:
        specialist = SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE)
        specialist.fit(X_tr[train_pair_mask], y_tr[train_pair_mask])
        cascade_pred[triggered] = specialist.predict(X_te[triggered])

    oof_cascade_pred[test_idx] = cascade_pred

print(f"\nRows where base model's top-1 prediction was facies 8 or 9: {n_triggered_total}")
cascade_results = per_facies_report(y, oof_cascade_pred, label="RE-VALIDATED: Cascade specialist (real XGBoost)")


# %% CELL 7: RE-VALIDATION 3 -- Depth-window rolling features (real XGBoost)
df_sorted = df.sort_values(['Well Name', 'Depth']).reset_index(drop=True)
WINDOW = 5
LOG_COLS = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']

rolling_feats = []
for well, g in df_sorted.groupby('Well Name'):
    g = g.sort_values('Depth')
    roll_mean = g[LOG_COLS].rolling(window=WINDOW, center=True, min_periods=1).mean()
    roll_std = g[LOG_COLS].rolling(window=WINDOW, center=True, min_periods=1).std().fillna(0)
    roll_mean.columns = [f'{c}_rollmean' for c in LOG_COLS]
    roll_std.columns = [f'{c}_rollstd' for c in LOG_COLS]
    combined = pd.concat([roll_mean, roll_std], axis=1)
    combined.index = g.index
    rolling_feats.append(combined)

rolling_df = pd.concat(rolling_feats).sort_index()
df_sorted = pd.concat([df_sorted, rolling_df], axis=1)

ROLL_FEATURES = FEATURES + list(rolling_df.columns)
y_sorted = df_sorted[TARGET] - 1
groups_sorted = df_sorted[GROUP_COL]
X_rolled = df_sorted[ROLL_FEATURES]

oof_rolling_pred = np.zeros(len(y_sorted), dtype=int)

for train_idx, test_idx in logo.split(X_rolled, y_sorted, groups=groups_sorted):
    X_tr, X_te = X_rolled.iloc[train_idx], X_rolled.iloc[test_idx]
    y_tr = y_sorted.iloc[train_idx]
    sw = np.sqrt(compute_sample_weight('balanced', y_tr))
    clf = make_xgb()
    clf.fit(X_tr, y_tr, sample_weight=sw)
    oof_rolling_pred[test_idx] = clf.predict(X_te)

rolling_results = per_facies_report(y_sorted, oof_rolling_pred, label="RE-VALIDATED: Depth-window rolling features (real XGBoost)")


# %% CELL 8: FINAL SUMMARY -- all four mechanisms, real XGBoost, same session
print("\n" + "=" * 70)
print("FINAL SUMMARY: All mechanisms re-validated with real XGBoost")
print("=" * 70)

summary = pd.DataFrame({
    'mechanism': ['Baseline (calibrated single model)', 'HMM/Viterbi smoothing',
                  'Cascade specialist', 'Depth-window rolling features'],
    'accuracy': [baseline_results['accuracy'], hmm_results['accuracy'],
                 cascade_results['accuracy'], rolling_results['accuracy']],
    'adjacent_accuracy': [baseline_results['adjacent_accuracy'], hmm_results['adjacent_accuracy'],
                          cascade_results['adjacent_accuracy'], rolling_results['adjacent_accuracy']],
})
print(summary.to_string(index=False))

print("\nPer-facies recall comparison:")
recall_comparison = pd.DataFrame({
    'facies': baseline_results['report']['facies'],
    'baseline': baseline_results['report']['recall'],
    'hmm': hmm_results['report']['recall'],
    'cascade': cascade_results['report']['recall'],
    'rolling': rolling_results['report']['recall'],
})
print(recall_comparison.to_string(index=False))

print("\nCopy the two tables above directly into the manuscript's Section 2.4")
print("and Section 4, replacing the 'preliminary' framing with confirmed,")
print("production-validated results.")
