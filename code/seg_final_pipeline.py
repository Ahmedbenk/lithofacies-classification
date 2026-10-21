"""
SEG/Hall 2016 Hugoton-Panoma facies dataset -- FINAL pipeline.

Settled methodology, after testing four hybrid mechanisms (NN stacking,
HMM/Viterbi sequence smoothing, cascade binary specialist, depth-window
rolling features) -- three did not beat a well-calibrated single XGBoost model; HMM/Viterbi
smoothing did overall but lowered recall on the weakest facies (see
revalidation/). The final approach is: single XGBoost, dampened (sqrt) class weighting,
diagnosed and reported per-class rather than forcing a hybrid fix where
it was not found to help the weak classes.

Paste each # %% CELL block into its own Colab cell, in order.
"""

# %% CELL 1: Mount Drive and load data
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

TRAIN_PATH = '/content/drive/MyDrive/facies_data.csv'
BLIND_LOGS_PATH = '/content/drive/MyDrive/validation_data_nofacies.csv'
BLIND_LABELS_PATH = '/content/drive/MyDrive/blind_stuart_crawford_core_facies.csv'

df = pd.read_csv(TRAIN_PATH)
df.columns = [c.strip() for c in df.columns]
print(f"Rows: {len(df)}, Wells: {df['Well Name'].nunique()}")


# %% CELL 2: Imports
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
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


# %% CELL 3: Adjacent-facies map, diagnosis categories, and evaluation helpers
ADJACENT_FACIES = {
    1: [2], 2: [1, 3], 3: [2], 4: [5], 5: [4, 6],
    6: [5, 7], 7: [6, 8], 8: [6, 7, 9], 9: [7, 8],
}

# Diagnosis category per facies, from the diagnostic work already done.
# This is reported alongside the metrics, not used to alter predictions --
# it documents WHY each class performs the way it does.
DIAGNOSIS = {
    1: "Weak baseline signal; adjacency to facies 2 suspected but not confirmed",
    2: "Adequate baseline performance",
    3: "Adequate baseline performance",
    4: "Adequate baseline performance",
    5: "Resolved-to-limit: dampened weighting + depth-window features tested, no improvement found",
    6: "Adequate baseline performance",
    7: "Adequate baseline performance, benefits from dampened weighting",
    8: "Resolved-to-limit: adjacency to facies 9; cascade specialist tested, no improvement found; HMM smoothing improved overall accuracy but lowered weak-facies recall",
    9: "Adjacency-limited: dampened weighting partially recovers recall vs. naive full-balanced weighting",
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
    report_df['diagnosis'] = report_df['facies'].map(DIAGNOSIS)
    print(report_df.to_string(index=False))
    return {'accuracy': acc, 'adjacent_accuracy': adj_acc, 'report': report_df}


# %% CELL 4: LOWO evaluation of the final single-model approach
logo = LeaveOneGroupOut()
oof_pred = np.zeros(len(y), dtype=int)

for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
    held_out_well = groups.iloc[test_idx].unique()[0]
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr = y.iloc[train_idx]

    # Dampened (sqrt) class weighting -- the one intervention that
    # consistently helped across both datasets without destabilizing
    # other classes.
    sw = np.sqrt(compute_sample_weight('balanced', y_tr))

    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric='mlogloss'
    )
    clf.fit(X_tr, y_tr, sample_weight=sw)
    oof_pred[test_idx] = clf.predict(X_te)

    print(f"Fold {fold_i + 1}/{groups.nunique()}: held out = {held_out_well}")

lowo_results = per_facies_report(y, oof_pred, label="Final model -- LOWO cross-validation")


# %% CELL 5: Final retrain on all 8 wells
final_clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    random_state=RANDOM_STATE, eval_metric='mlogloss'
)
sw_full = np.sqrt(compute_sample_weight('balanced', y))
final_clf.fit(X, y, sample_weight=sw_full)
print("Final model retrained on all 8 wells.")


# %% CELL 6: Predict blind wells and align to core-derived ground truth
blind_logs = pd.read_csv(BLIND_LOGS_PATH)
blind_logs.columns = [c.strip() for c in blind_logs.columns]
X_blind = blind_logs[FEATURES]
blind_pred = final_clf.predict(X_blind)
blind_logs = blind_logs.reset_index(drop=True)
blind_logs['pred_facies'] = blind_pred + 1  # back to 1-indexed

core = pd.read_csv(BLIND_LABELS_PATH)
core.columns = [c.strip() for c in core.columns]

aligned_frames = []
for well in blind_logs['Well Name'].unique():
    logs_w = blind_logs[blind_logs['Well Name'] == well].sort_values('Depth')
    core_w = core[core['WellName'] == well].sort_values('Depth.ft')
    merged = pd.merge_asof(
        logs_w, core_w, left_on='Depth', right_on='Depth.ft',
        direction='nearest', tolerance=1.0
    )
    aligned_frames.append(merged)

blind_aligned = pd.concat(aligned_frames, ignore_index=True)

n_unmatched = blind_aligned['LithCode'].isnull().sum()
if n_unmatched:
    print(f"WARNING: {n_unmatched} rows had no core sample within tolerance -- dropped.")
blind_aligned = blind_aligned.dropna(subset=['LithCode'])

n_out_of_scheme = (blind_aligned['LithCode'] == 11).sum()
if n_out_of_scheme:
    print(f"NOTE: {n_out_of_scheme} rows have LithCode 11 (out-of-scheme, 'Marine sandstone') "
          f"-- excluded from scoring, not part of the trained 9-facies scheme.")
blind_aligned = blind_aligned[blind_aligned['LithCode'] != 11]

y_blind_true = blind_aligned['LithCode'].astype(int) - 1
y_blind_pred = blind_aligned['pred_facies'].astype(int) - 1


# %% CELL 7: FINAL blind-well report (this is the headline, single-look result)
print("\n" + "=" * 60)
print("FINAL BLIND-WELL RESULTS (STUART + CRAWFORD)")
print("=" * 60)
blind_results = per_facies_report(y_blind_true, y_blind_pred, label="Final model -- blind wells")

print("\nConfusion matrix (rows=true, cols=predicted, 1-indexed facies):")
cm = confusion_matrix(y_blind_true + 1, y_blind_pred + 1, labels=sorted(y.unique() + 1))
print(pd.DataFrame(cm, index=sorted(y.unique() + 1), columns=sorted(y.unique() + 1)))

print("\n=== SUMMARY FOR PAPER ===")
print(f"LOWO accuracy: {lowo_results['accuracy']:.4f} | adjacent-facies: {lowo_results['adjacent_accuracy']:.4f}")
print(f"Blind-well accuracy: {blind_results['accuracy']:.4f} | adjacent-facies: {blind_results['adjacent_accuracy']:.4f}")
