"""
Hybrid XGBoost + Neural Network pipeline for lithofacies classification
from well-log data (Hall 2016 SEG facies benchmark).

Designed for Google Colab. Paste each "# %% CELL" block into its own
Colab cell, in order, or run this whole file as a script.

Pipeline:
  1. Leave-one-well-out (LOWO) CV on facies_data.csv to evaluate the
     hybrid vs. XGBoost-alone, with per-facies metrics (minority-class
     recall is the headline comparison).
  2. Final hybrid retrained on all 8 training wells.
  3. Blind-well evaluation on the official SEG contest test wells
     (STUART, CRAWFORD), scored against core-derived ground truth,
     including the adjacent-facies lenient accuracy used in the
     original SEG contest.

Requirements (pre-installed in Colab): pandas, numpy, scikit-learn,
xgboost, tensorflow.
"""

# %% CELL 1: Mount Drive and set paths
import os

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass  # not running in Colab

# EDIT THESE THREE PATHS to match your Drive locations
TRAIN_PATH = '/content/drive/MyDrive/facies_data.csv'
BLIND_LOGS_PATH = '/content/drive/MyDrive/validation_data_nofacies.csv'
BLIND_LABELS_PATH = '/content/drive/MyDrive/blind_stuart_crawford_core_facies.csv'

for p in [TRAIN_PATH, BLIND_LOGS_PATH, BLIND_LABELS_PATH]:
    assert os.path.exists(p), f"File not found: {p} -- check the path."


# %% CELL 2: Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# %% CELL 3: Load and prepare training data
df = pd.read_csv(TRAIN_PATH)
df.columns = [c.strip() for c in df.columns]

FEATURES = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
TARGET = 'Facies'
GROUP_COL = 'Well Name'
N_CLASSES = df[TARGET].nunique()

X = df[FEATURES].copy()
y = (df[TARGET] - 1).copy()  # zero-index for xgboost / keras
groups = df[GROUP_COL].copy()

print(f"Rows: {len(df)}, Wells: {groups.nunique()}, Classes: {N_CLASSES}")
print(df[TARGET].value_counts().sort_index())


# %% CELL 4: Adjacent-facies map and evaluation helpers
# From the original Hall (2016) SEG facies tutorial: some facies are
# geologically gradational/adjacent, so a lenient "adjacent accuracy"
# metric is standard in this literature alongside strict accuracy.
# Keys/values are 1-indexed facies numbers as used in the raw data.
ADJACENT_FACIES = {
    1: [2],
    2: [1, 3],
    3: [2],
    4: [5],
    5: [4, 6],
    6: [5, 7],
    7: [6, 8],
    8: [6, 7, 9],
    9: [7, 8],
}


def adjacent_accuracy(y_true_1idx, y_pred_1idx):
    """Fraction of predictions that are either exactly correct or in the
    true facies' adjacent set. y_true/y_pred are 1-indexed facies (1-9)."""
    correct = 0
    for t, p in zip(y_true_1idx, y_pred_1idx):
        if p == t or p in ADJACENT_FACIES.get(t, []):
            correct += 1
    return correct / len(y_true_1idx)


def per_facies_report(y_true_0idx, y_pred_0idx, label=""):
    """Per-facies precision/recall/F1, plus overall + adjacent accuracy.
    Inputs are 0-indexed (as used internally); converted to 1-indexed
    for the adjacent-facies lookup and for readable output."""
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
        'precision': prec, 'recall': rec, 'f1': f1, 'support': support
    })
    print(report_df.to_string(index=False))
    return {'accuracy': acc, 'adjacent_accuracy': adj_acc, 'report': report_df}


# %% CELL 5: Model builders
def make_nn(input_dim, n_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def get_xgb_oof_probs(X_tr, y_tr, groups_tr, n_classes, n_inner_splits=5):
    """Generate out-of-fold class probabilities for the training set using
    an inner GroupKFold (grouped by well, so correlated same-well rows
    never leak across inner train/val splits). This is what the NN
    meta-learner sees at training time -- never in-sample predictions."""
    n_splits = min(n_inner_splits, groups_tr.nunique())
    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros((len(X_tr), n_classes))

    for inner_tr_idx, inner_val_idx in gkf.split(X_tr, y_tr, groups=groups_tr):
        sw = compute_sample_weight('balanced', y_tr.iloc[inner_tr_idx])
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric='mlogloss'
        )
        model.fit(X_tr.iloc[inner_tr_idx], y_tr.iloc[inner_tr_idx], sample_weight=sw)
        oof_probs[inner_val_idx] = model.predict_proba(X_tr.iloc[inner_val_idx])

    return oof_probs


def train_hybrid(X_tr, y_tr, groups_tr, X_test, n_classes, epochs=60, verbose=0):
    """Train the full two-stage hybrid on (X_tr, y_tr) and return
    predictions for X_test, plus the standalone XGBoost predictions
    for comparison. Also returns the fitted stage-1/stage-2 objects
    and scaler so the final model can be reused for blind-well scoring."""

    # --- Stage 1: XGBoost, class-weighted, out-of-fold probs for training ---
    sw_full = compute_sample_weight('balanced', y_tr)
    xgb_final = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric='mlogloss'
    )
    xgb_final.fit(X_tr, y_tr, sample_weight=sw_full)

    oof_probs = get_xgb_oof_probs(X_tr, y_tr, groups_tr, n_classes)
    test_probs_from_final_xgb = xgb_final.predict_proba(X_test)
    xgb_only_pred = np.argmax(test_probs_from_final_xgb, axis=1)

    # --- Stage 2: NN meta-learner on [original features + OOF XGB probs] ---
    prob_cols = [f'xgb_p{i}' for i in range(n_classes)]
    X_tr_meta = pd.concat(
        [X_tr.reset_index(drop=True), pd.DataFrame(oof_probs, columns=prob_cols)], axis=1
    )
    X_test_meta = pd.concat(
        [X_test.reset_index(drop=True), pd.DataFrame(test_probs_from_final_xgb, columns=prob_cols)],
        axis=1
    )

    scaler = StandardScaler()
    X_tr_meta_scaled = scaler.fit_transform(X_tr_meta)
    X_test_meta_scaled = scaler.transform(X_test_meta)

    nn = make_nn(X_tr_meta_scaled.shape[1], n_classes)

    # keras wants a {class_index: weight} dict (not per-sample weights).
    # Using sqrt of the fully-balanced weight ("dampened" weighting) --
    # full inverse-frequency weighting (up to ~5x for the rarest class)
    # was found to destabilize training and hurt several mid-sized
    # classes (see LOWO diagnostic run), so this is a deliberately
    # gentler nudge rather than full rebalancing.
    unique_classes, counts = np.unique(y_tr, return_counts=True)
    total = len(y_tr)
    cw_dict = {
        c: np.sqrt(total / (len(unique_classes) * cnt))
        for c, cnt in zip(unique_classes, counts)
    }

    early_stop = EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True
    )

    nn.fit(
        X_tr_meta_scaled, y_tr, epochs=epochs, batch_size=32,
        validation_split=0.15, class_weight=cw_dict,
        callbacks=[early_stop], verbose=verbose
    )

    hybrid_pred = np.argmax(nn.predict(X_test_meta_scaled, verbose=0), axis=1)

    fitted = {
        'xgb_final': xgb_final, 'nn': nn, 'scaler': scaler, 'prob_cols': prob_cols
    }
    return xgb_only_pred, hybrid_pred, fitted


# %% CELL 6: Leave-one-well-out evaluation (XGBoost-only vs. Hybrid)
logo = LeaveOneGroupOut()
oof_xgb_pred = np.zeros(len(y), dtype=int)
oof_hybrid_pred = np.zeros(len(y), dtype=int)

for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
    held_out_well = groups.iloc[test_idx].unique()[0]
    print(f"\n=== LOWO fold {fold_i + 1}/{groups.nunique()}: held out = {held_out_well} ===")

    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    groups_tr = groups.iloc[train_idx]

    xgb_pred, hybrid_pred, _ = train_hybrid(X_tr, y_tr, groups_tr, X_te, N_CLASSES)

    oof_xgb_pred[test_idx] = xgb_pred
    oof_hybrid_pred[test_idx] = hybrid_pred

# Aggregate metrics across all 8 held-out wells combined
xgb_results = per_facies_report(y, oof_xgb_pred, label="XGBoost only (LOWO, all wells combined)")
hybrid_results = per_facies_report(y, oof_hybrid_pred, label="Hybrid XGBoost+NN (LOWO, all wells combined)")

print("\n=== Per-facies recall: XGBoost vs Hybrid ===")
compare_df = pd.DataFrame({
    'facies': xgb_results['report']['facies'],
    'support': xgb_results['report']['support'],
    'xgb_recall': xgb_results['report']['recall'],
    'hybrid_recall': hybrid_results['report']['recall'],
})
compare_df['recall_gain'] = compare_df['hybrid_recall'] - compare_df['xgb_recall']
print(compare_df.to_string(index=False))

# STOP AND REVIEW HERE before proceeding. If the hybrid isn't beating
# XGBoost-alone on minority-facies recall, that's the honest result --
# don't skip to the blind-well section hoping it looks better there.
# This LOWO comparison is what should drive any hyperparameter changes,
# not the blind-well score (which you only look at once, at the end).


# %% CELL 7: Final retrain on all 8 training wells
print("\n=== Retraining final hybrid on all training wells ===")
_, _, final_fitted = train_hybrid(X, y, groups, X, N_CLASSES, epochs=60, verbose=1)
# (X passed as both train and "test" here just to reuse the function --
#  we only keep final_fitted; the returned predictions on X are unused.)

xgb_final = final_fitted['xgb_final']
nn_final = final_fitted['nn']
scaler_final = final_fitted['scaler']
prob_cols = final_fitted['prob_cols']


# %% CELL 8: Load blind-well logs and predict
blind_logs = pd.read_csv(BLIND_LOGS_PATH)
blind_logs.columns = [c.strip() for c in blind_logs.columns]

X_blind = blind_logs[FEATURES]
xgb_blind_probs = xgb_final.predict_proba(X_blind)

X_blind_meta = pd.concat(
    [X_blind.reset_index(drop=True), pd.DataFrame(xgb_blind_probs, columns=prob_cols)],
    axis=1
)
X_blind_meta_scaled = scaler_final.transform(X_blind_meta)
blind_hybrid_pred = np.argmax(nn_final.predict(X_blind_meta_scaled, verbose=0), axis=1)
blind_xgb_pred = np.argmax(xgb_blind_probs, axis=1)

blind_logs = blind_logs.reset_index(drop=True)
blind_logs['pred_facies_xgb'] = blind_xgb_pred + 1       # back to 1-indexed
blind_logs['pred_facies_hybrid'] = blind_hybrid_pred + 1


# %% CELL 9: Align blind predictions to core-derived ground truth by depth
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

# LithCode 11 ("Marine sandstone") is not part of the 1-9 facies scheme
# the model was trained on -- exclude these from scoring and report the count.
n_out_of_scheme = (blind_aligned['LithCode'] == 11).sum()
if n_out_of_scheme:
    print(f"NOTE: {n_out_of_scheme} rows have LithCode 11 (out-of-scheme) -- excluded from scoring.")
blind_aligned = blind_aligned[blind_aligned['LithCode'] != 11]

y_blind_true = blind_aligned['LithCode'].astype(int) - 1  # zero-index to match model output
y_blind_xgb = blind_aligned['pred_facies_xgb'].astype(int) - 1
y_blind_hybrid = blind_aligned['pred_facies_hybrid'].astype(int) - 1


# %% CELL 10: Final blind-well report (the headline result)
print("\n" + "=" * 60)
print("FINAL BLIND-WELL RESULTS (STUART + CRAWFORD)")
print("=" * 60)
blind_xgb_results = per_facies_report(y_blind_true, y_blind_xgb, label="XGBoost only -- blind wells")
blind_hybrid_results = per_facies_report(y_blind_true, y_blind_hybrid, label="Hybrid XGBoost+NN -- blind wells")

print("\nConfusion matrix (Hybrid, rows=true, cols=predicted, 1-indexed facies):")
cm = confusion_matrix(y_blind_true + 1, y_blind_hybrid + 1, labels=sorted(y.unique() + 1))
print(pd.DataFrame(cm, index=sorted(y.unique() + 1), columns=sorted(y.unique() + 1)))


# %% NOTES (read before running / before writing up results)
#
# 1. The blind-well score in CELL 10 should be looked at ONCE, after
#    architecture/hyperparameters are already locked in from the LOWO
#    results in CELL 6. Don't tune anything based on the blind-well
#    number itself -- that defeats its purpose as an honest external
#    check and is exactly the kind of thing a reviewer will ask about.
#
# 2. If the hybrid underperforms XGBoost-alone on the LOWO comparison
#    (CELL 6), that is a valid and reportable result -- it means the
#    two-stage design doesn't help on this data, and the honest move is
#    to report that finding, not to keep tweaking until it looks better.
#
# 3. The 9 "LithCode 11" (Marine sandstone) rows in the blind set are
#    excluded from scoring since they fall outside the 9-facies scheme
#    the model was trained on. Report this exclusion explicitly in the
#    paper's methodology, don't just drop it silently.
#
# 4. Runtime: with 3232 rows, 8 outer folds x 5 inner folds x (XGBoost +
#    NN), this will take a few minutes on Colab's CPU runtime. If it's
#    too slow, reduce nn epochs (CELL 6 default is baked into
#    train_hybrid's epochs=60 default) or n_estimators in CELL 5/6.
#
# 5. The adjacent-facies matrix in CELL 4 comes from the original Hall
#    (2016) SEG tutorial convention -- cite that source if you report
#    the adjacent-accuracy metric in the paper.
