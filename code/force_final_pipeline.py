"""
FORCE 2020 lithology dataset -- FINAL pipeline.

Settled methodology: single XGBoost, dampened (sqrt) class weighting,
feature set includes PEF (confirmed to help Chalk/Anhydrite/Limestone)
but excludes NPHI (tested, made things worse) and the >50%-missing logs.
Basement is excluded from evaluation (confined to 1 of 98 wells --
structurally impossible to cross-validate fairly). Per-class results are
reported with an explicit diagnosis category, same as the SEG pipeline.

Paste each # %% CELL block into its own Colab cell, in order.
"""

# %% CELL 1: Mount Drive and load training data
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

TRAIN_PATH = '/content/drive/MyDrive/train.csv'
df = pd.read_csv(TRAIN_PATH, sep=';')
print(f"Shape: {df.shape}, Wells: {df['WELL'].nunique()}")


# %% CELL 2: Imports
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import xgboost as xgb

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# %% CELL 3: Settled feature set (PEF included, NPHI excluded per testing)
FEATURES = ['GR', 'RDEP', 'RMED', 'RHOB', 'DRHO', 'DTC', 'CALI', 'PEF']
print("Missingness of final feature set:")
print(df[FEATURES].isnull().mean().round(3))


# %% CELL 4: Label encoding, diagnosis categories, and evaluation helper
LITHOLOGY_NUMBERS = {
    30000: 'Sandstone', 65030: 'Sandstone/Shale', 65000: 'Shale',
    80000: 'Marl', 74000: 'Dolomite', 70000: 'Limestone',
    70032: 'Chalk', 88000: 'Halite', 86000: 'Anhydrite',
    99000: 'Tuff', 90000: 'Coal', 93000: 'Basement'
}
CLASS_NAMES = list(LITHOLOGY_NUMBERS.values())
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
N_CLASSES = len(CLASS_NAMES)

# Diagnosis category per class, from the diagnostic work already done on
# this dataset (dampened-weighting test, PEF-inclusion test, NPHI test).
DIAGNOSIS = {
    'Sandstone': 'Adequate baseline performance',
    'Sandstone/Shale': 'Persistently weak precision/recall; likely a genuinely ambiguous mixed-lithology class rather than a modeling failure',
    'Shale': 'Adequate baseline performance (majority class)',
    'Marl': 'Adequate baseline performance, benefits modestly from PEF',
    'Dolomite': 'Resolved-to-limit: dampened weighting, PEF, and NPHI all tested, no meaningful improvement found',
    'Limestone': 'Feature-fixed: PEF inclusion improved F1',
    'Chalk': 'Feature-fixed: PEF inclusion improved F1 substantially',
    'Halite': 'Low fold-coverage (present in only 3 wells) -- high precision but limited reliability',
    'Anhydrite': 'Feature-fixed: PEF inclusion improved F1; low fold-coverage (6 wells) caveat applies',
    'Tuff': 'Adequate baseline performance, benefits modestly from PEF',
    'Coal': 'Adequate baseline performance',
    'Basement': 'EXCLUDED from evaluation: confined to 1 of 98 wells, cannot be fairly cross-validated under any well-based split',
}

df['LITH_NAME'] = df['FORCE_2020_LITHOFACIES_LITHOLOGY'].map(LITHOLOGY_NUMBERS)
df['LITH_IDX'] = df['LITH_NAME'].map(CLASS_TO_IDX)

X = df[FEATURES]
y = df['LITH_IDX']
groups = df['WELL']

print(f"Rows: {len(df)}, Classes: {N_CLASSES}, Wells: {groups.nunique()}")


def per_class_report(y_true_idx, y_pred_idx, fold_presence=None, label=""):
    """Pooled per-class precision/recall/F1, annotated with diagnosis
    category and (if provided) fold-presence/well-count reliability info."""
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=range(N_CLASSES), zero_division=0
    )
    report_df = pd.DataFrame({
        'class': CLASS_NAMES, 'precision': prec, 'recall': rec,
        'f1': f1, 'support': support,
    })
    report_df['diagnosis'] = report_df['class'].map(DIAGNOSIS)

    if fold_presence is not None:
        report_df['n_folds_present'] = [(fold_presence[:, c] > 0).sum() for c in range(N_CLASSES)]

    acc = accuracy_score(y_true_idx, y_pred_idx)
    print(f"\n--- {label} ---")
    print(f"Overall accuracy: {acc:.4f}")
    print(report_df.to_string(index=False))
    return {'accuracy': acc, 'report': report_df}


# %% CELL 5: CV evaluation of the final single-model approach
N_SPLITS = 10
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_presence = np.zeros((N_SPLITS, N_CLASSES), dtype=int)
oof_pred = np.full(len(y), -1, dtype=int)

for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    y_test = y.iloc[test_idx]
    for c in range(N_CLASSES):
        fold_presence[fold_i, c] = (y_test == c).sum()

    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr = y.iloc[train_idx]

    imputer = SimpleImputer(strategy='median')
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    sw = np.sqrt(compute_sample_weight('balanced', y_tr))
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric='mlogloss', tree_method='hist'
    )
    clf.fit(X_tr_imp, y_tr, sample_weight=sw)
    oof_pred[test_idx] = clf.predict(X_te_imp)

    print(f"Fold {fold_i}: train wells={groups.iloc[train_idx].nunique()}, "
          f"test wells={groups.iloc[test_idx].nunique()}, test rows={len(test_idx)}")

assert (oof_pred >= 0).all(), "Some rows never got predicted -- check fold coverage."

cv_results = per_class_report(y, oof_pred, fold_presence=fold_presence,
                                label="Final model -- StratifiedGroupKFold cross-validation")


# %% CELL 6: Final retrain on all 98 wells
imputer_final = SimpleImputer(strategy='median')
X_imp_final = imputer_final.fit_transform(X)

sw_full = np.sqrt(compute_sample_weight('balanced', y))
final_clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    random_state=RANDOM_STATE, eval_metric='mlogloss', tree_method='hist'
)
final_clf.fit(X_imp_final, y, sample_weight=sw_full)
print("Final model retrained on all 98 wells.")


# %% CELL 7a: INSPECT the leaderboard blind-test files before assuming structure
# We haven't seen these files' actual column names yet -- check before
# writing merge/alignment logic, since the SEG blind files had real
# naming mismatches (WellName vs Well Name, Depth.ft vs Depth) that
# would have caused silent bugs if assumed rather than verified.
LEADERBOARD_FEATURES_PATH = '/content/drive/MyDrive/leaderboard_test_features.csv'
LEADERBOARD_TARGET_PATH = '/content/drive/MyDrive/leaderboard_test_target.csv'

lb_features = pd.read_csv(LEADERBOARD_FEATURES_PATH, sep=';')
lb_target = pd.read_csv(LEADERBOARD_TARGET_PATH, sep=';')

print("=== leaderboard_test_features.csv ===")
print(f"Shape: {lb_features.shape}")
print(f"Columns: {lb_features.columns.tolist()}")
print(lb_features.head(3))

print("\n=== leaderboard_test_target.csv ===")
print(f"Shape: {lb_target.shape}")
print(f"Columns: {lb_target.columns.tolist()}")
print(lb_target.head(3))

# STOP HERE and check the printed columns before running Cell 7b --
# the alignment/merge code in 7b will need to match whatever these
# actually are.
