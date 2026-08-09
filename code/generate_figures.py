"""
Publication figure generation for the lithofacies paper.

All data below is transcribed directly from actual Colab pipeline outputs
produced earlier in this project (SEG and FORCE final pipeline runs).
Only figures backed by production-model (real XGBoost/Keras) results are
included here -- the HMM, cascade-specialist, and depth-window-feature
experiments were only validated with a RandomForest stand-in in a
sandbox, NOT with the real model, and are deliberately excluded from
publication figures until re-run in Colab with the actual pipeline.

Run this in Colab or any environment with matplotlib installed. Figures
are saved as 300 DPI PNG (raster, fine for review) -- for camera-ready
submission, re-save as vector PDF/EPS if the journal requires it (change
FIG_FORMAT below).

Paste each # %% CELL block into its own Colab cell, or run as one script.
"""

# %% CELL 1: Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FIG_FORMAT = 'png'   # change to 'pdf' or 'eps' for camera-ready vector output
DPI = 300
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# %% CELL 2: Figure 1 -- SEG blind-well confusion matrix
seg_facies = list(range(1, 10))
seg_cm = np.array([
    [3, 9, 2, 0, 0, 0, 0, 0, 0],
    [13, 88, 11, 0, 1, 1, 0, 3, 0],
    [3, 60, 60, 0, 2, 0, 0, 11, 0],
    [0, 0, 0, 45, 4, 22, 12, 4, 0],
    [0, 0, 1, 3, 15, 19, 0, 17, 0],
    [0, 3, 3, 6, 19, 71, 4, 53, 9],
    [0, 1, 1, 3, 4, 5, 53, 26, 0],
    [0, 4, 4, 0, 6, 19, 7, 103, 2],
    [0, 0, 0, 0, 0, 0, 0, 2, 4],
])

# Row-normalize (recall-style) for interpretability -- raw counts also
# annotated on each cell
seg_cm_norm = seg_cm / seg_cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(seg_cm_norm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(len(seg_facies)))
ax.set_yticks(range(len(seg_facies)))
ax.set_xticklabels(seg_facies)
ax.set_yticklabels(seg_facies)
ax.set_xlabel('Predicted facies')
ax.set_ylabel('True facies')
ax.set_title('(a) SEG blind-well confusion matrix\n(STUART + CRAWFORD, row-normalized)')

for i in range(len(seg_facies)):
    for j in range(len(seg_facies)):
        count = seg_cm[i, j]
        if count > 0:
            color = 'white' if seg_cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, str(count), ha='center', va='center', color=color, fontsize=8)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Row-normalized fraction (recall)')
plt.tight_layout()
plt.savefig(f'fig1_seg_confusion_matrix.{FIG_FORMAT}', dpi=DPI, bbox_inches='tight')
plt.show()
print("Saved fig1_seg_confusion_matrix")


# %% CELL 3: Figure 2 -- FORCE blind-well confusion matrix
force_classes = ['Sandstone', 'Ss/Sh', 'Shale', 'Marl', 'Dolomite',
                  'Limestone', 'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement']
force_cm = np.array([
    [18474, 1997, 1391, 249, 11, 676, 7, 0, 0, 996, 247, 0],
    [2737, 6665, 7323, 204, 5, 232, 3, 0, 0, 260, 129, 0],
    [2088, 9778, 65820, 3387, 6, 1463, 1, 0, 0, 1047, 385, 0],
    [391, 649, 389, 830, 0, 1003, 21, 0, 0, 5, 18, 0],
    [15, 23, 287, 17, 0, 65, 0, 0, 0, 3, 6, 0],
    [648, 544, 784, 319, 3, 2238, 106, 62, 0, 91, 2, 1],
    [0, 0, 0, 2, 0, 606, 17, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 4, 34, 0, 0, 86, 0, 0, 0, 0, 0, 0],
    [63, 75, 503, 26, 0, 10, 0, 0, 0, 567, 1, 0],
    [65, 76, 79, 2, 0, 5, 0, 0, 0, 0, 463, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

row_sums = force_cm.sum(axis=1, keepdims=True)
# Halite and Basement have zero support in the blind set -- avoid div-by-zero
row_sums_safe = np.where(row_sums == 0, 1, row_sums)
force_cm_norm = force_cm / row_sums_safe

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(force_cm_norm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(len(force_classes)))
ax.set_yticks(range(len(force_classes)))
ax.set_xticklabels(force_classes, rotation=45, ha='right')
ax.set_yticklabels(force_classes)
ax.set_xlabel('Predicted lithology')
ax.set_ylabel('True lithology')
ax.set_title('(b) FORCE blind-well confusion matrix\n(official leaderboard set, row-normalized)')

for i in range(len(force_classes)):
    for j in range(len(force_classes)):
        count = force_cm[i, j]
        if count > 0:
            color = 'white' if force_cm_norm[i, j] > 0.5 else 'black'
            fontsize = 6 if count < 1000 else 7
            ax.text(j, i, f'{count}', ha='center', va='center', color=color, fontsize=fontsize)
        if row_sums[i, 0] == 0 and i == j:
            ax.text(j, i, 'n/a', ha='center', va='center', color='gray', fontsize=7, style='italic')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Row-normalized fraction (recall)')
plt.tight_layout()
plt.savefig(f'fig2_force_confusion_matrix.{FIG_FORMAT}', dpi=DPI, bbox_inches='tight')
plt.show()
print("Saved fig2_force_confusion_matrix")


# %% CELL 4: Figure 3 -- CV vs. blind-well recall for well-concentrated FORCE classes
# Includes contrast classes (Coal, Sandstone) that are well-distributed
# across many wells, to visually demonstrate the well-concentration effect
# rather than implying ALL classes generalize poorly.
comparison_data = pd.DataFrame({
    'class': ['Chalk', 'Anhydrite', 'Dolomite', 'Coal', 'Sandstone', 'Marl', 'Tuff'],
    'n_wells': [11, 6, 37, 51, 98, 73, 55],
    'cv_recall': [0.307334, 0.376959, 0.038507, 0.551309, 0.644329, 0.352516, 0.373500],
    'blind_recall': [0.027200, 0.000000, 0.000000, 0.671014, 0.768214, 0.251059, 0.455422],
})
comparison_data = comparison_data.sort_values('n_wells')

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(comparison_data))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_data['cv_recall'], width, label='Cross-validated recall', color='#4C72B0')
bars2 = ax.bar(x + width/2, comparison_data['blind_recall'], width, label='Blind-well recall', color='#C44E52')

ax.set_xticks(x)
labels = [f"{c}\n(n={w} wells)" for c, w in zip(comparison_data['class'], comparison_data['n_wells'])]
ax.set_xticklabels(labels)
ax.set_ylabel('Recall')
ax.set_title('(c) Cross-validated vs. blind-well recall by well-count\n(classes ordered by training-well count, low to high)')
ax.legend()
ax.set_ylim(0, 1.0)
ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(1.0, 0.92, 'Low well-count\n(concentrated)', ha='center', fontsize=8, style='italic', color='gray')
ax.text(5.0, 0.92, 'High well-count\n(well-distributed)', ha='center', fontsize=8, style='italic', color='gray')

plt.tight_layout()
plt.savefig(f'fig3_cv_vs_blind_recall.{FIG_FORMAT}', dpi=DPI, bbox_inches='tight')
plt.show()
print("Saved fig3_cv_vs_blind_recall")


# %% CELL 5: Figure 4 -- PEF/NPHI feature-inclusion ablation (F1 by class)
ablation_data = pd.DataFrame({
    'class': ['Chalk', 'Anhydrite', 'Limestone', 'Marl', 'Tuff', 'Dolomite', 'Coal'],
    'no_PEF': [0.307555, 0.450131, 0.457511, 0.283657, 0.246603, 0.024974, 0.291231],
    'plus_PEF': [0.383206, 0.509340, 0.474916, 0.293033, 0.263379, 0.030445, 0.265189],
    'plus_PEF_NPHI': [0.353749, 0.469932, 0.476194, 0.285158, 0.229974, 0.021675, 0.344946],
})

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(ablation_data))
width = 0.27

ax.bar(x - width, ablation_data['no_PEF'], width, label='Baseline (no PEF)', color='#8C8C8C')
ax.bar(x, ablation_data['plus_PEF'], width, label='+ PEF', color='#55A868')
ax.bar(x + width, ablation_data['plus_PEF_NPHI'], width, label='+ PEF + NPHI', color='#CCB974')

ax.set_xticks(x)
ax.set_xticklabels(ablation_data['class'], rotation=20, ha='right')
ax.set_ylabel('F1 score')
ax.set_title('(d) Feature-inclusion ablation: PEF and NPHI (FORCE dataset, cross-validated)')
ax.legend()
plt.tight_layout()
plt.savefig(f'fig4_pef_nphi_ablation.{FIG_FORMAT}', dpi=DPI, bbox_inches='tight')
plt.show()
print("Saved fig4_pef_nphi_ablation")


# %% CELL 6: Figure 5 -- SEG: calibrated baseline vs. NN-stacking hybrid (validated only)
# NOTE: only NN stacking is included here since it is the only hybrid
# mechanism actually validated with the real production model (XGBoost +
# Keras) in Colab. HMM, cascade, and rolling-feature results are
# excluded pending re-validation with the real model.
seg_hybrid_data = pd.DataFrame({
    'facies': list(range(1, 10)),
    'baseline_recall': [0.169884, 0.601626, 0.588618, 0.543478, 0.152074,
                         0.495671, 0.316327, 0.536145, 0.788820],
    'nn_stack_recall': [0.166023, 0.601626, 0.582114, 0.586957, 0.046083,
                         0.554113, 0.428571, 0.506024, 0.366460],
})

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(seg_hybrid_data))
width = 0.35

ax.bar(x - width/2, seg_hybrid_data['baseline_recall'], width,
       label='Calibrated single-model baseline', color='#4C72B0')
ax.bar(x + width/2, seg_hybrid_data['nn_stack_recall'], width,
       label='NN-stacking hybrid', color='#C44E52')

ax.set_xticks(x)
ax.set_xticklabels(seg_hybrid_data['facies'])
ax.set_xlabel('Facies')
ax.set_ylabel('Recall (LOWO cross-validation)')
ax.set_title('(e) SEG: calibrated baseline vs. NN-stacking hybrid, per facies')
ax.legend()
plt.tight_layout()
plt.savefig(f'fig5_seg_baseline_vs_nnstack.{FIG_FORMAT}', dpi=DPI, bbox_inches='tight')
plt.show()
print("Saved fig5_seg_baseline_vs_nnstack")

print("\nAll figures saved. Figures 1-4 use fully production-validated data.")
print("Figure 5 uses the SEG NN-stacking result, the only hybrid mechanism")
print("independently validated with the real XGBoost+Keras pipeline in Colab.")
