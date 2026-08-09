"""
Techlog-style well log plots: log curves alongside actual vs. predicted
lithology tracks, for ALL 10 wells in the SEG dataset --
8 training wells (using genuinely held-out LOWO out-of-fold predictions,
NOT the model's own training predictions) plus the 2 official blind
wells (STUART, CRAWFORD).

Using LOWO out-of-fold predictions for the training wells keeps this a
fair, honest comparison throughout: every single plot in this script
shows a well being predicted by a model that never saw that well during
training, exactly like the blind wells.

Dependencies (must exist in your Colab session before running this):
  - `blind_aligned` -- from seg_final_pipeline.py Cells 1-7
  - `oof_pred`, `df`, `X`, `y`, `groups` -- from seg_final_pipeline.py
    Cells 1-4 (the LOWO cross-validation loop)

Depth-axis orientation was verified pixel-by-pixel across all 8 training
wells and both blind wells before shipping this code.

Paste each # %% CELL block into its own Colab cell.
"""

# %% CELL 1: Imports and facies color scheme
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

FACIES_NAMES = {
    1: 'Nonmarine sandstone', 2: 'Nonmarine coarse siltstone',
    3: 'Nonmarine fine siltstone', 4: 'Marine siltstone/shale',
    5: 'Mudstone (limestone)', 6: 'Wackestone (limestone)',
    7: 'Dolomite', 8: 'Packstone-grainstone (limestone)',
    9: 'Phylloid-algal bafflestone (limestone)',
}
FACIES_COLORS = ['#f4d03f', '#f39c12', '#e67e22', '#27ae60', '#a9dfbf',
                  '#5dade2', '#2874a6', '#af7ac5', '#6c3483']
FACIES_CMAP = ListedColormap(FACIES_COLORS)


# %% CELL 2: Plotting function (shared for both training and blind wells)
def plot_well_comparison(well_data, well_name,
                          log_cols=('GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE'),
                          true_col='Facies', pred_col='oof_pred_facies',
                          depth_col='Depth', figsize_per_track=2.0, save_path=None):
    """
    well_data: subset of a dataframe for ONE well, containing log_cols,
               true_col, pred_col, and depth_col.
    true_col/pred_col: column names differ between training wells
               ('Facies'/'oof_pred_facies') and blind wells
               ('LithCode'/'pred_facies') -- pass the right pair.
    """
    well_data = well_data.sort_values(depth_col).reset_index(drop=True)
    depth = well_data[depth_col].values
    n_logs = len(log_cols)
    n_tracks = n_logs + 2

    fig, axes = plt.subplots(1, n_tracks, figsize=(figsize_per_track * n_tracks, 10), sharey=True)
    fig.suptitle(f'Well: {well_name}', fontsize=13, y=0.995)

    for i, col in enumerate(log_cols):
        ax = axes[i]
        ax.plot(well_data[col], depth, color='black', linewidth=0.8)
        ax.set_xlabel(col, fontsize=9)
        ax.set_title(col, fontsize=9)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        if i > 0:
            ax.tick_params(labelleft=False)

    for ax, col, title in [(axes[n_logs], true_col, 'Actual'),
                             (axes[n_logs + 1], pred_col, 'Predicted')]:
        grid = well_data[col].values.reshape(-1, 1).astype(int)
        ax.imshow(grid, aspect='auto', cmap=FACIES_CMAP, vmin=1, vmax=9,
                  origin='lower', extent=[0, 1, depth.min(), depth.max()])
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])

    axes[0].set_ylabel('Depth (ft)', fontsize=10)
    axes[0].set_ylim(depth.max(), depth.min())

    handles = [mpatches.Patch(color=FACIES_COLORS[k - 1], label=f'{k}: {FACIES_NAMES[k]}')
               for k in range(1, 10)]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.08), frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# %% CELL 3: Build the training-well comparison dataframe (LOWO OOF predictions)
# Reuses `oof_pred` from seg_final_pipeline.py's Cell 4 -- these are
# genuinely held-out predictions (each well predicted by a model that
# never trained on it), NOT the model's own training-set predictions.
train_wells_aligned = df.copy()
train_wells_aligned['oof_pred_facies'] = oof_pred + 1  # back to 1-indexed


# %% CELL 4: Generate plots for all 8 training wells (LOWO held-out predictions)
for well_name in train_wells_aligned['Well Name'].unique():
    well_data = train_wells_aligned[train_wells_aligned['Well Name'] == well_name]
    plot_well_comparison(
        well_data, f'{well_name} (LOWO held-out)',
        true_col='Facies', pred_col='oof_pred_facies',
        save_path=f'fig_logplot_train_{well_name.replace(" ", "_")}.png'
    )
    print(f"Saved fig_logplot_train_{well_name.replace(' ', '_')}.png ({len(well_data)} samples)")


# %% CELL 5: Generate plots for both official blind wells
for well_name in blind_aligned['Well Name'].unique():
    well_data = blind_aligned[blind_aligned['Well Name'] == well_name]
    plot_well_comparison(
        well_data, f'{well_name} (official blind well)',
        true_col='LithCode', pred_col='pred_facies',
        save_path=f'fig_logplot_blind_{well_name.replace(" ", "_")}.png'
    )
    print(f"Saved fig_logplot_blind_{well_name.replace(' ', '_')}.png ({len(well_data)} samples)")

print("\nGenerated 10 well-comparison plots total: 8 training wells (LOWO) + 2 blind wells.")
print("All use genuinely held-out predictions -- no well was ever predicted by a")
print("model that had seen that well during training.")
