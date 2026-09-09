from _paths import OUT_RL, REP_ROOT
import os, sys, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = REP_ROOT
OUT_DIR  = OUT_RL

# ── Load saved metrics ─────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT_DIR, 'table_month_retrain_robustness.csv'))
print('Loaded metrics:')
print(df[['Month', 'N_test', 'N_positive_test', 'Recall', 'Precision', 'ROC_AUC']].to_string(index=False))

# ── Reconstruct per-season CMs ─────────────────────────────────────────────
cms = []
for _, row in df.iterrows():
    tp = int(round(row['Recall'] * row['N_positive_test']))
    fn = int(row['N_positive_test']) - tp
    fp = int(round(tp / row['Precision'])) - tp if row['Precision'] > 0 else 0
    tn = int(row['N_test']) - int(row['N_positive_test']) - fp
    cms.append({
        'Month':     row['Month'],
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Recall':    tp / (tp + fn),
        'Precision': tp / (tp + fp),
        'ROC_AUC':   row['ROC_AUC'],
        'PR_AUC':    row['PR_AUC'],
    })
    print(f"  {row['Month']:35s}: TN={tn:>10,}  FP={fp:>7,}  FN={fn:>3,}  TP={tp:>4,}"
          f"  Recall={tp/(tp+fn):.3f}  Precision={tp/(tp+fp):.4f}")

cm_df = pd.DataFrame(cms)

# ── Average across four seasons ────────────────────────────────────────────
avg = {
    'TN':        cm_df['TN'].mean(),
    'FP':        cm_df['FP'].mean(),
    'FN':        cm_df['FN'].mean(),
    'TP':        cm_df['TP'].mean(),
    'Recall':    cm_df['Recall'].mean(),
    'Precision': cm_df['Precision'].mean(),
    'ROC_AUC':   cm_df['ROC_AUC'].mean(),
    'PR_AUC':    cm_df['PR_AUC'].mean(),
}
print(f"\nAveraged across Feb / May / Aug / Nov:")
for k, v in avg.items():
    print(f"  {k:12s}: {v:,.1f}" if k in ('TN','FP','FN','TP') else f"  {k:12s}: {v:.4f}")

avg_cm = np.array([[avg['TN'], avg['FP']],
                   [avg['FN'], avg['TP']]])

# ── Figure: single averaged confusion matrix ───────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

from matplotlib.colors import LogNorm
import os
im = ax.imshow(avg_cm, cmap='Blues',
               norm=LogNorm(vmin=1, vmax=avg_cm.max()))

class_names  = ['No Steel Plant', 'Steel Plant']
cell_labels  = [['TN', 'FP'], ['FN', 'TP']]
total        = avg_cm.sum()

for i in range(2):
    for j in range(2):
        val = avg_cm[i, j]
        pct = val / total * 100
        text_color = 'white' if val > avg_cm.max() / 3 else 'black'
        bold = cell_labels[i][j] in ('TP', 'TN')
        ax.text(j, i,
                f'{cell_labels[i][j]}\n{val:,.0f}\n({pct:.3f}%)',
                ha='center', va='center', fontsize=11,
                color=text_color,
                fontweight='bold' if bold else 'normal')

ax.set_xticks([0, 1]); ax.set_xticklabels(class_names, fontsize=10)
ax.set_yticks([0, 1]); ax.set_yticklabels(class_names, fontsize=10)
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual',    fontsize=11)
ax.set_title(
    f'Location Model — Average Across Four Seasons\n'
    f'Recall = {avg["Recall"]:.3f}   Precision = {avg["Precision"]:.4f}'
    f'   ROC-AUC = {avg["ROC_AUC"]:.3f}',
    fontsize=10, pad=8)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# ── Per-season summary table as text below figure ─────────────────────────
season_lines = []
for r in cms:
    season_lines.append(
        f"{r['Month'].split(' ')[0]:4s}  Recall={r['Recall']:.3f}  "
        f"Precision={r['Precision']:.4f}  ROC-AUC={r['ROC_AUC']:.3f}"
    )
fig.text(0.5, -0.04,
         'Per-season breakdown:   ' + '   |   '.join(season_lines),
         ha='center', va='top', fontsize=7, color='#444444',
         style='italic')

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'fig_month_cm_averaged.pdf')
fig.savefig(fig_path, bbox_inches='tight')
plt.close()
print(f'\nSaved: {fig_path}')
print('Done.')
