
import os, sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from _paths import DATA_PROCESSED, OUT_FIG, OUT_RL, need
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

# -- Paths --------------------------------------------------------------------
DATA_PATH = need(DATA_PROCESSED, 'non_zero_df_month.csv')
OUT_DIR   = OUT_FIG
OUT_RL    = OUT_RL
PAPER_FIG = OUT_FIG

EARTH_R = 6371.0

SIGNALS = ['CO_MEAN', 'NO2_MEAN', 'SO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN',
           'O3_MEAN', 'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN']

CHINA_HUBS = {
    'Tangshan': (39.6, 118.2), 'Wuhan':  (30.6, 114.3),
    'Anshan':   (41.1, 122.8), 'Rizhao': (35.4, 119.5),
    'Baotou':   (40.7, 109.8),
}
CHINA_PORTS = {
    'Shanghai':    (31.23, 121.47), 'Tianjin':     (38.98, 117.72),
    'Qingdao':     (36.07, 120.38), 'Ningbo':      (29.87, 121.55),
    'Guangzhou':   (23.10, 113.43), 'Dalian':      (38.92, 121.65),
    'Lianyungang': (34.75, 119.45), 'Yingkou':     (40.67, 122.23),
}

def haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((r2-r1)/2)**2 +
         np.cos(r1)*np.cos(r2)*np.sin(np.deg2rad(lon2-lon1)/2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

# -- Load and build features --------------------------------------------------
print('Loading data ...')
data = pd.read_csv(DATA_PATH)
data = data.drop(columns=['Unnamed: 0', 'Polygon_ID'], errors='ignore')
print(f'  Rows: {len(data):,}')

data = data.sort_values(['IDCode', 'Year', 'Month']).reset_index(drop=True)
for lag in range(1, 4):
    for col in SIGNALS:
        data[f'{col}_lag{lag}'] = data.groupby('IDCode')[col].shift(lag)

data['month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

lats, lons = data['Centroid_Lat'].values, data['Centroid_Long'].values
hub_d  = np.stack([haversine_km(lats, lons, la, lo) for la, lo in CHINA_HUBS.values()],  axis=1)
port_d = np.stack([haversine_km(lats, lons, la, lo) for la, lo in CHINA_PORTS.values()], axis=1)
data['dist_nearest_hub']  = hub_d.min(axis=1)
data['dist_nearest_port'] = port_d.min(axis=1)

# -- Ordered columns: signal groups first, extra features last ----------------
new_order_cols = []
for sig in SIGNALS:
    new_order_cols.append(sig)
    for lag in range(1, 4):
        new_order_cols.append(f'{sig}_lag{lag}')
new_order_cols += ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port']

data = data[new_order_cols].dropna()
print(f'  After dropna: {len(data):,} rows, {len(new_order_cols)} features')

# -- Tick labels: short signal name + lag -------------------------------------
SIG_SHORT = {
    'CO_MEAN':    'CO',    'NO2_MEAN':   'NO₂', 'SO2_MEAN':   'SO₂',
    'PM2_5_MEAN': 'PM₂.₅', 'PM10_MEAN': 'PM₁₀',
    'O3_MEAN':    'O₃',
    'LSTA_MEAN':  'LST_d', 'LSTT_MEAN':  'LST_n',   'NTL_MEAN':   'NTL',
}
LAG_TAG = ['t', 't-1', 't-2', 't-3']

tick_labels = []
for sig in SIGNALS:
    for lag in range(4):
        tick_labels.append(f'{SIG_SHORT[sig]}\n({LAG_TAG[lag]})')
tick_labels += ['sin\n(mon)', 'cos\n(mon)', 'dist\nhub', 'dist\nport']

data.columns = tick_labels

# -- Compute Pearson correlation ----------------------------------------------
print('Computing 40×40 correlation matrix ...')
corr = data.corr(method='pearson')

# -- Signal group colours for axis annotations --------------------------------
GROUP_COLORS = {
    'Air\npollutants\n(CO,NO₂,SO₂,\nPM₂.₅,PM₁₀,O₃)': ('#d0e8f5', 0,  24),
    'Thermal\n(LST_d,\nLST_n)':                        ('#fde8cc', 24, 32),
    'Light\n(NTL)':                                     ('#d8f0d8', 32, 36),
    'Temporal &\ndistance':                             ('#ece8f8', 36, 40),
}

# -- Plot ---------------------------------------------------------------------
N = 40
fig = plt.figure(figsize=(18, 16), dpi=200)
ax = fig.add_axes([0.14, 0.10, 0.78, 0.78])   # [left, bottom, width, height]

mask_upper = np.triu(np.ones((N, N), dtype=bool), k=1)

g = sns.heatmap(
    corr, ax=ax, mask=mask_upper, cmap='RdBu_r',
    vmin=-1, vmax=1, center=0, square=True,
    linewidths=0.15, linecolor='#e0e0e0',
    cbar_kws={'shrink': 0.55, 'label': 'Pearson r', 'pad': 0.02},
    xticklabels=tick_labels, yticklabels=tick_labels, annot=False,
)

ax.tick_params(axis='x', labelsize=7.5, rotation=90, pad=2)
ax.tick_params(axis='y', labelsize=7.5, rotation=0,  pad=2)

for i in range(0, 37, 4):
    lw = 2.0 if i == 36 else 1.0
    ax.axhline(i, color='white', linewidth=lw, zorder=3)
    ax.axvline(i, color='white', linewidth=lw, zorder=3)

ax_y = fig.add_axes([0.01, 0.10, 0.025, 0.78], sharey=None)
ax_y.set_xlim(0, 1); ax_y.set_ylim(N, 0)
ax_y.axis('off')

for label, (color, start, end) in GROUP_COLORS.items():
    mid = (start + end) / 2
    ax_y.add_patch(mpatches.FancyBboxPatch(
        (0.3, start), 0.7, end - start,
        boxstyle='square,pad=0', linewidth=0,
        facecolor=color, transform=ax_y.transData, clip_on=False))
    ax_y.text(0.1, mid, label, va='center', ha='right',
              fontsize=6.5, transform=ax_y.transData,
              rotation=0, multialignment='center')

for dest in [OUT_DIR, OUT_RL, PAPER_FIG]:
    os.makedirs(dest, exist_ok=True)
    fig.savefig(os.path.join(dest, 'correlation_matrix.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(dest, 'correlation_matrix.png'), bbox_inches='tight', dpi=200)
    print(f'Saved to {dest}')

plt.close()
print('Done.')
