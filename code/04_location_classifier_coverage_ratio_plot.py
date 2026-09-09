"""
04_location_classifier_coverage_ratio_plot.py
===============================================
Generates Figure 3: "Percentage of identified steel plants from the holdout
sample" — coverage ratio vs. tolerance distance (1–10 km) at three
probability thresholds (P≥0.5, P≥0.7, P≥0.9).

Method:
  1. Load the trained November MLP (from script 01) and preprocessing artifacts.
  2. Score all November grid cells from the full-population CSVs.
  3. Join predictions with grid-cell centroids (gridinfo_xy.csv) to get lat/lon.
  4. Define the holdout set: GEM plants in Location_Plants_Full.xlsx that
     were NOT included in the training sample (re_data / GridProd labels).
  5. For each threshold × distance combination, compute the fraction of holdout
     plants that lie within that distance of at least one prediction ≥ threshold.
  6. Save coverage_ratio_plot.pdf to output figures and paper Figure folder.
  7. Save table_coverage_by_threshold.csv (Output/tables) with the exact arrays
     plotted: radius_km, threshold, hit_rate, n_hit, n_total. Added 2026-07-24
     so the headline hit rates (e.g. ~88% within 1 km at P>=0.5) are auditable
     without re-running the model; purely additive, no computation changed.

Dependencies (run script 01 first):
  output/figures/classification_mlp_36feat.keras
  output/figures/classification_preprocessing.pkl

Requires confidential data:
  Data/processed/confidential/gridinfo_xy.csv   — IDCode → (lat, lon)
  Data/processed/confidential/GridProd_1922_monthly.csv — training labels
  Data/processed/confidential/Location_Plants_Full.xlsx — GEM plant list
"""
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, OUT_FIG, OUT_TAB, need

import os, sys, io, warnings, gc, pickle
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DATA    = DATA_PROCESSED
CSV1         = os.path.join(BASE_DATA, 'first_output_file.csv')
CSV2         = os.path.join(BASE_DATA, 'second_output_file.csv')
GRIDPROD_CSV = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GRIDINFO_CSV = need(DATA_CONFIDENTIAL, 'gridinfo_xy.csv')
PLANTS_XLSX  = need(DATA_CONFIDENTIAL, 'Location_Plants_Full.xlsx')

MODEL_DIR  = OUT_FIG
OUT_FIG    = OUT_FIG
OUT_TAB    = OUT_TAB
PAPER_FIG  = OUT_FIG

os.makedirs(OUT_FIG,   exist_ok=True)
os.makedirs(OUT_TAB,   exist_ok=True)
os.makedirs(PAPER_FIG, exist_ok=True)

SIGNALS    = ['CO_MEAN', 'NO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN', 'SO2_MEAN',
              'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN', 'O3_MEAN']
TARGET_MONTH = 11   # November — the baseline model month
CHUNKSIZE    = 500_000
BATCH_SIZE   = 1_000_000
R_EARTH      = 6371.0

# ── Load preprocessing artifacts and model ────────────────────────────────────
print('Loading preprocessing artifacts ...')
with open(os.path.join(MODEL_DIR, 'classification_preprocessing.pkl'), 'rb') as f:
    prep = pickle.load(f)
imputer = prep['imputer']
scaler  = prep['scaler']

from tensorflow.keras.models import load_model
import os
print('Loading MLP model ...')
model = load_model(os.path.join(MODEL_DIR, 'classification_mlp_36feat.keras'))
print('  Model loaded.')

# ── Load grid centroids (IDCode → lat/lon) ─────────────────────────────────────
print('Loading grid centroids ...')
grid_xy = pd.read_csv(GRIDINFO_CSV, usecols=['IDCode', 'Centroid_Lat', 'Centroid_Long'],
                      dtype={'IDCode': np.int32})
grid_xy = grid_xy.drop_duplicates('IDCode').set_index('IDCode')
print(f'  {len(grid_xy):,} grid cells with coordinates')

# ── Load November data from full-population CSVs ──────────────────────────────
print(f'Loading November (month={TARGET_MONTH}) data from full-population CSVs ...')
nov_chunks = []
for fpath in [CSV1, CSV2]:
    if not os.path.exists(fpath):
        print(f'  [skip] {os.path.basename(fpath)} — not found')
        continue
    for chunk in pd.read_csv(fpath, chunksize=CHUNKSIZE, low_memory=False,
                              usecols=['IDCode', 'Year', 'Month'] + SIGNALS,
                              dtype={'IDCode': np.int32, 'Year': np.int16,
                                     'Month': np.int8,
                                     **{s: np.float32 for s in SIGNALS}}):
        nov_chunks.append(chunk[chunk['Month'] == TARGET_MONTH])
    print(f'  {os.path.basename(fpath)} scanned.')

nov = pd.concat(nov_chunks, ignore_index=True)
del nov_chunks; gc.collect()
print(f'  November rows loaded: {len(nov):,}')

# ── Build lag features (lag1–3 grouped by IDCode) ─────────────────────────────
print('Computing lag features ...')
nov = nov.sort_values(['IDCode', 'Year', 'Month']).reset_index(drop=True)
for lag in range(1, 4):
    for sig in SIGNALS:
        nov[f'{sig}_lag{lag}'] = nov.groupby('IDCode')[sig].shift(lag)

nov = nov.dropna()
print(f'  After dropna: {len(nov):,} rows')

# ── Join coordinates ───────────────────────────────────────────────────────────
nov = nov.join(grid_xy, on='IDCode', how='inner')
print(f'  After coord join: {len(nov):,} rows')

# ── Feature matrix ────────────────────────────────────────────────────────────
feat_cols = SIGNALS + [f'{s}_lag{k}' for k in (1, 2, 3) for s in SIGNALS]
X = nov[feat_cols].to_numpy(dtype=np.float32)
lats = nov['Centroid_Lat'].to_numpy()
lons = nov['Centroid_Long'].to_numpy()
del nov; gc.collect()

# ── Score in batches ──────────────────────────────────────────────────────────
print(f'Scoring {len(X):,} November cells ...')
X = imputer.transform(X)
X = scaler.transform(X)
n = len(X)
y_prob = np.empty(n, dtype=np.float32)
for start in range(0, n, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n)
    y_prob[start:end] = model.predict(X[start:end], verbose=0).ravel()
    print(f'  Scored {end:,} / {n:,}')
del X; gc.collect()

# ── Define holdout: GEM plants NOT in training labels ─────────────────────────
print('Loading holdout plant set ...')
plants = pd.read_excel(PLANTS_XLSX)
plants.columns = [c.lower() for c in plants.columns]
lat_col = next(c for c in plants.columns if 'lat' in c)
lon_col = next(c for c in plants.columns if 'lon' in c)
# 'soestatus' also contains 'status' — require the operating-status column
status_col = next((c for c in plants.columns if 'operatingstatus' in c), None)

if status_col:
    plants = plants[plants[status_col].astype(str).str.lower().str.strip() == 'operating']
plants = plants[[lat_col, lon_col]].dropna().drop_duplicates()
plants.columns = ['Latitude', 'Longitude']

# Exclude plants in the training set (those with GridProd labels)
gp = pd.read_csv(GRIDPROD_CSV, usecols=['IDCode'])
training_ids = set(gp['IDCode'].unique())
del gp

# Find GEM plants that are NOT associated with any training IDCode
# (proxy: plants within 10 km of a training grid cell are considered "training plants")
train_mask = np.isin(
    np.arange(len(lats)),
    np.arange(len(lats))   # placeholder — see below
)

# Build BallTree on training grid cells to find which GEM plants are in training set
# A GEM plant is "training" if it's within 1 km of any grid cell with non-zero production label
# Here we use GridProd to get coordinates of training (non-holdout) plant grids
gp_full = pd.read_csv(GRIDPROD_CSV,
                       usecols=['IDCode', 'GridProd_Steel_tot'])
train_idcodes = set(gp_full[gp_full['GridProd_Steel_tot'] > 0]['IDCode'].unique())
del gp_full

# Get lat/lon of training plant grids
train_xy = grid_xy.loc[grid_xy.index.isin(train_idcodes)].dropna()
train_rad = np.radians(train_xy[['Centroid_Lat', 'Centroid_Long']].values.astype(float))
train_tree = BallTree(train_rad, metric='haversine')

# GEM plants within 1 km of a training grid = in training set
gem_rad = np.radians(plants[['Latitude', 'Longitude']].values.astype(float))
counts = train_tree.query_radius(gem_rad, r=1.0 / R_EARTH, count_only=True)
holdout = plants[counts == 0].copy().reset_index(drop=True)
print(f'  GEM plants total: {len(plants)} | training: {(counts>0).sum()} | holdout: {len(holdout)}')

# ── Build BallTree on predictions ─────────────────────────────────────────────
print('Building BallTree on November predictions ...')
pred_rad = np.radians(np.column_stack([lats, lons]))
pred_tree = BallTree(pred_rad, metric='haversine')

# ── Compute coverage ratios ────────────────────────────────────────────────────
THRESHOLDS = [0.5, 0.7, 0.9]
D_km_vals  = np.arange(1, 11)   # 1–10 km in 1 km steps

holdout_rad = np.radians(holdout[['Latitude', 'Longitude']].values.astype(float))

print('Computing coverage ratios ...')
results  = {thr: [] for thr in THRESHOLDS}
csv_rows = []   # additive: capture exactly what gets plotted

for thr in THRESHOLDS:
    above = y_prob >= thr
    print(f'  threshold={thr}: {above.sum():,} cells above threshold')
    for D in D_km_vals:
        r_rad = D / R_EARTH
        idxs  = pred_tree.query_radius(holdout_rad, r=r_rad)
        hits  = sum(1 for idx in idxs if len(idx) > 0 and above[idx].any())
        ratio = hits / len(holdout)
        results[thr].append(ratio)
        csv_rows.append({'radius_km': int(D), 'threshold': thr,
                         'hit_rate': ratio, 'n_hit': hits,
                         'n_total': len(holdout)})
    print(f'  D=1km: {results[thr][0]:.1%}, D=5km: {results[thr][4]:.1%}, D=10km: {results[thr][9]:.1%}')

# ── Save plotted arrays as CSV (additive; see docstring step 7) ───────────────
csv_path = os.path.join(OUT_TAB, 'table_coverage_by_threshold.csv')
pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
print(f'Saved: {csv_path}')

# ── Plot ───────────────────────────────────────────────────────────────────────
print('Plotting coverage ratio figure ...')
COLORS = {0.5: 'tab:red', 0.7: 'tab:blue', 0.9: 'tab:green'}
LABELS = {0.5: 'Predicted Prob > 0.5', 0.7: 'Predicted Prob > 0.7', 0.9: 'Predicted Prob > 0.9'}

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

for thr in THRESHOLDS:
    ax.plot(D_km_vals, [r * 100 for r in results[thr]],
            marker='o', markersize=5, linewidth=2,
            color=COLORS[thr], label=LABELS[thr])

ax.set_xlabel('Tolerance Distance (km)', fontsize=12)
ax.set_ylabel('Coverage Ratio (%)', fontsize=12)
ax.set_xticks(D_km_vals)
ax.set_ylim(60, 100)
ax.set_xlim(0.5, 10.5)
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

for dest in [OUT_FIG, PAPER_FIG]:
    out_path = os.path.join(dest, 'coverage_ratio_plot.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved: {out_path}')
plt.close()

print('\nDone. Figure 3 (coverage_ratio_plot.pdf) generated.')
