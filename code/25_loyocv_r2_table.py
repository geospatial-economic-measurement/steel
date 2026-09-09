import os, sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_FIG, OUT_TAB, need

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Paths ──────────────────────────────────────────────────────────────────────
GRID_DATA_PATH = need(DATA_PROCESSED, 'non_zero_df_month.csv')
SIGNALS_PATH   = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH  = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH       = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')
OUT_FIG        = OUT_FIG
OUT_TAB        = OUT_TAB
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)

EARTH_R    = 6371.0
HOLDOUT_YEARS = [2019, 2020, 2021, 2022]

CHINA_HUBS = {
    'Tangshan': (39.6, 118.2), 'Wuhan': (30.6, 114.3), 'Anshan': (41.1, 122.8),
    'Rizhao':   (35.4, 119.5), 'Baotou': (40.7, 109.8),
}
CHINA_PORTS = {
    'Shanghai':    (31.23, 121.47), 'Tianjin':     (38.98, 117.72),
    'Qingdao':     (36.07, 120.38), 'Ningbo':      (29.87, 121.55),
    'Guangzhou':   (23.10, 113.43), 'Dalian':      (38.92, 121.65),
    'Lianyungang': (34.75, 119.45), 'Yingkou':     (40.67, 122.23),
}

POLLUTANTS       = ['CO_MEAN', 'NO2_MEAN', 'SO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN',
                    'O3_MEAN', 'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN']
PLANT_POLLUTANTS = [f'Plant_{p}' for p in POLLUTANTS]

GRID_FEATURES = (
    POLLUTANTS +
    [f'{p}_lag{k}' for k in (1, 2, 3) for p in POLLUTANTS] +
    ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port']
)

PLANT_FEATURES = (
    PLANT_POLLUTANTS +
    [f'{p}_lag{k}' for k in (1, 2, 3) for p in PLANT_POLLUTANTS] +
    ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port', 'is_bof']
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((r2 - r1) / 2)**2 +
         np.cos(r1) * np.cos(r2) * np.sin(np.deg2rad(lon2 - lon1) / 2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def add_dist_features(df, lat_col, lon_col):
    lats, lons = df[lat_col].values, df[lon_col].values
    hub_d  = np.stack([haversine_km(lats, lons, la, lo)
                       for la, lo in CHINA_HUBS.values()], axis=1)
    port_d = np.stack([haversine_km(lats, lons, la, lo)
                       for la, lo in CHINA_PORTS.values()], axis=1)
    df = df.copy()
    df['dist_nearest_hub']  = hub_d.min(axis=1)
    df['dist_nearest_port'] = port_d.min(axis=1)
    return df

def loyocv(X_all, y_all, years_all, params, holdout_years=HOLDOUT_YEARS):
    """Leave-one-year-out CV. Returns dict year -> R²."""
    results = {}
    for yr in holdout_years:
        train_mask = years_all != yr
        test_mask  = years_all == yr
        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask],  y_all[test_mask]
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr)
        y_hat = m.predict(X_te)
        results[yr] = r2_score(y_te, y_hat)
        print(f"    {yr}: R² = {results[yr]:.4f}  (n_test={test_mask.sum()})")
    return results

# ── Build grid dataset ─────────────────────────────────────────────────────────
print("Loading grid data ...")
grid = pd.read_csv(GRID_DATA_PATH)
grid = grid.drop(columns=['Unnamed: 0', 'Polygon_ID'], errors='ignore')
# Dedup guard (2026-07): GridProd_1922_monthly.csv contains byte-identical
# duplicated Dec-2022 rows; dedup before any split/evaluation.
# Source files intentionally unmodified.
grid = grid.drop_duplicates(subset=['IDCode', 'Year', 'Month'])
grid['month_sin'] = np.sin(2 * np.pi * grid['Month'] / 12)
grid['month_cos'] = np.cos(2 * np.pi * grid['Month'] / 12)
grid = grid.sort_values(['IDCode', 'Year', 'Month']).reset_index(drop=True)
for lag in range(1, 4):
    for col in POLLUTANTS:
        grid[f'{col}_lag{lag}'] = grid.groupby('IDCode')[col].shift(lag)
grid = add_dist_features(grid, 'Centroid_Lat', 'Centroid_Long')
grid = grid.dropna(subset=GRID_FEATURES + ['GridProd_Steel_tot', 'Year'])
grid = grid[grid['GridProd_Steel_tot'] > 0]

X_grid  = grid[GRID_FEATURES].values
y_grid  = np.log(grid['GridProd_Steel_tot'].values)
yr_grid = grid['Year'].values
print(f"  Grid rows: {len(grid)}, features: {len(GRID_FEATURES)}")

# ── Build plant dataset ────────────────────────────────────────────────────────
print("Loading plant data ...")
signals  = pd.read_csv(SIGNALS_PATH)
signals  = signals.drop(columns=['PM1_MEAN'], errors='ignore')
gridprod = pd.read_csv(GRIDPROD_PATH)

gridprod = gridprod.drop_duplicates(subset=['IDCode', 'name_prod', 'Year', 'Month'])

bf_sheet   = pd.read_excel(GEM_PATH, sheet_name='Blast furnaces', header=0)
bf_gem_ids = set(bf_sheet.iloc[:, 0].dropna().astype(str).unique())
gem_tech   = (gridprod[['name_prod', 'GEMPlantID']]
              .drop_duplicates('name_prod').copy())
gem_tech['is_bof'] = gem_tech['GEMPlantID'].apply(
    lambda x: 1 if str(x) in bf_gem_ids else 0)

temp = gridprod[['IDCode', 'Longitude', 'Latitude', 'name_prod',
                 'Year', 'Month', 'Steel_Prod']].copy()
merged = pd.merge(signals, temp, on=['IDCode', 'Year', 'Month'], how='left')

agg_signals = {col: 'mean' for col in POLLUTANTS}
grouped = (merged
           .groupby(['name_prod', 'Year', 'Month'])
           .agg({**agg_signals,
                 'Longitude':  'mean',
                 'Latitude':   'mean',
                 'Steel_Prod': 'first'})
           .reset_index())
grouped.rename(columns={p: f'Plant_{p}' for p in POLLUTANTS}, inplace=True)
grouped = grouped.drop_duplicates(subset=['name_prod', 'Year', 'Month'])
grouped['plant_id'] = grouped['name_prod'].factorize()[0] + 1
grouped = grouped.merge(gem_tech[['name_prod', 'is_bof']], on='name_prod', how='left')
grouped['is_bof'] = grouped['is_bof'].fillna(0).astype(int)
grouped['month_sin'] = np.sin(2 * np.pi * grouped['Month'] / 12)
grouped['month_cos'] = np.cos(2 * np.pi * grouped['Month'] / 12)
grouped = grouped.sort_values(['plant_id', 'Year', 'Month']).reset_index(drop=True)
for lag in range(1, 4):
    for col in PLANT_POLLUTANTS:
        grouped[f'{col}_lag{lag}'] = grouped.groupby('plant_id')[col].shift(lag)
grouped = add_dist_features(grouped, 'Latitude', 'Longitude')
plant_data = grouped.dropna(subset=PLANT_FEATURES + ['Steel_Prod', 'Year']).copy()
plant_data = plant_data[plant_data['Steel_Prod'] > 0]

X_plant  = plant_data[PLANT_FEATURES].values
y_plant  = np.log1p(plant_data['Steel_Prod'].values)
yr_plant = plant_data['Year'].values
print(f"  Plant rows: {len(plant_data)}, features: {len(PLANT_FEATURES)}")

# ── XGBoost hyperparameters (level-specific) ───────────────────────────────────
GRID_PARAMS = dict(
    colsample_bytree=0.6, learning_rate=0.1, max_depth=6,
    n_estimators=300, subsample=0.8,
    alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0,
)
PLANT_PARAMS = dict(
    colsample_bytree=0.8, learning_rate=0.1, max_depth=6,
    n_estimators=300, subsample=0.6,
    alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0,
)

# ── Run LOYOCV ─────────────────────────────────────────────────────────────────
print("\nLOYOCV — Grid level:")
grid_r2 = loyocv(X_grid, y_grid, yr_grid, GRID_PARAMS)

print("\nLOYOCV — Plant level:")
plant_r2 = loyocv(X_plant, y_plant, yr_plant, PLANT_PARAMS)

# ── Build Panel A table ────────────────────────────────────────────────────────
rows = []
for yr in HOLDOUT_YEARS:
    rows.append({'Level': 'Grid',  'Year': yr, 'R2': grid_r2[yr]})
    rows.append({'Level': 'Plant', 'Year': yr, 'R2': plant_r2[yr]})
df_out = pd.DataFrame(rows)
pivot = df_out.pivot(index='Level', columns='Year', values='R2')
pivot['Mean'] = pivot.mean(axis=1)
pivot = pivot.loc[['Grid', 'Plant']]

print("\n── Panel A: LOYOCV R² by Year ──────────────────────────────────────────")
print(pivot.round(3).to_string())

pivot.reset_index().to_csv(
    os.path.join(OUT_TAB, 'table_a4_loyocv_r2.csv'), index=False)
print(f"\nSaved: {os.path.join(OUT_TAB, 'table_a4_loyocv_r2.csv')}")

# ── Figure: year-by-year R² line plot ─────────────────────────────────────────
print("\nPlotting figure ...")
fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
ax.plot(HOLDOUT_YEARS, [grid_r2[y]  for y in HOLDOUT_YEARS],
        marker='o', label='Grid', color='steelblue')
ax.plot(HOLDOUT_YEARS, [plant_r2[y] for y in HOLDOUT_YEARS],
        marker='s', label='Plant', color='darkorange')
ax.axhline(pivot.loc['Grid',  'Mean'], color='steelblue',  linestyle='--', lw=1, alpha=0.6)
ax.axhline(pivot.loc['Plant', 'Mean'], color='darkorange', linestyle='--', lw=1, alpha=0.6)
ax.set_xlabel('Holdout Year', fontsize=13)
ax.set_ylabel('$R^2$', fontsize=13)
ax.set_title('LOYOCV Year-by-Year $R^2$ (Grid vs Plant)', fontsize=14)
ax.set_xticks(HOLDOUT_YEARS)
ax.set_ylim(0.7, 1.0)
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

for ext in ('pdf', 'png'):
    p = os.path.join(OUT_FIG, f'figA4_loyocv_r2.{ext}')
    try:
        fig.savefig(p, bbox_inches='tight')
        print(f"  Saved: {p}")
    except PermissionError:
        p2 = os.path.join(OUT_FIG, f'figA4_loyocv_r2_v2.{ext}')
        fig.savefig(p2, bbox_inches='tight')
        print(f"  Saved (alt): {p2}")
plt.close()

print("\nDone.")
