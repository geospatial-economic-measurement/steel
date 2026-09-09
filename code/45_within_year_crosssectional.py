"""
45_within_year_crosssectional.py
================================
Response-letter Comment R2.2 — the LITERAL "year by year" exercise the
referee asked for: TRAIN THE MODEL ON EACH YEAR SEPARATELY and evaluate the
cross-section within that year.

Why this is distinct from Panel A (script 42):
  42_figR22_year_by_year_bars.py Panel A is LEAVE-ONE-YEAR-OUT (train on 3 years, predict the
  held-out year). That is an out-of-time *levels* forecast, and its R² ~0.8
  is dominated by cross-sectional carryover (plant identities/locations
  persist), so it does NOT cleanly isolate the cross-sectional component the
  referee (via Chen & Nordhaus 2019) is asking about.

  This script instead fits and evaluates WITHIN A SINGLE YEAR. Within one
  year there is essentially no temporal variation, so the resulting R² is a
  clean CROSS-SECTIONAL number: can the model tell units apart from satellite
  signals in that year? Reporting it for 2019-2022 shows how cross-sectional
  skill differs year by year — the exact disentangling requested.

Clean split (no leakage):
  For each year, GroupKFold(5) by unit (IDCode for grid, name_prod for plant)
  so that whole units are held out — a unit's own other months never appear in
  both train and test. OOF predictions over held-out units -> cross-sectional
  R²/RMSE for that year.

Spec is reconciled VERBATIM to scripts 25/42 (confirmed 40/41-feature set,
same data paths, same dedup guards, same XGBoost hyperparameters). Source
files intentionally unmodified.

Outputs:
  output/tables/table_within_year_crosssectional.csv
  console PASS summary
"""
import os, sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_TAB, need

warnings.filterwarnings('ignore')

# ── Paths (identical to scripts 25/26/42) ─────────────────────────────────────
GRID_DATA_PATH = need(DATA_PROCESSED, 'non_zero_df_month.csv')
SIGNALS_PATH   = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH  = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH       = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')
OUT_TAB        = OUT_TAB
os.makedirs(OUT_TAB, exist_ok=True)

EARTH_R       = 6371.0
YEARS         = [2019, 2020, 2021, 2022]
N_SPLITS      = 5

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

def within_year_cs(X, y, years, units, params, level):
    """For each year: GroupKFold(5) by unit within that year; OOF cross-sectional R²/RMSE."""
    out = []
    for yr in YEARS:
        m = years == yr
        Xy, yy, uy = X[m], y[m], units[m]
        n_units = len(np.unique(uy))
        k = min(N_SPLITS, n_units)
        gkf = GroupKFold(n_splits=k)
        oof = np.full(len(yy), np.nan)
        for tr, te in gkf.split(Xy, yy, groups=uy):
            model = xgb.XGBRegressor(**params)
            model.fit(Xy[tr], yy[tr])
            oof[te] = model.predict(Xy[te])
        r2   = r2_score(yy, oof)
        rmse = np.sqrt(mean_squared_error(yy, oof))
        out.append({'Level': level, 'Year': yr, 'R2': r2, 'RMSE_log': rmse,
                    'N_rows': len(yy), 'N_units': n_units, 'folds': k})
        print(f"    {level} {yr}: CS R² = {r2:.4f}  RMSE = {rmse:.4f}  "
              f"(n={len(yy)}, units={n_units}, folds={k})")
    return out

# ── Build grid dataset (verbatim from script 42) ──────────────────────────────
print("Loading grid data ...")
grid = pd.read_csv(GRID_DATA_PATH)
grid = grid.drop(columns=['Unnamed: 0', 'Polygon_ID'], errors='ignore')
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
X_grid, y_grid = grid[GRID_FEATURES].values, np.log(grid['GridProd_Steel_tot'].values)
yr_grid, unit_grid = grid['Year'].values, grid['IDCode'].values
print(f"  Grid rows: {len(grid)}")

# ── Build plant dataset (verbatim from script 42) ─────────────────────────────
print("Loading plant data ...")
signals  = pd.read_csv(SIGNALS_PATH).drop(columns=['PM1_MEAN'], errors='ignore')
gridprod = pd.read_csv(GRIDPROD_PATH)
gridprod = gridprod.drop_duplicates(subset=['IDCode', 'name_prod', 'Year', 'Month'])
bf_sheet   = pd.read_excel(GEM_PATH, sheet_name='Blast furnaces', header=0)
bf_gem_ids = set(bf_sheet.iloc[:, 0].dropna().astype(str).unique())
gem_tech   = gridprod[['name_prod', 'GEMPlantID']].drop_duplicates('name_prod').copy()
gem_tech['is_bof'] = gem_tech['GEMPlantID'].apply(lambda x: 1 if str(x) in bf_gem_ids else 0)
temp = gridprod[['IDCode', 'Longitude', 'Latitude', 'name_prod',
                 'Year', 'Month', 'Steel_Prod']].copy()
merged = pd.merge(signals, temp, on=['IDCode', 'Year', 'Month'], how='left')
agg_signals = {col: 'mean' for col in POLLUTANTS}
grouped = (merged.groupby(['name_prod', 'Year', 'Month'])
           .agg({**agg_signals, 'Longitude': 'mean', 'Latitude': 'mean',
                 'Steel_Prod': 'first'}).reset_index())
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
X_plant, y_plant = plant_data[PLANT_FEATURES].values, np.log1p(plant_data['Steel_Prod'].values)
yr_plant, unit_plant = plant_data['Year'].values, plant_data['name_prod'].values
print(f"  Plant rows: {len(plant_data)}")

GRID_PARAMS = dict(colsample_bytree=0.6, learning_rate=0.1, max_depth=6,
                   n_estimators=300, subsample=0.8, alpha=0.2, reg_lambda=0.5,
                   random_state=5, verbosity=0)
PLANT_PARAMS = dict(colsample_bytree=0.8, learning_rate=0.1, max_depth=6,
                    n_estimators=300, subsample=0.6, alpha=0.2, reg_lambda=0.5,
                    random_state=5, verbosity=0)

print("\nWithin-year cross-sectional (train each year separately, GroupKFold by unit):")
print("  Grid:")
rows  = within_year_cs(X_grid,  y_grid,  yr_grid,  unit_grid,  GRID_PARAMS,  'Grid')
print("  Plant:")
rows += within_year_cs(X_plant, y_plant, yr_plant, unit_plant, PLANT_PARAMS, 'Plant')

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_TAB, 'table_within_year_crosssectional.csv')
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

print("\n── Comparison: within-year CS (this script) vs LOYO levels (script 42) ──")
loyo = {'Grid':  {2019: 0.868, 2020: 0.875, 2021: 0.830, 2022: 0.795},
        'Plant': {2019: 0.853, 2020: 0.820, 2021: 0.759, 2022: 0.753}}
for lvl in ('Grid', 'Plant'):
    for yr in YEARS:
        cs = df[(df.Level == lvl) & (df.Year == yr)]['R2'].iloc[0]
        print(f"  {lvl} {yr}: within-year CS {cs:.3f}   |   LOYO levels {loyo[lvl][yr]:.3f}")
gmean = df[df.Level == 'Grid']['R2'].mean()
pmean = df[df.Level == 'Plant']['R2'].mean()
print(f"\n  Mean within-year CS R²:  Grid {gmean:.3f}   Plant {pmean:.3f}")

# ── Confound check: held-out UNIT but POOLED across all years ─────────────────
# Predicts never-seen units using ALL years of training data. Isolates whether
# the within-year collapse is driven by single-year training scarcity (sample
# size) vs a fundamental inability to predict unseen units from satellite alone.
def pooled_holdout_unit(X, y, units, params, level):
    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups=units):
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    r2 = r2_score(y, oof); rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"  {level}: pooled held-out-unit R² = {r2:.4f}  RMSE = {rmse:.4f}  "
          f"(n={len(y)}, units={len(np.unique(units))})")
    return r2

print("\n── Confound check: held-out UNIT, pooled across all years ──")
pooled_holdout_unit(X_grid,  y_grid,  unit_grid,  GRID_PARAMS,  'Grid')
pooled_holdout_unit(X_plant, y_plant, unit_plant, PLANT_PARAMS, 'Plant')
print("\nDone.")
