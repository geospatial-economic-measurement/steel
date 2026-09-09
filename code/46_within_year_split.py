
import os, sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_FIG, OUT_RL, OUT_TAB, need

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

GRID_DATA_PATH = need(DATA_PROCESSED, 'non_zero_df_month.csv')
SIGNALS_PATH   = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH  = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH       = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')
OUT_TAB        = OUT_TAB
os.makedirs(OUT_TAB, exist_ok=True)

EARTH_R = 6371.0
YEARS   = [2019, 2020, 2021, 2022]
SEED    = 42

CHINA_HUBS = {'Tangshan': (39.6, 118.2), 'Wuhan': (30.6, 114.3), 'Anshan': (41.1, 122.8),
              'Rizhao': (35.4, 119.5), 'Baotou': (40.7, 109.8)}
CHINA_PORTS = {'Shanghai': (31.23, 121.47), 'Tianjin': (38.98, 117.72),
               'Qingdao': (36.07, 120.38), 'Ningbo': (29.87, 121.55),
               'Guangzhou': (23.10, 113.43), 'Dalian': (38.92, 121.65),
               'Lianyungang': (34.75, 119.45), 'Yingkou': (40.67, 122.23)}

POLLUTANTS       = ['CO_MEAN', 'NO2_MEAN', 'SO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN',
                    'O3_MEAN', 'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN']
PLANT_POLLUTANTS = [f'Plant_{p}' for p in POLLUTANTS]
GRID_FEATURES = (POLLUTANTS + [f'{p}_lag{k}' for k in (1, 2, 3) for p in POLLUTANTS] +
                 ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port'])
PLANT_FEATURES = (PLANT_POLLUTANTS + [f'{p}_lag{k}' for k in (1, 2, 3) for p in PLANT_POLLUTANTS] +
                  ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port', 'is_bof'])

def haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((r2 - r1) / 2)**2 + np.cos(r1) * np.cos(r2) * np.sin(np.deg2rad(lon2 - lon1) / 2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def add_dist_features(df, lat_col, lon_col):
    lats, lons = df[lat_col].values, df[lon_col].values
    hub_d  = np.stack([haversine_km(lats, lons, la, lo) for la, lo in CHINA_HUBS.values()], axis=1)
    port_d = np.stack([haversine_km(lats, lons, la, lo) for la, lo in CHINA_PORTS.values()], axis=1)
    df = df.copy()
    df['dist_nearest_hub']  = hub_d.min(axis=1)
    df['dist_nearest_port'] = port_d.min(axis=1)
    return df

# ── Grid dataset (verbatim from script 42) ────────────────────────────────────
print("Loading grid data ...")
grid = pd.read_csv(GRID_DATA_PATH).drop(columns=['Unnamed: 0', 'Polygon_ID'], errors='ignore')
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
grid['_y'] = np.log(grid['GridProd_Steel_tot'].values)
print(f"  Grid rows: {len(grid)}")

# ── Plant dataset (verbatim from script 42) ───────────────────────────────────
print("Loading plant data ...")
signals  = pd.read_csv(SIGNALS_PATH).drop(columns=['PM1_MEAN'], errors='ignore')
gridprod = pd.read_csv(GRIDPROD_PATH).drop_duplicates(subset=['IDCode', 'name_prod', 'Year', 'Month'])
bf_sheet   = pd.read_excel(GEM_PATH, sheet_name='Blast furnaces', header=0)
bf_gem_ids = set(bf_sheet.iloc[:, 0].dropna().astype(str).unique())
gem_tech   = gridprod[['name_prod', 'GEMPlantID']].drop_duplicates('name_prod').copy()
gem_tech['is_bof'] = gem_tech['GEMPlantID'].apply(lambda x: 1 if str(x) in bf_gem_ids else 0)
temp = gridprod[['IDCode', 'Longitude', 'Latitude', 'name_prod', 'Year', 'Month', 'Steel_Prod']].copy()
merged = pd.merge(signals, temp, on=['IDCode', 'Year', 'Month'], how='left')
grouped = (merged.groupby(['name_prod', 'Year', 'Month'])
           .agg({**{c: 'mean' for c in POLLUTANTS}, 'Longitude': 'mean', 'Latitude': 'mean',
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
plant = grouped.dropna(subset=PLANT_FEATURES + ['Steel_Prod', 'Year']).copy()
plant = plant[plant['Steel_Prod'] > 0]
plant['_y'] = np.log1p(plant['Steel_Prod'].values)
print(f"  Plant rows: {len(plant)}")

GRID_PARAMS  = dict(colsample_bytree=0.6, learning_rate=0.1, max_depth=6, n_estimators=300,
                    subsample=0.8, alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0)
PLANT_PARAMS = dict(colsample_bytree=0.8, learning_rate=0.1, max_depth=6, n_estimators=300,
                    subsample=0.6, alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0)

def year_8020(df, feats, params, level):
    rows = []
    for yr in YEARS:
        sub = df[df['Year'] == yr]
        X, y = sub[feats].values, sub['_y'].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
        m = xgb.XGBRegressor(**params)
        m.fit(Xtr, ytr)
        yp = m.predict(Xte)
        r2 = r2_score(yte, yp); rmse = np.sqrt(mean_squared_error(yte, yp))
        rows.append({'Level': level, 'Year': yr, 'R2': r2, 'RMSE_log': rmse,
                     'N_train': len(ytr), 'N_test': len(yte)})
        print(f"    {level} {yr}: R² = {r2:.4f}  RMSE = {rmse:.4f}  "
              f"(train={len(ytr)}, test={len(yte)})")
    return rows

print("\nPer-year 80/20 split (train each year separately):")
print("  Grid:")
rows  = year_8020(grid,  GRID_FEATURES,  GRID_PARAMS,  'Grid')
print("  Plant:")
rows += year_8020(plant, PLANT_FEATURES, PLANT_PARAMS, 'Plant')

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_TAB, 'table_within_year_split_8020.csv')
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")
print(f"\nMean R²:  Grid {df[df.Level=='Grid']['R2'].mean():.3f}   "
      f"Plant {df[df.Level=='Plant']['R2'].mean():.3f}")

# ── Figure (2-panel, styled to match figR2.2_year_by_year_r2.pdf) ─────────────
print("Plotting figure ...")
OUT_FIG_LOCAL = OUT_RL
OUT_RL        = OUT_RL
OUT_FIG       = OUT_FIG
for d in (OUT_FIG_LOCAL, OUT_RL, OUT_FIG):
    os.makedirs(d, exist_ok=True)

grid_df, plant_df = df[df.Level == 'Grid'].sort_values('Year'), df[df.Level == 'Plant'].sort_values('Year')
x, width = np.arange(len(YEARS)), 0.35
fig = plt.figure(figsize=(10, 5), dpi=300)
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)
ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

ax1.bar(x - width/2, grid_df['R2'],  width, label='Grid',  color='steelblue',  alpha=0.85)
ax1.bar(x + width/2, plant_df['R2'], width, label='Plant', color='darkorange', alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(YEARS); ax1.set_ylim(0, 1.05)
ax1.set_ylabel('$R^2$'); ax1.set_title('(A) $R^2$ by Year (within-year 80/20 split)')
ax1.axhline(0.8, color='gray', linestyle='--', linewidth=0.8); ax1.legend()

ax2.bar(x - width/2, grid_df['RMSE_log'],  width, label='Grid',  color='steelblue',  alpha=0.85)
ax2.bar(x + width/2, plant_df['RMSE_log'], width, label='Plant', color='darkorange', alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(YEARS)
ax2.set_ylabel('RMSE (log output)'); ax2.set_title('(B) RMSE by Year (log output)'); ax2.legend()

fig.tight_layout()
for dest in (OUT_FIG_LOCAL, OUT_RL, OUT_FIG):
    out_path = os.path.join(dest, 'figR2.2_within_year_8020.pdf')
    try:
        fig.savefig(out_path, format='pdf', bbox_inches='tight')
        print(f"Saved: {out_path}")
    except PermissionError:
        alt = os.path.join(dest, 'figR2.2_within_year_8020_v2.pdf')
        fig.savefig(alt, format='pdf', bbox_inches='tight')
        print(f"Saved (alt, original locked by viewer): {alt}")
plt.close()
print("\nDone.")
