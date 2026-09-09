
import os, sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, OUT_TAB, need

warnings.filterwarnings('ignore')

GRID_DATA_PATH = need(DATA_PROCESSED, 'non_zero_df_month.csv')
GRIDPROD_PATH  = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
OUT_TAB        = OUT_TAB
os.makedirs(OUT_TAB, exist_ok=True)

EARTH_R = 6371.0
CHINA_HUBS = {'Tangshan': (39.6, 118.2), 'Wuhan': (30.6, 114.3), 'Anshan': (41.1, 122.8),
              'Rizhao': (35.4, 119.5), 'Baotou': (40.7, 109.8)}
CHINA_PORTS = {'Shanghai': (31.23, 121.47), 'Tianjin': (38.98, 117.72),
               'Qingdao': (36.07, 120.38), 'Ningbo': (29.87, 121.55),
               'Guangzhou': (23.10, 113.43), 'Dalian': (38.92, 121.65),
               'Lianyungang': (34.75, 119.45), 'Yingkou': (40.67, 122.23)}
POLLUTANTS = ['CO_MEAN', 'NO2_MEAN', 'SO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN',
              'O3_MEAN', 'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN']
GRID_FEATURES = (POLLUTANTS + [f'{p}_lag{k}' for k in (1, 2, 3) for p in POLLUTANTS] +
                 ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port'])

def haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((r2 - r1) / 2)**2 + np.cos(r1) * np.cos(r2) * np.sin(np.deg2rad(lon2 - lon1) / 2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def add_dist(df, la, lo):
    lats, lons = df[la].values, df[lo].values
    hub  = np.stack([haversine_km(lats, lons, a, b) for a, b in CHINA_HUBS.values()], axis=1)
    port = np.stack([haversine_km(lats, lons, a, b) for a, b in CHINA_PORTS.values()], axis=1)
    df = df.copy(); df['dist_nearest_hub'] = hub.min(1); df['dist_nearest_port'] = port.min(1)
    return df

def cs_r2(df, a, p, u):
    m = df.groupby(u)[[a, p]].mean()
    return r2_score(m[a], m[p])

# ── Grid dataset (verbatim from script 45) ────────────────────────────────────
print("Loading grid data (confirmed spec) ...")
g = pd.read_csv(GRID_DATA_PATH).drop(columns=['Unnamed: 0', 'Polygon_ID'], errors='ignore')
g = g.drop_duplicates(subset=['IDCode', 'Year', 'Month'])
g['month_sin'] = np.sin(2 * np.pi * g['Month'] / 12)
g['month_cos'] = np.cos(2 * np.pi * g['Month'] / 12)
g = g.sort_values(['IDCode', 'Year', 'Month']).reset_index(drop=True)
for lag in (1, 2, 3):
    for col in POLLUTANTS:
        g[f'{col}_lag{lag}'] = g.groupby('IDCode')[col].shift(lag)
g = add_dist(g, 'Centroid_Lat', 'Centroid_Long')
g = g.dropna(subset=GRID_FEATURES + ['GridProd_Steel_tot', 'Year'])
g = g[g['GridProd_Steel_tot'] > 0].reset_index(drop=True)
g['y'] = np.log(g['GridProd_Steel_tot'].values)
print(f"  Grid rows: {len(g)}  unique grids: {g['IDCode'].nunique()}")

# ── Plant map (name_prod, Steel_Prod) from GridProd_1922 ──────────────────────
gp = pd.read_csv(GRIDPROD_PATH).drop_duplicates(subset=['IDCode', 'name_prod', 'Year', 'Month'])
pmap = gp[['IDCode', 'Year', 'Month', 'name_prod', 'Steel_Prod']].dropna(subset=['name_prod', 'Steel_Prod'])

# ── Standard 5-fold OOF (confirmed XGB params) ────────────────────────────────
PARAMS = dict(colsample_bytree=0.6, learning_rate=0.1, max_depth=6, n_estimators=300,
              subsample=0.8, alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0)
X = g[GRID_FEATURES].values
y = g['y'].values
oof = np.full(len(g), np.nan)
kf = KFold(n_splits=5, shuffle=True, random_state=5)
for tr, te in kf.split(X):
    m = xgb.XGBRegressor(**PARAMS); m.fit(X[tr], y[tr])
    oof[te] = m.predict(X[te])
g['pred_log'] = oof
g['pred_kt']  = np.exp(g['pred_log'])
g['true_kt']  = g['GridProd_Steel_tot'].values

# ── Grid-level metrics (level AND log scale) ──────────────────────────────────
grid_overall = r2_score(g['true_kt'], g['pred_kt'])
grid_cs      = cs_r2(g, 'true_kt', 'pred_kt', 'IDCode')
grid_overall_log = r2_score(g['y'], g['pred_log'])
grid_cs_log      = cs_r2(g, 'y', 'pred_log', 'IDCode')
print(f"\nGrid  (individual): LEVEL overall {grid_overall:.4f} CS {grid_cs:.4f}  |  "
      f"LOG overall {grid_overall_log:.4f} CS {grid_cs_log:.4f}")

# ── Aggregate grid predictions -> plant level ─────────────────────────────────
gp_join = g[['IDCode', 'Year', 'Month', 'pred_kt']].merge(pmap, on=['IDCode', 'Year', 'Month'], how='inner')
plant = gp_join.groupby(['name_prod', 'Year', 'Month']).agg(
    actual=('Steel_Prod', 'first'), pred=('pred_kt', 'sum'), n_grids=('IDCode', 'count')).reset_index()
plant_overall = r2_score(plant['actual'], plant['pred'])
plant_cs      = cs_r2(plant, 'actual', 'pred', 'name_prod')
# log-scale plant metrics (log of aggregated levels)
plant['actual_log'] = np.log(plant['actual'].clip(lower=1e-9))
plant['pred_log']   = np.log(plant['pred'].clip(lower=1e-9))
plant_overall_log = r2_score(plant['actual_log'], plant['pred_log'])
plant_cs_log      = cs_r2(plant, 'actual_log', 'pred_log', 'name_prod')
print(f"Plant (sum grids) : LEVEL overall {plant_overall:.4f} CS {plant_cs:.4f}  |  "
      f"LOG overall {plant_overall_log:.4f} CS {plant_cs_log:.4f}")
print(f"  plant-months: {len(plant)}  plants: {plant['name_prod'].nunique()}  "
      f"avg grids/plant: {plant['n_grids'].mean():.1f}  grid rows mapped: {len(gp_join)}")

# ── Save + compare ────────────────────────────────────────────────────────────
out = pd.DataFrame([
    {'Level': 'Grid (individual)',    'Overall_R2': round(grid_overall, 4),  'CS_R2': round(grid_cs, 4),  'N': len(g)},
    {'Level': 'Plant (sum of grids)', 'Overall_R2': round(plant_overall, 4), 'CS_R2': round(plant_cs, 4), 'N': len(plant)},
])
path = os.path.join(OUT_TAB, 'tableR2.5_grid_to_plant_5fold_confirmed.csv')
out.to_csv(path, index=False)
print(f"\nSaved: {path}\n{out.to_string(index=False)}")
print("\nReference (OLD exhibit, LOYO + all-columns spec): Grid CS 0.898 -> Plant CS 0.920")
print(f"Error-cancellation holds (plant CS > grid CS): {plant_cs > grid_cs}")
