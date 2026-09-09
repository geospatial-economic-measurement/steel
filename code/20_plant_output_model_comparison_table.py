import os, sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                               StackingRegressor)
import lightgbm as lgb
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_TAB, need

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
SIGNALS_PATH  = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH      = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')
OUT_TAB       = OUT_TAB
os.makedirs(OUT_TAB, exist_ok=True)

SEED    = 1
EARTH_R = 6371.0

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

FEATURE_COLS = (
    PLANT_POLLUTANTS +
    [f'{p}_lag{k}' for k in (1, 2, 3) for p in PLANT_POLLUTANTS] +
    ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port', 'is_bof']
)

def haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((r2 - r1) / 2)**2 +
         np.cos(r1) * np.cos(r2) * np.sin(np.deg2rad(lon2 - lon1) / 2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def add_distance_features(df, lat_col='Latitude', lon_col='Longitude'):
    lats = df[lat_col].values
    lons = df[lon_col].values
    hub_d  = np.stack([haversine_km(lats, lons, la, lo)
                       for la, lo in CHINA_HUBS.values()], axis=1)
    port_d = np.stack([haversine_km(lats, lons, la, lo)
                       for la, lo in CHINA_PORTS.values()], axis=1)
    df = df.copy()
    df['dist_nearest_hub']  = hub_d.min(axis=1)
    df['dist_nearest_port'] = port_d.min(axis=1)
    return df

def calculate_wmape(y_true, y_pred):
    s = np.sum(np.abs(y_true))
    return np.sum(np.abs(y_true - y_pred)) / s if s != 0 else np.nan

print("Loading data ...")
signals  = pd.read_csv(SIGNALS_PATH)
signals  = signals.drop(columns=['PM1_MEAN'], errors='ignore')
gridprod = pd.read_csv(GRIDPROD_PATH)

# GEM technology lookup: is_bof = 1 if plant has a blast furnace
bf_sheet   = pd.read_excel(GEM_PATH, sheet_name='Blast furnaces', header=0)
bf_gem_ids = set(bf_sheet.iloc[:, 0].dropna().astype(str).unique())
gem_tech   = (gridprod[['name_prod', 'GEMPlantID']]
              .drop_duplicates('name_prod').copy())
gem_tech['is_bof'] = gem_tech['GEMPlantID'].apply(
    lambda x: 1 if str(x) in bf_gem_ids else 0)
print(f"  BOF plants: {gem_tech['is_bof'].sum()} / {len(gem_tech)} total")

# Merge satellite signals with production data
temp = gridprod[['IDCode', 'Longitude', 'Latitude', 'name_prod',
                 'Year', 'Month', 'Steel_Prod', 'Iron_Prod']].copy()
merged = pd.merge(signals, temp, on=['IDCode', 'Year', 'Month'], how='left')

# Aggregate to plant level
agg_signals = {col: 'mean' for col in POLLUTANTS}
grouped = (merged
           .groupby(['name_prod', 'Year', 'Month'])
           .agg({**agg_signals,
                 'Longitude': 'mean',
                 'Latitude':  'mean',
                 'Steel_Prod': 'first',
                 'Iron_Prod':  'first'})
           .reset_index())

grouped.rename(columns={p: f'Plant_{p}' for p in POLLUTANTS}, inplace=True)
grouped = grouped.drop_duplicates(subset=['name_prod', 'Year', 'Month'])
grouped['plant_id'] = grouped['name_prod'].factorize()[0] + 1
grouped = grouped.merge(gem_tech[['name_prod', 'is_bof']], on='name_prod', how='left')
grouped['is_bof'] = grouped['is_bof'].fillna(0).astype(int)

print(f"  Plants: {grouped['name_prod'].nunique()}, rows: {len(grouped)}")

grouped['month_sin'] = np.sin(2 * np.pi * grouped['Month'] / 12)
grouped['month_cos'] = np.cos(2 * np.pi * grouped['Month'] / 12)

grouped = grouped.sort_values(['plant_id', 'Year', 'Month']).reset_index(drop=True)
for lag in range(1, 4):
    for col in PLANT_POLLUTANTS:
        grouped[f'{col}_lag{lag}'] = grouped.groupby('plant_id')[col].shift(lag)

grouped = add_distance_features(grouped, lat_col='Latitude', lon_col='Longitude')

data = grouped.dropna(subset=FEATURE_COLS + ['Steel_Prod']).copy()
data = data[data['Steel_Prod'] > 0]
print(f"  After dropna: {len(data)} rows, {len(FEATURE_COLS)} features")

X = data[FEATURE_COLS].copy()
y = np.log1p(data['Steel_Prod'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)

# ── Model definitions (plant-level hyperparameters) ────────────────────────────
base_estimators = [
    ('lasso',        Lasso(alpha=0.01, random_state=SEED)),
    ('lightgbm',     lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                        learning_rate=0.05, n_estimators=720,
                                        verbose=-1)),
    ('random_forest', RandomForestRegressor(max_depth=4, max_features=9,
                                            n_estimators=300, random_state=5)),
    ('xgboost',      xgb.XGBRegressor(colsample_bytree=0.8, learning_rate=0.1,
                                       max_depth=6, n_estimators=300, subsample=0.6,
                                       alpha=0.2, reg_lambda=0.5, random_state=5,
                                       verbosity=0)),
]

models = {
    'Linear Regression': make_pipeline(StandardScaler(), LinearRegression()),
    'Lasso':             make_pipeline(StandardScaler(), Lasso(alpha=0.01, random_state=SEED)),
    'ElasticNet':        make_pipeline(StandardScaler(), ElasticNet(random_state=SEED)),
    'Gradient Boosting': GradientBoostingRegressor(random_state=SEED),
    'LightGBM':          lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                            learning_rate=0.05, n_estimators=720,
                                            verbose=-1),
    'Random Forest':     RandomForestRegressor(max_depth=4, max_features=9,
                                               n_estimators=300, random_state=5),
    'XGBoost':           xgb.XGBRegressor(colsample_bytree=0.8, learning_rate=0.1,
                                           max_depth=6, n_estimators=300, subsample=0.6,
                                           alpha=0.2, reg_lambda=0.5, random_state=5,
                                           verbosity=0),
    'Ensemble Learning': make_pipeline(
        StandardScaler(),
        StackingRegressor(estimators=base_estimators,
                          final_estimator=LinearRegression(), cv=5)
    ),
}

# ── Fit all models and compute metrics ────────────────────────────────────────
print("\nFitting models ...")
results = []
for name, model in models.items():
    print(f"  {name} ...", end=' ', flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse   = mean_squared_error(y_test, y_pred)
    rmse  = np.sqrt(mse)
    wmape = calculate_wmape(y_test.values, y_pred)
    r2    = r2_score(y_test, y_pred)
    results.append({'Model': name, 'MSE': mse, 'RMSE': rmse,
                    'WMAPE': wmape, 'R2 Score': r2})
    print(f"R2={r2:.4f}  RMSE={rmse:.4f}")

df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))

out_path = os.path.join(OUT_TAB, 'table_a7_plant_metrics.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
