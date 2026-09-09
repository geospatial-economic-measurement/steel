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
from _paths import DATA_PROCESSED, OUT_TAB, need

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = need(DATA_PROCESSED, 'non_zero_df_month.csv')
OUT_TAB   = OUT_TAB
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

POLLUTANTS = ['CO_MEAN', 'NO2_MEAN', 'SO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN',
              'O3_MEAN', 'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN']

FEATURE_COLS = (
    POLLUTANTS +
    [f'{p}_lag{k}' for k in (1, 2, 3) for p in POLLUTANTS] +
    ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port']
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((r2 - r1) / 2)**2 +
         np.cos(r1) * np.cos(r2) * np.sin(np.deg2rad(lon2 - lon1) / 2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def add_distance_features(df):
    lats, lons = df['Centroid_Lat'].values, df['Centroid_Long'].values
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

# ── Load and prepare data ──────────────────────────────────────────────────────
print("Loading data ...")
data = pd.read_csv(DATA_PATH)
data = data.drop(columns=['Unnamed: 0', 'Polygon_ID'], errors='ignore')

# Dedup guard (2026-07): GridProd_1922_monthly.csv contains byte-identical
# duplicated Dec-2022 rows; dedup before any split/evaluation.
# Source files intentionally unmodified.
data = data.drop_duplicates(subset=['IDCode', 'Year', 'Month'])

data['month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

data = data.sort_values(['IDCode', 'Year', 'Month']).reset_index(drop=True)
for lag in range(1, 4):
    for col in POLLUTANTS:
        data[f'{col}_lag{lag}'] = data.groupby('IDCode')[col].shift(lag)
data = data.dropna()
data = add_distance_features(data)

X = data[FEATURE_COLS].copy()
y = np.log(data['GridProd_Steel_tot'])
print(f"  X shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

# ── Model definitions (matching notebook 06 hyperparameters) ──────────────────
scaler = StandardScaler()

base_estimators = [
    ('lasso',        Lasso(alpha=0.01, random_state=SEED)),
    ('lightgbm',     lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                        learning_rate=0.05, n_estimators=720,
                                        verbose=-1)),
    ('random_forest', RandomForestRegressor(max_depth=4, max_features=9,
                                            n_estimators=300, random_state=5)),
    ('xgboost',      xgb.XGBRegressor(colsample_bytree=0.6, learning_rate=0.1,
                                       max_depth=6, n_estimators=300, subsample=0.8,
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
    'XGBoost':           xgb.XGBRegressor(colsample_bytree=0.6, learning_rate=0.1,
                                           max_depth=6, n_estimators=300, subsample=0.8,
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

out_path = os.path.join(OUT_TAB, 'table_a7_grid_metrics.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
