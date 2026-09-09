import os, sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_FIG, need

warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Paths ──────────────────────────────────────────────────────────────────────
GRID_DATA_PATH = need(DATA_PROCESSED, 'non_zero_df_month.csv')
SIGNALS_PATH   = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH  = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH       = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')

OUT_FIG   = OUT_FIG
PAPER_FIG = OUT_FIG

os.makedirs(OUT_FIG,   exist_ok=True)
os.makedirs(PAPER_FIG, exist_ok=True)

SEED    = 1
EARTH_R = 6371.0

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

POLLUTANTS = ['CO_MEAN', 'NO2_MEAN', 'SO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN',
              'O3_MEAN', 'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN']

# ── Helpers ────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((r2 - r1) / 2)**2 +
         np.cos(r1) * np.cos(r2) * np.sin(np.deg2rad(lon2 - lon1) / 2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def add_distance_features(df, lat_col='Centroid_Lat', lon_col='Centroid_Long'):
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

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def make_models_grid():
    """7 models for grid-level CV (hyperparameters from script 15)."""
    return {
        'Linear\nRegression': make_pipeline(StandardScaler(), LinearRegression()),
        'Lasso':              make_pipeline(StandardScaler(), Lasso(alpha=0.01, random_state=SEED)),
        'ElasticNet':         make_pipeline(StandardScaler(), ElasticNet(random_state=SEED)),
        'Gradient\nBoosting': GradientBoostingRegressor(random_state=SEED),
        'LightGBM':           lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                                 learning_rate=0.05, n_estimators=720,
                                                 verbose=-1),
        'Random\nForest':     RandomForestRegressor(max_depth=4, max_features=9,
                                                    n_estimators=300, random_state=5),
        'XGBoost':            xgb.XGBRegressor(colsample_bytree=0.6, learning_rate=0.1,
                                                max_depth=6, n_estimators=300, subsample=0.8,
                                                alpha=0.2, reg_lambda=0.5, random_state=5,
                                                verbosity=0),
    }

def make_models_plant():
    """7 models for plant-level CV (hyperparameters from script 20)."""
    return {
        'Linear\nRegression': make_pipeline(StandardScaler(), LinearRegression()),
        'Lasso':              make_pipeline(StandardScaler(), Lasso(alpha=0.01, random_state=SEED)),
        'ElasticNet':         make_pipeline(StandardScaler(), ElasticNet(random_state=SEED)),
        'Gradient\nBoosting': GradientBoostingRegressor(random_state=SEED),
        'LightGBM':           lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                                 learning_rate=0.05, n_estimators=720,
                                                 verbose=-1),
        'Random\nForest':     RandomForestRegressor(max_depth=4, max_features=9,
                                                    n_estimators=300, random_state=5),
        'XGBoost':            xgb.XGBRegressor(colsample_bytree=0.8, learning_rate=0.1,
                                                max_depth=6, n_estimators=300, subsample=0.6,
                                                alpha=0.2, reg_lambda=0.5, random_state=5,
                                                verbosity=0),
    }

def run_cv(models, X, y, kf):
    rmse_scorer_fn = make_scorer(rmse_scorer, greater_is_better=False)
    rmse_results = {}
    r2_results   = {}
    for name, model in models.items():
        short = name.replace('\n', ' ')
        print(f"  {short} ...", end=' ', flush=True)
        r2_scores   = cross_val_score(model, X, y, cv=kf, scoring='r2')
        rmse_scores = -cross_val_score(model, X, y, cv=kf, scoring=rmse_scorer_fn)
        r2_results[name]   = r2_scores
        rmse_results[name] = rmse_scores
        print(f"R2={r2_scores.mean():.4f}  RMSE={rmse_scores.mean():.4f}")
    return rmse_results, r2_results

def save_boxplot(results_dict, ylabel, title, outname, color):
    names  = list(results_dict.keys())
    values = [results_dict[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(values, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.0),
                    capprops=dict(linewidth=1.0),
                    flierprops=dict(marker='o', markersize=3,
                                    markerfacecolor=color, alpha=0.5))
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, pad=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for dest in [OUT_FIG, PAPER_FIG]:
        out_path = os.path.join(dest, outname)
        fig.savefig(out_path, bbox_inches='tight', dpi=200)
        print(f"  Saved: {out_path}")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Grid-level (40 features)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 1: Grid-level 5-fold CV (40 features)")
print("="*60)

GRID_FEATURES = (
    POLLUTANTS +
    [f'{p}_lag{k}' for k in (1, 2, 3) for p in POLLUTANTS] +
    ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port']
)

print("Loading grid data ...")
data = pd.read_csv(GRID_DATA_PATH)
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
data = add_distance_features(data, lat_col='Centroid_Lat', lon_col='Centroid_Long')

X_grid = data[GRID_FEATURES].copy()
y_grid = np.log(data['GridProd_Steel_tot'])
print(f"  X shape: {X_grid.shape}")

kf = KFold(n_splits=5, shuffle=True, random_state=1)

print("\nRunning 5-fold CV for grid models ...")
grid_models = make_models_grid()
grid_rmse, grid_r2 = run_cv(grid_models, X_grid, y_grid, kf)

print("\nSaving grid boxplots ...")
save_boxplot(grid_rmse,
             ylabel='RMSE (log tons)',
             title='Grid-Level Output Model: RMSE (5-Fold CV)',
             outname='model_rmse_performance_grid_filtered.pdf',
             color='#4C72B0')
save_boxplot(grid_r2,
             ylabel='R²',
             title='Grid-Level Output Model: R² (5-Fold CV)',
             outname='model_r2_performance_grid_filtered.pdf',
             color='#4C72B0')

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Plant-level (41 features)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 2: Plant-level 5-fold CV (41 features)")
print("="*60)

PLANT_POLLUTANTS = [f'Plant_{p}' for p in POLLUTANTS]
PLANT_FEATURES   = (
    PLANT_POLLUTANTS +
    [f'{p}_lag{k}' for k in (1, 2, 3) for p in PLANT_POLLUTANTS] +
    ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port', 'is_bof']
)

print("Loading plant data ...")
signals  = pd.read_csv(SIGNALS_PATH)
signals  = signals.drop(columns=['PM1_MEAN'], errors='ignore')
gridprod = pd.read_csv(GRIDPROD_PATH)

# Dedup guard (2026-07): GridProd_1922_monthly.csv contains byte-identical
# duplicated Dec-2022 rows; dedup before any split/evaluation.
# Source files intentionally unmodified.
gridprod = gridprod.drop_duplicates(subset=['IDCode', 'name_prod', 'Year', 'Month'])

bf_sheet   = pd.read_excel(GEM_PATH, sheet_name='Blast furnaces', header=0)
bf_gem_ids = set(bf_sheet.iloc[:, 0].dropna().astype(str).unique())
gem_tech   = (gridprod[['name_prod', 'GEMPlantID']]
              .drop_duplicates('name_prod').copy())
gem_tech['is_bof'] = gem_tech['GEMPlantID'].apply(
    lambda x: 1 if str(x) in bf_gem_ids else 0)
print(f"  BOF plants: {gem_tech['is_bof'].sum()} / {len(gem_tech)} total")

temp   = gridprod[['IDCode', 'Longitude', 'Latitude', 'name_prod',
                   'Year', 'Month', 'Steel_Prod', 'Iron_Prod']].copy()
merged = pd.merge(signals, temp, on=['IDCode', 'Year', 'Month'], how='left')

agg_signals = {col: 'mean' for col in POLLUTANTS}
grouped = (merged
           .groupby(['name_prod', 'Year', 'Month'])
           .agg({**agg_signals,
                 'Longitude': 'mean', 'Latitude': 'mean',
                 'Steel_Prod': 'first', 'Iron_Prod': 'first'})
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

plant_data = grouped.dropna(subset=PLANT_FEATURES + ['Steel_Prod']).copy()
plant_data = plant_data[plant_data['Steel_Prod'] > 0]
print(f"  After dropna: {len(plant_data)} rows, {len(PLANT_FEATURES)} features")

X_plant = plant_data[PLANT_FEATURES].copy()
y_plant = np.log1p(plant_data['Steel_Prod'])

print("\nRunning 5-fold CV for plant models ...")
plant_models = make_models_plant()
plant_rmse, plant_r2 = run_cv(plant_models, X_plant, y_plant, kf)

print("\nSaving plant boxplots ...")
save_boxplot(plant_rmse,
             ylabel='RMSE (log tons)',
             title='Plant-Level Output Model: RMSE (5-Fold CV)',
             outname='model_rmse_performance_v2.pdf',
             color='#DD8452')
save_boxplot(plant_r2,
             ylabel='R²',
             title='Plant-Level Output Model: R² (5-Fold CV)',
             outname='model_r2_performance_v2.pdf',
             color='#DD8452')

print("\nAll 4 figure files saved successfully.")
