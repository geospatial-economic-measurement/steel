import os, sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
from tqdm import tqdm
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_FIG, need

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Paths ──────────────────────────────────────────────────────────────────────
SIGNALS_PATH  = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH      = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')
OUT_FIG       = OUT_FIG
os.makedirs(OUT_FIG, exist_ok=True)

SEED         = 1
EARTH_R      = 6371.0
N_BOOTSTRAPS = 100   # set to 5000 for final submission

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

SIGNAL_MAP = {
    'Plant_CO_MEAN':    'CO',       'Plant_NO2_MEAN':   'NO$_2$',
    'Plant_SO2_MEAN':   'SO$_2$',   'Plant_PM2_5_MEAN': 'PM$_{2.5}$',
    'Plant_PM10_MEAN':  'PM$_{10}$','Plant_O3_MEAN':    'O$_3$',
    'Plant_LSTA_MEAN':  'LST (night)', 'Plant_LSTT_MEAN': 'LST (day)',
    'Plant_NTL_MEAN':   'NTL',
}

def pretty(col):
    for raw, nice in SIGNAL_MAP.items():
        if col == raw:           return nice
        if col.startswith(raw + '_lag'):
            return f'{nice} (lag {col.replace(raw + "_lag", "")})'
    if col == 'month_sin':          return 'Month (sin)'
    if col == 'month_cos':          return 'Month (cos)'
    if col == 'dist_nearest_hub':   return 'Dist. to Hub'
    if col == 'dist_nearest_port':  return 'Dist. to Port'
    if col == 'is_bof':             return 'BOF technology'
    return col

PRETTY_NAMES = [pretty(c) for c in FEATURE_COLS]

# ── Helpers ────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((r2 - r1) / 2)**2 +
         np.cos(r1) * np.cos(r2) * np.sin(np.deg2rad(lon2 - lon1) / 2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def add_distance_features(df):
    lats, lons = df['Latitude'].values, df['Longitude'].values
    hub_d  = np.stack([haversine_km(lats, lons, la, lo)
                       for la, lo in CHINA_HUBS.values()], axis=1)
    port_d = np.stack([haversine_km(lats, lons, la, lo)
                       for la, lo in CHINA_PORTS.values()], axis=1)
    df = df.copy()
    df['dist_nearest_hub']  = hub_d.min(axis=1)
    df['dist_nearest_port'] = port_d.min(axis=1)
    return df

# ── Load and prepare data ──────────────────────────────────────────────────────
print("Loading data ...")
signals  = pd.read_csv(SIGNALS_PATH)
signals  = signals.drop(columns=['PM1_MEAN'], errors='ignore')
gridprod = pd.read_csv(GRIDPROD_PATH)

# ── GEM technology lookup: is_bof = 1 if plant has a blast furnace ────────────
bf_sheet   = pd.read_excel(GEM_PATH, sheet_name='Blast furnaces', header=0)
bf_gem_ids = set(bf_sheet.iloc[:, 0].dropna().astype(str).unique())
gem_tech   = (gridprod[['name_prod', 'GEMPlantID']]
              .drop_duplicates('name_prod').copy())
gem_tech['is_bof'] = gem_tech['GEMPlantID'].apply(
    lambda x: 1 if str(x) in bf_gem_ids else 0)
print(f"  BOF plants: {gem_tech['is_bof'].sum()} / {len(gem_tech)} total")

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

grouped = add_distance_features(grouped)

data = grouped.dropna(subset=FEATURE_COLS + ['Steel_Prod']).copy()
data = data[data['Steel_Prod'] > 0]
print(f"  Rows: {len(data)}, Plants: {data['name_prod'].nunique()}")

X = data[FEATURE_COLS].copy()
y = np.log1p(data['Steel_Prod'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)

# ── Train XGBoost ──────────────────────────────────────────────────────────────
print("Training XGBoost ...")
xgb_model = xgb.XGBRegressor(
    colsample_bytree=0.8, learning_rate=0.1, max_depth=6,
    n_estimators=300, subsample=0.6,
    alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0,
)
xgb_model.fit(X_train, y_train)

# ── Bootstrap SHAP ─────────────────────────────────────────────────────────────
print(f"\nBootstrap SHAP (n={N_BOOTSTRAPS}) ...")
shap_bootstrap_list = []
for _ in tqdm(range(N_BOOTSTRAPS), desc="Bootstrapping"):
    idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    m = xgb.XGBRegressor(**xgb_model.get_params())
    m.fit(X_train.iloc[idx].values, y_train.iloc[idx].values)
    sv = shap.TreeExplainer(m).shap_values(X_test.values)
    shap_bootstrap_list.append(sv)

shap_bootstrap_arr = np.array(shap_bootstrap_list)   # (n_boot, n_test, n_feat)

# ── Select top 9 features by mean |SHAP| ──────────────────────────────────────
mean_abs_shap = np.abs(shap_bootstrap_arr).mean(axis=(0, 1))
top9_idx = np.argsort(mean_abs_shap)[::-1][:9]
print(f"\nTop 9 features: {[PRETTY_NAMES[i] for i in top9_idx]}")

# ── Compute per-observation 95% CI ────────────────────────────────────────────
l_95 = np.percentile(shap_bootstrap_arr, q=2.5,  axis=0)  # (n_test, n_feat)
u_95 = np.percentile(shap_bootstrap_arr, q=97.5, axis=0)
mean_shap_obs = shap_bootstrap_arr.mean(axis=0)            # (n_test, n_feat)

# ── Figure: 3x3 PDP-style local SHAP plots ─────────────
print("\nPlotting PDP figure ...")
fig, axes = plt.subplots(3, 3, figsize=(12, 8), dpi=300)
axes = axes.ravel()

for panel_idx, feat_idx in enumerate(top9_idx):
    ax = axes[panel_idx]
    feat_vals = X_test.values[:, feat_idx]
    order = np.argsort(feat_vals)

    ax.fill_between(
        feat_vals[order],
        l_95[order, feat_idx],
        u_95[order, feat_idx],
        color='lightblue', alpha=0.6,
    )
    ax.scatter(
        feat_vals[order],
        mean_shap_obs[order, feat_idx],
        s=10, color='black',
    )
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel(PRETTY_NAMES[feat_idx], fontsize=13)
    ax.set_ylabel('SHAP value', fontsize=13)

plt.tight_layout()

for ext in ('pdf', 'png'):
    p = os.path.join(OUT_FIG, f'PDP_plant.{ext}')
    try:
        fig.savefig(p, bbox_inches='tight')
        print(f"  Saved: {p}")
    except PermissionError:
        p2 = os.path.join(OUT_FIG, f'PDP_plant_v2.{ext}')
        fig.savefig(p2, bbox_inches='tight')
        print(f"  Saved (alt): {p2}")
plt.close()
print("Done.")
