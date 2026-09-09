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
from _paths import DATA_PROCESSED, OUT_FIG, need

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = need(DATA_PROCESSED, 'non_zero_df_month.csv')
OUT_FIG    = OUT_FIG
os.makedirs(OUT_FIG, exist_ok=True)

SEED         = 1
EARTH_R      = 6371.0
N_BOOTSTRAPS = 100   # set to 5000 for final submission (hours of compute)

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

# ── Pretty feature names ───────────────────────────────────────────────────────
SIGNAL_MAP = {
    'CO_MEAN':    'CO',
    'NO2_MEAN':   'NO$_2$',
    'SO2_MEAN':   'SO$_2$',
    'PM2_5_MEAN': 'PM$_{2.5}$',
    'PM10_MEAN':  'PM$_{10}$',
    'O3_MEAN':    'O$_3$',
    'LSTA_MEAN':  'LST (night)',
    'LSTT_MEAN':  'LST (day)',
    'NTL_MEAN':   'NTL',
}

def pretty(col):
    for raw, nice in SIGNAL_MAP.items():
        if col == raw:
            return nice
        if col.startswith(raw + '_lag'):
            lag = col.replace(raw + '_lag', '')
            return f'{nice} (lag {lag})'
    if col == 'month_sin':   return 'Month (sin)'
    if col == 'month_cos':   return 'Month (cos)'
    if col == 'dist_nearest_hub':  return 'Dist. to Hub'
    if col == 'dist_nearest_port': return 'Dist. to Port'
    return col

PRETTY_NAMES = [pretty(c) for c in FEATURE_COLS]

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
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)

# ── Train XGBoost (matching notebook 06) ──────────────────────────────────────
print("Training XGBoost ...")
xgb_model = xgb.XGBRegressor(
    colsample_bytree=0.6, learning_rate=0.1, max_depth=6,
    n_estimators=300, subsample=0.8, alpha=0.2, reg_lambda=0.5,
    random_state=5, verbosity=0,
)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"  Test R2={r2:.4f}  RMSE={rmse:.4f}")

# ── Figure 5(a): SHAP beeswarm summary plot ───────────────────────────────────
print("\nComputing SHAP values (TreeExplainer) ...")
explainer  = shap.TreeExplainer(xgb_model)
shap_vals  = explainer.shap_values(X_test)

plt.figure()   # let SHAP control layout, same as original notebook Cell 45
shap.summary_plot(
    shap_vals, X_test.values,
    feature_names=PRETTY_NAMES,
    show=False,
)
plt.tight_layout()

for ext in ('pdf', 'png'):
    p = os.path.join(OUT_FIG, f'shap_output_grid.{ext}')
    try:
        plt.savefig(p, bbox_inches='tight', dpi=800)
        print(f"  Saved: {p}")
    except PermissionError:
        p2 = os.path.join(OUT_FIG, f'shap_output_grid_v2.{ext}')
        plt.savefig(p2, bbox_inches='tight', dpi=800)
        print(f"  Saved (alt): {p2}")
plt.close()

# ── Figure 5(b): Bootstrap simulation SHAP bar chart ─────────────────────────
print(f"\nBootstrap SHAP (n={N_BOOTSTRAPS}) ...")

shap_bootstrap_list = []
for i in tqdm(range(N_BOOTSTRAPS), desc="Bootstrapping"):
    idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_s = X_train.iloc[idx].values
    y_s = y_train.iloc[idx].values
    m = xgb.XGBRegressor(**xgb_model.get_params())
    m.fit(X_s, y_s)
    sv = shap.TreeExplainer(m).shap_values(X_test.values)
    shap_bootstrap_list.append(sv)

shap_bootstrap_arr = np.array(shap_bootstrap_list)   # (n_boot, n_test, n_feat)

mean_shap   = np.abs(shap_bootstrap_arr).mean(axis=1).mean(axis=0)
lower_pct   = np.percentile(np.abs(shap_bootstrap_arr).mean(axis=1), q=2.5,  axis=0)
upper_pct   = np.percentile(np.abs(shap_bootstrap_arr).mean(axis=1), q=97.5, axis=0)
err_l = np.maximum(0, mean_shap - lower_pct)
err_u = np.maximum(0, upper_pct - mean_shap)

df_shap = pd.DataFrame({
    'Feature':    PRETTY_NAMES,
    'SHAP':       mean_shap,
    'SHAP_err_l': err_l,
    'SHAP_err_u': err_u,
}).sort_values('SHAP', ascending=True).tail(20)   # top 20 only, matching beeswarm

fig, ax = plt.subplots(figsize=(12, 10), dpi=160)
y_pos = np.arange(len(df_shap)) * 2   # same spacing as original Cell 46

ax.barh(
    y_pos,
    df_shap['SHAP'],
    xerr=df_shap[['SHAP_err_l', 'SHAP_err_u']].values.T,
    color='skyblue',
    capsize=3,
    edgecolor='black',
    alpha=0.8,
)
ax.set_yticks(y_pos)
ax.set_yticklabels(df_shap['Feature'], fontsize=12)
ax.set_xlabel('Mean SHAP Value (average impact on the outcome)', fontsize=14)
ax.set_ylabel('Features', fontsize=14)
ax.set_title('Global Feature Importance', fontsize=16)
ax.tick_params(axis='x', labelsize=12)
ax.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

for ext in ('pdf', 'png'):
    p = os.path.join(OUT_FIG, f'SHAP_simulation_grid.{ext}')
    try:
        fig.savefig(p, format=ext if ext != 'png' else None, bbox_inches='tight', dpi=800)
        print(f"  Saved: {p}")
    except PermissionError:
        p2 = os.path.join(OUT_FIG, f'SHAP_simulation_grid_v2.{ext}')
        fig.savefig(p2, bbox_inches='tight', dpi=800)
        print(f"  Saved (alt): {p2}")
plt.close()

print("\nDone.")
