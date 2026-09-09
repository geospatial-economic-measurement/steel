import os, sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb
import os
from _paths import DATA_PROCESSED, OUT_FIG, OUT_TAB, need

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = need(DATA_PROCESSED, 'non_zero_df_month.csv')
OUT_FIG   = OUT_FIG
OUT_TAB   = OUT_TAB
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)

SEED = 1   # matches notebook 06
EARTH_R = 6371.0

# ── China reference hubs & ports  ─────────────────────────────
CHINA_HUBS = {
    'Tangshan': (39.6, 118.2),
    'Wuhan':    (30.6, 114.3),
    'Anshan':   (41.1, 122.8),
    'Rizhao':   (35.4, 119.5),
    'Baotou':   (40.7, 109.8),
}
CHINA_PORTS = {
    'Shanghai':    (31.23, 121.47),
    'Tianjin':     (38.98, 117.72),
    'Qingdao':     (36.07, 120.38),
    'Ningbo':      (29.87, 121.55),
    'Guangzhou':   (23.10, 113.43),
    'Dalian':      (38.92, 121.65),
    'Lianyungang': (34.75, 119.45),
    'Yingkou':     (40.67, 122.23),
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

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ── Load and prepare data ──────────────────────────────────────────────────────
print("Loading data ...")
data = pd.read_csv(DATA_PATH)
data = data.drop(columns=['Unnamed: 0', 'Polygon_ID'], errors='ignore')

# Dedup guard (2026-07): GridProd_1922_monthly.csv contains byte-identical
# duplicated Dec-2022 rows; dedup before any split/evaluation.
# Source files intentionally unmodified.
data = data.drop_duplicates(subset=['IDCode', 'Year', 'Month'])

# Add month cyclical encoding
data['month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

# Add lag features (same as notebook 06)
data = data.sort_values(['IDCode', 'Year', 'Month']).reset_index(drop=True)
for lag in range(1, 4):
    for col in POLLUTANTS:
        data[f'{col}_lag{lag}'] = data.groupby('IDCode')[col].shift(lag)

data = data.dropna()

# Compute distance features from centroids then keep only the distance columns
data = add_distance_features(data)

# Build X and y (no lat/lon, no Year/Month raw, no GEM capacity)
X = data[FEATURE_COLS].copy()
y = np.log(data['GridProd_Steel_tot'])   # log (same as notebook 06)

print(f"  X shape: {X.shape}  |  Features: {len(FEATURE_COLS)}")
print(f"  Features: {FEATURE_COLS}")

# ── Train/test split + XGBoost (matching notebook 06) ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

xgb_model = xgb.XGBRegressor(
    colsample_bytree=0.6,
    learning_rate=0.1,
    max_depth=6,
    n_estimators=300,
    subsample=0.8,
    alpha=0.2,
    reg_lambda=0.5,
    random_state=5,
    verbosity=0,
)

# 5-fold CV (same as notebook 06)
print("Running 5-fold cross-validation ...")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_r2, cv_rmse = [], []

for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
    xgb_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    y_pred_fold = xgb_model.predict(X.iloc[te_idx])
    cv_r2.append(r2_score(y.iloc[te_idx], y_pred_fold))
    cv_rmse.append(calculate_rmse(y.iloc[te_idx], y_pred_fold))
    print(f"  Fold {fold+1}: R²={cv_r2[-1]:.4f}  RMSE={cv_rmse[-1]:.4f}")

print(f"\n  CV mean R²   = {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
print(f"  CV mean RMSE = {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")

# Final fit on train set, evaluate on test set (for figure)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = calculate_rmse(y_test, y_pred)
print(f"\n  Test R²={r2:.4f}  RMSE={rmse:.4f}")

# ── Save metrics ──────────────────────────────────────────────────────────────
metrics = pd.DataFrame({
    'Metric':   ['R²', 'RMSE', 'CV_R²_mean', 'CV_R²_std', 'CV_RMSE_mean', 'CV_RMSE_std'],
    'Value':    [r2, rmse, np.mean(cv_r2), np.std(cv_r2), np.mean(cv_rmse), np.std(cv_rmse)],
})
metrics.to_csv(os.path.join(OUT_TAB, 'fig4_grid_metrics.csv'), index=False)
print(f"  Saved metrics table.")

# ── Figure 4: KDE + Scatter (matching notebook 06 layout) ─────────────────────
print("Plotting Figure 4 ...")

# Tick positions in log-space mapped back to original scale
original_ticks = [0.01, 0.1, 1, 10, 100]
log_ticks = [np.log(t) for t in original_ticks]

fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

# ── Left: KDE density ─────────────────────────────────────────────────────────
ax = axes[0]
sns.kdeplot(y_test, color='blue',  fill=True, label='Actual',    ax=ax)
sns.kdeplot(y_pred, color='green', fill=True, label='Predicted', ax=ax)
ax.set_title('Density of Actual vs Predicted Values')
ax.set_xlabel('Values (10,000 tones)')
ax.set_xticks(log_ticks)
ax.set_xticklabels(original_ticks)
ax.legend()

# ── Right: Scatter ────────────────────────────────────────────────────────────
ax = axes[1]
ax.scatter(y_test, y_pred, alpha=0.5, color='purple')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_title('Actual vs Predicted Values')
ax.set_xlabel('Actual Values (10,000 tones)')
ax.set_ylabel('Predicted Values (10,000 tones)')
ax.set_xticks(log_ticks); ax.set_xticklabels(original_ticks)
ax.set_yticks(log_ticks); ax.set_yticklabels(original_ticks)

plt.tight_layout()

for ext in ('pdf', 'png'):
    fname = f'fig4_grid_actual_vs_predicted.{ext}'
    out_path = os.path.join(OUT_FIG, fname)
    try:
        fig.savefig(out_path, bbox_inches='tight')
        print(f"Saved: {out_path}")
    except PermissionError:
        alt_path = os.path.join(OUT_FIG, f'fig4_grid_actual_vs_predicted_v2.{ext}')
        fig.savefig(alt_path, bbox_inches='tight')
        print(f"Saved (alt): {alt_path}")

plt.close()
print("Done.")
