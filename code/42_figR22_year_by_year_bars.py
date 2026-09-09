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
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_FIG, OUT_RL, OUT_TAB, need

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Paths (identical to scripts 25/26) ────────────────────────────────────────
GRID_DATA_PATH = need(DATA_PROCESSED, 'non_zero_df_month.csv')
SIGNALS_PATH   = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH  = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH       = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')
OUT_FIG_LOCAL  = OUT_RL
OUT_RL         = OUT_RL
OUT_FIG        = OUT_FIG
OUT_TAB        = OUT_TAB
for d in (OUT_FIG_LOCAL, OUT_RL, OUT_FIG, OUT_TAB):
    os.makedirs(d, exist_ok=True)

EARTH_R       = 6371.0
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

# ── Helpers (verbatim from scripts 25/26) ─────────────────────────────────────
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

def loyocv_oof(X_all, y_all, years_all, units_all, params):
    """LOYOCV; return long DataFrame of OOF predictions (unit, year, y_true, y_pred)."""
    recs = []
    for yr in HOLDOUT_YEARS:
        train_mask = years_all != yr
        test_mask  = years_all == yr
        m = xgb.XGBRegressor(**params)
        m.fit(X_all[train_mask], y_all[train_mask])
        y_hat = m.predict(X_all[test_mask])
        recs.append(pd.DataFrame({'unit': units_all[test_mask],
                                  'year': yr,
                                  'y_true': y_all[test_mask],
                                  'y_pred': y_hat}))
        print(f"    {yr}: R² = {r2_score(y_all[test_mask], y_hat):.4f}  "
              f"(n_test={int(test_mask.sum())})")
    return pd.concat(recs, ignore_index=True)

def decompose_r2(df_pred):
    """Overall / cross-sectional / temporal R² (method of script 26)."""
    overall = r2_score(df_pred['y_true'], df_pred['y_pred'])
    means = df_pred.groupby('unit')[['y_true', 'y_pred']].mean()
    cs = r2_score(means['y_true'], means['y_pred'])
    df2 = df_pred.join(means.rename(columns={'y_true': 'mean_true',
                                             'y_pred': 'mean_pred'}), on='unit')
    temp = r2_score(df2['y_true'] - df2['mean_true'],
                    df2['y_pred'] - df2['mean_pred'])
    return overall, cs, temp

# ── Build grid dataset (verbatim from script 25) ──────────────────────────────
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

X_grid    = grid[GRID_FEATURES].values
y_grid    = np.log(grid['GridProd_Steel_tot'].values)
yr_grid   = grid['Year'].values
unit_grid = grid['IDCode'].values
print(f"  Grid rows: {len(grid)}, features: {len(GRID_FEATURES)}")

# ── Build plant dataset (verbatim from script 25) ─────────────────────────────
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

X_plant    = plant_data[PLANT_FEATURES].values
y_plant    = np.log1p(plant_data['Steel_Prod'].values)
yr_plant   = plant_data['Year'].values
unit_plant = plant_data['name_prod'].values
print(f"  Plant rows: {len(plant_data)}, features: {len(PLANT_FEATURES)}")

# ── XGBoost hyperparameters (identical to scripts 25/26) ──────────────────────
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

# ── Run LOYOCV ────────────────────────────────────────────────────────────────
print("\nLOYOCV — Grid level:")
oof_grid = loyocv_oof(X_grid, y_grid, yr_grid, unit_grid, GRID_PARAMS)
print("\nLOYOCV — Plant level:")
oof_plant = loyocv_oof(X_plant, y_plant, yr_plant, unit_plant, PLANT_PARAMS)

# ── Metrics ───────────────────────────────────────────────────────────────────
def year_metrics(oof):
    r2s, rmses, ns = [], [], []
    for yr in HOLDOUT_YEARS:
        sub = oof[oof['year'] == yr]
        r2s.append(r2_score(sub['y_true'], sub['y_pred']))
        rmses.append(np.sqrt(mean_squared_error(sub['y_true'], sub['y_pred'])))
        ns.append(len(sub))
    return r2s, rmses, ns

grid_r2, grid_rmse, grid_n    = year_metrics(oof_grid)
plant_r2, plant_rmse, plant_n = year_metrics(oof_plant)
g_overall, g_cs, g_temp = decompose_r2(oof_grid)
p_overall, p_cs, p_temp = decompose_r2(oof_plant)

# ── Verification against confirmed letter/appendix anchors ───────────────────
EXPECTED = {
    'grid_r2':  [0.868, 0.875, 0.830, 0.795],
    'plant_r2': [0.853, 0.820, 0.759, 0.753],
    'grid_cs': 0.897,  'grid_temp': -1.142,
    'plant_cs': 0.922, 'plant_temp': -0.580,
}
print("\n── Verification vs letter tab:byyear_r2 (3-dp rounding) ──")
ok = True
for name, got, exp in [('Grid  R²', grid_r2,  EXPECTED['grid_r2']),
                       ('Plant R²', plant_r2, EXPECTED['plant_r2'])]:
    for yr, g, e in zip(HOLDOUT_YEARS, got, exp):
        match = abs(round(g, 3) - e) < 1e-9
        ok &= match
        print(f"  {name} {yr}: computed {g:.4f} -> {round(g,3):.3f}  "
              f"expected {e:.3f}  {'MATCH' if match else 'MISMATCH'}")
for name, g, e in [('Grid CS', g_cs, EXPECTED['grid_cs']),
                   ('Grid temporal', g_temp, EXPECTED['grid_temp']),
                   ('Plant CS', p_cs, EXPECTED['plant_cs']),
                   ('Plant temporal', p_temp, EXPECTED['plant_temp'])]:
    match = abs(round(g, 3) - e) < 1e-9
    ok &= match
    print(f"  {name}: computed {g:.4f} -> {round(g,3):.3f}  "
          f"expected {e:.3f}  {'MATCH' if match else 'MISMATCH'}")
print(f"  OVERALL: {'PASS' if ok else 'FAIL'}")

# ── Backing CSV ───────────────────────────────────────────────────────────────
rows = []
for lvl, r2s, rmses, ns in [('Grid', grid_r2, grid_rmse, grid_n),
                            ('Plant', plant_r2, plant_rmse, plant_n)]:
    for yr, r2v, rm, n in zip(HOLDOUT_YEARS, r2s, rmses, ns):
        rows.append({'Level': lvl, 'Year': yr, 'R2': r2v,
                     'RMSE_log': rm, 'N_test': n})
for lvl, ov, cs, tp in [('Grid', g_overall, g_cs, g_temp),
                        ('Plant', p_overall, p_cs, p_temp)]:
    rows.append({'Level': lvl, 'Year': 'Overall/CS/Temporal', 'R2': ov,
                 'RMSE_log': cs, 'N_test': tp})
csv_out = pd.DataFrame(rows)
# Make the decomposition rows self-describing
csv_out.loc[csv_out['Year'] == 'Overall/CS/Temporal',
            'Note'] = 'R2=overall pooled OOF; RMSE_log column holds CS_R2; N_test column holds Temporal_R2'
csv_path = os.path.join(OUT_TAB, 'table_figR22_year_by_year.csv')
csv_out.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ── Figure (2-panel: Panel C cross-sectional/temporal removed 2026-07-26) ─────
print("Plotting figure ...")
fig = plt.figure(figsize=(10, 5), dpi=300)
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

x     = np.arange(len(HOLDOUT_YEARS))
width = 0.35

ax1.bar(x - width/2, grid_r2,  width, label='Grid',  color='steelblue',  alpha=0.85)
ax1.bar(x + width/2, plant_r2, width, label='Plant', color='darkorange', alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(HOLDOUT_YEARS)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel('$R^2$')
ax1.set_title('(A) $R^2$ by Year (LOYO out-of-fold)')
ax1.axhline(0.8, color='gray', linestyle='--', linewidth=0.8)
ax1.legend()

ax2.bar(x - width/2, grid_rmse,  width, label='Grid',  color='steelblue',  alpha=0.85)
ax2.bar(x + width/2, plant_rmse, width, label='Plant', color='darkorange', alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(HOLDOUT_YEARS)
ax2.set_ylabel('RMSE (log output)')
ax2.set_title('(B) RMSE by Year (log output)')
ax2.legend()

fig.tight_layout()
for dest in (OUT_FIG_LOCAL, OUT_RL, OUT_FIG):
    out_path = os.path.join(dest, 'figR2.2_year_by_year_r2.pdf')
    try:
        fig.savefig(out_path, format='pdf', bbox_inches='tight')
        print(f"Saved: {out_path}")
    except PermissionError:
        alt = os.path.join(dest, 'figR2.2_year_by_year_r2_v2.pdf')
        fig.savefig(alt, format='pdf', bbox_inches='tight')
        print(f"Saved (alt, original locked by viewer): {alt}")
plt.close()

print("\nDone.")
