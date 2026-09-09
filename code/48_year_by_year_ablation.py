
import os, sys, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, OUT_RL, REP_ROOT, need
warnings.filterwarnings('ignore')

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = REP_ROOT
CONF_DIR = DATA_CONFIDENTIAL
OUT_DIR  = OUT_RL
os.makedirs(OUT_DIR, exist_ok=True)

SEED  = 42
YEARS = [2019, 2020, 2021, 2022]

# ── Feature definitions (identical to Scripts 47-49) ─────────────────────────
STEEL_HUBS = {
    'Tangshan': (39.6, 118.2), 'Wuhan': (30.6, 114.3),
    'Anshan':   (41.1, 122.8), 'Rizhao': (35.4, 119.5),
    'Baotou':   (40.7, 109.8),
}
PORTS = {
    'Shanghai':    (31.23, 121.47), 'Tianjin':   (38.98, 117.72),
    'Qingdao':     (36.07, 120.38), 'Ningbo':    (29.87, 121.55),
    'Guangzhou':   (23.10, 113.43), 'Dalian':    (38.92, 121.65),
    'Lianyungang': (34.75, 119.45), 'Yingkou':   (40.67, 122.23),
}
POLLUTANTS = ['Plant_CO_MEAN', 'Plant_NO2_MEAN', 'Plant_PM2_5_MEAN', 'Plant_PM10_MEAN',
              'Plant_SO2_MEAN', 'Plant_LSTA_MEAN', 'Plant_LSTT_MEAN',
              'Plant_NTL_MEAN', 'Plant_O3_MEAN']
LAG_FEATS  = [f'{b}_lag{k}' for b in POLLUTANTS for k in [1, 2, 3]]
TEMP_FEATS = ['month_sin', 'month_cos']
HUB_COLS   = [f'dist_{h}' for h in STEEL_HUBS] + ['dist_nearest_hub']
PORT_COLS  = ['dist_nearest_port']
GEM_COLS   = ['nom_capacity_ttpa', 'bof_capacity_ttpa', 'eaf_capacity_ttpa', 'is_integrated']
LATLON     = ['Latitude', 'Longitude']

XGB_PARAMS = dict(colsample_bytree=0.6, learning_rate=0.1, max_depth=6,
                  n_estimators=300, subsample=0.8, alpha=0.2, reg_lambda=0.5,
                  random_state=SEED, verbosity=0)

# ── Helpers ───────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2-lat1), np.radians(lon2-lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def find_file(candidates):
    p = next((c for c in candidates if os.path.exists(c)), None)
    if p is None:
        raise FileNotFoundError(f'Not found: {candidates}')
    return p

def cs_temporal_r2(y_true, y_pred, pids):
    """Cross-sectional and temporal R² decomposition."""
    df = pd.DataFrame({'y': y_true, 'yhat': y_pred, 'pid': pids})
    means = df.groupby('pid')[['y','yhat']].mean()
    cs_r2 = r2_score(means['y'], means['yhat'])
    df['y_dm']   = df['y']   - df.groupby('pid')['y'].transform('mean')
    df['yhat_dm']= df['yhat']- df.groupby('pid')['yhat'].transform('mean')
    temp_r2 = r2_score(df['y_dm'], df['yhat_dm'])
    return cs_r2, temp_r2

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading data...')
re_data = pd.read_csv(find_file([need(DATA_CONFIDENTIAL, 're_data.csv'),
                                  os.path.join(CONF_DIR, 're_data.csv')]))
re_data = re_data.dropna(subset=['Steel_Prod', 'Year', 'Month', 'Latitude', 'Longitude'])
re_data['log_prod'] = np.log1p(re_data['Steel_Prod'])
print(f'  re_data: {re_data.shape}  years: {sorted(re_data["Year"].unique())}')

# ── Feature engineering (hub/port distances + GEM) ───────────────────────────
print('Engineering features...')
loc    = pd.read_excel(os.path.join(CONF_DIR, 'Location_Plants_Full.xlsx'))
loc_op = loc[loc['Capacityoperatingstatus'] == 'operating'].dropna(
             subset=['latitude', 'longitude'])
plants = re_data[['plant_id','Latitude','Longitude']].drop_duplicates(
             'plant_id').reset_index(drop=True)

for hub, (hlat, hlon) in STEEL_HUBS.items():
    plants[f'dist_{hub}'] = plants.apply(
        lambda r: haversine_km(r['Latitude'], r['Longitude'], hlat, hlon), axis=1)
plants['dist_nearest_hub'] = plants[[f'dist_{h}' for h in STEEL_HUBS]].min(axis=1)

for port, (plat, plon) in PORTS.items():
    plants[f'_p_{port}'] = plants.apply(
        lambda r: haversine_km(r['Latitude'], r['Longitude'], plat, plon), axis=1)
plants['dist_nearest_port'] = plants[[f'_p_{p}' for p in PORTS]].min(axis=1)
plants = plants.drop([f'_p_{p}' for p in PORTS], axis=1)

gem_tree = cKDTree(np.radians(loc_op[['latitude','longitude']].values))
_, idx   = gem_tree.query(np.radians(plants[['Latitude','Longitude']].values), k=1)
plants['nom_capacity_ttpa'] = loc_op.iloc[idx]['Nominalcrudesteelcapacitytt'].values
plants['bof_capacity_ttpa'] = loc_op.iloc[idx]['NominalBOFsteelcapacityttpa'].values
plants['eaf_capacity_ttpa'] = loc_op.iloc[idx]['NominalEAFsteelcapacityttpa'].values
plants['is_integrated']     = (loc_op.iloc[idx]['Mainproductionprocess'].values ==
                                'integrated (BF)').astype(float)
for col in ['nom_capacity_ttpa', 'bof_capacity_ttpa', 'eaf_capacity_ttpa']:
    plants[col] = plants[col].fillna(0)

re_data = re_data.merge(
    plants[['plant_id'] + HUB_COLS + PORT_COLS + GEM_COLS],
    on='plant_id', how='left')

# ── Build feature lists for each spec ────────────────────────────────────────
SAT_FEATS = [c for c in POLLUTANTS  if c in re_data.columns]
LAG_COLS  = [c for c in LAG_FEATS   if c in re_data.columns]
TMP_COLS  = [c for c in TEMP_FEATS  if c in re_data.columns]
DIST_COLS = [c for c in HUB_COLS + PORT_COLS + GEM_COLS if c in re_data.columns]
LL_COLS   = [c for c in LATLON if c in re_data.columns]

SAT_BASE = SAT_FEATS + LAG_COLS + TMP_COLS

SPECS = {
    'A: Lat/Lon only (no distances)': SAT_BASE + LL_COLS,
    'B: M3 — distances, no Lat/Lon':  SAT_BASE + DIST_COLS,
    'C: M3 + Lat/Lon':                SAT_BASE + DIST_COLS + LL_COLS,
}

# Drop rows missing any feature used across all specs or the target
all_feats = list(set(SAT_BASE + DIST_COLS + LL_COLS))
re_clean  = re_data.dropna(subset=all_feats + ['log_prod']).copy().reset_index(drop=True)
print(f'  Clean rows: {len(re_clean)}  plants: {re_clean["plant_id"].nunique()}')

# ── Leave-one-year-out CV ─────────────────────────────────────────────────────
def loyo_cv(feats, label):
    """Leave-one-year-out out-of-fold predictions for a given feature set."""
    records = []
    for test_yr in YEARS:
        tr_mask = re_clean['Year'] != test_yr
        te_mask = re_clean['Year'] == test_yr
        X_tr = re_clean.loc[tr_mask, feats].values
        X_te = re_clean.loc[te_mask, feats].values
        y_tr = re_clean.loc[tr_mask, 'log_prod'].values
        y_te = re_clean.loc[te_mask, 'log_prod'].values
        pids = re_clean.loc[te_mask, 'plant_id'].values

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s  = sc.transform(X_te)

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)

        r2   = r2_score(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        rho, _ = spearmanr(y_te, y_pred)
        cs_r2, temp_r2 = cs_temporal_r2(y_te, y_pred, pids)
        records.append({'Spec': label, 'Year': test_yr,
                        'R2': round(r2,4), 'RMSE': round(rmse,4),
                        'Spearman': round(rho,4),
                        'CS_R2': round(cs_r2,4), 'Temp_R2': round(temp_r2,4),
                        'N': len(y_te)})
    return pd.DataFrame(records)

print()
all_dfs = []
for name, feats in SPECS.items():
    feats_present = [f for f in feats if f in re_clean.columns]
    print(f'Running {name}  ({len(feats_present)} features)...')
    df = loyo_cv(feats_present, name)
    all_dfs.append(df)
    for _, row in df.iterrows():
        print(f"  {int(row['Year'])}: R²={row['R2']:.4f}  RMSE={row['RMSE']:.4f}"
              f"  ρ={row['Spearman']:.4f}  CS_R²={row['CS_R2']:.4f}  Temp_R²={row['Temp_R2']:.4f}")
    print()

results = pd.concat(all_dfs, ignore_index=True)

# ── Save tables ───────────────────────────────────────────────────────────────
out_yr  = os.path.join(OUT_DIR, 'table_r22_year_by_year_m3.csv')
out_cs  = os.path.join(OUT_DIR, 'table_r22_cs_temporal_m3.csv')
results.to_csv(out_yr, index=False)

cs_rows = []
for name in SPECS:
    sub = results[results['Spec'] == name]
    cs_rows.append({'Spec': name,
                    'CS_R2_mean':   round(sub['CS_R2'].mean(), 4),
                    'Temp_R2_mean': round(sub['Temp_R2'].mean(), 4),
                    'R2_mean':      round(sub['R2'].mean(), 4)})
cs_tbl = pd.DataFrame(cs_rows)
cs_tbl.to_csv(out_cs, index=False)
print(f'Tables saved to {OUT_DIR}')

# ── Within-year 80/20 R² by year (second panel) ─────────────────────────────
from sklearn.model_selection import train_test_split
def within_year_r2(feats):
    out = {}
    for yr in YEARS:
        sub = re_clean[re_clean['Year'] == yr]
        X = sub[feats].values; y = sub['log_prod'].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        m = xgb.XGBRegressor(**XGB_PARAMS); m.fit(Xtr, ytr)
        out[yr] = r2_score(yte, m.predict(Xte))
    return out

within = {name: within_year_r2([f for f in feats if f in re_clean.columns])
          for name, feats in SPECS.items()}

# ── Figure: clean 2-panel (LOYO + within-year R²; no temporal, no "M3") ──────
plt.rcParams['font.family'] = 'serif'
fig = plt.figure(figsize=(11, 4.5), dpi=150)
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.22)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])

colors = {'A: Lat/Lon only (no distances)': '#4e79a7',
          'B: M3 — distances, no Lat/Lon':  '#f28e2b',
          'C: M3 + Lat/Lon':                '#59a14f'}
labels_short = {'A: Lat/Lon only (no distances)': 'A: Lat/Lon only',
                'B: M3 — distances, no Lat/Lon':  'B: Distances (no Lat/Lon)',
                'C: M3 + Lat/Lon':                'C: Distances + Lat/Lon'}
x = np.arange(len(YEARS)); width = 0.25

for i, name in enumerate(SPECS):
    sub = results[results['Spec'] == name].sort_values('Year')
    ax1.bar(x + (i - 1)*width, sub['R2'], width,
            label=labels_short[name], color=colors[name], alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(YEARS, fontsize=9)
ax1.set_ylim(0, 1.05); ax1.set_ylabel('$R^2$', fontsize=10)
ax1.set_title('(A) Leave-one-year-out CV', fontsize=10, fontweight='bold')
ax1.axhline(0.8, color='gray', ls='--', lw=0.7); ax1.legend(fontsize=8, loc='lower left')

for i, name in enumerate(SPECS):
    vals = [within[name][yr] for yr in YEARS]
    ax2.bar(x + (i - 1)*width, vals, width, color=colors[name], alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(YEARS, fontsize=9)
ax2.set_ylim(0, 1.05); ax2.set_ylabel('$R^2$', fontsize=10)
ax2.set_title('(B) Within-year 80/20 split', fontsize=10, fontweight='bold')
ax2.axhline(0.8, color='gray', ls='--', lw=0.7)

fig.suptitle('Year-by-Year $R^2$ under Three Feature Specifications (Plant Level)',
             fontsize=11, fontweight='bold', y=1.02)
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, 'fig_r13_ablation.pdf')
fig.savefig(fig_path, bbox_inches='tight')
print(f'Figure saved: {fig_path}')

# ── Print summary ─────────────────────────────────────────────────────────────
print()
print('=' * 72)
print('SUMMARY — Cross-Sectional vs. Temporal Decomposition (mean across years)')
print('=' * 72)
print(cs_tbl[['Spec','R2_mean','CS_R2_mean','Temp_R2_mean']].to_string(index=False))

print()
print('=' * 72)
print('Year-by-Year R² Summary')
print('=' * 72)
pivot = results.pivot_table(index='Year', columns='Spec', values='R2')
print(pivot.to_string())

print('\nDone.')
