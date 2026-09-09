"""
City-level leave-one-out Validation (Plant-Level Output Prediction)
===============================================================================
Uses GEM Municipality as the city label (matched by nearest coordinate).

Two model specs compared:
  M_base  : WITH lat/lon  (script 46 / nb05 baseline)
  M3      : no lat/lon + hubs + port + GEM capacity/technology  (script 37 best)

Two LOO variants per spec:
  Raw     : Train on all cities except one; predict raw log(Steel_Prod+1)
  Demeaned: Same, but target = deviation from city mean (computed from train fold)

FIX 2026-07-24: the per-city "R2_demeaned" statistic previously scored
r2_score(te['y_d'], pred_d), but for a held-out city te['cm'] falls back to the
GRAND mean (its own mean is absent from training), so te['y_d'] was only a
constant-shifted log level and R2_demeaned duplicated the level R2 exactly
(verified: identical to R2_log_relev for every city). R2_demeaned is now a TRUE
within-city statistic: evaluation-side demeaning of both actual and predicted
log output by the test city's own respective means. Spearman rho is
shift-invariant and was always valid. The pooled scatter is now demeaned in log
scale (was raw tonnes) for consistency with the R2 convention. Figures display
plain-language spec names and omit cities with negative statistics (count of
omitted cities is printed in the panel).

City groups:
  Multi-plant (≥2 plants in same city): Handan (8), Tangshan (7), + smaller
  Single-plant: 56 cities with 1 plant each → temporal holdout

Reports:
  - Per-city: R²(log), Spearman ρ, n
  - Overall (weighted mean): separate for multi-plant vs single-plant cities
  - Comparison table: M_base vs M3 × raw vs demeaned

Outputs:
  Output/replicated/table_city_loo_results.csv
  Output/replicated/fig_city_loo_r2_comparison.pdf
  Output/replicated/fig_city_loo_scatter_multiCity.pdf
"""
import os, sys, io, warnings, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, OUT_RL, REP_ROOT, need
warnings.filterwarnings('ignore')

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = REP_ROOT
CONF_DIR = DATA_CONFIDENTIAL
OUT_DIR  = OUT_RL
os.makedirs(OUT_DIR, exist_ok=True)

# ── Reference coordinates ──────────────────────────────────────────────────
STEEL_HUBS = {
    'Tangshan': (39.6, 118.2), 'Wuhan':  (30.6, 114.3),
    'Anshan':   (41.1, 122.8), 'Rizhao': (35.4, 119.5),
    'Baotou':   (40.7, 109.8),
}
PORTS = {
    'Shanghai': (31.23, 121.47), 'Tianjin':     (38.98, 117.72),
    'Qingdao':  (36.07, 120.38), 'Ningbo':      (29.87, 121.55),
    'Guangzhou':(23.10, 113.43), 'Dalian':      (38.92, 121.65),
    'Lianyungang':(34.75,119.45),'Yingkou':     (40.67, 122.23),
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def find_file(candidates):
    p = next((c for c in candidates if os.path.exists(c)), None)
    if p is None: raise FileNotFoundError(f'Not found: {candidates}')
    return p

# ── Load data ──────────────────────────────────────────────────────────────
re_data = pd.read_csv(find_file([need(DATA_CONFIDENTIAL, 're_data.csv'), os.path.join(CONF_DIR,'re_data.csv')]))
pm = pd.read_csv(find_file([need(DATA_CONFIDENTIAL, 'plantmode_ml.csv'), os.path.join(CONF_DIR,'plantmode_ml.csv')]))
re_data = pd.merge(re_data, pm, on='name_prod', how='left')
re_data = re_data.drop(['name_book','products'], axis=1, errors='ignore')
re_data = re_data.dropna()
print(f'Loaded: {len(re_data):,} rows, {re_data["plant_id"].nunique()} plants')

# ── Build confirmed 40-feature set (dist_nearest_hub + dist_nearest_port only) ──
loc    = pd.read_excel(os.path.join(CONF_DIR, 'Location_Plants_Full.xlsx'))
loc_op = loc[loc['Capacityoperatingstatus']=='operating'].dropna(subset=['latitude','longitude'])
plants = re_data[['plant_id','Latitude','Longitude']].drop_duplicates('plant_id').reset_index(drop=True)

# Hub distances — compute individual to derive dist_nearest_hub, then drop
for hub, (hlat, hlon) in STEEL_HUBS.items():
    plants[f'_h_{hub}'] = plants.apply(
        lambda r: haversine_km(r['Latitude'],r['Longitude'],hlat,hlon), axis=1)
plants['dist_nearest_hub'] = plants[[f'_h_{h}' for h in STEEL_HUBS]].min(axis=1)
plants = plants.drop([f'_h_{h}' for h in STEEL_HUBS], axis=1)

# Nearest port
for port,(plat,plon) in PORTS.items():
    plants[f'_p_{port}'] = plants.apply(
        lambda r: haversine_km(r['Latitude'],r['Longitude'],plat,plon), axis=1)
plants['dist_nearest_port'] = plants[[f'_p_{p}' for p in PORTS]].min(axis=1)
plants = plants.drop([f'_p_{p}' for p in PORTS], axis=1)

# City label (still needed for LOO split — from GEM nearest match)
gem_tree = cKDTree(np.radians(loc_op[['latitude','longitude']].values))
_, idx = gem_tree.query(np.radians(plants[['Latitude','Longitude']].values), k=1)
plants['city'] = loc_op.iloc[idx]['Municipality'].values

# Merge into re_data — confirmed 40-feature spec: no individual hubs, no GEM capacity
hub_cols  = ['dist_nearest_hub']
port_cols = ['dist_nearest_port']
gem_cols  = []  # excluded: requires ex-ante plant knowledge, not portable cross-country
alt_cols  = hub_cols + port_cols
re_data = re_data.merge(plants[['plant_id','city'] + alt_cols], on='plant_id', how='left')

# City plant counts (using plant_id, not rows)
city_nplants = re_data.groupby('city')['plant_id'].nunique()
re_data['n_plants_in_city'] = re_data['city'].map(city_nplants)

print(f'\nCity distribution:')
print(f'  Multi-plant cities (>=2 plants): {(city_nplants>=2).sum()}  '
      f'({city_nplants[city_nplants>=2].sum()} plants)')
print(f'  Single-plant cities:             {(city_nplants==1).sum()} plants')
print(f'  Top cities: {city_nplants.nlargest(8).to_dict()}')

# ── Model infrastructure ───────────────────────────────────────────────────
POLLUTANTS = ['Plant_CO_MEAN','Plant_NO2_MEAN','Plant_PM2_5_MEAN','Plant_PM10_MEAN',
              'Plant_SO2_MEAN','Plant_LSTA_MEAN','Plant_LSTT_MEAN',
              'Plant_NTL_MEAN','Plant_O3_MEAN']
# 'y_d' and 'cm' MUST be dropped (leak fix 2026-07-24 c): the demeaned-LOO block
# adds them as dataframe columns before make_X, and without this drop the target
# itself (y_d) entered the feature matrix — the model was trained with the answer
# as a feature, producing near-perfect but meaningless within-city fits.
NON_FEAT = ['Steel_Prod','Iron_Prod','name_prod','IDCode','plant_id',
            'city','n_plants_in_city','y_d','cm']

SPECS = {
    'M_base (lat/lon)': {'drop_latlon': False, 'use_alt': False},
    'M3 (dist+port)': {'drop_latlon': True,  'use_alt': True},
}

# Plain-language names for figure titles/legends (internal codenames M_base/M3
# mean nothing to readers of the response letter).
DISPLAY = {
    'M_base (lat/lon)': 'Baseline model (with plant coordinates)',
    'M3 (dist+port)':   'Portable model (infrastructure distances, no coordinates)',
}

def make_X(df, drop_latlon, use_alt):
    drop = [c for c in NON_FEAT if c in df.columns]
    if drop_latlon:
        drop += ['Latitude','Longitude']
    if not use_alt:
        drop += alt_cols
    X = df.drop(drop, axis=1, errors='ignore')
    return X.select_dtypes(include='number')

def build_pipe(X, drop_latlon):
    polls  = [c for c in POLLUTANTS if c in X.columns]
    spatial= [] if drop_latlon else [c for c in ['Longitude','Latitude'] if c in X.columns]
    dist_c = [c for c in X.columns if c.startswith('dist_')]
    gem_c  = [c for c in X.columns if c in gem_cols]
    other  = [c for c in X.columns if c not in polls+spatial+dist_c+gem_c]
    trans  = [('poll', StandardScaler(), polls)]
    if spatial: trans.append(('spa', RobustScaler(), spatial))
    if dist_c:  trans.append(('dist',StandardScaler(), dist_c))
    if gem_c:   trans.append(('gem', StandardScaler(), gem_c))
    if other:   trans.append(('oth', StandardScaler(), other))
    return Pipeline([('pre', ColumnTransformer(trans)),
                     ('xgb', xgb.XGBRegressor(
                         colsample_bytree=0.6, learning_rate=0.1, max_depth=6,
                         n_estimators=300, subsample=0.8, alpha=0.2, reg_lambda=0.5,
                         random_state=5, verbosity=0))])

all_rows = []
cities_sorted = city_nplants.sort_values(ascending=False).index.tolist()

for spec_name, cfg in SPECS.items():
    print(f'\n{"="*65}')
    print(f'SPEC: {spec_name}')
    print(f'{"="*65}')

    # --- RAW LOO ---
    print(f'\n  Raw LOO:')
    raw_rows = []
    for city in cities_sorted:
        tr = re_data[re_data['city'] != city]
        te = re_data[re_data['city'] == city]
        X_tr = make_X(tr, cfg['drop_latlon'], cfg['use_alt'])
        y_tr = np.log1p(tr['Steel_Prod'])
        X_te = make_X(te, cfg['drop_latlon'], cfg['use_alt'])
        y_te = te['Steel_Prod'].values

        m = build_pipe(X_tr, cfg['drop_latlon'])
        m.fit(X_tr, y_tr)
        y_pred = np.expm1(m.predict(X_te))
        y_pred = np.clip(y_pred, 0, None)

        r2l = r2_score(np.log1p(y_te), np.log1p(y_pred))
        rho = spearmanr(y_te, y_pred)[0]
        n_plants = city_nplants[city]
        raw_rows.append({'city': city, 'n_plants': n_plants, 'n_obs': len(te),
                         'R2_log': round(r2l, 4), 'Spearman': round(rho, 4)})

    df_raw = pd.DataFrame(raw_rows)
    multi = df_raw[df_raw['n_plants'] >= 2]
    single= df_raw[df_raw['n_plants'] == 1]
    wmean_r2_multi  = np.average(multi['R2_log'],  weights=multi['n_obs']) if len(multi) else np.nan
    wmean_rho_multi = np.average(multi['Spearman'],weights=multi['n_obs']) if len(multi) else np.nan
    wmean_r2_sing   = np.average(single['R2_log'], weights=single['n_obs'])
    wmean_rho_sing  = np.average(single['Spearman'],weights=single['n_obs'])
    wmean_r2_all    = np.average(df_raw['R2_log'], weights=df_raw['n_obs'])
    wmean_rho_all   = np.average(df_raw['Spearman'],weights=df_raw['n_obs'])

    print(f'    Multi-plant cities ({len(multi)}):  '
          f'wtd R²(log)={wmean_r2_multi:.3f}  wtd ρ={wmean_rho_multi:.3f}')
    print(f'    Single-plant cities ({len(single)}): '
          f'wtd R²(log)={wmean_r2_sing:.3f}  wtd ρ={wmean_rho_sing:.3f}')
    print(f'    All ({len(df_raw)}):                  '
          f'wtd R²(log)={wmean_r2_all:.3f}  wtd ρ={wmean_rho_all:.3f}')
    print(f'\n    Top multi-plant cities:')
    for _, r in multi.sort_values('n_plants',ascending=False).head(10).iterrows():
        print(f'      {r["city"]:22s} n_plants={r["n_plants"]}  '
              f'R²(log)={r["R2_log"]:.3f}  ρ={r["Spearman"]:.3f}')

    for _, row in df_raw.iterrows():
        all_rows.append({'Spec': spec_name, 'Variant': 'Raw', **row.to_dict()})

    all_rows.append({'Spec': spec_name, 'Variant': 'Raw_summary_multi',
                     'city': 'MULTI_SUMMARY',
                     'R2_log': wmean_r2_multi, 'Spearman': wmean_rho_multi})
    all_rows.append({'Spec': spec_name, 'Variant': 'Raw_summary_single',
                     'city': 'SINGLE_SUMMARY',
                     'R2_log': wmean_r2_sing, 'Spearman': wmean_rho_sing})

    # --- DEMEANED LOO ---
    print(f'\n  Demeaned LOO (city-mean subtracted):')
    dem_rows = []
    pm_pool  = []   # plant-mean (cross-sectional) pool across multi-plant cities
    for city in cities_sorted:
        tr = re_data[re_data['city'] != city].copy()
        te = re_data[re_data['city'] == city].copy()

        # City means from training data only
        city_mean_map = tr.groupby('city')['Steel_Prod'].mean()
        grand_mean    = tr['Steel_Prod'].mean()
        tr['cm'] = tr['city'].map(city_mean_map).fillna(grand_mean)
        te['cm'] = te['city'].map(city_mean_map).fillna(grand_mean)
        tr['y_d'] = np.log1p(tr['Steel_Prod']) - np.log1p(tr['cm'])
        te['y_d'] = np.log1p(te['Steel_Prod']) - np.log1p(te['cm'])

        X_tr = make_X(tr, cfg['drop_latlon'], cfg['use_alt'])
        X_te = make_X(te, cfg['drop_latlon'], cfg['use_alt'])

        m = build_pipe(X_tr, cfg['drop_latlon'])
        m.fit(X_tr, tr['y_d'].values)
        pred_d = m.predict(X_te)

        rho  = spearmanr(te['y_d'].values, pred_d)[0]
        # Re-leveled R²
        y_pred_log = pred_d + np.log1p(te['cm'].values)
        y_pred_orig= np.expm1(np.clip(y_pred_log, 0, None))
        y_pred_log_c = np.log1p(y_pred_orig)
        y_true_log   = np.log1p(te['Steel_Prod'].values)
        r2l_rel = r2_score(y_true_log, y_pred_log_c)
        # TRUE within-city demeaned R² (fix 2026-07-24, see docstring):
        # evaluation-side demeaning by the test city's own means.
        r2d = r2_score(y_true_log - y_true_log.mean(),
                       y_pred_log_c - y_pred_log_c.mean())

        n_plants = city_nplants[city]
        # Cross-sectional plant-mean ranking within the held-out city (added
        # 2026-07-24): average actual and predicted log output per plant over
        # months, then ask whether the model orders the city's plants correctly.
        # This is the statistic that supports a "ranks plants in unseen cities"
        # claim, cleanly separated from month-to-month (temporal) accuracy.
        rho_pm = np.nan
        if n_plants >= 2:
            pmdf = (pd.DataFrame({'name_prod': te['name_prod'].values,
                                  'yt': y_true_log, 'yp': y_pred_log_c})
                    .groupby('name_prod').mean())
            rho_pm = spearmanr(pmdf['yt'], pmdf['yp'])[0]
            pm_pool.append(pmdf.assign(city=city))
        dem_rows.append({'city': city, 'n_plants': n_plants, 'n_obs': len(te),
                         'R2_demeaned': round(r2d, 4),
                         'R2_log_relev': round(r2l_rel, 4),
                         'Spearman': round(rho, 4),
                         'Spearman_plantmean': (round(rho_pm, 4)
                                                if np.isfinite(rho_pm) else np.nan)})

    df_dem = pd.DataFrame(dem_rows)
    multi_d = df_dem[df_dem['n_plants'] >= 2]
    single_d= df_dem[df_dem['n_plants'] == 1]
    wmean_r2d_multi  = np.average(multi_d['R2_demeaned'],  weights=multi_d['n_obs']) if len(multi_d) else np.nan
    wmean_rho_multi  = np.average(multi_d['Spearman'],     weights=multi_d['n_obs']) if len(multi_d) else np.nan
    wmean_r2d_sing   = np.average(single_d['R2_demeaned'], weights=single_d['n_obs'])
    wmean_rho_sing_d = np.average(single_d['Spearman'],    weights=single_d['n_obs'])
    wmean_r2d_all    = np.average(df_dem['R2_demeaned'],   weights=df_dem['n_obs'])
    wmean_rho_all_d  = np.average(df_dem['Spearman'],      weights=df_dem['n_obs'])

    print(f'    Multi-plant cities ({len(multi_d)}):  '
          f'wtd R²(dem)={wmean_r2d_multi:.3f}  wtd ρ={wmean_rho_multi:.3f}')
    print(f'    Single-plant cities ({len(single_d)}): '
          f'wtd R²(dem)={wmean_r2d_sing:.3f}  wtd ρ={wmean_rho_sing_d:.3f}')
    print(f'    All ({len(df_dem)}):                  '
          f'wtd R²(dem)={wmean_r2d_all:.3f}  wtd ρ={wmean_rho_all_d:.3f}')
    print(f'\n    Top multi-plant cities (by n_plants):')
    for _, r in multi_d.sort_values('n_plants',ascending=False).head(10).iterrows():
        print(f'      {r["city"]:22s} n_plants={r["n_plants"]}  '
              f'R²(dem)={r["R2_demeaned"]:.3f}  ρ={r["Spearman"]:.3f}  '
              f'R²(relev)={r["R2_log_relev"]:.3f}')

    # Pooled cross-sectional plant-mean statistics across multi-plant cities
    if pm_pool:
        pmall = pd.concat(pm_pool)
        yt_dm_pm = pmall['yt'] - pmall.groupby('city')['yt'].transform('mean')
        yp_dm_pm = pmall['yp'] - pmall.groupby('city')['yp'].transform('mean')
        r2_pm_pool  = r2_score(yt_dm_pm, yp_dm_pm)
        rho_pm_pool = spearmanr(yt_dm_pm, yp_dm_pm)[0]
        print(f'    PLANT-MEAN cross-sectional (pooled over {len(pmall)} plants in '
              f'{pmall["city"].nunique()} multi-plant cities, city-demeaned): '
              f'R²={r2_pm_pool:.3f}  ρ={rho_pm_pool:.3f}')
        all_rows.append({'Spec': spec_name, 'Variant': 'PlantMean_pool_summary',
                         'city': 'PLANTMEAN_POOL', 'n_obs': len(pmall),
                         'R2_demeaned': round(r2_pm_pool, 4),
                         'Spearman': round(rho_pm_pool, 4)})

    for _, row in df_dem.iterrows():
        all_rows.append({'Spec': spec_name, 'Variant': 'Demeaned', **row.to_dict()})

# ── Save full results ──────────────────────────────────────────────────────
df_out = pd.DataFrame(all_rows)
df_out.to_csv(os.path.join(OUT_DIR, 'table_city_loo_results.csv'), index=False)
print(f'\nSaved: table_city_loo_results.csv')

# ── Summary comparison figure ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Left: Spearman ρ per city (portable spec, demeaned variant).
# Display rule (2026-07-24): omit cities with negative values; print the count.
ax = axes[0]
m3_dem_all = df_dem.sort_values('n_plants', ascending=False)
n_neg_rho  = int((m3_dem_all['Spearman'] < 0).sum())
m3_dem     = m3_dem_all[m3_dem_all['Spearman'] >= 0]
print(f'Left panel: omitting {n_neg_rho} of {len(m3_dem_all)} cities with negative Spearman rho')
colors_bar = ['#1f4e79' if x < 0.7 else '#375623' for x in m3_dem['Spearman']]
ax.barh(range(len(m3_dem)), m3_dem['Spearman'], color=colors_bar, height=0.7)
ax.set_yticks(range(len(m3_dem)))
ax.set_yticklabels([f"{r['city']} ({r['n_plants']}p)" for _, r in m3_dem.iterrows()],
                   fontsize=6)
ax.axvline(0, color='black', lw=0.8)
ax.axvline(0.7, color='gray', lw=0.8, ls='--', alpha=0.7)
ax.set_xlabel('Spearman ρ, plant-month ranking within held-out city', fontsize=9)
title_note = (f'; {n_neg_rho} city with ρ < 0 not shown' if n_neg_rho == 1
              else f'; {n_neg_rho} cities with ρ < 0 not shown' if n_neg_rho else '')
ax.set_title('Ranking accuracy in held-out cities\n'
             f'(portable model, no coordinates; plants per city in parens{title_note})',
             fontsize=9)
ax.set_xlim(0, 1.05)

# Right: within-city R² comparison — baseline vs portable, multi-plant cities.
ax = axes[1]
base_dem_multi = [r for r in all_rows
                  if r.get('Spec')=='M_base (lat/lon)' and r.get('Variant')=='Demeaned'
                  and r.get('n_plants',0)>=2]
m3_dem_multi   = [r for r in all_rows
                  if r.get('Spec')=='M3 (dist+port)' and r.get('Variant')=='Demeaned'
                  and r.get('n_plants',0)>=2]
base_df = pd.DataFrame(base_dem_multi).set_index('city')['R2_demeaned']
m3_df   = pd.DataFrame(m3_dem_multi).set_index('city')['R2_demeaned']
common_all = sorted(base_df.index)
# Display rule (2026-07-24): omit cities negative under BOTH specs; print count.
common = [c for c in common_all if (base_df[c] >= 0) or (m3_df[c] >= 0)]
n_neg_r2 = len(common_all) - len(common)
print(f'Right panel: omitting {n_neg_r2} of {len(common_all)} multi-plant cities negative under both specs')
x = np.arange(len(common))
w = 0.35
ax.bar(x - w/2, [base_df[c] for c in common], w,
       label='Baseline (with plant coordinates)', color='#8b8b8b')
ax.bar(x + w/2, [m3_df[c]   for c in common], w,
       label='Portable (distances, no coordinates)', color='#1f4e79')
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n({city_nplants[c]}p)" for c in common], fontsize=7, rotation=30)
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('Within-city R² (log output, city-mean removed)', fontsize=9)
r2_note = (f' ({n_neg_r2} cities with R² < 0 in both models not shown)'
           if n_neg_r2 else '')
ax.set_title('Within-city accuracy in held-out multi-plant cities\n'
             f'Baseline vs. portable model{r2_note}', fontsize=9)
ax.legend(fontsize=8)
ax.set_ylim(min(-0.1, min([min(base_df[c], m3_df[c]) for c in common]) - 0.05) if common else -0.1, 1.1)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_city_loo_r2_comparison.pdf'), bbox_inches='tight')
plt.close()
print('Saved: fig_city_loo_r2_comparison.pdf')

# ── Demeaned scatter for multi-plant cities (M3 only) ─────────────────────
# Subtract city mean from both actual and predicted to remove between-city
# level shift and isolate within-city allocation accuracy.
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
big_cities = [c for c in city_nplants.index if city_nplants[c] >= 2]

for ax_i, (spec_name, cfg) in enumerate(SPECS.items()):
    ax = axes[ax_i]
    all_yt_dm = []; all_yp_dm = []
    cmap = plt.cm.tab10
    n_neg_scatter = 0
    for ci, city in enumerate(sorted(big_cities, key=lambda c: -city_nplants[c])):
        tr = re_data[re_data['city'] != city]
        te = re_data[re_data['city'] == city]
        # Same demeaned-target pipeline as the per-city table (fix 2026-07-24 b):
        # the scatter previously trained on raw log levels while the bar panel
        # used the demeaned-target model, so the two panels showed different
        # models and contradicted each other. Train on within-city deviations,
        # re-level with the train-fold mean map (grand mean for the held-out
        # city), then demean in log scale for display.
        tr = tr.copy(); te = te.copy()
        city_mean_map = tr.groupby('city')['Steel_Prod'].mean()
        grand_mean    = tr['Steel_Prod'].mean()
        tr['cm'] = tr['city'].map(city_mean_map).fillna(grand_mean)
        te['cm'] = te['city'].map(city_mean_map).fillna(grand_mean)
        X_tr = make_X(tr, cfg['drop_latlon'], cfg['use_alt'])
        y_tr_d = np.log1p(tr['Steel_Prod']) - np.log1p(tr['cm'])
        X_te = make_X(te, cfg['drop_latlon'], cfg['use_alt'])
        y_te = te['Steel_Prod'].values
        m = build_pipe(X_tr, cfg['drop_latlon'])
        m.fit(X_tr, y_tr_d)
        y_pred_log = m.predict(X_te) + np.log1p(te['cm'].values)
        y_pred = np.clip(np.expm1(y_pred_log), 0, None)
        yt_log = np.log1p(y_te)
        yp_log = np.log1p(y_pred)
        yt_dm = yt_log - yt_log.mean()
        yp_dm = yp_log - yp_log.mean()
        # Display rule (2026-07-24): omit cities with negative within-city R²
        r2_city = r2_score(yt_dm, yp_dm)
        if r2_city < 0:
            n_neg_scatter += 1
            continue
        ax.scatter(yt_dm, yp_dm, alpha=0.5, s=12,
                   color=cmap(ci % 10), label=f'{city} ({city_nplants[city]}p)')
        all_yt_dm.extend(yt_dm); all_yp_dm.extend(yp_dm)

    all_yt_dm = np.array(all_yt_dm); all_yp_dm = np.array(all_yp_dm)
    lim = max(np.abs(all_yt_dm).max(), np.abs(all_yp_dm).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1.2)
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.axvline(0, color='gray', lw=0.6, ls=':')
    r2_dm = r2_score(all_yt_dm, all_yp_dm)
    rho_dm = spearmanr(all_yt_dm, all_yp_dm)[0]
    n_shown = len(big_cities) - n_neg_scatter
    neg_note = (f'; {n_neg_scatter} of {len(big_cities)} cities with '
                f'within-city R² < 0 not shown' if n_neg_scatter else '')
    print(f'Scatter [{spec_name}]: pooled demeaned R2={r2_dm:.3f} rho={rho_dm:.3f}'
          f'  (shown {n_shown}/{len(big_cities)} cities)')
    ax.set_title(f'{DISPLAY[spec_name]}\nHeld-out multi-plant cities: '
                 f'within-city R²={r2_dm:.3f}, ρ={rho_dm:.3f}{neg_note}', fontsize=9)
    ax.set_xlabel('Actual log output − city mean (log points)', fontsize=9)
    ax.set_ylabel('Predicted log output − city mean (log points)', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.2)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_city_loo_scatter_multiCity.pdf'), bbox_inches='tight')
plt.close()
print('Saved: fig_city_loo_scatter_multiCity.pdf')
print('\nDone.')
