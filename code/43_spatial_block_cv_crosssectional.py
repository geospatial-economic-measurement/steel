"""
43_spatial_block_cv_crosssectional.py
=====================================
HONEST spatial-block cross-validation of the steel-output ML model, testing the
CROSS-SECTIONAL (level / ranking) dimension the model is actually good at — NOT
growth. Companion / candidate replacement for the growth-based geographic
holdout in Figure 7b (script 21), which shows the model cannot do growth.

Design
------
Leave-one-block-out spatial CV. For each held-out block, train on ALL other
blocks with the RAW log target, predict the held-out block's plant-month log
output, then aggregate to PLANT MEANS and score cross-sectional level/ranking.

Split schemes (plant level, 41-feature set from script 21):
  (A) North/South at 35 N, both directions.
  (B) Latitude bands: 4 bands by plant-latitude quantiles (~equal plant counts),
      leave-one-band-out.
  (C) K-means geographic clusters on (Latitude, Longitude), K=5, random_state=5
      (fixed / reproducible), leave-one-cluster-out.

Cross-sectional metrics per scheme (plant means of actual vs predicted log
output within each held-out block):
  1. Spearman rho — pooled AFTER subtracting each block's mean from both series
     (so between-block level gaps do not inflate pooling); and per-block.
  2. Region-demeaned R2 (evaluation-side demeaning by held-out block mean),
     pooled and per-block.
  3. Raw-level R2 (no demeaning), pooled — reference.
  4. n_plants held out.
CONTRAST (honesty): the SAME held-out blocks' plant-level matched-month annual
YoY growth r / R2 (expect ~0), reusing script 21's matched-month logic.

Grid level (schemes A and C only), 40-feature grid spec from script 12,
target log(GridProd_Steel_tot): pooled cross-sectional rho and demeaned R2.

INTEGRITY GUARDRAILS (a target-leakage bug was caught in a sibling script):
  * The feature matrix X is built by EXPLICIT column selection of the confirmed
    feature set only. An assertion verifies the target and any y-derived
    (mean / deviation) column is absent from X before every fit.
  * All demeaning is EVALUATION-SIDE only (subtract the held-out block mean from
    actual and predicted AFTER prediction) — never a feature, never in training.
  * Cross-sectional and temporal (growth) metrics are reported SEPARATELY.

Outputs (NEW names only; nothing existing overwritten):
  output/figures/fig_spatial_cv_crosssectional_scatter.pdf/.png
  output/figures/fig_spatial_cv_by_block.pdf/.png
    (+ copies to output_rl/)
  output/tables/table_spatial_block_cv.csv
    (+ copy to output_rl/)
  LEDGER.md row appended (DIAGNOSTIC — candidate Fig 7b replacement).
"""

import os, sys, io, warnings, shutil, datetime
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, pearsonr
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_FIG, OUT_RL, OUT_TAB, need

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Paths ──────────────────────────────────────────────────────────────────────
SIGNALS_PATH  = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH      = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')
GRID_DATA     = need(DATA_PROCESSED, 'non_zero_df_month.csv')
OUT_RL  = OUT_RL
for d in (OUT_FIG, OUT_TAB, OUT_RL):
    os.makedirs(d, exist_ok=True)

EARTH_R = 6371.0
KMEANS_SEED = 5          # fixed / reproducible k-means seed (noted in outputs)
K_CLUSTERS  = 5
MIN_MATCHED_MONTHS = 8   # matches script 21 / 27 computation A2
TRIM = 2.0               # |actual growth| > 200% excluded from growth contrast

JMP_BLUE = '#00468B'
JMP_RED  = '#ED0000'
JMP_GREY = '#737373'

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

# 41-feature plant set (identical to script 21)
FEATURE_COLS = (
    PLANT_POLLUTANTS +
    [f'{p}_lag{k}' for k in (1, 2, 3) for p in PLANT_POLLUTANTS] +
    ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port', 'is_bof']
)
# 40-feature grid set (identical to script 12)
GRID_FEATURE_COLS = (
    POLLUTANTS +
    [f'{p}_lag{k}' for k in (1, 2, 3) for p in POLLUTANTS] +
    ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port']
)

XGB_PARAMS = dict(                 # plant-level (script 21)
    colsample_bytree=0.8, learning_rate=0.1, max_depth=6,
    n_estimators=300, subsample=0.6,
    alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0,
)
XGB_PARAMS_GRID = dict(            # grid-level (script 12)
    colsample_bytree=0.6, learning_rate=0.1, max_depth=6,
    n_estimators=300, subsample=0.8,
    alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0,
)

# Columns that must NEVER appear in a feature matrix (target + any y-derived).
FORBIDDEN_IN_X = ['Steel_Prod', 'GridProd_Steel_tot', 'y', 'y_d', 'cm',
                  'actual_log', 'pred_log', 'pred', 'pred_orig',
                  'block_mean', 'demeaned', 'dev']


def assert_no_leak(X_cols, expected):
    """Guardrail: feature matrix is EXACTLY the confirmed set, no y-derived col."""
    xset = list(X_cols)
    assert xset == list(expected), (
        f'Feature columns differ from confirmed set. Extra='
        f'{set(xset)-set(expected)} Missing={set(expected)-set(xset)}')
    for bad in FORBIDDEN_IN_X:
        assert bad not in xset, f'LEAK: forbidden column {bad!r} in feature matrix'


def haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((r2 - r1) / 2)**2 +
         np.cos(r1) * np.cos(r2) * np.sin(np.deg2rad(lon2 - lon1) / 2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def add_dist_features(df, lat_col='Latitude', lon_col='Longitude'):
    lats, lons = df[lat_col].values, df[lon_col].values
    hub_d  = np.stack([haversine_km(lats, lons, la, lo)
                       for la, lo in CHINA_HUBS.values()], axis=1)
    port_d = np.stack([haversine_km(lats, lons, la, lo)
                       for la, lo in CHINA_PORTS.values()], axis=1)
    df = df.copy()
    df['dist_nearest_hub']  = hub_d.min(axis=1)
    df['dist_nearest_port'] = port_d.min(axis=1)
    return df


def save_fig(fig, name):
    for ext in ('pdf', 'png'):
        p = os.path.join(OUT_FIG, f'{name}.{ext}')
        try:
            fig.savefig(p, bbox_inches='tight', dpi=800)
            print(f'  Saved: {p}')
        except PermissionError:
            p = os.path.join(OUT_FIG, f'{name}_v2.{ext}')
            fig.savefig(p, bbox_inches='tight', dpi=800)
            print(f'  Saved (alt): {p}')


def copy_file(src, dst_dir):
    dst = os.path.join(dst_dir, os.path.basename(src))
    try:
        shutil.copy2(src, dst); print(f'  Copied: {dst}')
    except (PermissionError, OSError) as e:
        print(f'  COPY FAILED ({e}): {dst}')


# ── Matched-month annual growth (from script 21, computation A2) ────────────────
def matched_month_annual_growth(df):
    """Per-plant matched-month adjacent-year growth pairs. df needs columns
    name_prod, Year, Month, Steel_Prod, pred (pred in ORIGINAL scale)."""
    cur  = df[['name_prod', 'Year', 'Month', 'Steel_Prod', 'pred']].copy()
    base = cur.rename(columns={'Steel_Prod': 'base_actual', 'pred': 'base_pred'})
    base['Year'] = base['Year'] + 1
    pairs = cur.merge(base, on=['name_prod', 'Year', 'Month'], how='inner')
    pairs = pairs.rename(columns={'Steel_Prod': 'cur_actual', 'pred': 'cur_pred'})
    ann = (pairs.groupby(['name_prod', 'Year'])
                .agg(n_matched_months=('Month', 'nunique'),
                     cur_actual=('cur_actual', 'sum'),
                     base_actual=('base_actual', 'sum'),
                     cur_pred=('cur_pred', 'sum'),
                     base_pred=('base_pred', 'sum'))
                .reset_index())
    ann = ann[ann['n_matched_months'] >= MIN_MATCHED_MONTHS].copy()
    if len(ann) == 0:
        return ann
    p1 = np.percentile(np.concatenate([ann['cur_actual'].values,
                                       ann['base_actual'].values]), 1)
    ann = ann[ann['base_actual'] >= p1].copy()
    ann['actual_gr'] = ann['cur_actual'] / ann['base_actual'] - 1
    ann['pred_gr']   = ann['cur_pred']   / ann['base_pred']   - 1
    fin = np.isfinite(ann['actual_gr']) & np.isfinite(ann['pred_gr'])
    return ann[fin].copy()


# ════════════════════════════════════════════════════════════════════════════════
# PLANT-LEVEL DATA PREP (identical pipeline to script 21)
# ════════════════════════════════════════════════════════════════════════════════
print('Loading plant-level data ...')
signals  = pd.read_csv(SIGNALS_PATH).drop(columns=['PM1_MEAN'], errors='ignore')
gridprod = pd.read_csv(GRIDPROD_PATH)
gridprod = gridprod.drop_duplicates(subset=['IDCode', 'name_prod', 'Year', 'Month'])

bf_sheet   = pd.read_excel(GEM_PATH, sheet_name='Blast furnaces', header=0)
bf_gem_ids = set(bf_sheet.iloc[:, 0].dropna().astype(str).unique())
gem_tech   = gridprod[['name_prod', 'GEMPlantID']].drop_duplicates('name_prod').copy()
gem_tech['is_bof'] = gem_tech['GEMPlantID'].apply(lambda x: 1 if str(x) in bf_gem_ids else 0)

temp = gridprod[['IDCode', 'Longitude', 'Latitude', 'name_prod',
                 'Year', 'Month', 'Steel_Prod']].copy()
merged = pd.merge(signals, temp, on=['IDCode', 'Year', 'Month'], how='left')

agg_signals = {col: 'mean' for col in POLLUTANTS}
grouped = (merged.groupby(['name_prod', 'Year', 'Month'])
           .agg({**agg_signals, 'Longitude': 'mean', 'Latitude': 'mean',
                 'Steel_Prod': 'first'}).reset_index())
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
grouped = add_dist_features(grouped)

data = grouped.dropna(subset=FEATURE_COLS + ['Steel_Prod', 'Year', 'Latitude']).copy()
data = data[data['Steel_Prod'] > 0].reset_index(drop=True)
print(f'  Plant panel after dropna: {data["name_prod"].nunique()} plants, {len(data)} rows')


# ── Assign spatial blocks ───────────────────────────────────────────────────────
# Plant-level latitude/longitude (constant per plant; use mean to be safe).
plant_geo = data.groupby('name_prod')[['Latitude', 'Longitude']].mean()

# Scheme B: 4 latitude bands by plant-latitude quantiles (~equal plant counts)
plant_geo['lat_band'] = pd.qcut(plant_geo['Latitude'], 4,
                                labels=['B1_south', 'B2', 'B3', 'B4_north'])
# Scheme C: k-means on plant (Latitude, Longitude), fixed seed
km = KMeans(n_clusters=K_CLUSTERS, random_state=KMEANS_SEED, n_init=10)
plant_geo['kmeans'] = km.fit_predict(plant_geo[['Latitude', 'Longitude']].values)
plant_geo['kmeans'] = 'C' + plant_geo['kmeans'].astype(str)

data = data.merge(plant_geo[['lat_band', 'kmeans']], left_on='name_prod',
                  right_index=True, how='left')
data['ns35'] = np.where(data['Latitude'] >= 35, 'North_ge35', 'South_lt35')


# ── Leave-one-block-out runner (plant level, cross-sectional) ───────────────────
def run_lobo_plant(df, block_col, scheme_tag):
    """Leave-one-block-out. Returns dict of pooled metrics, per-block rows, and a
    plant-level dataframe (block, actual_dm, pred_dm) for the scatter."""
    blocks = sorted(df[block_col].dropna().unique())
    per_block, plant_rows, test_frames = [], [], []
    for b in blocks:
        train = df[df[block_col] != b]
        test  = df[df[block_col] == b].copy()

        X_train = train[FEATURE_COLS]
        X_test  = test[FEATURE_COLS]
        assert_no_leak(X_train.columns, FEATURE_COLS)   # leak guard (train)
        assert_no_leak(X_test.columns,  FEATURE_COLS)   # leak guard (test)

        y_train = np.log1p(train['Steel_Prod'].values)  # RAW log target only
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train.values, y_train)

        test['pred_log']   = model.predict(X_test.values)     # log1p space
        test['actual_log'] = np.log1p(test['Steel_Prod'].values)
        test['pred']       = np.expm1(test['pred_log'].values)  # for growth contrast
        test_frames.append(test)

        # Plant means (mean over months of actual & predicted log output)
        pm = (test.groupby('name_prod')
                  .agg(actual=('actual_log', 'mean'), pred=('pred_log', 'mean')))
        n_pl = len(pm)
        rho_b = spearmanr(pm['actual'], pm['pred'])[0] if n_pl >= 2 else np.nan
        a_dm = pm['actual'] - pm['actual'].mean()   # EVALUATION-side demeaning
        p_dm = pm['pred']   - pm['pred'].mean()
        r2_b = r2_score(a_dm, p_dm) if n_pl >= 2 else np.nan
        per_block.append(dict(scheme=scheme_tag, block=str(b), n_plants=n_pl,
                              spearman=rho_b, r2_demeaned=r2_b))
        for name, ad, pd_ in zip(pm.index, a_dm.values, p_dm.values):
            plant_rows.append(dict(scheme=scheme_tag, block=str(b),
                                   name_prod=name, actual_dm=ad, pred_dm=pd_,
                                   actual_raw=pm.loc[name, 'actual'],
                                   pred_raw=pm.loc[name, 'pred']))

    pdf = pd.DataFrame(plant_rows)
    rho_pool = spearmanr(pdf['actual_dm'], pdf['pred_dm'])[0]
    r2_dm_pool = r2_score(pdf['actual_dm'], pdf['pred_dm'])
    r2_raw_pool = r2_score(pdf['actual_raw'], pdf['pred_raw'])

    # Growth contrast on the SAME held-out blocks (out-of-block predictions,
    # each plant appears exactly once) — pooled matched-month annual growth.
    test_all = pd.concat(test_frames, ignore_index=True)
    ann = matched_month_annual_growth(test_all)
    ann_trim = ann[ann['actual_gr'].abs() <= TRIM]
    if len(ann_trim) >= 2:
        g_r  = pearsonr(ann_trim['actual_gr'], ann_trim['pred_gr'])[0]
        g_r2 = r2_score(ann_trim['actual_gr'], ann_trim['pred_gr'])
    else:
        g_r, g_r2 = np.nan, np.nan

    pb = pd.DataFrame(per_block)
    print(f'\n[{scheme_tag}] {len(blocks)} blocks, {len(pdf)} plants total')
    print(f'  pooled (block-demeaned): Spearman rho={rho_pool:.3f}  '
          f'demeaned R2={r2_dm_pool:.3f}  raw R2={r2_raw_pool:.3f}')
    print(f'  per-block Spearman rho range: '
          f'[{pb["spearman"].min():.3f}, {pb["spearman"].max():.3f}]  '
          f'(n>=2 blocks: {int((pb["n_plants"]>=2).sum())}/{len(pb)})')
    print(f'  CONTRAST growth (matched-month, |g|<=200%): '
          f'r={g_r:.3f}  R2={g_r2:.3f}  n_pairs={len(ann_trim)}')
    for _, r in pb.iterrows():
        print(f'    block {r["block"]:>12s}: n_plants={int(r["n_plants"]):3d}  '
              f'rho={r["spearman"]:.3f}  R2_dm={r["r2_demeaned"]:.3f}')

    return dict(scheme=scheme_tag, per_block=pb, plant_df=pdf,
                rho_pool=rho_pool, r2_dm_pool=r2_dm_pool, r2_raw_pool=r2_raw_pool,
                n_plants=len(pdf), growth_r=g_r, growth_r2=g_r2,
                n_growth=len(ann_trim))


print('\n' + '=' * 78)
print('PLANT-LEVEL SPATIAL BLOCK CV (cross-sectional)')
print('=' * 78)
results = []
# Scheme A: North/South both directions (leave-one-block-out on ns35 handles both)
results.append(run_lobo_plant(data, 'ns35',    'A_NorthSouth35'))
# Scheme B: 4 latitude bands
results.append(run_lobo_plant(data, 'lat_band', 'B_LatBands4'))
# Scheme C: k-means K=5
results.append(run_lobo_plant(data, 'kmeans',  f'C_KMeans{K_CLUSTERS}'))


# ════════════════════════════════════════════════════════════════════════════════
# GRID-LEVEL (schemes A and C), 40-feature spec (script 12), target log(GridProd)
# ════════════════════════════════════════════════════════════════════════════════
def prep_grid():
    g = pd.read_csv(GRID_DATA).drop(columns=['Unnamed: 0', 'Polygon_ID'],
                                    errors='ignore')
    g = g.drop_duplicates(subset=['IDCode', 'Year', 'Month'])   # dedup guard
    g['month_sin'] = np.sin(2 * np.pi * g['Month'] / 12)
    g['month_cos'] = np.cos(2 * np.pi * g['Month'] / 12)
    g = g.sort_values(['IDCode', 'Year', 'Month']).reset_index(drop=True)
    for lag in range(1, 4):
        for col in POLLUTANTS:
            g[f'{col}_lag{lag}'] = g.groupby('IDCode')[col].shift(lag)
    g = add_dist_features(g, 'Centroid_Lat', 'Centroid_Long')
    g = g.dropna(subset=GRID_FEATURE_COLS + ['GridProd_Steel_tot',
                                             'Centroid_Lat', 'Centroid_Long'])
    g = g[g['GridProd_Steel_tot'] > 0].copy()
    return g


def run_lobo_grid(g, block_col, scheme_tag):
    blocks = sorted(g[block_col].dropna().unique())
    per_block, cell_rows = [], []
    for b in blocks:
        train = g[g[block_col] != b]
        test  = g[g[block_col] == b].copy()
        assert_no_leak(train[GRID_FEATURE_COLS].columns, GRID_FEATURE_COLS)
        model = xgb.XGBRegressor(**XGB_PARAMS_GRID)
        model.fit(train[GRID_FEATURE_COLS].values,
                  np.log(train['GridProd_Steel_tot'].values))
        test['pred_log']   = model.predict(test[GRID_FEATURE_COLS].values)
        test['actual_log'] = np.log(test['GridProd_Steel_tot'].values)
        cm = (test.groupby('IDCode')
                  .agg(actual=('actual_log', 'mean'), pred=('pred_log', 'mean')))
        n = len(cm)
        rho_b = spearmanr(cm['actual'], cm['pred'])[0] if n >= 2 else np.nan
        a_dm = cm['actual'] - cm['actual'].mean()
        p_dm = cm['pred']   - cm['pred'].mean()
        r2_b = r2_score(a_dm, p_dm) if n >= 2 else np.nan
        per_block.append(dict(scheme=scheme_tag, block=str(b), n_plants=n,
                              spearman=rho_b, r2_demeaned=r2_b))
        for a, p in zip(a_dm.values, p_dm.values):
            cell_rows.append(dict(actual_dm=a, pred_dm=p))
    cdf = pd.DataFrame(cell_rows)
    rho_pool = spearmanr(cdf['actual_dm'], cdf['pred_dm'])[0]
    r2_dm_pool = r2_score(cdf['actual_dm'], cdf['pred_dm'])
    pb = pd.DataFrame(per_block)
    print(f'\n[GRID {scheme_tag}] {len(blocks)} blocks, {len(cdf)} grid cells')
    print(f'  pooled (block-demeaned): Spearman rho={rho_pool:.3f}  '
          f'demeaned R2={r2_dm_pool:.3f}')
    print(f'  per-block rho range: '
          f'[{pb["spearman"].min():.3f}, {pb["spearman"].max():.3f}]')
    return dict(scheme=scheme_tag, per_block=pb, rho_pool=rho_pool,
                r2_dm_pool=r2_dm_pool, r2_raw_pool=np.nan, n_plants=len(cdf),
                growth_r=np.nan, growth_r2=np.nan, n_growth=0, plant_df=None)


print('\n' + '=' * 78)
print('GRID-LEVEL SPATIAL BLOCK CV (cross-sectional, 40-feature spec)')
print('=' * 78)
grid_results = []
try:
    gdf = prep_grid()
    print(f'  Grid panel after prep: {gdf["IDCode"].nunique()} cells, {len(gdf)} rows')
    gdf['ns35'] = np.where(gdf['Centroid_Lat'] >= 35, 'North_ge35', 'South_lt35')
    cell_geo = gdf.groupby('IDCode')[['Centroid_Lat', 'Centroid_Long']].mean()
    kmg = KMeans(n_clusters=K_CLUSTERS, random_state=KMEANS_SEED, n_init=10)
    cell_geo['kmeans'] = 'C' + kmg.fit_predict(
        cell_geo[['Centroid_Lat', 'Centroid_Long']].values).astype(str)
    gdf = gdf.merge(cell_geo[['kmeans']], left_on='IDCode', right_index=True,
                    how='left')
    grid_results.append(run_lobo_grid(gdf, 'ns35',   'A_NorthSouth35_GRID'))
    grid_results.append(run_lobo_grid(gdf, 'kmeans', f'C_KMeans{K_CLUSTERS}_GRID'))
except Exception as e:
    print(f'  GRID level SKIPPED due to error: {e}')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 78)
print('FIGURES')
print('=' * 78)

# Fig 1: pooled predicted vs actual plant-mean log output (region-demeaned),
# colored by held-out block, k-means scheme (C).
res_c = next(r for r in results if r['scheme'].startswith('C_'))
pdf_c = res_c['plant_df']
fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=300)
blocks_c = sorted(pdf_c['block'].unique())
cmap = plt.cm.tab10
for i, b in enumerate(blocks_c):
    sub = pdf_c[pdf_c['block'] == b]
    ax.scatter(sub['actual_dm'], sub['pred_dm'], s=26, alpha=0.7,
               color=cmap(i % 10),
               label=f'{b} (n={len(sub)})')
lim = max(pdf_c['actual_dm'].abs().max(), pdf_c['pred_dm'].abs().max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], '--', lw=1.6, color=JMP_RED,
        label='45$^\\circ$ line')
ax.axhline(0, color=JMP_GREY, lw=0.7, ls=':')
ax.axvline(0, color=JMP_GREY, lw=0.7, ls=':')
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('Actual plant-mean log output $-$ block mean (log points)', fontsize=11)
ax.set_ylabel('Predicted plant-mean log output $-$ block mean (log points)', fontsize=11)
ax.set_title('Spatial block-CV cross-sectional prediction\n'
             '(k-means geographic clusters, leave-one-cluster-out)', fontsize=12)
ax.text(0.97, 0.05,
        f'pooled $\\rho$ = {res_c["rho_pool"]:.3f}\n'
        f'demeaned $R^2$ = {res_c["r2_dm_pool"]:.3f}\n'
        f'n = {res_c["n_plants"]} plants',
        transform=ax.transAxes, fontsize=11, ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
ax.legend(fontsize=8, loc='upper left', framealpha=0.85)
plt.tight_layout()
save_fig(fig, 'fig_spatial_cv_crosssectional_scatter')
plt.close(fig)

# Fig 2: per-block Spearman rho across all schemes (bar chart).
all_pb = pd.concat([r['per_block'] for r in results], ignore_index=True)
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
scheme_order = [r['scheme'] for r in results]
colors = {scheme_order[0]: JMP_BLUE, scheme_order[1]: JMP_RED,
          scheme_order[2]: JMP_GREY}
xpos, xticklabels, xtickpos, cursor = [], [], [], 0
for sc in scheme_order:
    sub = all_pb[all_pb['scheme'] == sc].reset_index(drop=True)
    xs = np.arange(len(sub)) + cursor
    ax.bar(xs, sub['spearman'], color=colors[sc], width=0.8, label=sc,
           edgecolor='black', linewidth=0.4)
    for x, blk in zip(xs, sub['block']):
        xticklabels.append(blk); xtickpos.append(x)
    cursor += len(sub) + 1
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(xtickpos)
ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Per-block Spearman $\\rho$ (plant-mean actual vs predicted)', fontsize=11)
ax.set_title('Cross-sectional ranking accuracy per held-out spatial block, all schemes',
             fontsize=12)
ax.legend(fontsize=9, loc='lower left')
ax.set_ylim(min(-0.1, all_pb['spearman'].min() - 0.05), 1.05)
plt.tight_layout()
save_fig(fig, 'fig_spatial_cv_by_block')
plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# BACKING TABLE
# ════════════════════════════════════════════════════════════════════════════════
rows = []
for r in results + grid_results:
    rows.append(dict(Scheme=r['scheme'], Level=('grid' if 'GRID' in r['scheme']
                                                else 'plant'),
                     Block='POOLED', n_plants=r['n_plants'],
                     Spearman_rho=round(r['rho_pool'], 4),
                     R2_demeaned=round(r['r2_dm_pool'], 4),
                     R2_raw=(round(r['r2_raw_pool'], 4)
                             if np.isfinite(r['r2_raw_pool']) else np.nan),
                     Contrast_growth_r=(round(r['growth_r'], 4)
                                        if np.isfinite(r['growth_r']) else np.nan),
                     Contrast_growth_R2=(round(r['growth_r2'], 4)
                                         if np.isfinite(r['growth_r2']) else np.nan),
                     n_growth_pairs=r['n_growth']))
    for _, pb in r['per_block'].iterrows():
        rows.append(dict(Scheme=r['scheme'],
                         Level=('grid' if 'GRID' in r['scheme'] else 'plant'),
                         Block=pb['block'], n_plants=int(pb['n_plants']),
                         Spearman_rho=round(pb['spearman'], 4),
                         R2_demeaned=round(pb['r2_demeaned'], 4),
                         R2_raw=np.nan, Contrast_growth_r=np.nan,
                         Contrast_growth_R2=np.nan, n_growth_pairs=np.nan))
table = pd.DataFrame(rows)
tab_path = os.path.join(OUT_TAB, 'table_spatial_block_cv.csv')
table.to_csv(tab_path, index=False, encoding='utf-8-sig')
print(f'\nSaved: {tab_path}  ({len(table)} rows)')

# ── Copies to output_rl ─────────────────────────────────────────────────────────
print('\nCopying outputs to output_rl ...')
for name in ('fig_spatial_cv_crosssectional_scatter.pdf',
             'fig_spatial_cv_by_block.pdf'):
    copy_file(os.path.join(OUT_FIG, name), OUT_RL)
copy_file(tab_path, OUT_RL)

# ── Append LEDGER row ───────────────────────────────────────────────────────────
ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
ledger = os.path.join(OUT_RL, 'LEDGER.md')
row = (
    f'\n## Spatial block-CV, CROSS-SECTIONAL — DIAGNOSTIC, candidate Fig 7b '
    f'replacement (not yet in letter/paper) — added {ts}\n\n'
    f'From `RC\\43_spatial_block_cv_crosssectional.py`. Leave-one-block-out '
    f'spatial CV testing the cross-sectional (level/ranking) dimension the model '
    f'is actually good at, NOT growth. Train on all other blocks with the RAW '
    f'log target, predict the held-out block, aggregate to plant means, score '
    f'cross-sectional rho / demeaned R2 (evaluation-side demeaning only — no '
    f'y-derived feature; leak-guard assertion passes). Growth contrast reported '
    f'separately (matched-month, expected ~0). K-means seed = {KMEANS_SEED}.\n\n'
    f'| Scheme | Level | pooled cross-sec rho | demeaned R2 | raw R2 | n_plants | contrast growth r |\n'
    f'|---|---|---|---|---|---|---|\n'
)
for r in results + grid_results:
    row += (f'| {r["scheme"]} | {"grid" if "GRID" in r["scheme"] else "plant"} '
            f'| {r["rho_pool"]:.3f} | {r["r2_dm_pool"]:.3f} '
            f'| {"" if not np.isfinite(r["r2_raw_pool"]) else f"{r['r2_raw_pool']:.3f}"} '
            f'| {r["n_plants"]} '
            f'| {"" if not np.isfinite(r["growth_r"]) else f"{r['growth_r']:.3f}"} |\n')
row += ('\nFigures: `fig_spatial_cv_crosssectional_scatter.pdf`, '
        '`fig_spatial_cv_by_block.pdf`; table `table_spatial_block_cv.csv` '
        '(all in `Output\\` + `output_rl\\`). DIAGNOSTIC — candidate Fig 7b '
        'replacement, not yet in letter/paper.\n')
try:
    with open(ledger, 'a', encoding='utf-8') as fh:
        fh.write(row)
    print(f'  Appended LEDGER row: {ledger}')
except Exception as e:
    print(f'  LEDGER append FAILED: {e}')

print('\nDone.')
