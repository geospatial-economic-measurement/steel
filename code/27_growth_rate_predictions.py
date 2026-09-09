import os, sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_FIG, OUT_TAB, need

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Paths ──────────────────────────────────────────────────────────────────────
GRID_DATA_PATH = need(DATA_PROCESSED, 'non_zero_df_month.csv')
SIGNALS_PATH   = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH  = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH       = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')
OUT_FIG        = OUT_FIG
OUT_TAB        = OUT_TAB
OUT_FIG_PAPER  = OUT_FIG
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_FIG_PAPER, exist_ok=True)

SEED    = 1
EARTH_R = 6371.0
YEARS   = [2019, 2020, 2021, 2022]

np.random.seed(SEED)

# ── Evaluation protocol: 'loyo' (default) or 'kfold' — see docstring ─────────
MODE      = 'kfold' if 'kfold' in [a.lower() for a in sys.argv[1:]] else 'loyo'
SUFFIX    = '_kfold' if MODE == 'kfold' else ''
TITLE_TAG = ' (5-fold CV)' if MODE == 'kfold' else ''
print(f'Evaluation protocol: {MODE.upper()}'
      + (' — 5-fold random OOF (within-year information allowed)'
         if MODE == 'kfold'
         else ' — leave-one-year-out OOF (unseen-year forecast test)'))

JMP_BLUE = '#00468B'
JMP_RED  = '#ED0000'

# ── China reference hubs & ports (identical to scripts 12/17) ─────────────────
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

GRID_FEATURE_COLS = (
    POLLUTANTS +
    [f'{p}_lag{k}' for k in (1, 2, 3) for p in POLLUTANTS] +
    ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port']
)

PLANT_POLLUTANTS = [f'Plant_{p}' for p in POLLUTANTS]

PLANT_FEATURE_COLS = (
    PLANT_POLLUTANTS +
    [f'{p}_lag{k}' for k in (1, 2, 3) for p in PLANT_POLLUTANTS] +
    ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port', 'is_bof']
)

# XGBoost hyperparameters — verbatim from scripts 12 and 17
GRID_XGB_PARAMS = dict(
    colsample_bytree=0.6, learning_rate=0.1, max_depth=6,
    n_estimators=300, subsample=0.8,
    alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0,
)
PLANT_XGB_PARAMS = dict(
    colsample_bytree=0.8, learning_rate=0.1, max_depth=6,
    n_estimators=300, subsample=0.6,
    alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0,
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((r2 - r1) / 2)**2 +
         np.cos(r1) * np.cos(r2) * np.sin(np.deg2rad(lon2 - lon1) / 2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))

def add_distance_features(df, lat_col, lon_col):
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
    """Save figure to both output directories, pdf + png."""
    for out_dir in (OUT_FIG, OUT_FIG_PAPER):
        for ext in ('pdf', 'png'):
            p = os.path.join(out_dir, f'{name}.{ext}')
            try:
                fig.savefig(p, bbox_inches='tight', dpi=300)
                print(f'  Saved: {p}')
            except PermissionError:
                p2 = os.path.join(out_dir, f'{name}_v2.{ext}')
                fig.savefig(p2, bbox_inches='tight', dpi=300)
                print(f'  Saved (alt): {p2}')

def loyo_oof_predict(data, feature_cols, y_log, xgb_params, inverse_fn, label):
    """Leave-one-year-out out-of-fold prediction.

    For each year Y, train on all other years, predict all obs of year Y.
    Returns OOF predicted LEVELS (inverse-transformed, clamped at 0).
    """
    oof_level = pd.Series(np.nan, index=data.index)
    for yr in YEARS:
        tr_mask = data['Year'] != yr
        te_mask = data['Year'] == yr
        if te_mask.sum() == 0:
            print(f'  [{label}] Year {yr}: no observations, skipped')
            continue
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(data.loc[tr_mask, feature_cols], y_log[tr_mask])
        pred_log = model.predict(data.loc[te_mask, feature_cols])
        oof_level.loc[te_mask] = np.clip(inverse_fn(pred_log), 0, None)
        print(f'  [{label}] Year {yr}: train n={tr_mask.sum()}, '
              f'predict n={te_mask.sum()}')
    assert oof_level.notna().all(), f'{label}: some observations lack OOF predictions'
    return oof_level

def kfold_oof_predict(data, feature_cols, y_log, xgb_params, inverse_fn, label):
    """5-fold random out-of-fold prediction (paper's headline CV convention).

    KFold(n_splits=5, shuffle=True, random_state=SEED) over the given
    (deduplicated) sample. Every observation is predicted by the fold model
    that excluded it; unlike LOYO, training folds contain the same
    unit-year's other months. Returns OOF predicted LEVELS (clamped at 0).
    """
    oof_level = pd.Series(np.nan, index=data.index)
    X = data[feature_cols]
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr_pos, te_pos) in enumerate(kf.split(X), start=1):
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X.iloc[tr_pos], y_log.iloc[tr_pos])
        pred_log = model.predict(X.iloc[te_pos])
        oof_level.iloc[te_pos] = np.clip(inverse_fn(pred_log), 0, None)
        print(f'  [{label}] Fold {fold}: train n={len(tr_pos)}, '
              f'predict n={len(te_pos)}')
    assert oof_level.notna().all(), f'{label}: some observations lack OOF predictions'
    return oof_level

def growth_metrics(g_actual, g_pred):
    """Pearson r, R2 (r2_score of g_pred vs g_actual), MAE in pp, n."""
    r   = pearsonr(g_actual, g_pred)[0]
    r2  = r2_score(g_actual, g_pred)
    mae = mean_absolute_error(g_actual, g_pred) * 100  # percentage points
    return r, r2, mae, len(g_actual)

def unit_growth_pairs(df, unit_col, level_actual_col, level_pred_col):
    """Build unit-level YoY growth pairs by merging (unit, Y, M) with (unit, Y-1, M)."""
    cur = df[[unit_col, 'Year', 'Month', level_actual_col, level_pred_col]].copy()
    base = cur.rename(columns={level_actual_col: 'base_actual',
                               level_pred_col:   'base_pred'})
    base['Year'] = base['Year'] + 1  # base year Y-1 matched to current year Y
    pairs = cur.merge(base, on=[unit_col, 'Year', 'Month'], how='inner')
    pairs = pairs.rename(columns={level_actual_col: 'cur_actual',
                                  level_pred_col:   'cur_pred'})
    return pairs

# ══════════════════════════════════════════════════════════════════════════════
# 1. GRID MODEL — data pipeline copied verbatim from script 12
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 78)
print('GRID MODEL: loading data (pipeline identical to script 12) ...')
data_g = pd.read_csv(GRID_DATA_PATH)
data_g = data_g.drop(columns=['Unnamed: 0', 'Polygon_ID'], errors='ignore')

data_g['month_sin'] = np.sin(2 * np.pi * data_g['Month'] / 12)
data_g['month_cos'] = np.cos(2 * np.pi * data_g['Month'] / 12)

data_g = data_g.sort_values(['IDCode', 'Year', 'Month']).reset_index(drop=True)
for lag in range(1, 4):
    for col in POLLUTANTS:
        data_g[f'{col}_lag{lag}'] = data_g.groupby('IDCode')[col].shift(lag)

data_g = data_g.dropna()
data_g = add_distance_features(data_g, lat_col='Centroid_Lat', lon_col='Centroid_Long')
data_g = data_g.reset_index(drop=True)

y_log_g = np.log(data_g['GridProd_Steel_tot'])
print(f'  Grid panel: {data_g["IDCode"].nunique()} grids, {len(data_g)} rows, '
      f'{len(GRID_FEATURE_COLS)} features')
print(f'  Years present: {sorted(data_g["Year"].unique())}')

data_g['actual_level'] = data_g['GridProd_Steel_tot']
if MODE == 'loyo':
    print('\nGRID MODEL: leave-one-year-out OOF prediction ...')
    data_g['oof_pred_level'] = loyo_oof_predict(
        data_g, GRID_FEATURE_COLS, y_log_g, GRID_XGB_PARAMS, np.exp, 'grid')

# ══════════════════════════════════════════════════════════════════════════════
# 2. PLANT MODEL — data pipeline copied verbatim from script 17
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 78)
print('PLANT MODEL: loading data (pipeline identical to script 17) ...')
signals  = pd.read_csv(SIGNALS_PATH)
signals  = signals.drop(columns=['PM1_MEAN'], errors='ignore')
gridprod = pd.read_csv(GRIDPROD_PATH)

bf_sheet   = pd.read_excel(GEM_PATH, sheet_name='Blast furnaces', header=0)
bf_gem_ids = set(bf_sheet.iloc[:, 0].dropna().astype(str).unique())
gem_tech   = (gridprod[['name_prod', 'GEMPlantID']]
              .drop_duplicates('name_prod').copy())
gem_tech['is_bof'] = gem_tech['GEMPlantID'].apply(
    lambda x: 1 if str(x) in bf_gem_ids else 0)
print(f'  BOF plants: {gem_tech["is_bof"].sum()} / {len(gem_tech)} total')

temp = gridprod[['IDCode', 'Longitude', 'Latitude', 'name_prod',
                 'Year', 'Month', 'Steel_Prod', 'Iron_Prod']].copy()
merged = pd.merge(signals, temp, on=['IDCode', 'Year', 'Month'], how='left')

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

grouped['month_sin'] = np.sin(2 * np.pi * grouped['Month'] / 12)
grouped['month_cos'] = np.cos(2 * np.pi * grouped['Month'] / 12)

grouped = grouped.sort_values(['plant_id', 'Year', 'Month']).reset_index(drop=True)
for lag in range(1, 4):
    for col in PLANT_POLLUTANTS:
        grouped[f'{col}_lag{lag}'] = grouped.groupby('plant_id')[col].shift(lag)

grouped = add_distance_features(grouped, lat_col='Latitude', lon_col='Longitude')

data_p = grouped.dropna(subset=PLANT_FEATURE_COLS + ['Steel_Prod']).copy()
data_p = data_p[data_p['Steel_Prod'] > 0].reset_index(drop=True)
print(f'  Plant panel: {data_p["name_prod"].nunique()} plants, {len(data_p)} rows, '
      f'{len(PLANT_FEATURE_COLS)} features')
print(f'  Years present: {sorted(data_p["Year"].unique())}')
mon_cov = data_p.groupby(['Year'])['Month'].nunique()
print(f'  Months per year in plant panel: {mon_cov.to_dict()}')

y_log_p = np.log1p(data_p['Steel_Prod'])

data_p['actual_level'] = data_p['Steel_Prod']
if MODE == 'loyo':
    print('\nPLANT MODEL: leave-one-year-out OOF prediction ...')
    data_p['oof_pred_level'] = loyo_oof_predict(
        data_p, PLANT_FEATURE_COLS, y_log_p, PLANT_XGB_PARAMS, np.expm1, 'plant')

# ══════════════════════════════════════════════════════════════════════════════
# 2b. Evaluation panels: deduplicate (unit, Year, Month)
#    
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 78)
n_dup_g = int(data_g.duplicated(subset=['IDCode', 'Year', 'Month']).sum())
eval_g = data_g.drop_duplicates(subset=['IDCode', 'Year', 'Month']).copy()
print(f'Evaluation panel dedup: grid {n_dup_g} duplicate unit-month rows removed '
      f'({len(data_g)} -> {len(eval_g)} rows)')
n_dup_p = int(data_p.duplicated(subset=['name_prod', 'Year', 'Month']).sum())
eval_p = data_p.drop_duplicates(subset=['name_prod', 'Year', 'Month']).copy()
print(f'Evaluation panel dedup: plant {n_dup_p} duplicate unit-month rows removed '
      f'({len(data_p)} -> {len(eval_p)} rows)')
eval_g = eval_g.reset_index(drop=True)
eval_p = eval_p.reset_index(drop=True)

if MODE == 'kfold':
    # K-fold OOF is trained on the deduplicated samples themselves, so the
    # duplicated Dec-2022 rows can never straddle train/test folds.
    print('\nGRID MODEL: 5-fold random OOF prediction (deduplicated sample) ...')
    eval_g['oof_pred_level'] = kfold_oof_predict(
        eval_g, GRID_FEATURE_COLS, np.log(eval_g['GridProd_Steel_tot']),
        GRID_XGB_PARAMS, np.exp, 'grid')
    print('\nPLANT MODEL: 5-fold random OOF prediction (deduplicated sample) ...')
    eval_p['oof_pred_level'] = kfold_oof_predict(
        eval_p, PLANT_FEATURE_COLS, np.log1p(eval_p['Steel_Prod']),
        PLANT_XGB_PARAMS, np.expm1, 'plant')

# ══════════════════════════════════════════════════════════════════════════════
# 3. Computation A: UNIT-LEVEL year-over-year growth
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 78)
print('A. UNIT-LEVEL YoY growth')

results = {}   # row_name -> (r, R2, MAE_pp, n)
unit_pairs = {}  # for figures
dropped_info = {}

for label, df, unit_col in (('grid',  eval_g, 'IDCode'),
                            ('plant', eval_p, 'name_prod')):
    pairs = unit_growth_pairs(df, unit_col, 'actual_level', 'oof_pred_level')
    n_raw = len(pairs)

    # Drop pairs with near-zero actual base level (< 1st pct of panel levels)
    p1_level = np.percentile(df['actual_level'], 1)
    small_base = pairs['base_actual'] < p1_level
    n_dropped = int(small_base.sum())
    pairs = pairs[~small_base].copy()
    dropped_info[label] = (n_dropped, n_raw, p1_level)

    pairs['g_actual'] = pairs['cur_actual'] / pairs['base_actual'] - 1
    pairs['g_pred']   = pairs['cur_pred']   / pairs['base_pred']   - 1
    fin = np.isfinite(pairs['g_actual']) & np.isfinite(pairs['g_pred'])
    n_nonfinite = int((~fin).sum())
    pairs = pairs[fin].copy()

    r, r2, mae, n = growth_metrics(pairs['g_actual'], pairs['g_pred'])
    results[f'{label}_unit'] = (r, r2, mae, n)
    unit_pairs[label] = pairs
    print(f'\n  [{label}] YoY pairs: {n_raw} raw; dropped {n_dropped} small-base '
          f'(base < 1st pct of levels = {p1_level:.4f}); '
          f'{n_nonfinite} non-finite dropped; {n} retained')
    print(f'  [{label}] pct growth:  r={r:.4f}  R2={r2:.4f}  MAE={mae:.2f}pp  n={n}')

    # Robustness: log-difference growth (log L_t - log L_{t-12})
    lp = pairs[(pairs['cur_actual'] > 0) & (pairs['base_actual'] > 0) &
               (pairs['cur_pred'] > 0) & (pairs['base_pred'] > 0)].copy()
    n_pos_dropped = len(pairs) - len(lp)
    ld_actual = np.log(lp['cur_actual']) - np.log(lp['base_actual'])
    ld_pred   = np.log(lp['cur_pred'])   - np.log(lp['base_pred'])
    r, r2, mae, n = growth_metrics(ld_actual, ld_pred)
    results[f'{label}_unit_logdiff'] = (r, r2, mae, n)
    print(f'  [{label}] log-diff:    r={r:.4f}  R2={r2:.4f}  '
          f'MAE={mae:.2f} (log pts x100)  n={n}'
          + (f'  ({n_pos_dropped} pairs with non-positive level dropped)'
             if n_pos_dropped else ''))

print('=' * 78)
print('A2. UNIT-LEVEL ANNUAL growth (matched-months construction)')

MIN_MATCHED_MONTHS = 8

annual_pairs = {}
annual_dropped = {}
for label, df, unit_col in (('grid',  eval_g, 'IDCode'),
                            ('plant', eval_p, 'name_prod')):
    pm = unit_growth_pairs(df, unit_col, 'actual_level', 'oof_pred_level')
    ann = (pm.groupby([unit_col, 'Year'])
             .agg(n_months=('Month', 'nunique'),
                  cur_actual=('cur_actual', 'sum'),
                  base_actual=('base_actual', 'sum'),
                  cur_pred=('cur_pred', 'sum'),
                  base_pred=('base_pred', 'sum'))
             .reset_index())
    n_raw = len(ann)

    insufficient = ann['n_months'] < MIN_MATCHED_MONTHS
    n_insuff = int(insufficient.sum())
    ann = ann[~insufficient].copy()

    # Small-base guard, analogous to monthly: 1st percentile of the matched
    # annual actual sums (base and current years pooled).
    p1_ann = np.percentile(np.concatenate([ann['cur_actual'].values,
                                           ann['base_actual'].values]), 1)
    small = ann['base_actual'] < p1_ann
    n_small = int(small.sum())
    ann = ann[~small].copy()

    ann['g_actual'] = ann['cur_actual'] / ann['base_actual'] - 1
    ann['g_pred']   = ann['cur_pred']   / ann['base_pred']   - 1
    fin = np.isfinite(ann['g_actual']) & np.isfinite(ann['g_pred'])
    ann = ann[fin].copy()

    r, r2, mae, n = growth_metrics(ann['g_actual'], ann['g_pred'])
    results[f'{label}_unit_annual'] = (r, r2, mae, n)
    annual_pairs[label] = ann
    annual_dropped[label] = (n_insuff, n_small, n_raw, p1_ann)
    print(f'\n  [{label}] annual year-pairs: {n_raw} raw; '
          f'dropped {n_insuff} with < {MIN_MATCHED_MONTHS} matched months; '
          f'dropped {n_small} small-base (base-year sum < 1st pct = {p1_ann:.4f}); '
          f'{n} retained')
    print(f'  [{label}] annual growth: r={r:.4f}  R2={r2:.4f}  '
          f'MAE={mae:.2f}pp  n={n}')

    # Log-difference variant on the identical retained sample
    lpa = ann[(ann['cur_actual'] > 0) & (ann['base_actual'] > 0) &
              (ann['cur_pred'] > 0) & (ann['base_pred'] > 0)]
    n_pos_dropped = len(ann) - len(lpa)
    ld_a = np.log(lpa['cur_actual']) - np.log(lpa['base_actual'])
    ld_p = np.log(lpa['cur_pred'])   - np.log(lpa['base_pred'])
    r, r2, mae, n = growth_metrics(ld_a, ld_p)
    results[f'{label}_unit_annual_logdiff'] = (r, r2, mae, n)
    print(f'  [{label}] annual log-diff: r={r:.4f}  R2={r2:.4f}  '
          f'MAE={mae:.2f} (log-pp)  n={n}'
          + (f'  ({n_pos_dropped} non-positive pairs dropped)'
             if n_pos_dropped else ''))

# ══════════════════════════════════════════════════════════════════════════════
# 4. Computation B: NATIONAL monthly YoY growth
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 78)
print('B. NATIONAL monthly YoY growth (BALANCED panels)')
print('  Balancing rule (as in Analysis/python/r25b_national_growth.py):')
print('  keep only units observed in every month of the post-lag-warm-up')
print('  common window, so composition changes never masquerade as growth.')


def national_series(df, units, unit_col):
    """National monthly totals over a fixed unit set, then YoY growth."""
    sub = df[df[unit_col].isin(units)]
    nat = (sub.groupby(['Year', 'Month'])[['actual_level', 'oof_pred_level']]
              .sum().reset_index()
              .sort_values(['Year', 'Month']).reset_index(drop=True))
    base = nat.rename(columns={'actual_level': 'base_actual',
                               'oof_pred_level': 'base_pred'})
    base['Year'] = base['Year'] + 1
    nat = nat.merge(base, on=['Year', 'Month'], how='inner')
    nat['g_actual'] = nat['actual_level']   / nat['base_actual'] - 1
    nat['g_pred']   = nat['oof_pred_level'] / nat['base_pred']   - 1
    nat['date'] = pd.to_datetime(dict(year=nat['Year'], month=nat['Month'], day=1))
    return nat.sort_values('date').reset_index(drop=True)


national = {}
national_n_units = {}
for label, df, unit_col in (('grid', eval_g, 'IDCode'),
                            ('plant', eval_p, 'name_prod')):
    n_window_months = len(df[['Year', 'Month']].drop_duplicates())
    counts = df.groupby(unit_col).size()   # deduped, so size = distinct months
    balanced_units = counts[counts == n_window_months].index
    print(f'\n  [{label}] common window: {n_window_months} months; '
          f'balanced units: {len(balanced_units)} / {counts.size} '
          f'(max observed months per unit = {counts.max()})')

    use_units, variant_note = balanced_units, 'fully balanced'
    if label == 'plant' and len(balanced_units) < 30:
        thr95 = int(np.ceil(0.95 * n_window_months))
        units95 = counts[counts >= thr95].index
        print(f'  [plant] fully balanced set < 30 plants; also reporting '
              f'>=95% variant (>= {thr95} months): {len(units95)} plants')
        nat95 = national_series(df, units95, unit_col)
        r, r2, mae, n = growth_metrics(nat95['g_actual'], nat95['g_pred'])
        results['national_plant_model_95pct'] = (r, r2, mae, n)
        national_n_units['national_plant_model_95pct'] = len(units95)
        print(f'  [plant model, >=95% panel] national YoY: r={r:.4f}  '
              f'R2={r2:.4f}  MAE={mae:.2f}pp  n={n}  n_units={len(units95)}')

    nat = national_series(df, use_units, unit_col)
    national[label] = nat
    national_n_units[f'national_{label}_model'] = len(use_units)

    r, r2, mae, n = growth_metrics(nat['g_actual'], nat['g_pred'])
    results[f'national_{label}_model'] = (r, r2, mae, n)
    print(f'  [{label} model, {variant_note}] national YoY: r={r:.4f}  '
          f'R2={r2:.4f}  MAE={mae:.2f}pp  n={n} months  '
          f'n_units={len(use_units)} '
          f'({nat["date"].min():%Y-%m} to {nat["date"].max():%Y-%m})')
    print(f'  [{label}] balanced national YoY range: actual '
          f'[{nat["g_actual"].min()*100:+.1f}%, {nat["g_actual"].max()*100:+.1f}%], '
          f'predicted [{nat["g_pred"].min()*100:+.1f}%, {nat["g_pred"].max()*100:+.1f}%]')

    # Log-difference variant on the identical balanced monthly sums
    ld_a = np.log(nat['actual_level'])   - np.log(nat['base_actual'])
    ld_p = np.log(nat['oof_pred_level']) - np.log(nat['base_pred'])
    r, r2, mae, n = growth_metrics(ld_a, ld_p)
    results[f'national_{label}_model_logdiff'] = (r, r2, mae, n)
    national_n_units[f'national_{label}_model_logdiff'] = len(use_units)
    print(f'  [{label} model] national log-diff: r={r:.4f}  R2={r2:.4f}  '
          f'MAE={mae:.2f} (log-pp)  n={n}')

# ══════════════════════════════════════════════════════════════════════════════
# 5. Table
# ══════════════════════════════════════════════════════════════════════════════
row_order = ['grid_unit', 'plant_unit', 'grid_unit_logdiff', 'plant_unit_logdiff',
             'grid_unit_annual', 'plant_unit_annual',
             'grid_unit_annual_logdiff', 'plant_unit_annual_logdiff',
             'national_grid_model', 'national_plant_model',
             'national_grid_model_logdiff', 'national_plant_model_logdiff']
if 'national_plant_model_95pct' in results:
    row_order.append('national_plant_model_95pct')
tab = pd.DataFrame(
    [(k,) + results[k] + (national_n_units.get(k, np.nan),) for k in row_order],
    columns=['row', 'r', 'R2', 'MAE_pp', 'n', 'n_units'])
tab_path = os.path.join(OUT_TAB, f'table_growth_metrics{SUFFIX}.csv')
tab.to_csv(tab_path, index=False)
print(f'\nSaved table: {tab_path}')
print(tab.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 6. Figure 1: unit-level growth scatter (grid | plant)
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 78)
print('Figure 1: unit-level YoY growth scatter ...')

fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
panel_specs = [('grid',  'a', 'Grid level'),
               ('plant', 'b', 'Plant level')]

for ax, (label, letter, title) in zip(axes, panel_specs):
    pairs = unit_pairs[label]
    actual_pct = pairs['g_actual'].values * 100
    pred_pct   = pairs['g_pred'].values   * 100
    r, r2, _, n = results[f'{label}_unit']

    ax.scatter(actual_pct, pred_pct, alpha=0.4, color='blue',
               s=12 if label == 'grid' else 25, edgecolors='none')
    # Axis clipping for display: 1st-99th percentile of actual growth
    p1, p99 = np.percentile(actual_pct, [1, 99])
    pad = 0.05 * (p99 - p1)
    lo, hi = p1 - pad, p99 + pad
    print(f'  [{label}] display axes clipped to 1st-99th pct of actual growth: '
          f'[{p1:.1f}%, {p99:.1f}%] (stats computed on all {n} retained pairs)')
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='45° line (perfect fit)')
    ax.axhline(0, color='gray', lw=0.8, linestyle=':')
    ax.axvline(0, color='gray', lw=0.8, linestyle=':')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual YoY Growth Rate (%)', fontsize=12)
    ax.set_ylabel('Predicted YoY Growth Rate (%)', fontsize=12)
    ax.set_title(f'({letter}) {title}{TITLE_TAG}', fontsize=13)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.8)
    ax.text(0.97, 0.05, f'$R^2$ = {r2:.3f}\n$r$ = {r:.3f}\nn = {n:,}',
            transform=ax.transAxes, fontsize=11, ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
save_fig(fig, f'fig_growth_unit_scatter{SUFFIX}')
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 6b. Figure 1b: unit-level ANNUAL growth scatter (grid | plant)
# ══════════════════════════════════════════════════════════════════════════════
print('Figure 1b: unit-level ANNUAL growth scatter ...')
ANNUAL_TAG = ' (annual, 5-fold CV)' if MODE == 'kfold' else ' (annual, LOYO)'

fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
for ax, (label, letter, title) in zip(axes, panel_specs):
    ann = annual_pairs[label]
    actual_pct = ann['g_actual'].values * 100
    pred_pct   = ann['g_pred'].values   * 100
    r, r2, _, n = results[f'{label}_unit_annual']

    ax.scatter(actual_pct, pred_pct, alpha=0.4, color='blue',
               s=12 if label == 'grid' else 25, edgecolors='none')
    # Axis clipping for display: 1st-99th percentile of actual growth
    p1, p99 = np.percentile(actual_pct, [1, 99])
    pad = 0.05 * (p99 - p1)
    lo, hi = p1 - pad, p99 + pad
    print(f'  [{label}] display axes clipped to 1st-99th pct of actual annual '
          f'growth: [{p1:.1f}%, {p99:.1f}%] (stats computed on all {n} pairs)')
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='45° line (perfect fit)')
    ax.axhline(0, color='gray', lw=0.8, linestyle=':')
    ax.axvline(0, color='gray', lw=0.8, linestyle=':')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual Annual Growth Rate (%)', fontsize=12)
    ax.set_ylabel('Predicted Annual Growth Rate (%)', fontsize=12)
    ax.set_title(f'({letter}) {title}{ANNUAL_TAG}', fontsize=13)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.8)
    ax.text(0.97, 0.05, f'$R^2$ = {r2:.3f}\n$r$ = {r:.3f}\nn = {n:,}',
            transform=ax.transAxes, fontsize=11, ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
save_fig(fig, f'fig_growth_unit_annual_scatter{SUFFIX}')
plt.close(fig)

# ── Figure 1c (conditional): logdiff-unit annual scatter, emitted only for
#    cells where the logdiff variant changes sign or moves r by > 0.10
#    relative to its percentage counterpart ────────────────────────────────────
for label, letter, title in panel_specs:
    r_pct = results[f'{label}_unit_annual'][0]
    r_ld, r2_ld, _, n_ld = results[f'{label}_unit_annual_logdiff']
    if not (np.sign(r_pct) != np.sign(r_ld) or abs(r_ld - r_pct) > 0.10):
        continue
    print(f'  [{label}] annual pct-vs-logdiff divergence '
          f'(r {r_pct:+.3f} -> {r_ld:+.3f}): generating logdiff-unit scatter ...')
    ann = annual_pairs[label]
    lpa = ann[(ann['cur_actual'] > 0) & (ann['base_actual'] > 0) &
              (ann['cur_pred'] > 0) & (ann['base_pred'] > 0)]
    x = (np.log(lpa['cur_actual']) - np.log(lpa['base_actual'])).values * 100
    yv = (np.log(lpa['cur_pred'])  - np.log(lpa['base_pred'])).values  * 100
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.scatter(x, yv, alpha=0.4, color='blue',
               s=12 if label == 'grid' else 25, edgecolors='none')
    p1, p99 = np.percentile(x, [1, 99])
    pad = 0.05 * (p99 - p1)
    lo, hi = p1 - pad, p99 + pad
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='45° line (perfect fit)')
    ax.axhline(0, color='gray', lw=0.8, linestyle=':')
    ax.axvline(0, color='gray', lw=0.8, linestyle=':')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual Annual Growth (log points x100)', fontsize=12)
    ax.set_ylabel('Predicted Annual Growth (log points x100)', fontsize=12)
    ax.set_title(f'{title} — log-difference{ANNUAL_TAG}', fontsize=13)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.8)
    ax.text(0.97, 0.05, f'$R^2$ = {r2_ld:.3f}\n$r$ = {r_ld:.3f}\nn = {n_ld:,}',
            transform=ax.transAxes, fontsize=11, ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.tight_layout()
    save_fig(fig, f'fig_growth_unit_annual_logdiff_{label}{SUFFIX}')
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 7. Figure 2: national YoY growth — time series + scatter
# ══════════════════════════════════════════════════════════════════════════════
print('Figure 2: national YoY growth ...')

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

# Panel A: time series
ax = axes[0]
nat_g, nat_p = national['grid'], national['plant']
ax.plot(nat_g['date'], nat_g['g_actual'] * 100, color='0.25', marker='o',
        ms=4, lw=1.5, label='Actual (balanced grid panel)')
ax.plot(nat_g['date'], nat_g['g_pred'] * 100, color=JMP_BLUE, marker='s',
        ms=4, lw=1.5, label='Grid-model prediction')
ax.plot(nat_p['date'], nat_p['g_pred'] * 100, color=JMP_RED, marker='^',
        ms=4, lw=1.5, linestyle='--', label='Plant-model prediction')
ax.axhline(0, color='gray', lw=0.8, linestyle=':')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('National YoY Growth Rate (%)', fontsize=12)
ax.set_title(f'(a) National YoY growth by month (balanced panels){TITLE_TAG}',
             fontsize=13)
ax.legend(fontsize=9, framealpha=0.8)
for lab in ax.get_xticklabels():
    lab.set_rotation(30)

# Panel B: scatter of predicted vs actual national growth
ax = axes[1]
r_g = results['national_grid_model'][0]
r_p = results['national_plant_model'][0]
ax.scatter(nat_g['g_actual'] * 100, nat_g['g_pred'] * 100, color=JMP_BLUE,
           marker='s', s=35, alpha=0.8, label=f'Grid model (r = {r_g:.3f})')
ax.scatter(nat_p['g_actual'] * 100, nat_p['g_pred'] * 100, color=JMP_RED,
           marker='^', s=35, alpha=0.8, label=f'Plant model (r = {r_p:.3f})')
all_vals = np.concatenate([nat_g['g_actual'], nat_g['g_pred'],
                           nat_p['g_actual'], nat_p['g_pred']]) * 100
lo, hi = all_vals.min() - 3, all_vals.max() + 3
ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='45° line (perfect fit)')
ax.axhline(0, color='gray', lw=0.8, linestyle=':')
ax.axvline(0, color='gray', lw=0.8, linestyle=':')
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel('Actual National YoY Growth Rate (%)', fontsize=12)
ax.set_ylabel('Predicted National YoY Growth Rate (%)', fontsize=12)
ax.set_title(f'(b) Predicted vs. actual national growth{TITLE_TAG}', fontsize=13)
ax.legend(fontsize=9, loc='upper left', framealpha=0.8)

plt.tight_layout()
save_fig(fig, f'fig_growth_national{SUFFIX}')
plt.close(fig)

if MODE == 'kfold':
    print('Manuscript figures (K-fold) ...')

    # ── 8a. Unit-level MONTHLY YoY log-difference scatter ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    for ax, (label, letter, title) in zip(axes, panel_specs):
        pairs = unit_pairs[label]
        lp = pairs[(pairs['cur_actual'] > 0) & (pairs['base_actual'] > 0) &
                   (pairs['cur_pred'] > 0) & (pairs['base_pred'] > 0)]
        x  = (np.log(lp['cur_actual']) - np.log(lp['base_actual'])).values * 100
        yv = (np.log(lp['cur_pred'])   - np.log(lp['base_pred'])).values   * 100
        r, r2, _, n = results[f'{label}_unit_logdiff']
        print(f'  [ms unit fig, {label}] annotation: r={r:.3f}  R2={r2:.3f}  '
              f'n={n:,}')
        ax.scatter(x, yv, alpha=0.4, color='blue',
                   s=12 if label == 'grid' else 25, edgecolors='none')
        p1, p99 = np.percentile(x, [1, 99])
        pad = 0.05 * (p99 - p1)
        lo, hi = p1 - pad, p99 + pad
        print(f'  [ms unit fig, {label}] display clipped to 1st-99th pct: '
              f'[{p1:.1f}, {p99:.1f}] log points')
        ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='45° line (perfect fit)')
        ax.axhline(0, color='gray', lw=0.8, linestyle=':')
        ax.axvline(0, color='gray', lw=0.8, linestyle=':')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('Actual YoY change (log points)', fontsize=12)
        ax.set_ylabel('Predicted YoY change (log points)', fontsize=12)
        ax.set_title(f'({letter}) {title}', fontsize=13)
        ax.legend(fontsize=10, loc='upper left', framealpha=0.8)
        ax.text(0.97, 0.05, f'$R^2$ = {r2:.3f}\n$r$ = {r:.3f}\nn = {n:,}',
                transform=ax.transAxes, fontsize=11, ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.tight_layout()
    save_fig(fig, 'fig_growth_unit_kfold_ms')
    plt.close(fig)

    # ── 8b. National figure without protocol suffix in panel titles ───────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    ax = axes[0]
    ax.plot(nat_g['date'], nat_g['g_actual'] * 100, color='0.25', marker='o',
            ms=4, lw=1.5, label='Actual (balanced grid panel)')
    ax.plot(nat_g['date'], nat_g['g_pred'] * 100, color=JMP_BLUE, marker='s',
            ms=4, lw=1.5, label='Grid-model prediction')
    ax.plot(nat_p['date'], nat_p['g_pred'] * 100, color=JMP_RED, marker='^',
            ms=4, lw=1.5, linestyle='--', label='Plant-model prediction')
    ax.axhline(0, color='gray', lw=0.8, linestyle=':')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('National YoY Growth Rate (%)', fontsize=12)
    ax.set_title('(a) National YoY growth by month (balanced panels)',
                 fontsize=13)
    ax.legend(fontsize=9, framealpha=0.8)
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)

    ax = axes[1]
    ax.scatter(nat_g['g_actual'] * 100, nat_g['g_pred'] * 100, color=JMP_BLUE,
               marker='s', s=35, alpha=0.8, label=f'Grid model (r = {r_g:.3f})')
    ax.scatter(nat_p['g_actual'] * 100, nat_p['g_pred'] * 100, color=JMP_RED,
               marker='^', s=35, alpha=0.8, label=f'Plant model (r = {r_p:.3f})')
    all_vals = np.concatenate([nat_g['g_actual'], nat_g['g_pred'],
                               nat_p['g_actual'], nat_p['g_pred']]) * 100
    lo, hi = all_vals.min() - 3, all_vals.max() + 3
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='45° line (perfect fit)')
    ax.axhline(0, color='gray', lw=0.8, linestyle=':')
    ax.axvline(0, color='gray', lw=0.8, linestyle=':')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual National YoY Growth Rate (%)', fontsize=12)
    ax.set_ylabel('Predicted National YoY Growth Rate (%)', fontsize=12)
    ax.set_title('(b) Predicted vs. actual national growth', fontsize=13)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.8)

    plt.tight_layout()
    save_fig(fig, 'fig_growth_national_kfold_ms')
    plt.close(fig)

# ── Summary ───────────────────────────────────────────────────────────────────
print('=' * 78)
print('SUMMARY')
for label in ('grid', 'plant'):
    n_dropped, n_raw, p1_level = dropped_info[label]
    print(f'  [{label}] monthly small-base pairs dropped: {n_dropped} / {n_raw} '
          f'(1st pct of levels = {p1_level:.4f})')
    n_insuff, n_small, n_raw_a, p1_ann = annual_dropped[label]
    print(f'  [{label}] annual pairs dropped: {n_insuff} (<{MIN_MATCHED_MONTHS} '
          f'matched months) + {n_small} small-base of {n_raw_a} raw '
          f'(1st pct of annual sums = {p1_ann:.4f})')
print('\nDone.')
