
import os, sys, io, warnings, shutil, time
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_RL, OUT_TAB, need

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
SIGNALS_PATH  = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH      = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')
GRID_DATA     = need(DATA_PROCESSED, 'non_zero_df_month.csv')
OUT_RL  = OUT_RL
for d in (OUT_TAB, OUT_RL):
    os.makedirs(d, exist_ok=True)

SEED = 1
EARTH_R = 6371.0

CHINA_HUBS = {'Tangshan': (39.6, 118.2), 'Wuhan': (30.6, 114.3), 'Anshan': (41.1, 122.8),
              'Rizhao': (35.4, 119.5), 'Baotou': (40.7, 109.8)}
CHINA_PORTS = {'Shanghai': (31.23, 121.47), 'Tianjin': (38.98, 117.72), 'Qingdao': (36.07, 120.38),
               'Ningbo': (29.87, 121.55), 'Guangzhou': (23.10, 113.43), 'Dalian': (38.92, 121.65),
               'Lianyungang': (34.75, 119.45), 'Yingkou': (40.67, 122.23)}
POLLUTANTS = ['CO_MEAN', 'NO2_MEAN', 'SO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN',
              'O3_MEAN', 'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN']
PLANT_POLLUTANTS = [f'Plant_{p}' for p in POLLUTANTS]

# Confirmed feature sets
GRID_FEATURES = (POLLUTANTS + [f'{p}_lag{k}' for k in (1, 2, 3) for p in POLLUTANTS]
                 + ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port'])
PLANT_FEATURES = (PLANT_POLLUTANTS + [f'{p}_lag{k}' for k in (1, 2, 3) for p in PLANT_POLLUTANTS]
                  + ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port', 'is_bof'])
# XGBoost hyperparameters — verbatim from scripts 12 (grid) and 09/13 (plant)
GRID_XGB = dict(colsample_bytree=0.6, learning_rate=0.1, max_depth=6, n_estimators=300,
                subsample=0.8, alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0)
PLANT_XGB = dict(colsample_bytree=0.8, learning_rate=0.1, max_depth=6, n_estimators=300,
                 subsample=0.6, alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0)

# Columns that must never enter a feature matrix (target + any y-derived)
FORBIDDEN_IN_X = ['Steel_Prod', 'GridProd_Steel_tot', 'oof', 'pred', 'pred_log',
                  'actual_log', 'y', 'y_d', 'cm', 'province']


def assert_no_leak(cols, expected):
    xset = list(cols)
    assert xset == list(expected), (f'feature set mismatch: extra={set(xset)-set(expected)} '
                                    f'missing={set(expected)-set(xset)}')
    for bad in FORBIDDEN_IN_X:
        assert bad not in xset, f'LEAK: forbidden column {bad!r} in feature matrix'


def haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((r2 - r1) / 2)**2 +
         np.cos(r1) * np.cos(r2) * np.sin(np.deg2rad(lon2 - lon1) / 2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def add_dist(df, lat, lon):
    lats, lons = df[lat].values, df[lon].values
    hub = np.stack([haversine_km(lats, lons, la, lo) for la, lo in CHINA_HUBS.values()], axis=1)
    prt = np.stack([haversine_km(lats, lons, la, lo) for la, lo in CHINA_PORTS.values()], axis=1)
    df = df.copy()
    df['dist_nearest_hub'] = hub.min(axis=1)
    df['dist_nearest_port'] = prt.min(axis=1)
    return df


def m_level(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return pearsonr(a, p)[0], r2_score(a, p), float(np.sqrt(np.mean((a - p)**2)))


def m_growth_pp(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return pearsonr(a, p)[0], r2_score(a, p), float(np.sqrt(np.mean((a - p)**2)) * 100)


def m_logdiff(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return pearsonr(a, p)[0], r2_score(a, p), float(np.sqrt(np.mean((a - p)**2)))


rows = []
def add(block, exercise, resolution, scale, n, r, r2, rmse, unit, letter_ref):
    rows.append(dict(block=block, exercise=exercise, resolution=resolution, scale=scale,
                     n=int(n), r=round(float(r), 4), R2=round(float(r2), 4),
                     RMSE=round(float(rmse), 4), RMSE_unit=unit, letter_ref=letter_ref))


# ══════════════════════════════════════════════════════════════════════════════
# PLANT panel (pipeline identical to scripts 17 / 21)
# ══════════════════════════════════════════════════════════════════════════════
print('Loading PLANT panel ...')
signals = pd.read_csv(SIGNALS_PATH).drop(columns=['PM1_MEAN'], errors='ignore')
gridprod = pd.read_csv(GRIDPROD_PATH)
gridprod = gridprod.drop_duplicates(subset=['IDCode', 'name_prod', 'Year', 'Month'])  # dedup guard
bf = pd.read_excel(GEM_PATH, sheet_name='Blast furnaces', header=0)
bf_ids = set(bf.iloc[:, 0].dropna().astype(str).unique())
gem_tech = gridprod[['name_prod', 'GEMPlantID']].drop_duplicates('name_prod').copy()
gem_tech['is_bof'] = gem_tech['GEMPlantID'].apply(lambda x: 1 if str(x) in bf_ids else 0)
temp = gridprod[['IDCode', 'Longitude', 'Latitude', 'name_prod', 'Year', 'Month', 'Steel_Prod']].copy()
merged = pd.merge(signals, temp, on=['IDCode', 'Year', 'Month'], how='left')
agg = {c: 'mean' for c in POLLUTANTS}
grp = (merged.groupby(['name_prod', 'Year', 'Month'])
       .agg({**agg, 'Longitude': 'mean', 'Latitude': 'mean', 'Steel_Prod': 'first'}).reset_index())
grp.rename(columns={p: f'Plant_{p}' for p in POLLUTANTS}, inplace=True)
grp = grp.drop_duplicates(subset=['name_prod', 'Year', 'Month'])
grp['plant_id'] = grp['name_prod'].factorize()[0] + 1
grp = grp.merge(gem_tech[['name_prod', 'is_bof']], on='name_prod', how='left')
grp['is_bof'] = grp['is_bof'].fillna(0).astype(int)
grp['month_sin'] = np.sin(2 * np.pi * grp['Month'] / 12)
grp['month_cos'] = np.cos(2 * np.pi * grp['Month'] / 12)
grp = grp.sort_values(['plant_id', 'Year', 'Month']).reset_index(drop=True)
for lag in range(1, 4):
    for col in PLANT_POLLUTANTS:
        grp[f'{col}_lag{lag}'] = grp.groupby('plant_id')[col].shift(lag)
grp = add_dist(grp, 'Latitude', 'Longitude')
plant = grp.dropna(subset=PLANT_FEATURES + ['Steel_Prod', 'Year']).copy()
plant = plant[plant['Steel_Prod'] > 0].reset_index(drop=True)
print(f'  plant: {plant["name_prod"].nunique()} plants, {len(plant)} rows')

# ══════════════════════════════════════════════════════════════════════════════
# GRID panel (pipeline identical to script 12, with dedup guard)
# ══════════════════════════════════════════════════════════════════════════════
print('Loading GRID panel ...')
grid = pd.read_csv(GRID_DATA).drop(columns=['Unnamed: 0', 'Polygon_ID'], errors='ignore')
grid = grid.drop_duplicates(subset=['IDCode', 'Year', 'Month'])                       # dedup guard
grid['month_sin'] = np.sin(2 * np.pi * grid['Month'] / 12)
grid['month_cos'] = np.cos(2 * np.pi * grid['Month'] / 12)
grid = grid.sort_values(['IDCode', 'Year', 'Month']).reset_index(drop=True)
for lag in range(1, 4):
    for col in POLLUTANTS:
        grid[f'{col}_lag{lag}'] = grid.groupby('IDCode')[col].shift(lag)
grid = grid.dropna()
grid = add_dist(grid, 'Centroid_Lat', 'Centroid_Long').reset_index(drop=True)
print(f'  grid: {grid["IDCode"].nunique()} cells, {len(grid)} rows')


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — cross-sectional level, random 80/20 split (LOG scale)
# ══════════════════════════════════════════════════════════════════════════════
def crosssec_level(df, feats, ycol, logfn, scale_label, res, ref):
    Xtr, Xte, itr, ite = train_test_split(df[feats], df.index, test_size=0.2, random_state=SEED)
    assert_no_leak(Xtr.columns, feats)                              # leak guard
    m = xgb.XGBRegressor(**(GRID_XGB if ycol == 'GridProd_Steel_tot' else PLANT_XGB))
    m.fit(Xtr.values, logfn(df.loc[itr, ycol].values))
    pred_log = m.predict(Xte.values)
    actual_log = logfn(df.loc[ite, ycol].values)
    add('1_crosssec_level', 'crosssec_8020', res, 'log', len(ite),
        *m_level(actual_log, pred_log), scale_label, ref)

crosssec_level(grid, GRID_FEATURES, 'GridProd_Steel_tot', np.log, 'log output', 'grid',
               'Table 8 row 1 (grid CS level)')
crosssec_level(plant, PLANT_FEATURES, 'Steel_Prod', np.log1p, 'log1p output', 'plant',
               'Table 8 row 2 (plant CS level)')


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — held-out year 2022 level (train 2019-2021), ORIGINAL scale (Fig 7a)
# ══════════════════════════════════════════════════════════════════════════════
def temporal_level(df, feats, ycol, logfn, invfn, res, ref):
    tr = df[df['Year'].isin([2019, 2020, 2021])]
    te = df[df['Year'] == 2022]
    assert_no_leak(tr[feats].columns, feats)                       # leak guard
    m = xgb.XGBRegressor(**(GRID_XGB if ycol == 'GridProd_Steel_tot' else PLANT_XGB))
    m.fit(tr[feats].values, logfn(tr[ycol].values))
    pred_orig = invfn(m.predict(te[feats].values))
    add('2_temporal_level', 'temporal_2022', res, 'orig', len(te),
        *m_level(te[ycol].values, pred_orig), '10k MT/mo (orig)', ref)
    return m

mg_temporal = temporal_level(grid, GRID_FEATURES, 'GridProd_Steel_tot', np.log, np.exp,
                             'grid', 'Table 8 row 3 (grid 2022 level)')
mp_temporal = temporal_level(plant, PLANT_FEATURES, 'Steel_Prod', np.log1p, np.expm1,
                             'plant', 'Table 8 row 4 (plant 2022 level)')


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — growth, annual LOG-DIFFERENCE, 5-fold within-sample OOF-differencing
# ══════════════════════════════════════════════════════════════════════════════
def kfold_oof_levels(df, feats, ycol, logfn, invfn):
    """5-fold shuffle OOF predicted LEVELS (clamped >=0)."""
    oof = np.full(len(df), np.nan)
    X = df[feats].values
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tri, tei in kf.split(X):
        assert_no_leak(df[feats].columns, feats)                   # leak guard
        m = xgb.XGBRegressor(**(GRID_XGB if ycol == 'GridProd_Steel_tot' else PLANT_XGB))
        m.fit(X[tri], logfn(df[ycol].values[tri]))
        oof[tei] = np.clip(invfn(m.predict(X[tei])), 0, None)
    return oof


def unit_annual_logdiff(df, unit, ycol, oofcol):
    """Unit annual sums (over observed months) -> adjacent-year log-difference."""
    ann = (df.groupby([unit, 'Year']).agg(a=(ycol, 'sum'), p=(oofcol, 'sum'))
           .reset_index().sort_values([unit, 'Year']))
    base = ann.rename(columns={'a': 'ba', 'p': 'bp', 'Year': 'Y0'})
    base['Year'] = base['Y0'] + 1
    m = ann.merge(base[[unit, 'Year', 'ba', 'bp']], on=[unit, 'Year'], how='inner')
    p1 = np.percentile(m['ba'], 1)                                 # 1st-pct small-base guard
    m = m[m['ba'] >= p1]
    m = m[(m['a'] > 0) & (m['ba'] > 0) & (m['p'] > 0) & (m['bp'] > 0)]  # log defined
    m['dla'] = np.log(m['a']) - np.log(m['ba'])
    m['dlp'] = np.log(m['p']) - np.log(m['bp'])
    return m


def national_monthly_growth(df, unit, ycol, oofcol):
    """Balanced-panel monthly YoY growth of the NATIONAL total (percent)."""
    nwin = len(df[['Year', 'Month']].drop_duplicates())
    cnt = df.groupby(unit).size()
    bal = cnt[cnt == nwin].index
    if len(bal) < 10:
        thr = int(np.ceil(0.95 * nwin)); bal = cnt[cnt >= thr].index
    sub = df[df[unit].isin(bal)]
    a = sub.groupby(['Year', 'Month']).agg(a=(ycol, 'sum'), p=(oofcol, 'sum')).reset_index()
    base = a.rename(columns={'a': 'ba', 'p': 'bp'}); base['Year'] += 1
    nat = a.merge(base, on=['Year', 'Month'], how='inner')
    nat['ga'] = nat['a'] / nat['ba'] - 1
    nat['gp'] = nat['p'] / nat['bp'] - 1
    return nat, len(bal)


print('5-fold OOF (grid) ...')
grid['oof'] = kfold_oof_levels(grid, GRID_FEATURES, 'GridProd_Steel_tot', np.log, np.exp)
print('5-fold OOF (plant) ...')
plant['oof'] = kfold_oof_levels(plant, PLANT_FEATURES, 'Steel_Prod', np.log1p, np.expm1)

gd = unit_annual_logdiff(grid, 'IDCode', 'GridProd_Steel_tot', 'oof')
add('3_growth_logdiff', 'kfold_annual_logdiff', 'grid', 'log', len(gd),
    *m_logdiff(gd['dla'], gd['dlp']), 'log points', 'Table 8 row 5 (grid unit log-diff)')
pd_ = unit_annual_logdiff(plant, 'name_prod', 'Steel_Prod', 'oof')
add('3_growth_logdiff', 'kfold_annual_logdiff', 'plant', 'log', len(pd_),
    *m_logdiff(pd_['dla'], pd_['dlp']), 'log points', 'Table 8 row 6 (plant unit log-diff)')

nat, nbal = national_monthly_growth(grid, 'IDCode', 'GridProd_Steel_tot', 'oof')
add('3_growth_logdiff', 'kfold_national_agg', 'national_grid', 'pct', len(nat),
    *m_growth_pp(nat['ga'], nat['gp']), 'pct points', 'Table 8 row 7 (grid national agg)')
print(f'  national balanced grid panel: {nbal} cells, {len(nat)} YoY months')


# ══════════════════════════════════════════════════════════════════════════════
# Write + print
# ══════════════════════════════════════════════════════════════════════════════
tab = pd.DataFrame(rows)
p1 = os.path.join(OUT_TAB, 'table_validation_summary_letter.csv')
tab.to_csv(p1, index=False, encoding='utf-8-sig')
for _ in range(8):
    try:
        shutil.copy2(p1, os.path.join(OUT_RL, 'table_validation_summary_letter.csv')); break
    except PermissionError:
        time.sleep(1.0)

# expected Table-8 values for a pass/fail check (rounded as printed in the letter)
EXPECT = {
    'Table 8 row 1 (grid CS level)':        (0.94, 0.875, 0.54),
    'Table 8 row 2 (plant CS level)':       (0.93, 0.861, 0.30),
    'Table 8 row 3 (grid 2022 level)':      (0.82, 0.60, 6.2),
    'Table 8 row 4 (plant 2022 level)':     (0.91, 0.79, 20.7),
    'Table 8 row 5 (grid unit log-diff)':   (0.85, 0.68, 0.21),
    'Table 8 row 6 (plant unit log-diff)':  (0.78, 0.58, 0.20),
    'Table 8 row 7 (grid national agg)':    (0.85, 0.61, 6.0),
}
print('\n' + '=' * 96)
print(f'{"letter_ref":40s} {"n":>6s} {"r":>7s} {"R2":>7s} {"RMSE":>8s}  {"unit":14s}  check')
print('-' * 96)
tol = 0.02
for _, r in tab.iterrows():
    er, er2, ermse = EXPECT[r['letter_ref']]
    ok = (abs(r['r']-er) <= tol and abs(r['R2']-er2) <= tol
          and abs(r['RMSE']-ermse) <= max(0.02, 0.03*abs(ermse)))
    print(f'{r["letter_ref"]:40s} {r["n"]:6d} {r["r"]:7.3f} {r["R2"]:7.3f} {r["RMSE"]:8.3f}  '
          f'{r["RMSE_unit"]:14s}  {"PASS" if ok else "CHECK"}  '
          f'(letter {er}/{er2}/{ermse})')
print('=' * 96)
print(f'Saved: {p1} (+ output_rl copy)')
