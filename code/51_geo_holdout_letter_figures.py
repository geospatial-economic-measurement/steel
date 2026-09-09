
import os, sys, io, warnings, shutil
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_RL, need

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Paths ──────────────────────────────────────────────────────────────────────
SIGNALS_PATH  = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH      = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')
REPL_OUT = OUT_RL
OUT_RL   = OUT_RL
for d in (REPL_OUT, OUT_RL):
    os.makedirs(d, exist_ok=True)

EARTH_R = 6371.0
LINTHRESH = 25
JMP_BLUE = '#00468B'; JMP_RED = '#ED0000'; JMP_GREY = '#737373'

CHINA_HUBS = {'Tangshan': (39.6, 118.2), 'Wuhan': (30.6, 114.3), 'Anshan': (41.1, 122.8),
              'Rizhao': (35.4, 119.5), 'Baotou': (40.7, 109.8)}
CHINA_PORTS = {'Shanghai': (31.23, 121.47), 'Tianjin': (38.98, 117.72), 'Qingdao': (36.07, 120.38),
               'Ningbo': (29.87, 121.55), 'Guangzhou': (23.10, 113.43), 'Dalian': (38.92, 121.65),
               'Lianyungang': (34.75, 119.45), 'Yingkou': (40.67, 122.23)}
POLLUTANTS = ['CO_MEAN', 'NO2_MEAN', 'SO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN',
              'O3_MEAN', 'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN']
PLANT_POLLUTANTS = [f'Plant_{p}' for p in POLLUTANTS]
FEATURE_COLS = (PLANT_POLLUTANTS + [f'{p}_lag{k}' for k in (1, 2, 3) for p in PLANT_POLLUTANTS]
                + ['month_sin', 'month_cos', 'dist_nearest_hub', 'dist_nearest_port', 'is_bof'])
XGB_PARAMS = dict(colsample_bytree=0.8, learning_rate=0.1, max_depth=6, n_estimators=300,
                  subsample=0.6, alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0)


def haversine_km(lat1, lon1, lat2, lon2):
    r1, r2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((r2 - r1) / 2)**2 +
         np.cos(r1) * np.cos(r2) * np.sin(np.deg2rad(lon2 - lon1) / 2)**2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def add_dist(df):
    lats, lons = df['Latitude'].values, df['Longitude'].values
    hub = np.stack([haversine_km(lats, lons, la, lo) for la, lo in CHINA_HUBS.values()], axis=1)
    prt = np.stack([haversine_km(lats, lons, la, lo) for la, lo in CHINA_PORTS.values()], axis=1)
    df = df.copy()
    df['dist_nearest_hub'] = hub.min(axis=1)
    df['dist_nearest_port'] = prt.min(axis=1)
    return df


def save_both(fig, name):
    for d in (REPL_OUT, OUT_RL):
        p = os.path.join(d, f'{name}.pdf')
        fig.savefig(p, bbox_inches='tight', dpi=300)
        print(f'  Saved: {p}')


# ── Load plant panel (identical pipeline to script 21) ──────────────────────────
print('Loading plant panel ...')
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
grp = add_dist(grp)
data = grp.dropna(subset=FEATURE_COLS + ['Steel_Prod', 'Year', 'Latitude']).copy()
data = data[data['Steel_Prod'] > 0].reset_index(drop=True)
print(f'  plant panel: {data["name_prod"].nunique()} plants, {len(data)} rows')

north = data[data['Latitude'] >= 35].copy()
south = data[data['Latitude'] < 35].copy()


def fit_predict(train_df, test_df):
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(train_df[FEATURE_COLS].values, np.log1p(train_df['Steel_Prod'].values))
    t = test_df.copy()
    t['pred'] = np.expm1(m.predict(t[FEATURE_COLS].values))
    return t


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 (R1.2 Figure 1): two-panel full-sample symlog growth scatter
# ══════════════════════════════════════════════════════════════════════════════
def unmatched_annual_growth(t):
    ann = (t.groupby(['name_prod', 'Year'])[['Steel_Prod', 'pred']]
           .sum().reset_index().sort_values(['name_prod', 'Year']))
    ann['actual_gr'] = ann.groupby('name_prod')['Steel_Prod'].pct_change()
    ann['pred_gr'] = ann.groupby('name_prod')['pred'].pct_change()
    ann = ann.dropna(subset=['actual_gr', 'pred_gr'])
    ann = ann[np.isfinite(ann['actual_gr']) & np.isfinite(ann['pred_gr'])].copy()
    return ann


def symlog_scatter(ax, ann):
    a = ann['actual_gr'].values * 100
    p = ann['pred_gr'].values * 100
    r = pearsonr(a / 100, p / 100)[0]
    r2 = r2_score(a / 100, p / 100)
    n = len(ann)
    ax.scatter(a, p, alpha=0.6, color='blue', s=25, zorder=3)
    ext = max(np.abs(a).max(), np.abs(p).max()) * 1.10
    lo, hi = -ext, ext
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, zorder=2, label='45° line')
    ax.axhline(0, color=JMP_GREY, lw=0.8, ls=':'); ax.axvline(0, color=JMP_GREY, lw=0.8, ls=':')
    ax.set_xscale('symlog', linthresh=LINTHRESH)
    ax.set_yscale('symlog', linthresh=LINTHRESH)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ticks = [t for t in (-1000, -500, -100, -50, -25, 0, 25, 50, 100, 500, 1000) if lo <= t <= hi]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        axis.set_minor_locator(NullLocator())
    ax.set_xlabel('Actual YoY Growth Rate (%)', fontsize=12)
    ax.set_ylabel('Predicted YoY Growth Rate (%)', fontsize=12)
    ax.text(0.97, 0.05, f'$R^2$ = {r2:.3f}\n$r$ = {r:.3f}\nn = {n}',
            transform=ax.transAxes, fontsize=11, ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    return r, r2, n


print('\nFigure 1 (two-panel full-sample symlog growth) ...')
ann_s = unmatched_annual_growth(fit_predict(north, south))   # South held out
ann_n = unmatched_annual_growth(fit_predict(south, north))   # North mirror
fig1, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
rs = symlog_scatter(axes[0], ann_s)
rn = symlog_scatter(axes[1], ann_n)
axes[0].set_title('(a) Southern China held out', fontsize=12)
axes[1].set_title('(b) Northern China held out (mirror)', fontsize=12)
plt.tight_layout()
save_both(fig1, 'fig_geo_holdout_2dir_full_symlog')
plt.close(fig1)
print(f'  (a) South: r={rs[0]:.3f} R2={rs[1]:.3f} n={rs[2]}')
print(f'  (b) North: r={rn[0]:.3f} R2={rn[1]:.3f} n={rn[2]}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 (R1.2 Figure 2): regional aggregate monthly YoY growth, 2x2
# ══════════════════════════════════════════════════════════════════════════════
def regional_aggregate(t):
    """Balanced plant panel; national total; monthly YoY growth of the total."""
    nwin = len(t[['Year', 'Month']].drop_duplicates())
    cnt = t.groupby('name_prod').size()
    bal = cnt[cnt == nwin].index
    if len(bal) < 10:
        thr = int(np.ceil(0.95 * nwin)); bal = cnt[cnt >= thr].index
    sub = t[t['name_prod'].isin(bal)]
    agg = sub.groupby(['Year', 'Month'])[['Steel_Prod', 'pred']].sum().reset_index()
    base = agg.rename(columns={'Steel_Prod': 'ba', 'pred': 'bp'}); base['Year'] += 1
    nat = agg.merge(base, on=['Year', 'Month'], how='inner')
    nat['g_actual'] = nat['Steel_Prod'] / nat['ba'] - 1
    nat['g_pred'] = nat['pred'] / nat['bp'] - 1
    nat['date'] = pd.to_datetime(dict(year=nat['Year'], month=nat['Month'], day=1))
    return nat.sort_values('date').reset_index(drop=True), len(bal)


print('\nFigure 2 (regional aggregate, 2x2) ...')
nat_s, nb_s = regional_aggregate(fit_predict(north, south))   # South held out
nat_n, nb_n = regional_aggregate(fit_predict(south, north))   # North held out
fig2, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
specs = [(nat_s, nb_s, 'Southern China', 0), (nat_n, nb_n, 'Northern China', 1)]
for nat, nb, region, col in specs:
    r = pearsonr(nat['g_actual'], nat['g_pred'])[0]
    r2 = r2_score(nat['g_actual'], nat['g_pred'])
    # top row: time series
    ax = axes[0, col]
    ax.plot(nat['date'], nat['g_actual'] * 100, color='0.25', marker='o', ms=4, lw=1.5,
            label='Actual (balanced panel)')
    ax.plot(nat['date'], nat['g_pred'] * 100, color=JMP_BLUE, marker='s', ms=4, lw=1.5,
            label='Predicted (geographic holdout)')
    ax.axhline(0, color=JMP_GREY, lw=0.8, ls=':')
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Regional YoY Growth Rate (%)', fontsize=11)
    ax.set_title(f'({chr(97+col*2)}) {region}: aggregate YoY growth by month', fontsize=12)
    ax.legend(fontsize=8.5, framealpha=0.85)
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
    # bottom row: scatter
    ax = axes[1, col]
    ax.scatter(nat['g_actual'] * 100, nat['g_pred'] * 100, color=JMP_BLUE, marker='s',
               s=35, alpha=0.8, label='Monthly observation')
    allv = np.concatenate([nat['g_actual'], nat['g_pred']]) * 100
    lo, hi = allv.min() - 3, allv.max() + 3
    ax.plot([lo, hi], [lo, hi], color=JMP_RED, ls='--', lw=1.5, label='45° line')
    ax.axhline(0, color=JMP_GREY, lw=0.8, ls=':'); ax.axvline(0, color=JMP_GREY, lw=0.8, ls=':')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual Regional YoY Growth Rate (%)', fontsize=11)
    ax.set_ylabel('Predicted Regional YoY Growth Rate (%)', fontsize=11)
    ax.set_title(f'({chr(97+col*2+1)}) {region}: predicted vs. actual', fontsize=12)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.85)
    ax.text(0.97, 0.05, f'$R^2$ = {r2:.3f}\n$r$ = {r:.3f}\nn = {len(nat)} months',
            transform=ax.transAxes, fontsize=11, ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    print(f'  {region}: r={r:.3f} R2={r2:.3f} n={len(nat)} balanced_plants={nb}')
plt.tight_layout()
save_both(fig2, 'fig_geo_holdout_agg')
plt.close(fig2)

print('\nDone.')
