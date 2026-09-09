import os, sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
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
OUT_FIG       = OUT_FIG
os.makedirs(OUT_FIG, exist_ok=True)

EARTH_R = 6371.0

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

XGB_PARAMS = dict(
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

def add_dist_features(df):
    lats, lons = df['Latitude'].values, df['Longitude'].values
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
        p = os.path.join(OUT_RL, f'{name}.{ext}')
        try:
            fig.savefig(p, bbox_inches='tight', dpi=800)
            print(f'  Saved: {p}')
        except PermissionError:
            p2 = os.path.join(OUT_RL, f'{name}_v2.{ext}')
            fig.savefig(p2, bbox_inches='tight', dpi=800)
            print(f'  Saved (alt): {p2}')

# ── Load and prepare data (same pipeline as fig6) ─────────────────────────────
print('Loading data ...')
signals  = pd.read_csv(SIGNALS_PATH)
signals  = signals.drop(columns=['PM1_MEAN'], errors='ignore')
gridprod = pd.read_csv(GRIDPROD_PATH)

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
print(f'  Plants: {grouped["name_prod"].nunique()}, rows: {len(grouped)}')

grouped['month_sin'] = np.sin(2 * np.pi * grouped['Month'] / 12)
grouped['month_cos'] = np.cos(2 * np.pi * grouped['Month'] / 12)
grouped = grouped.sort_values(['plant_id', 'Year', 'Month']).reset_index(drop=True)
for lag in range(1, 4):
    for col in PLANT_POLLUTANTS:
        grouped[f'{col}_lag{lag}'] = grouped.groupby('plant_id')[col].shift(lag)
grouped = add_dist_features(grouped)

data = grouped.dropna(subset=FEATURE_COLS + ['Steel_Prod', 'Year', 'Latitude']).copy()
data = data[data['Steel_Prod'] > 0].reset_index(drop=True)
print(f'  After dropna: {data["name_prod"].nunique()} plants, {len(data)} rows')

TICKS = [0.1, 1, 10, 100]

def scatter_panel(ax, y_actual, y_pred, title):
    ax.scatter(y_actual, y_pred, alpha=0.6, color='blue', s=25,
               label='Plant-month observation')
    lo = min(y_actual.min(), y_pred.min()) * 0.8
    hi = max(y_actual.max(), y_pred.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='45° line (perfect fit)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(TICKS); ax.set_xticklabels(TICKS)
    ax.set_yticks(TICKS); ax.set_yticklabels(TICKS)
    ax.set_xlabel('Actual Steel Output (10,000 MT/month)', fontsize=12)
    ax.set_ylabel('Predicted Steel Output (10,000 MT/month)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.8)
    r2   = r2_score(y_actual, y_pred)
    rho  = spearmanr(y_actual, y_pred).statistic
    ax.text(0.97, 0.05, f'$R^2$ = {r2:.3f}\n$\\rho$ = {rho:.3f}',
            transform=ax.transAxes, fontsize=11, ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    return r2, rho

# ── Panel A: Temporal holdout — train 2019–2021, test 2022 ────────────────────
print('\nPanel A: Temporal holdout (train 2019-2021, test 2022) ...')
train_a = data[data['Year'].isin([2019, 2020, 2021])]
test_a  = data[data['Year'] == 2022]

X_train_a = train_a[FEATURE_COLS].values
y_train_a = np.log1p(train_a['Steel_Prod'].values)
X_test_a  = test_a[FEATURE_COLS].values
y_test_a  = test_a['Steel_Prod'].values      # original scale for plotting

model_a = xgb.XGBRegressor(**XGB_PARAMS)
model_a.fit(X_train_a, y_train_a)
y_pred_a = np.expm1(model_a.predict(X_test_a))

r2_a  = r2_score(y_test_a, y_pred_a)
rho_a = spearmanr(y_test_a, y_pred_a).statistic
print(f'  2022 holdout: R²={r2_a:.4f}, Spearman rho={rho_a:.4f}')
print(f'  Train N={len(train_a)}, Test N={len(test_a)}')

fig_a, ax_a = plt.subplots(figsize=(6, 6), dpi=300)
scatter_panel(ax_a, y_test_a, y_pred_a,
              'Predicted vs. Actual Steel Output\n(2022 Temporal Holdout)')
plt.tight_layout()
save_fig(fig_a, 'fig_temporal_holdout_2022_diagnostic')
plt.close()

import shutil

OUT_TAB       = OUT_TAB
OUT_RL        = OUT_RL
REPL_OUT      = OUT_RL
PAPER_FIG_DIR = OUT_FIG
for d in (OUT_TAB, OUT_RL, REPL_OUT):
    os.makedirs(d, exist_ok=True)

JMP_BLUE = '#00468B'
JMP_RED  = '#ED0000'

MIN_MATCHED_MONTHS = 8  
TRIM = 2.0              


def matched_month_annual_growth(df):
    """Matched-month annual growth pairs per plant (script 27, computation A2).

    For each plant and adjacent year pair (Y-1, Y), sum Steel_Prod and pred
    over only the calendar months observed in BOTH years (post-lag-warm-up
    rows); require >= MIN_MATCHED_MONTHS matched months; small-base guard at
    the 1st percentile of matched annual sums (base and current years pooled).
    Returns (ann, info dict).
    """
    cur  = df[['name_prod', 'Year', 'Month', 'Steel_Prod', 'pred']].copy()
    base = cur.rename(columns={'Steel_Prod': 'base_actual', 'pred': 'base_pred'})
    base['Year'] = base['Year'] + 1          # base year Y-1 matched to year Y
    pairs = cur.merge(base, on=['name_prod', 'Year', 'Month'], how='inner')
    pairs = pairs.rename(columns={'Steel_Prod': 'cur_actual', 'pred': 'cur_pred'})

    ann = (pairs.groupby(['name_prod', 'Year'])
                .agg(n_matched_months=('Month', 'nunique'),
                     cur_actual=('cur_actual', 'sum'),
                     base_actual=('base_actual', 'sum'),
                     cur_pred=('cur_pred', 'sum'),
                     base_pred=('base_pred', 'sum'))
                .reset_index())
    n_raw = len(ann)

    insufficient = ann['n_matched_months'] < MIN_MATCHED_MONTHS
    n_insuff = int(insufficient.sum())
    ann = ann[~insufficient].copy()

    p1_ann = np.percentile(np.concatenate([ann['cur_actual'].values,
                                           ann['base_actual'].values]), 1)
    small = ann['base_actual'] < p1_ann
    n_small = int(small.sum())
    ann = ann[~small].copy()

    ann['actual_gr'] = ann['cur_actual'] / ann['base_actual'] - 1
    ann['pred_gr']   = ann['cur_pred']   / ann['base_pred']   - 1
    fin = np.isfinite(ann['actual_gr']) & np.isfinite(ann['pred_gr'])
    n_nonfinite = int((~fin).sum())
    ann = ann[fin].copy()

    return ann, dict(n_raw=n_raw, n_insuff=n_insuff, n_small=n_small,
                     n_nonfinite=n_nonfinite, p1_ann=p1_ann)


def regional_aggregate_growth(df, region_label):
    """Regional monthly YoY growth of the balanced-panel total.

    Balanced panel = plants observed in every month of the region's common
    post-warm-up window; if < 10 plants, relaxed to >= 95% of months (noted).
    g_t = T_t / T_{t-12} - 1 for actual and predicted totals.
    """
    n_window = len(df[['Year', 'Month']].drop_duplicates())
    counts = df.groupby('name_prod').size()   # panel deduped -> distinct months
    balanced = counts[counts == n_window].index
    note = f'fully balanced ({n_window}/{n_window} months)'
    if len(balanced) < 10:
        thr = int(np.ceil(0.95 * n_window))
        balanced = counts[counts >= thr].index
        note = (f'RELAXED to >=95% of months (>= {thr}/{n_window}); fully '
                f'balanced set had < 10 plants')
    print(f'  [{region_label}] aggregate panel: {len(balanced)} plants, {note}')

    sub = df[df['name_prod'].isin(balanced)]
    agg = (sub.groupby(['Year', 'Month'])[['Steel_Prod', 'pred']]
              .sum().reset_index())
    base = agg.rename(columns={'Steel_Prod': 'base_actual', 'pred': 'base_pred'})
    base['Year'] = base['Year'] + 1
    nat = agg.merge(base, on=['Year', 'Month'], how='inner')
    nat['g_actual'] = nat['Steel_Prod'] / nat['base_actual'] - 1
    nat['g_pred']   = nat['pred']       / nat['base_pred']   - 1
    nat['date'] = pd.to_datetime(dict(year=nat['Year'], month=nat['Month'], day=1))
    nat = nat.sort_values('date').reset_index(drop=True)
    return nat, len(balanced), note


def copy_file(src, dst_dir, dst_name=None):
    dst = os.path.join(dst_dir, dst_name or os.path.basename(src))
    try:
        shutil.copy2(src, dst)
        print(f'  Copied: {dst}')
    except (PermissionError, OSError) as e:
        print(f'  COPY FAILED ({e}): {dst}')


def growth_scatter_figure(ann_trim, r2_trim, r_trim, name, title):
    """Trimmed-consistent scatter: displayed points = metric sample."""
    actual_pct = ann_trim['actual_gr'].values * 100
    pred_pct   = ann_trim['pred_gr'].values   * 100
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.scatter(actual_pct, pred_pct, alpha=0.6, color='blue', s=25,
               label='Plant-year observation')
    lo = min(actual_pct.min(), pred_pct.min()) - 5
    hi = max(actual_pct.max(), pred_pct.max()) + 5
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='45° line (perfect fit)')
    ax.axhline(0, color='gray', lw=0.8, linestyle=':')
    ax.axvline(0, color='gray', lw=0.8, linestyle=':')
    ax.set_xlabel('Actual Annual Growth Rate (%)', fontsize=12)
    ax.set_ylabel('Predicted Annual Growth Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.8)
    # Stats box shows the SAME-SAMPLE metrics as the displayed points
    ax.text(0.97, 0.05,
            f'$R^2$ = {r2_trim:.3f}\n$r$ = {r_trim:.3f}\nn = {len(ann_trim)}',
            transform=ax.transAxes, fontsize=11, ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.tight_layout()
    save_fig(fig, name)
    plt.close(fig)


def aggregate_figure(nat, r_agg, r2_agg, name, region_label):
    """Two-panel regional aggregate growth figure: time series + scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    ax = axes[0]
    ax.plot(nat['date'], nat['g_actual'] * 100, color='0.25', marker='o',
            ms=4, lw=1.5, label='Actual (balanced panel)')
    ax.plot(nat['date'], nat['g_pred'] * 100, color=JMP_BLUE, marker='s',
            ms=4, lw=1.5, label='Predicted (geographic holdout)')
    ax.axhline(0, color='gray', lw=0.8, linestyle=':')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Regional YoY Growth Rate (%)', fontsize=12)
    ax.set_title(f'(a) {region_label} aggregate YoY growth by month', fontsize=13)
    ax.legend(fontsize=9, framealpha=0.8)
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)

    ax = axes[1]
    ax.scatter(nat['g_actual'] * 100, nat['g_pred'] * 100, color=JMP_BLUE,
               marker='s', s=35, alpha=0.8, label='Monthly observation')
    all_vals = np.concatenate([nat['g_actual'], nat['g_pred']]) * 100
    lo, hi = all_vals.min() - 3, all_vals.max() + 3
    ax.plot([lo, hi], [lo, hi], color=JMP_RED, linestyle='--', lw=1.5,
            label='45° line (perfect fit)')
    ax.axhline(0, color='gray', lw=0.8, linestyle=':')
    ax.axvline(0, color='gray', lw=0.8, linestyle=':')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual Regional YoY Growth Rate (%)', fontsize=12)
    ax.set_ylabel('Predicted Regional YoY Growth Rate (%)', fontsize=12)
    ax.set_title(f'(b) Predicted vs. actual {region_label.lower()} growth',
                 fontsize=13)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.8)
    ax.text(0.97, 0.05,
            f'$R^2$ = {r2_agg:.3f}\n$r$ = {r_agg:.3f}\nn = {len(nat)} months',
            transform=ax.transAxes, fontsize=11, ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.tight_layout()
    save_fig(fig, name)
    plt.close(fig)


def make_supp_table(ann):
    """Supplementary table with a flag for rows trimmed from metrics/figure."""
    supp = ann[['name_prod', 'Year', 'n_matched_months',
                'cur_actual', 'cur_pred', 'base_actual', 'base_pred',
                'actual_gr', 'pred_gr']].copy()
    supp['Trimmed_gt200pct'] = (supp['actual_gr'].abs() > TRIM).astype(int)
    supp['Actual_YoY_Growth_pct']    = (supp['actual_gr'] * 100).round(2)
    supp['Predicted_YoY_Growth_pct'] = (supp['pred_gr']   * 100).round(2)
    supp = supp.drop(columns=['actual_gr', 'pred_gr'])
    supp = supp.rename(columns={
        'name_prod':   'Plant',
        'n_matched_months': 'Matched_Months',
        'cur_actual':  'Actual_Output_10kMT_matched',
        'cur_pred':    'Predicted_Output_10kMT_matched',
        'base_actual': 'Actual_BaseYear_Output_10kMT_matched',
        'base_pred':   'Predicted_BaseYear_Output_10kMT_matched'})
    for c in ['Actual_Output_10kMT_matched', 'Predicted_Output_10kMT_matched',
              'Actual_BaseYear_Output_10kMT_matched',
              'Predicted_BaseYear_Output_10kMT_matched']:
        supp[c] = supp[c].round(2)
    return supp.sort_values(['Plant', 'Year']).reset_index(drop=True)


def run_geo_holdout(direction, train_df, test_df, scatter_name, scatter_title,
                    agg_name, region_label):
    """Full geographic-holdout exercise for one split direction."""
    print(f'\n{"-"*70}\nGeographic holdout [{direction}]')
    print(f'  Train: {train_df["name_prod"].nunique()} plants, {len(train_df)} rows')
    print(f'  Test:  {test_df["name_prod"].nunique()} plants, {len(test_df)} rows')

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(train_df[FEATURE_COLS].values,
              np.log1p(train_df['Steel_Prod'].values))
    test_df = test_df.copy()
    test_df['pred'] = np.expm1(model.predict(test_df[FEATURE_COLS].values))

    # A. Matched-month annual growth
    ann, info = matched_month_annual_growth(test_df)
    print(f'  Matched-month annual pairs: {info["n_raw"]} raw; '
          f'dropped {info["n_insuff"]} with < {MIN_MATCHED_MONTHS} matched months; '
          f'dropped {info["n_small"]} small-base (base-year matched sum < 1st pct '
          f'= {info["p1_ann"]:.4f}); {info["n_nonfinite"]} non-finite; '
          f'{len(ann)} retained')

    # B. Consistent metrics/display: trim |actual growth| > 200% from BOTH
    trimmed_mask = ann['actual_gr'].abs() > TRIM
    n_excluded = int(trimmed_mask.sum())
    ann_trim = ann[~trimmed_mask].copy()

    r2_full = r2_score(ann['actual_gr'], ann['pred_gr'])
    r_full  = pearsonr(ann['actual_gr'], ann['pred_gr'])[0]
    r2_trim = r2_score(ann_trim['actual_gr'], ann_trim['pred_gr'])
    r_trim  = pearsonr(ann_trim['actual_gr'], ann_trim['pred_gr'])[0]
    print(f'  FULL SAMPLE (incl. outliers, record only): '
          f'R2={r2_full:.4f}, r={r_full:.4f}, n={len(ann)}')
    print(f'  TRIMMED (|actual growth| <= {TRIM*100:.0f}%, used in figure): '
          f'R2={r2_trim:.4f}, r={r_trim:.4f}, n={len(ann_trim)} '
          f'({n_excluded} pair(s) excluded)')
    if n_excluded:
        for _, row in ann[trimmed_mask].iterrows():
            print(f'    excluded: {row["name_prod"]} {int(row["Year"])} '
                  f'actual {row["actual_gr"]*100:+.1f}%, '
                  f'pred {row["pred_gr"]*100:+.1f}% '
                  f'({int(row["n_matched_months"])} matched months)')

    growth_scatter_figure(ann_trim, r2_trim, r_trim, scatter_name, scatter_title)

    # C. Regional aggregate growth on a balanced plant panel
    nat, n_plants, panel_note = regional_aggregate_growth(test_df, region_label)
    r_agg  = pearsonr(nat['g_actual'], nat['g_pred'])[0]
    r2_agg = r2_score(nat['g_actual'], nat['g_pred'])
    print(f'  [{region_label}] aggregate monthly YoY: r={r_agg:.4f}, '
          f'R2={r2_agg:.4f}, n={len(nat)} months, n_plants={n_plants}')
    print(f'  [{region_label}] aggregate growth range: actual '
          f'[{nat["g_actual"].min()*100:+.1f}%, {nat["g_actual"].max()*100:+.1f}%], '
          f'pred [{nat["g_pred"].min()*100:+.1f}%, {nat["g_pred"].max()*100:+.1f}%]')
    aggregate_figure(nat, r_agg, r2_agg, agg_name, region_label)

    nat_out = nat[['Year', 'Month', 'Steel_Prod', 'pred',
                   'base_actual', 'base_pred', 'g_actual', 'g_pred']].copy()
    nat_out.insert(0, 'Direction', direction)
    nat_out['n_plants']   = n_plants
    nat_out['panel_note'] = panel_note
    nat_out = nat_out.rename(columns={'Steel_Prod': 'actual_total',
                                      'pred': 'pred_total'})

    return dict(ann=ann, ann_trim=ann_trim, n_excluded=n_excluded,
                r2_full=r2_full, r_full=r_full, r2_trim=r2_trim, r_trim=r_trim,
                nat=nat_out, r_agg=r_agg, r2_agg=r2_agg, n_plants=n_plants,
                panel_note=panel_note)


# ── Split (unchanged logic): North = Lat >= 35, South = Lat < 35 ──────────────
north = data[data['Latitude'] >= 35].copy()
south = data[data['Latitude'] <  35].copy()

# Main direction: train North → test South (Figure 7b)
res_s = run_geo_holdout(
    'North->South', north, south,
    scatter_name='fig_geo_holdout_south_matched_diagnostic',
    scatter_title=('Predicted vs. Actual Annual Growth Rate\n'
                   '(Southern China Geographic Holdout, matched months)'),
    agg_name='fig_geo_holdout_south_agg',
    region_label='Southern China')

# D. Mirror exercise: train South → test North (same thresholds, same reporting)
res_n = run_geo_holdout(
    'South->North', south, north,
    scatter_name='fig_geo_holdout_north_mirror',
    scatter_title=('Predicted vs. Actual Annual Growth Rate\n'
                   '(Northern China Mirror Holdout, matched months)'),
    agg_name='fig_geo_holdout_north_agg',
    region_label='Northern China')

# ── Supplementary tables ───────────────────────────────────────────────────────
print('\nWriting supplementary tables ...')
supp_s = make_supp_table(res_s['ann'])
p_s = os.path.join(OUT_TAB, 'table_s_geographic_holdout_growth.csv')
supp_s.to_csv(p_s, index=False, encoding='utf-8-sig')
print(f'  Saved: {p_s}  ({len(supp_s)} rows, '
      f'{supp_s["Plant"].nunique()} south plants, '
      f'{int(supp_s["Trimmed_gt200pct"].sum())} flagged trimmed)')

supp_n = make_supp_table(res_n['ann'])
p_n = os.path.join(OUT_TAB, 'table_s_geo_holdout_mirror.csv')
supp_n.to_csv(p_n, index=False, encoding='utf-8-sig')
print(f'  Saved: {p_n}  ({len(supp_n)} rows, '
      f'{supp_n["Plant"].nunique()} north plants, '
      f'{int(supp_n["Trimmed_gt200pct"].sum())} flagged trimmed)')

agg_both = pd.concat([res_s['nat'], res_n['nat']], ignore_index=True)
p_a = os.path.join(OUT_TAB, 'table_s_geo_holdout_agg.csv')
agg_both.to_csv(p_a, index=False, encoding='utf-8-sig')
print(f'  Saved: {p_a}  ({len(agg_both)} direction-months)')

print('\nCopying outputs ...')
copy_file(os.path.join(OUT_FIG, 'Figure_7b.pdf'), REPL_OUT)
copy_file(os.path.join(OUT_FIG, 'Figure_7b.pdf'), PAPER_FIG_DIR)
# New figures: replicated + output_rl
for fname in ('fig_geo_holdout_south_agg', 'fig_geo_holdout_north_mirror',
              'fig_geo_holdout_north_agg'):
    src = os.path.join(OUT_FIG, f'{fname}.pdf')
    copy_file(src, REPL_OUT)
    copy_file(src, OUT_RL)
# Tables: output_rl copies
for p in (p_s, p_n, p_a):
    copy_file(p, OUT_RL)

# ── Summary ────────────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('SUMMARY (Panel B revision)')
for tag, res in (('South holdout (North->South)', res_s),
                 ('North mirror  (South->North)', res_n)):
    print(f'  {tag}:')
    print(f'    matched-month pairs retained: {len(res["ann"])}; '
          f'excluded |g|>200%: {res["n_excluded"]}')
    print(f'    trimmed  R2={res["r2_trim"]:.4f}  r={res["r_trim"]:.4f}  '
          f'n={len(res["ann_trim"])}')
    print(f'    full     R2={res["r2_full"]:.4f}  r={res["r_full"]:.4f}  '
          f'n={len(res["ann"])}')
    print(f'    aggregate r={res["r_agg"]:.4f}  R2={res["r2_agg"]:.4f}  '
          f'n={len(res["nat"])} months  n_plants={res["n_plants"]} '
          f'({res["panel_note"]})')

print('\nDone.')
