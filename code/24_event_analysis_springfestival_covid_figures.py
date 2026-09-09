import os, sys, io, warnings
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split  # noqa: F401 (kept for reference)
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_FIG, OUT_RL, need

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Paths ──────────────────────────────────────────────────────────────────────
SIGNALS_PATH  = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH      = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')
OUT_FIG       = OUT_FIG
os.makedirs(OUT_FIG, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
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

# Spring Festival exact date ranges (matching original notebook)
SPRING_FESTIVAL_DATES = {
    '2019': ('2019-02-04', '2019-02-10'),
    '2020': ('2020-01-24', '2020-01-30'),
    '2021': ('2021-02-11', '2021-02-17'),
    '2022': ('2022-01-31', '2022-02-06'),
}

# COVID shading: Jan 2020 – Dec 2021
COVID_START = pd.Timestamp('2020-01-01')
COVID_END   = pd.Timestamp('2021-12-31')

# Wuhan definition (matching original notebook)
WUHAN_LAT   = 30.5928
WUHAN_LON   = 114.3055
WUHAN_DEG   = 0.45          # degree radius

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
    for ext in ('pdf',):   # paper store holds one file per figure
        p = os.path.join(OUT_FIG, f'{name}.{ext}')
        try:
            fig.savefig(p, bbox_inches='tight', dpi=800)
            print(f'  Saved: {p}')
        except PermissionError:
            p2 = os.path.join(OUT_FIG, f'{name}_v2.{ext}')
            fig.savefig(p2, bbox_inches='tight', dpi=800)
            print(f'  Saved (alt): {p2}')

# ── Load and prepare data (same pipeline as fig6/fig7) ────────────────────────
print('Loading data ...')
signals  = pd.read_csv(SIGNALS_PATH)
signals  = signals.drop(columns=['PM1_MEAN'], errors='ignore')
gridprod = pd.read_csv(GRIDPROD_PATH)

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
print(f'  Plants before dropna: {grouped["name_prod"].nunique()}, rows: {len(grouped)}')

grouped['month_sin'] = np.sin(2 * np.pi * grouped['Month'] / 12)
grouped['month_cos'] = np.cos(2 * np.pi * grouped['Month'] / 12)
grouped = grouped.sort_values(['plant_id', 'Year', 'Month']).reset_index(drop=True)
for lag in range(1, 4):
    for col in PLANT_POLLUTANTS:
        grouped[f'{col}_lag{lag}'] = grouped.groupby('plant_id')[col].shift(lag)
grouped = add_dist_features(grouped)

data = grouped.dropna(subset=FEATURE_COLS + ['Steel_Prod', 'Year', 'Latitude', 'Longitude']).copy()
data = data[data['Steel_Prod'] > 0].reset_index(drop=True)
data['date'] = pd.to_datetime(
    data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2) + '-01')
print(f'  After dropna: {data["name_prod"].nunique()} plants, {len(data)} rows')

# ── Row-level 80/20 split (matching original notebook exactly) ────────────────
# Split individual (plant, month) observations randomly; both train and test
# span all plants and all months — appropriate for event analysis.
rng = np.random.RandomState(1)
idx = np.arange(len(data))
rng.shuffle(idx)
split = int(len(idx) * 0.8)
train_idx, test_idx = idx[:split], idx[split:]

train_data = data.iloc[train_idx].copy()
test_data  = data.iloc[test_idx].copy()
print(f'\nRow-level split: train={len(train_data)}, test={len(test_data)}')

X_train = train_data[FEATURE_COLS].values
y_train = np.log1p(train_data['Steel_Prod'].values)

model = xgb.XGBRegressor(**XGB_PARAMS)
model.fit(X_train, y_train)
print('  Model trained.')

# ── Figure 8a: Spring Festival ─────────────────────────────────────────────────
print('\nPanel A: Spring Festival event analysis ...')

# Predict test-set rows only
test_data = test_data.copy()
test_data['pred'] = np.expm1(model.predict(test_data[FEATURE_COLS].values))

# Monthly mean / min / max across test-set plant-month observations
monthly_stats = (test_data
                 .groupby(test_data['date'].dt.to_period('M'))
                 .agg(Actual_mean=('Steel_Prod', 'mean'),
                      Actual_min =('Steel_Prod', 'min'),
                      Actual_max =('Steel_Prod', 'max'),
                      Pred_mean  =('pred', 'mean'),
                      Pred_min   =('pred', 'min'),
                      Pred_max   =('pred', 'max'))
                 .to_timestamp()
                 .reset_index()
                 .sort_values('date'))

dates = monthly_stats['date'].values

fig_a, ax_a = plt.subplots(figsize=(12, 6), dpi=300)

# Actual: mean line + asymmetric min/max error bars
yerr_actual = [
    monthly_stats['Actual_mean'] - monthly_stats['Actual_min'],
    monthly_stats['Actual_max']  - monthly_stats['Actual_mean'],
]
ax_a.errorbar(dates, monthly_stats['Actual_mean'], yerr=yerr_actual,
              fmt='-o', color='blue', ecolor='lightblue', capsize=3,
              lw=1.5, ms=4, label='Actual', zorder=3)

# Predicted: mean line + asymmetric min/max error bars
yerr_pred = [
    monthly_stats['Pred_mean'] - monthly_stats['Pred_min'],
    monthly_stats['Pred_max']  - monthly_stats['Pred_mean'],
]
ax_a.errorbar(dates, monthly_stats['Pred_mean'], yerr=yerr_pred,
              fmt='-x', color='maroon', ecolor='orange', capsize=3,
              lw=1.5, ms=5, label='Predicted', zorder=3)

# Shade exact Spring Festival date ranges
for yr, (start_str, end_str) in SPRING_FESTIVAL_DATES.items():
    ax_a.axvspan(pd.Timestamp(start_str), pd.Timestamp(end_str),
                 color='grey', alpha=0.3,
                 label='Spring Festival' if yr == '2019' else '_nolegend_')

ax_a.set_xlabel('Date', fontsize=12)
ax_a.set_ylabel('Plant Steel Output (10,000 MT/month)', fontsize=12)
ax_a.set_title('Monthly Plant Output (2019–2022) and Spring Festival Impact', fontsize=13)
ax_a.xaxis.set_major_locator(mdates.YearLocator())
ax_a.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax_a.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
ax_a.legend(loc='upper left', fontsize=10, framealpha=0.85)
ax_a.yaxis.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
save_fig(fig_a, 'Figure_06a')
plt.close()

# ── Figure 8b: COVID-19 ────────────────────────────────────────────────────────
print('\nPanel B: COVID-19 event analysis ...')

# Predict ALL plants (not just test set) using same model
data = data.copy()
data['pred'] = np.expm1(model.predict(data[FEATURE_COLS].values))

# Identify Wuhan plants: within WUHAN_DEG degrees of Wuhan center
plant_coords = (data.groupby('name_prod')[['Latitude', 'Longitude']]
                .mean().reset_index())
plant_coords['is_wuhan'] = (
    (np.abs(plant_coords['Latitude']  - WUHAN_LAT) <= WUHAN_DEG) &
    (np.abs(plant_coords['Longitude'] - WUHAN_LON) <= WUHAN_DEG)
)
wuhan_plants     = plant_coords.loc[plant_coords['is_wuhan'],  'name_prod'].values
non_wuhan_plants = plant_coords.loc[~plant_coords['is_wuhan'], 'name_prod'].values
print(f'  Wuhan plants: {len(wuhan_plants)}: {list(wuhan_plants)}')
print(f'  Non-Wuhan plants: {len(non_wuhan_plants)}')

# Monthly sum for each group, then normalize by Jan 2019 value
def group_monthly_normalized(df, plant_set, label):
    sub = df[df['name_prod'].isin(plant_set)]
    monthly = (sub.groupby(['Year', 'Month', 'date'])[['Steel_Prod', 'pred']]
               .sum().reset_index().sort_values('date'))
    # Normalize by first observation (Jan 2019)
    base_actual = monthly.iloc[0]['Steel_Prod']
    base_pred   = monthly.iloc[0]['pred']
    if base_actual == 0 or base_pred == 0:
        print(f'  WARNING: zero base value for {label} group')
        base_actual = max(base_actual, 1)
        base_pred   = max(base_pred,   1)
    monthly['actual_norm'] = monthly['Steel_Prod'] / base_actual
    monthly['pred_norm']   = monthly['pred']       / base_pred
    return monthly

wuhan_m     = group_monthly_normalized(data, wuhan_plants,     'Wuhan')
non_wuhan_m = group_monthly_normalized(data, non_wuhan_plants, 'Non-Wuhan')

fig_b, ax_b = plt.subplots(figsize=(10, 5), dpi=300)

# Wuhan lines
ax_b.plot(wuhan_m['date'], wuhan_m['actual_norm'],
          color='firebrick', lw=1.8, label='Wuhan (Actual)', zorder=3)
ax_b.plot(wuhan_m['date'], wuhan_m['pred_norm'],
          color='firebrick', lw=1.8, linestyle='--', label='Wuhan (Predicted)', zorder=3)

# Non-Wuhan lines
ax_b.plot(non_wuhan_m['date'], non_wuhan_m['actual_norm'],
          color='steelblue', lw=1.8, label='Non-Wuhan (Actual)', zorder=3)
ax_b.plot(non_wuhan_m['date'], non_wuhan_m['pred_norm'],
          color='steelblue', lw=1.8, linestyle='--', label='Non-Wuhan (Predicted)', zorder=3)

# Shade COVID period
ax_b.axvspan(COVID_START, COVID_END, alpha=0.15, color='gray', zorder=1,
             label='COVID-19 Period (Jan 2020 – Dec 2021)')

ax_b.axhline(1.0, color='black', lw=0.8, linestyle=':', alpha=0.5)
ax_b.set_xlabel('Month', fontsize=12)
ax_b.set_ylabel('Steel Output (Normalized, Jan 2019 = 1)', fontsize=12)
ax_b.set_title('COVID-19 Impact on Steel Output: Wuhan vs. Non-Wuhan', fontsize=13)
ax_b.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax_b.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
ax_b.legend(fontsize=9, framealpha=0.85, ncol=2)
ax_b.yaxis.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
save_fig(fig_b, 'Figure_06b')
plt.close()

# ── Figure 8b_2: COVID-19 counterfactual — train on 2019 only ─────────────────
# Train exclusively on 2019 data, then predict the full 2019–2022 period.
# The gap between predicted and actual in 2020 is a cleaner COVID shock estimate
# because the model has no exposure to COVID-period production levels.
print('\nPanel B (v2): COVID-19 counterfactual (train 2019 only) ...')

train_2019 = data[data['Year'] == 2019]
model_2019 = xgb.XGBRegressor(**XGB_PARAMS)
model_2019.fit(train_2019[FEATURE_COLS].values,
               np.log1p(train_2019['Steel_Prod'].values))

data_v2 = data.copy()
data_v2['pred'] = np.expm1(model_2019.predict(data_v2[FEATURE_COLS].values))

wuhan_v2     = group_monthly_normalized(data_v2, wuhan_plants,     'Wuhan')
non_wuhan_v2 = group_monthly_normalized(data_v2, non_wuhan_plants, 'Non-Wuhan')

fig_b2, ax_b2 = plt.subplots(figsize=(10, 5), dpi=300)

ax_b2.plot(wuhan_v2['date'], wuhan_v2['actual_norm'],
           color='firebrick', lw=1.8, label='Wuhan (Actual)', zorder=3)
ax_b2.plot(wuhan_v2['date'], wuhan_v2['pred_norm'],
           color='firebrick', lw=1.8, linestyle='--', label='Wuhan (Predicted)', zorder=3)

ax_b2.plot(non_wuhan_v2['date'], non_wuhan_v2['actual_norm'],
           color='steelblue', lw=1.8, label='Non-Wuhan (Actual)', zorder=3)
ax_b2.plot(non_wuhan_v2['date'], non_wuhan_v2['pred_norm'],
           color='steelblue', lw=1.8, linestyle='--', label='Non-Wuhan (Predicted)', zorder=3)

ax_b2.axvspan(COVID_START, COVID_END, alpha=0.15, color='gray', zorder=1,
              label='COVID-19 Period (Jan 2020 – Dec 2021)')
ax_b2.axhline(1.0, color='black', lw=0.8, linestyle=':', alpha=0.5)
ax_b2.set_xlabel('Month', fontsize=12)
ax_b2.set_ylabel('Steel Output (Normalized, Jan 2019 = 1)', fontsize=12)
ax_b2.set_title('COVID-19 Impact on Steel Output: Wuhan vs. Non-Wuhan\n'
                '(Counterfactual: model trained on 2019 only)', fontsize=13)
ax_b2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax_b2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax_b2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
ax_b2.legend(fontsize=9, framealpha=0.85, ncol=2)
ax_b2.yaxis.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
for _ext in ('pdf', 'png'):
    fig_b2.savefig(os.path.join(OUT_RL, 'fig_event_analysis_variant.' + _ext),
                   bbox_inches='tight', dpi=300)
print('  saved variant to output_rl')
plt.close()

print('\nDone. Summary:')
print(f'  Fig 6a — Spring Festival: {len(test_data["name_prod"].unique())} test plants, '
      f'{len(monthly_stats)} monthly observations')
print(f'  Fig 6b — COVID-19: {len(wuhan_plants)} Wuhan plants, '
      f'{len(non_wuhan_plants)} non-Wuhan plants')
