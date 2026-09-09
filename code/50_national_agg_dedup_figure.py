
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, OUT_RL

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

DATA = DATA_PROCESSED
CONF = DATA_CONFIDENTIAL
OUT  = OUT_RL
os.makedirs(OUT, exist_ok=True)

# -- 1. Load (same as notebook 33) -------------------------------------------
print('Loading data ...')
combined_df  = pd.read_csv(os.path.join(DATA, 'merged_SteelIron_pollutants_2019_2022.csv'))
grid_prod_df = pd.read_csv(os.path.join(CONF, 'GridProd_1922_monthly.csv'))
grid_info_df = pd.read_csv(os.path.join(CONF, 'gridinfo_xy.csv'))

# Dedup guard (2026-07): GridProd_1922_monthly.csv contains byte-identical
# duplicated Dec-2022 rows; dedup before any split/evaluation.
# Source files intentionally unmodified.
n_before = len(grid_prod_df)
grid_prod_df = grid_prod_df.drop_duplicates(subset=['IDCode', 'name_prod', 'Year', 'Month'])
print(f'Dedup guard removed {n_before - len(grid_prod_df)} duplicate GridProd rows')

combined_df = combined_df.drop(columns=['PM1_MEAN'], errors='ignore')

temp_df = grid_prod_df[['IDCode', 'Year', 'Month',
                        'GridProd_Steel_tot', 'GridProd_Iron_tot']].drop_duplicates()
final_df = pd.merge(combined_df, temp_df, on=['IDCode', 'Year', 'Month'], how='left')
final_df = pd.merge(final_df, grid_info_df, on=['IDCode'], how='left')

grid_data = final_df[
    final_df['GridProd_Steel_tot'].notna() &
    (final_df['GridProd_Steel_tot'] != 0) &
    final_df['GridProd_Iron_tot'].notna() &
    (final_df['GridProd_Iron_tot'] != 0)
].copy()
print(f'Grid obs after filter: {len(grid_data):,}')

# -- 2. LOYO OOF (same as notebook 33) ---------------------------------------
DROP_COLS = ['GridProd_Steel_tot', 'GridProd_Iron_tot', 'Polygon_ID', '_merge']
feat_cols = grid_data.drop(columns=DROP_COLS, errors='ignore').columns.tolist()
grid_data['log_target'] = np.log1p(grid_data['GridProd_Steel_tot'])

XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0
)

records = []
for test_year in [2019, 2020, 2021, 2022]:
    train = grid_data[grid_data['Year'] != test_year]
    test  = grid_data[grid_data['Year'] == test_year]
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(train[feat_cols])
    X_test  = imputer.transform(test[feat_cols])
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, train['log_target'].values)
    tmp = test[['IDCode', 'Year', 'Month']].copy().reset_index(drop=True)
    tmp['y_true_log'] = test['log_target'].values
    tmp['y_pred_log'] = model.predict(X_test)
    records.append(tmp)
    print(f'  Year {test_year}: grid R2 (log) = '
          f'{r2_score(tmp["y_true_log"], tmp["y_pred_log"]):.4f}  n={len(tmp)}')

oof = pd.concat(records, ignore_index=True)
oof['true_kt'] = np.expm1(oof['y_true_log'])
oof['pred_kt'] = np.expm1(oof['y_pred_log'])

# -- 3. National monthly aggregation (sum over ALL sampled grids) ------------
national = (oof.groupby(['Year', 'Month'], as_index=False)
               .agg(actual_kt=('true_kt', 'sum'),
                    pred_kt=('pred_kt', 'sum')))

# -- 4. Restrict to CISA-reported months (2019-2022; drops Jan/Feb) ----------
cisa = pd.read_excel(os.path.join(DATA, 'Steel_CISA.xlsx'))
cisa['Date'] = pd.to_datetime(cisa['Date'])
cisa['Year'] = cisa['Date'].dt.year
cisa['Month'] = cisa['Date'].dt.month
cisa = cisa[(cisa['Year'] >= 2019) & (cisa['Year'] <= 2022)]

national = national.merge(cisa[['Year', 'Month']], on=['Year', 'Month'], how='inner')
print(f'Sample after inner merge with CISA dates: n = {len(national)}')

# -- 5. Internal R2 (predicted vs own actual totals) -------------------------
r2_nat = r2_score(national['actual_kt'], national['pred_kt'])
print(f'National monthly internal R2 = {r2_nat:.4f}')

# -- 6. Figure ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
ax.scatter(national['actual_kt'], national['pred_kt'],
           color='#00468B', alpha=0.85, zorder=3)

lims = [min(national['actual_kt'].min(), national['pred_kt'].min()),
        max(national['actual_kt'].max(), national['pred_kt'].max())]
pad = 0.05 * (lims[1] - lims[0])
lims = [lims[0] - pad, lims[1] + pad]
ax.plot(lims, lims, 'k--', lw=1.2, zorder=2)
ax.set_xlim(lims); ax.set_ylim(lims)

ax.set_xlabel('Actual aggregate production (kt)', fontsize=12)
ax.set_ylabel('Predicted aggregate production (kt)', fontsize=12)
ax.set_title(f'National Monthly: $R^2$ = {r2_nat:.2f}', fontsize=13)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()
out_path = os.path.join(OUT, 'figR2.5_national_agg_panelA_dedup.pdf')
fig.savefig(out_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out_path}')
