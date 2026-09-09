import os, warnings, gc
import numpy as np
import pandas as pd
import matplotlib
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, OUT_RL, OUT_TAB, need
matplotlib.use("Agg")
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DATA    = DATA_PROCESSED
CSV1         = os.path.join(BASE_DATA, "first_output_file.csv")
CSV2         = os.path.join(BASE_DATA, "second_output_file.csv")
GRIDPROD_CSV = need(DATA_CONFIDENTIAL, "GridProd_1922_monthly.csv")

OUT_TABLES   = OUT_TAB
# 11_location_classifier_seasonal_cm_figure.py reads from this path:
LOCAL_OUT    = OUT_RL
os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(LOCAL_OUT,  exist_ok=True)

RANDOM_SEED  = 42
EPOCHS       = 11
NEG_SMOTE    = 50_000
N_TEST_NEG   = 1_000_000
CHUNKSIZE    = 500_000

SIGNALS = ['CO_MEAN', 'NO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN', 'SO2_MEAN',
           'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN', 'O3_MEAN']

MONTHS_TO_TRAIN = {
    2:  "February (Winter)",
    5:  "May (Spring)",
    8:  "August (Summer)",
    11: "November (Autumn)",
}

# ── Load full data once ────────────────────────────────────────────────────────
print("Loading all data (float32) ...")
id_parts, yr_parts, mo_parts, sig_parts = [], [], [], []
for fpath in [CSV1, CSV2]:
    if not os.path.exists(fpath):
        print(f"  [skip] {os.path.basename(fpath)} not found"); continue
    n_rows = 0
    for chunk in pd.read_csv(
            fpath, chunksize=CHUNKSIZE, low_memory=False,
            usecols=['IDCode', 'Year', 'Month'] + SIGNALS,
            dtype={'IDCode': np.int32, 'Year': np.int16, 'Month': np.int8,
                   **{col: np.float32 for col in SIGNALS}}):
        id_parts.append(chunk['IDCode'].to_numpy())
        yr_parts.append(chunk['Year'].to_numpy())
        mo_parts.append(chunk['Month'].to_numpy())
        sig_parts.append(chunk[SIGNALS].to_numpy(dtype=np.float32))
        n_rows += len(chunk)
    print(f"  {os.path.basename(fpath)}: {n_rows:,} rows")

id_arr  = np.concatenate(id_parts);  del id_parts
yr_arr  = np.concatenate(yr_parts);  del yr_parts
mo_arr  = np.concatenate(mo_parts);  del mo_parts
sig_arr = np.concatenate(sig_parts); del sig_parts
gc.collect()
print(f"  Total: {len(id_arr):,} rows")

# ── Sort ───────────────────────────────────────────────────────────────────────
print("Sorting ...")
order  = np.lexsort((mo_arr, yr_arr, id_arr))
id_arr = id_arr[order]; yr_arr = yr_arr[order]
mo_arr = mo_arr[order]; sig_arr = sig_arr[order]
del order; gc.collect()

# ── Lag features (36 total) ────────────────────────────────────────────────────
print("Computing lag features ...")
n_signals = len(SIGNALS)
n_lags    = 3
n_feats   = n_signals * (1 + n_lags)   # 36

X_all = np.empty((len(id_arr), n_feats), dtype=np.float32)
X_all[:, :n_signals] = sig_arr

boundaries   = np.where(np.diff(id_arr) != 0)[0] + 1
group_starts = np.concatenate([[0], boundaries])
group_ends   = np.concatenate([boundaries, [len(id_arr)]])

for lag in range(1, n_lags + 1):
    col_offset = n_signals * lag
    lag_block  = X_all[:, col_offset : col_offset + n_signals]
    lag_block[:] = np.nan
    for gs, ge in zip(group_starts, group_ends):
        if ge - gs > lag:
            lag_block[gs + lag : ge] = sig_arr[gs : ge - lag]

del sig_arr; gc.collect()

# Drop NaN rows (lag boundary rows)
valid_mask = ~np.isnan(X_all).any(axis=1)
X_all  = X_all[valid_mask]
id_v   = id_arr[valid_mask]; yr_v = yr_arr[valid_mask]; mo_v = mo_arr[valid_mask]
del id_arr, yr_arr, mo_arr, valid_mask; gc.collect()
print(f"  After dropna: {len(X_all):,} rows")

# ── GridProd labels ─────────────────────────────────────────────────────────────
print("Loading labels ...")
gp = pd.read_csv(GRIDPROD_CSV,
                 usecols=['IDCode', 'Year', 'Month', 'GridProd_Steel_tot'],
                 dtype={'IDCode': np.int32, 'Year': np.int16, 'Month': np.int8})
pos_keys = set(
    int(r.IDCode) * 10000 + int(r.Year) * 100 + int(r.Month)
    for r in gp.itertuples(index=False) if r.GridProd_Steel_tot > 0
)
del gp; gc.collect()

key_int = id_v.astype(np.int64) * 10000 + yr_v.astype(np.int64) * 100 + mo_v.astype(np.int64)
y_all   = np.zeros(len(X_all), dtype=np.int8)
y_all[np.isin(key_int, list(pos_keys))] = 1
del key_int, pos_keys; gc.collect()
print(f"  Positive rate: {y_all.mean():.5f}  ({y_all.sum():,} positives / {len(X_all):,} total)")

# ── Per-season training loop ───────────────────────────────────────────────────
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_score, recall_score)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

results = []

for month_num, season_label in MONTHS_TO_TRAIN.items():
    print(f"\n{'='*60}")
    print(f"Training: {season_label} (month={month_num}) ...")

    mask_m = (mo_v == month_num)
    X_m    = X_all[mask_m]
    y_m    = y_all[mask_m]

    pos_idx = np.where(y_m == 1)[0]
    neg_idx = np.where(y_m == 0)[0]
    print(f"  {len(X_m):,} rows | {len(pos_idx):,} pos | {len(neg_idx):,} neg")

    rng_m      = np.random.default_rng(RANDOM_SEED + month_num)
    n_test_pos = max(1, int(len(pos_idx) * 0.20))
    test_pos_i = rng_m.choice(pos_idx, size=n_test_pos, replace=False)
    train_pos_i = np.setdiff1d(pos_idx, test_pos_i)

    n_test_neg  = min(N_TEST_NEG, len(neg_idx))
    test_neg_i  = rng_m.choice(neg_idx, size=n_test_neg, replace=False)
    train_neg_i = np.setdiff1d(neg_idx, test_neg_i)

    X_test = np.concatenate([X_m[test_pos_i], X_m[test_neg_i]])
    y_test = np.concatenate([np.ones(n_test_pos, dtype=np.int8),
                              np.zeros(n_test_neg, dtype=np.int8)])

    n_neg_use    = min(NEG_SMOTE, len(train_neg_i))
    neg_sub_i    = rng_m.choice(train_neg_i, size=n_neg_use, replace=False)
    X_train_pool = np.concatenate([X_m[train_pos_i], X_m[neg_sub_i]])
    y_train_pool = np.concatenate([np.ones(len(train_pos_i), dtype=np.int8),
                                    np.zeros(n_neg_use, dtype=np.int8)])

    imputer  = SimpleImputer(strategy='mean')
    X_tr_sc  = imputer.fit_transform(X_train_pool)
    X_te_sc  = imputer.transform(X_test)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_sc)
    X_te_sc  = scaler.transform(X_test)

    smote       = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = smote.fit_resample(X_tr_sc, y_train_pool)
    del X_tr_sc, y_train_pool

    model = Sequential([
        Dense(64, input_shape=(n_feats,), activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_res, y_res, epochs=EPOCHS, batch_size=32, verbose=0)
    del X_res, y_res

    y_prob  = model.predict(X_te_sc, verbose=0).ravel()
    y_pred  = (y_prob >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)
    recall  = recall_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)

    print(f"  Recall={recall:.3f}  Precision={prec:.4f}  "
          f"ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")

    results.append({
        'Month':           season_label,
        'Season':          season_label.split('(')[1].rstrip(')'),
        'N_test':          len(y_test),
        'N_positive_test': n_test_pos,
        'Recall':          round(recall, 4),
        'Precision':       round(prec,   4),
        'ROC_AUC':         round(roc_auc, 4),
        'PR_AUC':          round(pr_auc,  4),
    })
    del X_m, y_m, X_test, y_test, X_te_sc, y_prob, model
    gc.collect()

# ── Save CSV ───────────────────────────────────────────────────────────────────
df_out   = pd.DataFrame(results)
csv_name = 'table_month_retrain_robustness.csv'
for out_dir in [OUT_TABLES, LOCAL_OUT]:
    path = os.path.join(out_dir, csv_name)
    df_out.to_csv(path, index=False)
    print(f"Saved: {path}")

print("\nSummary:")
print(df_out[['Month', 'N_positive_test', 'Recall', 'Precision', 'ROC_AUC', 'PR_AUC']].to_string(index=False))
print("\nDone. Run script 11 (41_month_cm_figure.py) to generate Fig 3a.")
