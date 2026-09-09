
import os, warnings, gc, pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, OUT_FIG, need

warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DATA    = DATA_PROCESSED
CSV1         = os.path.join(BASE_DATA, "first_output_file.csv")
CSV2         = os.path.join(BASE_DATA, "second_output_file.csv")
GRIDPROD_CSV = need(DATA_CONFIDENTIAL, "GridProd_1922_monthly.csv")

OUT_DIR   = OUT_FIG
PAPER_FIG = OUT_FIG
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED  = 42
EPOCHS       = 11
SHAP_BG_N    = 10_000
SHAP_EXPL_N  = 10_000
NEG_SMOTE    = 50_000      # negatives subsampled before SMOTE
N_TEST_NEG   = 1_000_000   # negatives held out for true-distribution evaluation
CHUNKSIZE    = 500_000

SIGNALS = ['CO_MEAN', 'NO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN', 'SO2_MEAN',
           'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN', 'O3_MEAN']

rng = np.random.default_rng(RANDOM_SEED)

# ── Step 1: Load ALL data as numpy float32 ─────────────────────────────────────
print("Loading all data (float32, numpy path) ...")
id_parts, yr_parts, mo_parts, sig_parts = [], [], [], []

for fpath in [CSV1, CSV2]:
    if not os.path.exists(fpath):
        print(f"  [skip] {os.path.basename(fpath)} not found")
        continue
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
print(f"  Total: {len(id_arr):,} rows | sig_arr: {sig_arr.nbytes/1e9:.1f} GB")

# ── Step 2: Sort ───────────────────────────────────────────────────────────────
print("Sorting ...")
order   = np.lexsort((mo_arr, yr_arr, id_arr))
id_arr  = id_arr[order];  yr_arr  = yr_arr[order]
mo_arr  = mo_arr[order];  sig_arr = sig_arr[order]
del order; gc.collect()

# ── Step 3: Compute lag features ──────────────────────────────────────────────
print("Computing lag features (numpy) ...")
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
print(f"  X_all shape: {X_all.shape} | memory: {X_all.nbytes/1e9:.1f} GB")

# ── Step 4: Drop NaN rows ─────────────────────────────────────────────────────
print("Dropping NaN rows ...")
valid_mask = ~np.isnan(X_all).any(axis=1)
X_all = X_all[valid_mask]
id_v  = id_arr[valid_mask];  yr_v = yr_arr[valid_mask];  mo_v = mo_arr[valid_mask]
del id_arr, yr_arr, mo_arr, valid_mask; gc.collect()
print(f"  After dropna: {len(X_all):,} rows")

# ── Step 5: Merge GridProd labels ─────────────────────────────────────────────
print("Merging GridProd labels ...")
gp = pd.read_csv(GRIDPROD_CSV,
                 usecols=['IDCode', 'Year', 'Month', 'GridProd_Steel_tot'],
                 dtype={'IDCode': np.int32, 'Year': np.int16, 'Month': np.int8})
gp['y'] = (gp['GridProd_Steel_tot'] > 0).astype(np.int8)
pos_keys = set(
    int(r.IDCode) * 10000 + int(r.Year) * 100 + int(r.Month)
    for r in gp.itertuples(index=False) if r.y == 1
)
del gp; gc.collect()

key_int = id_v.astype(np.int64) * 10000 + yr_v.astype(np.int64) * 100 + mo_v.astype(np.int64)
y_all   = np.zeros(len(X_all), dtype=np.int8)
y_all[np.isin(key_int, list(pos_keys))] = 1
del key_int, id_v, yr_v, mo_v, pos_keys; gc.collect()
n_pos = y_all.sum()
print(f"  Positive rate: {y_all.mean():.5f}  ({n_pos:,} positives / {len(X_all):,} total)")

# ── Step 6: Reserve true-distribution test set, then free X_all ───────────────
print("Reserving true-distribution test set ...")
pos_idx = np.where(y_all == 1)[0]
neg_idx = np.where(y_all == 0)[0]

# 20% of positives held out for test
n_test_pos    = int(len(pos_idx) * 0.20)
test_pos_idx  = rng.choice(pos_idx, size=n_test_pos, replace=False)
train_pos_idx = np.setdiff1d(pos_idx, test_pos_idx)

# 1M negatives held out for test
n_test_neg    = min(N_TEST_NEG, len(neg_idx))
test_neg_idx  = rng.choice(neg_idx, size=n_test_neg, replace=False)
train_neg_idx = np.setdiff1d(neg_idx, test_neg_idx)

# Extract compact test set (~144 MB)
X_test_true = np.concatenate([X_all[test_pos_idx], X_all[test_neg_idx]])
y_test_true = np.concatenate([np.ones(n_test_pos,  dtype=np.int8),
                               np.zeros(n_test_neg, dtype=np.int8)])

# Training pool: all train positives + 200K subsampled negatives
n_train_neg_sub   = min(200_000, len(train_neg_idx))
train_neg_sub_idx = rng.choice(train_neg_idx, size=n_train_neg_sub, replace=False)
X_train_pool = np.concatenate([X_all[train_pos_idx], X_all[train_neg_sub_idx]])
y_train_pool = np.concatenate([np.ones(len(train_pos_idx), dtype=np.int8),
                                np.zeros(n_train_neg_sub,   dtype=np.int8)])

del X_all, y_all, pos_idx, neg_idx
del test_pos_idx, test_neg_idx, train_pos_idx, train_neg_idx, train_neg_sub_idx
gc.collect()

baseline_rate = n_test_pos / (n_test_pos + n_test_neg)
print(f"  Test set: {n_test_pos:,} pos + {n_test_neg:,} neg "
      f"(baseline = {baseline_rate:.4f})")
print(f"  Train pool: {len(X_train_pool):,} rows | freed 16 GB array")

# Feature column names
feature_cols = list(SIGNALS)
for lag in range(1, n_lags + 1):
    for s in SIGNALS:
        feature_cols.append(f"{s}_lag{lag}")

# ── Step 7: Impute + scale (fit on training pool only) ────────────────────────
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

imputer    = SimpleImputer(strategy='mean')
X_train_sc = imputer.fit_transform(X_train_pool)
X_test_sc  = imputer.transform(X_test_true)
del X_train_pool

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sc)
X_test_sc  = scaler.transform(X_test_true)
del X_test_true; gc.collect()

# ── Step 8: Subsample negatives + SMOTE ────────────────────────────────────────
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

print("Subsampling negatives and applying SMOTE ...")
idx_pos     = np.where(y_train_pool == 1)[0]
idx_neg     = np.where(y_train_pool == 0)[0]
n_neg_use   = min(NEG_SMOTE, len(idx_neg))
idx_neg_sub = rng.choice(idx_neg, size=n_neg_use, replace=False)
X_smote = X_train_sc[np.concatenate([idx_pos, idx_neg_sub])]
y_smote = y_train_pool[np.concatenate([idx_pos, idx_neg_sub])]
print(f"  Before SMOTE: {len(idx_pos):,} pos + {n_neg_use:,} neg")
del X_train_sc, y_train_pool

smote = SMOTE(random_state=RANDOM_SEED)
X_res, y_res = smote.fit_resample(X_smote, y_smote)
del X_smote, y_smote
counts = np.bincount(y_res)
print(f"  After SMOTE: {counts[0]:,} neg + {counts[1]:,} pos")

X_train, X_val, y_train, y_val = train_test_split(
    X_res, y_res, test_size=0.125, stratify=y_res, random_state=RANDOM_SEED)
del X_res, y_res
print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}")

# ── Step 9: Train MLP (64->32->1) ──────────────────────────────────────────────
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

print(f"\nTraining MLP ({EPOCHS} epochs, batch=32) ...")
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=EPOCHS, batch_size=32)

# ── Step 10: Save model + preprocessing ───────────────────────────────────────
model.save(os.path.join(OUT_DIR, "classification_mlp_36feat.keras"))
with open(os.path.join(OUT_DIR, "classification_preprocessing.pkl"), "wb") as f:
    pickle.dump({"imputer": imputer, "scaler": scaler,
                 "feature_cols": feature_cols}, f)
print("Model and preprocessing artifacts saved.")

# ── Step 11: Evaluate on true-distribution test set ───────────────────────────
from sklearn.metrics import (classification_report, roc_auc_score,
                              roc_curve, precision_recall_curve,
                              average_precision_score)

print("\nEvaluating on true-distribution test set ...")
y_prob  = model.predict(X_test_sc).ravel()
y_pred  = (y_prob >= 0.5).astype(int)
roc_auc = roc_auc_score(y_test_true, y_prob)
ap      = average_precision_score(y_test_true, y_prob)
print(f"  ROC-AUC = {roc_auc:.4f}  |  AP = {ap:.4f}  |  baseline = {baseline_rate:.4f}")
print(classification_report(y_test_true, y_pred))

with open(os.path.join(OUT_DIR, "shap_classification_pred_cache.pkl"), "wb") as f:
    pickle.dump({"y_test": y_test_true, "y_prob": y_prob,
                 "y_pred": y_pred, "roc_auc": roc_auc, "ap": ap,
                 "baseline": baseline_rate}, f)
print("Prediction cache saved.")

# ROC curve
fpr, tpr, _ = roc_curve(y_test_true, y_prob)
fig_roc, ax_roc = plt.subplots(figsize=(4.5, 4), dpi=300)
ax_roc.plot(fpr, tpr, color="#1f77b4", lw=1.8,
            label=f"MLP (AUC = {roc_auc:.3f})")
ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
ax_roc.set_xlabel("False Positive Rate", fontsize=10)
ax_roc.set_ylabel("True Positive Rate", fontsize=10)
ax_roc.set_title("ROC Curve", fontsize=11)
ax_roc.legend(fontsize=9); ax_roc.tick_params(labelsize=9)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig_roc.savefig(os.path.join(OUT_DIR, f"fig_roc_classification_trainsplit.{ext}"),
                    bbox_inches="tight")
plt.close(fig_roc)
print("ROC curve saved.")

# Precision-Recall curve
prec_vals, rec_vals, _ = precision_recall_curve(y_test_true, y_prob)
fig_pr, ax_pr = plt.subplots(figsize=(4.5, 4), dpi=300)
ax_pr.plot(rec_vals, prec_vals, color="#d62728", lw=1.8,
           label=f"MLP (AP = {ap:.3f})")
ax_pr.axhline(baseline_rate, color="gray", lw=1, linestyle="--",
              label=f"Random ({baseline_rate:.4f})")
ax_pr.set_xlabel("Recall", fontsize=10)
ax_pr.set_ylabel("Precision", fontsize=10)
ax_pr.set_title("Precision-Recall Curve", fontsize=11)
ax_pr.legend(fontsize=9); ax_pr.tick_params(labelsize=9)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig_pr.savefig(os.path.join(OUT_DIR, f"fig_pr_classification_trainsplit.{ext}"),
                   bbox_inches="tight")
plt.close(fig_pr)
print("PR curve saved.")

del X_test_sc, y_test_true; gc.collect()

# ── Step 12: SHAP on SMOTE-balanced training sample ───────────────────────────
import shap

print(f"\nComputing SHAP (background={SHAP_BG_N}, explain={SHAP_EXPL_N}) ...")
rng2     = np.random.default_rng(RANDOM_SEED + 1)
bg_idx   = rng2.choice(len(X_train), size=min(SHAP_BG_N,   len(X_train)), replace=False)
expl_idx = rng2.choice(len(X_train), size=min(SHAP_EXPL_N, len(X_train)), replace=False)
background_np = X_train[bg_idx]
X_explain_np  = X_train[expl_idx]

explainer       = shap.Explainer(model, background_np)
shap_values_raw = explainer(X_explain_np)

with open(os.path.join(OUT_DIR, "shap_classification_cache.pkl"), "wb") as f:
    pickle.dump({"shap_values":  shap_values_raw.values,
                 "base_values":  shap_values_raw.base_values,
                 "X_explain":    X_explain_np,
                 "feature_cols": feature_cols}, f)
print("SHAP cache saved.")

# ── Step 13: Pretty feature names ─────────────────────────────────────────────
SIGNAL_MAP = {
    "CO_MEAN":    "CO",         "NO2_MEAN":   "NO2",
    "PM2_5_MEAN": "PM2.5",     "PM10_MEAN":  "PM10",
    "SO2_MEAN":   "SO2",       "LSTA_MEAN":  "LST (night)",
    "LSTT_MEAN":  "LST (day)", "NTL_MEAN":   "NTL",
    "O3_MEAN":    "O3",
}

def pretty(col):
    for raw, nice in SIGNAL_MAP.items():
        if col == raw:
            return nice
        if col.startswith(raw + "_lag"):
            return f"{nice} (lag {col.replace(raw + '_lag', '')})"
    return col

pretty_names = [pretty(c) for c in feature_cols]

# ── Step 14: Plot SHAP beeswarm (top 15) ──────────────────────────────────────
print("Plotting SHAP beeswarm ...")
shap_exp = shap.Explanation(
    values        = shap_values_raw.values,
    base_values   = shap_values_raw.base_values,
    data          = X_explain_np,
    feature_names = pretty_names,
)
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
plt.sca(ax)
shap.plots.beeswarm(shap_exp, max_display=15, show=False)
ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
ax.tick_params(axis="y", labelsize=8)
ax.tick_params(axis="x", labelsize=9)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT_DIR, f"fig_shap_classification.{ext}"),
                bbox_inches="tight")
plt.close()
print("SHAP beeswarm saved.")

# ── Step 15: Copy all figures to paper ────────────────────────────────────────
import shutil
for fname in ["fig_shap_classification.pdf", "fig_shap_classification.png",
              "fig_roc_classification.pdf",   "fig_roc_classification.png",
              "fig_pr_classification.pdf",    "fig_pr_classification.png"]:
    shutil.copy2(os.path.join(OUT_DIR, fname), os.path.join(PAPER_FIG, fname))
print("All figures copied to paper Figure directory.")
print("\nDone.")
