

import os, warnings, gc, pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, OUT_FIG, SCRATCH, need

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

OUT_DIR      = OUT_FIG
PAPER_FIG    = OUT_FIG
MEMMAP_PATH  = os.path.join(SCRATCH, "tmp_X_all.dat")

CHUNKSIZE  = 500_000
BATCH_SIZE = 1_000_000

SIGNALS = ['CO_MEAN', 'NO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN', 'SO2_MEAN',
           'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN', 'O3_MEAN']

# ── Load preprocessing artifacts ──────────────────────────────────────────────
print("Loading preprocessing artifacts ...")
with open(os.path.join(OUT_DIR, "classification_preprocessing.pkl"), "rb") as f:
    prep = pickle.load(f)
imputer      = prep["imputer"]
scaler       = prep["scaler"]
feature_cols = prep["feature_cols"]
print(f"  Preprocessing loaded. Features: {len(feature_cols)}")

# ── Pre-scan CSVs to get total row count ──────────────────────────────────────
print("Pre-scanning CSVs for row counts ...")
file_rows = {}
for fpath in [CSV1, CSV2]:
    if not os.path.exists(fpath):
        print(f"  [skip] {os.path.basename(fpath)}")
        file_rows[fpath] = 0
        continue
    n = 0
    for chunk in pd.read_csv(fpath, chunksize=CHUNKSIZE, usecols=['IDCode']):
        n += len(chunk)
    file_rows[fpath] = n
    print(f"  {os.path.basename(fpath)}: {n:,} rows")

n_total_raw = sum(file_rows.values())
print(f"  Grand total: {n_total_raw:,} rows")

# ── Pre-allocate output arrays (no list accumulation) ─────────────────────────
print("Pre-allocating arrays ...")
id_arr  = np.empty(n_total_raw, dtype=np.int32)
yr_arr  = np.empty(n_total_raw, dtype=np.int16)
mo_arr  = np.empty(n_total_raw, dtype=np.int8)
sig_arr = np.empty((n_total_raw, len(SIGNALS)), dtype=np.float32)
print(f"  id_arr+yr+mo: {(id_arr.nbytes+yr_arr.nbytes+mo_arr.nbytes)/1e9:.2f} GB")
print(f"  sig_arr: {sig_arr.nbytes/1e9:.2f} GB")

# ── Fill arrays directly from CSV chunks ──────────────────────────────────────
print("Loading all data ...")
offset = 0
for fpath in [CSV1, CSV2]:
    if file_rows[fpath] == 0:
        continue
    loaded = 0
    for chunk in pd.read_csv(
            fpath, chunksize=CHUNKSIZE, low_memory=False,
            usecols=['IDCode', 'Year', 'Month'] + SIGNALS,
            dtype={'IDCode': np.int32, 'Year': np.int16, 'Month': np.int8,
                   **{col: np.float32 for col in SIGNALS}}):
        n = len(chunk)
        id_arr [offset : offset+n] = chunk['IDCode'].to_numpy()
        yr_arr [offset : offset+n] = chunk['Year'].to_numpy()
        mo_arr [offset : offset+n] = chunk['Month'].to_numpy()
        sig_arr[offset : offset+n] = chunk[SIGNALS].to_numpy(dtype=np.float32)
        offset += n
        loaded += n
    print(f"  {os.path.basename(fpath)}: {loaded:,} rows")

gc.collect()
print(f"  Total loaded: {offset:,} rows")

# ── Sort ───────────────────────────────────────────────────────────────────────
print("Sorting ...")
order   = np.lexsort((mo_arr, yr_arr, id_arr))
id_arr  = id_arr[order];  yr_arr  = yr_arr[order]
mo_arr  = mo_arr[order];  sig_arr = sig_arr[order]
del order; gc.collect()

# ── Compute lag features into memory-mapped file (avoids 15 GiB RAM) ──────────
print("Computing lag features (disk-backed memmap) ...")
n_signals = len(SIGNALS)
n_lags    = 3
n_feats   = n_signals * (1 + n_lags)   # 36

print(f"  Allocating memmap: {n_total_raw:,} × {n_feats} @ float32 ...")
X_mm = np.memmap(MEMMAP_PATH, dtype=np.float32, mode='w+',
                 shape=(n_total_raw, n_feats))

X_mm[:, :n_signals] = sig_arr   # contemporaneous

boundaries   = np.where(np.diff(id_arr) != 0)[0] + 1
group_starts = np.concatenate([[0], boundaries])
group_ends   = np.concatenate([boundaries, [n_total_raw]])

for lag in range(1, n_lags + 1):
    col_offset = n_signals * lag
    X_mm[:, col_offset : col_offset + n_signals] = np.nan
    for gs, ge in zip(group_starts, group_ends):
        if ge - gs > lag:
            X_mm[gs + lag : ge, col_offset : col_offset + n_signals] = sig_arr[gs : ge - lag]

X_mm.flush()
del sig_arr; gc.collect()
print(f"  Memmap written. Disk size: {X_mm.nbytes/1e9:.1f} GB")

# ── Detect valid rows (no NaN) ─────────────────────────────────────────────────
print("Scanning for NaN rows ...")
valid_mask = np.ones(n_total_raw, dtype=bool)
for start in range(0, n_total_raw, BATCH_SIZE):
    end   = min(start + BATCH_SIZE, n_total_raw)
    chunk = np.array(X_mm[start:end])
    valid_mask[start:end] = ~np.isnan(chunk).any(axis=1)
    del chunk

n_valid = valid_mask.sum()
print(f"  Valid rows: {n_valid:,}  (dropped {n_total_raw - n_valid:,} NaN rows)")

id_v   = id_arr[valid_mask];  yr_v = yr_arr[valid_mask];  mo_v = mo_arr[valid_mask]
del id_arr, yr_arr, mo_arr; gc.collect()

# ── Merge GridProd labels ──────────────────────────────────────────────────────
print("Merging labels ...")
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
y_all   = np.zeros(n_valid, dtype=np.int8)
y_all[np.isin(key_int, list(pos_keys))] = 1
del key_int, id_v, yr_v, mo_v, pos_keys; gc.collect()
n_pos    = int(y_all.sum())
baseline = float(y_all.mean())
print(f"  Positive rate: {baseline:.6f}  ({n_pos:,} pos / {n_valid:,} total)")

# ── Load model now (X_mm is on disk; sig_arr is freed) ────────────────────────
from tensorflow.keras.models import load_model
print("Loading saved model ...")
model = load_model(os.path.join(OUT_DIR, "classification_mlp_36feat.keras"))
print("  Model loaded.")

# ── Score all valid rows in batches ───────────────────────────────────────────
print(f"Scoring {n_valid:,} rows in batches of {BATCH_SIZE:,} ...")
y_prob        = np.empty(n_valid, dtype=np.float32)
valid_indices = np.where(valid_mask)[0]
del valid_mask; gc.collect()

for i, start in enumerate(range(0, n_valid, BATCH_SIZE)):
    end     = min(start + BATCH_SIZE, n_valid)
    raw_idx = valid_indices[start:end]
    batch   = np.array(X_mm[raw_idx])
    batch   = imputer.transform(batch)
    batch   = scaler.transform(batch)
    y_prob[start:end] = model.predict(batch, verbose=0).ravel()
    del batch
    if (i + 1) % 10 == 0 or end == n_valid:
        print(f"  Scored {end:,} / {n_valid:,}")

print("  Done scoring.")
del valid_indices, X_mm; gc.collect()

# ── Delete temp memmap ─────────────────────────────────────────────────────────
try:
    os.remove(MEMMAP_PATH)
    print("Temp memmap file deleted.")
except Exception as e:
    print(f"  Could not delete memmap: {e}")

# ── Metrics ───────────────────────────────────────────────────────────────────
from sklearn.metrics import (classification_report, roc_auc_score,
                              roc_curve, precision_recall_curve,
                              average_precision_score)

print("\nComputing metrics on full population ...")
y_pred  = (y_prob >= 0.5).astype(np.int8)
roc_auc = roc_auc_score(y_all, y_prob)
ap      = average_precision_score(y_all, y_prob)
print(f"  ROC-AUC = {roc_auc:.4f}  |  AP = {ap:.6f}  |  baseline = {baseline:.6f}")
print(classification_report(y_all, y_pred))

# Save updated prediction cache
with open(os.path.join(OUT_DIR, "shap_classification_pred_cache.pkl"), "wb") as f:
    pickle.dump({"y_test": y_all, "y_prob": y_prob,
                 "y_pred": y_pred, "roc_auc": roc_auc,
                 "ap": ap, "baseline": baseline,
                 "note": f"Full population: {n_valid:,} rows"}, f)
print("Prediction cache updated.")

# ── ROC curve ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_all, y_prob)
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
    fig_roc.savefig(os.path.join(OUT_DIR, f"fig_roc_classification.{ext}"),
                    bbox_inches="tight")
plt.close(fig_roc)
print("ROC curve saved.")

# ── PR curve ──────────────────────────────────────────────────────────────────
prec_vals, rec_vals, _ = precision_recall_curve(y_all, y_prob)
fig_pr, ax_pr = plt.subplots(figsize=(4.5, 4), dpi=300)
ax_pr.plot(rec_vals, prec_vals, color="#d62728", lw=1.8,
           label=f"MLP (AP = {ap:.4f})")
ax_pr.axhline(baseline, color="gray", lw=1, linestyle="--",
              label=f"Random ({baseline:.6f})")
ax_pr.set_xlabel("Recall", fontsize=10)
ax_pr.set_ylabel("Precision", fontsize=10)
ax_pr.set_title("Precision-Recall Curve", fontsize=11)
ax_pr.legend(fontsize=9); ax_pr.tick_params(labelsize=9)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig_pr.savefig(os.path.join(OUT_DIR, f"fig_pr_classification.{ext}"),
                   bbox_inches="tight")
plt.close(fig_pr)
print("PR curve saved.")

# ── Copy to paper ─────────────────────────────────────────────────────────────
import shutil
for fname in ["fig_roc_classification.pdf", "fig_roc_classification.png",
              "fig_pr_classification.pdf",   "fig_pr_classification.png"]:
    shutil.copy2(os.path.join(OUT_DIR, fname), os.path.join(PAPER_FIG, fname))
print("Figures copied to paper Figure directory.")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"""
=== UPDATE PAPER WITH THESE VALUES ===
ROC-AUC  : {roc_auc:.3f}
AP       : {ap:.4f}
Baseline : {baseline:.6f}  (~1 in {1/baseline:.0f})
N pos    : {n_pos:,}
N total  : {n_valid:,}
N neg    : {n_valid - n_pos:,}
======================================
""")
print("Done.")
