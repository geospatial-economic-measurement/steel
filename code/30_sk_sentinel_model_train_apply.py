import io, os, sys, warnings, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import os
from _paths import CODEBOOKS, DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_SK, MODEL_ARTIFACTS, REP_ROOT, need

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

BASE        = REP_ROOT
CHINA_SENT  = need(DATA_PROCESSED, 'china_plant_sentinel_2018q4_2022.csv')
RE_DATA     = need(DATA_CONFIDENTIAL, "re_data.csv")
GEM_SK      = need(CODEBOOKS, "sk_gem_plants.csv")
SK_DATA     = need(DATA_SK, 'sk_data_prepared.csv')
OUT_DIR     = os.path.join(DATA_SK, "sk_outputs")
CONF_DIR    = MODEL_ARTIFACTS
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)

WSA_SK    = {2019:71.42, 2020:67.14, 2021:70.38, 2022:65.91, 2023:66.76, 2024:62.04}
BUFFER_KM = 10.0

SIGNALS = [
    "Plant_CO_MEAN", "Plant_NO2_MEAN", "Plant_SO2_MEAN", "Plant_O3_MEAN",
    "Plant_PM10_MEAN", "Plant_PM2_5_MEAN", "Plant_LSTT_MEAN",
    "Plant_LSTA_MEAN", "Plant_NTL_MEAN",
]
FEAT_COLS = (
    [s + suf for s in SIGNALS for suf in ["", "_lag1", "_lag2", "_lag3"]]
    + ["month_sin", "month_cos", "dist_nearest_hub", "dist_nearest_port"]
)

CHINA_HUBS = {
    "Tangshan": (39.63, 118.18), "Wuhan":   (30.59, 114.30),
    "Anshan":   (41.12, 122.99), "Rizhao":  (35.42, 119.53),
    "Baotou":   (40.65, 109.85),
}
CHINA_PORTS = {
    "Shanghai":     (31.23, 121.47), "Tianjin":       (39.00, 117.72),
    "Qingdao":      (36.07, 120.37), "Ningbo":        (29.87, 121.55),
    "Guangzhou":    (23.11, 113.25), "Dalian":        (38.91, 121.64),
    "Lianyungang":  (34.60, 119.22), "Yingkou":       (40.67, 122.23),
}
SK_HUBS = {
    "Pohang":    (36.009, 129.395),
    "Gwangyang": (34.920, 127.749),
    "Dangjin":   (36.986, 126.697),
}
SK_PORTS = {
    "Busan":        (35.10, 129.04), "Incheon":    (37.46, 126.61),
    "Ulsan":        (35.54, 129.32), "Gwangyang_pt":(34.92, 127.75),
    "Pyeongtaek":   (36.97, 126.86), "Pohang_pt":  (36.02, 129.37),
}

def hav(lat1, lon1, lat2, lon2):
    R = 6371.
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def nearest_dist(lat, lon, ref_dict):
    return min(hav(lat, lon, rlat, rlon) for rlat, rlon in ref_dict.values())

model_path = os.path.join(CONF_DIR, "china_plant_sentinel_model.pkl")

# ═══════════════════════════════════════════════════════════════
# PART A — China Sentinel model training
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PART A — China Sentinel model")
print("=" * 60)

if os.path.exists(model_path):
    print(f"Model already exists — loading from {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    pipe      = bundle["pipeline"]
    FEAT_COLS = bundle["feature_cols"]
    print(f"  Loaded. Features: {len(FEAT_COLS)}")
else:
    # 1. Load satellite signals
    sent = pd.read_csv(CHINA_SENT)
    print(f"Sentinel data: {len(sent):,} rows  ({sent['plant_id'].nunique()} plants × "
          f"{sent[['year','month']].drop_duplicates().__len__()} months)")

    # 2. Load Steel_Prod from re_data
    re = pd.read_csv(RE_DATA, usecols=["plant_id", "Year", "Month", "Steel_Prod",
                                        "Longitude", "Latitude"])
    re = re.rename(columns={"Year": "year", "Month": "month"})

    # 3. Merge
    df = sent.merge(re, on=["plant_id", "year", "month"], how="inner")
    print(f"After merge with re_data: {len(df):,} rows  ({df['plant_id'].nunique()} plants)")

    # 4. Compute lags within each plant time series
    df = df.sort_values(["plant_id", "year", "month"]).reset_index(drop=True)
    for sig in SIGNALS:
        if sig not in df.columns:
            df[sig] = np.nan
        for lag in [1, 2, 3]:
            df[f"{sig}_lag{lag}"] = df.groupby("plant_id")[sig].shift(lag)

    # 5. Distance features
    coords = df[["plant_id", "Longitude", "Latitude"]].drop_duplicates("plant_id")
    coords = coords.copy()
    coords["dist_nearest_hub"]  = coords.apply(
        lambda r: nearest_dist(r["Latitude"], r["Longitude"], CHINA_HUBS), axis=1)
    coords["dist_nearest_port"] = coords.apply(
        lambda r: nearest_dist(r["Latitude"], r["Longitude"], CHINA_PORTS), axis=1)
    df = df.merge(coords[["plant_id", "dist_nearest_hub", "dist_nearest_port"]],
                  on="plant_id", how="left")

    # 6. Temporal encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # 7. Target: log1p(Steel_Prod)
    df = df[df["Steel_Prod"] > 0].copy()
    df["target"] = np.log1p(df["Steel_Prod"])
    # Drop rows where ALL lag1/2/3 cols are NaN (e.g. Oct-Dec 2018 have no prior data)
    all_lag_cols = [f"{s}_lag{k}" for s in SIGNALS for k in [1, 2, 3]]
    df = df.dropna(subset=all_lag_cols, how="all").reset_index(drop=True)

    # 8. Train / test split
    train = df[df["year"] <= 2021].copy()
    test  = df[df["year"] == 2022].copy()
    print(f"Train: {len(train):,} rows  Test: {len(test):,} rows")

    X_train = train[FEAT_COLS].values;  y_train = train["target"].values
    X_test  = test[FEAT_COLS].values;   y_test  = test["target"].values

    # 9. XGBoost M1 pipeline
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("xgb", XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        )),
    ])
    pipe.fit(X_train, y_train)

    pred_test  = pipe.predict(X_test)
    r2_test = 1 - np.sum((y_test-pred_test)**2) / np.sum((y_test-y_test.mean())**2)
    r_test, _ = pearsonr(pred_test, y_test)
    mae_test = np.mean(np.abs(np.expm1(pred_test) - np.expm1(y_test)))
    print(f"\nChina Sentinel M1:  Test R²={r2_test:.3f}  r={r_test:.3f}  MAE={mae_test:.1f} kt")

    # Save model
    bundle = {"pipeline": pipe, "feature_cols": FEAT_COLS}
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Model saved -> {model_path}")

print(f"\nUsing model: {model_path}")

# ═══════════════════════════════════════════════════════════════
# PART B — SK plant-level prediction
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("PART B — SK plant-level prediction (Sentinel model)")
print("=" * 60)

# Load SK GEM plants
gem = pd.read_csv(GEM_SK)
gem = gem[gem["status"] == "operating"].reset_index(drop=True)
print(f"SK GEM operating plants: {len(gem)}")

# SK distance features
gem["dist_nearest_hub"]  = gem.apply(
    lambda r: nearest_dist(r["lat"], r["lon"], SK_HUBS), axis=1)
gem["dist_nearest_port"] = gem.apply(
    lambda r: nearest_dist(r["lat"], r["lon"], SK_PORTS), axis=1)

plant_lats = gem["lat"].values
plant_lons = gem["lon"].values
n_plants   = len(gem)
lat_margin = BUFFER_KM / 111.0
lon_margin = BUFFER_KM / (111.0 * np.cos(np.radians(36.5)))

# Check SK data columns
sk_sample = pd.read_csv(SK_DATA, nrows=2)
print(f"SK data columns: {sk_sample.columns.tolist()}")

# Map SK column names → Plant_* names used in model
# sk_data_prepared has buffer-aggregated means: CO_MEAN, NO2_MEAN etc.
# with lags already present as CO_MEAN_lag1 etc.
SK_COL_MAP = {
    "CO_MEAN":       "Plant_CO_MEAN",
    "NO2_MEAN":      "Plant_NO2_MEAN",
    "SO2_MEAN":      "Plant_SO2_MEAN",
    "O3_MEAN":       "Plant_O3_MEAN",
    "PM10_MEAN":     "Plant_PM10_MEAN",
    "PM2_5_MEAN":    "Plant_PM2_5_MEAN",
    "LSTT_MEAN":     "Plant_LSTT_MEAN",
    "LSTA_MEAN":     "Plant_LSTA_MEAN",
    "NTL_MEAN":      "Plant_NTL_MEAN",
}
# Also map lag columns
for base_sk, base_pl in list(SK_COL_MAP.items()):
    for lag in [1, 2, 3]:
        SK_COL_MAP[f"{base_sk}_lag{lag}"] = f"{base_pl}_lag{lag}"

SK_ALL_FEATS = list(SK_COL_MAP.keys())

# Buffer extraction from sk_data_prepared.csv
acc_sum   = defaultdict(lambda: defaultdict(lambda: np.zeros(len(SK_ALL_FEATS))))
acc_count = defaultdict(lambda: defaultdict(lambda: np.zeros(len(SK_ALL_FEATS), dtype=int)))

chunk_size = 500_000
total_rows = matched_rows = 0

print(f"\nExtracting {BUFFER_KM} km buffers from SK data...")
for ci, chunk in enumerate(pd.read_csv(SK_DATA, chunksize=chunk_size)):
    total_rows += len(chunk)
    if ci % 5 == 0:
        print(f"  chunk {ci:3d} | rows: {total_rows:,} | matched: {matched_rows:,}")

    # Detect lat/lon column names
    lat_col = "latitude" if "latitude" in chunk.columns else "lat"
    lon_col = "longitude" if "longitude" in chunk.columns else "lon"
    yr_col  = "Year"  if "Year"  in chunk.columns else "year"
    mo_col  = "Month" if "Month" in chunk.columns else "month"

    lats = chunk[lat_col].values
    lons = chunk[lon_col].values

    for pi in range(n_plants):
        plat, plon = plant_lats[pi], plant_lons[pi]
        mask = ((lats >= plat-lat_margin) & (lats <= plat+lat_margin) &
                (lons >= plon-lon_margin) & (lons <= plon+lon_margin))
        if not mask.any():
            continue
        sub = chunk[mask].copy()
        dists = hav(sub[lat_col].values, sub[lon_col].values, plat, plon)
        sub = sub[dists <= BUFFER_KM]
        if len(sub) == 0:
            continue
        matched_rows += len(sub)

        present = [c for c in SK_ALL_FEATS if c in sub.columns]
        for _, grp in sub.groupby([yr_col, mo_col]):
            ym  = (int(grp[yr_col].iloc[0]), int(grp[mo_col].iloc[0]))
            idx = [SK_ALL_FEATS.index(c) for c in present]
            vals = grp[present].values.astype(float)
            acc_sum[pi][ym][idx]   += np.nansum(vals, axis=0)
            acc_count[pi][ym][idx] += np.sum(~np.isnan(vals), axis=0)

print(f"\nDone. Total: {total_rows:,} | Matched: {matched_rows:,}")

# Build plant × month DataFrame with Plant_* column names
records = []
for pi in range(n_plants):
    for (yr, mo), sig_sum in acc_sum[pi].items():
        cnt = acc_count[pi][(yr, mo)]
        sig_mean = np.where(cnt > 0, sig_sum / np.maximum(cnt, 1), np.nan)
        rec = {"plant_name": gem.iloc[pi]["plant_name"], "Year": yr, "Month": mo}
        for j, sk_col in enumerate(SK_ALL_FEATS):
            pl_col = SK_COL_MAP[sk_col]
            rec[pl_col] = sig_mean[j]
        records.append(rec)

plant_sig = pd.DataFrame(records)
print(f"Plant-signal rows: {len(plant_sig):,}  "
      f"({plant_sig['plant_name'].nunique()} plants)")

# ── Extract 2018 Q4 from monthly SK files to provide real lag context ────────
SK_OUT_DIR = DATA_SK
RAW_TO_PLANT = {
    "CO":  "Plant_CO_MEAN",  "NO2": "Plant_NO2_MEAN",  "SO2": "Plant_SO2_MEAN",
    "O3":  "Plant_O3_MEAN",  "PM10":"Plant_PM10_MEAN",  "PM25":"Plant_PM2_5_MEAN",
    "LST": "Plant_LSTT_MEAN","LSTA":"Plant_LSTA_MEAN",  "NTL": "Plant_NTL_MEAN",
}
patch_records = []
missing_q4 = []
for yr, mo in [(2018, 10), (2018, 11), (2018, 12)]:
    fpath = os.path.join(SK_OUT_DIR, f"sk_{yr}_{mo:02d}.csv")
    if not os.path.exists(fpath):
        missing_q4.append(os.path.basename(fpath)); continue
    raw = pd.read_csv(fpath)
    rlats = raw["lat"].values; rlons = raw["lon"].values
    for pi in range(n_plants):
        plat, plon = plant_lats[pi], plant_lons[pi]
        mask = ((rlats >= plat-lat_margin) & (rlats <= plat+lat_margin) &
                (rlons >= plon-lon_margin) & (rlons <= plon+lon_margin))
        if not mask.any(): continue
        sub = raw[mask].copy()
        dists = hav(sub["lat"].values, sub["lon"].values, plat, plon)
        sub = sub[dists <= BUFFER_KM]
        if len(sub) == 0: continue
        rec = {"plant_name": gem.iloc[pi]["plant_name"], "Year": yr, "Month": mo}
        for raw_col, pl_col in RAW_TO_PLANT.items():
            if raw_col in sub.columns:
                vals = sub[raw_col].values.astype(float)
                rec[pl_col] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else np.nan
        patch_records.append(rec)

if patch_records:
    patch_df = pd.DataFrame(patch_records)
    print(f"2018 Q4 SK extraction: {len(patch_df)} plant-month rows")
    # Keep only base signal columns (no pre-computed lags)
    base_pl_cols = list(RAW_TO_PLANT.values())
    keep = ["plant_name", "Year", "Month"] + [c for c in base_pl_cols if c in patch_df.columns]
    # Also drop pre-computed lag cols from main extraction before combining
    drop_lag_cols = [c for c in plant_sig.columns if "_lag" in c]
    plant_sig = plant_sig.drop(columns=drop_lag_cols, errors="ignore")
    plant_sig = pd.concat([plant_sig, patch_df[keep]], ignore_index=True)
    plant_sig = plant_sig.sort_values(["plant_name", "Year", "Month"]).reset_index(drop=True)
    print(f"Combined rows (incl. 2018 Q4): {len(plant_sig):,}")
else:
    raise SystemExit(
        "\nABORTING: the 2018 Q4 lag patch is missing.\n"
        "  not found in data/SK: %s\n\n"
        "  Jan-Mar 2019 depend on Oct-Dec 2018 for their lag features. Without\n"
        "  them the lags are imputed and the 2019 national total comes out at\n"
        "  ~12,995 kt instead of 12,480, which moves Table 1's first row from\n"
        "  -3.2%% to -7.0%% and Pearson r from 0.981 to 0.872.\n\n"
        "  Run 29_sk_china_sentinel_2018q4_patch_gee.py first (needs Google\n"
        "  Earth Engine credentials). See 'Careful with 2019' in the README."
        % (", ".join(missing_q4) or "sk_2018_10.csv, sk_2018_11.csv, sk_2018_12.csv"))

# Always recompute lags from the full time series (ensures 2018 Q4 is used as context)
print("Recomputing lags from combined time series...")
plant_sig = plant_sig.sort_values(["plant_name", "Year", "Month"]).reset_index(drop=True)
for sig in SIGNALS:
    if sig not in plant_sig.columns:
        plant_sig[sig] = np.nan
    for lag in [1, 2, 3]:
        plant_sig[f"{sig}_lag{lag}"] = plant_sig.groupby("plant_name")[sig].shift(lag)

# Drop 2018 rows — they were only needed to supply lag context for Jan-Mar 2019
plant_sig = plant_sig[plant_sig["Year"] >= 2019].reset_index(drop=True)
print(f"After dropping 2018 context rows: {len(plant_sig):,}")

# Merge SK distance features
plant_sig = plant_sig.merge(
    gem[["plant_name", "dist_nearest_hub", "dist_nearest_port"]],
    on="plant_name", how="left"
)
plant_sig["month_sin"] = np.sin(2 * np.pi * plant_sig["Month"] / 12)
plant_sig["month_cos"] = np.cos(2 * np.pi * plant_sig["Month"] / 12)

# Add any missing feature columns as NaN
for c in FEAT_COLS:
    if c not in plant_sig.columns:
        plant_sig[c] = np.nan

# Predict
X_sk = plant_sig[FEAT_COLS].values
plant_sig["pred_log"] = pipe.predict(X_sk)
plant_sig["pred_kt"]  = np.expm1(plant_sig["pred_log"])

print(f"\nPredicted range: {plant_sig['pred_kt'].min():.1f} – "
      f"{plant_sig['pred_kt'].max():.1f} kt/month")

# ── Annual aggregation & growth rates ──────────────────────────────────────
annual = plant_sig.groupby(["plant_name", "Year"])["pred_kt"].sum().reset_index()
annual.columns = ["plant_name", "Year", "pred_kt_annual"]
nat = annual.groupby("Year")["pred_kt_annual"].sum()
yrs = sorted(nat.index)
transitions = [f"{yrs[i]}->{yrs[i+1]}" for i in range(len(yrs)-1)]

print()
print("=" * 60)
print("SK GROWTH RATE RESULTS (Sentinel M1)")
print("=" * 60)
print(f"  {'Transition':>12}  {'Model%':>8}  {'WSA%':>8}  {'Diff':>8}")
print("  " + "-" * 45)

model_gr, wsa_gr = [], []
for t in transitions:
    y0, y1 = int(t.split("->")[0]), int(t.split("->")[1])
    if y0 not in nat.index or y1 not in nat.index:
        continue
    mg = (nat[y1] - nat[y0]) / nat[y0] * 100
    if y0 in WSA_SK and y1 in WSA_SK:
        wg = (WSA_SK[y1] - WSA_SK[y0]) / WSA_SK[y0] * 100
        wsa_gr.append(wg)
        model_gr.append(mg)
        diff = mg - wg
        print(f"  {t:>12}  {mg:>7.1f}%  {wg:>7.1f}%  {diff:>+7.1f}pp")

if len(model_gr) >= 2:
    r_gr, _ = pearsonr(model_gr, wsa_gr)
    mae_gr  = np.mean(np.abs(np.array(model_gr) - np.array(wsa_gr)))
    print(f"\n  Pearson r = {r_gr:.3f}   MAE = {mae_gr:.1f}pp   (n={len(model_gr)})")

# Per-plant growth table
print()
print("Per-plant YoY growth rates (%):")
gr_rows = []
for pname, grp in annual.groupby("plant_name"):
    grp = grp.sort_values("Year")
    row = {"Plant": pname}
    for i in range(len(grp)-1):
        y0r, y1r = grp.iloc[i], grp.iloc[i+1]
        if y1r["Year"] != y0r["Year"] + 1:
            continue
        v0 = y0r["pred_kt_annual"]; v1 = y1r["pred_kt_annual"]
        if not v0 or v0 == 0:
            continue
        t = f"{int(y0r['Year'])}->{int(y1r['Year'])}"
        row[t] = round((v1 - v0) / v0 * 100, 1)
    gr_rows.append(row)

gr_df = pd.DataFrame(gr_rows).set_index("Plant")
wsa_row_vals = {t: round((WSA_SK[int(t.split("->")[1])] - WSA_SK[int(t.split("->")[0])]) /
                         WSA_SK[int(t.split("->")[0])] * 100, 1)
                for t in gr_df.columns if int(t.split("->")[0]) in WSA_SK}
gr_df.loc["WSA actual"] = wsa_row_vals
print(gr_df.to_string())

# Save outputs
# Published national totals (kt) behind Table 1; guards against a silently
# different 2019, which is the failure mode when the Q4-2018 patch is absent.
PUBLISHED_TOTALS = {2019: 12480.3, 2020: 12084.1, 2021: 12142.9,
                    2022: 11682.5, 2023: 11694.6, 2024: 11356.6}
_nat = plant_sig.groupby("Year")["pred_kt"].sum()
_off = {int(y): (float(_nat[y]), v) for y, v in PUBLISHED_TOTALS.items()
        if y in _nat.index and abs(float(_nat[y]) - v) > 1.0}
if _off:
    print("\n  WARNING: national totals differ from the published series:")
    for y, (got, want) in sorted(_off.items()):
        print("    %d  got %.1f kt, published %.1f kt" % (y, got, want))
    print("  Table 1 and Figure 8b will not match the paper. Check that\n"
          "  29_sk_china_sentinel_2018q4_patch_gee.py ran first.")
else:
    print("  national totals match the published series")

plant_sig[["plant_name", "Year", "Month", "pred_kt"]].to_csv(
    os.path.join(OUT_DIR, "sk_plant_sentinel_monthly.csv"), index=False)
gr_df.reset_index().to_csv(
    os.path.join(OUT_DIR, "sk_plant_sentinel_growth_rates.csv"), index=False)

print(f"\nOutputs saved to {OUT_DIR}/sk_plant_sentinel_*.csv")
print("DONE")
