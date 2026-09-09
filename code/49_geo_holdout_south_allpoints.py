"""Figure 7b (rebuild): southern-China geographic holdout, ALL observations shown.

Same data, model and matched-month construction as
`21_holdout_validation_temporal_geo_figures.py`. The only change is display:

  * every plant-year pair is plotted, including the ones the matched-month
    rules exclude from the statistics (<8 matched months, small base,
    |actual growth| > 200%);
  * the reported r / R^2 / n still come from the valid retained sample, so the
    headline statistic is unchanged from the shipped figure;
  * excluded points are drawn in a distinct style and annotated, and symlog
    axes keep the central cloud legible while showing the extremes.

Rationale: the LEDGER records that the older full-sample statistic
(R^2 = 0.840, r = 0.982, n = 120) was artifact-driven -- one pair with a single
matched month (+1077% / +1455%) plus 2019 short-year inflation common to both
series. Those artifacts are excluded from the statistics here, but the points
are no longer hidden from the reader.

DIAGNOSTIC ONLY -- this is not the paper figure. The paper's Figure 7b /
Figure_08b uses the full-sample construction and is produced by
`23_geo_holdout_south_full_symlog_paper.py`.

Outputs: fig_geo_holdout_south_matched_allpoints.pdf/.png
         table_s_geo_holdout_south_allpoints.csv
         -> Replication/output_rl/
"""
import io
import os
import sys
import warnings

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_RL, need

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── Paths (identical to script 21) ────────────────────────────────────────────
SIGNALS_PATH = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')

# DIAGNOSTIC ONLY. This script must not write Figure_7b / Figure_08b -- the
# paper figure is produced by 23_geo_holdout_south_full_symlog_paper.py.
OUT_DIRS = [OUT_RL]
FIG_NAME = "fig_geo_holdout_south_matched_allpoints"

EARTH_R = 6371.0
CHINA_HUBS = {
    "Tangshan": (39.6, 118.2), "Wuhan": (30.6, 114.3), "Anshan": (41.1, 122.8),
    "Rizhao": (35.4, 119.5), "Baotou": (40.7, 109.8),
}
CHINA_PORTS = {
    "Shanghai": (31.23, 121.47), "Tianjin": (38.98, 117.72),
    "Qingdao": (36.07, 120.38), "Ningbo": (29.87, 121.55),
    "Guangzhou": (23.10, 113.43), "Dalian": (38.92, 121.65),
    "Lianyungang": (34.75, 119.45), "Yingkou": (40.67, 122.23),
}

POLLUTANTS = ["CO_MEAN", "NO2_MEAN", "SO2_MEAN", "PM2_5_MEAN", "PM10_MEAN",
              "O3_MEAN", "LSTA_MEAN", "LSTT_MEAN", "NTL_MEAN"]
PLANT_POLLUTANTS = ["Plant_%s" % p for p in POLLUTANTS]
FEATURE_COLS = (
    PLANT_POLLUTANTS
    + ["%s_lag%d" % (p, k) for k in (1, 2, 3) for p in PLANT_POLLUTANTS]
    + ["month_sin", "month_cos", "dist_nearest_hub", "dist_nearest_port", "is_bof"]
)
XGB_PARAMS = dict(
    colsample_bytree=0.8, learning_rate=0.1, max_depth=6,
    n_estimators=300, subsample=0.6,
    alpha=0.2, reg_lambda=0.5, random_state=5, verbosity=0,
)

MIN_MATCHED_MONTHS = 8
TRIM = 2.0
LINTHRESH = 25.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def add_dist_features(df):
    for name, (la, lo) in CHINA_HUBS.items():
        df["d_hub_%s" % name] = haversine_km(df["Latitude"], df["Longitude"], la, lo)
    for name, (la, lo) in CHINA_PORTS.items():
        df["d_port_%s" % name] = haversine_km(df["Latitude"], df["Longitude"], la, lo)
    df["dist_nearest_hub"] = df[["d_hub_%s" % k for k in CHINA_HUBS]].min(axis=1)
    df["dist_nearest_port"] = df[["d_port_%s" % k for k in CHINA_PORTS]].min(axis=1)
    return df


# ── Load and prepare data (identical pipeline to script 21) ───────────────────
print("Loading data ...")
signals = pd.read_csv(SIGNALS_PATH).drop(columns=["PM1_MEAN"], errors="ignore")
gridprod = pd.read_csv(GRIDPROD_PATH).drop_duplicates(
    subset=["IDCode", "name_prod", "Year", "Month"])

bf_sheet = pd.read_excel(GEM_PATH, sheet_name="Blast furnaces", header=0)
bf_gem_ids = set(bf_sheet.iloc[:, 0].dropna().astype(str).unique())
gem_tech = gridprod[["name_prod", "GEMPlantID"]].drop_duplicates("name_prod").copy()
gem_tech["is_bof"] = gem_tech["GEMPlantID"].apply(
    lambda x: 1 if str(x) in bf_gem_ids else 0)

temp = gridprod[["IDCode", "Longitude", "Latitude", "name_prod",
                 "Year", "Month", "Steel_Prod"]].copy()
merged = pd.merge(signals, temp, on=["IDCode", "Year", "Month"], how="left")

grouped = (merged.groupby(["name_prod", "Year", "Month"])
           .agg({**{c: "mean" for c in POLLUTANTS},
                 "Longitude": "mean", "Latitude": "mean", "Steel_Prod": "first"})
           .reset_index())
grouped.rename(columns={p: "Plant_%s" % p for p in POLLUTANTS}, inplace=True)
grouped = grouped.drop_duplicates(subset=["name_prod", "Year", "Month"])
grouped["plant_id"] = grouped["name_prod"].factorize()[0] + 1
grouped = grouped.merge(gem_tech[["name_prod", "is_bof"]], on="name_prod", how="left")
grouped["is_bof"] = grouped["is_bof"].fillna(0).astype(int)

grouped["month_sin"] = np.sin(2 * np.pi * grouped["Month"] / 12)
grouped["month_cos"] = np.cos(2 * np.pi * grouped["Month"] / 12)
grouped = grouped.sort_values(["plant_id", "Year", "Month"]).reset_index(drop=True)
for lag in range(1, 4):
    for col in PLANT_POLLUTANTS:
        grouped["%s_lag%d" % (col, lag)] = grouped.groupby("plant_id")[col].shift(lag)
grouped = add_dist_features(grouped)

data = grouped.dropna(subset=FEATURE_COLS + ["Steel_Prod", "Year", "Latitude"]).copy()
data = data[data["Steel_Prod"] > 0].reset_index(drop=True)
print("  After dropna: %d plants, %d rows" % (data["name_prod"].nunique(), len(data)))

north = data[data["Latitude"] >= 35].copy()
south = data[data["Latitude"] < 35].copy()
print("  Train (north >=35N): %d plants | Test (south <35N): %d plants"
      % (north["name_prod"].nunique(), south["name_prod"].nunique()))

# ── Fit on north, predict south ───────────────────────────────────────────────
model = xgb.XGBRegressor(**XGB_PARAMS)
model.fit(north[FEATURE_COLS].values, np.log1p(north["Steel_Prod"].values))
test = south.copy()
test["pred"] = np.expm1(model.predict(test[FEATURE_COLS].values))

# ── Matched-month annual growth, KEEPING the excluded pairs and flagging them ──
cur = test[["name_prod", "Year", "Month", "Steel_Prod", "pred"]].copy()
base = cur.rename(columns={"Steel_Prod": "base_actual", "pred": "base_pred"})
base["Year"] = base["Year"] + 1
pairs = cur.merge(base, on=["name_prod", "Year", "Month"], how="inner")
pairs = pairs.rename(columns={"Steel_Prod": "cur_actual", "pred": "cur_pred"})

ann = (pairs.groupby(["name_prod", "Year"])
       .agg(n_matched_months=("Month", "nunique"),
            cur_actual=("cur_actual", "sum"), base_actual=("base_actual", "sum"),
            cur_pred=("cur_pred", "sum"), base_pred=("base_pred", "sum"))
       .reset_index())
print("  Raw matched-month annual pairs: %d" % len(ann))

ann["insufficient"] = ann["n_matched_months"] < MIN_MATCHED_MONTHS

# Small-base guard is defined on the sufficient-months subset, exactly as in script 21.
suff = ann[~ann["insufficient"]]
p1_ann = np.percentile(
    np.concatenate([suff["cur_actual"].values, suff["base_actual"].values]), 1)
ann["small_base"] = (~ann["insufficient"]) & (ann["base_actual"] < p1_ann)

ann["actual_gr"] = ann["cur_actual"] / ann["base_actual"] - 1
ann["pred_gr"] = ann["cur_pred"] / ann["base_pred"] - 1
ann["nonfinite"] = ~(np.isfinite(ann["actual_gr"]) & np.isfinite(ann["pred_gr"]))
ann["trimmed"] = (~ann["insufficient"]) & (~ann["small_base"]) & \
                 (~ann["nonfinite"]) & (ann["actual_gr"].abs() > TRIM)

ann["retained"] = ~(ann["insufficient"] | ann["small_base"] |
                    ann["nonfinite"] | ann["trimmed"])

kept = ann[ann["retained"]]
dropped = ann[~ann["retained"] & ~ann["nonfinite"]]

print("  Excluded: %d insufficient months, %d small base, %d nonfinite, %d >|%d%%|"
      % (int(ann["insufficient"].sum()), int(ann["small_base"].sum()),
         int(ann["nonfinite"].sum()), int(ann["trimmed"].sum()), int(TRIM * 100)))
print("  Retained (statistics sample): n = %d" % len(kept))

r2_kept = r2_score(kept["actual_gr"], kept["pred_gr"])
r_kept = pearsonr(kept["actual_gr"], kept["pred_gr"])[0]
print("  RETAINED  R2 = %+.3f   r = %+.3f   n = %d" % (r2_kept, r_kept, len(kept)))

if len(dropped):
    print("  Points shown but excluded from statistics:")
    for _, row in dropped.iterrows():
        reason = ("only %d matched month(s)" % row["n_matched_months"]
                  if row["insufficient"] else
                  "small base year" if row["small_base"] else
                  "|actual growth| > %d%%" % int(TRIM * 100))
        print("     %-14s %d  actual %+9.1f%%  pred %+9.1f%%   [%s]"
              % (row["name_prod"], int(row["Year"]), row["actual_gr"] * 100,
                 row["pred_gr"] * 100, reason))

# ── Figure: all points shown, statistics from the retained sample ─────────────
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

ax.scatter(kept["actual_gr"] * 100, kept["pred_gr"] * 100,
           alpha=0.6, color="blue", s=25, zorder=3,
           label="Plant-year observation (n = %d)" % len(kept))

if len(dropped):
    ax.scatter(dropped["actual_gr"] * 100, dropped["pred_gr"] * 100,
               facecolors="none", edgecolors="dimgray", s=70, linewidths=1.4,
               zorder=4, label="Excluded from statistics (n = %d)" % len(dropped))
    for _, row in dropped.iterrows():
        if row["insufficient"]:
            note = "%d matched month%s" % (
                row["n_matched_months"], "" if row["n_matched_months"] == 1 else "s")
        elif row["small_base"]:
            note = "small base year"
        else:
            note = "|growth| > %d%%" % int(TRIM * 100)
        # Both excluded points sit near y = 0; fan the labels apart so they
        # do not collide with each other or with the central cloud.
        if row["actual_gr"] < 0:
            off, ha = (-10, -20), "right"
        else:
            off, ha = (10, 16), "left"
        ax.annotate(note, (row["actual_gr"] * 100, row["pred_gr"] * 100),
                    textcoords="offset points", xytext=off,
                    fontsize=8, color="dimgray", ha=ha,
                    arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.6))

allx = np.concatenate([kept["actual_gr"].values, dropped["actual_gr"].values]) * 100
ally = np.concatenate([kept["pred_gr"].values, dropped["pred_gr"].values]) * 100
lim = max(np.abs(allx).max(), np.abs(ally).max()) * 1.15
ax.plot([-lim, lim], [-lim, lim], "r--", lw=2, zorder=2,
        label="45$^\\circ$ line (perfect fit)")
ax.axhline(0, color="gray", lw=0.8, linestyle=":", zorder=1)
ax.axvline(0, color="gray", lw=0.8, linestyle=":", zorder=1)

# Linear axes: under the matched-month construction the range is modest
# (max |growth| well under 100%), so symlog would only compress the cloud.
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("Actual Annual Growth Rate (%)", fontsize=12)
ax.set_ylabel("Predicted Annual Growth Rate (%)", fontsize=12)
ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
ax.text(0.97, 0.05,
        "$R^2$ = %.3f\n$r$ = %.3f\nn = %d" % (r2_kept, r_kept, len(kept)),
        transform=ax.transAxes, fontsize=11, ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
plt.tight_layout()

for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(d, "%s.%s" % (FIG_NAME, ext))
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print("  saved %s" % p)
plt.close(fig)

# Companion table so every plotted point is auditable.
out = ann[["name_prod", "Year", "n_matched_months", "actual_gr", "pred_gr",
           "retained", "insufficient", "small_base", "trimmed"]].copy()
out["actual_gr"] = (out["actual_gr"] * 100).round(2)
out["pred_gr"] = (out["pred_gr"] * 100).round(2)
out = out.rename(columns={"actual_gr": "Actual_YoY_Growth_pct",
                          "pred_gr": "Predicted_YoY_Growth_pct"})
tp = os.path.join(OUT_DIRS[0], "table_s_geo_holdout_south_allpoints.csv")
out.to_csv(tp, index=False, encoding="utf-8-sig")
print("  saved %s (%d rows)" % (tp, len(out)))
