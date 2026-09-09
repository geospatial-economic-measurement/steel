"""Figure 7b (paper): southern-China geographic holdout, FULL sample, symlog axes.

Reproduces `fig_geo_holdout_south_full_symlog.pdf` as a standalone single panel
and writes it as the paper's Figure_7b. Until now that exhibit existed only as
an ad-hoc split of panel (a) of `fig_geo_holdout_2dir_full_symlog.pdf`
(script 51) with no script of its own; this file supplies one so the code and
the published figure agree.

Method (identical to `51_geo_holdout_letter_figures.py`):
  * train XGBoost on northern China (Latitude >= 35), predict southern China;
  * `unmatched_annual_growth`: sum Steel_Prod and pred per plant-year over
    whatever months are present, then take pct_change across consecutive years
    within each plant. No matched-month requirement, no minimum-month rule, no
    small-base guard and no trimming -- every finite pair is kept and plotted;
  * symlog axes (linthresh = 25) with fixed decade-style ticks;
  * r and R^2 computed on the same full sample that is displayed.

Expected output: r = 0.982, R^2 = 0.840, n = 120.

NOTE ON PROVENANCE. This is the full-sample construction. The alternative
matched-month construction in `21_holdout_validation_temporal_geo_figures.py`
(>= 8 matched months, small-base guard) yields r = -0.012, R^2 = -0.952,
n = 118 on the same model and data. The two differ because the unmatched
construction compares annual sums built from different month coverage across
years: the 马长江 2020 pair contributes a single matched month (+1077% actual /
+1455% predicted here versus -5.9% / -1.0% under matched months), and 2019 is a
short year for several plants. Whoever maintains this file should keep both
numbers in view when describing the exercise.

Outputs: Figure_7b.pdf/.png -> E:\\project\\NC_steel\\Final\\Figure\\
         fig_geo_holdout_south_full_symlog.pdf -> output_rl\\ (regenerated)
         table_s_geo_holdout_south_full.csv     -> output_rl\\
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
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_FIG, OUT_RL, need, save_diagnostic, save_fig

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

SIGNALS_PATH = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')

PAPER_FIG = OUT_FIG          # manuscript \includegraphics

EARTH_R = 6371.0
LINTHRESH = 25
JMP_GREY = "#737373"

CHINA_HUBS = {"Tangshan": (39.6, 118.2), "Wuhan": (30.6, 114.3),
              "Anshan": (41.1, 122.8), "Rizhao": (35.4, 119.5),
              "Baotou": (40.7, 109.8)}
CHINA_PORTS = {"Shanghai": (31.23, 121.47), "Tianjin": (38.98, 117.72),
               "Qingdao": (36.07, 120.38), "Ningbo": (29.87, 121.55),
               "Guangzhou": (23.10, 113.43), "Dalian": (38.92, 121.65),
               "Lianyungang": (34.75, 119.45), "Yingkou": (40.67, 122.23)}

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

model = xgb.XGBRegressor(**XGB_PARAMS)
model.fit(north[FEATURE_COLS].values, np.log1p(north["Steel_Prod"].values))
test = south.copy()
test["pred"] = np.expm1(model.predict(test[FEATURE_COLS].values))


def unmatched_annual_growth(t):
    """Full-sample annual growth: plant-year sums, then within-plant pct_change."""
    ann = (t.groupby(["name_prod", "Year"])[["Steel_Prod", "pred"]]
           .sum().reset_index().sort_values(["name_prod", "Year"]))
    ann["actual_gr"] = ann.groupby("name_prod")["Steel_Prod"].pct_change()
    ann["pred_gr"] = ann.groupby("name_prod")["pred"].pct_change()
    ann = ann.dropna(subset=["actual_gr", "pred_gr"])
    ann = ann[np.isfinite(ann["actual_gr"]) & np.isfinite(ann["pred_gr"])].copy()
    return ann


ann = unmatched_annual_growth(test)
a = ann["actual_gr"].values * 100
p = ann["pred_gr"].values * 100
r = pearsonr(ann["actual_gr"], ann["pred_gr"])[0]
r2 = r2_score(ann["actual_gr"], ann["pred_gr"])
n = len(ann)
print("  FULL SAMPLE  r = %+.3f   R2 = %+.3f   n = %d" % (r, r2, n))

top = ann.reindex(ann["actual_gr"].abs().sort_values(ascending=False).index).head(3)
print("  largest |actual growth| observations:")
for _, row in top.iterrows():
    print("     %-12s %d  actual %+9.1f%%  predicted %+9.1f%%"
          % (row["name_prod"], int(row["Year"]),
             row["actual_gr"] * 100, row["pred_gr"] * 100))

# ── Figure: single panel, identical style to script 51's symlog_scatter ───────
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
ax.scatter(a, p, alpha=0.6, color="blue", s=25, zorder=3,
           label="Plant-year observation")
ext = max(np.abs(a).max(), np.abs(p).max()) * 1.10
lo, hi = -ext, ext
ax.plot([lo, hi], [lo, hi], "r--", lw=2, zorder=2, label="45$^\\circ$ line (perfect fit)")
ax.axhline(0, color=JMP_GREY, lw=0.8, ls=":")
ax.axvline(0, color=JMP_GREY, lw=0.8, ls=":")
ax.set_xscale("symlog", linthresh=LINTHRESH)
ax.set_yscale("symlog", linthresh=LINTHRESH)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ticks = [t for t in (-1000, -500, -100, -50, -25, 0, 25, 50, 100, 500, 1000)
         if lo <= t <= hi]
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    axis.set_minor_locator(NullLocator())
ax.set_xlabel("Actual Annual Growth Rate (%)", fontsize=12)
ax.set_ylabel("Predicted Annual Growth Rate (%)", fontsize=12)
ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
ax.text(0.97, 0.05, "$R^2$ = %.3f\n$r$ = %.3f\nn = %d" % (r2, r, n),
        transform=ax.transAxes, fontsize=11, ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
plt.tight_layout()

for d in (PAPER_FIG, OUT_FIG, OUT_RL):
    os.makedirs(d, exist_ok=True)

# The manuscript \includegraphics uses the unpadded name; the canonical
# paper-figure store uses the zero-padded accepted-PDF numbering (Fig. 8b).
# Both must be written or the two locations drift apart.
save_fig(fig, "Figure_07b")


fp = os.path.join(OUT_RL, "fig_geo_holdout_south_full_symlog.pdf")
fig.savefig(fp, bbox_inches="tight", dpi=300)
print("  saved %s" % fp)
plt.close(fig)

out = ann[["name_prod", "Year", "actual_gr", "pred_gr"]].copy()
out["actual_gr"] = (out["actual_gr"] * 100).round(2)
out["pred_gr"] = (out["pred_gr"] * 100).round(2)
out = out.rename(columns={"actual_gr": "Actual_YoY_Growth_pct",
                          "pred_gr": "Predicted_YoY_Growth_pct"})
tp = os.path.join(OUT_RL, "table_s_geo_holdout_south_full.csv")
out.to_csv(tp, index=False, encoding="utf-8-sig")
print("  saved %s (%d rows)" % (tp, len(out)))
