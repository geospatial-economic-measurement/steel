"""Figure 8a (paper): 2022 temporal holdout, no in-figure title.

Same data, model and construction as Panel A of
`21_holdout_validation_temporal_geo_figures.py`. Two display changes so the
panel matches its companion (Figure 8b, produced by
`23_geo_holdout_south_full_symlog_paper.py`):

  * the in-figure title is removed -- Nature Portfolio requires titles and
    legends to live in the manuscript, not inside the figure file, and the two
    panels should look identical side by side;
  * legend sizing matches 13c.

Train 2019-2021, test 2022, plant-month observations on log-log axes.
Expected: R^2 = 0.79, Spearman rho = 0.88.

Outputs: Figure_7a.pdf/.png  -> Final\\Figure\\            (manuscript include)
         Figure_08a.pdf      -> Replication\\output\\figures\\ (canonical store)
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
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_RAW, OUT_FIG, need, save_diagnostic, save_fig

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

SIGNALS_PATH = need(DATA_PROCESSED, 'merged_SteelIron_pollutants_2019_2022.csv')
GRIDPROD_PATH = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GEM_PATH = need(DATA_RAW, 'Iron-unit-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx')

PAPER_FIG = OUT_FIG

EARTH_R = 6371.0
TICKS = [0.1, 1, 10, 100]

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

# ── Temporal holdout: train 2019-2021, test 2022 ──────────────────────────────
train = data[data["Year"].isin([2019, 2020, 2021])]
test = data[data["Year"] == 2022]

model = xgb.XGBRegressor(**XGB_PARAMS)
model.fit(train[FEATURE_COLS].values, np.log1p(train["Steel_Prod"].values))
y_actual = test["Steel_Prod"].values
y_pred = np.expm1(model.predict(test[FEATURE_COLS].values))

r2 = r2_score(y_actual, y_pred)
rho = spearmanr(y_actual, y_pred).statistic
print("  Train N=%d, Test N=%d" % (len(train), len(test)))
print("  2022 holdout: R2 = %.4f, Spearman rho = %.4f" % (r2, rho))

# ── Figure: no title (moved to the manuscript caption) ────────────────────────
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
ax.scatter(y_actual, y_pred, alpha=0.6, color="blue", s=25,
           label="Plant-month observation")
lo = min(y_actual.min(), y_pred.min()) * 0.8
hi = max(y_actual.max(), y_pred.max()) * 1.2
ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="45$^\\circ$ line (perfect fit)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(TICKS)
ax.set_xticklabels(TICKS)
ax.set_yticks(TICKS)
ax.set_yticklabels(TICKS)
ax.set_xlabel("Actual Steel Output (10,000 MT/month)", fontsize=12)
ax.set_ylabel("Predicted Steel Output (10,000 MT/month)", fontsize=12)
ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
ax.text(0.97, 0.05, "$R^2$ = %.3f\n$\\rho$ = %.3f" % (r2, rho),
        transform=ax.transAxes, fontsize=11, ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
plt.tight_layout()

for d in (PAPER_FIG, OUT_FIG):
    os.makedirs(d, exist_ok=True)
save_fig(fig, "Figure_07a")
plt.close(fig)
