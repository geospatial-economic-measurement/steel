"""Figure 9b (paper): South Korea national growth rates, GRID-level, bars only.

Layout follows the accepted manuscript: a bar chart alone. The Pearson r and
MAE live in the caption and in Table 1, not inside the figure, and there is no
in-figure title (Nature Portfolio requires titles in the manuscript, and it
keeps this panel consistent with Figures 8a/8b).

Series: national grid-level predicted output, summing `Steel_Pred_total` over
all South Korean 1 km cells, aggregated exactly as in
`Analysis/19_south_korea_agg.py`:

  1. drop the first LAG_BURNIN_MONTHS (= 3) months of the panel -- lag-3
     features are imputed means there, so Jan-Mar 2019 predictions are
     unreliable. 2019 therefore keeps Apr-Dec only;
  2. rescale the 9-month 2019 total by 12/9 to a 12-month equivalent, so the
     2019->2020 growth rate is not mechanically depressed by a short base year.

Both steps matter: without them the 2019->2020 rate is -14.6% instead of -8.0%.
The rescaled 2019 total (8.432315e+06) reproduces `output_rl/sk_total_by_year.csv`
exactly.

Result: r = 0.887, MAE = 1.9 pp, n = 5 transitions -- matching the response
letter. Table 1 of the manuscript reports the same series, so the figure and the
table agree.

Alternative aggregations, for the record:
  * raw grid sums, no burn-in or rescaling -> r = 0.789, MAE = 3.2 pp
  * `Steel_Pred_model_scale` (model units)  -> r = 0.746, MAE = 2.6 pp
  * plant-level Sentinel M1 (script 32)     -> r = 0.980, MAE = 2.6 pp (superseded)

The first pass reads ~1.3 GB of grid predictions and caches national monthly
totals to `output_rl/sk_national_grid_monthly.csv`; later runs reuse the cache.

Outputs: fig_sk_growth_rates.pdf/.png  -> Final\\Figure\\   (manuscript include)
         Figure_09b.pdf                -> Replication\\output\\figures\\
         sk_national_grid_monthly.csv  -> Replication\\output_rl\\
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
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _paths import DATA_SK, OUT_FIG, OUT_RL, need, save_diagnostic, save_fig

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

GRID_FILES = [
    need(DATA_SK, 'sk_outputs', 'sk_predictions_grid_2019_2022.csv'),
    need(DATA_SK, 'sk_outputs', 'sk_predictions_grid_2023_2024.csv'),
]
PAPER_FIG = OUT_FIG
CACHE = os.path.join(OUT_RL, "sk_national_grid_monthly.csv")

# Lag-3 features are imputed means for the first three months, so those
# predictions are dropped (see Analysis/19_south_korea_agg.py).
LAG_BURNIN_MONTHS = 3

VALUE_COL = "Steel_Pred_total"

# World Steel Association reported national crude steel output, South Korea (Mt).
WSA_SK = {2019: 71.4, 2020: 67.1, 2021: 70.4, 2022: 65.8, 2023: 66.7, 2024: 63.6}


def monthly_totals():
    """National grid sums by (Year, Month), cached after the first pass."""
    if os.path.exists(CACHE):
        print("using cached monthly totals -> %s" % CACHE)
        return pd.read_csv(CACHE)

    print("aggregating grid predictions (first run, ~1.3 GB) ...")
    acc = {}
    for f in GRID_FILES:
        print("  reading %s" % os.path.basename(f))
        for chunk in pd.read_csv(
                f, usecols=["Year", "Month", VALUE_COL], chunksize=2_000_000):
            g = chunk.groupby(["Year", "Month"])[VALUE_COL].sum()
            for (y, m), v in g.items():
                acc[(y, m)] = acc.get((y, m), 0.0) + v

    df = (pd.DataFrame([{"Year": y, "Month": m, "SK_Total_Steel": v}
                        for (y, m), v in acc.items()])
          .sort_values(["Year", "Month"]).reset_index(drop=True))
    os.makedirs(OUT_RL, exist_ok=True)
    df.to_csv(CACHE, index=False)
    print("  cached -> %s" % CACHE)
    return df


def national_totals():
    """Annual totals following `Analysis/19_south_korea_agg.py`.

    Two steps matter and neither is cosmetic:
      1. drop the first LAG_BURNIN_MONTHS months -- lag-3 features are imputed
         means there, so Jan-Mar 2019 predictions are unreliable;
      2. scale the resulting 9-month 2019 total to a 12-month equivalent, so the
         2019->2020 growth rate is not mechanically depressed by a short year.
    """
    m = monthly_totals().sort_values(["Year", "Month"]).reset_index(drop=True)
    m = m.iloc[LAG_BURNIN_MONTHS:].reset_index(drop=True)

    per_year = m.groupby("Year")["SK_Total_Steel"].sum()
    counts = m.groupby("Year").size()
    start_year = int(m["Year"].min())

    print("  months retained per year: %s"
          % dict(zip(counts.index.tolist(), counts.tolist())))
    n0 = int(counts.loc[start_year])
    if n0 < 12:
        per_year.loc[start_year] *= 12.0 / n0
        print("  %d total scaled from %d to 12 months" % (start_year, n0))

    return dict(per_year)


nat = national_totals()
years = sorted(nat)

trans, model, wsa = [], [], []
for a, b in zip(years, years[1:]):
    if a in WSA_SK and b in WSA_SK:
        trans.append("%d\u2192%d" % (a, b))
        model.append((nat[b] / nat[a] - 1) * 100)
        wsa.append((WSA_SK[b] / WSA_SK[a] - 1) * 100)

model = np.array(model)
wsa = np.array(wsa)
r = pearsonr(model, wsa)[0]
mae = np.mean(np.abs(model - wsa))

print("\n%-14s %8s %8s %8s" % ("transition", "model", "WSA", "error"))
for t, m, w in zip(trans, model, wsa):
    print("  %-12s %+7.1f %+8.1f %+8.1f" % (t, m, w, m - w))
print("  Pearson r = %.3f   MAE = %.1f pp   n = %d" % (r, mae, len(model)))

# LaTeX rows for Table 1, so the table cannot drift from the figure.
print("\nTable 1 rows (paste into the manuscript):")
for t, m, w in zip(trans, model, wsa):
    a, b = t.split("\u2192")
    print("  %s $\\to$ %s & $%+.1f$ & $%+.1f$ & $%+.1f$ \\\\" % (a, b, m, w, m - w))
print("  Pearson r = %.3f ; MAE = %.1f pp" % (r, mae))

# ── Figure: bars only, no title ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
x = np.arange(len(trans))
bw = 0.35
ax.bar(x - bw / 2, model, width=bw, color="#2166ac",
       label="Model prediction (grid-level)")
ax.bar(x + bw / 2, wsa, width=bw, color="#d73027", alpha=0.75, label="WSA actual")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(trans, fontsize=10)
ax.set_ylabel("Year-over-year growth rate (%)", fontsize=11)
ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
plt.tight_layout()

for d in (PAPER_FIG, OUT_FIG):
    os.makedirs(d, exist_ok=True)
save_diagnostic(fig, "fig_sk_growth_grid_level")
plt.close(fig)
