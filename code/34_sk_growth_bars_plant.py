"""Figure 9b (alternative): South Korea national growth rates, PLANT level, bars only.

Companion to `33_sk_growth_bars_paper.py`, which produces the grid-level
version. Same layout in both -- bars only, no in-figure title, identical
styling -- so the two can be compared directly and either can be dropped into
the manuscript.

Series: Sentinel M1 plant-level predictions, summed over the 15 South Korean
GEM plants (10 km buffer extraction). Coverage is 12 months in every year from
2019 to 2024, so no lag burn-in exclusion or rescaling is needed -- unlike the
grid series, where the first three months are unusable and 2019 must be
rescaled from 9 to 12 months.

Result: r = 0.980, MAE = 2.6 pp, n = 5 transitions. This is the series the
accepted manuscript reported (there against slightly different WSA figures,
giving r = 0.966, MAE = 3.0 pp).

Which level goes in the paper:

| Level | Script | r | MAE | 2019->2020 |
|---|---|---|---|---|
| Plant | `34` (this file) | 0.980 | 2.6 pp | -3.2% |
| Grid  | `33`             | 0.887 | 1.9 pp | -8.0% |

Whichever is used, Table 1 and the figure must be regenerated together: both
scripts print the Table 1 LaTeX rows for that purpose.

Outputs: fig_sk_growth_rates_plant.pdf/.png -> Final\\Figure\\
         Figure_09b_plant.pdf               -> Replication\\output\\figures\\
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
from _paths import DATA_SK, OUT_FIG, need, save_diagnostic, save_fig

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

SRC = need(DATA_SK, 'sk_outputs', 'sk_plant_sentinel_monthly.csv')
PAPER_FIG = OUT_FIG

# World Steel Association reported national crude steel output, South Korea (Mt).
WSA_SK = {2019: 71.4, 2020: 67.1, 2021: 70.4, 2022: 65.8, 2023: 66.7, 2024: 63.6}

# Published plant-level model column of Table 1, used as a verification target.
TABLE1_MODEL = [-3.2, 0.5, -3.8, 0.1, -2.9]

df = pd.read_csv(SRC)
print("rows: %d | plants: %d" % (len(df), df["plant_name"].nunique()))

counts = df.groupby("Year")["Month"].nunique()
print("months per year: %s" % dict(zip(counts.index.tolist(), counts.tolist())))
if set(counts.tolist()) != {12}:
    print("  WARNING: unbalanced coverage -- annual sums are not directly comparable.")

nat = df.groupby("Year")["pred_kt"].sum()
years = sorted(nat.index)

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

dev = np.abs(model - np.array(TABLE1_MODEL))
print("\nmax deviation from the published Table 1 model column: %.2f pp" % dev.max())
if dev.max() > 0.06:
    print("  WARNING: does not reproduce Table 1 -- inspect before use.")

print("\nTable 1 rows (paste into the manuscript):")
for t, m, w in zip(trans, model, wsa):
    a, b = t.split("\u2192")
    print("  %s $\\to$ %s & $%+.1f$ & $%+.1f$ & $%+.1f$ \\\\" % (a, b, m, w, m - w))
print("  Pearson r = %.3f ; MAE = %.1f pp" % (r, mae))

# ── Figure: bars only, no title (identical styling to 22b) ────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
x = np.arange(len(trans))
bw = 0.35
ax.bar(x - bw / 2, model, width=bw, color="#2166ac",
       label="Model prediction (plant-level)")
ax.bar(x + bw / 2, wsa, width=bw, color="#d73027", alpha=0.75, label="WSA actual")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(trans, fontsize=10)
ax.set_ylabel("Year-over-year growth rate (%)", fontsize=11)
ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
plt.tight_layout()

for d in (PAPER_FIG, OUT_FIG):
    os.makedirs(d, exist_ok=True)
save_fig(fig, "Figure_08b")
save_diagnostic(fig, "fig_sk_growth_plant_level")
plt.close(fig)
