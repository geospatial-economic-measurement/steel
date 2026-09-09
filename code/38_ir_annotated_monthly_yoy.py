
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
from _paths import DATA_IR, OUT_RL

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

IR_OUT_DIR  = os.path.join(DATA_IR, 'ir_outputs')
MONTHLY_CSV = os.path.join(IR_OUT_DIR, "ir_totals_by_month_thresh.csv")
REPLICATED  = OUT_RL
os.makedirs(REPLICATED, exist_ok=True)

WSA = {2019: 25.6, 2020: 29.0, 2021: 28.3, 2022: 30.6, 2023: 30.7, 2024: 31.4}


def main():
    # ── Load and compute YoY ──────────────────────────────────────────────────
    df = pd.read_csv(MONTHLY_CSV)
    df["date"] = pd.to_datetime(
        dict(year=df["Year"].astype(int), month=df["Month"].astype(int), day=1))
    df = df.set_index("date").sort_index()
    steel_col = "Total_Steel" if "Total_Steel" in df.columns else "Steel_Pred_total"
    df = df.rename(columns={steel_col: "Total_Steel"})
    df["yoy_pct"] = df["Total_Steel"].pct_change(periods=12, fill_method=None) * 100

    # ── WSA annual benchmarks ─────────────────────────────────────────────────
    wsa_sorted = sorted(WSA.items())
    wsa_growth = {}
    for i in range(1, len(wsa_sorted)):
        yr_prev, v_prev = wsa_sorted[i - 1]
        yr_curr, v_curr = wsa_sorted[i]
        rate = (v_curr / v_prev - 1) * 100
        wsa_growth[f"{yr_prev}->{yr_curr}"] = (
            rate, pd.Timestamp(f"{yr_curr}-01-01"), pd.Timestamp(f"{yr_curr}-12-01"))

    val_end   = pd.Timestamp(f"{max(WSA.keys())}-12-01")
    app_start = pd.Timestamp(f"{max(WSA.keys())+1}-01-01")

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor("#f9f9f9")

    # Shaded period backgrounds
    ax.axvspan(df.index.min(), val_end, alpha=0.06, color="steelblue",
               label="Validation period")
    if app_start <= df.index.max():
        ax.axvspan(app_start, df.index.max(), alpha=0.06, color="darkorange",
                   label="Application period")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # 2022 crisis window (slightly stronger shade)
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-01"),
               alpha=0.14, color="red", zorder=0, label="2022 energy crisis")

    # ── Model YoY line ────────────────────────────────────────────────────────
    valid = df["yoy_pct"].dropna()
    val_mask = valid.index <= val_end
    ax.plot(valid.index[val_mask], valid[val_mask], color="steelblue",
            linewidth=1.8, marker="o", markersize=3.5,
            label="Model YoY growth (validation)")
    if (~val_mask).any():
        ax.plot(valid.index[~val_mask], valid[~val_mask], color="darkorange",
                linewidth=1.8, marker="o", markersize=3.5, linestyle="--",
                label="Model YoY growth (application)")

    # ── WSA annual benchmark bars ─────────────────────────────────────────────
    for label, (rate, t0, t1) in wsa_growth.items():
        ax.hlines(rate, t0, t1, colors="crimson", linewidth=2.5)
        ax.text(t1 + pd.Timedelta(days=10), rate,
                f"WSA {rate:+.1f}%",
                fontsize=7, color="crimson", va="center")

    # ── Annotations (spread around the 2022 window) ───────────────────────────
    # Helper: get nearest model value at a given date
    def series_val(date_str):
        pt = pd.Timestamp(date_str)
        return float(valid.reindex([pt], method="nearest").iloc[0])

    arrow_kw = dict(arrowstyle="->", color="#333333", lw=1.1,
                    connectionstyle="arc3,rad=0.2")

    # Box 1 — Energy supply cuts (upper-LEFT, before window)
    ax.annotate(
        "Gas supply: only 15/40 Mcm/day\n(−62% shortfall to industry);\n"
        "50% electricity cut to heavy industry",
        xy=(pd.Timestamp("2022-01-01"), series_val("2022-01-01")),
        xytext=(pd.Timestamp("2020-09-01"), 28),
        fontsize=7.5, color="#333333", ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0,
                        connectionstyle="arc3,rad=-0.25"),
    )

    # Box 2 — H1 output decline (below, inside window)
    ax.annotate(
        "Jan–Jun 2022: output −10–11% YoY\n(monthly peaks −17–21% in Apr–May)",
        xy=(pd.Timestamp("2022-05-01"), series_val("2022-05-01")),
        xytext=(pd.Timestamp("2022-02-01"), -36),
        fontsize=7.5, color="#333333", ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0,
                        connectionstyle="arc3,rad=0.15"),
    )

    # Box 3 — Export collapse + structural cause (upper-RIGHT, after window)
    ax.annotate(
        "Steel exports −$2.5B: Russian steel\ndisplaces IR in Asian markets;\n"
        "sanctions cut capex imports\n$20B (2011) → $8B (2022)",
        xy=(pd.Timestamp("2022-09-01"), series_val("2022-09-01")),
        xytext=(pd.Timestamp("2023-02-01"), 26),
        fontsize=7.5, color="#333333", ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0,
                        connectionstyle="arc3,rad=0.2"),
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    wsa_handle   = Line2D([0], [0], color="crimson", linewidth=2.5,
                          label="WSA annual benchmark")
    crisis_patch = mpatches.Patch(color="red", alpha=0.25,
                                  label="2022 energy rationing crisis")
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles + [wsa_handle, crisis_patch],
              lbls + ["WSA annual benchmark", "2022 energy rationing crisis"],
              fontsize=7.5, loc="lower left", framealpha=0.9,
              ncol=2)

    # ── Labels and formatting ─────────────────────────────────────────────────
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("YoY Growth Rate (%)", fontsize=10)
    ax.tick_params(labelsize=8.5)
    # Expand y-axis slightly to give annotation boxes room
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(ymin, -42), max(ymax, 38))
    fig.autofmt_xdate()
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    out1 = os.path.join(IR_OUT_DIR, "ir_monthly_yoy_thresh_annotated.png")
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")

    out2 = os.path.join(REPLICATED, "ir_monthly_yoy_thresh_annotated.png")
    shutil.copy2(out1, out2)
    print(f"Copied to: {out2}")


if __name__ == "__main__":
    main()
