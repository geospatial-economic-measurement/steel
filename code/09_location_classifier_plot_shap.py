

import os, pickle, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shap
import os
from _paths import OUT_FIG, SCRATCH

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

CACHE_FILE = os.path.join(SCRATCH, "shap_classification_cache.pkl")
OUT_DIR    = OUT_FIG
PAPER_FIG  = OUT_FIG

SIGNAL_MAP = {
    "CO_MEAN":    "CO",
    "NO2_MEAN":   "NO₂",
    "PM2_5_MEAN": "PM₂.₅",
    "PM10_MEAN":  "PM₁₀",
    "SO2_MEAN":   "SO₂",
    "LSTA_MEAN":  "LST (night)",
    "LSTT_MEAN":  "LST (day)",
    "NTL_MEAN":   "NTL",
    "O3_MEAN":    "O₃",
}

def pretty(col):
    for raw, nice in SIGNAL_MAP.items():
        if col == raw:
            return nice
        if col.startswith(raw + "_lag"):
            lag_n = col.replace(raw + "_lag", "")
            return f"{nice} (lag {lag_n})"
    return col

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_display", type=int, default=15)
    parser.add_argument("--width",       type=float, default=6)
    parser.add_argument("--height",      type=float, default=6)
    args = parser.parse_args()

    print(f"Loading SHAP cache: {CACHE_FILE}")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)

    shap_vals   = cache["shap_values"]
    base_vals   = cache["base_values"]
    X_explain   = cache["X_explain"]
    feature_cols = cache["feature_cols"]

    pretty_names = [pretty(c) for c in feature_cols]

    shap_exp = shap.Explanation(
        values        = shap_vals,
        base_values   = base_vals,
        data          = X_explain,
        feature_names = pretty_names,
    )

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=300)
    plt.sca(ax)
    shap.plots.beeswarm(shap_exp, max_display=args.max_display, show=False)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        out_path = os.path.join(OUT_DIR, f"fig_shap_classification.{ext}")
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
        # Also copy to paper Figure directory
        import shutil
        shutil.copy2(out_path, os.path.join(PAPER_FIG, f"fig_shap_classification.{ext}"))
        print(f"Copied to paper Figure/")

    plt.close()
    print("Done.")

if __name__ == "__main__":
    main()
