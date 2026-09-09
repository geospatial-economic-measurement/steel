
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from _paths import DATA_IR

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "Data"))
RAW_DIR    = os.path.join(DATA_ROOT, "raw")

IR_DATA_DIR = DATA_IR
IR_PREP_CSV = os.path.join(IR_DATA_DIR, "ir_data_prepared.csv")
IR_OUT_DIR  = os.path.join(IR_DATA_DIR, "ir_outputs")
IR_PREV_CSV = os.path.join(IR_OUT_DIR, "ir_location_predictions.csv")

GEM_STEEL_PATH = os.path.join(RAW_DIR,
    "Plant-level-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx")

GEN_MODEL   = os.path.join(SCRIPT_DIR, "steel_location_mlp.h5")
GEN_IMPUTER = os.path.join(SCRIPT_DIR, "location_imputer.pkl")
GEN_SCALER  = os.path.join(SCRIPT_DIR, "location_scaler.pkl")
GEN_COLS    = os.path.join(SCRIPT_DIR, "model_columns.pkl")

EARTH_R   = 6371.0
THRESHOLD = 0.5
GAMMA     = 0.22
IR_BBOX   = dict(lon_min=44.0, lon_max=64.0, lat_min=24.0, lat_max=40.0)

os.makedirs(IR_OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def classify_tech(equip):
    if not isinstance(equip, str):
        return "Unknown"
    u = equip.upper()
    if "BF" in u:
        return "BF-BOF"
    if "EAF" in u or "DRI" in u:
        return "EAF"
    return "Unknown"


def load_gem_ir():
    gem = pd.read_excel(GEM_STEEL_PATH, sheet_name="Plant data")
    sub = gem[gem["Country/area"] == "Iran"].copy()
    coords = sub["Coordinates"].str.split(",", expand=True)
    sub["lat"] = pd.to_numeric(coords[0], errors="coerce")
    sub["lon"] = pd.to_numeric(coords[1], errors="coerce")
    sub = sub.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    sub["tech"] = sub["Main production equipment"].apply(classify_tech)
    # Robust plant_name column detection
    if "Plant name" in sub.columns:
        sub = sub.rename(columns={"Plant name": "plant_name"})
    else:
        name_col = [c for c in sub.columns if "name" in c.lower()]
        if name_col:
            sub = sub.rename(columns={name_col[0]: "plant_name"})
        else:
            sub["plant_name"] = sub.index.astype(str)
    sub = sub[
        (sub["lat"] >= IR_BBOX["lat_min"]) & (sub["lat"] <= IR_BBOX["lat_max"]) &
        (sub["lon"] >= IR_BBOX["lon_min"]) & (sub["lon"] <= IR_BBOX["lon_max"])
    ].reset_index(drop=True)
    return sub[["plant_name", "lat", "lon", "tech", "Main production equipment"]]


def compute_coverage(land_cells, prob_col, gem, radius_km=10):
    tree = BallTree(np.deg2rad(land_cells[["lat", "lon"]].values), metric="haversine")
    results = []
    for _, p in gem.iterrows():
        q   = np.deg2rad([[p["lat"], p["lon"]]])
        idx = tree.query_radius(q, r=radius_km / EARTH_R)[0]
        mp  = float(land_cells[prob_col].iloc[idx].max()) if len(idx) else 0.0
        results.append({
            "name": p["plant_name"], "tech": p["tech"],
            "lat": p["lat"], "lon": p["lon"],
            "prob": mp, "detected": mp >= THRESHOLD,
        })
    return pd.DataFrame(results)


def make_grid(land_cells, prob_col, step=0.009):
    lat_arr = np.arange(IR_BBOX["lat_min"], IR_BBOX["lat_max"] + step, step)
    lon_arr = np.arange(IR_BBOX["lon_min"], IR_BBOX["lon_max"] + step, step)
    prob_grid = np.full((len(lat_arr), len(lon_arr)), np.nan)
    land_grid = np.zeros_like(prob_grid, dtype=bool)
    for _, row in land_cells.iterrows():
        i = int(round((row["lat"] - lat_arr[0]) / step))
        j = int(round((row["lon"] - lon_arr[0]) / step))
        if 0 <= i < len(lat_arr) and 0 <= j < len(lon_arr):
            prob_grid[i, j] = row[prob_col]
            land_grid[i, j] = True
    return lat_arr, lon_arr, prob_grid, land_grid


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load prepared data
    print("Loading IR prepared data ...")
    df = pd.read_csv(IR_PREP_CSV)
    cell_df = df.groupby("IDCode").first().reset_index()
    for old, new in [("latitude", "lat"), ("longitude", "lon")]:
        if old in cell_df.columns and new not in cell_df.columns:
            cell_df = cell_df.rename(columns={old: new})
    print(f"  {len(cell_df):,} unique grid cells")

    pred_out = os.path.join(IR_OUT_DIR, "ir_location_predictions_single.csv")

    if os.path.exists(pred_out):
        # Load from cache — skip slow prediction step
        print(f"  Loading cached predictions from {pred_out} ...")
        land_cells = pd.read_csv(pred_out)
        print(f"  Land cells: {len(land_cells):,}")
    else:
        # 2. Load general model
        print("Loading general MLP model ...")
        from tensorflow.keras.models import load_model as keras_load
        import joblib
        gen_model   = keras_load(GEN_MODEL)
        gen_imputer = joblib.load(GEN_IMPUTER)
        gen_scaler  = joblib.load(GEN_SCALER)
        with open(GEN_COLS, "rb") as f:
            gen_cols = pickle.load(f)
        print(f"  Feature columns: {len(gen_cols)}")

        # 3. Predict
        print("Applying general classifier ...")
        X = cell_df.reindex(columns=gen_cols, fill_value=0).copy()
        X_imp = gen_imputer.transform(X)
        X_sc  = gen_scaler.transform(X_imp)
        cell_df["prob_all"] = gen_model.predict(X_sc).flatten()

        # 4. Land mask
        ntl_col = next((c for c in cell_df.columns if "NTL" in c.upper()), None)
        land_cells = cell_df.dropna(subset=[ntl_col]).copy() if ntl_col else cell_df.copy()
        print(f"  Land cells: {len(land_cells):,}")

        # 5. Save predictions
        land_cells[["IDCode", "lat", "lon", "prob_all"]].to_csv(pred_out, index=False)
        print(f"  Saved: {pred_out}")

    # 6. GEM plants
    print("Loading GEM plants ...")
    gem = load_gem_ir()
    gem_bf  = gem[gem["tech"] == "BF-BOF"].reset_index(drop=True)
    gem_eaf = gem[gem["tech"] == "EAF"].reset_index(drop=True)
    print(f"  GEM total: {len(gem)}  (BF-BOF: {len(gem_bf)}, EAF: {len(gem_eaf)})")

    # 7. Coverage — single classifier vs all / BF / EAF subsets
    cov_all = compute_coverage(land_cells, "prob_all", gem)
    cov_bf  = compute_coverage(land_cells, "prob_all", gem_bf)
    cov_eaf = compute_coverage(land_cells, "prob_all", gem_eaf)

    n_all = cov_all["detected"].sum()
    n_bf  = cov_bf["detected"].sum()
    n_eaf = cov_eaf["detected"].sum()

    # 8. Also load tech-stratified results for comparison
    print("Loading tech-stratified predictions for comparison ...")
    prev = pd.read_csv(IR_PREV_CSV)
    # Merge on IDCode
    merged = land_cells[["IDCode", "lat", "lon", "prob_all"]].merge(
        prev[["IDCode", "prob_bf", "prob_eaf"]], on="IDCode", how="left"
    )
    merged["prob_max_tech"] = merged[["prob_bf", "prob_eaf"]].max(axis=1)

    cov_tech_all = compute_coverage(
        merged.rename(columns={"lat": "lat", "lon": "lon"}),
        "prob_max_tech", gem
    )
    n_tech = cov_tech_all["detected"].sum()
    cov_tech_bf  = compute_coverage(merged, "prob_bf",  gem_bf)
    cov_tech_eaf = compute_coverage(merged, "prob_eaf", gem_eaf)

    # 9. Report
    report_path = os.path.join(IR_OUT_DIR, "ir_detection_report_single.txt")
    lines = [
        "Iran — Single vs Tech-Stratified Classifier Comparison",
        "=" * 60,
        f"Threshold: P >= {THRESHOLD}  |  Search radius: 10 km",
        f"GEM plants in bbox: {len(gem)} total  "
        f"({len(gem_bf)} BF-BOF, {len(gem_eaf)} EAF)",
        "",
        "SINGLE GENERAL CLASSIFIER:",
        f"  All plants:    {n_all}/{len(gem)} ({100*n_all/len(gem):.0f}%)",
        f"  BF-BOF subset: {n_bf}/{len(gem_bf)} ({100*n_bf/len(gem_bf):.0f}%)",
        f"  EAF subset:    {n_eaf}/{len(gem_eaf)} ({100*n_eaf/len(gem_eaf):.0f}%)",
        "",
        "TECH-STRATIFIED (BF model→BF plants, EAF model→EAF plants):",
        f"  Combined:      {n_tech}/{len(gem)} ({100*n_tech/len(gem):.0f}%)",
        f"  BF-BOF:        {cov_tech_bf['detected'].sum()}/{len(gem_bf)} "
        f"({100*cov_tech_bf['detected'].sum()/len(gem_bf):.0f}%)",
        f"  EAF:           {cov_tech_eaf['detected'].sum()}/{len(gem_eaf)} "
        f"({100*cov_tech_eaf['detected'].sum()/len(gem_eaf):.0f}%)",
        "",
        "PLANT-LEVEL DETAIL (single classifier):",
    ]
    for _, r in cov_all.iterrows():
        s = "DETECTED" if r["detected"] else "missed  "
        lines.append(f"  [{s}] [{r['tech']:6s}] {r['name'][:45]:45s} P={r['prob']:.4f}")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved report: {report_path}")

    # Print summary to console
    print("\n" + "\n".join(lines[:18]))

    # 10. Heatmap — single panel
    print("\nRendering heatmap ...")
    lat_arr, lon_arr, prob_grid, land_grid = make_grid(land_cells, "prob_all")
    detected = cov_all[cov_all["detected"]]
    missed   = cov_all[~cov_all["detected"]]

    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#07071a")
    ax.set_facecolor("#07071a")
    LONM, LATM = np.meshgrid(lon_arr, lat_arr)

    land_cmap = plt.cm.gray.copy(); land_cmap.set_bad("#07071a")
    ax.pcolormesh(LONM, LATM,
                  np.ma.masked_invalid(np.where(land_grid, 0.08, np.nan)),
                  cmap=land_cmap, vmin=0, vmax=1, shading="auto")

    vis = np.power(np.where(np.isnan(prob_grid), np.nan, prob_grid), GAMMA)
    inferno = plt.cm.inferno.copy(); inferno.set_bad("#00000000")
    im = ax.pcolormesh(LONM, LATM, np.ma.masked_invalid(vis),
                       cmap=inferno, vmin=0, vmax=1, shading="auto")

    for _, d in detected.iterrows():
        ax.plot(d["lon"], d["lat"], "*", color="lime", ms=9, zorder=6)
        ax.annotate(str(d["name"])[:22], (d["lon"], d["lat"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=5.0, color="white", zorder=7)
    for _, m in missed.iterrows():
        ax.plot(m["lon"], m["lat"], "*", color="deepskyblue", ms=8, zorder=6)

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], marker="*", color="w", markerfacecolor="lime",        ms=10, label=f"Detected ({n_all})"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="deepskyblue", ms=10, label=f"Missed ({len(gem)-n_all})"),
    ]
    ax.legend(handles=legend_els, loc="lower left", fontsize=9,
              facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    cbar_ax = fig.add_axes([0.93, 0.08, 0.013, 0.80])
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("P(steel plant)^0.22  [γ-compressed]", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    ax.set_xlim(lon_arr[0], lon_arr[-1])
    ax.set_ylim(lat_arr[0], lat_arr[-1])
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    fig.suptitle(
        f"Iran: General (Single) Classifier  —  {n_all}/{len(gem)} GEM plants detected "
        f"({100*n_all/len(gem):.0f}%)\n"
        f"[vs tech-stratified: {n_tech}/{len(gem)} ({100*n_tech/len(gem):.0f}%)]",
        color="white", fontsize=12, y=0.97
    )

    out_fig = os.path.join(IR_OUT_DIR, "ir_heatmap_combined_techfilter.png")
    fig.savefig(out_fig, dpi=150, bbox_inches="tight", facecolor="#07071a")
    plt.close()
    print(f"  Saved: {out_fig}")
    print("\nDone.")


if __name__ == "__main__":
    main()
