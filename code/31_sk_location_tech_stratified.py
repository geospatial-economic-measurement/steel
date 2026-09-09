
import os
import sys
import pickle

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_SK, need

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "sk_outputs")
DATA_ROOT  = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "Data"))
GEM_SK_PATH = os.path.join(DATA_ROOT, "codebooks", "sk_gem_plants.csv")
GEM_GLOBAL_PATH = os.path.join(
    DATA_ROOT, "raw",
    "Plant-level-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx")

SK_PRED_PATH = os.path.join(OUT_DIR, "SK_steel_location_predictions.csv")

# China training data (for Part B)
CHINA_TRAIN_DIR = DATA_PROCESSED
CHINA_CSV1      = os.path.join(CHINA_TRAIN_DIR, "first_output_file.csv")
CHINA_CSV2      = os.path.join(CHINA_TRAIN_DIR, "second_output_file.csv")
GRIDPROD_CSV    = need(DATA_CONFIDENTIAL, 'GridProd_1922_monthly.csv')
GRIDINFO_CSV    = need(DATA_CONFIDENTIAL, 'gridinfo_xy.csv')

EARTH_R   = 6371.0
THRESHOLD = 0.5
TECH_JOIN_KM = 15.0  # assign a cell's technology by nearest GEM plant within 15 km

os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: technology classifier
# ─────────────────────────────────────────────────────────────────────────────
def classify_tech(tech_str: str) -> str:
    """
    Assign a plant to 'BF-BOF' or 'EAF' based on its technology string.
    Mixed plants (BF+EAF) are classified as 'BF-BOF' because their satellite
    signature is dominated by blast furnace SO2/CO emissions.
    Pure EAF plants → 'EAF'.
    """
    if not isinstance(tech_str, str):
        return "Unknown"
    t = tech_str.upper()
    if "BF" in t:
        return "BF-BOF"
    if "BOF" in t and "EAF" not in t:
        return "BF-BOF"
    if "EAF" in t:
        return "EAF"
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# PART A — Detection analysis with existing predictions
# ─────────────────────────────────────────────────────────────────────────────
def run_part_a():
    print("\n" + "=" * 60)
    print("PART A — Technology-stratified detection analysis")
    print("=" * 60)

    # Load SK predictions
    if not os.path.exists(SK_PRED_PATH):
        print(f"ERROR: SK predictions not found at {SK_PRED_PATH}")
        print("Run 18_south_korea_prediction.py first.")
        return None

    pred = pd.read_csv(SK_PRED_PATH)
    cell_max = pred.groupby(["latitude", "longitude"])["SteelProb"].max().reset_index()
    print(f"Loaded {len(pred):,} rows → {len(cell_max):,} unique cells")

    # Load SK GEM plants and classify technology
    gem = pd.read_csv(GEM_SK_PATH)
    gem["tech_group"] = gem["technology"].apply(classify_tech)

    print(f"\nSK GEM plants by technology:")
    for tg, sub in gem.groupby("tech_group"):
        print(f"  {tg}: {len(sub)} plants")
        for _, p in sub.iterrows():
            print(f"    {p['plant_name']}")

    # Coverage check by technology group
    grid_tree = BallTree(
        np.deg2rad(cell_max[["latitude", "longitude"]].values),
        metric="haversine"
    )

    results = []
    for _, plant in gem.iterrows():
        q   = np.deg2rad([[plant["lat"], plant["lon"]]])
        idx = grid_tree.query_radius(q, r=10.0 / EARTH_R)[0]
        mp  = float(cell_max["SteelProb"].iloc[idx].max()) if len(idx) else 0.0
        results.append({
            "plant_name": plant["plant_name"],
            "company":    plant.get("company", ""),
            "lat":        plant["lat"],
            "lon":        plant["lon"],
            "tech_group": plant["tech_group"],
            "max_prob":   mp,
            "detected":   mp >= THRESHOLD,
        })
    det_df = pd.DataFrame(results)

    # Summary by technology group
    report_lines = [
        "South Korea — Technology-Stratified Detection Report",
        "=" * 60,
        f"Threshold: P >= {THRESHOLD}  |  Search radius: 10 km",
        "",
    ]

    for tg in ["BF-BOF", "EAF", "Other"]:
        sub = det_df[det_df["tech_group"] == tg]
        if len(sub) == 0:
            continue
        n_det = sub["detected"].sum()
        n_tot = len(sub)
        pct   = 100.0 * n_det / n_tot
        report_lines.append(f"{tg} plants: {n_det}/{n_tot} detected ({pct:.0f}%)")
        for _, r in sub.iterrows():
            status = "DETECTED" if r["detected"] else "missed"
            report_lines.append(
                f"  [{status:8s}] {r['plant_name']:45s} P={r['max_prob']:.4f}"
            )
        report_lines.append("")

    # Overall
    n_all = det_df["detected"].sum()
    report_lines.append(f"Overall: {n_all}/{len(det_df)} detected")
    report_lines.append("")
    report_lines.append(
        "Interpretation:\n"
        "  BF-BOF plants emit high SO2 and CO from coke combustion and\n"
        "  hot-metal production. The MLP was trained on Chinese data where\n"
        "  ~60% of plants are BF-BOF, so the model is effectively a BF-BOF\n"
        "  detector. EAF plants produce steel from scrap via electric arc\n"
        "  with near-zero SO2/CO — their satellite signature is much weaker\n"
        "  and overlaps with non-steel industries.\n"
        "\n"
        "  Run Part B to train a dedicated EAF classifier on Chinese EAF\n"
        "  plants and apply it to South Korea."
    )

    report_str = "\n".join(report_lines)
    print(report_str)
    report_path = os.path.join(OUT_DIR, "sk_tech_detection_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str + "\n")
    print(f"\nSaved report: {report_path}")

    # Static map
    _plot_tech_map(cell_max, det_df, OUT_DIR)

    return det_df


def _plot_tech_map(cell_max, det_df, out_dir):
    """
    Heatmap with markers colour-coded by technology and detection status.
    Legend: green star = BF-BOF detected, red star = BF-BOF missed,
            blue diamond = EAF detected, orange diamond = EAF missed
    """
    land_file = need(DATA_SK, 'sk_land_mask.csv')
    if os.path.exists(land_file):
        land = pd.read_csv(land_file)
        land_set = set(zip(land.loc[land["valid_count"] >= 1, "latitude"].round(4),
                           land.loc[land["valid_count"] >= 1, "longitude"].round(4)))
        cell_max["is_land"] = cell_max.apply(
            lambda r: (round(r["latitude"], 4), round(r["longitude"], 4)) in land_set,
            axis=1)
        plot_cells = cell_max[cell_max["is_land"]]
    else:
        plot_cells = cell_max

    lat_step, lon_step = 0.009, 0.009
    lat_arr = np.arange(33.0, 38.85, lat_step)
    lon_arr = np.arange(125.8, 130.45, lon_step)
    prob_grid = np.full((len(lat_arr), len(lon_arr)), np.nan)
    for _, row in plot_cells.iterrows():
        i = int(round((row["latitude"]  - lat_arr[0]) / lat_step))
        j = int(round((row["longitude"] - lon_arr[0]) / lon_step))
        if 0 <= i < len(lat_arr) and 0 <= j < len(lon_arr):
            prob_grid[i, j] = row["SteelProb"]

    LONM, LATM = np.meshgrid(lon_arr, lat_arr)
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_facecolor("#07071a")
    fig.patch.set_facecolor("#07071a")

    prob_vis = np.power(np.where(np.isnan(prob_grid), np.nan, prob_grid), 0.22)
    inferno  = plt.cm.inferno.copy()
    inferno.set_bad("#00000000")
    im = ax.pcolormesh(LONM, LATM, np.ma.masked_invalid(prob_vis),
                       cmap=inferno, vmin=0, vmax=1, shading="auto")

    # Markers by technology group and detection status
    STYLES = {
        ("BF-BOF", True):  dict(marker="*", s=320, c="#00ff88", label="BF-BOF detected"),
        ("BF-BOF", False): dict(marker="*", s=320, c="#ff4444", label="BF-BOF missed"),
        ("EAF",    True):  dict(marker="D", s=90,  c="#00ccff", label="EAF detected"),
        ("EAF",    False): dict(marker="D", s=90,  c="#ffaa00", label="EAF missed"),
        ("Other",  True):  dict(marker="o", s=80,  c="#ffffff", label="Other detected"),
        ("Other",  False): dict(marker="o", s=80,  c="#888888", label="Other missed"),
    }

    legend_added = set()
    for _, r in det_df.iterrows():
        key = (r["tech_group"], bool(r["detected"]))
        style = STYLES.get(key, dict(marker="o", s=80, c="gray"))
        label = style.get("label") if key not in legend_added else ""
        legend_added.add(key)
        ax.scatter(r["lon"], r["lat"],
                   marker=style["marker"], s=style["s"], c=style["c"],
                   edgecolors="white", linewidths=0.6, zorder=8,
                   label=label if label else "_nolegend_")

    # Colorbar
    cbar_ticks  = [0, 0.1**0.22, 0.3**0.22, 0.5**0.22, 0.9**0.22, 1.0]
    cbar_labels = ["0", "0.1", "0.3", "0.5", "0.9", "1.0"]
    cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                        fraction=0.025, pad=0.01, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_labels, color="white", fontsize=8)
    cbar.set_label("P(steel plant)", color="white", fontsize=10)
    cbar.ax.tick_params(colors="white")

    ax.set_xlim(125.8, 130.4)
    ax.set_ylim(33.0, 38.8)
    ax.set_xlabel("Longitude", color="white", fontsize=10)
    ax.set_ylabel("Latitude",  color="white", fontsize=10)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#333")
    ax.grid(True, color="#1a1a2e", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_title(
        "South Korea: Steel Location Probability\nby Furnace Technology",
        color="white", fontsize=12, fontweight="bold", pad=12
    )
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.6,
              labelcolor="white", facecolor="#1a1a2e", edgecolor="#555",
              markerscale=0.9)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_sk_location.png")
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor="#07071a")
    plt.close()
    print(f"Saved map: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PART B — Train technology-stratified classifiers
# ─────────────────────────────────────────────────────────────────────────────
def run_part_b():
    print("\n" + "=" * 60)
    print("PART B — Train BF-BOF and EAF location classifiers")
    print("=" * 60)

    for path in [CHINA_CSV1, CHINA_CSV2, GRIDPROD_CSV, GRIDINFO_CSV]:
        if not os.path.exists(path):
            print(f"ERROR: Required file not found:\n  {path}")
            print("Part B requires the full China training dataset. Skipping.")
            return False

    if not os.path.exists(GEM_GLOBAL_PATH):
        print(f"ERROR: GEM global plant file not found:\n  {GEM_GLOBAL_PATH}")
        return False

    # ── Load GEM China plants and classify technology ────────────────────────
    print("Loading GEM China plants ...")
    gem_global = pd.read_excel(GEM_GLOBAL_PATH, sheet_name="Plant data")
    gem_china  = gem_global[gem_global["Country/area"] == "China"].copy()
    gem_china["tech_group"] = gem_china["Main production equipment"].apply(classify_tech)
    gem_china[["lat", "lon"]] = (
        gem_china["Coordinates"].str.split(",", expand=True).astype(float)
    )
    gem_china = gem_china.dropna(subset=["lat", "lon"])
    print(f"  China plants: {len(gem_china):,}  "
          f"(BF-BOF={len(gem_china[gem_china['tech_group']=='BF-BOF'])}, "
          f"EAF={len(gem_china[gem_china['tech_group']=='EAF'])})")

    # ── Load China training data (Aug–Nov months only) ───────────────────────
    print("Loading China training CSVs (months 8-11) ...")

    def load_aug_nov(path, chunksize=500_000):
        parts = []
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            if "Month" in chunk.columns:
                parts.append(chunk[chunk["Month"].isin([8, 9, 10, 11])])
            else:
                parts.append(chunk)
        return pd.concat(parts, ignore_index=True)

    df1 = load_aug_nov(CHINA_CSV1)
    df2 = load_aug_nov(CHINA_CSV2)
    combined = pd.concat([df1, df2], ignore_index=True)
    del df1, df2
    print(f"  Combined: {len(combined):,} rows")

    # ── Feature engineering (identical to train_location_model.py) ───────────
    combined = combined.dropna()
    if "PM1_MEAN" in combined.columns:
        combined = combined.drop(columns="PM1_MEAN")
    combined = combined.sort_values(["IDCode", "Year", "Month"]).reset_index(drop=True)

    pollutants = ["CO_MEAN", "NO2_MEAN", "PM2_5_MEAN", "PM10_MEAN",
                  "SO2_MEAN", "LSTA_MEAN", "LSTT_MEAN", "NTL_MEAN", "O3_MEAN"]
    pollutants = [p for p in pollutants if p in combined.columns]

    print("Adding lag features ...")
    for lag in range(1, 4):
        for col in pollutants:
            combined[f"{col}_lag{lag}"] = combined.groupby("IDCode")[col].shift(lag)

    combined = combined.dropna()
    combined = combined[combined["Month"] == 11].copy()
    print(f"  After lag + November filter: {len(combined):,} rows")

    # ── Merge GridProd + gridinfo ─────────────────────────────────────────────
    print("Merging with GridProd and gridinfo ...")
    grid_prod = pd.read_csv(GRIDPROD_CSV)
    grid_info = pd.read_csv(GRIDINFO_CSV)

    combined = pd.merge(
        combined,
        grid_prod[["IDCode", "Year", "Month", "GridProd_Steel_tot"]],
        on=["IDCode", "Year", "Month"], how="left"
    )
    combined = pd.merge(combined, grid_info, on="IDCode", how="left")

    # ── Assign technology label to positive training cells ───────────────────
    print("Assigning technology labels via spatial join with GEM China ...")
    pos_mask = combined["GridProd_Steel_tot"].fillna(0) > 0
    pos_cells = combined[pos_mask].copy()
    neg_cells = combined[~pos_mask].copy()

    # Build BallTree over GEM China plants
    gem_coords_rad = np.deg2rad(gem_china[["lat", "lon"]].values)
    gem_tree = BallTree(gem_coords_rad, metric="haversine")

    # For each positive cell, find nearest GEM plant within TECH_JOIN_KM
    pos_lat = pos_cells["Centroid_Lat"].values
    pos_lon = pos_cells["Centroid_Long"].values
    pos_rad  = np.deg2rad(np.column_stack([pos_lat, pos_lon]))
    dist_rad, idx_arr = gem_tree.query(pos_rad, k=1)
    dist_km = dist_rad[:, 0] * EARTH_R
    nearest_idx = idx_arr[:, 0]

    tech_labels = []
    for d, gi in zip(dist_km, nearest_idx):
        if d <= TECH_JOIN_KM:
            tech_labels.append(gem_china["tech_group"].iloc[gi])
        else:
            tech_labels.append("Unknown")  # positive cell far from any GEM plant

    pos_cells = pos_cells.copy()
    pos_cells["tech_label"] = tech_labels

    bf_cells  = pos_cells[pos_cells["tech_label"] == "BF-BOF"]
    eaf_cells = pos_cells[pos_cells["tech_label"] == "EAF"]
    unk_cells = pos_cells[pos_cells["tech_label"] == "Unknown"]
    print(f"  Positive cells: {len(pos_cells):,}  "
          f"(BF-BOF={len(bf_cells):,}, EAF={len(eaf_cells):,}, "
          f"Unknown={len(unk_cells):,})")

    # ── Build X / y ──────────────────────────────────────────────────────────
    drop_cols_base = [
        "IDCode", "GridProd_Steel_tot", "Month", "Year", "_merge",
        "Polygon_ID", "Centroid_Long", "Centroid_Lat",
        "PM2_5_MEAN", "PM2_5_MEAN_lag1", "PM2_5_MEAN_lag2", "PM2_5_MEAN_lag3",
        "LSTA_MEAN",  "LSTA_MEAN_lag1",  "LSTA_MEAN_lag2",  "LSTA_MEAN_lag3",
        "tech_label",
    ]
    # Also drop any extra iron production col if present
    for extra in ["GridProd_Iron_tot"]:
        if extra in combined.columns:
            drop_cols_base.append(extra)

    NEG_RATIO = 50  # negatives per positive — keeps dataset small; SMOTE then balances

    def _train_tech_model(pos_subset, neg_df, label_name, model_out_path):
        """Train MLP on pos_subset (positive) + subsampled negatives."""
        n_neg = min(len(neg_df), NEG_RATIO * len(pos_subset))
        neg_sample = neg_df.sample(n=n_neg, random_state=42)
        pos_c = pos_subset.copy(); pos_c["y"] = 1
        neg_c = neg_sample.copy(); neg_c["y"] = 0
        subset = pd.concat([pos_c, neg_c], ignore_index=True)
        print(f"  [{label_name}] neg subsampled: {n_neg:,} (ratio {NEG_RATIO}:1)")

        drop_cols = [c for c in drop_cols_base + ["y"] if c in subset.columns]
        X = subset.drop(columns=drop_cols)
        y_vals = subset["y"]

        print(f"\n  [{label_name}] X shape: {X.shape}  "
              f"positive rate: {y_vals.mean():.4f}")

        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import train_test_split
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.models import Sequential

        imputer = SimpleImputer(strategy="mean")
        X_imp   = imputer.fit_transform(X)
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X_imp)

        smote     = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_sc, y_vals)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tr, y_tr, test_size=0.125, random_state=42, stratify=y_tr)

        model = Sequential([
            Dense(64, input_shape=(X_tr.shape[1],), activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["accuracy"])
        model.fit(X_tr, y_tr,
                  validation_data=(X_va, y_va),
                  epochs=10, batch_size=512, verbose=1)

        from sklearn.metrics import classification_report
        y_pred = (model.predict(X_te) >= 0.5).astype(int).flatten()
        print(f"\n  [{label_name}] Test report:")
        print(classification_report(y_te, y_pred))

        model.save(model_out_path)
        imp_path = model_out_path.replace(".h5", "_imputer.pkl")
        sc_path  = model_out_path.replace(".h5", "_scaler.pkl")
        cols_path = model_out_path.replace(".h5", "_columns.pkl")
        joblib.dump(imputer, imp_path)
        joblib.dump(scaler,  sc_path)
        with open(cols_path, "wb") as f:
            pickle.dump(list(X.columns), f)
        print(f"  Saved: {model_out_path}")
        print(f"  Saved: {imp_path}, {sc_path}, {cols_path}")
        return model, imputer, scaler, list(X.columns)

    bf_model_path  = os.path.join(SCRIPT_DIR, "steel_location_mlp_bf.h5")
    eaf_model_path = os.path.join(SCRIPT_DIR, "steel_location_mlp_eaf.h5")

    print("\nTraining BF-BOF classifier ...")
    _train_tech_model(bf_cells, neg_cells, "BF-BOF", bf_model_path)

    print("\nTraining EAF classifier ...")
    _train_tech_model(eaf_cells, neg_cells, "EAF", eaf_model_path)

    print("\nPart B complete.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PART C — Apply tech-stratified models to SK and evaluate
# ─────────────────────────────────────────────────────────────────────────────
def run_part_c():
    print("\n" + "=" * 60)
    print("PART C — Apply tech-stratified models to SK")
    print("=" * 60)

    bf_model_path  = os.path.join(SCRIPT_DIR, "steel_location_mlp_bf.h5")
    eaf_model_path = os.path.join(SCRIPT_DIR, "steel_location_mlp_eaf.h5")

    for path in [bf_model_path, eaf_model_path]:
        if not os.path.exists(path):
            print(f"ERROR: Model not found: {path}")
            print("Run Part B first to train the technology-specific models.")
            return

    from tensorflow.keras.models import load_model as keras_load

    sk_data_path = need(DATA_SK, 'sk_data_prepared_nov.csv')
    if not os.path.exists(sk_data_path):
        print(f"ERROR: SK data not found: {sk_data_path}")
        return

    print(f"Loading SK data ...")
    sk = pd.read_csv(sk_data_path)
    print(f"  {len(sk):,} rows")

    gem = pd.read_csv(GEM_SK_PATH)
    gem["tech_group"] = gem["technology"].apply(classify_tech)

    def apply_model(model_path, sk_df, label):
        imp_path  = model_path.replace(".h5", "_imputer.pkl")
        sc_path   = model_path.replace(".h5", "_scaler.pkl")
        cols_path = model_path.replace(".h5", "_columns.pkl")

        model   = keras_load(model_path)
        imputer = joblib.load(imp_path)
        scaler  = joblib.load(sc_path)
        with open(cols_path, "rb") as f:
            model_cols = pickle.load(f)

        X = sk_df.reindex(columns=model_cols, fill_value=np.nan)
        X = X.apply(pd.to_numeric, errors="coerce")
        X_imp = imputer.transform(X)
        X_sc  = scaler.transform(X_imp)

        prob = model.predict(X_sc).flatten()
        pred = (prob >= THRESHOLD).astype(int)
        print(f"  [{label}] P >= {THRESHOLD}: {pred.sum():,} / {len(pred):,} cells")

        keep = [c for c in ["IDCode", "Year", "Month", "longitude", "latitude"]
                if c in sk_df.columns]
        out = sk_df[keep].copy()
        out["SteelProb"] = prob
        out["SteelPred"] = pred
        return out

    bf_pred  = apply_model(bf_model_path,  sk, "BF-BOF")
    eaf_pred = apply_model(eaf_model_path, sk, "EAF")

    # Save predictions
    bf_out  = os.path.join(OUT_DIR, "sk_location_predictions_bf.csv")
    eaf_out = os.path.join(OUT_DIR, "sk_location_predictions_eaf.csv")
    bf_pred.to_csv(bf_out,   index=False)
    eaf_pred.to_csv(eaf_out, index=False)
    print(f"  Saved: {bf_out}")
    print(f"  Saved: {eaf_out}")

    # Coverage check — BF model vs BF-BOF plants, EAF model vs EAF plants
    def coverage(pred_df, gem_sub, d_km=10.0):
        cell_max = pred_df.groupby(["latitude", "longitude"])["SteelProb"].max().reset_index()
        tree = BallTree(np.deg2rad(cell_max[["latitude", "longitude"]].values),
                        metric="haversine")
        rows = []
        for _, plant in gem_sub.iterrows():
            q   = np.deg2rad([[plant["lat"], plant["lon"]]])
            idx = tree.query_radius(q, r=d_km / EARTH_R)[0]
            mp  = float(cell_max["SteelProb"].iloc[idx].max()) if len(idx) else 0.0
            rows.append({"plant": plant["plant_name"],
                         "max_prob": mp, "detected": mp >= THRESHOLD})
        return pd.DataFrame(rows)

    gem_bf  = gem[gem["tech_group"] == "BF-BOF"]
    gem_eaf = gem[gem["tech_group"] == "EAF"]

    cov_bf  = coverage(bf_pred,  gem_bf)
    cov_eaf = coverage(eaf_pred, gem_eaf)

    print(f"\n  BF-BOF model vs BF-BOF plants: "
          f"{cov_bf['detected'].sum()}/{len(cov_bf)} detected")
    for _, r in cov_bf.iterrows():
        s = "DETECTED" if r["detected"] else "missed"
        print(f"    [{s:8s}] {r['plant']:40s} P={r['max_prob']:.4f}")

    print(f"\n  EAF model vs EAF plants: "
          f"{cov_eaf['detected'].sum()}/{len(cov_eaf)} detected")
    for _, r in cov_eaf.iterrows():
        s = "DETECTED" if r["detected"] else "missed"
        print(f"    [{s:8s}] {r['plant']:40s} P={r['max_prob']:.4f}")

    # Compare with original model recall
    orig_pred = pd.read_csv(SK_PRED_PATH)
    orig_max  = orig_pred.groupby(["latitude", "longitude"])["SteelProb"].max().reset_index()
    orig_tree = BallTree(np.deg2rad(orig_max[["latitude", "longitude"]].values),
                         metric="haversine")
    orig_rows = []
    for _, plant in gem.iterrows():
        q   = np.deg2rad([[plant["lat"], plant["lon"]]])
        idx = orig_tree.query_radius(q, r=10.0 / EARTH_R)[0]
        mp  = float(orig_max["SteelProb"].iloc[idx].max()) if len(idx) else 0.0
        orig_rows.append({"plant": plant["plant_name"],
                          "tech":  plant["tech_group"],
                          "max_prob": mp, "detected": mp >= THRESHOLD})
    orig_df = pd.DataFrame(orig_rows)

    orig_bf_det  = orig_df[orig_df["tech"] == "BF-BOF"]["detected"].sum()
    orig_eaf_det = orig_df[orig_df["tech"] == "EAF"]["detected"].sum()

    print("\n  ── Summary: original vs tech-stratified ──────────────────────")
    print(f"  Original model  | BF-BOF: {orig_bf_det}/{len(gem_bf)} | "
          f"EAF: {orig_eaf_det}/{len(gem_eaf)}")
    print(f"  Tech-stratified | BF-BOF: {cov_bf['detected'].sum()}/{len(gem_bf)} | "
          f"EAF: {cov_eaf['detected'].sum()}/{len(gem_eaf)}")

    print("\nPart C complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("a", "all"):
        run_part_a()

    if mode in ("b", "all"):
        run_part_b()

    if mode in ("c", "all"):
        run_part_c()

    print("\nDone.")
