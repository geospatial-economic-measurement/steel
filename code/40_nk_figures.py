import os
import sys
import time
import pickle

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from _paths import OUT_FIG, REP_ROOT

# -- Paths ---------------------------------------------------------------------
REP_ROOT   = REP_ROOT
ART_DIR    = os.path.join(REP_ROOT, "code", "model_artifacts")
NK_DIR     = os.path.join(REP_ROOT, "data", "NK")
NK_2024    = os.path.join(NK_DIR, "nk_data_2024.csv")           # legacy (unused)
# Fresh GEE panel built by 39_nk_download_prepare_gee.py — complete coverage
# Jan 2023 - Dec 2024 (24 months x 472,368 cells, lag features precomputed).
NK_PANEL   = os.path.join(NK_DIR, "nk_data_prepared_v2.csv")
# v2 panel uses lowercase meta names; canonicalize to the legacy convention.
PANEL_RENAME = {"year": "Year", "month": "Month",
                "lon": "longitude", "lat": "latitude"}
GEM_PATH   = os.path.join(REP_ROOT, "data", "raw",
    "Plant-level-data-Global-Iron-and-Steel-Tracker-March-2026-V1.xlsx")

MLP_MODEL   = os.path.join(ART_DIR, "steel_location_mlp.h5")
MLP_IMPUTER = os.path.join(ART_DIR, "location_imputer.pkl")
MLP_SCALER  = os.path.join(ART_DIR, "location_scaler.pkl")
MLP_COLS    = os.path.join(ART_DIR, "model_columns.pkl")
XGB_PIPE    = os.path.join(ART_DIR, "china_xgb_pipeline.pkl")

FIG_DIRS = [
    OUT_FIG,
    OUT_FIG,
]

PRED_CSV   = os.path.join(NK_DIR, "nk_location_predictions.csv")
TOTALS_CSV = os.path.join(NK_DIR, "nk_totals_by_month.csv")

# -- Constants -----------------------------------------------------------------
EARTH_R   = 6371.0
THRESHOLD = 0.5
RADIUS_KM = 10.0
GAMMA     = 0.22
GRID_STEP = 1.0 / 111.0   # native NK grid spacing (0.009009... deg)
NK_BBOX   = dict(lon_min=123.5, lon_max=131.2, lat_min=37.3, lat_max=43.3)
CHUNK     = 1_000_000
JMP_BLUE  = "#00468B"

# NK hub / port coordinates (lat, lon)
NK_HUBS = {
    "Chongjin": (41.79, 129.79),
    "Kimchaek": (40.67, 129.20),
    "Songnim":  (38.75, 125.65),
    "Nampo":    (38.73, 125.41),
}
NK_PORTS = {
    "Nampo":         (38.71, 125.40),
    "Chongjin_port": (41.78, 129.82),
    "Wonsan":        (39.17, 127.44),
    "Hungnam":       (39.83, 127.62),
}

for d in FIG_DIRS:
    os.makedirs(d, exist_ok=True)


_NK_GEOM = None


def in_north_korea(lat, lon, buffer_deg=0.02):
    """True for coordinates inside North Korea (Natural Earth 10m polygon).

    The data bbox necessarily includes Chinese and South Korean territory
    (Dandong/Yanji in the north, the Seoul area in the south). All NK maps
    and aggregates are restricted to the DPRK polygon, slightly buffered so
    coastal grid-cell centroids on the shoreline are retained. Requires
    internet once to fetch the Natural Earth shapefile (cached afterwards).
    """
    global _NK_GEOM
    if _NK_GEOM is None:
        from cartopy.io import shapereader
        import shapely
        from shapely.ops import unary_union
        shp = shapereader.natural_earth(resolution="10m", category="cultural",
                                        name="admin_0_countries")
        recs = [r.geometry for r in shapereader.Reader(shp).records()
                if r.attributes.get("ADMIN") == "North Korea"]
        _NK_GEOM = unary_union(recs).buffer(buffer_deg)
    import shapely
    return shapely.contains_xy(_NK_GEOM,
                               np.asarray(lon, dtype=float),
                               np.asarray(lat, dtype=float))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -- Generic helpers (mirrored from 33_multi_country_output.py) -----------------
def safe_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None


def ensure_schema(df, feature_names):
    out = pd.DataFrame(index=df.index)
    for c in feature_names:
        out[c] = df[c] if c in df.columns else 0.0
    return out[feature_names]


def haversine_km(lat1, lon1, lat2, lon2):
    rlat1, rlon1 = np.deg2rad(lat1), np.deg2rad(lon1)
    rlat2, rlon2 = np.deg2rad(lat2), np.deg2rad(lon2)
    dlat, dlon = rlat2 - rlat1, rlon2 - rlon1
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def add_distance_features(df, ports, hubs, lat_col="latitude", lon_col="longitude"):
    df = df.copy()
    lats, lons = df[lat_col].values, df[lon_col].values
    port_d = {}
    for name, (plat, plon) in ports.items():
        d = haversine_km(lats, lons, plat, plon)
        df[f"dist_{name}"] = d
        port_d[name] = d
    hub_d = {}
    for name, (hlat, hlon) in hubs.items():
        hub_d[name] = haversine_km(lats, lons, hlat, hlon)
    df["dist_nearest_port"] = np.stack(list(port_d.values()), axis=1).min(axis=1)
    df["dist_nearest_hub"]  = np.stack(list(hub_d.values()),  axis=1).min(axis=1)
    return df


def add_gem_capacity(df, gem, lat_col="latitude", lon_col="longitude",
                     radius_km=15.0):
    """Copied from 33_multi_country_output.py. NOTE: china_xgb_pipeline's 43
    features include no capacity columns, so this is only applied if the model
    schema requests them (it does not for the current pipeline)."""
    df = df.copy()
    for c in ["nom_capacity_ttpa", "bof_capacity_ttpa", "eaf_capacity_ttpa"]:
        df[c] = 0.0
    if gem is None or len(gem) == 0:
        return df
    plant_rad = np.deg2rad(gem[["lat", "lon"]].values.astype(float))
    tree = BallTree(plant_rad, metric="haversine")
    grid_rad = np.deg2rad(df[[lat_col, lon_col]].values.astype(float))
    dists_rad, idx = tree.query(grid_rad, k=1)
    dists_km = dists_rad[:, 0] * EARTH_R
    idx = idx[:, 0]
    within = dists_km <= radius_km
    for c in ["nom_capacity_ttpa", "bof_capacity_ttpa", "eaf_capacity_ttpa"]:
        if c in gem.columns:
            df[c] = np.where(within, gem[c].values[idx].astype(float), 0.0)
    return df


def load_gem_nk():
    """All North Korean GEM plants regardless of status (status reported)."""
    plants = pd.read_excel(GEM_PATH, sheet_name="Plant data", engine="openpyxl")
    plants.columns = [c.strip() for c in plants.columns]
    sub = plants[plants["Country/area"] == "North Korea"].copy()
    coords = sub["Coordinates"].astype(str).str.split(",", expand=True)
    sub["lat"] = pd.to_numeric(coords[0], errors="coerce")
    sub["lon"] = pd.to_numeric(coords[1], errors="coerce")
    sub = sub.rename(columns={"Plant name (English)": "plant_name"})
    sub = sub.dropna(subset=["lat", "lon"])

    caps = pd.read_excel(GEM_PATH, sheet_name="Plant capacities and status",
                         engine="openpyxl")
    caps.columns = [c.strip() for c in caps.columns]
    caps = caps[["GEM plant ID", "Status"]].drop_duplicates("GEM plant ID")
    sub = sub.merge(caps, on="GEM plant ID", how="left")
    sub["Status"] = sub["Status"].fillna("unknown")
    return sub[["plant_name", "lat", "lon", "Status"]].reset_index(drop=True)


def compute_coverage(land_cells, prob_col, gem):
    tree = BallTree(np.deg2rad(land_cells[["lat", "lon"]].values),
                    metric="haversine")
    results = []
    for _, p in gem.iterrows():
        q   = np.deg2rad([[p["lat"], p["lon"]]])
        idx = tree.query_radius(q, r=RADIUS_KM / EARTH_R)[0]
        mp  = float(land_cells[prob_col].iloc[idx].max()) if len(idx) else 0.0
        results.append({
            "name": p["plant_name"], "status": p["Status"],
            "lat": p["lat"], "lon": p["lon"],
            "prob": mp, "detected": mp >= THRESHOLD,
        })
    return pd.DataFrame(results)


# ================================================================================
# PART 1 — Location heatmap
# ================================================================================
def run_location():
    log("PART 1: location heatmap")

    if os.path.exists(PRED_CSV):
        log(f"Loading cached predictions: {PRED_CSV}")
        cells = pd.read_csv(PRED_CSV)
    else:
        log("Loading MLP location model ...")
        from tensorflow.keras.models import load_model as keras_load
        mlp     = keras_load(MLP_MODEL)
        imputer = joblib.load(MLP_IMPUTER)
        scaler  = joblib.load(MLP_SCALER)
        with open(MLP_COLS, "rb") as f:
            cols = pickle.load(f)

        # Stream the prepared panel for Year==2024 (all 12 months) to obtain
        # the per-cell MAX monthly probability.
        raw_header = pd.read_csv(NK_PANEL, nrows=0).columns.tolist()
        header = [PANEL_RENAME.get(c, c) for c in raw_header]
        inv = {v: k for k, v in PANEL_RENAME.items()}
        missing = [c for c in cols if c not in header]
        log(f"  model expects {len(cols)} features; "
            f"missing in NK panel (filled 0): {missing if missing else 'none'}")

        meta = ["IDCode", "Year", "Month", "longitude", "latitude"]
        use_cols = [inv.get(c, c) for c in header if c in set(meta) | set(cols)]
        dtypes = {c: "float32" for c in use_cols
                  if c not in ("IDCode", "year", "month")}
        dtypes.update({"year": "int16", "month": "int8", "IDCode": "str"})

        best, t0, chunk_i = None, time.time(), 0
        log(f"Streaming {NK_PANEL} ({CHUNK:,}-row chunks, 2024 Jan-Dec) ...")
        for chunk in pd.read_csv(NK_PANEL, usecols=use_cols, dtype=dtypes,
                                 chunksize=CHUNK):
            chunk_i += 1
            chunk = chunk.rename(columns=PANEL_RENAME)
            chunk = chunk[chunk["Year"] == 2024]
            if len(chunk) == 0:
                log(f"  chunk {chunk_i}: 0 rows kept "
                    f"({time.time()-t0:.0f}s elapsed)")
                continue
            X  = chunk.reindex(columns=cols, fill_value=0)
            xb = scaler.transform(imputer.transform(X))
            pr = mlp.predict(xb, batch_size=16384, verbose=0).ravel()
            part = pd.DataFrame({
                "IDCode": chunk["IDCode"].values,
                "lat": chunk["latitude"].values.astype(float),
                "lon": chunk["longitude"].values.astype(float),
                "max_prob": pr,
                "has_ntl": chunk["NTL_MEAN"].notna().values.astype(int),
            })
            part = (part.groupby("IDCode")
                        .agg(lat=("lat", "first"), lon=("lon", "first"),
                             max_prob=("max_prob", "max"),
                             has_ntl=("has_ntl", "max"))
                        .reset_index())
            best = part if best is None else (
                pd.concat([best, part], ignore_index=True)
                  .groupby("IDCode")
                  .agg(lat=("lat", "first"), lon=("lon", "first"),
                       max_prob=("max_prob", "max"),
                       has_ntl=("has_ntl", "max"))
                  .reset_index())
            log(f"  chunk {chunk_i}: {len(chunk):,} rows scored, "
                f"{len(best):,} cells tracked ({time.time()-t0:.0f}s elapsed)")

        cells = best
        n0 = len(cells)
        cells = cells[cells["has_ntl"] == 1].drop(columns=["has_ntl"])
        log(f"  {len(cells):,} land cells (dropped {n0 - len(cells):,} no-NTL)")

        cells.to_csv(PRED_CSV, index=False)
        log(f"  Saved: {PRED_CSV}")

    # Exclude cells south of the Military Demarcation Line (South Korean territory)
    n_before = len(cells)
    cells = cells[in_north_korea(cells["lat"].values, cells["lon"].values)].copy()
    log(f"DPRK polygon mask: {n_before:,} -> {len(cells):,} cells (removed {n_before - len(cells):,} outside North Korea)")

    # GEM plants + coverage (operating plants only for the headline count)
    gem = load_gem_nk()
    log(f"GEM NK plants: {len(gem)}  status: "
        + ", ".join(f"{k}={v}" for k, v in gem["Status"].value_counts().items()))
    gem = gem[gem["Status"].astype(str).str.lower().str.strip() == "operating"].reset_index(drop=True)
    log(f"Operating plants used for detection: {len(gem)}")
    cov = compute_coverage(cells, "max_prob", gem)
    n_det = int(cov["detected"].sum())
    log(f"Detection: {n_det}/{len(gem)} operating plants (P>={THRESHOLD} within {RADIUS_KM:.0f} km)")
    for _, r in cov.iterrows():
        log(f"  [{'DETECTED' if r['detected'] else 'missed  '}] "
            f"[{r['status']:>9s}] {r['name'][:45]:45s} P={r['prob']:.4f}")

    # -- Heatmap (style mirrors 36_iran_location_classifier.py) -----------------
    log("Rendering heatmap ...")
    bb = NK_BBOX
    lat_arr = np.arange(bb["lat_min"], bb["lat_max"] + GRID_STEP, GRID_STEP)
    lon_arr = np.arange(bb["lon_min"], bb["lon_max"] + GRID_STEP, GRID_STEP)
    prob_grid = np.full((len(lat_arr), len(lon_arr)), np.nan)
    land_grid = np.zeros_like(prob_grid, dtype=bool)

    inb = cells[
        cells["lat"].between(bb["lat_min"], bb["lat_max"]) &
        cells["lon"].between(bb["lon_min"], bb["lon_max"])
    ]
    ii = np.round((inb["lat"].values - lat_arr[0]) / GRID_STEP).astype(int)
    jj = np.round((inb["lon"].values - lon_arr[0]) / GRID_STEP).astype(int)
    ok = (ii >= 0) & (ii < len(lat_arr)) & (jj >= 0) & (jj < len(lon_arr))
    prob_grid[ii[ok], jj[ok]] = inb["max_prob"].values[ok]
    land_grid[ii[ok], jj[ok]] = True

    detected = cov[cov["detected"]]
    missed   = cov[~cov["detected"]]

    # Light basemap with country boundaries (cartopy Natural Earth; downloads
    # NE shapefiles on first use — needs internet once, then cached).
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import LinearSegmentedColormap

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(11, 8.6), facecolor="white")
    ax = plt.axes(projection=proj)
    ax.set_extent([bb["lon_min"], bb["lon_max"], bb["lat_min"], bb["lat_max"]],
                  crs=proj)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dbe9f4", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"),  facecolor="#f2f0eb", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6,
                   edgecolor="0.35", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=1.1,
                   edgecolor="0.15", linestyle="-", zorder=4)

    LONM, LATM = np.meshgrid(lon_arr, lat_arr)
    vis = np.power(np.where(np.isnan(prob_grid), np.nan, prob_grid), GAMMA)
    # transparent -> gold -> red ramp (reads on the light basemap)
    heat = LinearSegmentedColormap.from_list(
        "nkheat", [(0.00, (1.0, 0.85, 0.2, 0.0)),
                   (0.25, (1.0, 0.78, 0.1, 0.55)),
                   (0.60, (0.95, 0.35, 0.05, 0.85)),
                   (1.00, (0.75, 0.0, 0.05, 0.95))])
    ax.pcolormesh(LONM, LATM, np.ma.masked_invalid(vis),
                  cmap=heat, vmin=0, vmax=1, shading="auto",
                  transform=proj, zorder=3)

    offsets = [(7, 7), (7, -13), (7, 20), (-7, -13), (7, 32)]
    for i, (_, d) in enumerate(detected.iterrows()):
        ax.plot(d["lon"], d["lat"], "*", color="#1a9c1a", ms=13,
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=6, transform=proj)
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(str(d["name"])[:40], (d["lon"], d["lat"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color="0.15", zorder=7,
                    ha="left" if dx > 0 else "right")
    for i, (_, m) in enumerate(missed.iterrows()):
        ax.plot(m["lon"], m["lat"], "*", color="#1668c8", ms=12,
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=6, transform=proj)
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(str(m["name"])[:40], (m["lon"], m["lat"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color="0.15", zorder=7,
                    ha="left" if dx > 0 else "right")

    legend_els = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#1a9c1a",
               markeredgecolor="white", ms=13, linestyle="",
               label=f"Detected ({n_det})"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#1668c8",
               markeredgecolor="white", ms=12, linestyle="",
               label=f"Missed ({len(gem) - n_det})"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=9,
              facecolor="white", edgecolor="0.5")

    cbar_ax = fig.add_axes([0.92, 0.12, 0.013, 0.74])
    sm = plt.cm.ScalarMappable(cmap=heat, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("P(steel plant)$^{0.22}$  [$\\gamma$-compressed]", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="0.75",
                      linestyle=":", x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    ax.set_title(
        f"North Korea: Steel Plant Location Prediction — "
        f"{n_det}/{len(gem)} operating GEM plants detected",
        fontsize=12, pad=12)

    for d in FIG_DIRS:
        out = os.path.join(d, "nk_location_heatmap.png")
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        log(f"  Saved: {out}")
    plt.close(fig)
    return n_det, len(gem)


# ================================================================================
# PART 2 — Monthly national output totals
# ================================================================================
def stream_totals(year_filter=None):
    """Stream the 5 GB panel through china_xgb_pipeline; return monthly totals."""
    log(f"Loading XGB pipeline: {XGB_PIPE} ...")
    model = joblib.load(XGB_PIPE)
    feat_names = safe_feature_names(model)
    log(f"  model expects {len(feat_names)} features")

    raw_header = pd.read_csv(NK_PANEL, nrows=0).columns.tolist()
    header = [PANEL_RENAME.get(c, c) for c in raw_header]
    inv = {v: k for k, v in PANEL_RENAME.items()}
    missing = [c for c in feat_names if c not in header + [
        "IDCode", "Centroid_Long", "Centroid_Lat", "month_sin", "month_cos"]]
    log(f"  features missing from NK panel (filled 0): {missing}")

    meta = ["Year", "Month", "longitude", "latitude"]
    use_cols = [inv.get(c, c) for c in header if c in set(meta) | set(feat_names)]

    cap_feats = [c for c in ("nom_capacity_ttpa", "bof_capacity_ttpa",
                             "eaf_capacity_ttpa") if c in feat_names]
    gem_cap = load_gem_nk() if cap_feats else None

    bb = NK_BBOX
    sums = {}   # (Year, Month) -> running total
    row_offset, chunk_i, t0 = 0, 0, time.time()
    log(f"Streaming {NK_PANEL} in {CHUNK:,}-row chunks"
        + (f" (Year=={year_filter} only)" if year_filter else "") + " ...")

    for chunk in pd.read_csv(NK_PANEL, usecols=use_cols, chunksize=CHUNK,
                             low_memory=False):
        chunk_i += 1
        n_read = chunk_i * CHUNK
        chunk = chunk.rename(columns=PANEL_RENAME)
        # Coerce to numeric: defensive against non-numeric rows.
        for c in chunk.columns:
            if chunk[c].dtype == object:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        chunk = chunk.dropna(subset=["Year", "Month", "longitude", "latitude"])
        chunk["Year"]  = chunk["Year"].astype(int)
        chunk["Month"] = chunk["Month"].astype(int)
        if year_filter is not None:
            chunk = chunk[chunk["Year"] == year_filter]
        mask = (
            chunk["longitude"].between(bb["lon_min"], bb["lon_max"]) &
            chunk["latitude"].between(bb["lat_min"], bb["lat_max"]) &
            in_north_korea(chunk["latitude"].values, chunk["longitude"].values)
        )
        chunk = chunk[mask].copy()
        if len(chunk) == 0:
            log(f"  chunk {chunk_i}: 0 rows kept (~{n_read:,} read, "
                f"{time.time()-t0:.0f}s elapsed)")
            continue

        chunk["IDCode"] = np.arange(row_offset, row_offset + len(chunk),
                                    dtype=float)
        row_offset += len(chunk)

        chunk = add_distance_features(chunk, NK_PORTS, NK_HUBS)
        if cap_feats:
            chunk = add_gem_capacity(chunk, gem_cap)
        chunk["Centroid_Long"] = chunk["dist_nearest_port"]
        chunk["Centroid_Lat"]  = chunk["dist_nearest_hub"]
        chunk["month_sin"] = np.sin(2 * np.pi * chunk["Month"] / 12.0)
        chunk["month_cos"] = np.cos(2 * np.pi * chunk["Month"] / 12.0)

        Xc = ensure_schema(chunk, feat_names)
        yhat = model.predict(Xc)
        del Xc
        pred = np.maximum(0, np.expm1(yhat))

        g = (pd.DataFrame({"Year": chunk["Year"].values,
                           "Month": chunk["Month"].values,
                           "p": pred})
             .groupby(["Year", "Month"])["p"].agg(["sum", "count"]))
        for k, row in g.iterrows():
            key = (int(k[0]), int(k[1]))
            s, n = sums.get(key, (0.0, 0))
            sums[key] = (s + float(row["sum"]), n + int(row["count"]))
        del chunk
        log(f"  chunk {chunk_i}: {row_offset:,} NK rows predicted "
            f"(~{n_read:,} read, {time.time()-t0:.0f}s elapsed)")

    totals = (pd.DataFrame([(y, m, v[0], v[1]) for (y, m), v in sums.items()],
                           columns=["Year", "Month", "Total_Steel", "N_cells"])
              .sort_values(["Year", "Month"]).reset_index(drop=True))
    # Keep Jan 2023 - Dec 2024 (fresh v2 panel has complete coverage)
    totals = totals[totals["Year"].isin([2023, 2024])].reset_index(drop=True)
    # Satellite coverage relative to the best-covered month. The GEE download
    # is incomplete for some months (e.g. Dec 2023 has ~28% of cells);
    # downstream plotting flags months with coverage < 95%.
    totals["coverage"] = totals["N_cells"] / totals["N_cells"].max()
    return totals


def merge_totals(new_totals):
    """Merge partial-year totals into TOTALS_CSV (replacing same Year rows)."""
    if os.path.exists(TOTALS_CSV):
        old = pd.read_csv(TOTALS_CSV)
        old = old[~old["Year"].isin(new_totals["Year"].unique())]
        new_totals = pd.concat([old, new_totals], ignore_index=True)
    new_totals = new_totals.sort_values(["Year", "Month"]).reset_index(drop=True)
    new_totals.to_csv(TOTALS_CSV, index=False)
    log(f"Saved: {TOTALS_CSV} ({len(new_totals)} month rows)")
    return new_totals


def plot_totals():
    """Paper-style monthly totals figure from TOTALS_CSV."""
    totals = pd.read_csv(TOTALS_CSV).sort_values(["Year", "Month"])
    totals["date"] = pd.to_datetime(dict(year=totals["Year"],
                                         month=totals["Month"], day=1))
    peak = totals["Total_Steel"].max()
    log(f"Monthly totals: min={totals['Total_Steel'].min():,.0f}, "
        f"max={peak:,.0f} (model units), {len(totals)} months")

    if "coverage" not in totals.columns:
        totals["coverage"] = 1.0
    complete = totals[totals["coverage"] >= 0.95]
    partial  = totals[totals["coverage"] < 0.95]
    if len(partial):
        log("Months with incomplete satellite coverage (<95%), shown as open markers: "
            + ", ".join(f"{int(r.Year)}-{int(r.Month):02d} ({r.coverage:.0%})"
                        for r in partial.itertuples()))

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(complete["date"], complete["Total_Steel"],
            color=JMP_BLUE, linewidth=1.8, marker="o", markersize=5,
            markerfacecolor=JMP_BLUE,
            label="Complete satellite coverage")
    if len(partial):
        ax.plot(partial["date"], partial["Total_Steel"],
                linestyle="", marker="o", markersize=6,
                markerfacecolor="white", markeredgecolor="grey",
                markeredgewidth=1.2,
                label="Incomplete satellite coverage (excluded)")
        ax.legend(fontsize=9, frameon=False, loc="lower right")

    ax.set_xlabel("Month", fontsize=12)
    # Model predicts in same units as GridProd_Steel_tot; NK absolute levels
    # are anchored separately in the paper, so keep raw model units.
    ax.set_ylabel("Total predicted output (model units)", fontsize=12)

    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", color="grey", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate(rotation=0, ha="center")
    plt.tight_layout()

    for d in FIG_DIRS:
        out = os.path.join(d, "nk_total_by_month_v2.png")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        log(f"  Saved: {out}")
    plt.close(fig)
    return totals


def run_output(year_filter=None, make_fig=True):
    log("PART 2: monthly national output totals")
    totals = stream_totals(year_filter=year_filter)
    totals = merge_totals(totals)
    have = set(zip(totals["Year"], totals["Month"]))
    complete = all((2023, m) in have for m in range(1, 13)) and \
               all((2024, m) in have for m in range(1, 11))
    if make_fig and complete:
        plot_totals()
    elif make_fig:
        log("Totals CSV incomplete (Jan 2023 - Oct 2024 not all present); "
            "run remaining year then 'outputfig'.")


# ================================================================================
def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if mode in ("all", "location"):
        run_location()
    if mode == "all" or mode == "output":
        run_output()
    elif mode == "output2023":
        run_output(year_filter=2023, make_fig=False)
    elif mode == "output2024":
        run_output(year_filter=2024)
    elif mode == "outputfig":
        plot_totals()
    log("Done.")


if __name__ == "__main__":
    main()
