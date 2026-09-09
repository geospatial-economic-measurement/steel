import calendar
import itertools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from _paths import DATA_NK

# -- Config ---------------------------------------------------------------------
NK_DIR   = DATA_NK
RAW_DIR  = os.path.join(NK_DIR, "raw")
MERGED   = os.path.join(NK_DIR, "nk_satellite_2022_2024_v2.csv")
PREPARED = os.path.join(NK_DIR, "nk_data_prepared_v2.csv")

CFG = dict(
    name="North Korea",
    lon_min=124.0, lon_max=130.8,
    lat_min=37.5,  lat_max=43.1,
    prefix="nk", tile_label="NK",
)

# (year, month) range: Oct 2022 .. Dec 2024
YM = [(2022, m) for m in (10, 11, 12)] \
   + [(y, m) for y in (2023, 2024) for m in range(1, 13)]

SIGNALS     = ["CO", "SO2", "NO2", "O3", "PM10", "NTL", "LST"]
SCALE_M     = 1000
TILE_SIZE   = 0.5
MAX_WORKERS = 5
TILE_TIMEOUT = 120
GEE_PROJECT = "ee-jianweiairuc"

os.makedirs(RAW_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -- GEE download (identical machinery to script 35) ------------------------------
def init_gee():
    import ee
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
    return ee


def month_date_range(year, month):
    last = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"


def build_monthly_image(ee, year, month, bbox):
    start, end = month_date_range(year, month)

    def s5p(cid, band):
        return (ee.ImageCollection(cid).filterDate(start, end)
                  .filterBounds(bbox).select(band).mean())

    def modis(cid, band, sf):
        return (ee.ImageCollection(cid).filterDate(start, end)
                  .filterBounds(bbox).select(band).mean().multiply(sf))

    co   = s5p("COPERNICUS/S5P/OFFL/L3_CO",  "CO_column_number_density")
    no2  = s5p("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density")
    so2  = s5p("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density")
    o3   = s5p("COPERNICUS/S5P/OFFL/L3_O3",  "O3_column_number_density")
    pm10 = modis("MODIS/061/MCD19A2_GRANULES", "Optical_Depth_047", 0.001)
    lst  = modis("MODIS/061/MOD11A1",           "LST_Day_1km",       0.02)
    ntl  = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
              .filterDate(start, end).filterBounds(bbox).select("avg_rad").mean())

    return (co.rename("CO").addBands(no2.rename("NO2")).addBands(so2.rename("SO2"))
              .addBands(o3.rename("O3")).addBands(pm10.rename("PM10"))
              .addBands(lst.rename("LST")).addBands(ntl.rename("NTL")))


def fetch_tile(ee, image, lon0, lat0, lon1, lat1):
    tile_bbox = ee.Geometry.Rectangle([lon0, lat0, lon1, lat1])
    fc = image.sample(region=tile_bbox, scale=SCALE_M, geometries=True,
                      dropNulls=False)
    features = fc.getInfo()["features"]
    rows = []
    for feat in features:
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        row = {"lon": coords[0], "lat": coords[1]}
        row.update(props)
        rows.append(row)
    return rows


def image_to_df_tiled(ee, image, year, month):
    lon_edges = np.arange(CFG["lon_min"], CFG["lon_max"], TILE_SIZE)
    lat_edges = np.arange(CFG["lat_min"], CFG["lat_max"], TILE_SIZE)
    tiles = []
    for lon0 in lon_edges:
        lon1 = min(lon0 + TILE_SIZE, CFG["lon_max"])
        for lat0 in lat_edges:
            lat1 = min(lat0 + TILE_SIZE, CFG["lat_max"])
            tiles.append((lon0, lat0, lon1, lat1))

    all_rows, errors = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        fut_map = {exe.submit(fetch_tile, ee, image, lo, la, lo1, la1):
                   (lo, la, lo1, la1) for lo, la, lo1, la1 in tiles}
        for fut in as_completed(fut_map, timeout=TILE_TIMEOUT * len(tiles)):
            try:
                all_rows.extend(fut.result(timeout=TILE_TIMEOUT))
            except Exception as exc:
                errors.append((fut_map[fut], str(exc)[:120]))

    if errors:
        log(f"  WARNING: {len(errors)} of {len(tiles)} tiles failed:")
        for t, e in errors[:5]:
            log(f"    tile {t}: {e}")
        raise RuntimeError(f"{len(errors)} tiles failed — month left for retry")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["year"] = year
        df["month"] = month
    return df


def download(months=None):
    ee = init_gee()
    todo = months if months else YM
    total = len(todo)
    for i, (year, month) in enumerate(todo, 1):
        out_path = os.path.join(RAW_DIR, f"nk_{year}_{month:02d}.csv")
        if os.path.exists(out_path):
            n = sum(1 for _ in open(out_path)) - 1
            log(f"[{i}/{total}] SKIP {year}-{month:02d} ({n:,} rows exist)")
            continue
        log(f"[{i}/{total}] Downloading {year}-{month:02d} ...")
        try:
            img = build_monthly_image(ee, year, month,
                ee.Geometry.Rectangle([CFG["lon_min"], CFG["lat_min"],
                                       CFG["lon_max"], CFG["lat_max"]]))
            df = image_to_df_tiled(ee, img, year, month)
            if df.empty:
                log(f"  WARNING: no data for {year}-{month:02d}")
                continue
            df["grid_id"] = (df["lon"].round(4).astype(str) + "_"
                             + df["lat"].round(4).astype(str))
            df["tile"] = CFG["tile_label"]
            cols = ["grid_id", "year", "month", "lon", "lat", "tile"] + SIGNALS
            df = df[[c for c in cols if c in df.columns]]
            df.to_csv(out_path, index=False)
            log(f"  -> OK  {len(df):,} rows  {os.path.basename(out_path)}")
        except Exception as exc:
            log(f"  ERROR {year}-{month:02d}: {str(exc)[:200]}")


def prepare():
    """Merge monthly raw CSVs and build the lagged prediction panel."""
    log("Merging monthly CSVs ...")
    dfs, missing = [], []
    for year, month in YM:
        p = os.path.join(RAW_DIR, f"nk_{year}_{month:02d}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
        else:
            missing.append(f"{year}-{month:02d}")
    if missing:
        log(f"  WARNING: missing months (rerun download): {', '.join(missing)}")
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(MERGED, index=False)
    log(f"  Merged: {len(merged):,} rows -> {os.path.basename(MERGED)}")

    # Rename to the IR/China *_MEAN convention (LST day series -> LSTT_MEAN,
    # matching ir_data_prepared.csv / the legacy NK panel).
    ren = {s: f"{s}_MEAN" for s in ("CO", "SO2", "NO2", "O3", "PM10", "NTL")}
    ren["LST"] = "LSTT_MEAN"
    df = merged.rename(columns=ren)
    sigs = list(ren.values())

    df["IDCode"] = df["grid_id"]
    df = (df[["IDCode", "year", "month", "lon", "lat"] + sigs]
          .sort_values(["IDCode", "year", "month"]).reset_index(drop=True))

    log("Computing lag features (1-3 months, within grid cell) ...")
    df["t"] = (df["year"] * 12 + df["month"]).astype(int)
    for lag in (1, 2, 3):
        shifted = df.groupby("IDCode")[sigs + ["t"]].shift(lag)
        gap_ok = (df["t"] - shifted["t"]) == lag
        for s in sigs:
            df[f"{s}_lag{lag}"] = shifted[s].where(gap_ok)
    df = df.drop(columns=["t"])

    n0 = len(df)
    lag_cols = [f"{s}_lag{k}" for s in sigs for k in (1, 2, 3)]
    df = df.dropna(subset=lag_cols, how="all")
    # Keep Jan 2023+ (2022 Q4 is lag context only)
    df = df[(df["year"] >= 2023)].reset_index(drop=True)
    log(f"  Panel: {n0:,} -> {len(df):,} rows (2023+, lag features attached)")

    df.to_csv(PREPARED, index=False)
    log(f"Saved: {PREPARED}")
    per_month = df.groupby(["year", "month"]).size()
    log("Rows per month:\n" + per_month.to_string())


def main():
    args = sys.argv[1:]
    if not args:
        download()
        prepare()
    elif args[0] == "download":
        if len(args) == 3:
            download([(int(args[1]), int(args[2]))])
        else:
            download()
    elif args[0] == "prepare":
        prepare()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
