
import calendar
import itertools
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from _paths import DATA, DATA_IR

# ── Country configurations ───────────────────────────────────────────────────
COUNTRIES = {
    "JP": dict(
        name="Japan",
        lon_min=129.5, lon_max=146.0,
        lat_min=30.5,  lat_max=45.5,
        data_dir=os.path.join(DATA, "JP"),
        prefix="jp",
        tile_label="JP",
    ),
    "UA": dict(
        name="Ukraine",
        lon_min=22.0,  lon_max=40.5,
        lat_min=44.0,  lat_max=52.5,
        data_dir=os.path.join(DATA, "UA"),
        prefix="ua",
        tile_label="UA",
    ),
    "IR": dict(
        name="Iran",
        lon_min=44.0,  lon_max=64.0,
        lat_min=24.0,  lat_max=40.0,
        data_dir=DATA_IR,
        prefix="ir",
        tile_label="IR",
    ),
}

# Download 2019–2024 — matches SK pipeline (2019-2022 validation, 2023-2024 application)
ALL_YEARS  = list(range(2019, 2025))
ALL_MONTHS = list(range(1, 13))

SCALE_M     = 1000   # ~1 km
TILE_SIZE   = 0.5    # degrees per tile (≈ 3100 pixels, under GEE 5000-element limit)
MAX_WORKERS = 5
GEE_PROJECT = "ee-jianweiairuc"


# ── GEE helpers (identical to script 26) ────────────────────────────────────
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


TILE_TIMEOUT = 120  # seconds per tile before giving up


def fetch_tile(ee, image, lon0, lat0, lon1, lat1):
    import signal, platform
    tile_bbox = ee.Geometry.Rectangle([lon0, lat0, lon1, lat1])
    fc = image.sample(region=tile_bbox, scale=SCALE_M, geometries=True, dropNulls=False)
    # getInfo() can hang indefinitely on GEE errors; use concurrent timeout instead
    features = fc.getInfo()["features"]
    rows = []
    for feat in features:
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        row = {"lon": coords[0], "lat": coords[1]}
        row.update(props)
        rows.append(row)
    return rows


def image_to_df_tiled(ee, image, cfg, year, month):
    lon_edges = np.arange(cfg["lon_min"], cfg["lon_max"], TILE_SIZE)
    lat_edges = np.arange(cfg["lat_min"], cfg["lat_max"], TILE_SIZE)
    tiles = []
    for lon0 in lon_edges:
        lon1 = min(lon0 + TILE_SIZE, cfg["lon_max"])
        for lat0 in lat_edges:
            lat1 = min(lat0 + TILE_SIZE, cfg["lat_max"])
            tiles.append((lon0, lat0, lon1, lat1))

    all_rows, errors = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        fut_map = {exe.submit(fetch_tile, ee, image, lo, la, lo1, la1): (lo, la, lo1, la1)
                   for lo, la, lo1, la1 in tiles}
        for fut in as_completed(fut_map, timeout=TILE_TIMEOUT * len(tiles)):
            try:
                all_rows.extend(fut.result(timeout=TILE_TIMEOUT))
            except Exception as exc:
                errors.append((fut_map[fut], str(exc)))

    if errors:
        print(f"  WARNING: {len(errors)} tiles failed")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["year"] = year
        df["month"] = month
    return df


def download_country(ee, cc, cfg):
    os.makedirs(cfg["data_dir"], exist_ok=True)
    full_bbox = ee.Geometry.Rectangle(
        [cfg["lon_min"], cfg["lat_min"], cfg["lon_max"], cfg["lat_max"]])
    total = len(ALL_YEARS) * len(ALL_MONTHS)
    done = 0

    for year, month in itertools.product(ALL_YEARS, ALL_MONTHS):
        out_path = os.path.join(cfg["data_dir"],
                                f"{cfg['prefix']}_{year}_{month:02d}.csv")
        if os.path.exists(out_path):
            n = sum(1 for _ in open(out_path)) - 1
            print(f"[{cc} {done+1}/{total}] SKIP {year}-{month:02d} ({n:,} rows exist)")
            done += 1
            continue

        print(f"[{cc} {done+1}/{total}] Downloading {year}-{month:02d} ...", flush=True)
        try:
            img = build_monthly_image(ee, year, month, full_bbox)
            df  = image_to_df_tiled(ee, img, cfg, year, month)
            if df.empty:
                print(f"  WARNING: no data for {year}-{month:02d}")
                done += 1
                continue

            df["grid_id"] = (df["lon"].round(4).astype(str) + "_"
                             + df["lat"].round(4).astype(str))
            df["tile"] = cfg["tile_label"]
            cols = ["grid_id", "year", "month", "lon", "lat", "tile",
                    "CO", "SO2", "NO2", "O3", "PM10", "NTL", "LST"]
            df = df[[c for c in cols if c in df.columns]]
            df.to_csv(out_path, index=False)
            print(f"  -> OK  {len(df):,} rows  {os.path.basename(out_path)}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
        done += 1

    # Merge into single file
    print(f"\nMerging {cc} CSVs ...")
    dfs = []
    for year in ALL_YEARS:
        for month in ALL_MONTHS:
            p = os.path.join(cfg["data_dir"], f"{cfg['prefix']}_{year}_{month:02d}.csv")
            if os.path.exists(p):
                dfs.append(pd.read_csv(p))
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        out = os.path.join(cfg["data_dir"],
                           f"{cfg['prefix']}_satellite_{ALL_YEARS[0]}_{ALL_YEARS[-1]}.csv")
        merged.to_csv(out, index=False)
        print(f"  -> {os.path.basename(out)}  ({len(merged):,} rows)")


def main():
    requested = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else list(COUNTRIES.keys())
    invalid   = [cc for cc in requested if cc not in COUNTRIES]
    if invalid:
        print(f"Unknown country codes: {invalid}. Valid: {list(COUNTRIES.keys())}")
        sys.exit(1)

    print(f"Downloading satellite data for: {requested}")
    ee = init_gee()
    for cc in requested:
        print(f"\n{'='*60}")
        print(f"  {COUNTRIES[cc]['name']} ({cc})")
        print(f"  bbox: lon [{COUNTRIES[cc]['lon_min']}, {COUNTRIES[cc]['lon_max']}]  "
              f"lat [{COUNTRIES[cc]['lat_min']}, {COUNTRIES[cc]['lat_max']}]")
        print(f"{'='*60}")
        download_country(ee, cc, COUNTRIES[cc])
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
