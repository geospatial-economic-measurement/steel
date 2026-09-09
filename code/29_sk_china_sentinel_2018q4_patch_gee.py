

import calendar, itertools, os, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, DATA_SK, REP_ROOT, need

BASE       = REP_ROOT
RE_DATA    = need(DATA_CONFIDENTIAL, "re_data.csv")
CN_OUT_DIR = DATA_PROCESSED
SK_OUT_DIR = DATA_SK
BUFFER_M   = 10_000
SCALE_M    = 1_000
PATCH_MONTHS = [(2018, 10), (2018, 11), (2018, 12)]
GEE_PROJECT  = "ee-jianweiairuc"

SK_LON_MIN, SK_LON_MAX = 126.0, 130.0
SK_LAT_MIN, SK_LAT_MAX = 33.5,  38.5
TILE_SIZE   = 0.5
MAX_WORKERS = 5

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
        return ee.ImageCollection(cid).filterDate(start,end).filterBounds(bbox).select(band).mean()
    def modis(cid, band, sf):
        return ee.ImageCollection(cid).filterDate(start,end).filterBounds(bbox).select(band).mean().multiply(sf)
    co   = s5p("COPERNICUS/S5P/OFFL/L3_CO",  "CO_column_number_density")
    no2  = s5p("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density")
    so2  = s5p("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density")
    o3   = s5p("COPERNICUS/S5P/OFFL/L3_O3",  "O3_column_number_density")
    pm10 = modis("MODIS/061/MCD19A2_GRANULES", "Optical_Depth_047",  0.001)
    pm25 = (ee.ImageCollection("ECMWF/CAMS/NRT")
              .filterDate(start,end).filterBounds(bbox)
              .select("particulate_matter_d_less_than_25_um_surface").mean().multiply(1e9))
    lstt = modis("MODIS/061/MOD11A1", "LST_Day_1km",   0.02)
    lsta = modis("MODIS/061/MOD11A1", "LST_Night_1km", 0.02)
    ntl  = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
              .filterDate(start,end).filterBounds(bbox).select("avg_rad").mean())
    return (co.rename("CO").addBands(no2.rename("NO2")).addBands(so2.rename("SO2"))
              .addBands(o3.rename("O3")).addBands(pm10.rename("PM10"))
              .addBands(pm25.rename("PM25")).addBands(lstt.rename("LSTT"))
              .addBands(lsta.rename("LSTA")).addBands(ntl.rename("NTL")))

# ── PART A: China plant buffers ───────────────────────────────────────────────
def download_china_month(ee, year, month, plants_fc, china_bbox):
    import ee as ee_lib
    out_path = os.path.join(CN_OUT_DIR, f"china_{year}_{month:02d}.csv")
    if os.path.exists(out_path):
        print(f"  China {year}-{month:02d}: SKIP (exists)")
        return
    print(f"  China {year}-{month:02d}: downloading...", flush=True)
    img  = build_monthly_image(ee_lib, year, month, china_bbox)
    result = img.reduceRegions(collection=plants_fc, reducer=ee_lib.Reducer.mean(), scale=SCALE_M)
    features = result.getInfo()["features"]
    rows = []
    for feat in features:
        p = feat["properties"]
        rows.append({"plant_id": p.get("plant_id"), "name_prod": p.get("name_prod"),
                     "lon": p.get("lon"), "lat": p.get("lat"),
                     "Plant_CO_MEAN":   p.get("CO"),   "Plant_NO2_MEAN":  p.get("NO2"),
                     "Plant_SO2_MEAN":  p.get("SO2"),  "Plant_O3_MEAN":   p.get("O3"),
                     "Plant_PM10_MEAN": p.get("PM10"), "Plant_PM2_5_MEAN":p.get("PM25"),
                     "Plant_LSTT_MEAN": p.get("LSTT"), "Plant_LSTA_MEAN": p.get("LSTA"),
                     "Plant_NTL_MEAN":  p.get("NTL"),  "year": year, "month": month})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  China {year}-{month:02d}: OK  {len(df)} plants")

# ── PART B: SK tiled grid ─────────────────────────────────────────────────────
def fetch_sk_tile(ee, image, lon0, lat0, lon1, lat1):
    tile_bbox = ee.Geometry.Rectangle([lon0, lat0, lon1, lat1])
    fc = image.sample(region=tile_bbox, scale=SCALE_M, geometries=True, dropNulls=False)
    rows = []
    for feat in fc.getInfo()["features"]:
        props = feat["properties"]; coords = feat["geometry"]["coordinates"]
        row = {"lon": coords[0], "lat": coords[1]}; row.update(props); rows.append(row)
    return rows

def download_sk_month(ee, year, month):
    import ee as ee_lib
    out_path = os.path.join(SK_OUT_DIR, f"sk_{year}_{month:02d}.csv")
    if os.path.exists(out_path):
        print(f"  SK    {year}-{month:02d}: SKIP (exists)")
        return
    print(f"  SK    {year}-{month:02d}: downloading...", flush=True)
    full_bbox = ee_lib.Geometry.Rectangle([SK_LON_MIN, SK_LAT_MIN, SK_LON_MAX, SK_LAT_MAX])
    img = build_monthly_image(ee_lib, year, month, full_bbox)

    lon_edges = np.arange(SK_LON_MIN, SK_LON_MAX, TILE_SIZE)
    lat_edges = np.arange(SK_LAT_MIN, SK_LAT_MAX, TILE_SIZE)
    tiles = [(lo, la, min(lo+TILE_SIZE, SK_LON_MAX), min(la+TILE_SIZE, SK_LAT_MAX))
             for lo in lon_edges for la in lat_edges]

    all_rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_sk_tile, ee_lib, img, *t): t for t in tiles}
        for fut in as_completed(futures):
            try:
                all_rows.extend(fut.result())
            except Exception as e:
                print(f"    tile error: {e}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print(f"  SK    {year}-{month:02d}: WARNING empty"); return
    df["year"] = year; df["month"] = month
    df["grid_id"] = df["lon"].round(4).astype(str) + "_" + df["lat"].round(4).astype(str)
    df["tile"] = "SK"
    # Rename to match existing SK schema
    df = df.rename(columns={"CO":"CO","NO2":"NO2","SO2":"SO2","O3":"O3",
                             "PM10":"PM10","NTL":"NTL","LSTT":"LST","LSTA":"LSTA","PM25":"PM25"})
    cols = ["grid_id","year","month","lon","lat","tile","CO","SO2","NO2","O3","PM10","NTL","LST","PM25","LSTA"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out_path, index=False)
    print(f"  SK    {year}-{month:02d}: OK  {len(df):,} rows")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    import ee as ee_lib
    ee = init_gee()

    # China: build plant FeatureCollection
    plants = (pd.read_csv(RE_DATA, usecols=["plant_id","name_prod","Longitude","Latitude"])
                .drop_duplicates("plant_id").reset_index(drop=True))
    features = []
    for _, row in plants.iterrows():
        pt  = ee_lib.Geometry.Point([row["Longitude"], row["Latitude"]])
        buf = pt.buffer(BUFFER_M)
        features.append(ee_lib.Feature(buf, {
            "plant_id": int(row["plant_id"]), "name_prod": str(row["name_prod"]),
            "lon": float(row["Longitude"]),   "lat": float(row["Latitude"])}))
    plants_fc   = ee_lib.FeatureCollection(features)
    china_bbox  = ee_lib.Geometry.Rectangle([73.0, 18.0, 135.0, 53.0])

    print("=" * 50)
    print("Downloading 2018 Q4 — China (plant buffers)")
    print("=" * 50)
    for year, month in PATCH_MONTHS:
        try:
            download_china_month(ee, year, month, plants_fc, china_bbox)
        except Exception as e:
            print(f"  ERROR China {year}-{month:02d}: {e}")

    print()
    print("=" * 50)
    print("Downloading 2018 Q4 — South Korea (full grid)")
    print("=" * 50)
    for year, month in PATCH_MONTHS:
        try:
            download_sk_month(ee, year, month)
        except Exception as e:
            print(f"  ERROR SK {year}-{month:02d}: {e}")

    # ── Rebuild China merged file (2018 Q4 + 2019-2022) ──────────────────────
    print()
    print("Rebuilding china_plant_sentinel_2018q4_2022.csv ...")
    all_years  = [(2018,10),(2018,11),(2018,12)] + [(y,m) for y in range(2019,2023) for m in range(1,13)]
    dfs = []
    for y, m in all_years:
        p = os.path.join(CN_OUT_DIR, f"china_{y}_{m:02d}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        out = os.path.join(CN_OUT_DIR, "china_plant_sentinel_2018q4_2022.csv")
        merged.to_csv(out, index=False)
        print(f"  -> {out}  ({len(merged):,} rows)")

    print("\nDone. Run 30_sk_sentinel_model_train_apply.py with CHINA_SENT pointing to china_plant_sentinel_2018q4_2022.csv")

if __name__ == "__main__":
    main()
