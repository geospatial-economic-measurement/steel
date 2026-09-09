

import calendar
import itertools
import os

import numpy as np
import pandas as pd
from _paths import DATA_CONFIDENTIAL, DATA_PROCESSED, REP_ROOT, need

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE     = REP_ROOT
RE_DATA  = need(DATA_CONFIDENTIAL, "re_data.csv")
OUT_DIR  = DATA_PROCESSED
BUFFER_M = 10_000      # 10 km buffer radius
SCALE_M  = 1_000       # 1 km resolution for reduceRegions
YEARS    = list(range(2019, 2023))   # 2019-2022 training period
MONTHS   = list(range(1, 13))
GEE_PROJECT = "ee-jianweiairuc"

# --------------------------------------------------------------------------- #
# GEE initialisation
# --------------------------------------------------------------------------- #
def init_gee():
    import ee
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
    return ee


# --------------------------------------------------------------------------- #
# Load plant coordinates from re_data.csv
# --------------------------------------------------------------------------- #
def load_plants():
    df = pd.read_csv(RE_DATA, usecols=["name_prod", "plant_id", "Longitude", "Latitude"])
    plants = (df.drop_duplicates(subset="plant_id")
                .reset_index(drop=True)
                [["plant_id", "name_prod", "Longitude", "Latitude"]])
    print(f"Loaded {len(plants)} unique China plants from re_data.csv")
    return plants


# --------------------------------------------------------------------------- #
# Build monthly 9-band image
# --------------------------------------------------------------------------- #
def month_date_range(year, month):
    last = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"


def build_monthly_image(ee, year, month, bbox):
    start, end = month_date_range(year, month)

    def s5p(cid, band):
        return (ee.ImageCollection(cid)
                  .filterDate(start, end)
                  .filterBounds(bbox)
                  .select(band)
                  .mean())

    def modis(cid, band, sf):
        return (ee.ImageCollection(cid)
                  .filterDate(start, end)
                  .filterBounds(bbox)
                  .select(band)
                  .mean()
                  .multiply(sf))

    co   = s5p("COPERNICUS/S5P/OFFL/L3_CO",  "CO_column_number_density")
    no2  = s5p("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density")
    so2  = s5p("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density")
    o3   = s5p("COPERNICUS/S5P/OFFL/L3_O3",  "O3_column_number_density")
    pm10 = modis("MODIS/061/MCD19A2_GRANULES", "Optical_Depth_047",  0.001)
    pm25 = (ee.ImageCollection("ECMWF/CAMS/NRT")
              .filterDate(start, end)
              .filterBounds(bbox)
              .select("particulate_matter_d_less_than_25_um_surface")
              .mean()
              .multiply(1e9))   # kg/m3 → µg/m3
    lstt = modis("MODIS/061/MOD11A1", "LST_Day_1km",   0.02)
    lsta = modis("MODIS/061/MOD11A1", "LST_Night_1km", 0.02)
    ntl  = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
              .filterDate(start, end)
              .filterBounds(bbox)
              .select("avg_rad")
              .mean())

    return (co.rename("CO")
              .addBands(no2.rename("NO2"))
              .addBands(so2.rename("SO2"))
              .addBands(o3.rename("O3"))
              .addBands(pm10.rename("PM10"))
              .addBands(pm25.rename("PM25"))
              .addBands(lstt.rename("LSTT"))
              .addBands(lsta.rename("LSTA"))
              .addBands(ntl.rename("NTL")))


# --------------------------------------------------------------------------- #
# Extract per-plant buffer means via reduceRegions
# --------------------------------------------------------------------------- #
def extract_plant_means(ee, image, plants_fc):
    """
    Run image.reduceRegions over 10 km plant buffers.
    Returns list of dicts with plant_id + band means.
    """
    result = image.reduceRegions(
        collection=plants_fc,
        reducer=ee.Reducer.mean(),
        scale=SCALE_M,
    )
    features = result.getInfo()["features"]
    rows = []
    for feat in features:
        props = feat["properties"]
        rows.append({
            "plant_id":  props.get("plant_id"),
            "name_prod": props.get("name_prod"),
            "lon":       props.get("lon"),
            "lat":       props.get("lat"),
            "CO":        props.get("CO"),
            "NO2":       props.get("NO2"),
            "SO2":       props.get("SO2"),
            "O3":        props.get("O3"),
            "PM10":      props.get("PM10"),
            "PM25":      props.get("PM25"),
            "LSTT":      props.get("LSTT"),
            "LSTA":      props.get("LSTA"),
            "NTL":       props.get("NTL"),
        })
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ee      = init_gee()
    plants  = load_plants()

    # Build GEE FeatureCollection of 10 km buffers around each plant
    features = []
    for _, row in plants.iterrows():
        pt = ee.Geometry.Point([row["Longitude"], row["Latitude"]])
        buf = pt.buffer(BUFFER_M)
        feat = ee.Feature(buf, {
            "plant_id":  int(row["plant_id"]),
            "name_prod": str(row["name_prod"]),
            "lon":       float(row["Longitude"]),
            "lat":       float(row["Latitude"]),
        })
        features.append(feat)
    plants_fc = ee.FeatureCollection(features)

    # China bounding box for image filtering (server-side, fine at this extent)
    china_bbox = ee.Geometry.Rectangle([73.0, 18.0, 135.0, 53.0])

    total = len(YEARS) * len(MONTHS)
    done  = 0

    for year, month in itertools.product(YEARS, MONTHS):
        out_path = os.path.join(OUT_DIR, f"china_{year}_{month:02d}.csv")
        if os.path.exists(out_path):
            n = sum(1 for _ in open(out_path)) - 1
            print(f"[{done+1}/{total}] SKIP {year}-{month:02d} (exists, {n} rows)")
            done += 1
            continue

        print(f"[{done+1}/{total}] Downloading {year}-{month:02d} ...", flush=True)
        try:
            img  = build_monthly_image(ee, year, month, china_bbox)
            rows = extract_plant_means(ee, img, plants_fc)

            df = pd.DataFrame(rows)
            df["year"]  = year
            df["month"] = month

            # Rename to match re_data Plant_*_MEAN convention
            df = df.rename(columns={
                "CO":   "Plant_CO_MEAN",
                "NO2":  "Plant_NO2_MEAN",
                "SO2":  "Plant_SO2_MEAN",
                "O3":   "Plant_O3_MEAN",
                "PM10": "Plant_PM10_MEAN",
                "PM25": "Plant_PM2_5_MEAN",
                "LSTT": "Plant_LSTT_MEAN",
                "LSTA": "Plant_LSTA_MEAN",
                "NTL":  "Plant_NTL_MEAN",
            })

            col_order = ["plant_id", "name_prod", "lon", "lat", "year", "month",
                         "Plant_CO_MEAN", "Plant_NO2_MEAN", "Plant_SO2_MEAN",
                         "Plant_O3_MEAN", "Plant_PM10_MEAN", "Plant_PM2_5_MEAN",
                         "Plant_LSTT_MEAN", "Plant_LSTA_MEAN", "Plant_NTL_MEAN"]
            df = df[[c for c in col_order if c in df.columns]]
            df.to_csv(out_path, index=False)
            print(f"  -> OK  {len(df)} plants  saved: {os.path.basename(out_path)}")

        except Exception as exc:
            print(f"  ERROR: {exc}")

        done += 1

    # ── Merge into single panel ──────────────────────────────────────────────
    print("\nMerging into china_plant_sentinel_2019_2022.csv ...")
    dfs = []
    for year, month in itertools.product(YEARS, MONTHS):
        p = os.path.join(OUT_DIR, f"china_{year}_{month:02d}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        out_merged = os.path.join(OUT_DIR, "china_plant_sentinel_2019_2022.csv")
        merged.to_csv(out_merged, index=False)
        print(f"  -> {out_merged}  ({len(merged):,} rows, "
              f"{merged['plant_id'].nunique()} plants × "
              f"{merged[['year','month']].drop_duplicates().__len__()} months)")
    else:
        print("  WARNING: no monthly files found to merge.")

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
