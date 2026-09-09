
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import os
from _paths import DATA_PROCESSED, OUT_FIG

LAYERS = os.path.join(DATA_PROCESSED, 'fig1a_layers')
OUT1 = os.path.join(OUT_FIG, "ESRI_prediction_holdout_v2.png")
OUT2 = OUT1

EXTENT = [72, 136, 14, 55]  # lon_min, lon_max, lat_min, lat_max

print("Loading layers ...")
flagged = pd.read_csv(LAYERS + r"\predicted_cells_nov.csv")
train_p = pd.read_csv(LAYERS + r"\training_plants.csv")
hold_p  = pd.read_csv(LAYERS + r"\holdout_plants.csv")
print(f"  flagged cells: {len(flagged):,} | training: {len(train_p)} | holdout: {len(hold_p)}")

proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(14, 10), dpi=200)
ax = plt.axes(projection=proj)
ax.set_extent(EXTENT, crs=proj)

# ── Basemap: Esri World Imagery tiles (same source as original); fallback stock_img
try:
    from cartopy.io.img_tiles import GoogleTiles

    class EsriImagery(GoogleTiles):
        def _image_url(self, tile):
            x, y, z = tile
            return ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                    f"World_Imagery/MapServer/tile/{z}/{y}/{x}.jpg")

    ax.add_image(EsriImagery(cache=True), 5, interpolation="spline36")
    print("  Esri World Imagery basemap OK")
except Exception as e:
    print(f"  Tile fetch failed ({e}); using stock_img fallback")
    ax.stock_img()

# ── Heat layer of predicted cells (P >= 0.5), smoothed like the original
BIN = 0.15  # degrees
lon_edges = np.arange(EXTENT[0], EXTENT[1] + BIN, BIN)
lat_edges = np.arange(EXTENT[2], EXTENT[3] + BIN, BIN)
H, _, _ = np.histogram2d(flagged["lon"], flagged["lat"], bins=[lon_edges, lat_edges])
Hs = gaussian_filter(H, sigma=1.6)
Hs = np.ma.masked_less(Hs, 0.05)
# purple -> cyan -> green ramp, transparent at the low end (matches original palette)
cmap = LinearSegmentedColormap.from_list(
    "pred", [(0.0, (0.45, 0.15, 0.75, 0.00)),
             (0.15, (0.45, 0.15, 0.75, 0.55)),
             (0.55, (0.10, 0.75, 0.80, 0.75)),
             (1.0, (0.30, 0.95, 0.35, 0.90))])
LON, LAT = np.meshgrid(lon_edges[:-1] + BIN / 2, lat_edges[:-1] + BIN / 2)
ax.pcolormesh(LON, LAT, np.power(Hs.T, 0.4), cmap=cmap, shading="auto",
              transform=proj, zorder=3)

# ── Plant markers
ax.scatter(train_p["Longitude"], train_p["Latitude"], s=60, c="#1440d6",
           marker="o", edgecolors="white", linewidths=0.6, transform=proj,
           zorder=5, label="Training-Test Sample")
ax.scatter(hold_p["Longitude"], hold_p["Latitude"], s=55, c="#e01010",
           marker="^", edgecolors="white", linewidths=0.6, transform=proj,
           zorder=5, label="Hold-out Sample")

# ── Legend styled like the original "Map Legend" box
handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#1440d6",
           markeredgecolor="white", markersize=10, label="Training-Test Sample"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor="#e01010",
           markeredgecolor="white", markersize=10, label="Hold-out Sample"),
    Patch(facecolor=(0.45, 0.15, 0.75, 0.8), label="Predicted Locations"),
]
leg = ax.legend(handles=handles, loc="lower left", fontsize=11,
                title="Map Legend", title_fontsize=12, framealpha=0.95,
                edgecolor="0.4", facecolor="white")
leg.get_title().set_fontweight("bold")
leg._legend_box.align = "left"

ax.set_axis_off()
plt.tight_layout(pad=0.2)
for out in (OUT1, OUT2):
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {out}")
plt.close()
print("Done.")
