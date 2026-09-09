
import warnings, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter
import os
from _paths import CODEBOOKS, DATA_SK, OUT_FIG, OUT_RL, OUT_TAB, REP_ROOT, need
warnings.filterwarnings("ignore")

BASE      = REP_ROOT
GEM_PATH  = need(CODEBOOKS, "sk_gem_plants.csv")
GRID_1922 = need(DATA_SK, "sk_outputs", "sk_predictions_grid_2019_2022.csv")
GRID_2324 = need(DATA_SK, "sk_outputs", "sk_predictions_grid_2023_2024.csv")
SENT_MON  = need(DATA_SK, "sk_outputs", "sk_plant_sentinel_monthly.csv")
OUT_PATH  = os.path.join(OUT_TAB, "intermediate", "sk_plant_predictions_combined.xlsx")
BUFFER_KM = 10.0
WSA_SK    = {2019:71.42, 2020:67.14, 2021:70.42, 2022:65.85, 2023:66.68, 2024:63.61}

def hav(lat1, lon1, lat2, lon2):
    R = 6371.
    dlat = np.radians(lat2-lat1); dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a,0,1)))

def yoy(v0, v1):
    return round((v1-v0)/v0*100, 1) if v0 and v0 > 0 else None

gem = pd.read_csv(GEM_PATH)
gem = gem[gem["status"] == "operating"].reset_index(drop=True)
plant_lats  = gem["lat"].values
plant_lons  = gem["lon"].values
plant_names = gem["plant_name"].values
n_plants    = len(gem)
lat_margin  = BUFFER_KM / 111.0
lon_margin  = BUFFER_KM / (111.0 * np.cos(np.radians(36.5)))

# ── METHOD 1: Grid aggregation ───────────────────────────────────────────────
print("Building Method 1 (Grid aggregation)...")
acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
for fpath in [GRID_1922, GRID_2324]:
    for chunk in pd.read_csv(fpath, chunksize=500_000,
                             usecols=["Year","Month","longitude","latitude","Steel_Pred_total"]):
        chunk = chunk[~((chunk["Year"]==2019) & (chunk["Month"]<=3))]
        if len(chunk) == 0: continue
        lats = chunk["latitude"].values; lons = chunk["longitude"].values
        preds = chunk["Steel_Pred_total"].values
        years = chunk["Year"].values; months = chunk["Month"].values
        for pi in range(n_plants):
            plat, plon = plant_lats[pi], plant_lons[pi]
            mask = ((lats>=plat-lat_margin) & (lats<=plat+lat_margin) &
                    (lons>=plon-lon_margin) & (lons<=plon+lon_margin))
            if not mask.any(): continue
            idx = np.where(mask)[0]
            dists = hav(lats[idx], lons[idx], plat, plon)
            in_buf = idx[dists <= BUFFER_KM]
            for i in in_buf:
                ym = (int(years[i]), int(months[i]))
                acc[pi][ym][0] += float(preds[i]); acc[pi][ym][1] += 1

grid_annual = {}
for pi in range(n_plants):
    yr_totals = defaultdict(float)
    for (yr, mo), (s, _) in acc[pi].items():
        yr_totals[yr] += s
    if 2019 in yr_totals:
        yr_totals[2019] *= 12/9
    grid_annual[plant_names[pi]] = dict(yr_totals)

# ── METHOD 2: Sentinel M1 ────────────────────────────────────────────────────
print("Building Method 2 (Sentinel M1)...")
sent_mon = pd.read_csv(SENT_MON)
sent_annual_df = sent_mon.groupby(["plant_name","Year"])["pred_kt"].sum().reset_index()
sent_annual_df.columns = ["plant_name","Year","val"]
sent_annual = {pname: dict(zip(grp["Year"], grp["val"]))
               for pname, grp in sent_annual_df.groupby("plant_name")}

# ── Shared structure helpers ─────────────────────────────────────────────────
yrs_grid = sorted({y for d in grid_annual.values() for y in d})
yrs_sent = sorted(sent_annual_df["Year"].unique())
trans_grid = [f"{yrs_grid[i]}->{yrs_grid[i+1]}" for i in range(len(yrs_grid)-1)]
trans_sent = [f"{yrs_sent[i]}->{yrs_sent[i+1]}" for i in range(len(yrs_sent)-1)]

nat_grid = {yr: sum(grid_annual[p].get(yr,0) for p in plant_names) for yr in yrs_grid}
nat_sent = {yr: sum(sent_annual.get(p,{}).get(yr,0) for p in plant_names) for yr in yrs_sent}

def build_level_df(annual_dict, yrs, plant_names, gem):
    rows = []
    for pi, pname in enumerate(plant_names):
        row = {"Plant Name": pname,
               "Latitude":        round(plant_lats[pi], 4),
               "Longitude":       round(plant_lons[pi], 4),
               "Capacity (ttpa)": int(gem.iloc[pi]["nom_capacity_ttpa"]),
               "Technology":      gem.iloc[pi]["technology"].split(";")[0].strip()}
        for yr in yrs:
            row[yr] = round(annual_dict[pname].get(yr, np.nan), 1)
        rows.append(row)
    nat_row = {"Plant Name":"NATIONAL (model, 10k tonnes)","Latitude":"","Longitude":"",
               "Capacity (ttpa)":"","Technology":""}
    wsa_row = {"Plant Name":"WSA actual (million tonnes)","Latitude":"","Longitude":"",
               "Capacity (ttpa)":"","Technology":""}
    nat = {yr: sum(annual_dict[p].get(yr,0) for p in plant_names) for yr in yrs}
    for yr in yrs:
        nat_row[yr] = round(nat.get(yr, np.nan), 1)
        wsa_row[yr] = WSA_SK.get(yr, "")
    return pd.concat([pd.DataFrame(rows),
                      pd.DataFrame([nat_row]),
                      pd.DataFrame([wsa_row])], ignore_index=True)

def build_gr_df(annual_dict, transitions, plant_names, gem):
    nat = {yr: sum(annual_dict[p].get(yr,0) for p in plant_names)
           for yr in {y for d in annual_dict.values() for y in d}}
    rows = []
    for pi, pname in enumerate(plant_names):
        row = {"Plant Name": pname,
               "Latitude":        round(plant_lats[pi], 4),
               "Longitude":       round(plant_lons[pi], 4),
               "Capacity (ttpa)": int(gem.iloc[pi]["nom_capacity_ttpa"]),
               "Technology":      gem.iloc[pi]["technology"].split(";")[0].strip()}
        for t in transitions:
            y0,y1 = int(t.split("->")[0]), int(t.split("->")[1])
            row[t] = yoy(annual_dict[pname].get(y0,0), annual_dict[pname].get(y1,0))
        rows.append(row)
    nat_row = {"Plant Name":"NATIONAL (model)","Latitude":"","Longitude":"",
               "Capacity (ttpa)":"","Technology":""}
    wsa_row = {"Plant Name":"WSA actual","Latitude":"","Longitude":"",
               "Capacity (ttpa)":"","Technology":""}
    for t in transitions:
        y0,y1 = int(t.split("->")[0]), int(t.split("->")[1])
        nat_row[t] = yoy(nat.get(y0), nat.get(y1))
        wsa_row[t]  = yoy(WSA_SK.get(y0), WSA_SK.get(y1))
    return pd.concat([pd.DataFrame(rows),
                      pd.DataFrame([nat_row]),
                      pd.DataFrame([wsa_row])], ignore_index=True)

grid_level_df = build_level_df(grid_annual, yrs_grid, plant_names, gem)
grid_gr_df    = build_gr_df(grid_annual, trans_grid, plant_names, gem)
sent_annual_kt = {p: {yr: v * 10 for yr, v in d.items()} for p, d in sent_annual.items()}
sent_level_df = build_level_df(sent_annual_kt, yrs_sent, plant_names, gem)
sent_gr_df    = build_gr_df(sent_annual, trans_sent, plant_names, gem)

# ── Cross-method comparison sheet ───────────────────────────────────────────
shared_trans = [t for t in trans_sent if t in trans_grid]
wsa_vals = {t: yoy(WSA_SK.get(int(t.split("->")[0])), WSA_SK.get(int(t.split("->")[1])))
            for t in shared_trans}
comp_rows = []
for t in shared_trans:
    y0,y1 = int(t.split("->")[0]), int(t.split("->")[1])
    gv = yoy(nat_grid.get(y0), nat_grid.get(y1))
    sv = yoy(nat_sent.get(y0), nat_sent.get(y1))
    wv = wsa_vals[t]
    comp_rows.append({"Transition": t,
                      "Grid agg (YoY%)":     gv,
                      "Sentinel M1 (YoY%)":  sv,
                      "WSA actual (YoY%)":   wv,
                      "Grid error (pp)":     round(gv-wv,1) if gv and wv else None,
                      "Sentinel error (pp)": round(sv-wv,1) if sv and wv else None})
comp_df = pd.DataFrame(comp_rows)

# Pearson r
gv_list = [r["Grid agg (YoY%)"]    for r in comp_rows if r["WSA actual (YoY%)"] is not None]
sv_list = [r["Sentinel M1 (YoY%)"] for r in comp_rows if r["WSA actual (YoY%)"] is not None]
wv_list = [r["WSA actual (YoY%)"]  for r in comp_rows if r["WSA actual (YoY%)"] is not None]
r_grid, _ = pearsonr(gv_list, wv_list)
r_sent, _ = pearsonr(sv_list, wv_list)
mae_grid = np.mean(np.abs(np.array(gv_list)-np.array(wv_list)))
mae_sent = np.mean(np.abs(np.array(sv_list)-np.array(wv_list)))
print(f"Grid agg:    Pearson r={r_grid:.3f}  MAE={mae_grid:.1f}pp")
print(f"Sentinel M1: Pearson r={r_sent:.3f}  MAE={mae_sent:.1f}pp")

# Add summary rows to comparison sheet
summary_rows = [
    {"Transition":"---","Grid agg (YoY%)":None,"Sentinel M1 (YoY%)":None,
     "WSA actual (YoY%)":None,"Grid error (pp)":None,"Sentinel error (pp)":None},
    {"Transition":"Pearson r","Grid agg (YoY%)":round(r_grid,3),"Sentinel M1 (YoY%)":round(r_sent,3),
     "WSA actual (YoY%)":None,"Grid error (pp)":None,"Sentinel error (pp)":None},
    {"Transition":"MAE (pp)","Grid agg (YoY%)":round(mae_grid,1),"Sentinel M1 (YoY%)":round(mae_sent,1),
     "WSA actual (YoY%)":None,"Grid error (pp)":None,"Sentinel error (pp)":None},
]
comp_df = pd.concat([comp_df, pd.DataFrame(summary_rows)], ignore_index=True)

# ── Excel formatting ─────────────────────────────────────────────────────────
header_fill  = PatternFill("solid", fgColor="1F4E79")
header_font  = Font(color="FFFFFF", bold=True)
nat_fill     = PatternFill("solid", fgColor="D6E4F0")
wsa_fill     = PatternFill("solid", fgColor="FCE4D6")
alt_fill     = PatternFill("solid", fgColor="F2F2F2")
pos_font     = Font(color="1F6B2E", bold=True)
neg_font     = Font(color="C00000", bold=True)
method1_fill = PatternFill("solid", fgColor="E8F4FD")  # light blue — grid
method2_fill = PatternFill("solid", fgColor="E8F9EF")  # light green — Sentinel

SHEETS = [
    ("Grid — Level (10k t)",      grid_level_df,  False, "grid"),
    ("Grid — Growth rates (YoY%)", grid_gr_df,    True,  "grid"),
    ("Sentinel — Level (thousand t)", sent_level_df, False, "sent"),
    ("Sentinel — Growth (YoY%)",  sent_gr_df,     True,  "sent"),
    ("Cross-method comparison",   comp_df,         True,  "comp"),
]

UNIT_NOTES = {
    "Grid — Level (10k t)":
        "Method 1 — Grid aggregation: Steel_Pred_total summed within 10 km buffer of each plant. "
        "Unit: 10,000 tonnes (10k t). 2019 = Apr–Dec observed × 12/9. "
        "NATIONAL = sum of 15 plants. WSA = actual SK production (million tonnes, different scale).",
    "Sentinel — Level (thousand t)":
        "Method 2 — Sentinel M1: XGBoost trained on China Sentinel-5P/MODIS/VIIRS signals, "
        "applied to SK 10 km plant buffers. Unit: thousands of tonnes (kt). "
        "Absolute levels not comparable to WSA (China BOF calibration ≠ SK EAF). "
        "Growth rates are the meaningful comparison.",
}

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    for sname, df, is_gr, method in SHEETS:
        has_note = sname in UNIT_NOTES
        startrow = 2 if has_note else 0
        df.to_excel(writer, sheet_name=sname, index=False, startrow=startrow)
        ws = writer.sheets[sname]

        if has_note:
            nc = ws.cell(1, 1, UNIT_NOTES[sname])
            nc.font      = Font(italic=True, color="595959", size=9)
            nc.alignment = Alignment(wrap_text=True)
            ws.row_dimensions[1].height = 36
            ws.merge_cells(start_row=1, start_column=1,
                           end_row=1, end_column=max(ws.max_column, 8))

        hdr_row  = startrow + 1
        data_row = hdr_row + 1

        # Method colour stripe on header
        mfill = method1_fill if method == "grid" else (method2_fill if method == "sent" else header_fill)

        for cell in ws[hdr_row]:
            cell.fill = header_fill; cell.font = header_font
            cell.alignment = Alignment(horizontal="center")

        for ri, row in enumerate(ws.iter_rows(min_row=data_row), start=data_row):
            pname = str(ws.cell(ri, 1).value or "")
            is_nat = "NATIONAL" in pname
            is_wsa = "WSA" in pname and "actual" in pname.lower()
            is_sep = pname == "---"
            for cell in row:
                cell.alignment = Alignment(horizontal="center")
                if is_nat:
                    cell.fill = nat_fill; cell.font = Font(bold=True)
                elif is_wsa:
                    cell.fill = wsa_fill; cell.font = Font(bold=True)
                elif is_sep:
                    pass
                elif ri % 2 == 0:
                    cell.fill = alt_fill
                if is_gr and cell.column > 1 and not is_nat and not is_wsa and not is_sep:
                    try:
                        v = float(cell.value)
                        cell.font = pos_font if v > 0 else (neg_font if v < 0 else Font())
                    except (TypeError, ValueError):
                        pass

        ws.column_dimensions["A"].width = 46
        for ci in range(2, ws.max_column + 1):
            cw = 11 if ci <= 5 else 15
            ws.column_dimensions[get_column_letter(ci)].width = cw
        ws.freeze_panes = f"B{data_row}"

print(f"\nExcel saved -> {OUT_PATH}")

# ── Two-panel validation figure (Sentinel M1 plant-level vs WSA) ─────────────
import os, matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

E_FIG = OUT_RL
os.makedirs(E_FIG, exist_ok=True)

# Build arrays for Sentinel M1 vs WSA national YoY
fig_trans, fig_sv, fig_wv = [], [], []
for t in trans_sent:
    y0, y1 = int(t.split("->")[0]), int(t.split("->")[1])
    if y0 in WSA_SK and y1 in WSA_SK:
        sv = yoy(nat_sent.get(y0), nat_sent.get(y1))
        wv_val = yoy(WSA_SK.get(y0), WSA_SK.get(y1))
        if sv is not None and wv_val is not None:
            fig_trans.append(f"{y0}→{y1}")
            fig_sv.append(sv)
            fig_wv.append(wv_val)

sv_arr = np.array(fig_sv)
wv_arr = np.array(fig_wv)
r_fig, _ = pearsonr(sv_arr, wv_arr)
mae_fig = np.mean(np.abs(sv_arr - wv_arr))

fig, (ax_bar, ax_scat) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

# Panel A: grouped bar chart
x = np.arange(len(fig_trans))
bw = 0.35
ax_bar.bar(x - bw/2, fig_sv, width=bw, color="#2166ac", label="Model prediction (plant-level)")
ax_bar.bar(x + bw/2, fig_wv, width=bw, color="#d73027", alpha=0.75, label="WSA actual")
ax_bar.axhline(0, color="black", linewidth=0.8)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(fig_trans, fontsize=9)
ax_bar.set_ylabel("Year-over-year growth rate (%)", fontsize=10)
ax_bar.legend(fontsize=9, framealpha=0.9)
ax_bar.set_title("South Korea: national steel output growth rates", fontsize=10, pad=8)

# Panel B: scatter with 45° line
lim = max(np.abs(sv_arr).max(), np.abs(wv_arr).max()) * 1.15
ax_scat.scatter(wv_arr, sv_arr, color="#2166ac", s=60, zorder=3)
ax_scat.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="45° line")
ax_scat.axhline(0, color="gray", lw=0.6, ls=":")
ax_scat.axvline(0, color="gray", lw=0.6, ls=":")
for t, wg, mg in zip(fig_trans, wv_arr, sv_arr):
    ax_scat.annotate(t, (wg, mg), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
ax_scat.set_xlim(-lim, lim)
ax_scat.set_ylim(-lim, lim)
ax_scat.set_xlabel("WSA reported growth rate (%)", fontsize=10)
ax_scat.set_ylabel("Model predicted growth rate (%)", fontsize=10)
ax_scat.set_title(
    f"Predicted vs. WSA growth rates\nPearson $r={r_fig:.3f}$, MAE$={mae_fig:.1f}$ pp",
    fontsize=10, pad=8
)
ax_scat.legend(fontsize=9, framealpha=0.9)

plt.tight_layout()
fig_out = os.path.join(E_FIG, "fig_sk_growth_rates_2panel_diagnostic.pdf")
plt.savefig(fig_out, bbox_inches="tight")
plt.savefig(fig_out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_out}")
