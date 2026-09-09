
import os, sys, io, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, OUT_RL, REP_ROOT
warnings.filterwarnings('ignore')

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = REP_ROOT
PROC_DIR = os.path.join(BASE_DIR, 'Data', 'processed')
CONF_DIR = DATA_CONFIDENTIAL
RAW_DIR  = os.path.join(BASE_DIR, 'Data', 'raw')
OUT_DIR  = OUT_RL

TRAIN_MONTH = 11
D_MAX       = 10.0          # outer boundary for labeling
D_VALUES    = [1, 2, 3, 5, 10]  # thresholds to test
R_EARTH     = 6371.0
SEED        = 42
CHUNKSIZE   = 500_000
MIN_CELLS   = 200           # skip D if any class has fewer cells

FEATURES = ['CO_MEAN', 'NO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN', 'SO2_MEAN',
            'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN', 'O3_MEAN']

FEAT_FILES = [
    os.path.join(PROC_DIR, 'first_output_file.csv'),
    os.path.join(PROC_DIR, 'second_output_file.csv'),
]

CACHE = os.path.join(OUT_DIR, 'feat_labeled_D10.parquet')

# ── Load / build labeled feature dataset ──────────────────────────────────
if os.path.exists(CACHE):
    print(f'Loading cached labeled features from {os.path.basename(CACHE)}...')
    feat_df = pd.read_parquet(CACHE)
    print(f'  Loaded {len(feat_df):,} rows')
else:
    print('Building labeled feature dataset (D_MAX=10 km)...')

    # ── Plant locations (verbatim from 03_location_classifier_negative_control_umap.py) ─────────────────────
    coal_raw = pd.read_excel(
        os.path.join(RAW_DIR, 'China-Coal-Plant-Tracker-July-2025.xlsx'), sheet_name='Sheet1')
    sc = 'Status' if 'Status' in coal_raw.columns else \
         [c for c in coal_raw.columns if 'status' in c.lower()][0]
    lc = 'Latitude'  if 'Latitude'  in coal_raw.columns else \
         [c for c in coal_raw.columns if c.lower() == 'latitude'][0]
    oc = 'Longitude' if 'Longitude' in coal_raw.columns else \
         [c for c in coal_raw.columns if c.lower() == 'longitude'][0]
    coal = (coal_raw[coal_raw[sc].str.lower().str.strip() == 'operating'][[lc, oc]]
            .dropna().rename(columns={lc: 'Latitude', oc: 'Longitude'}).copy())

    cem_raw = pd.read_excel(
        os.path.join(RAW_DIR, 'Global-Cement-and-Concrete-Tracker_July-2025.xlsx'),
        sheet_name='Plant Data')
    cem_raw = cem_raw[
        (cem_raw['Country/Area'] == 'China') &
        (cem_raw['Operating status'].str.lower().str.strip() == 'operating')].copy()
    cs2 = cem_raw['Coordinates'].str.split(',', expand=True)
    cem_raw['Latitude']  = pd.to_numeric(cs2[0].str.strip(), errors='coerce')
    cem_raw['Longitude'] = pd.to_numeric(cs2[1].str.strip(), errors='coerce')
    cement = cem_raw[['Latitude', 'Longitude']].dropna().copy()

    loc = pd.read_excel(os.path.join(CONF_DIR, 'Location_Plants_Full.xlsx'))
    steel = (loc[loc['Capacityoperatingstatus'] == 'operating']
             [['latitude', 'longitude']].dropna().drop_duplicates().copy())
    steel.columns = ['Latitude', 'Longitude']
    print(f'  Steel: {len(steel):,}  Coal: {len(coal):,}  Cement: {len(cement):,}')

    # ── BallTrees ─────────────────────────────────────────────────────────
    def make_tree(df):
        return BallTree(np.radians(df[['Latitude', 'Longitude']].values.astype(float)),
                        metric='haversine')

    r_max = D_MAX / R_EARTH
    steel_tree  = make_tree(steel)
    coal_tree   = make_tree(coal)
    cement_tree = make_tree(cement)

    # ── Label gridinfo with ACTUAL minimum distance per plant type ────────
    print('Labeling grid cells and computing actual distances...')
    gridinfo = pd.read_csv(
        os.path.join(CONF_DIR, 'gridinfo_xy.csv'),
        usecols=['IDCode', 'Centroid_Lat', 'Centroid_Long'])
    pts = np.radians(gridinfo[['Centroid_Lat', 'Centroid_Long']].values.astype(float))

    # dist_* = actual haversine distance in km to nearest plant of each type
    dist_steel_raw,  _ = steel_tree.query(pts,  k=1)
    dist_coal_raw,   _ = coal_tree.query(pts,   k=1)
    dist_cement_raw, _ = cement_tree.query(pts, k=1)

    dist_steel  = dist_steel_raw.ravel()  * R_EARTH
    dist_coal   = dist_coal_raw.ravel()   * R_EARTH
    dist_cement = dist_cement_raw.ravel() * R_EARTH

    near_steel  = dist_steel  <= D_MAX
    near_coal   = dist_coal   <= D_MAX
    near_cement = dist_cement <= D_MAX
    n_types = near_steel.astype(int) + near_coal.astype(int) + near_cement.astype(int)

    # Assign label = type of nearest plant if that distance <= D_MAX AND only one type nearby
    label    = np.full(len(gridinfo), '', dtype=object)
    min_dist = np.full(len(gridinfo), np.nan)

    mask_steel  = near_steel  & (n_types == 1)
    mask_coal   = near_coal   & (n_types == 1)
    mask_cement = near_cement & (n_types == 1)

    label[mask_steel]  = 'steel'
    label[mask_coal]   = 'coal'
    label[mask_cement] = 'cement'
    min_dist[mask_steel]  = dist_steel[mask_steel]
    min_dist[mask_coal]   = dist_coal[mask_coal]
    min_dist[mask_cement] = dist_cement[mask_cement]

    gridinfo['label']    = label
    gridinfo['min_dist'] = min_dist
    labeled  = gridinfo[gridinfo['label'] != ''].copy()
    print(f'  Labeled (D<=10 km, unambiguous): {len(labeled):,}')
    print(f'  {labeled.groupby("label").size().to_dict()}')
    labeled_ids = set(labeled['IDCode'])

    # ── Load November features for labeled cells ──────────────────────────
    print(f'\nLoading November features from feature files...')
    parts = []
    for fpath in FEAT_FILES:
        if not os.path.exists(fpath):
            print(f'  [skip] {os.path.basename(fpath)} not found')
            continue
        n_chunks = 0
        t0 = time.time()
        for chunk in pd.read_csv(fpath, chunksize=CHUNKSIZE,
                                  usecols=['IDCode', 'Year', 'Month'] + FEATURES):
            sub = chunk[(chunk['Month'] == TRAIN_MONTH) &
                        (chunk['IDCode'].isin(labeled_ids))]
            if len(sub):
                parts.append(sub)
            n_chunks += 1
            if n_chunks % 100 == 0:
                print(f'  {os.path.basename(fpath)}: chunk {n_chunks}, '
                      f'{(time.time()-t0)/60:.1f} min')
        print(f'  {os.path.basename(fpath)}: done ({n_chunks} chunks, '
              f'{(time.time()-t0)/60:.1f} min)')

    feat_df = pd.concat(parts, ignore_index=True)
    id_to_label = dict(zip(labeled['IDCode'], labeled['label']))
    id_to_dist  = dict(zip(labeled['IDCode'], labeled['min_dist']))
    feat_df['label']    = feat_df['IDCode'].map(id_to_label)
    feat_df['min_dist'] = feat_df['IDCode'].map(id_to_dist)
    feat_df = feat_df.dropna(subset=FEATURES + ['label', 'min_dist'])

    # Save cache
    feat_df.to_parquet(CACHE, index=False)
    print(f'\nCached to {os.path.basename(CACHE)}: {len(feat_df):,} rows')

print(f'\nFull dataset: {len(feat_df):,} rows')
print(feat_df.groupby('label').size().to_string())
print(f'min_dist stats: mean={feat_df["min_dist"].mean():.2f} km, '
      f'median={feat_df["min_dist"].median():.2f} km')

# ── Sensitivity loop over D values ─────────────────────────────────────────
print('\n' + '=' * 68)
print('SENSITIVITY ANALYSIS: XGBoost metrics by distance threshold D')
print('=' * 68)

label_enc = {'steel': 0, 'coal': 1, 'cement': 2}
label_names = ['steel', 'coal', 'cement']

rows = []
for D in D_VALUES:
    sub = feat_df[feat_df['min_dist'] <= D].copy()
    counts = sub.groupby('label').size().to_dict()
    n_steel  = counts.get('steel', 0)
    n_coal   = counts.get('coal', 0)
    n_cement = counts.get('cement', 0)
    n_total  = len(sub)

    print(f'\nD = {D} km: N={n_total:,}  '
          f'(steel={n_steel:,}, coal={n_coal:,}, cement={n_cement:,})')

    if min(n_steel, n_coal, n_cement) < MIN_CELLS:
        print(f'  [SKIP] Fewer than {MIN_CELLS} cells in one class — skipping.')
        rows.append({'D_km': D, 'N_total': n_total, 'N_steel': n_steel,
                     'N_coal': n_coal, 'N_cement': n_cement,
                     'accuracy': np.nan, 'steel_precision': np.nan,
                     'steel_recall': np.nan, 'steel_f1': np.nan,
                     'coal_precision': np.nan, 'cement_precision': np.nan})
        continue

    X = StandardScaler().fit_transform(sub[FEATURES].values)
    y = np.array([label_enc[l] for l in sub['label'].values])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=SEED, verbosity=0, num_class=3,
        objective='multi:softmax')
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    rep = classification_report(y_te, y_pred, target_names=label_names,
                                 output_dict=True)
    acc = (y_pred == y_te).mean() * 100
    print(f'  Accuracy:         {acc:.1f}%  (chance=33.3%)')
    print(f'  Steel:   prec={rep["steel"]["precision"]:.3f}  '
          f'rec={rep["steel"]["recall"]:.3f}  F1={rep["steel"]["f1-score"]:.3f}')
    print(f'  Coal:    prec={rep["coal"]["precision"]:.3f}  '
          f'rec={rep["coal"]["recall"]:.3f}')
    print(f'  Cement:  prec={rep["cement"]["precision"]:.3f}  '
          f'rec={rep["cement"]["recall"]:.3f}')

    rows.append({
        'D_km':             D,
        'N_total':          n_total,
        'N_steel':          n_steel,
        'N_coal':           n_coal,
        'N_cement':         n_cement,
        'accuracy':         round(acc, 2),
        'steel_precision':  round(rep['steel']['precision'], 4),
        'steel_recall':     round(rep['steel']['recall'],    4),
        'steel_f1':         round(rep['steel']['f1-score'],  4),
        'coal_precision':   round(rep['coal']['precision'],  4),
        'cement_precision': round(rep['cement']['precision'], 4),
    })

results = pd.DataFrame(rows)
out_csv = os.path.join(OUT_DIR, 'table_sensitivity_D.csv')
results.to_csv(out_csv, index=False)
print(f'\nSaved: table_sensitivity_D.csv')
print(results.to_string(index=False))

# ── Figure ─────────────────────────────────────────────────────────────────
valid = results.dropna(subset=['accuracy'])
if len(valid) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    # Left: accuracy
    axes[0].plot(valid['D_km'], valid['accuracy'], 'o-', color='#1f4e79', lw=2, ms=7,
                 label='Overall accuracy')
    axes[0].axhline(33.3, color='gray', ls='--', lw=1, label='Chance (33.3%)')
    axes[0].set_xlabel('Distance threshold D (km)', fontsize=10)
    axes[0].set_ylabel('3-class accuracy (%)', fontsize=10)
    axes[0].set_title('(A) Overall Accuracy vs. D', fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].set_xticks(D_VALUES)
    axes[0].set_ylim(0, 100)

    # Right: per-class metrics for steel
    axes[1].plot(valid['D_km'], valid['steel_precision'] * 100, 's-',
                 color='#1f4e79', lw=2, ms=7, label='Steel precision')
    axes[1].plot(valid['D_km'], valid['steel_recall'] * 100, '^--',
                 color='#c00000', lw=2, ms=7, label='Steel recall')
    axes[1].plot(valid['D_km'], valid['coal_precision'] * 100, 's:',
                 color='#808080', lw=1.5, ms=5, label='Coal precision')
    axes[1].plot(valid['D_km'], valid['cement_precision'] * 100, '^:',
                 color='#ed7d31', lw=1.5, ms=5, label='Cement precision')
    axes[1].set_xlabel('Distance threshold D (km)', fontsize=10)
    axes[1].set_ylabel('Metric (%)', fontsize=10)
    axes[1].set_title('(B) Per-Class Metrics vs. D', fontsize=10, fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].set_xticks(D_VALUES)
    axes[1].set_ylim(0, 100)

    # Secondary x-axis: N_steel
    ax2 = axes[0].twiny()
    ax2.set_xlim(axes[0].get_xlim())
    ax2.set_xticks(valid['D_km'].tolist())
    ax2.set_xticklabels([f'{int(n/1000)}k' for n in valid['N_steel']], fontsize=7)
    ax2.set_xlabel('N steel cells (thousands)', fontsize=8, color='gray')

    fig.suptitle('Sensitivity: 3-Class Plant-Type Classification vs. Distance Threshold',
                 fontsize=10, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_sensitivity_D.pdf')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f'Saved: fig_sensitivity_D.pdf')

print('\nDone.')
