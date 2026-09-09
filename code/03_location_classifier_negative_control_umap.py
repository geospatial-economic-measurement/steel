"""
Plant-type clustering — Steel, Coal, Cement (R1.4 Extension)
========================================================================
Tests whether raw satellite signals in 9-dimensional feature space
naturally separate steel, coal, and cement plants without any labels.

Two analyses:
  Phase 1 — Unsupervised k-means clustering (K=3) + UMAP visualization
  Phase 2 — Supervised XGBoost 3-class validation

Grid cells used: within 10 km of exactly one plant type (November, 2019-2022 pooled).
Ambiguous cells (within 10 km of 2+ plant types) are excluded.
Exclusion adjustment: coal/cement cells within 10 km of steel are re-labeled steel.

Outputs:
  table_cluster_purity.csv           K-means purity (cluster x plant type)
  table_multiclass_accuracy.csv      XGBoost precision/recall/F1 per class
  fig_plant_clustering_umap.pdf      UMAP scatter colored by plant type + cluster
  fig_cluster_confusion.pdf          XGBoost confusion matrix
"""
import os, sys, io, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
from _paths import DATA_CONFIDENTIAL, OUT_RL, REP_ROOT
warnings.filterwarnings('ignore')

try:
    import umap
    HAS_UMAP = True
except ImportError:
    print('[WARN] umap-learn not installed; UMAP figure will be skipped')
    HAS_UMAP = False

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = REP_ROOT
PROC_DIR = os.path.join(BASE_DIR, 'Data', 'processed')
CONF_DIR = DATA_CONFIDENTIAL
RAW_DIR  = os.path.join(BASE_DIR, 'Data', 'raw')
OUT_DIR  = OUT_RL
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_MONTH = 11          # November
D_KM        = 10.0        # proximity threshold
R_EARTH     = 6371.0
SEED        = 42
CHUNKSIZE   = 500_000

FEATURES = ['CO_MEAN', 'NO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN', 'SO2_MEAN',
            'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN', 'O3_MEAN']

FEAT_FILES = [
    os.path.join(PROC_DIR, 'first_output_file.csv'),
    os.path.join(PROC_DIR, 'second_output_file.csv'),
]

LABEL_NAMES = ['steel', 'coal', 'cement']
COLORS      = {'steel': '#1f4e79', 'coal': '#c00000', 'cement': '#ed7d31'}
MARKERS     = {'steel': 'o', 'coal': 's', 'cement': '^'}

# ── Load plant locations (verbatim from 32_coverage_curve.py) ──────────────
print('Loading plant locations...')

# Coal
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
print(f'  Coal: {len(coal):,} plants')

# Cement
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
print(f'  Cement: {len(cement):,} plants')

# Steel
loc = pd.read_excel(os.path.join(CONF_DIR, 'Location_Plants_Full.xlsx'))
steel = (loc[loc['Capacityoperatingstatus'] == 'operating']
         [['latitude', 'longitude']].dropna().drop_duplicates().copy())
steel.columns = ['Latitude', 'Longitude']
print(f'  Steel: {len(steel):,} plants')

# ── Build BallTrees for each plant type ───────────────────────────────────
def make_tree(df):
    coords = np.radians(df[['Latitude', 'Longitude']].values.astype(float))
    return BallTree(coords, metric='haversine')

r_rad = D_KM / R_EARTH
steel_tree  = make_tree(steel)
coal_tree   = make_tree(coal)
cement_tree = make_tree(cement)

# ── Label each grid cell by nearest plant type ────────────────────────────
print('\nLoading gridinfo and labeling grid cells...')
gridinfo = pd.read_csv(
    os.path.join(CONF_DIR, 'gridinfo_xy.csv'),
    usecols=['IDCode', 'Centroid_Lat', 'Centroid_Long'])
print(f'  Grid cells: {len(gridinfo):,}')

pts = np.radians(gridinfo[['Centroid_Lat', 'Centroid_Long']].values.astype(float))

near_steel  = steel_tree.query_radius(pts,  r=r_rad, count_only=True) > 0
near_coal   = coal_tree.query_radius(pts,   r=r_rad, count_only=True) > 0
near_cement = cement_tree.query_radius(pts, r=r_rad, count_only=True) > 0

# Count how many plant types are within D_KM
n_types = near_steel.astype(int) + near_coal.astype(int) + near_cement.astype(int)

# Label: single-type cells only (exclude ambiguous)
label = np.full(len(gridinfo), '', dtype=object)
label[near_steel  & (n_types == 1)] = 'steel'
label[near_coal   & (n_types == 1)] = 'coal'
label[near_cement & (n_types == 1)] = 'cement'

gridinfo['label'] = label
labeled = gridinfo[gridinfo['label'] != ''].copy()
print(f'  Labeled cells: {len(labeled):,}  '
      f'(steel={( labeled["label"]=="steel").sum()}, '
      f'coal={(labeled["label"]=="coal").sum()}, '
      f'cement={(labeled["label"]=="cement").sum()})')
print(f'  Ambiguous (excluded): {(n_types > 1).sum():,}')

labeled_ids = set(labeled['IDCode'])
id_to_label = dict(zip(labeled['IDCode'], labeled['label']))

# ── Load November features for labeled cells only (chunked) ───────────────
print(f'\nLoading November features (month={TRAIN_MONTH}) for labeled cells...')
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
                  f'{(time.time()-t0)/60:.1f} min, rows so far: '
                  f'{sum(len(p) for p in parts):,}')
    print(f'  {os.path.basename(fpath)}: done ({n_chunks} chunks, '
          f'{(time.time()-t0)/60:.1f} min)')

feat_df = pd.concat(parts, ignore_index=True)
feat_df['label'] = feat_df['IDCode'].map(id_to_label)
feat_df = feat_df.dropna(subset=FEATURES + ['label'])
print(f'\nFeature dataset: {len(feat_df):,} rows')
print(feat_df.groupby('label').size().to_string())

# Drop NaN in feature columns and standardize
X_raw = feat_df[FEATURES].values
y_str = feat_df['label'].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── Phase 1: K-means Clustering ───────────────────────────────────────────
print('\n--- Phase 1: K-means Clustering ---')

# Silhouette scores for K=2..6
print('Silhouette scores:')
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km_labels = km.fit_predict(X)
    sil = silhouette_score(X, km_labels, sample_size=min(10000, len(X)))
    print(f'  K={k}: silhouette={sil:.4f}')

# Main result: K=3
km3 = KMeans(n_clusters=3, random_state=SEED, n_init=20)
cluster_labels = km3.fit_predict(X)

# Purity table: cluster x plant type
purity = pd.crosstab(cluster_labels, y_str,
                     rownames=['Cluster'], colnames=['Plant type'])
purity['Total']   = purity.sum(axis=1)
purity['Purity%'] = purity[['steel', 'coal', 'cement']].max(axis=1) / purity['Total'] * 100
purity['Dominant'] = purity[['steel', 'coal', 'cement']].idxmax(axis=1)
print('\nK-means purity (K=3):')
print(purity.to_string())
purity.to_csv(os.path.join(OUT_DIR, 'table_cluster_purity.csv'))
print(f'Saved: table_cluster_purity.csv')

# ── Phase 1b: UMAP Visualization ─────────────────────────────────────────
if HAS_UMAP:
    print('\nRunning UMAP...')
    # Subsample for speed if large
    MAX_UMAP = 30_000
    if len(X) > MAX_UMAP:
        idx = np.random.default_rng(SEED).choice(len(X), MAX_UMAP, replace=False)
        X_u = X[idx]; y_u = y_str[idx]; c_u = cluster_labels[idx]
    else:
        X_u = X; y_u = y_str; c_u = cluster_labels

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean',
                        random_state=SEED)
    emb = reducer.fit_transform(X_u)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Left: colored by plant type
    for ptype in LABEL_NAMES:
        mask = y_u == ptype
        axes[0].scatter(emb[mask, 0], emb[mask, 1],
                        c=COLORS[ptype], marker=MARKERS[ptype],
                        s=5, alpha=0.4, label=ptype.capitalize(), rasterized=True)
    axes[0].set_title('(A) Ground-Truth Plant Type', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')
    axes[0].legend(fontsize=9, markerscale=3)

    # Right: colored by k-means cluster (dominant label shown)
    cluster_colors = {0: '#1a9850', 1: '#d73027', 2: '#4575b4'}
    # Map cluster number to dominant plant type for legend
    dominant_map = purity['Dominant'].to_dict()
    for k in range(3):
        mask = c_u == k
        dom = dominant_map.get(k, str(k))
        axes[1].scatter(emb[mask, 0], emb[mask, 1],
                        c=cluster_colors[k], s=5, alpha=0.4,
                        label=f'Cluster {k} ({dom})', rasterized=True)
    axes[1].set_title('(B) K-Means Cluster Assignment (K=3)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')
    axes[1].legend(fontsize=9, markerscale=3)

    fig.suptitle('UMAP: 9-Dimensional Satellite Features Separate Industrial Plant Types',
                 fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_plant_clustering_umap.pdf')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f'Saved: fig_plant_clustering_umap.pdf')
else:
    print('[SKIP] UMAP figure skipped (umap-learn not installed)')

# ── Phase 2: XGBoost Multi-Class Classifier ──────────────────────────────
print('\n--- Phase 2: XGBoost Multi-Class Classifier ---')

label_enc = {'steel': 0, 'coal': 1, 'cement': 2}
y_int = np.array([label_enc[l] for l in y_str])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_int, test_size=0.2, random_state=SEED, stratify=y_int)

clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=SEED, verbosity=0, num_class=3,
    objective='multi:softmax')
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)

# Per-class report
label_names_ord = ['steel', 'coal', 'cement']
report = classification_report(y_te, y_pred, target_names=label_names_ord, output_dict=True)
report_df = pd.DataFrame(report).T
print('\nXGBoost multi-class report:')
print(report_df.round(4).to_string())
report_df.to_csv(os.path.join(OUT_DIR, 'table_multiclass_accuracy.csv'))
print(f'Saved: table_multiclass_accuracy.csv')

# Confusion matrix figure
cm = confusion_matrix(y_te, y_pred)
fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names_ord)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title('XGBoost 3-Class Confusion Matrix\n(Steel / Coal / Cement)',
             fontsize=10, fontweight='bold')
fig.tight_layout()
fig_path2 = os.path.join(OUT_DIR, 'fig_cluster_confusion.pdf')
fig.savefig(fig_path2, bbox_inches='tight')
plt.close()
print(f'Saved: fig_cluster_confusion.pdf')

# ── Summary ───────────────────────────────────────────────────────────────
print()
print('=' * 68)
print('SUMMARY')
print('=' * 68)
print(f'Labeled cells: {len(feat_df):,}  '
      f'(steel={(y_str=="steel").sum():,}, '
      f'coal={(y_str=="coal").sum():,}, '
      f'cement={(y_str=="cement").sum():,})')
print()
print('K-means (K=3) purity:')
for k in range(3):
    dom = purity.loc[k, 'Dominant']
    pur = purity.loc[k, 'Purity%']
    tot = purity.loc[k, 'Total']
    print(f'  Cluster {k}: dominant={dom:<7}  purity={pur:.1f}%  N={tot:,}')
print()
acc = (y_pred == y_te).mean() * 100
print(f'XGBoost 3-class accuracy: {acc:.1f}%  (chance=33.3%)')
for lbl in label_names_ord:
    prec = report[lbl]['precision']
    rec  = report[lbl]['recall']
    f1   = report[lbl]['f1-score']
    print(f'  {lbl:<8}: precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}')
print()
print('Done.')
