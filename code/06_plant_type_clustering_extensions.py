
import os, sys, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, roc_auc_score,
                              accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb
import umap
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

CACHE    = os.path.join(OUT_DIR, 'feat_labeled_D10.parquet')
FEATURES = ['CO_MEAN', 'NO2_MEAN', 'PM2_5_MEAN', 'PM10_MEAN', 'SO2_MEAN',
            'LSTA_MEAN', 'LSTT_MEAN', 'NTL_MEAN', 'O3_MEAN']
D_VALUES = [1, 2, 3, 5, 10]
SEED     = 42
R_EARTH  = 6371.0
MIN_CELLS = 100   # skip D if any class has fewer

# ── UMAP subsample to keep runtime manageable ──────────────────────────────
UMAP_MAX = 30_000   # total rows (random from full pool, preserving class proportions)


# ── Helper: k-means purity ──────────────────────────────────────────────────
def kmeans_purity(X, true_labels, k, seed=42):
    """
    Run k-means with k clusters.  Purity = sum_i max_j count(cluster_i, label_j) / N.
    Returns purity (0–1) and cluster assignments.
    """
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(X)
    c = km.labels_
    total = len(c)
    correct = 0
    for ci in range(k):
        mask = c == ci
        if mask.sum() == 0:
            continue
        # most common true label in this cluster
        vals, cnts = np.unique(true_labels[mask], return_counts=True)
        correct += cnts.max()
    return correct / total, c


# ── Helper: UMAP plot ───────────────────────────────────────────────────────
COLOR_MAP = {'steel': '#1f4e79', 'coal': '#808080', 'cement': '#ed7d31'}

def plot_umap(X_scaled, labels_arr, title, fig_path, subsample=UMAP_MAX):
    unique_labels = sorted(set(labels_arr))
    # random sample from full pool — preserves true class proportions
    rng = np.random.default_rng(SEED)
    n = len(labels_arr)
    if n > subsample:
        idx = rng.choice(n, subsample, replace=False)
    else:
        idx = np.arange(n)
    X_sub   = X_scaled[idx]
    lbl_sub = labels_arr[idx]

    print(f'  Running UMAP on {len(idx):,} points...')
    reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
    emb = reducer.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for lbl in unique_labels:
        mask = lbl_sub == lbl
        color = COLOR_MAP.get(lbl, '#333333')
        ax.scatter(emb[mask, 0], emb[mask, 1], s=2, alpha=0.35,
                   color=color, label=lbl.capitalize(), rasterized=True)
    ax.set_xlabel('UMAP-1', fontsize=9)
    ax.set_ylabel('UMAP-2', fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(fontsize=8, markerscale=4)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {os.path.basename(fig_path)}')


# ── Load cache ──────────────────────────────────────────────────────────────
print('=' * 68)
print('Loading feat_labeled_D10.parquet ...')
if not os.path.exists(CACHE):
    raise FileNotFoundError(f'Cache not found: {CACHE}\nRun 05_plant_type_clustering_sensitivity.py first.')
feat_df = pd.read_parquet(CACHE)
print(f'  Loaded {len(feat_df):,} rows')
print(feat_df.groupby('label').size().to_string())

# ── Load steel plant coordinates (for Exercise 2 exclusion) ─────────────────
print('\nLoading steel plant locations...')
loc = pd.read_excel(os.path.join(CONF_DIR, 'Location_Plants_Full.xlsx'))
steel_plants = (loc[loc['Capacityoperatingstatus'] == 'operating']
                [['latitude', 'longitude']].dropna().drop_duplicates().copy())
steel_plants.columns = ['Latitude', 'Longitude']
print(f'  Steel plants: {len(steel_plants):,}')

steel_tree = BallTree(
    np.radians(steel_plants[['Latitude', 'Longitude']].values.astype(float)),
    metric='haversine')

# ── We also need IDCode → lat/lon to do the exclusion check ─────────────────
print('Loading gridinfo_xy.csv for lat/lon lookup...')
gridinfo = pd.read_csv(os.path.join(CONF_DIR, 'gridinfo_xy.csv'),
                       usecols=['IDCode', 'Centroid_Lat', 'Centroid_Long'])
id_to_lat = dict(zip(gridinfo['IDCode'], gridinfo['Centroid_Lat']))
id_to_lon = dict(zip(gridinfo['IDCode'], gridinfo['Centroid_Long']))
feat_df['Centroid_Lat']  = feat_df['IDCode'].map(id_to_lat)
feat_df['Centroid_Long'] = feat_df['IDCode'].map(id_to_lon)
feat_df = feat_df.dropna(subset=['Centroid_Lat', 'Centroid_Long'])
print(f'  After lat/lon merge: {len(feat_df):,} rows')


# ============================================================
# EXERCISE 1: Steel vs. Cement Binary
# ============================================================
print('\n' + '=' * 68)
print('EXERCISE 1: Steel vs. Cement Binary')
print('=' * 68)

sc_df = feat_df[feat_df['label'].isin(['steel', 'cement'])].copy()
print(f'Steel+cement rows: {len(sc_df):,}')
print(sc_df.groupby('label').size().to_string())

X_sc_all   = StandardScaler().fit_transform(sc_df[FEATURES].values)
lbl_sc_all = sc_df['label'].values

# ── Unsupervised: K-means K=2 on full D=10 km dataset ─────────────────────
print('\nK-means K=2 (steel vs cement, D≤10 km)...')
purity_2, km_labels_2 = kmeans_purity(X_sc_all, lbl_sc_all, k=2, seed=SEED)
print(f'  Purity = {purity_2*100:.1f}%  (baseline 3-class: 53.1%)')
# cluster contents
for ci in range(2):
    mask = km_labels_2 == ci
    vals, cnts = np.unique(lbl_sc_all[mask], return_counts=True)
    print(f'  Cluster {ci}: {dict(zip(vals, cnts))}')

# ── UMAP (D=10 km, steel+cement) ──────────────────────────────────────────
print('\nUMAP (steel vs cement, D≤10 km)...')
plot_umap(X_sc_all, lbl_sc_all,
          'UMAP: Steel vs. Cement (D ≤ 10 km, Nov)',
          os.path.join(OUT_DIR, 'fig_steel_cement_umap.pdf'))

# ── Supervised: Binary XGBoost sweep over D ───────────────────────────────
print('\nBinary XGBoost sensitivity over D...')
label_enc_bin = {'cement': 0, 'steel': 1}

rows_bin = []
for D in D_VALUES:
    sub = sc_df[sc_df['min_dist'] <= D].copy()
    n_s = (sub['label'] == 'steel').sum()
    n_c = (sub['label'] == 'cement').sum()
    n_total = len(sub)
    print(f'\n  D = {D} km: N={n_total:,}  (steel={n_s:,}, cement={n_c:,})')

    if min(n_s, n_c) < MIN_CELLS:
        print(f'  [SKIP] < {MIN_CELLS} cells in one class')
        rows_bin.append({'D_km': D, 'N_total': n_total, 'N_steel': n_s,
                         'N_cement': n_c, 'kmeans_purity': np.nan,
                         'accuracy': np.nan, 'steel_precision': np.nan,
                         'steel_recall': np.nan, 'steel_f1': np.nan, 'auc': np.nan})
        continue

    # K-means purity at this D
    X_d = StandardScaler().fit_transform(sub[FEATURES].values)
    lbl_d = sub['label'].values
    pur_d, _ = kmeans_purity(X_d, lbl_d, k=2, seed=SEED)
    print(f'  K-means purity (K=2): {pur_d*100:.1f}%')

    # Supervised
    y = np.array([label_enc_bin[l] for l in lbl_d])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_d, y, test_size=0.2, random_state=SEED, stratify=y)

    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=SEED, verbosity=0)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_te, y_pred) * 100
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_te, y_prob)
    except Exception:
        auc = np.nan

    print(f'  Accuracy={acc:.1f}%  Prec={prec:.3f}  Rec={rec:.3f}  '
          f'F1={f1:.3f}  AUC={auc:.3f}')

    rows_bin.append({'D_km': D, 'N_total': n_total, 'N_steel': n_s,
                     'N_cement': n_c, 'kmeans_purity': round(pur_d*100, 1),
                     'accuracy': round(acc, 2), 'steel_precision': round(prec, 4),
                     'steel_recall': round(rec, 4), 'steel_f1': round(f1, 4),
                     'auc': round(auc, 4)})

results_bin = pd.DataFrame(rows_bin)
out_csv1 = os.path.join(OUT_DIR, 'table_steel_cement_binary.csv')
results_bin.to_csv(out_csv1, index=False)
print(f'\nSaved: table_steel_cement_binary.csv')
print(results_bin.to_string(index=False))

# ── Figure: supervised sensitivity (steel vs cement) ──────────────────────
valid1 = results_bin.dropna(subset=['accuracy'])
if len(valid1) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    axes[0].plot(valid1['D_km'], valid1['accuracy'], 'o-', color='#1f4e79',
                 lw=2, ms=7, label='Binary accuracy')
    axes[0].axhline(50, color='gray', ls='--', lw=1, label='Chance (50%)')
    axes[0].set_xlabel('Distance threshold D (km)', fontsize=10)
    axes[0].set_ylabel('Accuracy (%)', fontsize=10)
    axes[0].set_title('(A) Supervised Accuracy vs. D', fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].set_xticks(D_VALUES); axes[0].set_ylim(0, 100)

    axes[1].plot(valid1['D_km'], valid1['steel_precision'] * 100, 's-',
                 color='#1f4e79', lw=2, ms=7, label='Steel precision')
    axes[1].plot(valid1['D_km'], valid1['steel_recall'] * 100, '^--',
                 color='#c00000', lw=2, ms=7, label='Steel recall')
    axes[1].plot(valid1['D_km'], valid1['auc'] * 100, 'D:',
                 color='#70ad47', lw=1.5, ms=5, label='AUC')
    axes[1].plot(valid1['D_km'], valid1['kmeans_purity'], 'v:',
                 color='#ed7d31', lw=1.5, ms=5, label='K-means purity (K=2)')
    axes[1].set_xlabel('Distance threshold D (km)', fontsize=10)
    axes[1].set_ylabel('Metric (%)', fontsize=10)
    axes[1].set_title('(B) Metrics vs. D', fontsize=10, fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].set_xticks(D_VALUES); axes[1].set_ylim(0, 100)

    fig.suptitle('Exercise 1: Steel vs. Cement Binary Classification',
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_steel_cement_sensitivity.pdf'),
                bbox_inches='tight')
    plt.close()
    print('Saved: fig_steel_cement_sensitivity.pdf')


# ============================================================
# EXERCISE 2: 50 km Exclusion Zone
# ============================================================
print('\n' + '=' * 68)
print('EXERCISE 2: 50 km Exclusion Zone')
print('=' * 68)

# ── Compute distance from each coal/cement cell to nearest steel plant ─────
coal_cem = feat_df[feat_df['label'].isin(['coal', 'cement'])].copy()
steel_only = feat_df[feat_df['label'] == 'steel'].copy()
print(f'Coal+cement rows: {len(coal_cem):,}  Steel rows: {len(steel_only):,}')

print('Computing distance from coal/cement cells to nearest steel plant...')
pts_cc = np.radians(coal_cem[['Centroid_Lat', 'Centroid_Long']].values.astype(float))
dist_raw, _ = steel_tree.query(pts_cc, k=1)
coal_cem['dist_to_steel_km'] = dist_raw.ravel() * R_EARTH

D_EXCL = 50.0
keep_cc = coal_cem[coal_cem['dist_to_steel_km'] > D_EXCL].copy()
excl_cc = coal_cem[coal_cem['dist_to_steel_km'] <= D_EXCL]
print(f'  Coal/cement cells excluded (within {D_EXCL} km of steel): {len(excl_cc):,}')
print(f'  Coal/cement cells remaining: {len(keep_cc):,}')
print(f'  Remaining breakdown: {keep_cc.groupby("label").size().to_dict()}')

# Geographic check on remaining coal/cement cells
print('\nGeographic check on remaining coal/cement cells:')
for lbl in ['coal', 'cement']:
    sub_g = keep_cc[keep_cc['label'] == lbl]
    if len(sub_g) == 0:
        continue
    print(f'  {lbl}: N={len(sub_g):,}  '
          f'Lat mean={sub_g["Centroid_Lat"].mean():.1f}  '
          f'Lon mean={sub_g["Centroid_Long"].mean():.1f}  '
          f'(Lat range [{sub_g["Centroid_Lat"].min():.1f}, '
          f'{sub_g["Centroid_Lat"].max():.1f}])')

# Combine with all steel cells
ex2_df = pd.concat([steel_only, keep_cc], ignore_index=True)
ex2_df = ex2_df.dropna(subset=FEATURES + ['label'])
print(f'\nExercise 2 dataset: {len(ex2_df):,} rows')
print(ex2_df.groupby('label').size().to_string())

X_ex2_all   = StandardScaler().fit_transform(ex2_df[FEATURES].values)
lbl_ex2_all = ex2_df['label'].values

# ── Unsupervised: K-means K=3 ──────────────────────────────────────────────
print('\nK-means K=3 (after 50 km exclusion)...')
purity_3_ex2, km_labels_3_ex2 = kmeans_purity(X_ex2_all, lbl_ex2_all, k=3, seed=SEED)
print(f'  Purity = {purity_3_ex2*100:.1f}%  (baseline 03_location_classifier_negative_control_umap.py: 53.1%)')
for ci in range(3):
    mask = km_labels_3_ex2 == ci
    vals, cnts = np.unique(lbl_ex2_all[mask], return_counts=True)
    print(f'  Cluster {ci}: {dict(zip(vals, cnts))}')

# ── UMAP (after exclusion) ─────────────────────────────────────────────────
print('\nUMAP (after 50 km exclusion, 3-class)...')
plot_umap(X_ex2_all, lbl_ex2_all,
          f'UMAP: After {int(D_EXCL)} km Exclusion Zone (3-class, Nov)',
          os.path.join(OUT_DIR, 'fig_50km_umap.pdf'))

# ── Supervised: 3-class XGBoost ───────────────────────────────────────────
print('\n3-class XGBoost (after 50 km exclusion)...')
label_enc3 = {'steel': 0, 'coal': 1, 'cement': 2}
label_names3 = ['steel', 'coal', 'cement']

n_steel  = (ex2_df['label'] == 'steel').sum()
n_coal   = (ex2_df['label'] == 'coal').sum()
n_cement = (ex2_df['label'] == 'cement').sum()

rows_ex2 = []
if min(n_steel, n_coal, n_cement) < MIN_CELLS:
    print(f'  [SKIP] Fewer than {MIN_CELLS} cells in one class — skipping.')
    rows_ex2.append({'Setup': f'{D_EXCL} km exclusion (D=10 km labels)',
                     'N_steel': n_steel, 'N_coal': n_coal, 'N_cement': n_cement,
                     'kmeans_purity_pct': round(purity_3_ex2 * 100, 1),
                     'accuracy': np.nan, 'steel_precision': np.nan,
                     'steel_recall': np.nan, 'steel_f1': np.nan})
else:
    y_ex2 = np.array([label_enc3[l] for l in lbl_ex2_all])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_ex2_all, y_ex2, test_size=0.2, random_state=SEED, stratify=y_ex2)

    clf3 = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=SEED, verbosity=0,
        num_class=3, objective='multi:softmax')
    clf3.fit(X_tr, y_tr)
    y_pred3 = clf3.predict(X_te)

    rep3 = classification_report(y_te, y_pred3, target_names=label_names3,
                                  output_dict=True)
    acc3 = (y_pred3 == y_te).mean() * 100
    print(f'  Accuracy: {acc3:.1f}%  (baseline 03_location_classifier_negative_control_umap.py: 68.0%, chance: 33.3%)')
    print(f'  Steel:  prec={rep3["steel"]["precision"]:.3f}  '
          f'rec={rep3["steel"]["recall"]:.3f}  F1={rep3["steel"]["f1-score"]:.3f}')
    print(f'  Coal:   prec={rep3["coal"]["precision"]:.3f}  '
          f'rec={rep3["coal"]["recall"]:.3f}')
    print(f'  Cement: prec={rep3["cement"]["precision"]:.3f}  '
          f'rec={rep3["cement"]["recall"]:.3f}')

    rows_ex2.append({
        'Setup':               f'{int(D_EXCL)} km exclusion (D=10 km labels)',
        'N_steel':             n_steel,
        'N_coal':              n_coal,
        'N_cement':            n_cement,
        'kmeans_purity_pct':   round(purity_3_ex2 * 100, 1),
        'accuracy':            round(acc3, 2),
        'steel_precision':     round(rep3['steel']['precision'], 4),
        'steel_recall':        round(rep3['steel']['recall'], 4),
        'steel_f1':            round(rep3['steel']['f1-score'], 4),
        'coal_precision':      round(rep3['coal']['precision'], 4),
        'cement_precision':    round(rep3['cement']['precision'], 4),
    })

# Baseline from 03_location_classifier_negative_control_umap.py
rows_ex2.insert(0, {
    'Setup': 'Baseline (no exclusion)',
    'N_steel': 99_442, 'N_coal': 307_668, 'N_cement': 319_614,
    'kmeans_purity_pct': 53.1,
    'accuracy': 68.0, 'steel_precision': 0.748, 'steel_recall': 0.202,
    'steel_f1': 0.317, 'coal_precision': 0.658, 'cement_precision': 0.718,
})

results_ex2 = pd.DataFrame(rows_ex2)
out_csv2 = os.path.join(OUT_DIR, 'table_50km_exclusion.csv')
results_ex2.to_csv(out_csv2, index=False)
print(f'\nSaved: table_50km_exclusion.csv')
print(results_ex2.to_string(index=False))


# ============================================================
# EXERCISE 3: Steel vs. Cement + 50 km Exclusion (Strongest Test)
# ============================================================
print('\n' + '=' * 68)
print('EXERCISE 3: Steel vs. Cement Binary + 50 km Exclusion Zone')
print('=' * 68)

# cement cells that are > 50 km from any steel plant (already computed above)
cement_excl = keep_cc[keep_cc['label'] == 'cement'].copy()
ex3_df = pd.concat([steel_only, cement_excl], ignore_index=True)
ex3_df = ex3_df.dropna(subset=FEATURES + ['label'])
n_s3 = (ex3_df['label'] == 'steel').sum()
n_c3 = (ex3_df['label'] == 'cement').sum()
print(f'Steel: {n_s3:,}  Cement (>50 km from steel): {n_c3:,}  Total: {len(ex3_df):,}')

X_ex3_all   = StandardScaler().fit_transform(ex3_df[FEATURES].values)
lbl_ex3_all = ex3_df['label'].values

# ── Unsupervised: K-means K=2 ──────────────────────────────────────────────
print('\nK-means K=2 (steel vs cement, 50 km excluded)...')
purity_2_ex3, _ = kmeans_purity(X_ex3_all, lbl_ex3_all, k=2, seed=SEED)
print(f'  Purity = {purity_2_ex3*100:.1f}%  '
      f'(Ex1 no-exclusion baseline: {purity_2*100:.1f}%)')

# ── UMAP ───────────────────────────────────────────────────────────────────
print('\nUMAP (steel vs cement, 50 km exclusion)...')
plot_umap(X_ex3_all, lbl_ex3_all,
          'UMAP: Steel vs. Cement, Cement >50 km from Steel',
          os.path.join(OUT_DIR, 'fig_steel_cement_excl_umap.pdf'))

# ── Supervised: Binary XGBoost ─────────────────────────────────────────────
print('\nBinary XGBoost (steel vs cement, 50 km excluded)...')
if min(n_s3, n_c3) < MIN_CELLS:
    print('  [SKIP] too few cells')
    acc3b, prec3b, rec3b, f1_3b, auc3b = [np.nan] * 5
else:
    y_ex3 = np.array([label_enc_bin[l] for l in lbl_ex3_all])
    X_tr3, X_te3, y_tr3, y_te3 = train_test_split(
        X_ex3_all, y_ex3, test_size=0.2, random_state=SEED, stratify=y_ex3)

    clf3b = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=SEED, verbosity=0)
    clf3b.fit(X_tr3, y_tr3)
    y_pred3b = clf3b.predict(X_te3)
    y_prob3b = clf3b.predict_proba(X_te3)[:, 1]

    acc3b  = accuracy_score(y_te3, y_pred3b) * 100
    prec3b = precision_score(y_te3, y_pred3b, zero_division=0)
    rec3b  = recall_score(y_te3, y_pred3b, zero_division=0)
    f1_3b  = f1_score(y_te3, y_pred3b, zero_division=0)
    try:
        auc3b = roc_auc_score(y_te3, y_prob3b)
    except Exception:
        auc3b = np.nan

    print(f'  Accuracy={acc3b:.1f}%  Prec={prec3b:.3f}  Rec={rec3b:.3f}  '
          f'F1={f1_3b:.3f}  AUC={auc3b:.3f}')

r10_ex3 = results_bin[results_bin['D_km'] == 10].iloc[0]
results_ex3 = pd.DataFrame([
    {'Setup': 'Ex1: Steel vs Cement (no exclusion)',
     'N_steel': 99_442, 'N_cement': 319_614,
     'kmeans_purity_pct': round(purity_2*100, 1),
     'accuracy': round(r10_ex3['accuracy'], 2),
     'steel_precision': r10_ex3['steel_precision'],
     'steel_recall': r10_ex3['steel_recall'], 'auc': r10_ex3['auc']},
    {'Setup': 'Ex3: Steel vs Cement (cement >50 km from steel)',
     'N_steel': n_s3, 'N_cement': n_c3,
     'kmeans_purity_pct': round(purity_2_ex3*100, 1),
     'accuracy': round(acc3b, 2) if not np.isnan(acc3b) else np.nan,
     'steel_precision': round(prec3b, 4) if not np.isnan(prec3b) else np.nan,
     'steel_recall': round(rec3b, 4) if not np.isnan(rec3b) else np.nan,
     'auc': round(auc3b, 4) if not np.isnan(auc3b) else np.nan},
])
out_csv3 = os.path.join(OUT_DIR, 'table_steel_cement_excl.csv')
results_ex3.to_csv(out_csv3, index=False)
print(f'\nSaved: table_steel_cement_excl.csv')
print(results_ex3.to_string(index=False))

# ============================================================
# FINAL SUMMARY
# ============================================================
print('\n' + '=' * 68)
print('SUMMARY FOR SLIDES')
print('=' * 68)
print(f'\nExercise 1 — Steel vs. Cement Binary (D=10 km):')
print(f'  K-means purity (K=2):    {purity_2*100:.1f}%  (3-class baseline: 53.1%)')
r10 = results_bin[results_bin['D_km'] == 10].iloc[0]
print(f'  XGBoost accuracy (D=10): {r10["accuracy"]:.1f}%  (3-class baseline: 68.0%)')
print(f'  Steel precision (D=10):  {r10["steel_precision"]:.3f}  recall: {r10["steel_recall"]:.3f}')
print(f'\nExercise 2 — 50 km Exclusion Zone (3-class):')
print(f'  K-means purity (K=3):    {purity_3_ex2*100:.1f}%  (baseline: 53.1%)')
if len(rows_ex2) > 1:
    r_ex2 = results_ex2.iloc[-1]
    print(f'  XGBoost accuracy:        {r_ex2["accuracy"]:.1f}%  (baseline: 68.0%)')
    print(f'  Steel precision:         {r_ex2["steel_precision"]:.3f}  recall: {r_ex2["steel_recall"]:.3f}')
print(f'\nExercise 3 — Steel vs. Cement + 50 km Exclusion (strongest test):')
print(f'  K-means purity (K=2):    {purity_2_ex3*100:.1f}%  (Ex1 baseline: {purity_2*100:.1f}%)')
if not np.isnan(acc3b):
    print(f'  XGBoost accuracy:        {acc3b:.1f}%')
    print(f'  Steel precision:         {prec3b:.3f}  recall: {rec3b:.3f}  AUC: {auc3b:.3f}')
print('\nDone.')
