# Sumber: lampiran kode pada laporan Tugas 05; paragraf penjelasan laporan dijadikan komentar.
# Kode pada lampiran ini adalah versi yang dirapikan dan diberi komentar dari skrip analisis yang sebenarnya dijalankan untuk menghasilkan seluruh angka, tabel, dan gambar dalam laporan (bagian kosmetik visualisasi seperti pewarnaan dan tata letak sumbu diringkas agar lebih mudah dibaca). Skrip lengkap dapat diminta dari penulis.
# A.1 Persiapan Data, Profil Awal, dan Rekayasa Fitur
# Memuat data, memeriksa kebersihan (nilai hilang dan duplikat), menguji kemenyerupaan sebaran atribut numerik terhadap distribusi seragam, menguji asosiasi antarvariabel kategorikal, dan membentuk dua himpunan fitur untuk Skenario 1 (metadata) dan Skenario 2 (kata pada judul), sesuai Bab 3 dan Subbab 4.1–4.2.
import json, math, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
 
CSV = "trending_topics_2026_synthetic.csv"
RS  = 42                                    # random_state tetap di seluruh skrip
df  = pd.read_csv(CSV)
df["date"]    = pd.to_datetime(df["date"])
df["day"]     = df.date.dt.day
df["weekday"] = df.date.dt.weekday
 
CATS = ["ai_and_tech", "economy", "entertainment",
        "global_events", "sports"]
y = df.topic_category.map({c: i for i, c in enumerate(CATS)}).values
 
# --- profil dasar & kebersihan data (Subbab 4.1) --------------------
n_missing = int(df.isna().sum().sum())
n_dup     = int(df.duplicated(
    subset=[c for c in df.columns if c not in ["id", "day", "weekday"]]
).sum())
ks_eng   = stats.kstest((df.engagement_score - 50) / 9950, "uniform")
ks_trend = stats.kstest((df.trend_score - .1) / .9, "uniform")
# p-value tinggi pada kedua uji KS -> sebaran mendekati uniform,
# konsisten dengan data yang dibangkitkan secara sintetis (Subbab 3.1).
 
# --- uji asosiasi kategorikal: Cramer's V dan chi-kuadrat ------------
def cramers_v(a, b):
    ct = pd.crosstab(a, b)
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    n = ct.values.sum(); k = min(ct.shape) - 1
    return math.sqrt(chi2 / (n * k)), chi2, p, dof
 
for col in ["sentiment", "source", "country", "language"]:
    v, chi2, p, dof = cramers_v(df.topic_category, df[col])
    print(f"{col:10s} V={v:.3f}  chi2={chi2:7.2f}  p={p:.3f}  dof={dof}")
 
# --- uji beda rerata numerik antar kelas (ANOVA & Kruskal-Wallis) ----
for col in ["engagement_score", "trend_score"]:
    groups = [df[df.topic_category == k][col] for k in CATS]
    F, pF = stats.f_oneway(*groups)
    H_, pH = stats.kruskal(*groups)
    print(f"{col:18s} ANOVA F={F:.3f} p={pF:.3f} | Kruskal H={H_:.3f} p={pH:.3f}")
 
# --- rekayasa fitur: dua skenario prediktor yang dibandingkan --------
# Skenario 1 (metadata) - topic_subcategory dan short_text DIBUANG
# karena keduanya membocorkan label topic_category (Subbab 3.3).
XA = pd.get_dummies(
        df[["source", "language", "country", "sentiment"]]
     ).astype(int)
XA["engagement_score"] = df.engagement_score
XA["trend_score"]      = df.trend_score
XA["day"]              = df.day
XA["weekday"]          = df.weekday
 
# Skenario 2 (teks) - bag-of-words biner dari kolom headline
cv_words = CountVectorizer(
    binary=True, lowercase=True,
    token_pattern=r"(?u)\b[a-zA-Z0-9]{2,}\b")
XB = pd.DataFrame(cv_words.fit_transform(df.headline).toarray(),
                   columns=cv_words.get_feature_names_out())
print("Ukuran kosakata (Skenario 2):", XB.shape[1], "kata")
# A.2 Entropi, Gini, Information Gain, dan Gain Ratio
# Implementasi manual ukuran ketidakmurnian dan pencarian split terbaik yang menjadi dasar Tabel 8–10 serta Gambar 12–14, sekaligus menjadi penanda bahwa perhitungan tersebut telah diverifikasi secara independen dari fungsi bawaan pustaka scikit-learn.
# Fungsi ketidakmurnian dan pencarian split terbaik (Subbab 4.3),
# dipakai untuk menghitung information gain, gain ratio, dan
# penurunan Gini yang mendasari Tabel 8-10 serta Gambar 12-14.
 
def entropy(counts):
    p = np.asarray(counts, float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())
 
def gini(counts):
    p = np.asarray(counts, float) / np.sum(counts)
    return float(1 - (p ** 2).sum())
 
def split_stats(xcat, yv, k=5):
    """Split multi-arah pada atribut kategorikal xcat."""
    n = len(yv)
    H  = entropy(np.bincount(yv, minlength=k))
    G  = gini(np.bincount(yv, minlength=k))
    H2 = G2 = si = 0.0
    for v in np.unique(xcat):
        m = xcat == v
        w = m.sum() / n
        H2 += w * entropy(np.bincount(yv[m], minlength=k))
        G2 += w * gini(np.bincount(yv[m], minlength=k))
        si -= w * math.log2(w)                 # split information
    ig = H - H2
    return dict(ig=ig, gr=ig / si if si > 0 else 0,
                gini_dec=G - G2, split_info=si, H_after=H2)
 
def best_numeric(x, yv, k=5, nq=60):
    """Cari titik potong biner terbaik pada atribut numerik x
    dengan menyisir sejumlah kuantil (di Orange: pencarian threshold
    dilakukan pada setiap titik unik nilai yang sudah diurutkan)."""
    n = len(yv)
    H = entropy(np.bincount(yv, minlength=k))
    best = None
    for t in np.unique(np.quantile(x, np.linspace(.02, .98, nq))):
        m = x <= t; w = m.mean()
        if w <= 0 or w >= 1:
            continue
        H2 = (w * entropy(np.bincount(yv[m], minlength=k)) +
              (1 - w) * entropy(np.bincount(yv[~m], minlength=k)))
        si = -(w * math.log2(w) + (1 - w) * math.log2(1 - w))
        r = dict(ig=H - H2, gr=(H - H2) / si, thr=float(t))
        if best is None or r["ig"] > best["ig"]:
            best = r
    return best
 
# Peringkat atribut Skenario 1 menurut information gain (Tabel 10)
rankA = []
for c in ["source", "language", "country", "sentiment"]:
    rankA.append(dict(attr=c, kind="kategorik", **split_stats(df[c].values, y)))
for c in ["engagement_score", "trend_score", "day", "weekday"]:
    rankA.append(dict(attr=c, kind="numerik",
                       **best_numeric(df[c].values.astype(float), y)))
rankA.sort(key=lambda r: -r["ig"])
for r in rankA:
    print(r["attr"], "IG=", round(r["ig"], 4), "bit",
          "GR=", round(r.get("gr", 0), 4))
# A.3 Pembagian Data, Pohon Keputusan, dan Pencarian Grid
# Pembagian data latih/uji berstrata, definisi pohon bergaya default Orange, pencarian grid untuk pemangkasan pada kedua skenario, dan perbandingan model pada skema validasi silang 10-lipatan yang meniru widget Test and Score.
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      GridSearchCV, cross_validate)
 
# --- pembagian data: 80% latih, 20% uji, berstrata pada label -------
idx     = np.arange(len(df))
tr, te  = train_test_split(idx, test_size=.2, random_state=RS, stratify=y)
skf10   = StratifiedKFold(10, shuffle=True, random_state=RS)   # Test&Score
skf5    = StratifiedKFold(5,  shuffle=True, random_state=RS)   # grid search
 
def orange_tree(criterion="gini", **kw):
    """Meniru parameter widget Tree bawaan Orange:
    min instances in leaves = 2, minimal 5 instans untuk displit,
    tanpa batas kedalaman."""
    p = dict(min_samples_leaf=2, min_samples_split=5,
             max_depth=100, random_state=RS)
    p.update(kw)
    return DecisionTreeClassifier(criterion=criterion, **p)
 
# --- pencarian grid (pruning) untuk Skenario 1 dan Skenario 2 -------
grid = {"criterion": ["gini", "entropy"],
        "max_depth": [1, 2, 3, 4, 5, 6, 8, 10, 15, None],
        "min_samples_leaf": [1, 5, 10, 25, 50, 100]}
 
gridA = GridSearchCV(DecisionTreeClassifier(random_state=RS), grid,
                      cv=skf5, scoring="accuracy", n_jobs=-1
        ).fit(XA.iloc[tr], y[tr])
bestA = gridA.best_estimator_
print("Skenario 1 - parameter terbaik:", gridA.best_params_,
      "CV:", round(gridA.best_score_, 4))
 
gridB = GridSearchCV(DecisionTreeClassifier(random_state=RS), grid,
                      cv=skf5, scoring="accuracy", n_jobs=-1
        ).fit(XB.iloc[tr], y[tr])
# Banyak konfigurasi mencapai akurasi sempurna pada Skenario 2 (Lampiran D);
# di antaranya dipilih yang paling sederhana (kedalaman dan daun terkecil).
res  = pd.DataFrame(gridB.cv_results_)
top  = res[res.mean_test_score >= res.mean_test_score.max() - 1e-12].copy()
top["d"] = top.param_max_depth.apply(
    lambda v: 999 if v is None or (isinstance(v, float) and np.isnan(v)) else int(v))
bp = top.sort_values(["d", "param_min_samples_leaf"]).iloc[0]["params"]
bestB = DecisionTreeClassifier(random_state=RS, **bp).fit(XB.iloc[tr], y[tr])
print("Skenario 2 - konfigurasi paling sederhana:", bp)
 
# --- model pembanding pada widget Test and Score (Tabel 13, F41) ----
def models_for(bp_tuned):
    return {
        "Constant (mayoritas)": DummyClassifier(strategy="prior"),
        "Tree (default Orange)": orange_tree(),
        "Tree (Entropy, default)": orange_tree("entropy"),
        "Tree (tuning/pruning)": DecisionTreeClassifier(random_state=RS, **bp_tuned),
        "Random Forest": RandomForestClassifier(200, random_state=RS, n_jobs=-1),
        "Logistic Regression": make_pipeline(StandardScaler(),
                                              LogisticRegression(max_iter=2000)),
    }
 
def score_models(X, models, cv):
    rows = []
    for nm, mdl in models.items():
        r = cross_validate(mdl, X, y, cv=cv,
                scoring=["accuracy", "f1_weighted", "precision_weighted",
                         "recall_weighted", "roc_auc_ovr"])
        rows.append(dict(model=nm, ca=r["test_accuracy"].mean(),
                          auc=r["test_roc_auc_ovr"].mean(),
                          f1=r["test_f1_weighted"].mean()))
    return rows
 
scoreA = score_models(XA, models_for(gridA.best_params_), skf10)
scoreB = score_models(XB, models_for(bp), skf10)
for row in scoreA + scoreB:
    print(row["model"], "CA=", round(row["ca"], 3),
          "AUC=", round(row["auc"], 3), "F1=", round(row["f1"], 3))
 
# --- aturan pohon terpilih, ditampilkan sebagai teks (Lampiran B) ---
rulesB = export_text(bestB, feature_names=list(XB.columns),
                      class_names=CATS, max_depth=6)
print(rulesB)
# A.4 Evaluasi Model: Confusion Matrix, ROC, Uji Permutasi, dan Kalibrasi
# Evaluasi hold-out 20%, confusion matrix dan ROC one-vs-rest dari prediksi validasi silang, uji permutasi label untuk menilai signifikansi akurasi Skenario 1, serta perhitungan Expected Calibration Error yang mendasari Subbab 4.4 dan Gambar 22–28, 44–45.
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve,
                              auc, precision_recall_fscore_support)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import label_binarize
 
# --- evaluasi hold-out 20% pada pohon terpilih (Tabel 12, Gambar 34) -
def holdout(mdl, X):
    mdl.fit(X.iloc[tr], y[tr])
    pred = mdl.predict(X.iloc[te])
    p, r, f, s = precision_recall_fscore_support(
        y[te], pred, labels=range(5), zero_division=0)
    return dict(acc=accuracy_score(y[te], pred),
                cm=confusion_matrix(y[te], pred, labels=range(5)).tolist(),
                depth=int(mdl.get_depth()), leaves=int(mdl.get_n_leaves()))
 
holdA = holdout(bestA, XA)
holdB = holdout(bestB, XB)
print("Akurasi uji - Skenario 1:", round(holdA["acc"], 4),
      "| Skenario 2:", round(holdB["acc"], 4))
 
# --- confusion matrix dan ROC (one-vs-rest) dari prediksi 10-fold ---
# (mencerminkan widget Confusion Matrix dan ROC Analysis di Orange,
# Gambar 22-27)
mA = DecisionTreeClassifier(random_state=RS, **gridA.best_params_)
pA = cross_val_predict(mA, XA, y, cv=skf10)
cmA = confusion_matrix(y, pA)
 
def roc_ovr(mdl, X):
    proba = cross_val_predict(mdl, X, y, cv=skf10, method="predict_proba")
    Yb = label_binarize(y, classes=range(5))
    aucs = []
    for k in range(5):
        fpr, tpr, _ = roc_curve(Yb[:, k], proba[:, k])
        aucs.append(auc(fpr, tpr))
    return aucs, proba
aucA, probaA = roc_ovr(mA, XA)
print("AUC per kelas (Skenario 1):", [round(a, 3) for a in aucA])
 
# --- uji permutasi: apakah akurasi Skenario 1 berbeda dari acak? ----
# (Subbab 4.4.4, Gambar 28) - label diacak 200 kali, model dilatih
# ulang dengan cross-validation yang sama; p-value = proporsi hasil
# acak yang menyamai/melampaui akurasi asli.
permA = DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=RS)
obs = cross_val_score(permA, XA, y, cv=skf5).mean()
rng = np.random.RandomState(RS)
null = []
for _ in range(200):
    y_shuffled = rng.permutation(y)
    null.append(cross_val_score(permA, XA, y_shuffled, cv=skf5).mean())
null = np.array(null)
p_value = (np.sum(null >= obs) + 1) / (len(null) + 1)
print(f"Akurasi asli={obs:.4f}  rerata acak={null.mean():.4f}  p={p_value:.3f}")
 
# --- kalibrasi probabilitas (ECE) dan kurva lift (Gambar 44-45) -----
def calibration_ece(proba, y_true, n_bins=10):
    allp = proba.ravel(); ally = np.eye(5)[y_true].ravel()
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    b = np.digitize(allp, bins[1:-1])
    for bi in range(n_bins):
        m = b == bi
        if m.sum():
            ece += m.mean() * abs(allp[m].mean() - ally[m].mean())
    return ece
print("Expected Calibration Error (Skenario 1):",
      round(calibration_ece(probaA, y), 4))
# A.5 Rekonstruksi Visualisasi Bergaya Orange Data Mining
# Inti komputasi di balik visualisasi pada Subbab 4.6, yaitu PCA, t-SNE, MDS, dendrogram (jarak Jaccard), k-Means dengan silhouette plot, tabel korelasi, pivot table, serta rekursi pembagian persegi pada pohon dan hutan Pythagoras. Kode kosmetik untuk pewarnaan dan anotasi gambar diringkas.
# Rekonstruksi sejumlah widget Orange yang tidak dapat dijalankan
# langsung pada lingkungan ini (lihat Lampiran C untuk langkah aslinya
# di aplikasi Orange). Bagian ini menunjukkan inti komputasinya saja;
# kode plotting kosmetik (warna, anotasi) diringkas.
 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
 
ZA = StandardScaler().fit_transform(XA.values.astype(float))
 
# -- PCA (Gambar 39): scree plot & proyeksi dua komponen utama -------
pca = PCA().fit(ZA)
ev  = pca.explained_variance_ratio_          # proporsi varians per PC
Pz  = pca.transform(ZA)
k90 = int(np.searchsorted(np.cumsum(ev), .9) + 1)
print(f"Komponen untuk 90% varians: {k90} dari {len(ev)} (tanpa struktur kuat)")
 
# -- t-SNE (Gambar 39): proyeksi non-linear metadata vs kata judul ---
ts_a = TSNE(2, perplexity=30, random_state=RS, init="pca").fit_transform(ZA)
ts_b = TSNE(2, perplexity=30, random_state=RS, init="pca"
            ).fit_transform(XB.values.astype(float))
 
# -- MDS & Dendrogram (Gambar 39): 20 headline unik, jarak Jaccard ---
uh = df.drop_duplicates("headline")[["headline", "topic_category"]]
Uw = pd.DataFrame(cv_words.transform(uh.headline).toarray().astype(bool))
Dmat = squareform(pdist(Uw.values, "jaccard"))
mds = MDS(2, dissimilarity="precomputed", random_state=RS,
          n_init=8, init="random").fit(Dmat)
L  = linkage(pdist(Uw.values, "jaccard"), "average")
uy = uh.topic_category.map({c: i for i, c in enumerate(CATS)}).values
cl = fcluster(L, 5, "maxclust")
print("ARI dendrogram vs kelas asli:", round(adjusted_rand_score(uy, cl), 3))
 
# -- k-Means + Silhouette Plot (Gambar 40) ----------------------------
for Xk, nm in [(ZA, "Skenario 1"), (XB.values.astype(float), "Skenario 2")]:
    km  = KMeans(5, n_init=10, random_state=RS).fit(Xk)
    sil = silhouette_samples(Xk, km.labels_)
    ari = adjusted_rand_score(y, km.labels_)
    print(f"{nm:12s} rerata siluet={sil.mean():.3f}  ARI vs kelas={ari:.3f}")
 
# -- Correlations (Gambar 41): 12 pasangan |r| terbesar antar fitur --
nmA = list(XA.columns); cm_ = XA.corr().values
grp = lambda c: c.split("_")[0]
pairs = [(abs(cm_[i, j]), cm_[i, j], nmA[i], nmA[j])
         for i in range(len(nmA)) for j in range(i + 1, len(nmA))
         if grp(nmA[i]) != grp(nmA[j])]
pairs.sort(reverse=True)
for _, r, a, b in pairs[:5]:
    print(f"{a} x {b}: r = {r:+.3f}")
 
# -- Pivot Table (Gambar 42): rerata engagement per hari x kelas -----
pv = df.pivot_table(index="weekday", columns="topic_category",
                     values="engagement_score", aggfunc="mean")[CATS]
print(pv.round(1))
 
# -- Pohon dan Hutan Pythagoras (Gambar 15-18, 43): rekursi luas ------
# persegi tiap simpul, sisi anak sebanding akar(jumlah_sampel anak).
def draw_pythagoras(tree_, node, B0, B1, depth, ax, max_depth=7):
    B0, B1 = np.array(B0, float), np.array(B1, float)
    u = B1 - B0; s = np.linalg.norm(u); nv = np.array([-u[1], u[0]])
    T0, T1 = B0 + nv, B1 + nv
    ax.add_patch(Polygon([B0, B1, T1, T0], closed=True))   # gambar persegi simpul
    left, right = tree_.children_left[node], tree_.children_right[node]
    if left == -1 or depth >= max_depth:
        return
    nl = tree_.n_node_samples[left]; nr = tree_.n_node_samples[right]
    a = s * math.sqrt(nl / (nl + nr)); b = s * math.sqrt(nr / (nl + nr))
    Q = T0 + (a * a / s) * (u / s) + (a * b / s) * (nv / s)
    draw_pythagoras(tree_, left,  T0, Q,  depth + 1, ax, max_depth)
    draw_pythagoras(tree_, right, Q,  T1, depth + 1, ax, max_depth)
