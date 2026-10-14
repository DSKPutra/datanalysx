# -*- coding: utf-8 -*-
"""
=====================================================================
 Studi Kasus 07 — K-Nearest Neighbors (KNN)
 Klasifikasi "Viral Hit" vs "High Reach" pada Instagram Reels 2026
---------------------------------------------------------------------
 Penulis     : Dea Saka Kurnia Putra (2026)
---------------------------------------------------------------------
 Alur mengikuti CRISP-DM:
   1. Business Understanding   -> definisi masalah & target
   2. Data Understanding       -> struktur, missing, duplikat, outlier, EDA
   3. Data Preparation         -> rekayasa fitur, encoding, capping IQR, scaling
   4. Modeling                 -> KNN + pemilihan K + GridSearchCV
   5. Evaluation               -> metrik, baseline, pembanding, analisis lanjutan
   6. Deployment (rekomendasi) -> implikasi bagi kreator/marketer
 Seluruh angka yang dilaporkan disimpan ke outputs/model_summary.json.
 Cara menjalankan:  python knn_instagram_reels.py
 (butuh: pandas, numpy, scikit-learn>=1.3, matplotlib, seaborn)
=====================================================================
"""

# =====================================================================
# 0. IMPORT, KONFIGURASI, DAN TEMA GRAFIK
# =====================================================================
import os, re, json, time, zipfile, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import (train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
                                     GridSearchCV, cross_validate, learning_curve, validation_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             balanced_accuracy_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, average_precision_score)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ZIP = os.path.join(BASE, "Instagram Reels- Virality  Engagement Trends 2026.zip")
DATA_CSV = os.path.join(BASE, "instagram_reels_clean.csv")
OUT = os.path.join(BASE, "outputs")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# Palet tema hijau zamrud – hitam – emas
EMERALD, DEEP, BLACK, GOLD = "#0B6E4F", "#064E3B", "#111111", "#C9A227"
MINT, SAGE, GREY, LIGHT = "#52B788", "#B7E4C7", "#8A8A8A", "#EEF6F1"
CLS_COL = {0: GREY, 1: EMERALD}        # 0 = High Reach, 1 = Viral Hit
CLS_NAME = {0: "High Reach", 1: "Viral Hit"}
CMAP_G = LinearSegmentedColormap.from_list("gk", ["#E8F5EE", SAGE, MINT, EMERALD, DEEP, BLACK])
CMAP_DIV = LinearSegmentedColormap.from_list("div", [BLACK, "#555555", "#FFFFFF", MINT, DEEP])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.edgecolor": "#333333", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})
SUMMARY = {}          # semua angka hasil eksekusi dikumpulkan di sini
FIG_LIST = []


def savefig(name, title):
    """Simpan grafik ke outputs/figures dan catat judulnya."""
    path = os.path.join(FIG, name + ".png")
    plt.savefig(path, facecolor="white")
    plt.close("all")
    FIG_LIST.append({"file": name + ".png", "title": title})


def r(x, n=4):
    """Pembulatan aman untuk JSON."""
    return None if x is None else float(round(float(x), n))


# =====================================================================
# 1. BUSINESS UNDERSTANDING
# ---------------------------------------------------------------------
# Pertanyaan bisnis: dapatkah atribut yang SUDAH diketahui kreator sebelum
# / saat unggah (durasi video, panjang caption, jumlah hashtag, niche, jam &
# hari unggah, usia unggahan) dipakai untuk memprediksi apakah sebuah Reel
# akan masuk tier "Viral Hit" (virality_score >= 75)?
# =====================================================================

# =====================================================================
# 2. DATA UNDERSTANDING
# =====================================================================
# 2.1 Ekstraksi ZIP & identifikasi file CSV
if os.path.exists(DATA_ZIP):
    with zipfile.ZipFile(DATA_ZIP) as z:
        members = z.namelist()
        z.extractall(BASE)
    SUMMARY["zip_members"] = members
df_raw = pd.read_csv(DATA_CSV)

# 2.2 Struktur data
du = {
    "n_rows": int(df_raw.shape[0]), "n_cols": int(df_raw.shape[1]),
    "dtypes": {c: str(t) for c, t in df_raw.dtypes.items()},
    "missing_total": int(df_raw.isna().sum().sum()),
    "missing_per_col": {c: int(v) for c, v in df_raw.isna().sum().items()},
    "duplicate_rows": int(df_raw.duplicated().sum()),
    "duplicate_post_id": int(df_raw["post_id"].duplicated().sum()),
    "duplicate_shortcode": int(df_raw["shortcode"].duplicated().sum()),
    "n_creators": int(df_raw["creator_handle"].nunique()),
    "n_niches": int(df_raw["niche"].nunique()),
    "ts_min": str(df_raw["timestamp_utc"].min()), "ts_max": str(df_raw["timestamp_utc"].max()),
    "tier_counts": df_raw["virality_tier"].value_counts().to_dict(),
    "niche_counts": df_raw["niche"].value_counts().to_dict(),
    "creator_counts": df_raw["creator_handle"].value_counts().to_dict(),
}
num_raw = ["video_duration_sec", "caption_word_count", "hashtag_count", "views", "likes",
           "comments", "engagement_rate_pct", "virality_score"]
desc = df_raw[num_raw].describe().T
du["describe"] = {c: {k: r(desc.loc[c, k], 3) for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]}
                  for c in num_raw}
# 2.3 Deteksi outlier metode IQR (1,5 x IQR) pada kolom numerik mentah
iqr_rep = {}
for c in num_raw:
    q1, q3 = df_raw[c].quantile([.25, .75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    n_out = int(((df_raw[c] < lo) | (df_raw[c] > hi)).sum())
    iqr_rep[c] = {"q1": r(q1, 3), "q3": r(q3, 3), "lower": r(lo, 3), "upper": r(hi, 3),
                  "n_outlier": n_out, "pct": r(100 * n_out / len(df_raw), 2)}
du["iqr_outliers"] = iqr_rep
SUMMARY["data_understanding"] = du

# =====================================================================
# 3. DATA PREPARATION
# =====================================================================
df = df_raw.copy()

# 3.1 Definisi target (biner). Pada data ini hanya ada 2 tier yang muncul
# (min virality_score = 51,4), sehingga target alami: Viral Hit (skor>=75) = 1,
# High Reach (50–74,9) = 0.
df["target"] = (df["virality_tier"] == "Viral Hit").astype(int)
assert (df["target"] == (df["virality_score"] >= 75)).all()

# 3.2 CEGAH DATA LEAKAGE: kolom pembentuk target TIDAK boleh menjadi fitur.
# virality_score = f(views, engagement_rate); engagement_rate = f(likes, comments, views)
LEAKAGE_COLS = ["views", "likes", "comments", "engagement_rate_pct", "virality_score", "virality_tier"]
ID_COLS = ["post_id", "shortcode", "caption", "timestamp_utc", "creator_handle"]

# 3.3 Rekayasa fitur (hanya informasi yang tersedia saat/ sebelum unggah)
ts = pd.to_datetime(df["timestamp_utc"], utc=True)
ts_local = ts.dt.tz_convert("Asia/Kolkata")         # kreator mayoritas berbasis di India (IST)
ref_date = ts.max()                                   # tanggal pengambilan data (unggahan terbaru)
df["post_hour"] = ts_local.dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["post_hour"] / 24)   # encoding siklik: jam 23 dekat jam 0
df["hour_cos"] = np.cos(2 * np.pi * df["post_hour"] / 24)
df["day_of_week"] = ts_local.dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["post_age_days"] = (ref_date - ts).dt.total_seconds() / 86400.0
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")
df["caption_char_len"] = df["caption"].str.len()
df["emoji_count"] = df["caption"].apply(lambda s: len(EMOJI_RE.findall(s)))
df["mention_count"] = df["caption"].str.count(r"@\w+")
df["has_question"] = df["caption"].str.contains(r"\?").astype(int)

NUM_FEATS = ["video_duration_sec", "caption_word_count", "caption_char_len", "hashtag_count",
             "emoji_count", "mention_count", "post_age_days", "hour_sin", "hour_cos"]
BIN_FEATS = ["is_weekend", "has_question"]
CAT_FEATS = ["niche"]
FEATURES = NUM_FEATS + BIN_FEATS + CAT_FEATS
assert not set(FEATURES) & set(LEAKAGE_COLS)

SUMMARY["preparation"] = {
    "target_def": "Viral Hit (virality_score >= 75) = 1; High Reach = 0",
    "leakage_cols_excluded": LEAKAGE_COLS, "id_cols_excluded": ID_COLS,
    "num_feats": NUM_FEATS, "bin_feats": BIN_FEATS, "cat_feats": CAT_FEATS,
    "n_features_raw": len(FEATURES), "ref_date": str(ref_date),
    "class_counts": {CLS_NAME[k]: int(v) for k, v in df["target"].value_counts().items()},
    "class_pct": {CLS_NAME[k]: r(100 * v, 2) for k, v in df["target"].value_counts(normalize=True).items()},
    "imbalance_ratio": r(df["target"].value_counts().max() / df["target"].value_counts().min(), 3),
    "corr_leak_with_target": {c: r(df[c].corr(df["target"]), 3) for c in
                              ["views", "likes", "comments", "engagement_rate_pct", "virality_score"]},
    "feature_describe": {c: {k: r(v, 3) for k, v in df[c].describe().items()} for c in NUM_FEATS + BIN_FEATS},
}
# Statistik per kelas (dipakai dalam narasi laporan)
SUMMARY["preparation"]["median_by_class"] = {c: {CLS_NAME[k]: r(v, 2) for k, v in df.groupby("target")[c].median().items()}
                                             for c in ["post_hour", "caption_word_count", "video_duration_sec", "hashtag_count", "post_age_days"]}
SUMMARY["preparation"]["corr_word_char"] = r(df["caption_word_count"].corr(df["caption_char_len"]), 3)
_dw = pd.crosstab(df["day_of_week"], df["target"]).reindex(range(7), fill_value=0)
SUMMARY["preparation"]["day_counts"] = {["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][i]:
                                        {"High Reach": int(_dw.loc[i, 0]), "Viral Hit": int(_dw.loc[i, 1])} for i in range(7)}
SUMMARY["preparation"]["top4_creators_share"] = int(df["creator_handle"].value_counts().head(4).sum())
# IQR pada fitur hasil rekayasa (informasi untuk capping)
SUMMARY["preparation"]["iqr_features"] = {}
for c in NUM_FEATS:
    q1, q3 = df[c].quantile([.25, .75]); lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    SUMMARY["preparation"]["iqr_features"][c] = int(((df[c] < lo) | (df[c] > hi)).sum())


class IQRCapper(BaseEstimator, TransformerMixin):
    """Winsorization metode IQR. Batas dihitung HANYA dari data latih (fit),
    lalu diterapkan ke data uji (transform) -> tidak ada kebocoran informasi."""
    def __init__(self, k=1.5):
        self.k = k
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
        self.lo_, self.hi_ = q1 - self.k * (q3 - q1), q3 + self.k * (q3 - q1)
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def make_prep(scaler="standard"):
    """ColumnTransformer: numerik -> IQR capping -> scaling; biner -> passthrough;
    kategorikal -> one-hot. scaler: 'none' | 'standard' | 'minmax'."""
    sc = {"none": FunctionTransformer(feature_names_out="one-to-one"),
          "standard": StandardScaler(), "minmax": MinMaxScaler()}[scaler]
    return ColumnTransformer([
        ("num", Pipeline([("cap", IQRCapper(1.5)), ("scaler", sc)]), NUM_FEATS),
        ("bin", "passthrough", BIN_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS),
    ])


def make_knn(scaler="standard", **kw):
    return Pipeline([("prep", make_prep(scaler)), ("knn", KNeighborsClassifier(**kw))])


X, y = df[FEATURES], df["target"].values

# 3.4 Train–test split terstratifikasi (75:25)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
SUMMARY["split"] = {"n_train": int(len(X_train)), "n_test": int(len(X_test)),
                    "train_pos": int(y_train.sum()), "test_pos": int(y_test.sum()),
                    "train_pos_pct": r(100 * y_train.mean(), 2), "test_pos_pct": r(100 * y_test.mean(), 2)}
CV_TR = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)   # untuk kurva
CV_GS = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)    # untuk GridSearch

# =====================================================================
# 2.x  EXPLORATORY DATA ANALYSIS (GRAFIK)
# =====================================================================
# G01 Distribusi target + virality score
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
cnt = df["target"].value_counts().sort_index()
bars = ax[0].bar([CLS_NAME[i] for i in cnt.index], cnt.values, color=[GREY, EMERALD], edgecolor=BLACK)
for b, v in zip(bars, cnt.values):
    ax[0].text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v} ({100*v/len(df):.1f}%)", ha="center", fontweight="bold")
ax[0].set_title("Distribusi Kelas Target"); ax[0].set_ylabel("Jumlah Reels"); ax[0].set_ylim(0, cnt.max() * 1.2)
ax[1].hist(df["virality_score"], bins=18, color=MINT, edgecolor=DEEP)
ax[1].axvline(75, color=GOLD, lw=2.5, ls="--", label="Ambang Viral Hit = 75")
ax[1].set_title("Distribusi virality_score"); ax[1].set_xlabel("virality_score"); ax[1].set_ylabel("Frekuensi"); ax[1].legend()
savefig("G01_distribusi_target", "Distribusi kelas target dan virality_score dengan ambang 75")

# G02 Histogram fitur numerik per kelas
feats_plot = ["video_duration_sec", "caption_word_count", "caption_char_len", "hashtag_count",
              "emoji_count", "mention_count", "post_age_days", "post_hour"]
fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
for a, c in zip(axes.ravel(), feats_plot):
    for k in [0, 1]:
        a.hist(df.loc[df.target == k, c], bins=12, alpha=0.65, color=CLS_COL[k], edgecolor="white", label=CLS_NAME[k])
    a.set_title(c, fontsize=10)
axes[0, 0].legend(fontsize=8)
plt.suptitle("Histogram Fitur Numerik per Kelas", fontweight="bold", y=1.01)
plt.tight_layout()
savefig("G02_histogram_fitur", "Histogram fitur numerik menurut kelas target")

# G03 Boxplot fitur per kelas (outlier IQR terlihat sebagai titik)
fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
for a, c in zip(axes.ravel(), feats_plot):
    data = [df.loc[df.target == k, c] for k in [0, 1]]
    bp = a.boxplot(data, patch_artist=True, widths=0.55, flierprops=dict(marker="o", mfc=GOLD, mec=BLACK, ms=5))
    for p, k in zip(bp["boxes"], [0, 1]):
        p.set_facecolor(CLS_COL[k]); p.set_alpha(.8)
    for m in bp["medians"]:
        m.set_color(BLACK); m.set_linewidth(2)
    a.set_xticks([1, 2]); a.set_xticklabels(["High Reach", "Viral Hit"], fontsize=8); a.set_title(c, fontsize=10)
plt.suptitle("Boxplot Fitur per Kelas (titik emas = outlier IQR 1,5×)", fontweight="bold", y=1.01)
plt.tight_layout()
savefig("G03_boxplot_fitur", "Boxplot fitur numerik per kelas dan outlier IQR")

# G04 Heatmap korelasi (fitur + kolom leakage + target)
corr_cols = NUM_FEATS + BIN_FEATS + ["views", "likes", "comments", "engagement_rate_pct", "virality_score", "target"]
cm_ = df[corr_cols].corr()
plt.figure(figsize=(10.5, 8.5))
sns.heatmap(cm_, cmap=CMAP_DIV, center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=.4, linecolor="white", cbar_kws={"shrink": .75})
plt.title("Heatmap Korelasi Pearson: Fitur, Kolom Leakage, dan Target")
savefig("G04_heatmap_korelasi", "Heatmap korelasi fitur, kolom leakage, dan target")
SUMMARY["preparation"]["corr_feat_target"] = {c: r(cm_.loc[c, "target"], 3) for c in NUM_FEATS + BIN_FEATS}

# G05 Niche & hari unggah vs target
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
ct = pd.crosstab(df["niche"], df["target"]).rename(columns=CLS_NAME)
ct.plot(kind="barh", stacked=True, color=[GREY, EMERALD], ax=ax[0], edgecolor=BLACK)
ax[0].set_title("Niche vs Kelas Target"); ax[0].set_xlabel("Jumlah Reels"); ax[0].set_ylabel("")
days = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
ct2 = pd.crosstab(df["day_of_week"], df["target"]).reindex(range(7), fill_value=0).rename(columns=CLS_NAME)
ct2.index = days
ct2.plot(kind="bar", color=[GREY, EMERALD], ax=ax[1], edgecolor=BLACK, rot=0)
ax[1].set_title("Hari Unggah (IST) vs Kelas Target"); ax[1].set_ylabel("Jumlah Reels")
plt.tight_layout()
savefig("G05_niche_hari", "Sebaran kelas target menurut niche dan hari unggah")
SUMMARY["preparation"]["niche_viral_rate"] = {k: r(v, 3) for k, v in df.groupby("niche")["target"].mean().items()}

# =====================================================================
# 3.5 PERBANDINGAN SCALING (tanpa vs StandardScaler vs MinMaxScaler)
# =====================================================================
Xtr_none = make_prep("none").fit_transform(X_train)
Xtr_std = make_prep("standard").fit_transform(X_train)
Xtr_mm = make_prep("minmax").fit_transform(X_train)
n_num = len(NUM_FEATS)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for a, M, t in zip(axes, [Xtr_none, Xtr_std, Xtr_mm], ["Tanpa Scaling (setelah capping)", "StandardScaler", "MinMaxScaler"]):
    bp = a.boxplot(M[:, :n_num], vert=False, patch_artist=True, widths=.6,
                   flierprops=dict(marker="o", mfc=GOLD, ms=3))
    for p in bp["boxes"]:
        p.set_facecolor(MINT)
    for m in bp["medians"]:
        m.set_color(BLACK)
    a.set_yticks(range(1, n_num + 1)); a.set_yticklabels(NUM_FEATS if a is axes[0] else [""] * n_num, fontsize=8)
    a.set_title(t, fontsize=11)
plt.suptitle("Rentang Fitur Numerik Sebelum dan Sesudah Scaling (data latih)", fontweight="bold", y=1.02)
plt.tight_layout()
savefig("G06_sebelum_sesudah_scaling", "Rentang fitur numerik sebelum dan sesudah scaling")
SUMMARY["scaling_ranges"] = {"none_min": r(Xtr_none[:, :n_num].min(), 3), "none_max": r(Xtr_none[:, :n_num].max(), 3),
                             "range_ratio_none": r((Xtr_none[:, :n_num].max(0) - Xtr_none[:, :n_num].min(0)).max() /
                                                   max(1e-9, (Xtr_none[:, :n_num].max(0) - Xtr_none[:, :n_num].min(0)).min()), 2)}

scale_res = {}
for s in ["none", "standard", "minmax"]:
    # (a) K=5 default, (b) K terbaik hasil tuning K ganjil 1..49 dengan metric euclidean-uniform
    cv5 = cross_validate(make_knn(s, n_neighbors=5), X_train, y_train, cv=CV_TR,
                         scoring=["accuracy", "f1", "roc_auc"])
    gs = GridSearchCV(make_knn(s), {"knn__n_neighbors": list(range(1, 50, 2))}, cv=CV_TR, scoring="accuracy").fit(X_train, y_train)
    yp = gs.predict(X_test)
    scale_res[s] = {"cv_acc_k5": r(cv5["test_accuracy"].mean()), "cv_acc_k5_std": r(cv5["test_accuracy"].std()),
                    "cv_f1_k5": r(cv5["test_f1"].mean()), "cv_auc_k5": r(cv5["test_roc_auc"].mean()),
                    "best_k": int(gs.best_params_["knn__n_neighbors"]), "cv_acc_bestk": r(gs.best_score_),
                    "test_acc_bestk": r(accuracy_score(y_test, yp)), "test_f1_bestk": r(f1_score(y_test, yp))}
SUMMARY["scaling_comparison"] = scale_res
fig, ax = plt.subplots(figsize=(9, 4))
labels = ["Tanpa Scaling", "StandardScaler", "MinMaxScaler"]
xs = np.arange(3); w = .26
v1 = [scale_res[s]["cv_acc_k5"] for s in scale_res]; e1 = [scale_res[s]["cv_acc_k5_std"] for s in scale_res]
v2 = [scale_res[s]["cv_acc_bestk"] for s in scale_res]; v3 = [scale_res[s]["test_acc_bestk"] for s in scale_res]
for off, v, col, lab in [(-w, v1, GREY, "CV akurasi (K=5)"), (0, v2, EMERALD, "CV akurasi (K terbaik)"), (w, v3, BLACK, "Akurasi uji (K terbaik)")]:
    bb = ax.bar(xs + off, v, w, color=col, label=lab, edgecolor=BLACK, yerr=e1 if off == -w else None, capsize=3)
    for b, vv in zip(bb, v):
        ax.text(b.get_x() + b.get_width() / 2, vv + .02, f"{vv:.3f}", ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n(K*={scale_res[s]['best_k']})" for l, s in zip(labels, scale_res)])
ax.set_ylim(0, 1.05); ax.set_ylabel("Akurasi"); ax.legend(loc="lower right", fontsize=8)
ax.set_title("Pengaruh Feature Scaling terhadap Kinerja KNN")
savefig("G07_perbandingan_scaling", "Perbandingan akurasi KNN tanpa scaling, StandardScaler, dan MinMaxScaler")

# =====================================================================
# 4. MODELING — PEMILIHAN K
# =====================================================================
# 4.1 Kurva error vs K (K ganjil 1–49) dengan CV berulang pada data latih
K_RANGE = list(range(1, 50, 2))
curve = {"k": K_RANGE}
for wgt in ["uniform", "distance"]:
    tr_s, va_s = validation_curve(make_knn("standard", weights=wgt), X_train, y_train,
                                  param_name="knn__n_neighbors", param_range=K_RANGE, cv=CV_TR, scoring="accuracy")
    curve[f"train_acc_{wgt}"] = [r(v) for v in tr_s.mean(1)]
    curve[f"cv_acc_{wgt}"] = [r(v) for v in va_s.mean(1)]
    curve[f"cv_std_{wgt}"] = [r(v) for v in va_s.std(1)]
test_err = []
for k in K_RANGE:
    m = make_knn("standard", n_neighbors=k).fit(X_train, y_train)
    test_err.append(r(1 - accuracy_score(y_test, m.predict(X_test))))
curve["test_err_uniform"] = test_err
best_k_elbow = K_RANGE[int(np.argmax(curve["cv_acc_uniform"]))]
curve["best_k_uniform"] = best_k_elbow
curve["best_cv_acc_uniform"] = max(curve["cv_acc_uniform"])
curve["cv_acc_k1_uniform"] = curve["cv_acc_uniform"][0]
curve["train_acc_k1_uniform"] = curve["train_acc_uniform"][0]
curve["cv_acc_k49_uniform"] = curve["cv_acc_uniform"][-1]
SUMMARY["k_curve"] = curve

fig, ax = plt.subplots(figsize=(10, 4.4))
cv_err = 1 - np.array(curve["cv_acc_uniform"]); sd = np.array(curve["cv_std_uniform"])
ax.plot(K_RANGE, 1 - np.array(curve["train_acc_uniform"]), "o--", color=GREY, ms=4, label="Error latih (uniform)")
ax.plot(K_RANGE, cv_err, "o-", color=EMERALD, ms=5, lw=2, label="Error validasi CV (uniform)")
ax.fill_between(K_RANGE, cv_err - sd, cv_err + sd, color=SAGE, alpha=.35, label="±1 SD CV")
ax.plot(K_RANGE, 1 - np.array(curve["cv_acc_distance"]), "s-", color=BLACK, ms=4, lw=1.3, label="Error validasi CV (distance)")
ax.plot(K_RANGE, test_err, "^:", color=GOLD, ms=5, label="Error uji (informasi, bukan untuk seleksi)")
ax.axvline(best_k_elbow, color=GOLD, lw=1.5, ls="--")
ax.annotate(f"K optimal (CV) = {best_k_elbow}", (best_k_elbow, cv_err.min()), xytext=(best_k_elbow + 4, cv_err.min() - 0.08),
            arrowprops=dict(arrowstyle="->", color=BLACK), fontweight="bold")
ax.set_xlabel("K (jumlah tetangga)"); ax.set_ylabel("Error rate = 1 − akurasi"); ax.set_xticks(K_RANGE[::2])
ax.set_title("Kurva Error vs K (Metode Elbow), StandardScaler + Euclidean"); ax.legend(fontsize=8, ncol=2)
savefig("G08_error_vs_k", "Kurva error rate terhadap K (elbow)")

# 4.2 GridSearchCV: n_neighbors × weights × metric × scaler
param_grid = [
    {"prep__num__scaler": [StandardScaler(), MinMaxScaler()], "knn__n_neighbors": K_RANGE,
     "knn__weights": ["uniform", "distance"], "knn__metric": ["euclidean", "manhattan", "chebyshev", "cosine"],
     "knn__algorithm": ["brute"]},
    {"prep__num__scaler": [StandardScaler(), MinMaxScaler()], "knn__n_neighbors": K_RANGE,
     "knn__weights": ["uniform", "distance"], "knn__metric": ["minkowski"], "knn__p": [3], "knn__algorithm": ["brute"]},
]
t0 = time.time()
gs = GridSearchCV(make_knn("standard"), param_grid, cv=CV_GS, n_jobs=-1,
                  scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}, refit="accuracy")
gs.fit(X_train, y_train)
gs_time = time.time() - t0
res = pd.DataFrame(gs.cv_results_)
res["scaler"] = res["param_prep__num__scaler"].apply(lambda s: type(s).__name__)
res["metric"] = res.apply(lambda rr: "minkowski(p=3)" if rr["param_knn__metric"] == "minkowski" else rr["param_knn__metric"], axis=1)
bp_ = gs.best_params_
best_scaler = type(bp_["prep__num__scaler"]).__name__
best_metric = bp_["knn__metric"]; best_p = bp_.get("knn__p", 2)
best_k = int(bp_["knn__n_neighbors"]); best_w = bp_["knn__weights"]
top10 = res.sort_values("rank_test_accuracy").head(10)
SUMMARY["gridsearch"] = {
    "n_candidates": int(len(res)), "n_splits_total": int(CV_GS.get_n_splits()), "n_fits": int(len(res) * CV_GS.get_n_splits()),
    "time_sec": r(gs_time, 2), "best_scaler": best_scaler, "best_metric": best_metric, "best_p": int(best_p),
    "best_k": best_k, "best_weights": best_w, "best_cv_acc": r(gs.best_score_),
    "best_cv_acc_std": r(res.loc[gs.best_index_, "std_test_accuracy"]),
    "best_cv_f1": r(res.loc[gs.best_index_, "mean_test_f1"]), "best_cv_auc": r(res.loc[gs.best_index_, "mean_test_roc_auc"]),
    "top10": [{"scaler": t.scaler, "metric": t.metric, "k": int(t.param_knn__n_neighbors), "weights": t.param_knn__weights,
               "cv_acc": r(t.mean_test_accuracy), "cv_std": r(t.std_test_accuracy), "cv_f1": r(t.mean_test_f1),
               "cv_auc": r(t.mean_test_roc_auc)} for t in top10.itertuples()],
    "best_by_metric": {m: r(g["mean_test_accuracy"].max()) for m, g in res.groupby("metric")},
    "best_by_scaler": {m: r(g["mean_test_accuracy"].max()) for m, g in res.groupby("scaler")},
    "best_by_weights": {m: r(g["mean_test_accuracy"].max()) for m, g in res.groupby("param_knn__weights")},
}
# G09 Heatmap GridSearch (K × metric) untuk scaler & weights terbaik
sub = res[(res.scaler == best_scaler) & (res.param_knn__weights == best_w)]
hm = sub.pivot_table(index="metric", columns="param_knn__n_neighbors", values="mean_test_accuracy")
plt.figure(figsize=(13, 3.6))
sns.heatmap(hm, cmap=CMAP_G, annot=True, fmt=".2f", annot_kws={"size": 7}, linewidths=.4, linecolor="white",
            cbar_kws={"label": "CV akurasi"})
plt.title(f"Heatmap GridSearchCV: K × Metric (scaler={best_scaler}, weights={best_w})")
plt.xlabel("K (n_neighbors)"); plt.ylabel("Metric jarak")
savefig("G09_heatmap_gridsearch", "Heatmap hasil GridSearchCV K × metric jarak")

best_model = gs.best_estimator_

# =====================================================================
# 5. EVALUATION
# =====================================================================
def evaluate(model, name):
    yp = model.predict(X_test); pp = model.predict_proba(X_test)[:, 1]
    return {"model": name, "accuracy": r(accuracy_score(y_test, yp)), "precision": r(precision_score(y_test, yp, zero_division=0)),
            "recall": r(recall_score(y_test, yp)), "f1": r(f1_score(y_test, yp)),
            "balanced_acc": r(balanced_accuracy_score(y_test, yp)),
            "roc_auc": r(roc_auc_score(y_test, pp)), "avg_precision": r(average_precision_score(y_test, pp))}, yp, pp

knn_eval, yp_knn, pp_knn = evaluate(best_model, "KNN (tuned)")
cmat = confusion_matrix(y_test, yp_knn)
knn_eval["confusion_matrix"] = cmat.tolist()
knn_eval["tn"], knn_eval["fp"], knn_eval["fn"], knn_eval["tp"] = [int(v) for v in cmat.ravel()]
knn_eval["classification_report"] = classification_report(y_test, yp_knn, target_names=["High Reach", "Viral Hit"], output_dict=True)
knn_eval["specificity"] = r(cmat[0, 0] / cmat[0].sum())
SUMMARY["knn_test"] = knn_eval

# G10 Confusion matrix
fig, ax = plt.subplots(figsize=(4.8, 4.2))
sns.heatmap(cmat, annot=True, fmt="d", cmap=CMAP_G, cbar=False, linewidths=2, linecolor="white", annot_kws={"size": 18, "weight": "bold"},
            xticklabels=["High Reach", "Viral Hit"], yticklabels=["High Reach", "Viral Hit"], ax=ax)
ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title(f"Confusion Matrix KNN (K={best_k}, data uji)")
savefig("G10_confusion_matrix", "Confusion matrix KNN pada data uji")

# 5.1 Baseline & model pembanding (semua dalam Pipeline yang sama)
comparators = {
    "Dummy (most frequent)": Pipeline([("prep", make_prep("standard")), ("clf", DummyClassifier(strategy="most_frequent"))]),
    "Dummy (stratified)": Pipeline([("prep", make_prep("standard")), ("clf", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE))]),
    "Logistic Regression": Pipeline([("prep", make_prep("standard")), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
    "Gaussian Naive Bayes": Pipeline([("prep", make_prep("none")), ("clf", GaussianNB())]),
    "Decision Tree": Pipeline([("prep", make_prep("none")), ("clf", DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE))]),
    "Random Forest": Pipeline([("prep", make_prep("none")), ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]),
    "KNN default (K=5, tanpa scaling)": make_knn("none", n_neighbors=5),
    "KNN (tuned)": clone(best_model),
}
CV_ALL = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
comp_rows, probas = [], {}
for name, mdl in comparators.items():
    mdl.fit(X_train, y_train)
    ev, _, pp = evaluate(mdl, name)
    probas[name] = pp
    t0 = time.time(); mdl.fit(X_train, y_train); fit_t = time.time() - t0
    t0 = time.time(); mdl.predict(X_test); pred_t = time.time() - t0
    cvr = cross_validate(clone(mdl), X, y, cv=CV_ALL, scoring=["accuracy", "f1", "roc_auc"])
    ev.update({"cv_acc": r(cvr["test_accuracy"].mean()), "cv_acc_std": r(cvr["test_accuracy"].std()),
               "cv_f1": r(cvr["test_f1"].mean()), "cv_auc": r(np.nanmean(cvr["test_roc_auc"])),
               "fit_ms": r(1000 * fit_t, 3), "predict_ms": r(1000 * pred_t, 3)})
    comp_rows.append(ev)
SUMMARY["model_comparison"] = comp_rows

# G11 ROC curve
fig, ax = plt.subplots(figsize=(6.2, 5.2))
style = {"KNN (tuned)": (EMERALD, "-", 2.8), "Logistic Regression": (BLACK, "--", 1.6), "Gaussian Naive Bayes": (GOLD, "-.", 1.6),
         "Random Forest": (MINT, ":", 2), "KNN default (K=5, tanpa scaling)": (GREY, "-", 1.3)}
for name, (c, ls, lw) in style.items():
    fpr, tpr, _ = roc_curve(y_test, probas[name])
    ax.plot(fpr, tpr, color=c, ls=ls, lw=lw, label=f"{name} (AUC={roc_auc_score(y_test, probas[name]):.3f})")
ax.plot([0, 1], [0, 1], color="#BBBBBB", ls="--", label="Acak (AUC=0,5)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)"); ax.set_title("Kurva ROC pada Data Uji")
ax.legend(fontsize=7.5, loc="lower right")
savefig("G11_roc_curve", "Kurva ROC KNN dan model pembanding")

# G12 Precision–Recall curve
fig, ax = plt.subplots(figsize=(6.2, 5.2))
for name, (c, ls, lw) in style.items():
    pr, rc, _ = precision_recall_curve(y_test, probas[name])
    ax.step(rc, pr, where="post", color=c, ls=ls, lw=lw, label=f"{name} (AP={average_precision_score(y_test, probas[name]):.3f})")
ax.axhline(y_test.mean(), color="#BBBBBB", ls="--", label=f"Prevalensi = {y_test.mean():.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Kurva Precision–Recall pada Data Uji"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.5, loc="lower left")
savefig("G12_precision_recall", "Kurva precision-recall KNN dan model pembanding")

# =====================================================================
# 5.2 CONTOH PERHITUNGAN MANUAL (1 data uji nyata) + VERIFIKASI assert
# =====================================================================
prep_f = best_model.named_steps["prep"]; knn_f = best_model.named_steps["knn"]
feat_names = list(prep_f.get_feature_names_out())
Xtr_t = prep_f.transform(X_train); Xte_t = prep_f.transform(X_test)
# Dipilih data uji PERTAMA (urutan split) yang K tetangganya tidak seragam kelasnya,
# agar mekanisme voting (berbobot) terlihat jelas; bila tidak ada, dipakai data uji ke-0.
_nb_all = knn_f.kneighbors(Xte_t, return_distance=False)
_mixed = [i for i in range(len(Xte_t)) if len(set(y_train[_nb_all[i]])) > 1]
q_pos = _mixed[0] if _mixed else 0
xq = Xte_t[q_pos]


def dist(a, B, metric, p=2):
    """Rumus jarak manual (lihat Persamaan 2.1–2.5)."""
    if metric == "euclidean":
        return np.sqrt(((B - a) ** 2).sum(1))
    if metric == "manhattan":
        return np.abs(B - a).sum(1)
    if metric == "chebyshev":
        return np.abs(B - a).max(1)
    if metric == "minkowski":
        return (np.abs(B - a) ** p).sum(1) ** (1 / p)
    if metric == "cosine":
        return 1 - (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a))
    raise ValueError(metric)


d_all = dist(xq, Xtr_t, best_metric, best_p)                   # Langkah 1: jarak ke semua data latih
order = np.argsort(d_all, kind="stable")                        # Langkah 2: urutkan
nn_idx = order[:best_k]                                         # Langkah 3: ambil K terdekat
if best_w == "distance":                                        # Langkah 4: voting (berbobot 1/d)
    wts = 1 / np.maximum(d_all[nn_idx], 1e-12)
else:
    wts = np.ones(best_k)
votes = np.array([wts[y_train[nn_idx] == c].sum() for c in [0, 1]])
proba_manual = votes / votes.sum()
pred_manual = int(np.argmax(proba_manual))                      # Langkah 5: kelas dengan suara terbanyak
# Verifikasi terhadap scikit-learn
sk_dist, sk_idx = knn_f.kneighbors(xq.reshape(1, -1))
sk_proba = best_model.predict_proba(X_test.iloc[[q_pos]])[0]
sk_pred = int(best_model.predict(X_test.iloc[[q_pos]])[0])
assert np.allclose(np.sort(d_all[nn_idx]), np.sort(sk_dist[0]), atol=1e-10), "Jarak tetangga berbeda!"
assert set(nn_idx) == set(sk_idx[0]), "Himpunan tetangga berbeda!"
assert np.allclose(proba_manual, sk_proba, atol=1e-10), "Probabilitas berbeda!"
assert pred_manual == sk_pred, "Prediksi berbeda!"
# rincian kontribusi per fitur terhadap jarak ke tetangga terdekat
n1 = nn_idx[0]
diff = xq - Xtr_t[n1]
contrib = (diff ** 2) if best_metric == "euclidean" else (np.abs(diff) if best_metric in ("manhattan", "chebyshev") else
          (np.abs(diff) ** best_p if best_metric == "minkowski" else xq * Xtr_t[n1]))
q_row = X_test.iloc[q_pos]
SUMMARY["worked_example"] = {
    "q_pos": int(q_pos), "n_mixed_test": len(_mixed),
    "test_row_index": int(X_test.index[q_pos]), "shortcode": str(df.loc[X_test.index[q_pos], "shortcode"]),
    "creator": str(df.loc[X_test.index[q_pos], "creator_handle"]),
    "timestamp_utc": str(df.loc[X_test.index[q_pos], "timestamp_utc"]),
    "actual": int(y_test[q_pos]), "actual_label": CLS_NAME[int(y_test[q_pos])],
    "raw_features": {c: (q_row[c] if isinstance(q_row[c], str) else r(q_row[c], 4)) for c in FEATURES},
    "transformed": {f: r(v, 4) for f, v in zip(feat_names, xq)},
    "metric": best_metric, "p": int(best_p), "k": best_k, "weights": best_w,
    "neighbors": [{"rank": i + 1, "train_row_index": int(X_train.index[j]),
                   "shortcode": str(df.loc[X_train.index[j], "shortcode"]),
                   "niche": str(X_train.iloc[j]["niche"]), "duration": r(X_train.iloc[j]["video_duration_sec"], 1),
                   "words": int(X_train.iloc[j]["caption_word_count"]), "hashtags": int(X_train.iloc[j]["hashtag_count"]),
                   "age_days": r(X_train.iloc[j]["post_age_days"], 1),
                   "distance": r(d_all[j], 4), "weight": r(wts[i], 4), "label": CLS_NAME[int(y_train[j])]}
                  for i, j in enumerate(nn_idx)],
    "next_after_k": [{"rank": best_k + 1 + i, "distance": r(d_all[j], 4), "label": CLS_NAME[int(y_train[j])]}
                     for i, j in enumerate(order[best_k:best_k + 3])],
    "nearest_contrib": [{"feature": f, "query": r(a, 4), "neighbor": r(b, 4), "diff": r(a - b, 4), "contrib": r(c, 4)}
                        for f, a, b, c in zip(feat_names, xq, Xtr_t[n1], contrib)],
    "nearest_distance_manual": r(d_all[n1], 6),
    "nearest_dot": r(float(xq @ Xtr_t[n1]), 6), "norm_query": r(float(np.linalg.norm(xq)), 6),
    "norm_neighbor": r(float(np.linalg.norm(Xtr_t[n1])), 6),
    "votes": {"High Reach": r(votes[0], 4), "Viral Hit": r(votes[1], 4)},
    "proba_manual": [r(v, 6) for v in proba_manual], "proba_sklearn": [r(v, 6) for v in sk_proba],
    "pred_manual": CLS_NAME[pred_manual], "pred_sklearn": CLS_NAME[sk_pred], "assert_passed": True,
    "max_abs_diff_proba": float(np.abs(proba_manual - sk_proba).max()),
    "d_min": r(d_all.min(), 4), "d_max": r(d_all.max(), 4), "d_median": r(np.median(d_all), 4),
}

# G17 Ilustrasi K tetangga terdekat (proyeksi PCA 2D ruang fitur ter-scaling)
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
_A = np.vstack([Xtr_t, xq])                        # MDS mempertahankan jarak metric terpilih sebaik mungkin di 2D
_D = pairwise_distances(_A, metric=best_metric) if best_metric != "minkowski" else pairwise_distances(_A, metric="minkowski", p=best_p)
_emb = MDS(n_components=2, metric="precomputed", init="random", random_state=RANDOM_STATE, n_init=4).fit(_D)
P, pq = _emb.embedding_[:-1], _emb.embedding_[-1]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
for c in [0, 1]:
    ax[0].scatter(P[y_train == c, 0], P[y_train == c, 1], s=30, c=CLS_COL[c], alpha=.55, edgecolor="white", label=CLS_NAME[c])
for j in nn_idx:
    ax[0].plot([pq[0], P[j, 0]], [pq[1], P[j, 1]], color=GOLD, lw=.9, alpha=.8, zorder=1)
    ax[0].scatter(P[j, 0], P[j, 1], s=90, facecolor="none", edgecolor=GOLD, lw=1.8, zorder=3)
ax[0].scatter(*pq, marker="*", s=380, c=BLACK, edgecolor=GOLD, lw=1.2, zorder=4, label="Data uji (query)")
ax[0].set_title(f"{best_k} Tetangga Terdekat (lingkar emas), peta MDS jarak {best_metric}"); ax[0].set_xlabel("Dimensi MDS 1"); ax[0].set_ylabel("Dimensi MDS 2")
ax[0].legend(fontsize=8, loc="best")
ds = d_all[order[:min(len(order), best_k + 10)]]
cols = [CLS_COL[int(y_train[j])] for j in order[:len(ds)]]
ax[1].bar(range(1, len(ds) + 1), ds, color=cols, edgecolor=[GOLD if i < best_k else "white" for i in range(len(ds))], lw=1.5)
ax[1].axvline(best_k + .5, color=BLACK, ls="--"); ax[1].text(best_k + .7, ds.max() * .95, f"batas K={best_k}", fontsize=9)
ax[1].set_xlabel("Peringkat tetangga"); ax[1].set_ylabel(f"Jarak {best_metric}"); ax[1].set_title("Jarak Terurut ke Data Latih")
from matplotlib.patches import Patch
ax[1].legend(handles=[Patch(color=GREY, label="High Reach"), Patch(color=EMERALD, label="Viral Hit")], fontsize=8, loc="lower right")
plt.tight_layout()
savefig("G17_ilustrasi_tetangga", "Ilustrasi K tetangga terdekat untuk contoh perhitungan manual")

# =====================================================================
# 5.3 ANALISIS LANJUTAN
# =====================================================================
# G13 Learning curve (seluruh data, CV berulang) — ukuran latih minimal > K
sizes = np.unique(np.linspace(max(best_k + 2, 12), int(len(X) * 0.8), 8).astype(int))
ls_sizes, ls_tr, ls_va = learning_curve(clone(best_model), X, y, train_sizes=sizes, cv=CV_ALL, scoring="accuracy",
                                        shuffle=True, random_state=RANDOM_STATE)
fig, ax = plt.subplots(figsize=(8.5, 4.3))
for s_, c, lab, mk in [(ls_tr, GREY, "Skor latih", "o--"), (ls_va, EMERALD, "Skor validasi CV", "o-")]:
    m_, sd_ = s_.mean(1), s_.std(1)
    ax.plot(ls_sizes, m_, mk, color=c, lw=2, label=lab); ax.fill_between(ls_sizes, m_ - sd_, m_ + sd_, color=c, alpha=.15)
ax.axhline(max(y.mean(), 1 - y.mean()), color=GOLD, ls=":", label="Baseline mayoritas")
ax.set_xlabel("Jumlah data latih"); ax.set_ylabel("Akurasi"); ax.set_title(f"Learning Curve KNN (K={best_k}, {best_w}, {best_metric})")
ax.legend(fontsize=8); ax.set_ylim(0.3, 1.05)
savefig("G13_learning_curve", "Learning curve KNN")
SUMMARY["learning_curve"] = {"sizes": [int(v) for v in ls_sizes], "train": [r(v) for v in ls_tr.mean(1)],
                             "val": [r(v) for v in ls_va.mean(1)], "val_std": [r(v) for v in ls_va.std(1)]}

# G14 Validation curve (K) untuk konfigurasi terbaik
vc_tr, vc_va = validation_curve(clone(best_model), X_train, y_train, param_name="knn__n_neighbors",
                                param_range=K_RANGE, cv=CV_TR, scoring="accuracy")
fig, ax = plt.subplots(figsize=(9, 4.2))
for s_, c, lab, mk in [(vc_tr, GREY, "Skor latih", "o--"), (vc_va, EMERALD, "Skor validasi CV", "o-")]:
    m_, sd_ = s_.mean(1), s_.std(1)
    ax.plot(K_RANGE, m_, mk, color=c, lw=2, ms=4, label=lab); ax.fill_between(K_RANGE, m_ - sd_, m_ + sd_, color=c, alpha=.15)
ax.axvline(best_k, color=GOLD, ls="--", label=f"K terpilih = {best_k}")
ax.text(2, 0.42, "← overfitting\n(varians tinggi)", fontsize=9, color=BLACK)
ax.text(38, 0.42, "underfitting →\n(bias tinggi)", fontsize=9, color=BLACK)
ax.set_xlabel("K"); ax.set_ylabel("Akurasi"); ax.set_ylim(0.35, 1.05); ax.legend(fontsize=8)
ax.set_title(f"Validation Curve terhadap K ({best_scaler}, {best_metric}, {best_w})")
savefig("G14_validation_curve", "Validation curve KNN terhadap K")
SUMMARY["validation_curve"] = {"train": [r(v) for v in vc_tr.mean(1)], "val": [r(v) for v in vc_va.mean(1)]}

# G15 Permutation importance (dirata-rata pada fold validasi CV berulang, tingkat fitur mentah)
imp = {f: [] for f in FEATURES}
for tr_i, va_i in RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=RANDOM_STATE).split(X, y):
    m = clone(best_model).fit(X.iloc[tr_i], y[tr_i])
    pi = permutation_importance(m, X.iloc[va_i], y[va_i], n_repeats=20, random_state=RANDOM_STATE, scoring="accuracy")
    for f, v in zip(FEATURES, pi.importances_mean):
        imp[f].append(v)
imp_df = pd.DataFrame({"feature": FEATURES, "mean": [np.mean(imp[f]) for f in FEATURES],
                       "std": [np.std(imp[f]) for f in FEATURES]}).sort_values("mean")
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.barh(imp_df.feature, imp_df["mean"], xerr=imp_df["std"], color=[EMERALD if v > 0 else GREY for v in imp_df["mean"]],
        edgecolor=BLACK, capsize=3, error_kw={"ecolor": GOLD})
ax.axvline(0, color=BLACK, lw=1); ax.set_xlabel("Penurunan akurasi rata-rata saat fitur diacak")
ax.set_title("Permutation Feature Importance KNN (CV 5×4, 20 pengacakan)")
savefig("G15_permutation_importance", "Permutation feature importance KNN")
SUMMARY["permutation_importance"] = [{"feature": t.feature, "mean": r(t.mean), "std": r(t.std)}
                                     for t in imp_df.sort_values("mean", ascending=False).itertuples()]

# G16 Decision boundary pada ruang PCA 2D
pca_pipe = Pipeline([("prep", clone(best_model.named_steps["prep"])), ("pca", PCA(2, random_state=RANDOM_STATE)),
                     ("knn", clone(knn_f))]).fit(X_train, y_train)
Ptr = pca_pipe[:-1].transform(X_train); Pte = pca_pipe[:-1].transform(X_test)
pad = 0.6
xx, yy = np.meshgrid(np.linspace(min(Ptr[:, 0].min(), Pte[:, 0].min()) - pad, max(Ptr[:, 0].max(), Pte[:, 0].max()) + pad, 300),
                     np.linspace(min(Ptr[:, 1].min(), Pte[:, 1].min()) - pad, max(Ptr[:, 1].max(), Pte[:, 1].max()) + pad, 300))
Z = pca_pipe.named_steps["knn"].predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
db_acc = {}
for a, mt in zip(ax, ["euclidean", best_metric]):
    kk = best_k
    kn = clone(knn_f).set_params(metric=mt).fit(Ptr, y_train)
    Zk = kn.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    a.contourf(xx, yy, Zk, levels=np.linspace(0, 1, 11), cmap=LinearSegmentedColormap.from_list("b", ["#E6E6E6", "#FFFFFF", SAGE, MINT]), alpha=.9)
    a.contour(xx, yy, Zk, levels=[0.5], colors=BLACK, linewidths=1.4)
    for c in [0, 1]:
        a.scatter(Ptr[y_train == c, 0], Ptr[y_train == c, 1], c=CLS_COL[c], s=28, edgecolor="white", label=f"Latih: {CLS_NAME[c]}")
        a.scatter(Pte[y_test == c, 0], Pte[y_test == c, 1], c=CLS_COL[c], s=70, marker="D", edgecolor=GOLD, lw=1.3, label=f"Uji: {CLS_NAME[c]}")
    acc2 = accuracy_score(y_test, kn.predict(Pte)); db_acc[mt] = r(acc2)
    a.set_title(f"Decision Boundary PCA 2D — {mt}, K={kk} (akurasi uji 2D={acc2:.3f})", fontsize=10.5)
    a.set_xlabel(f"PC1 ({100*pca_pipe.named_steps['pca'].explained_variance_ratio_[0]:.1f}% varians)")
    a.set_ylabel(f"PC2 ({100*pca_pipe.named_steps['pca'].explained_variance_ratio_[1]:.1f}% varians)")
ax[0].legend(fontsize=7, loc="best")
plt.tight_layout()
savefig("G16_decision_boundary_pca", "Decision boundary KNN pada ruang PCA 2D")
SUMMARY["pca"] = {"evr": [r(v) for v in pca_pipe.named_steps["pca"].explained_variance_ratio_],
                  "evr_sum": r(pca_pipe.named_steps["pca"].explained_variance_ratio_.sum()),
                  "test_acc_2d_bestk": r(accuracy_score(y_test, pca_pipe.predict(X_test))),
                  "test_acc_2d_by_metric": db_acc}

# G19 Curse of dimensionality
rng = np.random.default_rng(RANDOM_STATE)
noise_dims = [0, 2, 5, 10, 20, 50, 100, 200]
cod = {"noise_dims": noise_dims, "knn": [], "knn_std": [], "knn_euc_std": [], "lr": [], "contrast": []}
Xall_t_base = None
for nd in noise_dims:
    Xn = X.copy()
    for i in range(nd):
        Xn[f"noise_{i}"] = rng.normal(size=len(Xn))
    num_n = NUM_FEATS + [f"noise_{i}" for i in range(nd)]
    prep_n = ColumnTransformer([("num", Pipeline([("cap", IQRCapper()), ("scaler", clone(prep_f.named_transformers_["num"].named_steps["scaler"]))]), num_n),
                                ("bin", "passthrough", BIN_FEATS),
                                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS)])
    knn_n = Pipeline([("prep", prep_n), ("knn", clone(knn_f))])
    lr_n = Pipeline([("prep", clone(prep_n)), ("clf", LogisticRegression(max_iter=3000))])
    s1 = cross_validate(knn_n, Xn, y, cv=CV_ALL, scoring="accuracy")["test_score"]
    s2 = cross_validate(lr_n, Xn, y, cv=CV_ALL, scoring="accuracy")["test_score"]
    prep_e = ColumnTransformer([("num", Pipeline([("cap", IQRCapper()), ("scaler", StandardScaler())]), num_n),
                                ("bin", "passthrough", BIN_FEATS),
                                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS)])
    s3 = cross_validate(Pipeline([("prep", prep_e), ("knn", KNeighborsClassifier(n_neighbors=best_k, metric="euclidean"))]),
                        Xn, y, cv=CV_ALL, scoring="accuracy")["test_score"]
    cod["knn_euc_std"].append(r(s3.mean()))
    Mt = clone(prep_n).fit_transform(Xn)
    D = np.sqrt(((Mt[:, None, :] - Mt[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(D, np.nan)
    contrast = np.nanmean((np.nanmax(D, 1) - np.nanmin(D, 1)) / np.nanmin(D, 1))
    cod["knn"].append(r(s1.mean())); cod["knn_std"].append(r(s1.std())); cod["lr"].append(r(s2.mean())); cod["contrast"].append(r(contrast))
# kontras jarak teoretis pada data uniform acak
dims_th = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]; th = []
for dd in dims_th:
    U = rng.uniform(size=(300, dd)); Dq = np.sqrt(((U[1:] - U[0]) ** 2).sum(1)); th.append(r((Dq.max() - Dq.min()) / Dq.min()))
cod["dims_theory"] = dims_th; cod["contrast_theory"] = th
SUMMARY["curse_of_dimensionality"] = cod
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
tot = [len(FEATURES) + d for d in noise_dims]
ax[0].errorbar(tot, cod["knn"], yerr=cod["knn_std"], fmt="o-", color=EMERALD, lw=2, capsize=3, label=f"KNN tuned ({best_scaler}, {best_metric})")
ax[0].plot(tot, cod["knn_euc_std"], "^-.", color=GOLD, lw=1.6, label=f"KNN StandardScaler + euclidean (K={best_k})")
ax[0].plot(tot, cod["lr"], "s--", color=BLACK, label="Logistic Regression")
ax[0].axhline(max(y.mean(), 1 - y.mean()), color=GREY, ls=":", label="Baseline mayoritas")
ax[0].set_xscale("log"); ax[0].set_xticks(tot); ax[0].set_xticklabels([str(t) for t in tot], fontsize=8); ax[0].minorticks_off(); ax[0].set_xlabel("Jumlah fitur mentah (asli + fitur noise acak)"); ax[0].set_ylabel("Akurasi CV (5×10)")
ax[0].set_title("Akurasi vs Dimensi: Dampak Fitur Tidak Relevan"); ax[0].legend(fontsize=8)
ax[1].loglog(dims_th, th, "o-", color=EMERALD, lw=2, label="Data uniform acak (n=300)")
ax[1].loglog(tot, cod["contrast"], "s--", color=BLACK, label="Data Reels + noise")
ax[1].set_xlabel("Dimensi"); ax[1].set_ylabel("Kontras jarak (Dmax − Dmin)/Dmin")
ax[1].set_title("Hilangnya Kontras Jarak pada Dimensi Tinggi"); ax[1].legend(fontsize=8)
plt.tight_layout()
savefig("G19_curse_of_dimensionality", "Akurasi dan kontras jarak terhadap jumlah dimensi")

# G23 Akurasi vs jumlah fitur (fitur ditambahkan berurutan menurut permutation importance)
rank_feats = [d["feature"] for d in SUMMARY["permutation_importance"]]
fs_res = {"order": rank_feats, "n": [], "knn": [], "knn_std": []}
for n in range(1, len(rank_feats) + 1):
    sel = rank_feats[:n]
    nf, bf, cf = [c for c in NUM_FEATS if c in sel], [c for c in BIN_FEATS if c in sel], [c for c in CAT_FEATS if c in sel]
    trs = []
    if nf: trs.append(("num", Pipeline([("cap", IQRCapper()), ("scaler", clone(prep_f.named_transformers_["num"].named_steps["scaler"]))]), nf))
    if bf: trs.append(("bin", "passthrough", bf))
    if cf: trs.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cf))
    mdl = Pipeline([("prep", ColumnTransformer(trs)), ("knn", clone(knn_f))])
    sc = cross_validate(mdl, X[sel], y, cv=CV_ALL, scoring="accuracy")["test_score"]
    fs_res["n"].append(n); fs_res["knn"].append(r(sc.mean())); fs_res["knn_std"].append(r(sc.std()))
fs_res["best_n"] = int(fs_res["n"][int(np.argmax(fs_res["knn"]))]); fs_res["best_acc"] = max(fs_res["knn"])
SUMMARY["feature_count_curve"] = fs_res
fig, ax = plt.subplots(figsize=(10, 4.3))
m_, sd_ = np.array(fs_res["knn"]), np.array(fs_res["knn_std"])
ax.plot(fs_res["n"], m_, "o-", color=EMERALD, lw=2); ax.fill_between(fs_res["n"], m_ - sd_, m_ + sd_, color=SAGE, alpha=.35)
ax.axhline(max(y.mean(), 1 - y.mean()), color=GOLD, ls=":", label="Baseline mayoritas")
ax.axvline(fs_res["best_n"], color=BLACK, ls="--", lw=1, label=f"Puncak: {fs_res['best_n']} fitur (akurasi {fs_res['best_acc']:.3f})")
ax.set_xticks(fs_res["n"]); ax.set_xticklabels([f"{i}: +{f}" for i, f in zip(fs_res["n"], rank_feats)], fontsize=7.5, rotation=35, ha="right")
ax.set_ylabel("Akurasi CV (5×10, seluruh data)"); ax.set_xlabel("Jumlah fitur (ditambahkan menurut urutan importance)")
ax.set_title("Akurasi KNN vs Jumlah Fitur Asli"); ax.legend(fontsize=8)
savefig("G23_akurasi_vs_jumlah_fitur", "Akurasi KNN terhadap jumlah fitur asli yang digunakan")

# G20 Waktu komputasi vs ukuran data (bukti lazy learner)
Xbase = Xtr_std  # ruang fitur ter-transformasi (data latih)
sizes_t = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
n_query = 1000
timing = {"sizes": sizes_t, "knn_fit": [], "knn_pred": [], "knn_kd_fit": [], "knn_kd_pred": [], "lr_fit": [], "lr_pred": []}
Q = Xbase[rng.integers(0, len(Xbase), n_query)] + rng.normal(0, .05, (n_query, Xbase.shape[1]))
for n in sizes_t:
    idx = rng.integers(0, len(Xbase), n)
    Xs = Xbase[idx] + rng.normal(0, .05, (n, Xbase.shape[1])); ys = y_train[idx]
    for key, mdl in [("knn", KNeighborsClassifier(n_neighbors=best_k, algorithm="brute")),
                     ("knn_kd", KNeighborsClassifier(n_neighbors=best_k, algorithm="kd_tree")),
                     ("lr", LogisticRegression(max_iter=2000))]:
        ft, pt = [], []
        for _ in range(3):
            t0 = time.perf_counter(); mdl.fit(Xs, ys); ft.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); mdl.predict(Q); pt.append(time.perf_counter() - t0)
        timing[f"{key}_fit"].append(r(1000 * np.median(ft), 3)); timing[f"{key}_pred"].append(r(1000 * np.median(pt), 3))
SUMMARY["timing"] = timing
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for key, c, mk, lab in [("knn", EMERALD, "o-", "KNN (brute force)"), ("knn_kd", MINT, "^-", "KNN (KD-Tree)"), ("lr", BLACK, "s--", "Logistic Regression")]:
    ax[0].loglog(sizes_t, timing[f"{key}_fit"], mk, color=c, lw=2, label=lab)
    ax[1].loglog(sizes_t, timing[f"{key}_pred"], mk, color=c, lw=2, label=lab)
ax[0].set_title("Waktu Training (fit)"); ax[1].set_title(f"Waktu Prediksi {n_query} Query")
for a in ax:
    a.set_xlabel("Ukuran data latih (baris)"); a.set_ylabel("Waktu (ms, skala log)"); a.legend(fontsize=8)
plt.suptitle("Bukti Empiris Lazy Learner: Biaya KNN Berpindah ke Tahap Prediksi", fontweight="bold", y=1.02)
plt.tight_layout()
savefig("G20_waktu_komputasi", "Waktu fit dan prediksi terhadap ukuran data")

# G21 Penanganan class imbalance (Oversampling, SMOTE manual, Undersampling)
def smote(Xm, ym, k=5, seed=0):
    """SMOTE sederhana (Chawla dkk., 2002): interpolasi antara sampel minoritas dan
    salah satu dari k tetangga minoritasnya. Dijalankan HANYA pada fold latih."""
    rg = np.random.default_rng(seed)
    cls, cnts = np.unique(ym, return_counts=True); mino = cls[np.argmin(cnts)]; need = cnts.max() - cnts.min()
    Xmin = Xm[ym == mino]
    D = np.sqrt(((Xmin[:, None] - Xmin[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    nbr = np.argsort(D, 1)[:, :min(k, len(Xmin) - 1)]
    base = rg.integers(0, len(Xmin), need); pick = nbr[base, rg.integers(0, nbr.shape[1], need)]
    gap = rg.uniform(size=(need, 1))
    Xnew = Xmin[base] + gap * (Xmin[pick] - Xmin[base])
    return np.vstack([Xm, Xnew]), np.concatenate([ym, np.full(need, mino)])


def ros(Xm, ym, seed=0):
    rg = np.random.default_rng(seed); cls, cnts = np.unique(ym, return_counts=True); mino = cls[np.argmin(cnts)]
    add = rg.choice(np.where(ym == mino)[0], cnts.max() - cnts.min()); return np.vstack([Xm, Xm[add]]), np.concatenate([ym, ym[add]])


def rus(Xm, ym, seed=0):
    rg = np.random.default_rng(seed); cls, cnts = np.unique(ym, return_counts=True); majo = cls[np.argmax(cnts)]
    keep = np.concatenate([rg.choice(np.where(ym == majo)[0], cnts.min(), replace=False), np.where(ym != majo)[0]])
    return Xm[keep], ym[keep]


imb = {}
for name, fn in [("Tanpa resampling", None), ("Random Oversampling", ros), ("SMOTE", smote), ("Random Undersampling", rus)]:
    acc, rec, f1s, bal, prc = [], [], [], [], []
    for i, (tr_i, va_i) in enumerate(CV_ALL.split(X_train, y_train)):
        pr = clone(prep_f).fit(X_train.iloc[tr_i]); Xa = pr.transform(X_train.iloc[tr_i]); ya = y_train[tr_i]
        if fn is not None:
            Xa, ya = fn(Xa, ya, seed=i)
        kn = clone(knn_f).fit(Xa, ya); yp = kn.predict(pr.transform(X_train.iloc[va_i])); yt = y_train[va_i]
        acc.append(accuracy_score(yt, yp)); rec.append(recall_score(yt, yp)); f1s.append(f1_score(yt, yp))
        bal.append(balanced_accuracy_score(yt, yp)); prc.append(precision_score(yt, yp, zero_division=0))
    imb[name] = {"acc": r(np.mean(acc)), "recall_viral": r(np.mean(rec)), "precision_viral": r(np.mean(prc)),
                 "f1_viral": r(np.mean(f1s)), "balanced_acc": r(np.mean(bal))}
SUMMARY["imbalance"] = imb
fig, ax = plt.subplots(figsize=(10, 4.2))
mets = ["acc", "balanced_acc", "precision_viral", "recall_viral", "f1_viral"]
mlab = ["Akurasi", "Balanced Acc.", "Precision (Viral)", "Recall (Viral)", "F1 (Viral)"]
cols4 = [GREY, EMERALD, MINT, BLACK]; w = .2
for i, (nm, c) in enumerate(zip(imb, cols4)):
    vals = [imb[nm][m] for m in mets]
    bb = ax.bar(np.arange(5) + (i - 1.5) * w, vals, w, color=c, edgecolor=BLACK, label=nm)
    for b, v in zip(bb, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90)
ax.set_xticks(range(5)); ax.set_xticklabels(mlab); ax.set_ylim(0, 1.1); ax.set_ylabel("Skor CV (5×10, data latih)")
ax.set_title("Dampak Strategi Penanganan Class Imbalance pada KNN"); ax.legend(fontsize=8, ncol=4, loc="upper center")
savefig("G21_class_imbalance", "Perbandingan strategi penanganan class imbalance")

# G18 Perbandingan antar model (akurasi uji & CV, F1, AUC)
cmp_df = pd.DataFrame(comp_rows)
order_m = ["Dummy (most frequent)", "Dummy (stratified)", "Gaussian Naive Bayes", "Logistic Regression", "Decision Tree",
           "Random Forest", "KNN default (K=5, tanpa scaling)", "KNN (tuned)"]
cmp_df = cmp_df.set_index("model").loc[order_m].reset_index()
fig, ax = plt.subplots(figsize=(11, 4.8))
w = .2; xs = np.arange(len(cmp_df))
for i, (m, c, lab) in enumerate([("accuracy", GREY, "Akurasi uji"), ("cv_acc", EMERALD, "Akurasi CV 5×10"),
                                 ("f1", BLACK, "F1 uji"), ("roc_auc", GOLD, "ROC-AUC uji")]):
    ax.bar(xs + (i - 1.5) * w, cmp_df[m], w, color=c, edgecolor=BLACK, label=lab)
ax.set_xticks(xs); ax.set_xticklabels([t.replace(" (", "\n(").replace("Gaussian Naive Bayes", "Gaussian\nNaive Bayes").replace("Logistic Regression", "Logistic\nRegression").replace("Random Forest", "Random\nForest").replace("Decision Tree", "Decision\nTree") for t in cmp_df.model], fontsize=7.5)
ax.set_ylim(0, 1.1); ax.set_ylabel("Skor"); ax.legend(fontsize=8, ncol=4, loc="upper left")
ax.set_title("Perbandingan KNN dengan Baseline dan Model Pembanding")
savefig("G18_perbandingan_model", "Perbandingan kinerja KNN dengan baseline dan model pembanding")

# G22 Scatter durasi vs panjang caption + usia unggahan
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for c in [0, 1]:
    s = df[df.target == c]
    ax[0].scatter(s.video_duration_sec, s.caption_word_count, c=CLS_COL[c], s=40, edgecolor="white", alpha=.85, label=CLS_NAME[c])
    ax[1].scatter(s.post_age_days, s.hashtag_count, c=CLS_COL[c], s=40, edgecolor="white", alpha=.85, label=CLS_NAME[c])
ax[0].set_xlabel("video_duration_sec"); ax[0].set_ylabel("caption_word_count"); ax[0].set_title("Durasi Video vs Panjang Caption")
ax[1].set_xlabel("post_age_days"); ax[1].set_ylabel("hashtag_count"); ax[1].set_title("Usia Unggahan vs Jumlah Hashtag"); ax[1].set_xscale("symlog")
ax[0].legend(fontsize=8)
plt.tight_layout()
savefig("G22_scatter_fitur", "Sebaran dua dimensi fitur utama menurut kelas")

# G00 Diagram alur CRISP-DM penelitian
fig, ax = plt.subplots(figsize=(11, 3.4)); ax.axis("off")
steps = [("1. Business\nUnderstanding", "Target Viral Hit\nvs High Reach"), ("2. Data\nUnderstanding", "88 Reels, 15 kolom\nmissing, duplikat, IQR"),
         ("3. Data\nPreparation", "Anti-leakage, fitur baru\none-hot, capping, scaling"), ("4. Modeling", "KNN + kurva K\nGridSearchCV"),
         ("5. Evaluation", "Metrik, baseline,\npembanding, analisis"), ("6. Deployment", "Rekomendasi\nkreator/marketer")]
for i, (t, s) in enumerate(steps):
    x0 = i * 1.8
    ax.add_patch(plt.Rectangle((x0, 0.9), 1.5, 1.1, color=DEEP if i % 2 == 0 else BLACK, ec=GOLD, lw=1.5))
    ax.text(x0 + .75, 1.45, t, ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    ax.add_patch(plt.Rectangle((x0, 0.0), 1.5, .75, color=LIGHT, ec=EMERALD, lw=1))
    ax.text(x0 + .75, .375, s, ha="center", va="center", color=BLACK, fontsize=7.8)
    if i < 5:
        ax.annotate("", xy=(x0 + 1.78, 1.45), xytext=(x0 + 1.52, 1.45), arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2))
ax.annotate("", xy=(0.75, 2.05), xytext=(9.75, 2.05), arrowprops=dict(arrowstyle="-|>", color=EMERALD, lw=1.3, connectionstyle="arc3,rad=0.12"))
ax.text(5.25, 2.5, "iterasi (evaluasi dapat mengembalikan ke tahap persiapan data)", ha="center", fontsize=8, color=EMERALD, style="italic")
ax.set_xlim(-.1, 10.7); ax.set_ylim(-.1, 2.8)
savefig("G00_crisp_dm", "Diagram alur CRISP-DM penelitian")

# =====================================================================
# 6. SIMPAN DATASET BERSIH, SAMPEL, DAN RINGKASAN
# =====================================================================
clean_cols = ["post_id", "shortcode", "creator_handle", "niche", "timestamp_utc", "post_hour", "day_of_week"] + \
             [c for c in FEATURES if c != "niche"] + ["views", "likes", "comments", "engagement_rate_pct", "virality_score",
                                                      "virality_tier", "target"]
clean = df[clean_cols].copy()
clean["split"] = np.where(clean.index.isin(X_test.index), "test", "train")
clean.to_csv(os.path.join(OUT, "instagram_reels_knn_clean.csv"), index=False)
SUMMARY["sample_rows"] = df.loc[X_train.index[:6], ["shortcode", "niche", "video_duration_sec", "caption_word_count",
                                                     "hashtag_count", "emoji_count", "post_hour", "post_age_days", "virality_tier"]] \
    .round(2).astype(object).to_dict(orient="records")
SUMMARY["figures"] = FIG_LIST
SUMMARY["final_feature_names"] = feat_names
SUMMARY["n_features_transformed"] = len(feat_names)
SUMMARY["versions"] = {"python": os.sys.version.split()[0], "sklearn": __import__("sklearn").__version__,
                       "pandas": pd.__version__, "numpy": np.__version__}
with open(os.path.join(OUT, "model_summary.json"), "w", encoding="utf-8") as f:
    json.dump(SUMMARY, f, indent=2, ensure_ascii=False, default=str)

print("=" * 70)
print(f"Data: {du['n_rows']} baris × {du['n_cols']} kolom | train={len(X_train)} test={len(X_test)}")
print(f"Best params: scaler={best_scaler}, metric={best_metric}, K={best_k}, weights={best_w}, CV acc={gs.best_score_:.4f}")
print(f"Test: acc={knn_eval['accuracy']} f1={knn_eval['f1']} auc={knn_eval['roc_auc']}")
print(f"Manual vs sklearn proba: {proba_manual} vs {sk_proba} -> assert OK")
print(f"{len(FIG_LIST)} grafik disimpan ke {FIG}")
