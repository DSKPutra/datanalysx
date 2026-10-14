"""
=====================================================================================
  NAIVE BAYES UNTUK PREDIKSI ARAH PERGERAKAN IHSG BERBASIS INDIKATOR MAKRO-FINANSIAL
=====================================================================================
Studi Kasus 06 — Naive Bayes (proyek tim)
 
Skrip ini bersifat end-to-end dan reproducible: menjalankan skrip ini dari awal
akan menghasilkan seluruh angka, tabel, dan grafik yang dikutip di dalam laporan.
Dataset  : Daily Multi-Asset Macroeconomic and Financial Time Series Dataset
           (Indonesia/Global Markets), 2010-01-01 s.d. 2026-05-29, 4.277 observasi harian.
 
Alur kerja:
  1. Pemuatan & pembersihan data (missing value, tipe data, pengurutan tanggal)
  2. Rekayasa fitur (return harian, label arah pergerakan, lag, indikator makro)
  3. Eksplorasi Data (EDA) + visualisasi
  4. Pembagian data latih/uji berbasis waktu (time-based split, bukan random split)
  5. Pelatihan model Gaussian Naive Bayes
  6. Evaluasi: akurasi, presisi, recall, F1, confusion matrix, ROC-AUC
  7. Validasi silang deret waktu (TimeSeriesSplit) & kurva pembelajaran
  8. Analisis probabilitas prior/likelihood per kelas (interpretasi model Naive Bayes)
  9. Ekspor seluruh grafik (PNG 300dpi) dan ringkasan metrik (JSON) untuk laporan
=====================================================================================
"""
 
import json
import warnings
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report,
    precision_recall_curve, average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings("ignore")
 
# --------------------------------------------------------------------------------
# 0. KONFIGURASI TAMPILAN & TEMA WARNA (selaras dengan tema laporan hijau-hitam)
# --------------------------------------------------------------------------------
GREEN_DARK   = "#0B3D24"
GREEN_MAIN   = "#12864B"
GREEN_LIGHT  = "#4CD07D"
GREEN_ACCENT = "#8CF5B0"
BLACK_MAIN   = "#111315"
GREY_MAIN    = "#5A6169"
GOLD_ACCENT  = "#C9A227"
RED_DOWN     = "#B3261E"
 
sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": "#22262B",
    "axes.labelcolor": "#111315",
    "text.color": "#111315",
    "xtick.color": "#111315",
    "ytick.color": "#111315",
    "grid.color": "#DDE3DF",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
 
BASE = Path(__file__).resolve().parent.parent
CHART_DIR = BASE / "charts"
DATA_DIR = BASE / "data"
CHART_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
 
RAW_CSV = "/root/.claude/uploads/f8c8de33-3c10-59ff-a92b-2d9b7b576cdb/da88b0ad-Daily_Multi-Asset_Macroeconomic_and_Financial_Time_Series_Dataset_IndonesiaGlobal_Markets.csv"
 
summary = {}
 
# --------------------------------------------------------------------------------
# 1. PEMUATAN & PEMBERSIHAN DATA
# --------------------------------------------------------------------------------
df = pd.read_csv(RAW_CSV)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
 
summary["n_rows_raw"] = int(len(df))
summary["date_min"] = str(df["Date"].min().date())
summary["date_max"] = str(df["Date"].max().date())
summary["missing_before"] = df.isna().sum().to_dict()
 
# Interpolasi linear terbatas untuk harga aset (weekend/hari libur bursa berbeda antar
# pasar); variabel makro (CPI, BI_rate, US_rate) sudah lengkap (rilis bulanan yang di-
# forward-fill pada sumber data asli).
price_cols = ["OIL", "GOLD", "USDIDR", "SP500", "IHSG", "VIX"]
df[price_cols] = df[price_cols].interpolate(method="linear", limit_direction="both")
 
summary["missing_after"] = df.isna().sum().to_dict()
summary["n_rows_clean"] = int(len(df))
 
# --------------------------------------------------------------------------------
# 2. REKAYASA FITUR
# --------------------------------------------------------------------------------
for c in ["OIL", "GOLD", "USDIDR", "SP500", "IHSG"]:
    df[f"{c}_ret"] = df[c].pct_change() * 100
 
df["VIX_chg"] = df["VIX"].diff()
df["rate_diff"] = df["BI_rate"] - df["US_rate"]          # selisih suku bunga RI-AS
df["CPI_chg"] = df["CPI"].diff()
 
# --------------------------------------------------------------------------------
# 2b. DETEKSI & PENANGANAN OUTLIER PADA VARIABEL RETURN (metode IQR)
# --------------------------------------------------------------------------------
# Pemeriksaan data mentah menemukan lonjakan return USDIDR yang tidak wajar secara
# ekonomi (mis. +902,9% pada 2010-11-02 diikuti -90,0% pada 2010-11-01, dan pola
# serupa pada 2012-02-07/08) -- indikasi kuat kesalahan entri/anomali sumber data,
# bukan pergerakan pasar riil (depresiasi rupiah ekstrem >900% dalam sehari secara
# empiris tidak pernah terjadi di pasar valas). Nilai-nilai ini di-winsorize dengan
# metode IQR (batas Q1-3*IQR s.d. Q3+3*IQR) agar tidak mendominasi estimasi varians
# Gaussian pada model Naive Bayes, yang sangat sensitif terhadap outlier.
ret_cols = ["OIL_ret", "GOLD_ret", "USDIDR_ret", "SP500_ret", "IHSG_ret"]
outlier_log = {}
for c in ret_cols:
    q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
    outlier_log[c] = {"batas_bawah": round(float(lo), 3), "batas_atas": round(float(hi), 3), "jumlah_outlier": n_out}
    df[c] = df[c].clip(lower=lo, upper=hi)
 
# Target: arah pergerakan IHSG besok (1 = naik, 0 = turun/tetap), diprediksi
# menggunakan HANYA informasi yang tersedia pada hari t (tidak ada leakage).
df["IHSG_ret_next"] = df["IHSG_ret"].shift(-1)
df["target"] = (df["IHSG_ret_next"] > 0).astype(int)
 
FEATURES = [
    "IHSG_ret", "SP500_ret", "OIL_ret", "GOLD_ret", "USDIDR_ret",
    "VIX", "VIX_chg", "rate_diff", "CPI_chg",
]
 
model_df = df.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)
# buang baris terakhir (target NaN karena shift) sudah otomatis oleh dropna
summary["n_rows_model"] = int(len(model_df))
summary["class_balance"] = model_df["target"].value_counts(normalize=True).round(4).to_dict()
summary["features"] = FEATURES
summary["outlier_treatment"] = outlier_log
 
model_df.to_csv(DATA_DIR / "clean_feature_dataset.csv", index=False)
 
# --------------------------------------------------------------------------------
# 3. SPLIT DATA BERBASIS WAKTU (80% latih paling awal, 20% uji paling akhir)
# --------------------------------------------------------------------------------
split_idx = int(len(model_df) * 0.8)
train_df = model_df.iloc[:split_idx]
test_df = model_df.iloc[split_idx:]
 
X_train, y_train = train_df[FEATURES].values, train_df["target"].values
X_test, y_test = test_df[FEATURES].values, test_df["target"].values
 
summary["train_period"] = f"{train_df['Date'].min().date()} s.d. {train_df['Date'].max().date()} (n={len(train_df)})"
summary["test_period"] = f"{test_df['Date'].min().date()} s.d. {test_df['Date'].max().date()} (n={len(test_df)})"
summary["train_class_balance"] = train_df["target"].value_counts(normalize=True).round(4).to_dict()
summary["test_class_balance"] = test_df["target"].value_counts(normalize=True).round(4).to_dict()
 
# Catatan metodologis: Gaussian Naive Bayes mengestimasi mean (theta) dan varians (var)
# per fitur secara terpisah untuk tiap kelas, sehingga --berbeda dengan model berbasis
# jarak seperti KNN atau SVM-- standardisasi skala TIDAK mengubah hasil prediksi model
# ini secara matematis. Oleh karena itu fitur dilatih pada skala aslinya agar parameter
# theta (rata-rata) dan sigma (simpangan baku) yang dilaporkan tetap dalam satuan yang
# mudah diinterpretasikan (%, poin indeks VIX, dsb).
X_train_s = X_train
X_test_s = X_test
 
# --------------------------------------------------------------------------------
# 4. PELATIHAN MODEL GAUSSIAN NAIVE BAYES
# --------------------------------------------------------------------------------
gnb = GaussianNB()
gnb.fit(X_train_s, y_train)
 
y_pred = gnb.predict(X_test_s)
y_proba = gnb.predict_proba(X_test_s)[:, 1]
 
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
brier = brier_score_loss(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
 
summary["metrics"] = {
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1_score": round(float(f1), 4),
    "roc_auc": round(float(auc), 4),
    "brier_score": round(float(brier), 4),
    "average_precision": round(float(ap), 4),
    "confusion_matrix": cm.tolist(),
}
summary["classification_report"] = classification_report(y_test, y_pred, target_names=["Turun/Tetap (0)", "Naik (1)"], output_dict=True)
 
# Baseline naif (selalu prediksi kelas mayoritas) sebagai pembanding
majority_class = int(pd.Series(y_train).mode()[0])
baseline_pred = np.full_like(y_test, majority_class)
summary["baseline_majority_accuracy"] = round(float(accuracy_score(y_test, baseline_pred)), 4)
 
# --------------------------------------------------------------------------------
# 5. PARAMETER MODEL: PRIOR & LIKELIHOOD (INTERPRETASI KHAS NAIVE BAYES)
# --------------------------------------------------------------------------------
class_prior = dict(zip(["Turun/Tetap (0)", "Naik (1)"], gnb.class_prior_.round(4).tolist()))
theta = pd.DataFrame(gnb.theta_, columns=FEATURES, index=["Turun/Tetap (0)", "Naik (1)"])
sigma = pd.DataFrame(np.sqrt(gnb.var_), columns=FEATURES, index=["Turun/Tetap (0)", "Naik (1)"])
 
summary["class_prior"] = class_prior
summary["theta_mean_per_class"] = theta.round(4).to_dict()
summary["sigma_std_per_class"] = sigma.round(4).to_dict()
 
# --------------------------------------------------------------------------------
# 5b. CONTOH PERHITUNGAN MANUAL LANGKAH-DEMI-LANGKAH (WORKED EXAMPLE)
# --------------------------------------------------------------------------------
# Untuk menelusuri cara kerja model secara matematis (bukan sekadar hasil akhir),
# satu observasi nyata dari data uji dipilih dan dihitung ulang secara manual:
# prior -> likelihood Gaussian per fitur -> gabungan log-probabilitas -> normalisasi
# posterior -> keputusan MAP. Hasilnya diverifikasi identik dengan predict_proba().
def gaussian_pdf(x, mu, var):
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mu) ** 2) / (2 * var))
 
CLASS_LABELS = ["Turun/Tetap (0)", "Naik (1)"]
example_idx = 6  # observasi ke-7 pada data uji (baris pertama dengan prediksi & label yang tegas)
ex_row = test_df.iloc[example_idx]
x_vec = ex_row[FEATURES].values.astype(float)
 
log_prior = np.log(gnb.class_prior_)
per_feature = {lbl: [] for lbl in CLASS_LABELS}
loglik_sum = np.zeros(2)
for cls in [0, 1]:
    for i, feat in enumerate(FEATURES):
        mu_ci, var_ci = gnb.theta_[cls, i], gnb.var_[cls, i]
        pdf_ci = gaussian_pdf(x_vec[i], mu_ci, var_ci)
        loglik_sum[cls] += np.log(pdf_ci)
        per_feature[CLASS_LABELS[cls]].append({
            "fitur": feat, "x": round(float(x_vec[i]), 4),
            "mu": round(float(mu_ci), 4), "sigma": round(float(np.sqrt(var_ci)), 4),
            "pdf": round(float(pdf_ci), 6), "log_pdf": round(float(np.log(pdf_ci)), 4),
        })
 
log_joint = log_prior + loglik_sum                       # log[P(y)*prod P(xi|y)], belum ternormalisasi
m_stable = log_joint.max()                                # trik log-sum-exp agar stabil secara numerik
post_unnorm = np.exp(log_joint - m_stable)
posterior = post_unnorm / post_unnorm.sum()               # P(y|x) = P(y)*prod P(xi|y) / P(x)
 
pred_manual = int(np.argmax(posterior))
proba_sklearn = gnb.predict_proba(x_vec.reshape(1, -1))[0]
assert np.allclose(posterior, proba_sklearn, atol=1e-8), "Perhitungan manual tidak sama dengan predict_proba()!"
 
summary["worked_example"] = {
    "tanggal": str(ex_row["Date"].date()),
    "index_in_test": int(example_idx),
    "fitur_observasi": {f: round(float(v), 4) for f, v in zip(FEATURES, x_vec)},
    "target_aktual": int(ex_row["target"]),
    "ihsg_ret_next_aktual": round(float(ex_row["IHSG_ret_next"]), 4),
    "log_prior": {CLASS_LABELS[i]: round(float(log_prior[i]), 4) for i in range(2)},
    "per_feature_likelihood": per_feature,
    "sum_log_likelihood": {CLASS_LABELS[i]: round(float(loglik_sum[i]), 4) for i in range(2)},
    "log_joint_unnormalized": {CLASS_LABELS[i]: round(float(log_joint[i]), 4) for i in range(2)},
    "posterior": {CLASS_LABELS[i]: round(float(posterior[i]), 6) for i in range(2)},
    "prediksi_manual": CLASS_LABELS[pred_manual],
    "proba_sklearn_cek": {CLASS_LABELS[i]: round(float(proba_sklearn[i]), 6) for i in range(2)},
    "cocok_dengan_sklearn": bool(np.allclose(posterior, proba_sklearn, atol=1e-8)),
}
 
# --------------------------------------------------------------------------------
# 6. VALIDASI SILANG DERET WAKTU (TimeSeriesSplit, 5-fold expanding window)
# --------------------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)
X_full = model_df[FEATURES].values
y_full = model_df["target"].values
 
cv_rows = []
for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_full), start=1):
    Xtr, Xva = X_full[tr_idx], X_full[va_idx]
    m = GaussianNB().fit(Xtr, y_full[tr_idx])
    proba = m.predict_proba(Xva)[:, 1]
    pred = m.predict(Xva)
    cv_rows.append({
        "fold": fold,
        "n_train": len(tr_idx),
        "n_val": len(va_idx),
        "accuracy": accuracy_score(y_full[va_idx], pred),
        "roc_auc": roc_auc_score(y_full[va_idx], proba),
        "f1": f1_score(y_full[va_idx], pred, zero_division=0),
    })
cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(DATA_DIR / "cv_results.csv", index=False)
summary["cv_results"] = cv_df.round(4).to_dict(orient="records")
summary["cv_auc_mean_std"] = [round(float(cv_df["roc_auc"].mean()), 4), round(float(cv_df["roc_auc"].std()), 4)]
summary["cv_accuracy_mean_std"] = [round(float(cv_df["accuracy"].mean()), 4), round(float(cv_df["accuracy"].std()), 4)]
 
# ==================================================================================
#  VISUALISASI  — seluruh grafik disimpan sebagai PNG 300dpi ke folder charts/
# ==================================================================================
 
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(CHART_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
 
 
# G00 — Boxplot sebelum/sesudah penanganan outlier (USDIDR_ret sebagai contoh kasus)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
raw_usdidr = pd.read_csv(RAW_CSV)
raw_usdidr["ret"] = raw_usdidr["USDIDR"].pct_change() * 100
sns.boxplot(y=raw_usdidr["ret"].dropna(), ax=axes[0], color=RED_DOWN, width=0.35)
axes[0].set_title("Sebelum Penanganan Outlier")
axes[0].set_ylabel("Return Harian USDIDR (%)")
sns.boxplot(y=model_df["USDIDR_ret"], ax=axes[1], color=GREEN_MAIN, width=0.35)
axes[1].set_title("Sesudah Winsorization (IQR)")
axes[1].set_ylabel("Return Harian USDIDR (%)")
fig.suptitle("Dampak Penanganan Outlier pada Fitur USDIDR_ret", fontsize=14, y=1.02)
savefig(fig, "G00_outlier_treatment.png")
 
# G01 — Tren harga seluruh aset (dinormalisasi ke basis 100)
fig, ax = plt.subplots(figsize=(11, 5.5))
norm = df.set_index("Date")[["OIL", "GOLD", "SP500", "IHSG"]].dropna()
norm = norm / norm.iloc[0] * 100
colors = [GREEN_DARK, GOLD_ACCENT, GREY_MAIN, GREEN_MAIN]
for col, c in zip(norm.columns, colors):
    ax.plot(norm.index, norm[col], label=col, color=c, linewidth=1.4)
ax.set_title("Tren Harga Aset Ternormalisasi (Basis 100 = 1 Januari 2010)")
ax.set_ylabel("Indeks (Basis 100)")
ax.set_xlabel("Tanggal")
ax.legend(frameon=False, ncol=4, loc="upper left")
savefig(fig, "G01_tren_harga_normalisasi.png")
 
# G02 — IHSG dengan overlay VIX
fig, ax1 = plt.subplots(figsize=(11, 5.5))
ax1.plot(df["Date"], df["IHSG"], color=GREEN_MAIN, linewidth=1.3, label="IHSG")
ax1.set_ylabel("IHSG (poin)", color=GREEN_DARK)
ax1.set_xlabel("Tanggal")
ax2 = ax1.twinx()
ax2.plot(df["Date"], df["VIX"], color=RED_DOWN, linewidth=0.8, alpha=0.6, label="VIX")
ax2.set_ylabel("VIX (indeks volatilitas)", color=RED_DOWN)
ax2.grid(False)
ax1.set_title("Pergerakan IHSG vs. Indeks Volatilitas Global (VIX)")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
savefig(fig, "G02_ihsg_vs_vix.png")
 
# G03 — Distribusi return harian IHSG
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(model_df["IHSG_ret"], bins=80, color=GREEN_MAIN, kde=True, ax=ax, edgecolor="white")
ax.axvline(0, color=BLACK_MAIN, linestyle="--", linewidth=1)
ax.set_title("Distribusi Return Harian IHSG")
ax.set_xlabel("Return Harian (%)")
ax.set_ylabel("Frekuensi")
savefig(fig, "G03_distribusi_return_ihsg.png")
 
# G04 — Heatmap korelasi antar fitur
fig, ax = plt.subplots(figsize=(9, 7.5))
corr = model_df[FEATURES + ["target"]].corr()
cmap = sns.diverging_palette(10, 145, s=70, l=45, as_cmap=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0, square=True,
            linewidths=0.6, linecolor="white", cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Matriks Korelasi Antar-Fitur dan Target")
savefig(fig, "G04_heatmap_korelasi.png")
 
# G05 — Boxplot / distribusi fitur numerik per kelas (grid)
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for ax, feat in zip(axes.flat, FEATURES):
    bp = sns.boxplot(data=model_df, x="target", y=feat, ax=ax,
                      showfliers=False, width=0.55)
    for i, box_color in enumerate([GREY_MAIN, GREEN_MAIN]):
        if i < len(ax.patches):
            ax.patches[i].set_facecolor(box_color)
    ax.set_xticklabels(["Turun/Tetap", "Naik"])
    ax.set_xlabel("")
    ax.set_title(feat, fontsize=11)
fig.suptitle("Distribusi Fitur Berdasarkan Kelas Target (Arah IHSG Besok)", fontsize=15, y=1.01)
savefig(fig, "G05_boxplot_fitur_per_kelas.png")
 
# G06 — Kurva Gaussian likelihood P(x|kelas) untuk fitur paling diskriminatif
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
top_feats = ["IHSG_ret", "SP500_ret", "VIX", "USDIDR_ret"]
for ax, feat in zip(axes.flat, top_feats):
    idx = FEATURES.index(feat)
    xs = np.linspace(model_df[feat].quantile(0.01), model_df[feat].quantile(0.99), 300)
    for cls, color, lbl in [(0, GREY_MAIN, "Turun/Tetap"), (1, GREEN_MAIN, "Naik")]:
        mu = theta.loc[["Turun/Tetap (0)", "Naik (1)"][cls], feat]
        sd = sigma.loc[["Turun/Tetap (0)", "Naik (1)"][cls], feat]
        ys = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
        ax.plot(xs, ys, color=color, label=f"{lbl} (μ={mu:.2f}, σ={sd:.2f})", linewidth=1.8)
        ax.fill_between(xs, ys, alpha=0.15, color=color)
    ax.set_title(f"Likelihood Gaussian: {feat}")
    ax.set_xlabel(feat)
    ax.set_ylabel("Densitas Probabilitas")
    ax.legend(frameon=False, fontsize=8.5)
fig.suptitle("Distribusi Likelihood Gaussian Naive Bayes per Fitur dan Kelas", fontsize=15, y=1.02)
savefig(fig, "G06_likelihood_gaussian.png")
 
# G07 — Confusion Matrix
fig, ax = plt.subplots(figsize=(6.2, 5.4))
cmap2 = sns.light_palette(GREEN_MAIN, as_cmap=True)
sns.heatmap(cm, annot=True, fmt="d", cmap=cmap2, cbar=False, square=True,
            linewidths=1, linecolor="white", annot_kws={"size": 16, "weight": "bold"}, ax=ax,
            xticklabels=["Prediksi\nTurun/Tetap", "Prediksi\nNaik"],
            yticklabels=["Aktual\nTurun/Tetap", "Aktual\nNaik"])
ax.set_title("Confusion Matrix — Gaussian Naive Bayes (Data Uji)")
savefig(fig, "G07_confusion_matrix.png")
 
# G08 — ROC Curve
fig, ax = plt.subplots(figsize=(7, 6.3))
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax.plot(fpr, tpr, color=GREEN_MAIN, linewidth=2.2, label=f"Gaussian NB (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color=GREY_MAIN, linestyle="--", linewidth=1.2, label="Tebakan Acak (AUC = 0.500)")
ax.fill_between(fpr, tpr, alpha=0.12, color=GREEN_MAIN)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Kurva ROC — Model Gaussian Naive Bayes")
ax.legend(frameon=False, loc="lower right")
savefig(fig, "G08_roc_curve.png")
 
# G09 — Precision-Recall Curve
fig, ax = plt.subplots(figsize=(7, 6.3))
prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba)
ax.plot(rec_c, prec_c, color=GREEN_DARK, linewidth=2.2, label=f"AP = {ap:.3f}")
base_rate = y_test.mean()
ax.axhline(base_rate, color=GREY_MAIN, linestyle="--", linewidth=1.2, label=f"Baseline (prevalensi = {base_rate:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Kurva Precision-Recall — Model Gaussian Naive Bayes")
ax.legend(frameon=False, loc="lower left")
savefig(fig, "G09_precision_recall_curve.png")
 
# G10 — Kalibrasi probabilitas
fig, ax = plt.subplots(figsize=(7, 6.3))
frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
ax.plot(mean_pred, frac_pos, marker="o", color=GREEN_MAIN, linewidth=2, label=f"Gaussian NB (Brier = {brier:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color=GREY_MAIN, label="Kalibrasi Sempurna")
ax.set_xlabel("Rata-rata Probabilitas Prediksi")
ax.set_ylabel("Proporsi Aktual Positif")
ax.set_title("Kurva Kalibrasi Probabilitas")
ax.legend(frameon=False)
savefig(fig, "G10_kurva_kalibrasi.png")
 
# G11 — Validasi silang deret waktu per fold
fig, ax = plt.subplots(figsize=(9, 5.5))
width = 0.35
xs = np.arange(len(cv_df))
ax.bar(xs - width/2, cv_df["accuracy"], width, label="Akurasi", color=GREEN_MAIN)
ax.bar(xs + width/2, cv_df["roc_auc"], width, label="ROC-AUC", color=GOLD_ACCENT)
ax.axhline(0.5, color=RED_DOWN, linestyle="--", linewidth=1, label="Batas Tebakan Acak")
ax.set_xticks(xs)
ax.set_xticklabels([f"Fold {i}" for i in cv_df["fold"]])
ax.set_ylim(0.3, max(0.75, cv_df[["accuracy","roc_auc"]].values.max()+0.05))
ax.set_title("Validasi Silang Deret Waktu (TimeSeriesSplit, 5-Fold)")
ax.legend(frameon=False)
savefig(fig, "G11_cv_timeseries.png")
 
# G12 — Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    GaussianNB(), X_full, y_full,
    cv=TimeSeriesSplit(n_splits=5), scoring="roc_auc",
    train_sizes=np.linspace(0.1, 1.0, 8), shuffle=False
)
fig, ax = plt.subplots(figsize=(9, 5.8))
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color=GREEN_DARK, label="Skor Latih (ROC-AUC)")
ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color=GREEN_MAIN, label="Skor Validasi (ROC-AUC)")
ax.fill_between(train_sizes, train_scores.mean(1)-train_scores.std(1), train_scores.mean(1)+train_scores.std(1), color=GREEN_DARK, alpha=0.12)
ax.fill_between(train_sizes, val_scores.mean(1)-val_scores.std(1), val_scores.mean(1)+val_scores.std(1), color=GREEN_MAIN, alpha=0.15)
ax.axhline(0.5, color=RED_DOWN, linestyle="--", linewidth=1, label="Batas Tebakan Acak")
ax.set_xlabel("Jumlah Data Latih")
ax.set_ylabel("ROC-AUC")
ax.set_title("Kurva Pembelajaran (Learning Curve) — Gaussian Naive Bayes")
ax.legend(frameon=False)
savefig(fig, "G12_learning_curve.png")
 
# G13 — Perbandingan model vs baseline mayoritas
fig, ax = plt.subplots(figsize=(7.5, 5.5))
comp = pd.Series({
    "Baseline\n(Kelas Mayoritas)": summary["baseline_majority_accuracy"],
    "Gaussian\nNaive Bayes": acc,
})
bars = ax.bar(comp.index, comp.values, color=[GREY_MAIN, GREEN_MAIN], width=0.5)
for b, v in zip(bars, comp.values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
ax.set_ylim(0, max(0.65, comp.values.max() + 0.1))
ax.set_ylabel("Akurasi")
ax.set_title("Perbandingan Akurasi: Model vs. Baseline Naif")
savefig(fig, "G13_perbandingan_baseline.png")
 
# G14 — Rolling volatility IHSG (30 hari) untuk konteks makro
fig, ax = plt.subplots(figsize=(11, 5.2))
roll_vol = model_df.set_index("Date")["IHSG_ret"].rolling(30).std()
ax.plot(roll_vol.index, roll_vol.values, color=GREEN_DARK, linewidth=1.1)
ax.fill_between(roll_vol.index, roll_vol.values, color=GREEN_MAIN, alpha=0.2)
ax.set_title("Volatilitas Bergulir 30 Hari — Return Harian IHSG")
ax.set_ylabel("Std. Dev. Return 30 Hari (%)")
ax.set_xlabel("Tanggal")
savefig(fig, "G14_rolling_volatility.png")
 
# G15 — Prior kelas (pie chart)
fig, ax = plt.subplots(figsize=(6, 6))
vals = [class_prior["Turun/Tetap (0)"], class_prior["Naik (1)"]]
ax.pie(vals, labels=["Turun/Tetap", "Naik"], autopct="%1.1f%%", startangle=90,
       colors=[GREY_MAIN, GREEN_MAIN], wedgeprops={"edgecolor": "white", "linewidth": 2},
       textprops={"fontsize": 11, "fontweight": "bold"})
ax.set_title("Probabilitas Prior Kelas P(y) — Data Latih")
savefig(fig, "G15_prior_kelas.png")
 
# --------------------------------------------------------------------------------
# EKSPOR RINGKASAN METRIK KE JSON (dipakai untuk mengisi angka pada laporan Word)
# --------------------------------------------------------------------------------
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o
 
with open(DATA_DIR / "model_summary.json", "w") as f:
    json.dump(clean(summary), f, indent=2, ensure_ascii=False)
 
print("SELESAI. Grafik tersimpan di:", CHART_DIR)
print("Ringkasan metrik tersimpan di:", DATA_DIR / "model_summary.json")
print(json.dumps(clean(summary["metrics"]), indent=2))
 