#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================================
 STUDI KASUS 08 - KONSEP DAN IMPLEMENTASI ALGORITMA K-MEANS
 Pengelompokan Hari Perdagangan Saham AAPL 2026 ke dalam "Rezim Pasar"
-------------------------------------------------------------------------------------
 Penulis : Dea Saka Kurnia Putra (2026)
-------------------------------------------------------------------------------------
 Cara menjalankan :  python kmeans_aapl_stock.py
 Masukan          :  AAPL_2026_stock_data.csv (satu folder dengan skrip ini)
 Keluaran         :  outputs/charts/G00..G31_*.png   (grafik)
                     outputs/aapl_kmeans_clustered.csv (data bersih + fitur + label klaster)
                     outputs/model_summary.json        (SELURUH angka yang dikutip laporan)
 Alur (CRISP-DM)  :  1 Data Understanding -> 2 Feature Engineering -> 3 EDA
                     -> 4 Bukti harga mentah -> 5 Outlier -> 6 Scaling -> 7 Pemilihan K
                     -> 8 Inisialisasi & konvergensi -> 9 Model final & profil klaster
                     -> 10 Perhitungan manual -> 11 Pembanding algoritma
                     -> 12 Analisis lanjutan -> 13 Simpan keluaran
 Catatan          :  seluruh proses acak dikunci dengan RANDOM_STATE = 42 sehingga hasil
                     klaster identik antar-run. Hanya angka waktu komputasi yang berubah.
=====================================================================================
"""

import json
import os
import platform
import time
import warnings
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from scipy import stats
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans, kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

# =====================================================================================
# 0. KONFIGURASI UMUM
# =====================================================================================
RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "AAPL_2026_stock_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# --- Palet tema hijau zamrud - hitam - emas -------------------------------------------
EMERALD, DEEP, BLACK, GOLD = "#0B7A53", "#0B4F3A", "#111111", "#B8860B"
RUST, BLUE, PURPLE, GREY, LIGHT = "#C0452B", "#2F6DB5", "#8A4FA8", "#6B7280", "#E8F3EE"
# Warna & bentuk marker per klaster (lolos uji keterbedaan; marker = encoding kedua)
PALETTE = [EMERALD, RUST, GOLD, BLUE, PURPLE, GREY, "#1F2933", "#2AA198", "#D17A00", "#7B8B3A"]
MARKERS = ["^", "v", "D", "o", "s", "P", "X", "*", "h", "<"]

plt.rcParams.update(
    {
        "font.family": ["Carlito", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": BLACK,
        "axes.labelsize": 11,
        "axes.edgecolor": "#9CA3AF",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

SUMMARY = {"figures": []}  # semua angka laporan dikumpulkan di sini
TIMING = {}  # angka waktu (tidak deterministik) dipisah


def r(x, nd=4):
    """Bulatkan angka/array/dict secara rekursif agar JSON rapi."""
    if isinstance(x, dict):
        return {str(k): r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [r(v, nd) for v in list(x)]
    if isinstance(x, (np.floating, float)):
        return None if (np.isnan(x) or np.isinf(x)) else round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%Y-%m-%d")
    return x


def savefig(fig, code, title):
    """Simpan grafik sebagai PNG dan catat di ringkasan."""
    fname = f"{code}.png"
    fig.savefig(os.path.join(CHART_DIR, fname))
    plt.close(fig)
    SUMMARY["figures"].append({"file": fname, "title": title})
    print(f"   [grafik] {fname} - {title}")


def boxplot_compat(ax, values, names, **kw):
    """Boxplot yang kompatibel lintas versi matplotlib (argumen 'labels' berganti nama di versi baru)."""
    bp = ax.boxplot(values, **kw)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)
    return bp


def eta_squared(values, labels):
    """Ukuran efek eta^2 = SS_between / SS_total (proporsi varians yang dijelaskan klaster)."""
    values, labels = np.asarray(values, float), np.asarray(labels)
    grand = values.mean()
    ss_t = ((values - grand) ** 2).sum()
    ss_b = sum(
        len(values[labels == g]) * (values[labels == g].mean() - grand) ** 2 for g in np.unique(labels)
    )
    return ss_b / ss_t if ss_t > 0 else 0.0


def match_labels(ref, lab, k=None):
    """Samakan penomoran klaster 'lab' dengan 'ref' (Hungarian pada tabel kontingensi)."""
    ref, lab = np.asarray(ref), np.asarray(lab)
    ur, ul = np.unique(ref), np.unique(lab)
    ct = np.array([[np.sum((ref == a) & (lab == b)) for b in ul] for a in ur])
    ri, ci = linear_sum_assignment(-ct)
    mapping = {ul[c]: ur[r_] for r_, c in zip(ri, ci)}
    nxt = max(ur) + 1
    out = np.empty_like(lab)
    for b in ul:
        if b not in mapping:
            mapping[b] = nxt
            nxt += 1
        out[lab == b] = mapping[b]
    return out


print("=" * 88)
print(" K-MEANS CLUSTERING - REZIM PASAR SAHAM AAPL 2026")
print("=" * 88)

# =====================================================================================
# 1. DATA UNDERSTANDING - audit struktur dan kualitas data mentah
# =====================================================================================
print("\n[1] DATA UNDERSTANDING")
raw = pd.read_csv(DATA_PATH)
n_rows_raw, n_cols_raw = raw.shape
dtypes_raw = {c: str(t) for c, t in raw.dtypes.items()}

# --- 1a. Parsing tanggal & urutan kronologis ------------------------------------------
raw["Date"] = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="raise")
was_sorted = bool(raw["Date"].is_monotonic_increasing)
raw = raw.sort_values("Date").reset_index(drop=True)
assert raw["Date"].is_monotonic_increasing and raw["Date"].is_unique

# --- 1b. Missing value, duplikat -------------------------------------------------------
missing = raw.isna().sum()
missing = {c: int(v) for c, v in missing.items() if v > 0}
dup_rows = int(raw.duplicated().sum())
dup_dates = int(raw["Date"].duplicated().sum())

# --- 1c. Hari bursa yang hilang (dibandingkan hari kerja Senin-Jumat) -------------------
#     Daftar hari libur bursa AS (NYSE/Nasdaq) 2026 dalam rentang data.
US_HOLIDAYS_2026 = {
    "2026-01-01": "New Year's Day",
    "2026-01-19": "Martin Luther King Jr. Day",
    "2026-02-16": "Washington's Birthday",
    "2026-04-03": "Good Friday",
    "2026-05-25": "Memorial Day",
    "2026-06-19": "Juneteenth",
    "2026-07-03": "Independence Day (observed)",
}
bdays = pd.bdate_range(raw["Date"].min(), raw["Date"].max())
missing_days = [d.strftime("%Y-%m-%d") for d in bdays.difference(raw["Date"])]
missing_explained = {d: US_HOLIDAYS_2026.get(d, "TIDAK TERJELASKAN") for d in missing_days}
unexplained = [d for d, v in missing_explained.items() if v == "TIDAK TERJELASKAN"]

# --- 1d. Konsistensi logis OHLC dan kolom turunan bawaan dataset ------------------------
ohlc_ok = bool(
    (
        (raw.High >= raw[["Open", "Close"]].max(axis=1))
        & (raw.Low <= raw[["Open", "Close"]].min(axis=1))
        & (raw.Volume > 0)
    ).all()
)
chk_ret = float((raw.Close.pct_change() * 100 - raw.Daily_Return_Pct).abs().max())
chk_ma7 = float((raw.Close.rolling(7, min_periods=1).mean() - raw.MA_7).abs().max())
chk_ma30 = float((raw.Close.rolling(30, min_periods=1).mean() - raw.MA_30).abs().max())
chk_ma50 = float((raw.Close.rolling(50, min_periods=1).mean() - raw.MA_50).abs().max())
chk_vol = float(
    (raw.Close.pct_change().mul(100).rolling(10, min_periods=2).std() - raw.Volatility_10D_Pct).abs().max()
)
# MA_30 & MA_50 bawaan dihitung dengan min_periods=1 (rata-rata "expanding" pada baris awal)
ma30_first_true = int(raw.index[29])  # baris pertama dengan 30 observasi penuh
# Is_Month_End: bandingkan dengan hari bursa terakhir tiap bulan yang lengkap
last_td = raw.groupby(raw.Date.dt.to_period("M")).Date.max()
last_td = last_td[last_td.index < raw.Date.max().to_period("M")]  # bulan terakhir belum selesai
flag_true = set(raw.loc[raw.Is_Month_End, "Date"].dt.strftime("%Y-%m-%d"))
month_end_missed = sorted(set(last_td.dt.strftime("%Y-%m-%d")) - flag_true)

# --- 1e. Outlier pada kolom numerik mentah (IQR 1,5 dan |z| > 3) ------------------------
RAW_NUM = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj_Close",
    "Volume",
    "Daily_Return_Pct",
    "MA_7",
    "MA_30",
    "MA_50",
    "Volatility_10D_Pct",
]


def outlier_counts(s):
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z = (s - s.mean()) / s.std(ddof=0)
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": lo,
        "upper": hi,
        "n_iqr": int(((s < lo) | (s > hi)).sum()),
        "n_z3": int((z.abs() > 3).sum()),
    }


raw_desc = {}
for c in RAW_NUM:
    d = raw[c].describe()
    raw_desc[c] = {
        "count": int(d["count"]),
        "mean": d["mean"],
        "std": d["std"],
        "min": d["min"],
        "median": d["50%"],
        "max": d["max"],
        **outlier_counts(raw[c]),
    }
vol_top = raw.nlargest(5, "Volume")[["Date", "Volume", "Daily_Return_Pct"]]

SUMMARY["data_audit"] = r(
    {
        "file": os.path.basename(DATA_PATH),
        "n_rows": n_rows_raw,
        "n_cols": n_cols_raw,
        "columns": list(raw.columns),
        "dtypes_as_read": dtypes_raw,
        "date_min": raw.Date.min(),
        "date_max": raw.Date.max(),
        "was_sorted": was_sorted,
        "missing_values": missing,
        "missing_total": int(sum(missing.values())),
        "duplicate_rows": dup_rows,
        "duplicate_dates": dup_dates,
        "business_days_in_range": len(bdays),
        "missing_business_days": missing_explained,
        "n_missing_business_days": len(missing_days),
        "n_unexplained_missing_days": len(unexplained),
        "ohlc_consistent": ohlc_ok,
        "max_abs_diff_recomputed": {
            "Daily_Return_Pct": chk_ret,
            "MA_7": chk_ma7,
            "MA_30_minperiods1": chk_ma30,
            "MA_50_minperiods1": chk_ma50,
            "Volatility_10D_Pct": chk_vol,
        },
        "is_month_end_true_dates": sorted(flag_true),
        "is_month_end_missed_last_trading_days": month_end_missed,
        "weekday_counts": raw.Day_of_Week.value_counts().to_dict(),
        "n_up_candles": int((raw.Close >= raw.Open).sum()),
        "n_down_candles": int((raw.Close < raw.Open).sum()),
        "close_first": raw.Close.iloc[0],
        "close_last": raw.Close.iloc[-1],
        "close_min": raw.Close.min(),
        "close_max": raw.Close.max(),
        "ytd_change_pct": (raw.Close.iloc[-1] / raw.Close.iloc[0] - 1) * 100,
        "volume_mean": raw.Volume.mean(),
        "volume_median": raw.Volume.median(),
        "top5_volume_days": [
            {"date": d, "volume": int(v), "ret_pct": rt} for d, v, rt in vol_top.itertuples(index=False)
        ],
        "raw_describe": raw_desc,
    },
    4,
)
print(f"   {n_rows_raw} baris x {n_cols_raw} kolom | {raw.Date.min().date()} s.d. {raw.Date.max().date()}")
print(
    f"   missing={missing} | duplikat baris={dup_rows} | hari bursa hilang={len(missing_days)} "
    f"(tak terjelaskan={len(unexplained)}) | OHLC konsisten={ohlc_ok}"
)

# =====================================================================================
# 2. FEATURE ENGINEERING - dari OHLCV ke fitur stasioner, TANPA look-ahead
# =====================================================================================
print("\n[2] FEATURE ENGINEERING")
VOL_WIN, VOLAT_WIN, MA_WIN, RSI_WIN, MOM_WIN = 20, 10, 20, 14, 5


def build_features(df):
    """Bangun 10 fitur kandidat dari OHLCV.

    Prinsip anti look-ahead: fitur hari t hanya memakai informasi sampai penutupan hari t.
    Statistik pembanding (rata-rata volume, volatilitas, moving average) dihitung dari
    jendela t-N..t-1 (memakai .shift(1)) sehingga nilai hari t tidak ikut membentuk
    pembandingnya sendiri dan tidak ada data masa depan yang terpakai.
    """
    C, O, H, L, V = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    f["log_ret"] = np.log(C / C.shift(1)) * 100  # return log harian (%)
    f["range_pct"] = (H - L) / C * 100  # rentang intraday (%)
    f["gap_pct"] = (O - C.shift(1)) / C.shift(1) * 100  # gap pembukaan (%)
    f["clv"] = (C - L) / (H - L)  # posisi close dalam rentang (0-1)
    f["vol_chg"] = np.log(V / V.shift(1))  # perubahan volume (log)
    f["rel_vol"] = V / V.shift(1).rolling(VOL_WIN).mean()  # volume relatif thd rata-rata 20 hari lalu
    f["volat_10"] = f["log_ret"].shift(1).rolling(VOLAT_WIN).std()  # volatilitas rolling 10 hari lalu (%)
    f["dist_ma20"] = (C / C.shift(1).rolling(MA_WIN).mean() - 1) * 100  # jarak harga thd MA20 (%)
    delta = C.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / RSI_WIN, adjust=False, min_periods=RSI_WIN).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_WIN, adjust=False, min_periods=RSI_WIN).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss)  # RSI Wilder 14 hari
    f["mom_5"] = np.log(C / C.shift(MOM_WIN)) * 100  # momentum 5 hari (%)
    return f


ALL_FEATS = [
    "log_ret",
    "range_pct",
    "gap_pct",
    "clv",
    "vol_chg",
    "rel_vol",
    "volat_10",
    "dist_ma20",
    "rsi_14",
    "mom_5",
]
# Enam fitur inti: satu wakil untuk tiap dimensi perilaku pasar (lihat Bab III laporan)
CORE = ["log_ret", "range_pct", "gap_pct", "rel_vol", "volat_10", "dist_ma20"]
EXTRA = [c for c in ALL_FEATS if c not in CORE]
FEAT_LABEL = {
    "log_ret": "Log return harian (%)",
    "range_pct": "Rentang intraday (%)",
    "gap_pct": "Gap pembukaan (%)",
    "clv": "Posisi close dalam rentang",
    "vol_chg": "Perubahan volume (log)",
    "rel_vol": "Volume relatif (x rata-rata 20 hari)",
    "volat_10": "Volatilitas 10 hari (%)",
    "dist_ma20": "Jarak harga ke MA20 (%)",
    "rsi_14": "RSI 14 hari",
    "mom_5": "Momentum 5 hari (%)",
}
FEAT_SHORT = {
    "log_ret": "log_ret",
    "range_pct": "range_pct",
    "gap_pct": "gap_pct",
    "clv": "clv",
    "vol_chg": "vol_chg",
    "rel_vol": "rel_vol",
    "volat_10": "volat_10",
    "dist_ma20": "dist_ma20",
    "rsi_14": "rsi_14",
    "mom_5": "mom_5",
}

feats_full = build_features(raw)

# --- 2a. UJI LOOK-AHEAD: fitur hari t harus sama walaupun data sesudah t dibuang -------
lookahead_checked = 0
for t in range(25, len(raw) + 1, 7):
    f_trunc = build_features(raw.iloc[:t])
    a, b = f_trunc.iloc[-1].values.astype(float), feats_full.iloc[t - 1].values.astype(float)
    assert np.allclose(a, b, equal_nan=True, atol=1e-10), f"Look-ahead terdeteksi pada baris {t - 1}"
    lookahead_checked += 1

# --- 2b. Gabung dan buang baris pemanasan (NaN akibat rolling) --------------------------
data_full = pd.concat([raw, feats_full], axis=1)
data_full["next_ret"] = data_full["log_ret"].shift(-1)  # HANYA untuk analisis deskriptif (bukan fitur)
nan_per_feat = feats_full.isna().sum().to_dict()
warm_mask = feats_full[ALL_FEATS].isna().any(axis=1)
n_warm = int(warm_mask.sum())
data = data_full.loc[~warm_mask].reset_index(drop=True)
N = len(data)
assert data[ALL_FEATS].isna().sum().sum() == 0 and data.duplicated(subset="Date").sum() == 0

corr_all = data[ALL_FEATS].corr()
high_corr = [
    (a, b, corr_all.loc[a, b]) for a, b in combinations(ALL_FEATS, 2) if abs(corr_all.loc[a, b]) >= 0.60
]
feat_desc = {}
for c in ALL_FEATS:
    s = data[c]
    feat_desc[c] = {
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "q1": s.quantile(0.25),
        "median": s.median(),
        "q3": s.quantile(0.75),
        "max": s.max(),
        "skew": s.skew(),
        "kurt_excess": s.kurt(),
        "variance": s.var(),
        **{k: v for k, v in outlier_counts(s).items() if k in ("lower", "upper", "n_iqr", "n_z3")},
    }

# Uji sifat deret return (untuk menilai seberapa "mirip pasar nyata" data ini)
lr = data_full["log_ret"].dropna()
sh_w, sh_p = stats.shapiro(lr)
jb_s, jb_p = stats.jarque_bera(lr)
SUMMARY["features"] = r(
    {
        "all_features": ALL_FEATS,
        "core_features": CORE,
        "extra_features": EXTRA,
        "windows": {"rel_vol": VOL_WIN, "volat": VOLAT_WIN, "ma": MA_WIN, "rsi": RSI_WIN, "mom": MOM_WIN},
        "nan_per_feature": nan_per_feat,
        "n_warmup_rows_dropped": n_warm,
        "n_clean": N,
        "clean_date_min": data.Date.min(),
        "clean_date_max": data.Date.max(),
        "lookahead_assert_checks": lookahead_checked,
        "describe": feat_desc,
        "corr": {a: corr_all.loc[a].to_dict() for a in ALL_FEATS},
        "high_corr_pairs": [
            {"a": a, "b": b, "r": v} for a, b, v in sorted(high_corr, key=lambda x: -abs(x[2]))
        ],
        "log_return_series": {
            "n": len(lr),
            "mean": lr.mean(),
            "std": lr.std(),
            "skew": lr.skew(),
            "kurt_excess": lr.kurt(),
            "min": lr.min(),
            "max": lr.max(),
            "pct_positive": (lr > 0).mean() * 100,
            "shapiro_w": sh_w,
            "shapiro_p": sh_p,
            "jarque_bera": jb_s,
            "jarque_bera_p": jb_p,
            "acf1_return": lr.autocorr(1),
            "acf1_abs_return": lr.abs().autocorr(1),
            "acf1_sq_return": (lr**2).autocorr(1),
        },
    },
    4,
)
print(
    f"   {len(ALL_FEATS)} fitur kandidat, {len(CORE)} fitur inti | baris pemanasan dibuang={n_warm} | n bersih={N}"
)
print(f"   uji look-ahead lolos pada {lookahead_checked} titik potong")

# =====================================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================================
print("\n[3] EDA")

# --- G00 Diagram alur CRISP-DM ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 4.6))
ax.axis("off")
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
steps = [
    ("1. Business\nUnderstanding", "Tujuan: memetakan\nrezim pasar harian"),
    ("2. Data\nUnderstanding", "Audit 136 baris,\nmissing, hari libur,\noutlier"),
    ("3. Data\nPreparation", "10 fitur stasioner,\nanti look-ahead,\nscaling dlm Pipeline"),
    ("4. Modeling", "K-Means (Lloyd,\nk-means++), K = 2-10,\nperhitungan manual"),
    ("5. Evaluation", "Silhouette, DBI, CH,\nGap; stabilitas ARI;\n6 algoritma pembanding"),
    ("6. Deployment", "Profil rezim, matriks\ntransisi, kode & data\nreproducible"),
]
for i, (t1, t2) in enumerate(steps):
    x = 0.15 + i * 1.98
    ax.add_patch(
        FancyBboxPatch(
            (x, 2.75),
            1.72,
            1.35,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            fc=DEEP if i % 2 == 0 else EMERALD,
            ec=GOLD,
            lw=1.4,
        )
    )
    ax.text(x + 0.86, 3.42, t1, ha="center", va="center", color="white", fontsize=11.5, fontweight="bold")
    ax.add_patch(
        FancyBboxPatch(
            (x, 0.75),
            1.72,
            1.55,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            fc=LIGHT,
            ec="#9CC9B5",
            lw=1.0,
        )
    )
    ax.text(x + 0.86, 1.52, t2, ha="center", va="center", color=BLACK, fontsize=9.6)
    ax.add_patch(
        FancyArrowPatch((x + 0.86, 2.72), (x + 0.86, 2.36), arrowstyle="-|>", mutation_scale=12, color=GREY)
    )
    if i < 5:
        ax.add_patch(
            FancyArrowPatch(
                (x + 1.76, 3.42), (x + 1.96, 3.42), arrowstyle="-|>", mutation_scale=15, color=BLACK, lw=1.4
            )
        )
ax.add_patch(
    FancyArrowPatch(
        (10.9, 4.14),
        (1.0, 4.14),
        connectionstyle="arc3,rad=0.13",
        arrowstyle="-|>",
        mutation_scale=14,
        color=GOLD,
        lw=1.5,
        ls="--",
    )
)
ax.text(
    6,
    0.32,
    "panah putus-putus: proses bersifat iteratif, hasil evaluasi dapat mengembalikan analisis ke tahap sebelumnya",
    ha="center",
    fontsize=9.6,
    color=GOLD,
    style="italic",
)
ax.set_title("Alur Penelitian Berbasis CRISP-DM", pad=6)
savefig(fig, "G00_crispdm", "Diagram alur CRISP-DM penelitian")

# --- G01 Deret waktu harga penutupan dan volume (dua panel, satu sumbu per panel) -------
ma20 = raw.Close.rolling(20).mean()
ma50 = raw.Close.rolling(50).mean()
fig, (a1, a2) = plt.subplots(
    2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.08}
)
a1.plot(raw.Date, raw.Close, color=BLACK, lw=1.8, label="Harga penutupan")
a1.plot(raw.Date, ma20, color=EMERALD, lw=1.6, label="MA 20 hari")
a1.plot(raw.Date, ma50, color=GOLD, lw=1.6, ls="--", label="MA 50 hari")
a1.axvspan(raw.Date.iloc[0], data.Date.iloc[0], color="#F3F4F6", zorder=0)
a1.text(
    raw.Date.iloc[1],
    raw.Close.max() * 0.985,
    "masa pemanasan fitur\n(20 hari, tidak diklaster)",
    fontsize=9,
    color=GREY,
    va="top",
)
a1.set_ylabel("Harga (USD)")
a1.legend(loc="lower right", ncol=3)
a1.set_title("Harga Penutupan dan Volume Perdagangan AAPL, 2 Januari - 20 Juli 2026")
vmean = raw.Volume.mean()
a2.bar(
    raw.Date,
    raw.Volume / 1e6,
    color=np.where(raw.Volume > 1.5 * raw.Volume.median(), GOLD, EMERALD),
    width=0.9,
)
a2.axhline(vmean / 1e6, color=BLACK, lw=1, ls=":")
a2.text(raw.Date.iloc[-1], vmean / 1e6 * 1.12, f"rata-rata {vmean / 1e6:.1f} juta", ha="right", fontsize=9)
a2.set_ylabel("Volume (juta lembar)")
a2.legend(
    handles=[Patch(color=EMERALD, label="volume normal"), Patch(color=GOLD, label="> 1,5 x median")],
    loc="upper right",
    ncol=2,
)
a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
savefig(fig, "G01_harga_volume", "Deret waktu harga penutupan dan volume AAPL 2026")

# --- G02 Candlestick ringkas -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.6))
x = np.arange(len(raw))
up = raw.Close >= raw.Open
ax.vlines(x, raw.Low, raw.High, color=np.where(up, EMERALD, RUST), lw=0.9)
ax.bar(
    x,
    (raw.Close - raw.Open).abs().clip(lower=0.15),
    bottom=raw[["Open", "Close"]].min(axis=1),
    color=np.where(up, EMERALD, RUST),
    width=0.7,
)
ticks = [i for i in range(len(raw)) if i == 0 or raw.Date[i].month != raw.Date[i - 1].month]
ax.set_xticks(ticks)
ax.set_xticklabels([raw.Date[i].strftime("%b %Y") for i in ticks])
ax.set_ylabel("Harga (USD)")
ax.set_xlim(-1, len(raw))
ax.legend(
    handles=[
        Patch(color=EMERALD, label=f"close >= open ({int(up.sum())} hari)"),
        Patch(color=RUST, label=f"close < open ({int((~up).sum())} hari)"),
    ],
    loc="upper left",
)
ax.set_title("Grafik Candlestick OHLC Harian AAPL 2026 (136 Hari Perdagangan)")
savefig(fig, "G02_candlestick", "Candlestick OHLC harian AAPL 2026")

# --- G03 Distribusi return harian --------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
a1.hist(lr, bins=22, color=EMERALD, edgecolor="white", density=True, alpha=0.9, label="log return harian")
xs = np.linspace(lr.min() - 0.5, lr.max() + 0.5, 300)
a1.plot(
    xs, stats.norm.pdf(xs, lr.mean(), lr.std()), color=BLACK, lw=2, label="kurva normal (mean, sd sampel)"
)
a1.axvline(lr.mean(), color=GOLD, lw=1.8, ls="--", label=f"rata-rata = {lr.mean():.3f}%")
a1.set_xlabel("Log return harian (%)")
a1.set_ylabel("Kepadatan")
a1.legend(loc="upper left", fontsize=9)
a1.set_title("Histogram Log Return Harian")
(osm, osr), (slope, inter, rr) = stats.probplot(lr, dist="norm")
a2.scatter(osm, osr, s=16, color=EMERALD, edgecolor="white", lw=0.4)
a2.plot(osm, slope * osm + inter, color=BLACK, lw=1.6)
a2.set_xlabel("Kuantil teoretis normal")
a2.set_ylabel("Kuantil sampel (%)")
a2.set_title(f"Q-Q Plot (Shapiro-Wilk p = {sh_p:.3f})")
savefig(fig, "G03_distribusi_return", "Distribusi log return harian dan Q-Q plot")

# --- G04 Histogram 10 fitur kandidat -----------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(13, 5.4))
for ax, c in zip(axes.ravel(), ALL_FEATS):
    ax.hist(data[c], bins=18, color=EMERALD if c in CORE else GREY, edgecolor="white")
    ax.axvline(data[c].mean(), color=GOLD, lw=1.6, ls="--")
    ax.set_title(c + ("" if c in CORE else " (tambahan)"), fontsize=11)
    ax.tick_params(labelsize=8.5)
fig.suptitle(
    "Histogram 10 Fitur Hasil Rekayasa (hijau = fitur inti, abu-abu = fitur tambahan; garis emas = rata-rata)",
    fontsize=12.5,
    fontweight="bold",
    y=1.0,
)
fig.tight_layout()
savefig(fig, "G04_histogram_fitur", "Histogram sepuluh fitur hasil rekayasa")

# --- G05 Boxplot fitur terstandar + outlier ----------------------------------------------
zall = (data[ALL_FEATS] - data[ALL_FEATS].mean()) / data[ALL_FEATS].std(ddof=0)
fig, ax = plt.subplots(figsize=(11, 4.4))
bp = boxplot_compat(
    ax,
    [zall[c] for c in ALL_FEATS],
    ALL_FEATS,
    patch_artist=True,
    widths=0.55,
    flierprops=dict(
        marker="o", markersize=5, markerfacecolor=GOLD, markeredgecolor=BLACK, markeredgewidth=0.5
    ),
    medianprops=dict(color=BLACK, lw=1.6),
)
for patch, c in zip(bp["boxes"], ALL_FEATS):
    patch.set_facecolor(EMERALD if c in CORE else "#C7CCD1")
    patch.set_alpha(0.85)
for i, c in enumerate(ALL_FEATS):
    ax.text(
        i + 1,
        zall.max().max() + 0.35,
        f"IQR: {feat_desc[c]['n_iqr']}\n|z|>3: {feat_desc[c]['n_z3']}",
        ha="center",
        fontsize=8.6,
        color=BLACK,
    )
ax.axhline(3, color=RUST, lw=0.9, ls=":")
ax.axhline(-3, color=RUST, lw=0.9, ls=":")
ax.set_ylim(zall.min().min() - 0.4, zall.max().max() + 1.5)
ax.set_ylabel("Nilai terstandar (z-score)")
ax.set_title("Boxplot Fitur Terstandar dan Jumlah Outlier (titik emas = di luar pagar 1,5 x IQR)")
savefig(fig, "G05_boxplot_outlier", "Boxplot fitur terstandar dan jumlah outlier IQR serta z-score")

# --- G06 Heatmap korelasi ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 6.6))
cmap = sns.diverging_palette(25, 150, s=85, l=40, as_cmap=True)
sns.heatmap(
    corr_all,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=1.5,
    linecolor="white",
    annot_kws={"size": 9},
    cbar_kws={"shrink": 0.8, "label": "Korelasi Pearson"},
    ax=ax,
)
ax.grid(False)
ax.set_title("Heatmap Korelasi Pearson antar 10 Fitur Kandidat (n = %d)" % N)
ax.tick_params(axis="x", rotation=40)
savefig(fig, "G06_heatmap_korelasi", "Heatmap korelasi antarfitur")

# --- G07 Pairplot fitur inti -------------------------------------------------------------
pg = sns.pairplot(
    data[CORE],
    corner=True,
    height=1.75,
    plot_kws=dict(s=14, color=EMERALD, edgecolor="white", linewidth=0.3, alpha=0.85),
    diag_kws=dict(color=DEEP, edgecolor="white", bins=15),
)
pg.fig.suptitle("Pairplot Enam Fitur Inti (sebelum scaling)", fontsize=13, fontweight="bold", y=1.01)
savefig(pg.fig, "G07_pairplot", "Pairplot enam fitur inti")

# =====================================================================================
# 4. PERCOBAAN PEMBANDING: MENGAPA TIDAK MENGKLASTER HARGA MENTAH
# =====================================================================================
print("\n[4] BUKTI: CLUSTERING HARGA MENTAH HANYA MENGIKUTI WAKTU")
K_DEMO = 3
RAW_PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


def temporal_stats(labels):
    """Seberapa 'menempel pada waktu' sebuah partisi: jumlah perpindahan, panjang run, eta^2 waktu."""
    labels = np.asarray(labels)
    switches = int((labels[1:] != labels[:-1]).sum())
    runs = np.diff(np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True]))
    t = np.arange(len(labels))
    quart = pd.qcut(t, len(np.unique(labels)), labels=False)
    return {
        "n_switches": switches,
        "n_runs": int(len(runs)),
        "mean_run_length": float(runs.mean()),
        "max_run_length": int(runs.max()),
        "eta2_time_index": eta_squared(t, labels),
        "nmi_vs_time_blocks": normalized_mutual_info_score(quart, labels),
        "ari_vs_time_blocks": adjusted_rand_score(quart, labels),
    }


X_rawprice = StandardScaler().fit_transform(data[RAW_PRICE_COLS])
km_rawprice = KMeans(K_DEMO, n_init=20, random_state=RANDOM_STATE).fit(X_rawprice)
X_core_std = StandardScaler().fit_transform(data[CORE])
km_feat_demo = KMeans(K_DEMO, n_init=20, random_state=RANDOM_STATE).fit(X_core_std)
# urutkan label harga-mentah menurut rata-rata Close agar warna runtut
order = data.groupby(km_rawprice.labels_).Close.mean().sort_values().index
lab_rawprice = pd.Series(km_rawprice.labels_).map({o: i for i, o in enumerate(order)}).values
ts_raw, ts_feat = temporal_stats(lab_rawprice), temporal_stats(km_feat_demo.labels_)
adf_note = float(np.corrcoef(np.arange(N), data.Close)[0, 1])
SUMMARY["raw_price_experiment"] = r(
    {
        "k": K_DEMO,
        "raw_price_columns": RAW_PRICE_COLS,
        "raw_price": {
            **ts_raw,
            "silhouette": silhouette_score(X_rawprice, lab_rawprice),
            "sizes": np.bincount(lab_rawprice).tolist(),
            "close_range_per_cluster": [
                [data.Close[lab_rawprice == k].min(), data.Close[lab_rawprice == k].max()]
                for k in range(K_DEMO)
            ],
            "eta2_close": eta_squared(data.Close, lab_rawprice),
        },
        "engineered_features": {
            **ts_feat,
            "silhouette": silhouette_score(X_core_std, km_feat_demo.labels_),
            "sizes": np.bincount(km_feat_demo.labels_).tolist(),
            "eta2_close": eta_squared(data.Close, km_feat_demo.labels_),
        },
        "corr_close_vs_time": adf_note,
        "ari_between_two_partitions": adjusted_rand_score(lab_rawprice, km_feat_demo.labels_),
    },
    4,
)
print(
    f"   harga mentah: perpindahan={ts_raw['n_switches']}, eta2 waktu={ts_raw['eta2_time_index']:.3f} | "
    f"fitur: perpindahan={ts_feat['n_switches']}, eta2 waktu={ts_feat['eta2_time_index']:.3f}"
)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"hspace": 0.22})
for ax, lab, ttl, st in [
    (a1, lab_rawprice, "K-Means pada harga mentah (Open, High, Low, Close, Volume)", ts_raw),
    (a2, km_feat_demo.labels_, "K-Means pada enam fitur stasioner hasil rekayasa", ts_feat),
]:
    ax.plot(data.Date, data.Close, color="#9CA3AF", lw=1, zorder=1)
    for k in range(K_DEMO):
        m = lab == k
        ax.scatter(
            data.Date[m],
            data.Close[m],
            s=30,
            color=PALETTE[k],
            marker=MARKERS[k],
            edgecolor="white",
            lw=0.4,
            zorder=2,
            label=f"klaster {k + 1}",
        )
    ax.set_title(
        f"{ttl}\nperpindahan klaster = {st['n_switches']} kali; rata-rata bertahan = {st['mean_run_length']:.1f} hari; "
        f"eta² terhadap waktu = {st['eta2_time_index']:.2f}",
        fontsize=11.5,
    )
    ax.set_ylabel("Harga penutupan (USD)")
a1.legend(ncol=4, loc="upper left")
a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
savefig(fig, "G08_harga_mentah_vs_fitur", "Percobaan pembanding: klaster harga mentah mengikuti waktu")

# =====================================================================================
# 5. PENANGANAN OUTLIER - dideteksi, TIDAK langsung dibuang, lalu dibandingkan
# =====================================================================================
print("\n[5] OUTLIER")


class IQRWinsorizer(BaseEstimator, TransformerMixin):
    """Memangkas (winsorize) nilai di luar pagar Q1 - k*IQR dan Q3 + k*IQR.
    Batas dipelajari saat fit sehingga aman dipakai di dalam Pipeline."""

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        q1, q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
        self.lower_, self.upper_ = q1 - self.k * (q3 - q1), q3 + self.k * (q3 - q1)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lower_, self.upper_)


Xc = data[CORE].values
zc = np.abs((Xc - Xc.mean(0)) / Xc.std(0))
iqr_w = IQRWinsorizer().fit(Xc)
iqr_flag = (Xc < iqr_w.lower_) | (Xc > iqr_w.upper_)
out_iqr_rows, out_z_rows = iqr_flag.any(1), (zc > 3).any(1)
outlier_days = []
for i in np.flatnonzero(out_iqr_rows):
    outlier_days.append(
        {
            "date": data.Date[i],
            "features_iqr": [CORE[j] for j in np.flatnonzero(iqr_flag[i])],
            "max_abs_z": zc[i].max(),
            "z3": bool(out_z_rows[i]),
            **{c: data[c][i] for c in CORE},
            "volume": int(data.Volume[i]),
        }
    )


def fit_km(X, k, seed=RANDOM_STATE, n_init=20, init="k-means++"):
    return KMeans(n_clusters=k, init=init, n_init=n_init, random_state=seed).fit(X)


K_WORK = 3  # K kerja untuk percobaan praproses; pembenaran K ada di bagian 7
base_pipe = Pipeline(
    [("scaler", StandardScaler()), ("kmeans", KMeans(K_WORK, n_init=20, random_state=RANDOM_STATE))]
).fit(Xc)
lab_base = base_pipe.named_steps["kmeans"].labels_
X_std = base_pipe.named_steps["scaler"].transform(Xc)
# (a) winsorize IQR di dalam Pipeline
wins_pipe = Pipeline(
    [
        ("winsor", IQRWinsorizer(1.5)),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(K_WORK, n_init=20, random_state=RANDOM_STATE)),
    ]
).fit(Xc)
lab_wins = wins_pipe.named_steps["kmeans"].labels_
X_wins = wins_pipe[:-1].transform(Xc)
# (b) baris outlier dibuang (|z| > 3)
keep = ~out_z_rows
drop_pipe = Pipeline(
    [("scaler", StandardScaler()), ("kmeans", KMeans(K_WORK, n_init=20, random_state=RANDOM_STATE))]
).fit(Xc[keep])
lab_drop = drop_pipe.named_steps["kmeans"].labels_
# (c) baris outlier IQR dibuang
keep2 = ~out_iqr_rows
drop2_pipe = Pipeline(
    [("scaler", StandardScaler()), ("kmeans", KMeans(K_WORK, n_init=20, random_state=RANDOM_STATE))]
).fit(Xc[keep2])
lab_drop2 = drop2_pipe.named_steps["kmeans"].labels_


def centroid_shift(c_ref, c_new):
    """Rata-rata & maksimum jarak antar centroid sepadan (dipasangkan dgn Hungarian), skala terstandar."""
    D = cdist(c_ref, c_new)
    ri, ci = linear_sum_assignment(D)
    return float(D[ri, ci].mean()), float(D[ri, ci].max())


sc0 = base_pipe.named_steps["scaler"]
cen_base = base_pipe.named_steps["kmeans"].cluster_centers_
cen_wins = sc0.transform(
    wins_pipe.named_steps["scaler"].inverse_transform(wins_pipe.named_steps["kmeans"].cluster_centers_)
)
cen_drop = sc0.transform(
    drop_pipe.named_steps["scaler"].inverse_transform(drop_pipe.named_steps["kmeans"].cluster_centers_)
)
cen_drop2 = sc0.transform(
    drop2_pipe.named_steps["scaler"].inverse_transform(drop2_pipe.named_steps["kmeans"].cluster_centers_)
)
out_cluster_of_outliers = pd.Series(lab_base[out_iqr_rows]).value_counts().to_dict()
SUMMARY["outliers"] = r(
    {
        "per_feature_iqr": {c: int(iqr_flag[:, j].sum()) for j, c in enumerate(CORE)},
        "per_feature_z3": {c: int((zc[:, j] > 3).sum()) for j, c in enumerate(CORE)},
        "iqr_bounds": {c: [iqr_w.lower_[j], iqr_w.upper_[j]] for j, c in enumerate(CORE)},
        "n_rows_iqr": int(out_iqr_rows.sum()),
        "n_rows_z3": int(out_z_rows.sum()),
        "outlier_days": outlier_days,
        "comparison": {
            "no_handling": {
                "n": N,
                "silhouette": silhouette_score(X_std, lab_base),
                "dbi": davies_bouldin_score(X_std, lab_base),
                "ch": calinski_harabasz_score(X_std, lab_base),
                "sizes": sorted(np.bincount(lab_base).tolist(), reverse=True),
            },
            "winsorize_iqr": {
                "n": N,
                "silhouette": silhouette_score(X_wins, lab_wins),
                "dbi": davies_bouldin_score(X_wins, lab_wins),
                "ch": calinski_harabasz_score(X_wins, lab_wins),
                "sizes": sorted(np.bincount(lab_wins).tolist(), reverse=True),
                "ari_vs_no_handling": adjusted_rand_score(lab_base, lab_wins),
                "n_days_changed_cluster": int((match_labels(lab_base, lab_wins) != lab_base).sum()),
                "centroid_shift_mean_max": centroid_shift(cen_base, cen_wins),
            },
            "drop_z3": {
                "n": int(keep.sum()),
                "silhouette": silhouette_score(
                    drop_pipe.named_steps["scaler"].transform(Xc[keep]), lab_drop
                ),
                "dbi": davies_bouldin_score(drop_pipe.named_steps["scaler"].transform(Xc[keep]), lab_drop),
                "ch": calinski_harabasz_score(
                    drop_pipe.named_steps["scaler"].transform(Xc[keep]), lab_drop
                ),
                "sizes": sorted(np.bincount(lab_drop).tolist(), reverse=True),
                "ari_vs_no_handling_common_rows": adjusted_rand_score(lab_base[keep], lab_drop),
                "centroid_shift_mean_max": centroid_shift(cen_base, cen_drop),
            },
            "drop_iqr": {
                "n": int(keep2.sum()),
                "silhouette": silhouette_score(
                    drop2_pipe.named_steps["scaler"].transform(Xc[keep2]), lab_drop2
                ),
                "dbi": davies_bouldin_score(
                    drop2_pipe.named_steps["scaler"].transform(Xc[keep2]), lab_drop2
                ),
                "ch": calinski_harabasz_score(
                    drop2_pipe.named_steps["scaler"].transform(Xc[keep2]), lab_drop2
                ),
                "sizes": sorted(np.bincount(lab_drop2).tolist(), reverse=True),
                "ari_vs_no_handling_common_rows": adjusted_rand_score(lab_base[keep2], lab_drop2),
                "centroid_shift_mean_max": centroid_shift(cen_base, cen_drop2),
            },
        },
    },
    4,
)
print(
    f"   baris outlier IQR={int(out_iqr_rows.sum())}, |z|>3={int(out_z_rows.sum())} | "
    f"ARI winsorize vs tanpa penanganan={adjusted_rand_score(lab_base, lab_wins):.3f}"
)

# =====================================================================================
# 6. PENGARUH FEATURE SCALING
# =====================================================================================
print("\n[6] FEATURE SCALING")
SCALERS = {
    "Tanpa scaling": FunctionTransformer(),
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
}
scal_res, scal_lab, scal_X = {}, {}, {}
for name, sc in SCALERS.items():
    pipe = Pipeline([("scaler", sc), ("kmeans", KMeans(K_WORK, n_init=20, random_state=RANDOM_STATE))]).fit(
        Xc
    )
    Xt = pipe.named_steps["scaler"].transform(Xc)
    lab = pipe.named_steps["kmeans"].labels_
    scal_lab[name], scal_X[name] = lab, Xt
    var_share = Xt.var(0) / Xt.var(0).sum() * 100
    scal_res[name] = {
        "inertia_own_space": pipe.named_steps["kmeans"].inertia_,
        "silhouette_own_space": silhouette_score(Xt, lab),
        "dbi_own_space": davies_bouldin_score(Xt, lab),
        "ch_own_space": calinski_harabasz_score(Xt, lab),
        "silhouette_in_standard_space": silhouette_score(X_std, lab),
        "dbi_in_standard_space": davies_bouldin_score(X_std, lab),
        "sizes": sorted(np.bincount(lab).tolist(), reverse=True),
        "variance_share_pct": {c: var_share[j] for j, c in enumerate(CORE)},
        "eta2_per_feature": {c: eta_squared(Xc[:, j], lab) for j, c in enumerate(CORE)},
        "feature_range_after": {c: [Xt[:, j].min(), Xt[:, j].max()] for j, c in enumerate(CORE)},
    }
for name in SCALERS:
    scal_res[name]["ari_vs_standard"] = adjusted_rand_score(scal_lab["StandardScaler"], scal_lab[name])
    e = scal_res[name]["eta2_per_feature"]
    scal_res[name]["dominant_feature"] = max(e, key=e.get)
    scal_res[name]["n_features_eta2_ge_0.30"] = int(sum(v >= 0.30 for v in e.values()))
SUMMARY["scaling"] = r(
    {
        "k": K_WORK,
        "results": scal_res,
        "raw_variance": {c: float(np.var(Xc[:, j], ddof=1)) for j, c in enumerate(CORE)},
        "raw_variance_ratio_max_min": float(Xc.var(0).max() / Xc.var(0).min()),
    },
    4,
)
for n_, v in scal_res.items():
    print(
        f"   {n_:15s} sil(ruang sendiri)={v['silhouette_own_space']:.3f} sil(ruang standar)={v['silhouette_in_standard_space']:.3f} "
        f"ARI vs Std={v['ari_vs_standard']:.3f} dominan={v['dominant_feature']}"
    )

# --- G09 Sebelum-sesudah scaling ---------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(13, 4.0))
for ax, (name, Xt) in zip(axes, scal_X.items()):
    bp = boxplot_compat(
        ax,
        [Xt[:, j] for j in range(len(CORE))],
        CORE,
        patch_artist=True,
        widths=0.6,
        flierprops=dict(
            marker="o", markersize=3.5, markerfacecolor=GOLD, markeredgecolor=BLACK, markeredgewidth=0.4
        ),
        medianprops=dict(color=BLACK, lw=1.4),
    )
    for p in bp["boxes"]:
        p.set_facecolor(EMERALD)
        p.set_alpha(0.85)
    ax.set_title(name, fontsize=11.5)
    ax.tick_params(axis="x", rotation=55, labelsize=8.8)
fig.suptitle("Sebaran Enam Fitur Inti Sebelum dan Sesudah Scaling", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
savefig(fig, "G09_scaling_sebelum_sesudah", "Sebaran fitur sebelum dan sesudah scaling")

# --- G10 Pengaruh scaling terhadap klaster -----------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={"width_ratios": [1.25, 1]})
E = pd.DataFrame({n_: v["eta2_per_feature"] for n_, v in scal_res.items()}).T[CORE]
sns.heatmap(
    E,
    annot=True,
    fmt=".2f",
    cmap=sns.light_palette(EMERALD, as_cmap=True),
    vmin=0,
    vmax=1,
    linewidths=1.5,
    linecolor="white",
    cbar_kws={"label": "eta² (porsi varians fitur yang dijelaskan klaster)"},
    ax=a1,
    annot_kws={"size": 10},
)
a1.grid(False)
a1.set_title('Fitur mana yang "mengendalikan" klaster?')
a1.tick_params(axis="x", rotation=30)
a1.tick_params(axis="y", rotation=0)
names = list(scal_res)
v_own = [scal_res[n_]["silhouette_own_space"] for n_ in names]
v_std = [scal_res[n_]["silhouette_in_standard_space"] for n_ in names]
xx = np.arange(len(names))
b1 = a2.bar(xx - 0.2, v_own, 0.38, color=GOLD, label="di ruang fitur masing-masing")
b2 = a2.bar(xx + 0.2, v_std, 0.38, color=EMERALD, label="dievaluasi ulang di ruang terstandar")
for b in list(b1) + list(b2):
    a2.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() + 0.008,
        f"{b.get_height():.3f}",
        ha="center",
        fontsize=9,
    )
a2.set_xticks(xx)
a2.set_xticklabels(
    [n_.replace("Scaler", "\nScaler").replace("Tanpa ", "Tanpa\n") for n_ in names], fontsize=9.5
)
a2.set_ylabel(f"Silhouette score (K = {K_WORK})")
a2.set_ylim(0, max(v_own) * 1.3)
a2.legend(loc="upper right", fontsize=9)
a2.set_title("Skor validasi menurut metode scaling")
fig.tight_layout()
savefig(fig, "G10_pengaruh_scaling", "Pengaruh scaling terhadap pengendali klaster dan skor silhouette")

# =====================================================================================
# 7. PEMILIHAN JUMLAH KLASTER K (K = 2..10)
# =====================================================================================
print("\n[7] PEMILIHAN K")
X = X_std.copy()  # ruang kerja: 6 fitur inti, StandardScaler
K_RANGE = list(range(2, 11))
N_INIT_SEL = 50


def gap_statistic(X, ks, B=100, seed=RANDOM_STATE):
    """Gap statistic (Tibshirani dkk., 2001): bandingkan log(W_k) data dengan rata-rata log(W_k)
    dari B himpunan acuan seragam pada kotak pembatas data."""
    rng = np.random.default_rng(seed)
    lo, hi = X.min(0), X.max(0)
    refs = [rng.uniform(lo, hi, size=X.shape) for _ in range(B)]
    out = {}
    for k in ks:
        wk = np.log(inertia[k])
        wb = np.array([np.log(KMeans(k, n_init=5, random_state=seed).fit(ref).inertia_) for ref in refs])
        out[k] = {
            "log_wk": wk,
            "exp_log_wk": wb.mean(),
            "gap": wb.mean() - wk,
            "sk": wb.std(ddof=0) * np.sqrt(1 + 1 / B),
        }
    return out


inertia = {1: float(((X - X.mean(0)) ** 2).sum())}
ksel, labels_k, centers_k, landscape = {}, {}, {}, {}
N_LAND = 300  # jumlah inisialisasi tunggal untuk memetakan minimum lokal
for k in K_RANGE:
    # (i) petakan "lanskap" solusi: 300 run n_init=1 -> solusi terbaik yang diketahui untuk K ini
    singles = [KMeans(k, n_init=1, random_state=s).fit(X) for s in range(N_LAND)]
    best_m = min(singles, key=lambda m: m.inertia_)
    labels_k[k], centers_k[k], inertia[k] = best_m.labels_, best_m.cluster_centers_, best_m.inertia_
    iners = np.array([m.inertia_ for m in singles])
    aris = np.array([adjusted_rand_score(best_m.labels_, m.labels_) for m in singles])
    rival = [
        (m.inertia_, a) for m, a in zip(singles, aris) if a < 0.80
    ]  # solusi yang "berbeda secara substansial"
    rival_best = min(rival) if rival else None
    # (ii) reprodusibilitas pada n_init = 50: apakah 20 random_state menghasilkan partisi yang sama?
    rep = [KMeans(k, n_init=N_INIT_SEL, random_state=s).fit(X) for s in range(20)]
    rep_ari = np.array([adjusted_rand_score(best_m.labels_, m.labels_) for m in rep])
    landscape[k] = {
        "n_runs": N_LAND,
        "best_inertia": best_m.inertia_,
        "n_distinct_local_minima": int(len(np.unique(np.round(iners, 2)))),
        "pct_single_runs_reach_best": float((iners <= best_m.inertia_ * (1 + 1e-6)).mean() * 100),
        "pct_single_runs_ari_ge_0.95": float((aris >= 0.95).mean() * 100),
        "single_run_ari_mean": float(aris.mean()),
        "single_run_ari_min": float(aris.min()),
        "inertia_cv_pct": float(iners.std(ddof=1) / iners.mean() * 100),
        "rival_exists": rival_best is not None,
        "rival_inertia": None if rival_best is None else float(rival_best[0]),
        "rival_excess_inertia_pct": None
        if rival_best is None
        else float((rival_best[0] / best_m.inertia_ - 1) * 100),
        "rival_ari_vs_best": None if rival_best is None else float(rival_best[1]),
        "ninit50_pct_identical": float((rep_ari > 0.9999).mean() * 100),
        "ninit50_ari_mean": float(rep_ari.mean()),
        "ninit50_ari_min": float(rep_ari.min()),
    }
    ss = silhouette_samples(X, best_m.labels_)
    ksel[k] = {
        "inertia": best_m.inertia_,
        "silhouette": ss.mean(),
        "dbi": davies_bouldin_score(X, best_m.labels_),
        "ch": calinski_harabasz_score(X, best_m.labels_),
        "sizes": sorted(np.bincount(best_m.labels_).tolist(), reverse=True),
        "min_cluster_size": int(np.bincount(best_m.labels_).min()),
        "pct_negative_silhouette": float((ss < 0).mean() * 100),
        "init_stability_ari_mean": float(aris.mean()),
        "init_stability_ari_min": float(aris.min()),
        "ninit50_pct_identical": landscape[k]["ninit50_pct_identical"],
    }
gap = gap_statistic(X, range(1, 11), B=100)
for k in K_RANGE:
    ksel[k].update({"gap": gap[k]["gap"], "gap_sk": gap[k]["sk"]})
# Aturan Tibshirani: K terkecil dengan Gap(K) >= Gap(K+1) - s(K+1)
gap_k = next((k for k in range(1, 10) if gap[k]["gap"] >= gap[k + 1]["gap"] - gap[k + 1]["sk"]), 10)
gap_kmax = max(gap, key=lambda k: gap[k]["gap"])
# Siku (elbow): jarak tegak lurus terbesar dari tali yang menghubungkan titik K=1 dan K=10
ks_all = np.array(sorted(inertia))
w = np.array([inertia[k] for k in ks_all])
xn, yn = (ks_all - ks_all[0]) / (ks_all[-1] - ks_all[0]), (w - w[-1]) / (w[0] - w[-1])
elbow_dist = np.abs(xn + yn - 1) / np.sqrt(2)
elbow_k = int(ks_all[np.argmax(elbow_dist)])
drop_pct = {int(k): float((inertia[k - 1] - inertia[k]) / inertia[k - 1] * 100) for k in ks_all[1:]}
best = {
    "silhouette": max(K_RANGE, key=lambda k: ksel[k]["silhouette"]),
    "dbi": min(K_RANGE, key=lambda k: ksel[k]["dbi"]),
    "ch": max(K_RANGE, key=lambda k: ksel[k]["ch"]),
    "gap_tibshirani_rule": gap_k,
    "gap_global_max": gap_kmax,
    "elbow": elbow_k,
    "reproducibility_ninit50": max(
        K_RANGE, key=lambda k: (ksel[k]["ninit50_pct_identical"], ksel[k]["init_stability_ari_mean"])
    ),
}
# Hubungan solusi K=3 dan K=4 (apakah K=4 hanya membelah satu klaster K=3?)
ct34 = pd.crosstab(labels_k[3], labels_k[4])
SUMMARY["k_selection"] = r(
    {
        "n_init": N_INIT_SEL,
        "inertia_k1": inertia[1],
        "per_k": ksel,
        "gap_all": gap,
        "inertia_drop_pct": drop_pct,
        "elbow_distance": {int(k): float(d) for k, d in zip(ks_all, elbow_dist)},
        "best_k_by_method": best,
        "landscape": landscape,
        "ari_k3_vs_k4": adjusted_rand_score(labels_k[3], labels_k[4]),
        "crosstab_k3_k4": ct34.values.tolist(),
    },
    4,
)
print("   K terbaik per metode:", best)

# --- G11 Elbow ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.plot(ks_all, w, "-o", color=EMERALD, lw=2, ms=7, markeredgecolor="white")
ax.plot([ks_all[0], ks_all[-1]], [w[0], w[-1]], color=GREY, lw=1, ls=":")
ax.scatter([elbow_k], [inertia[elbow_k]], s=190, facecolor="none", edgecolor=GOLD, lw=2.4, zorder=5)
ax.annotate(
    f"siku geometris: K = {elbow_k}",
    (elbow_k, inertia[elbow_k]),
    (elbow_k + 1.1, inertia[elbow_k] + 75),
    arrowprops=dict(arrowstyle="->", color=BLACK),
    fontsize=10.5,
)
for k in ks_all[1:]:
    ax.text(k, inertia[k] + 16, f"-{drop_pct[int(k)]:.0f}%", ha="center", fontsize=8.6, color=GREY)
ax.set_xlabel("Jumlah klaster K")
ax.set_ylabel("Inertia / WCSS")
ax.set_xticks(ks_all)
ax.set_title("Kurva Elbow: Inertia (WCSS) terhadap K (angka abu-abu = penurunan dari K sebelumnya)")
savefig(fig, "G11_elbow", "Kurva elbow inertia (WCSS) terhadap K")

# --- G12 Silhouette vs K -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.2))
sv = [ksel[k]["silhouette"] for k in K_RANGE]
ax.plot(K_RANGE, sv, "-o", color=EMERALD, lw=2, ms=7, markeredgecolor="white")
ax.scatter([best["silhouette"]], [max(sv)], s=190, facecolor="none", edgecolor=GOLD, lw=2.4, zorder=5)
for k, v in zip(K_RANGE, sv):
    ax.text(k, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
ax.axhline(0.25, color=RUST, lw=1, ls="--")
ax.text(10, 0.252, '0,25: batas "struktur lemah" (Kaufman & Rousseeuw)', ha="right", fontsize=9, color=RUST)
ax.set_ylim(min(sv) - 0.03, 0.275)
ax.set_xticks(K_RANGE)
ax.set_xlabel("Jumlah klaster K")
ax.set_ylabel("Rata-rata silhouette")
ax.set_title("Silhouette Score terhadap K")
savefig(fig, "G12_silhouette_vs_k", "Silhouette score terhadap K")

# --- G13 Silhouette plot per klaster (K = 2..5) -------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.6), sharex=True)
for ax, k in zip(axes, [2, 3, 4, 5]):
    lab = labels_k[k]
    ss = silhouette_samples(X, lab)
    y0 = 6
    for c in np.argsort(-np.bincount(lab)):
        v = np.sort(ss[lab == c])
        col = PALETTE[list(np.argsort(-np.bincount(lab))).index(c)]
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=col, alpha=0.9)
        ax.text(-0.19, y0 + len(v) / 2, f"n={len(v)}", fontsize=8.6, va="center")
        y0 += len(v) + 6
    ax.axvline(ss.mean(), color=BLACK, ls="--", lw=1.3)
    ax.set_title(
        f"K = {k} | rata-rata = {ss.mean():.3f}\nnegatif = {(ss < 0).mean() * 100:.1f}%", fontsize=11
    )
    ax.set_yticks([])
    ax.set_xlim(-0.22, 0.62)
    ax.set_xlabel("Koefisien silhouette")
fig.suptitle(
    "Silhouette Plot per Klaster untuk K = 2 sampai 5 (garis putus-putus = rata-rata)",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
savefig(fig, "G13_silhouette_plot", "Silhouette plot per klaster untuk K = 2 sampai 5")

# --- G14 Davies-Bouldin & Calinski-Harabasz ------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.1))
dv = [ksel[k]["dbi"] for k in K_RANGE]
cv_ = [ksel[k]["ch"] for k in K_RANGE]
a1.plot(K_RANGE, dv, "-o", color=RUST, lw=2, ms=7, markeredgecolor="white")
a1.scatter([best["dbi"]], [min(dv)], s=190, facecolor="none", edgecolor=GOLD, lw=2.4, zorder=5)
a1.set_title("Davies-Bouldin Index (lebih kecil lebih baik)")
a1.set_xlabel("Jumlah klaster K")
a1.set_ylabel("DBI")
a1.set_xticks(K_RANGE)
a2.plot(K_RANGE, cv_, "-o", color=EMERALD, lw=2, ms=7, markeredgecolor="white")
a2.scatter([best["ch"]], [max(cv_)], s=190, facecolor="none", edgecolor=GOLD, lw=2.4, zorder=5)
a2.set_title("Calinski-Harabasz Index (lebih besar lebih baik)")
a2.set_xlabel("Jumlah klaster K")
a2.set_ylabel("CH")
a2.set_xticks(K_RANGE)
for k, v in zip(K_RANGE, dv):
    a1.text(k, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.8)
for k, v in zip(K_RANGE, cv_):
    a2.text(k, v + 0.35, f"{v:.1f}", ha="center", fontsize=8.8)
fig.tight_layout()
savefig(fig, "G14_dbi_ch", "Davies-Bouldin dan Calinski-Harabasz index terhadap K")

# --- G15 Gap statistic ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.2))
gk = list(range(1, 11))
gv = [gap[k]["gap"] for k in gk]
gs = [gap[k]["sk"] for k in gk]
ax.errorbar(
    gk,
    gv,
    yerr=gs,
    fmt="-o",
    color=EMERALD,
    ecolor=BLACK,
    elinewidth=1,
    capsize=4,
    lw=2,
    ms=7,
    markeredgecolor="white",
)
ax.scatter([gap_k], [gap[gap_k]["gap"]], s=210, facecolor="none", edgecolor=GOLD, lw=2.4, zorder=5)
ax.annotate(
    f"aturan Tibshirani: K = {gap_k}",
    (gap_k, gap[gap_k]["gap"]),
    (gap_k + 1.2, gap[gap_k]["gap"] - 0.045),
    arrowprops=dict(arrowstyle="->", color=BLACK),
    fontsize=10.5,
)
ax.set_xticks(gk)
ax.set_xlabel("Jumlah klaster K")
ax.set_ylabel("Gap(K)")
ax.set_title("Gap Statistic dengan Galat Baku (B = 100 himpunan acuan seragam)")
savefig(fig, "G15_gap_statistic", "Gap statistic terhadap K")

# =====================================================================================
# 8. INISIALISASI DAN KONVERGENSI
# =====================================================================================
print("\n[8] INISIALISASI & KONVERGENSI")
K_FINAL = 3


def lloyd_trace(X, init_centers, max_iter=300):
    """Algoritma Lloyd ditulis ulang dengan NumPy agar inertia tiap iterasi dapat direkam."""
    C = np.array(init_centers, float)
    trace = []
    for it in range(1, max_iter + 1):
        D = cdist(X, C, "sqeuclidean")
        lab = D.argmin(1)  # langkah assignment
        trace.append(float(D[np.arange(len(X)), lab].sum()))
        C_new = np.array(
            [X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(len(C))]
        )  # langkah update
        if np.allclose(C_new, C, atol=1e-12):
            break
        C = C_new
    return lab, C, trace


# acuan "terbaik yang diketahui": solusi berinertia minimum dari pemetaan lanskap pada bagian 7
BEST_INERTIA = inertia[K_FINAL]
ref_labels = labels_k[K_FINAL]

init_cmp, traces = {}, {"k-means++": [], "random": []}
rng0 = np.random.RandomState(RANDOM_STATE)
for init in ("k-means++", "random"):
    iners, iters, aris = [], [], []
    for s in range(100):
        m = KMeans(K_FINAL, n_init=1, init=init, random_state=s).fit(X)
        iners.append(m.inertia_)
        iters.append(m.n_iter_)
        aris.append(adjusted_rand_score(ref_labels, m.labels_))
    iners = np.array(iners)
    init_cmp[init] = {
        "inertia_mean": iners.mean(),
        "inertia_std": iners.std(ddof=1),
        "inertia_min": iners.min(),
        "inertia_max": iners.max(),
        "inertia_cv_pct": iners.std(ddof=1) / iners.mean() * 100,
        "pct_runs_reach_best": float((iners <= BEST_INERTIA * (1 + 1e-6)).mean() * 100),
        "pct_runs_ari_ge_0.95": float((np.array(aris) >= 0.95).mean() * 100),
        "n_distinct_solutions": int(len(np.unique(np.round(iners, 3)))),
        "iter_mean": float(np.mean(iters)),
        "iter_max": int(np.max(iters)),
        "ari_vs_best_mean": float(np.mean(aris)),
        "ari_vs_best_min": float(np.min(aris)),
        "excess_inertia_mean_pct": float((iners.mean() / BEST_INERTIA - 1) * 100),
    }
    init_cmp[init]["_inertias"] = iners
# kurva konvergensi (inertia per iterasi) dengan Lloyd tulis-ulang, 12 seed tiap metode
init_first_inertia = {"k-means++": [], "random": []}
for s in range(12):
    rs = np.random.RandomState(s)
    c_pp, _ = kmeans_plusplus(X, K_FINAL, random_state=s)
    c_rd = X[rs.choice(len(X), K_FINAL, replace=False)]
    for name, c0 in (("k-means++", c_pp), ("random", c_rd)):
        lab_t, C_t, tr = lloyd_trace(X, c0)
        sk_m = KMeans(K_FINAL, init=c0, n_init=1, algorithm="lloyd", tol=0, max_iter=300).fit(X)
        assert np.isclose(tr[-1], sk_m.inertia_, rtol=1e-8), (
            "Lloyd tulis-ulang tidak sama dengan scikit-learn"
        )
        traces[name].append(tr)
        init_first_inertia[name].append(tr[0])
mw = stats.mannwhitneyu(
    init_cmp["k-means++"]["_inertias"], init_cmp["random"]["_inertias"], alternative="two-sided"
)

# pengaruh n_init
NINIT = [1, 2, 5, 10, 20, 50, 100]
ninit_res = {}
for init in ("k-means++", "random"):
    ninit_res[init] = {}
    for ni in NINIT:
        iners, aris = [], []
        for s in range(30):
            m = KMeans(K_FINAL, n_init=ni, init=init, random_state=1000 + s).fit(X)
            iners.append(m.inertia_)
            aris.append(adjusted_rand_score(ref_labels, m.labels_))
        iners = np.array(iners)
        ninit_res[init][ni] = {
            "inertia_mean": iners.mean(),
            "inertia_std": iners.std(ddof=1),
            "inertia_max": iners.max(),
            "pct_reach_best": float((iners <= BEST_INERTIA * (1 + 1e-6)).mean() * 100),
            "ari_mean": float(np.mean(aris)),
            "ari_min": float(np.min(aris)),
        }
# stabilitas konfigurasi final lintas 50 random_state
N_INIT_FINAL = 100
stab_iner, stab_ari = [], []
stab_labels = []
for s in range(50):
    m = KMeans(K_FINAL, n_init=N_INIT_FINAL, init="k-means++", random_state=s).fit(X)
    stab_iner.append(m.inertia_)
    stab_ari.append(adjusted_rand_score(ref_labels, m.labels_))
    stab_labels.append(m.labels_)
pair_ari = [adjusted_rand_score(a, b) for a, b in combinations(stab_labels[:30], 2)]
single_ari = [
    adjusted_rand_score(ref_labels, KMeans(K_FINAL, n_init=1, random_state=s).fit(X).labels_)
    for s in range(50)
]
single_iner = [KMeans(K_FINAL, n_init=1, random_state=s).fit(X).inertia_ for s in range(50)]
SUMMARY["init_convergence"] = r(
    {
        "k": K_FINAL,
        "best_known_inertia": BEST_INERTIA,
        "n_reference_inits": N_LAND,
        "n_init_1_100_seeds": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in init_cmp.items()
        },
        "mannwhitney_inertia_pp_vs_random": {"U": mw.statistic, "p": mw.pvalue},
        "first_iteration_inertia_mean": {k: float(np.mean(v)) for k, v in init_first_inertia.items()},
        "trace_iterations_mean": {k: float(np.mean([len(t) for t in v])) for k, v in traces.items()},
        "trace_iterations_max": {k: int(np.max([len(t) for t in v])) for k, v in traces.items()},
        "n_init_effect": ninit_res,
        "final_config_50_seeds": {
            "n_init": N_INIT_FINAL,
            "inertia_mean": np.mean(stab_iner),
            "inertia_std": np.std(stab_iner, ddof=1),
            "inertia_min": np.min(stab_iner),
            "inertia_max": np.max(stab_iner),
            "inertia_cv_pct": np.std(stab_iner, ddof=1) / np.mean(stab_iner) * 100,
            "ari_vs_best_mean": np.mean(stab_ari),
            "ari_vs_best_min": np.min(stab_ari),
            "pairwise_ari_mean": np.mean(pair_ari),
            "pairwise_ari_min": np.min(pair_ari),
            "pct_identical_to_best": float(np.mean(np.array(stab_ari) > 0.9999) * 100),
        },
        "single_init_50_seeds": {
            "ari_mean": np.mean(single_ari),
            "ari_min": np.min(single_ari),
            "ari_std": np.std(single_ari, ddof=1),
            "inertia_mean": np.mean(single_iner),
            "inertia_std": np.std(single_iner, ddof=1),
            "inertia_cv_pct": np.std(single_iner, ddof=1) / np.mean(single_iner) * 100,
            "pct_identical_to_best": float(np.mean(np.array(single_ari) > 0.9999) * 100),
        },
    },
    4,
)
print(
    f"   inertia terbaik={BEST_INERTIA:.3f} | n_init=1: k-means++ capai terbaik {init_cmp['k-means++']['pct_runs_reach_best']:.0f}% run, "
    f"random {init_cmp['random']['pct_runs_reach_best']:.0f}% | final (n_init={N_INIT_FINAL}) ARI min={np.min(stab_ari):.3f}"
)

# --- G16 Kurva konvergensi ---------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
for name, col, ls in (("random", GOLD, "-"), ("k-means++", EMERALD, "-")):
    for tr in traces[name]:
        a1.plot(range(1, len(tr) + 1), tr, color=col, lw=1.2, alpha=0.75, ls=ls, marker="o", ms=3)
a1.axhline(BEST_INERTIA, color=BLACK, lw=1, ls=":")
a1.text(
    a1.get_xlim()[1], BEST_INERTIA - 11, f"inertia terbaik = {BEST_INERTIA:.1f}", ha="right", fontsize=9
)
a1.set_xlabel("Iterasi Lloyd")
a1.set_ylabel("Inertia (WCSS)")
a1.legend(
    handles=[
        Line2D([], [], color=EMERALD, lw=2, label="init = k-means++"),
        Line2D([], [], color=GOLD, lw=2, label="init = random"),
    ]
)
a1.set_title(f"Inertia per Iterasi (12 seed tiap metode, K = {K_FINAL})")
bp = boxplot_compat(
    a2,
    [init_cmp["k-means++"]["_inertias"], init_cmp["random"]["_inertias"]],
    ["k-means++", "random"],
    patch_artist=True,
    widths=0.5,
    medianprops=dict(color=BLACK, lw=1.6),
    flierprops=dict(marker="o", markersize=4, markerfacecolor=GREY),
)
for p, col in zip(bp["boxes"], (EMERALD, GOLD)):
    p.set_facecolor(col)
    p.set_alpha(0.85)
a2.axhline(BEST_INERTIA, color=BLACK, lw=1, ls=":")
a2.set_ylabel("Inertia akhir (n_init = 1)")
a2.set_title("Sebaran Inertia Akhir, 100 Seed")
fig.tight_layout()
savefig(fig, "G16_konvergensi", "Kurva konvergensi inertia per iterasi dan sebaran inertia akhir")

# --- G17 Stabilitas & n_init ---------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
xx = np.arange(len(NINIT))
for off, init, col in ((-0.2, "k-means++", EMERALD), (0.2, "random", GOLD)):
    v = [ninit_res[init][ni]["pct_reach_best"] for ni in NINIT]
    b = a1.bar(xx + off, v, 0.38, color=col, label=f"init = {init}")
    for bb in b:
        a1.text(
            bb.get_x() + bb.get_width() / 2,
            bb.get_height() + 1.5,
            f"{bb.get_height():.0f}",
            ha="center",
            fontsize=9,
        )
a1.set_xticks(xx)
a1.set_xticklabels(NINIT)
a1.set_xlabel("n_init (jumlah inisialisasi ulang)")
a1.set_ylabel("% run yang mencapai inertia terbaik")
a1.set_ylim(0, 118)
a1.legend(loc="upper left")
a1.set_title("Pengaruh n_init (30 random_state tiap konfigurasi)")
rs_ = np.random.RandomState(1)
a2.scatter(
    rs_.uniform(-0.12, 0.12, 50), single_ari, s=32, color=GOLD, edgecolor=BLACK, lw=0.4, label="n_init = 1"
)
a2.scatter(
    1 + rs_.uniform(-0.12, 0.12, 50),
    stab_ari,
    s=32,
    color=EMERALD,
    edgecolor=BLACK,
    lw=0.4,
    marker="D",
    label=f"n_init = {N_INIT_FINAL}",
)
a2.set_xticks([0, 1])
a2.set_xticklabels(
    [
        f"n_init = 1\n(rata-rata ARI {np.mean(single_ari):.3f})",
        f"n_init = {N_INIT_FINAL}\n(rata-rata ARI {np.mean(stab_ari):.3f})",
    ]
)
a2.set_xlim(-0.5, 1.5)
a2.set_ylim(0, 1.06)
a2.set_ylabel("ARI terhadap solusi terbaik")
a2.set_title("Stabilitas Label lintas 50 random_state")
fig.tight_layout()
savefig(fig, "G17_stabilitas", "Pengaruh n_init dan stabilitas hasil lintas random_state")

# =====================================================================================
# 9. MODEL FINAL DAN PROFIL KLASTER
# =====================================================================================
print("\n[9] MODEL FINAL")
final_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "kmeans",
            KMeans(
                n_clusters=K_FINAL,
                init="k-means++",
                n_init=N_INIT_FINAL,
                max_iter=300,
                tol=1e-4,
                algorithm="lloyd",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
).fit(data[CORE].values)
scaler_f, km_f = final_pipe.named_steps["scaler"], final_pipe.named_steps["kmeans"]
assert np.isclose(km_f.inertia_, BEST_INERTIA, rtol=1e-6), (
    "Model final tidak mencapai inertia terbaik yang diketahui"
)
lab_raw_final = km_f.labels_
cent_orig_raw = scaler_f.inverse_transform(km_f.cluster_centers_)
# centroid dalam skala asli harus sama dengan rata-rata fitur tiap klaster
assert np.allclose(cent_orig_raw, data[CORE].groupby(lab_raw_final).mean().values, atol=1e-8)

# --- Penamaan klaster berbasis aturan yang eksplisit (bukan selera) -----------------------
ci = pd.DataFrame(cent_orig_raw, columns=CORE)
i_bull = int(ci.log_ret.idxmax())  # klaster dengan rata-rata return tertinggi
rest = [i for i in range(K_FINAL) if i != i_bull]
i_act = max(rest, key=lambda i: ci.rel_vol[i])  # dari sisanya: volume relatif tertinggi
i_weak = [i for i in rest if i != i_act][0]  # sisanya: return terendah, volume di bawah rata-rata
ORDER = [i_bull, i_weak, i_act]
REMAP = {old: new for new, old in enumerate(ORDER)}
NAMES = ["Bullish / Reli", "Melemah / Konsolidasi", "Aktif - Volatil"]
SHORT = ["Bullish", "Melemah", "Aktif-Volatil"]
CODE = ["C1", "C2", "C3"]
assert ci.log_ret[i_weak] == ci.log_ret.min(), (
    "Aturan penamaan tidak konsisten: klaster 'Melemah' bukan return terendah"
)
labels = np.array([REMAP[l] for l in lab_raw_final])
centers_std = km_f.cluster_centers_[ORDER]
centers_orig = cent_orig_raw[ORDER]
data["cluster"] = labels
data["cluster_code"] = [CODE[l] for l in labels]
data["cluster_name"] = [NAMES[l] for l in labels]
sil_s = silhouette_samples(X, labels)
data["silhouette"] = sil_s
data["dist_to_centroid"] = np.linalg.norm(X - centers_std[labels], axis=1)
sizes = np.bincount(labels, minlength=K_FINAL)

profile = {}
for k in range(K_FINAL):
    m = labels == k
    sub = data.loc[m]
    profile[CODE[k]] = {
        "name": NAMES[k],
        "n": int(m.sum()),
        "pct": float(m.mean() * 100),
        "centroid_original": {c: centers_orig[k, j] for j, c in enumerate(CORE)},
        "centroid_z": {c: centers_std[k, j] for j, c in enumerate(CORE)},
        "desc": {
            c: {
                "mean": sub[c].mean(),
                "median": sub[c].median(),
                "std": sub[c].std(),
                "min": sub[c].min(),
                "max": sub[c].max(),
            }
            for c in ALL_FEATS
        },
        "silhouette_mean": float(sil_s[m].mean()),
        "pct_negative_silhouette": float((sil_s[m] < 0).mean() * 100),
        "pct_up_days": float((sub.log_ret > 0).mean() * 100),
        "mean_volume_million": float(sub.Volume.mean() / 1e6),
        "wcss": float(((X[m] - centers_std[k]) ** 2).sum()),
        "representative_day": sub.loc[sub.dist_to_centroid.idxmin(), "Date"],
        "first_date": sub.Date.min(),
        "last_date": sub.Date.max(),
    }
# jarak antar centroid (ruang terstandar)
cdm = cdist(centers_std, centers_std)

# --- Uji beda antar klaster: ANOVA satu arah & Kruskal-Wallis + eta^2 ---------------------
tests = {}
for c in ALL_FEATS + ["next_ret"]:
    groups = [data.loc[(labels == k) & data[c].notna(), c].values for k in range(K_FINAL)]
    F, pF = stats.f_oneway(*groups)
    H, pH = stats.kruskal(*groups)
    lev = stats.levene(*groups)
    ok = data[c].notna().values
    tests[c] = {
        "anova_F": F,
        "anova_p": pF,
        "kruskal_H": H,
        "kruskal_p": pH,
        "eta2": eta_squared(data.loc[ok, c], labels[ok]),
        "levene_p": lev.pvalue,
        "used_in_clustering": c in CORE,
    }
SUMMARY["final_model"] = r(
    {
        "pipeline": "StandardScaler -> KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=300, tol=1e-4, algorithm='lloyd', random_state=42)",
        "features": CORE,
        "n": N,
        "k": K_FINAL,
        "inertia": km_f.inertia_,
        "n_iter": int(km_f.n_iter_),
        "tss": inertia[1],
        "bss_over_tss_pct": (1 - km_f.inertia_ / inertia[1]) * 100,
        "silhouette": silhouette_score(X, labels),
        "dbi": davies_bouldin_score(X, labels),
        "ch": calinski_harabasz_score(X, labels),
        "pct_negative_silhouette": float((sil_s < 0).mean() * 100),
        "scaler_mean": {c: scaler_f.mean_[j] for j, c in enumerate(CORE)},
        "scaler_std": {c: scaler_f.scale_[j] for j, c in enumerate(CORE)},
        "names": dict(zip(CODE, NAMES)),
        "short_names": dict(zip(CODE, SHORT)),
        "sizes": dict(zip(CODE, sizes.tolist())),
        "size_ratio_max_min": float(sizes.max() / sizes.min()),
        "profile": profile,
        "centroid_distance_matrix": cdm.tolist(),
        "closest_centroid_pair": [
            CODE[i] for i in np.unravel_index(np.argmin(cdm + np.eye(K_FINAL) * 99), cdm.shape)
        ],
        "farthest_centroid_pair": [CODE[i] for i in np.unravel_index(np.argmax(cdm), cdm.shape)],
        "tests": tests,
    },
    4,
)
print(
    f"   inertia={km_f.inertia_:.3f} silhouette={silhouette_score(X, labels):.4f} DBI={davies_bouldin_score(X, labels):.4f} "
    f"CH={calinski_harabasz_score(X, labels):.3f} | ukuran={dict(zip(SHORT, sizes.tolist()))}"
)


# --- Alternatif K = 4 (analisis sensitivitas; TIDAK dipakai sebagai model final) -----------
def describe_alt(lab_alt, cen_alt_std):
    ct = pd.crosstab(labels, lab_alt).reindex(index=range(K_FINAL), fill_value=0)
    cen_o = scaler_f.inverse_transform(cen_alt_std)
    rows = []
    for j in range(cen_alt_std.shape[0]):
        col = ct[j].values
        rows.append(
            {
                "n": int(col.sum()),
                "parent": CODE[int(col.argmax())],
                "purity_pct": float(col.max() / col.sum() * 100),
                "from_each_k3_cluster": dict(zip(CODE, col.tolist())),
                "centroid_original": dict(zip(CORE, cen_o[j])),
            }
        )
    return sorted(rows, key=lambda d: (d["parent"], -d["n"]))


singles4 = [KMeans(4, n_init=1, random_state=s).fit(X) for s in range(N_LAND)]
best4 = min(singles4, key=lambda m: m.inertia_)
riv4 = min(
    (m for m in singles4 if adjusted_rand_score(best4.labels_, m.labels_) < 0.80), key=lambda m: m.inertia_
)
SUMMARY["k4_alternative"] = r(
    {
        "best": {
            "inertia": best4.inertia_,
            "silhouette": silhouette_score(X, best4.labels_),
            "sizes": sorted(np.bincount(best4.labels_).tolist(), reverse=True),
            "ari_vs_final_k3": adjusted_rand_score(labels, best4.labels_),
            "clusters": describe_alt(best4.labels_, best4.cluster_centers_),
        },
        "rival": {
            "inertia": riv4.inertia_,
            "silhouette": silhouette_score(X, riv4.labels_),
            "sizes": sorted(np.bincount(riv4.labels_).tolist(), reverse=True),
            "ari_vs_final_k3": adjusted_rand_score(labels, riv4.labels_),
            "ari_vs_best_k4": adjusted_rand_score(best4.labels_, riv4.labels_),
            "excess_inertia_pct": (riv4.inertia_ / best4.inertia_ - 1) * 100,
            "clusters": describe_alt(riv4.labels_, riv4.cluster_centers_),
            "n_days_differently_grouped": int(
                (match_labels(best4.labels_, riv4.labels_) != best4.labels_).sum()
            ),
        },
    },
    4,
)

CL_COL = PALETTE[:K_FINAL]  # C1 hijau, C2 merah bata, C3 emas
CL_MRK = MARKERS[:K_FINAL]
leg_handles = [
    Line2D(
        [],
        [],
        marker=CL_MRK[k],
        color="none",
        markerfacecolor=CL_COL[k],
        markeredgecolor="white",
        markersize=9,
        label=f"{CODE[k]} {SHORT[k]} (n={sizes[k]})",
    )
    for k in range(K_FINAL)
]

# --- G18 PCA 2D + centroid ---------------------------------------------------------------
pca = PCA(n_components=len(CORE), random_state=RANDOM_STATE).fit(X)
Z = pca.transform(X)[:, :2]
Zc = pca.transform(centers_std)[:, :2]
evr = pca.explained_variance_ratio_
fig, (ax, axl) = plt.subplots(1, 2, figsize=(12.8, 5.6), gridspec_kw={"width_ratios": [1.75, 1]})
for k in range(K_FINAL):
    m = labels == k
    ax.scatter(
        Z[m, 0], Z[m, 1], s=46, color=CL_COL[k], marker=CL_MRK[k], edgecolor="white", lw=0.6, alpha=0.92
    )
    ax.scatter(Zc[k, 0], Zc[k, 1], s=360, color=CL_COL[k], marker="*", edgecolor=BLACK, lw=1.4, zorder=6)
    off = Zc[k] / (np.linalg.norm(Zc[k]) + 1e-9) * 0.95 + np.array([0, 0.25])
    ax.annotate(
        f"{CODE[k]} {SHORT[k]}",
        Zc[k],
        Zc[k] + off,
        fontsize=10.5,
        fontweight="bold",
        color=BLACK,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=CL_COL[k], lw=1.2),
        zorder=7,
    )
ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% varians)")
ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% varians)")
ax.legend(
    handles=leg_handles
    + [
        Line2D(
            [],
            [],
            marker="*",
            color="none",
            markerfacecolor="white",
            markeredgecolor=BLACK,
            markersize=14,
            label="centroid",
        )
    ],
    loc="lower right",
    fontsize=9.5,
)
ax.set_title(f"Klaster K-Means pada PCA 2D (PC1 + PC2 = {evr[:2].sum() * 100:.1f}% varians)")
load = pca.components_[:2].T
axl.add_patch(plt.Circle((0, 0), 1, fill=False, color="#D1D5DB", lw=1))
for j, c in enumerate(CORE):
    axl.annotate(
        "",
        xy=(load[j, 0], load[j, 1]),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=DEEP, lw=1.6),
    )
    va = "bottom" if load[j, 1] >= 0 else "top"
    axl.text(
        load[j, 0] * 1.1,
        load[j, 1] * 1.1 + (0.05 if c == "log_ret" else (-0.05 if c == "gap_pct" else 0)),
        c,
        fontsize=10,
        ha="center",
        va=va,
        color=BLACK,
    )
axl.axhline(0, color="#9CA3AF", lw=0.8)
axl.axvline(0, color="#9CA3AF", lw=0.8)
axl.set_xlim(-1.15, 1.15)
axl.set_ylim(-1.15, 1.15)
axl.set_aspect("equal")
axl.set_xlabel("Loading pada PC1")
axl.set_ylabel("Loading pada PC2")
axl.set_title("Arah fitur (loading PCA)")
fig.tight_layout()
savefig(fig, "G18_pca_klaster", "Sebaran klaster pada PCA 2D beserta centroid")

# --- G19 t-SNE sebagai pembanding ----------------------------------------------------------
tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=RANDOM_STATE)
T = tsne.fit_transform(X)
tw_pca, tw_tsne = trustworthiness(X, Z, n_neighbors=10), trustworthiness(X, T, n_neighbors=10)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5))
for ax, emb, ttl in (
    (a1, Z, f"PCA 2D (linear) - trustworthiness = {tw_pca:.3f}"),
    (a2, T, f"t-SNE (nonlinear, perplexity 30) - trustworthiness = {tw_tsne:.3f}"),
):
    for k in range(K_FINAL):
        m = labels == k
        ax.scatter(emb[m, 0], emb[m, 1], s=40, color=CL_COL[k], marker=CL_MRK[k], edgecolor="white", lw=0.5)
    ax.set_title(ttl, fontsize=11.5)
    ax.set_xlabel("Dimensi 1")
    ax.set_ylabel("Dimensi 2")
a1.legend(handles=leg_handles, loc="upper left", fontsize=9)
fig.tight_layout()
savefig(fig, "G19_pca_vs_tsne", "Perbandingan visualisasi klaster PCA dan t-SNE")
SUMMARY["projection"] = r(
    {
        "pca_explained_variance_ratio": evr.tolist(),
        "pca_cumulative": np.cumsum(evr).tolist(),
        "pca_loadings_pc1": dict(zip(CORE, pca.components_[0])),
        "pca_loadings_pc2": dict(zip(CORE, pca.components_[1])),
        "n_components_for_80pct": int(np.searchsorted(np.cumsum(evr), 0.80) + 1),
        "silhouette_in_pca2d": silhouette_score(Z, labels),
        "trustworthiness_pca": tw_pca,
        "trustworthiness_tsne": tw_tsne,
        "tsne": {"perplexity": 30, "kl_divergence": float(tsne.kl_divergence_)},
        "silhouette_in_tsne2d": silhouette_score(T, labels),
        "umap_available": False,
    },
    4,
)

# --- G20 Profil centroid: heatmap z-score + radar -----------------------------------------
fig = plt.figure(figsize=(13, 4.9))
a1 = fig.add_subplot(1, 2, 1)
Hm = pd.DataFrame(centers_std, index=[f"{CODE[k]} {SHORT[k]}" for k in range(K_FINAL)], columns=CORE)
ann = np.array(
    [
        [f"{centers_std[k, j]:+.2f}\n({centers_orig[k, j]:.2f})" for j in range(len(CORE))]
        for k in range(K_FINAL)
    ]
)
sns.heatmap(
    Hm,
    annot=ann,
    fmt="",
    cmap=cmap,
    center=0,
    vmin=-2,
    vmax=2,
    linewidths=1.5,
    linecolor="white",
    ax=a1,
    annot_kws={"size": 9.5},
    cbar_kws={"label": "Centroid (z-score)"},
)
a1.grid(False)
a1.set_title("Centroid: z-score dan (nilai skala asli)")
a1.tick_params(axis="x", rotation=30)
a1.tick_params(axis="y", rotation=0)
a2 = fig.add_subplot(1, 2, 2, projection="polar")
ang = np.linspace(0, 2 * np.pi, len(CORE), endpoint=False).tolist()
ang += ang[:1]
for k in range(K_FINAL):
    v = centers_std[k].tolist()
    v += v[:1]
    a2.plot(ang, v, color=CL_COL[k], lw=2, marker=CL_MRK[k], ms=6, label=f"{CODE[k]} {SHORT[k]}")
    a2.fill(ang, v, color=CL_COL[k], alpha=0.08)
a2.plot(np.linspace(0, 2 * np.pi, 100), np.zeros(100), color=BLACK, lw=0.9, ls="--")
a2.set_xticks(ang[:-1])
a2.set_xticklabels(CORE, fontsize=9.5)
a2.set_ylim(-1.9, 2.1)
a2.set_yticks([-1, 0, 1, 2])
a2.set_yticklabels(["-1", "0", "+1", "+2"], fontsize=8)
a2.legend(loc="upper right", bbox_to_anchor=(1.42, 1.12), fontsize=9.5)
a2.set_title("Radar centroid (z-score; garis putus-putus = rata-rata)", pad=18)
fig.tight_layout()
savefig(fig, "G20_profil_centroid", "Profil centroid tiap klaster dalam heatmap dan radar")

# --- G21 Ukuran klaster & sebaran fitur per klaster --------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.1), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
b = axes[0].bar([f"{CODE[k]}\n{SHORT[k]}" for k in range(K_FINAL)], sizes, color=CL_COL, width=0.62)
for bb, s_ in zip(b, sizes):
    axes[0].text(
        bb.get_x() + bb.get_width() / 2,
        bb.get_height() + 0.8,
        f"{s_} hari\n({s_ / N * 100:.1f}%)",
        ha="center",
        fontsize=9.5,
    )
axes[0].set_ylim(0, sizes.max() * 1.25)
axes[0].set_ylabel("Jumlah hari perdagangan")
axes[0].set_title("Ukuran klaster")
axes[0].tick_params(axis="x", labelsize=9.2)
for ax, c in zip(axes[1:], ["log_ret", "range_pct", "rel_vol"]):
    bp = boxplot_compat(
        ax,
        [data.loc[labels == k, c] for k in range(K_FINAL)],
        CODE,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color=BLACK, lw=1.5),
        flierprops=dict(marker="o", markersize=4, markerfacecolor=GREY),
    )
    for p, col in zip(bp["boxes"], CL_COL):
        p.set_facecolor(col)
        p.set_alpha(0.85)
    ax.axhline(data[c].mean(), color=BLACK, ls=":", lw=1)
    ax.set_title(FEAT_LABEL[c], fontsize=11)
fig.tight_layout()
savefig(fig, "G21_ukuran_dan_sebaran", "Ukuran klaster dan sebaran fitur kunci per klaster")

# =====================================================================================
# 10. CONTOH PERHITUNGAN MANUAL (mengikuti alur contoh Khomarudin, 2016)
# =====================================================================================
print("\n[10] PERHITUNGAN MANUAL")
M_N, M_K, M_FEATS = 10, 3, ["log_ret", "range_pct"]
sub = data.loc[: M_N - 1, ["Date"] + M_FEATS].copy()  # 10 hari perdagangan pertama pada data bersih
P = sub[M_FEATS].round(2).values  # dibulatkan 2 desimal agar mudah dihitung tangan
init_idx = sorted(
    np.random.default_rng(RANDOM_STATE).choice(M_N, M_K, replace=False).tolist()
)  # pusat awal = M_K data acak
C0 = P[init_idx].copy()
manual_iters = []
C_cur = C0.copy()
prev_assign = None
for it in range(1, 50):
    # langkah (b): jarak Euclidean D(i,j) tiap data i ke pusat klaster j
    D = np.sqrt(((P[:, None, :] - C_cur[None, :, :]) ** 2).sum(axis=2))
    assign = D.argmin(axis=1)  # langkah (c): masuk klaster terdekat
    sse = float((D[np.arange(M_N), assign] ** 2).sum())  # SSE terhadap pusat yang dipakai pada iterasi ini
    C_new = np.array(
        [P[assign == j].mean(axis=0) if np.any(assign == j) else C_cur[j] for j in range(M_K)]
    )  # langkah (d)
    sse_after = float(((P - C_new[assign]) ** 2).sum())  # SSE setelah pusat diperbarui
    moved = None if prev_assign is None else int((assign != prev_assign).sum())
    manual_iters.append(
        {
            "iteration": it,
            "centroids_used": C_cur.copy(),
            "distances": D.copy(),
            "assignment": assign.copy(),
            "new_centroids": C_new.copy(),
            "sse_before_update": sse,
            "sse_after_update": sse_after,
            "n_reassigned": moved,
            "members": [np.flatnonzero(assign == j).tolist() for j in range(M_K)],
        }
    )
    converged = np.allclose(C_new, C_cur, atol=1e-12)  # langkah (e): berhenti bila pusat tidak berubah
    C_cur, prev_assign = C_new, assign
    if converged:
        break
manual_labels, manual_centers = prev_assign, C_cur
# --- VERIFIKASI: harus IDENTIK dengan scikit-learn untuk pusat awal yang sama --------------
sk = KMeans(n_clusters=M_K, init=C0, n_init=1, algorithm="lloyd", max_iter=300, tol=0).fit(P)
assert np.array_equal(sk.labels_, manual_labels), "Label manual berbeda dengan scikit-learn"
assert np.allclose(sk.cluster_centers_, manual_centers, atol=1e-10), (
    "Centroid manual berbeda dengan scikit-learn"
)
assert np.isclose(sk.inertia_, manual_iters[-1]["sse_after_update"], atol=1e-10), (
    "SSE manual berbeda dengan scikit-learn"
)
max_diff_centers = float(np.abs(sk.cluster_centers_ - manual_centers).max())
SUMMARY["manual_example"] = r(
    {
        "n": M_N,
        "k": M_K,
        "features": M_FEATS,
        "rounding_decimals": 2,
        "dates": [d for d in sub.Date],
        "points": P.tolist(),
        "init_indices_1based": [i + 1 for i in init_idx],
        "init_centroids": C0.tolist(),
        "n_iterations": len(manual_iters),
        "iterations": [
            {
                "iteration": m["iteration"],
                "centroids_used": m["centroids_used"].tolist(),
                "distances": m["distances"].tolist(),
                "assignment_1based": (m["assignment"] + 1).tolist(),
                "new_centroids": m["new_centroids"].tolist(),
                "members_1based": [[i + 1 for i in mm] for mm in m["members"]],
                "sse_before_update": m["sse_before_update"],
                "sse_after_update": m["sse_after_update"],
                "n_reassigned": m["n_reassigned"],
            }
            for m in manual_iters
        ],
        "final_labels_1based": (manual_labels + 1).tolist(),
        "final_centroids": manual_centers.tolist(),
        "final_sse": manual_iters[-1]["sse_after_update"],
        "sklearn": {
            "labels_1based": (sk.labels_ + 1).tolist(),
            "centroids": sk.cluster_centers_.tolist(),
            "inertia": sk.inertia_,
            "n_iter": int(sk.n_iter_),
        },
        "labels_identical": bool(np.array_equal(sk.labels_, manual_labels)),
        "max_abs_diff_centroids": max_diff_centers,
        "abs_diff_sse": abs(sk.inertia_ - manual_iters[-1]["sse_after_update"]),
    },
    6,
)
print(
    f"   {M_N} hari, {len(M_FEATS)} fitur, K={M_K}, pusat awal=data ke-{[i + 1 for i in init_idx]} | konvergen pada iterasi {len(manual_iters)} "
    f"| identik scikit-learn (selisih centroid maks {max_diff_centers:.1e})"
)

# --- G22 Ilustrasi iterasi perhitungan manual --------------------------------------------
n_it = len(manual_iters)
fig, axes = plt.subplots(1, n_it, figsize=(4.3 * n_it, 4.5), sharex=True, sharey=True)
axes = np.atleast_1d(axes)
MC = [EMERALD, GOLD, RUST]
MM = ["o", "s", "^"]
for ax, m in zip(axes, manual_iters):
    for j in range(M_K):
        mm = m["assignment"] == j
        ax.scatter(P[mm, 0], P[mm, 1], s=95, color=MC[j], marker=MM[j], edgecolor="white", lw=0.8, zorder=3)
        for i in np.flatnonzero(mm):
            ax.plot(
                [P[i, 0], m["centroids_used"][j, 0]],
                [P[i, 1], m["centroids_used"][j, 1]],
                color=MC[j],
                lw=0.7,
                alpha=0.5,
                zorder=1,
            )
        ax.scatter(
            *m["centroids_used"][j], s=260, marker="X", color=MC[j], edgecolor=BLACK, lw=1.3, zorder=5
        )
        ax.scatter(
            *m["new_centroids"][j], s=260, marker="*", color="white", edgecolor=MC[j], lw=1.8, zorder=6
        )
        ax.annotate(
            "",
            xy=m["new_centroids"][j],
            xytext=m["centroids_used"][j],
            arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.1),
            zorder=4,
        )
    for i in range(M_N):
        ax.text(P[i, 0] + 0.09, P[i, 1] + 0.1, str(i + 1), fontsize=9)
    ax.set_title(
        f"Iterasi {m['iteration']}: SSE = {m['sse_before_update']:.3f} -> {m['sse_after_update']:.3f}",
        fontsize=11.5,
    )
    ax.set_xlabel("log_ret (%)")
axes[0].set_ylabel("range_pct (%)")
fig.suptitle(
    "Ilustrasi Perhitungan Manual K-Means: Pergeseran Pusat Klaster Tiap Iterasi (angka = nomor data)",
    fontsize=12.5,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
fig.legend(
    handles=[
        Line2D(
            [],
            [],
            marker="X",
            color="none",
            markerfacecolor=GREY,
            markeredgecolor=BLACK,
            markersize=11,
            label="pusat klaster yang dipakai pada iterasi tersebut",
        ),
        Line2D(
            [],
            [],
            marker="*",
            color="none",
            markerfacecolor="white",
            markeredgecolor=GREY,
            markersize=13,
            label="pusat baru (rata-rata anggota)",
        ),
        Line2D([], [], color=BLACK, lw=1.1, label="arah pergeseran pusat"),
    ],
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.07),
    fontsize=9.5,
)
savefig(fig, "G22_iterasi_manual", "Ilustrasi iterasi perhitungan manual K-Means")

# =====================================================================================
# 11. PERBANDINGAN DENGAN ALGORITMA CLUSTERING LAIN (metrik internal yang sama)
# =====================================================================================
print("\n[11] PEMBANDING ALGORITMA")


def pam_kmedoids(X, k, max_iter=100):
    """K-Medoids (PAM: BUILD + SWAP) ditulis manual karena paket scikit-learn-extra tidak tersedia.
    Deterministik; pusat klaster adalah titik data nyata (medoid)."""
    D = cdist(X, X)
    n = len(X)
    med = [int(np.argmin(D.sum(1)))]  # BUILD
    while len(med) < k:
        dmin = D[:, med].min(1)
        gains = [(np.maximum(dmin - D[:, c], 0).sum(), c) for c in range(n) if c not in med]
        med.append(max(gains)[1])
    cost = D[:, med].min(1).sum()
    for _ in range(max_iter):  # SWAP
        best_cost, best_swap = cost, None
        for i in range(k):
            for h in range(n):
                if h in med:
                    continue
                trial = med.copy()
                trial[i] = h
                c_ = D[:, trial].min(1).sum()
                if c_ < best_cost - 1e-12:
                    best_cost, best_swap = c_, (i, h)
        if best_swap is None:
            break
        med[best_swap[0]] = best_swap[1]
        cost = best_cost
    return D[:, med].argmin(1), np.array(med), float(cost)


def internal_metrics(Xm, lab):
    lab = np.asarray(lab)
    mask = lab >= 0
    n_cl = len(np.unique(lab[mask]))
    if n_cl < 2:
        return {
            "n_clusters": n_cl,
            "n_noise": int((~mask).sum()),
            "silhouette": None,
            "dbi": None,
            "ch": None,
            "sizes": sorted(np.bincount(lab[mask]).tolist(), reverse=True) if mask.any() else [],
        }
    return {
        "n_clusters": n_cl,
        "n_noise": int((~mask).sum()),
        "silhouette": silhouette_score(Xm[mask], lab[mask]),
        "dbi": davies_bouldin_score(Xm[mask], lab[mask]),
        "ch": calinski_harabasz_score(Xm[mask], lab[mask]),
        "sizes": sorted(np.bincount(lab[mask]).tolist(), reverse=True),
    }


algo_labels = {"K-Means": labels}
t0 = time.perf_counter()
mb = MiniBatchKMeans(K_FINAL, n_init=20, batch_size=64, random_state=RANDOM_STATE).fit(X)
algo_labels["MiniBatchKMeans"] = mb.labels_
pam_lab, pam_med, pam_cost = pam_kmedoids(X, K_FINAL)
algo_labels["K-Medoids (PAM)"] = pam_lab
ward = AgglomerativeClustering(K_FINAL, linkage="ward").fit(X)
algo_labels["Agglomerative (Ward)"] = ward.labels_
avg = AgglomerativeClustering(K_FINAL, linkage="average").fit(X)
algo_labels["Agglomerative (Average)"] = avg.labels_
gmm = GaussianMixture(K_FINAL, covariance_type="full", n_init=20, random_state=RANDOM_STATE).fit(X)
algo_labels["GMM (full)"] = gmm.predict(X)
gmm_proba = gmm.predict_proba(X)

# DBSCAN: eps dipilih dari grafik k-distance; lalu grid kecil untuk melihat kepekaan
MIN_S = 2 * len(CORE)  # heuristik min_samples = 2 x dimensi
kd = np.sort(NearestNeighbors(n_neighbors=MIN_S).fit(X).kneighbors(X)[0][:, -1])
xs_ = np.linspace(0, 1, len(kd))
yn_ = (kd - kd[0]) / (kd[-1] - kd[0])
eps_knee = float(kd[np.argmax(xs_ - yn_)])
db_main = DBSCAN(eps=eps_knee, min_samples=MIN_S).fit(X)
algo_labels["DBSCAN"] = db_main.labels_
db_grid = []
for ms in (4, 6, 8, 12):
    for eps in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5):
        lab_ = DBSCAN(eps=eps, min_samples=ms).fit(X).labels_
        im = internal_metrics(X, lab_)
        db_grid.append({"eps": eps, "min_samples": ms, **im, "noise_pct": im["n_noise"] / N * 100})
db_multi = [g for g in db_grid if g["n_clusters"] >= 2]

algo_res = {}
for name, lab in algo_labels.items():
    im = internal_metrics(X, lab)
    mask = np.asarray(lab) >= 0
    algo_res[name] = {
        **im,
        "ari_vs_kmeans": adjusted_rand_score(labels, lab),
        "ari_vs_kmeans_non_noise": adjusted_rand_score(labels[mask], np.asarray(lab)[mask])
        if mask.sum() > 1
        else None,
        "noise_pct": im["n_noise"] / N * 100,
    }
algo_res["K-Medoids (PAM)"]["medoid_dates"] = [data.Date[i] for i in pam_med]
algo_res["K-Medoids (PAM)"]["total_cost"] = pam_cost
algo_res["DBSCAN"].update({"eps": eps_knee, "min_samples": MIN_S})
algo_res["GMM (full)"].update(
    {
        "mean_max_posterior": float(gmm_proba.max(1).mean()),
        "pct_uncertain_lt_0.6": float((gmm_proba.max(1) < 0.6).mean() * 100),
        "bic": gmm.bic(X),
        "aic": gmm.aic(X),
    }
)
# BIC GMM untuk berbagai K dan bentuk kovarians
bic = {}
for ct in ("spherical", "diag", "tied", "full"):
    bic[ct] = {
        k: GaussianMixture(k, covariance_type=ct, n_init=5, random_state=RANDOM_STATE).fit(X).bic(X)
        for k in range(1, 9)
    }
bic_best = min(((ct, k, v) for ct, d in bic.items() for k, v in d.items()), key=lambda t: t[2])
# Dendrogram Ward
Zl = linkage(X, method="ward")
coph, _ = cophenet(Zl, pdist(X))
cut_h = float((Zl[-K_FINAL, 2] + Zl[-K_FINAL + 1, 2]) / 2)
merge_gaps = {
    int(k): float(Zl[-k + 1, 2] - Zl[-k, 2]) for k in range(2, 9)
}  # celah tinggi penggabungan utk tiap K
SUMMARY["algorithms"] = r(
    {
        "k": K_FINAL,
        "results": algo_res,
        "dbscan_grid": db_grid,
        "dbscan_n_configs": len(db_grid),
        "dbscan_n_configs_multi_cluster": len(db_multi),
        "dbscan_best_multi": max(db_multi, key=lambda g: g["silhouette"]) if db_multi else None,
        "gmm_bic": bic,
        "gmm_bic_best": {"covariance_type": bic_best[0], "k": bic_best[1], "bic": bic_best[2]},
        "ward": {
            "cophenetic_corr": coph,
            "cut_height": cut_h,
            "merge_height_gaps": merge_gaps,
            "k_largest_gap": max(merge_gaps, key=merge_gaps.get),
        },
        "crosstab_kmeans_vs_gmm": pd.crosstab(
            labels, match_labels(labels, algo_labels["GMM (full)"])
        ).values.tolist(),
        "crosstab_kmeans_vs_ward": pd.crosstab(
            labels, match_labels(labels, algo_labels["Agglomerative (Ward)"])
        ).values.tolist(),
    },
    4,
)
for n_, v in algo_res.items():
    print(
        f"   {n_:26s} klaster={v['n_clusters']} noise={v['n_noise']:3d} sil={v['silhouette'] if v['silhouette'] is None else round(v['silhouette'], 3)} "
        f"ARI vs KMeans={v['ari_vs_kmeans']:.3f} ukuran={v['sizes']}"
    )

# --- G23 Dendrogram ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 4.6))
from scipy.cluster.hierarchy import set_link_color_palette

set_link_color_palette([EMERALD, GOLD, RUST, BLUE, PURPLE])
dendrogram(Zl, no_labels=True, color_threshold=cut_h, above_threshold_color=BLACK, ax=ax)
ax.axhline(cut_h, color=RUST, lw=1.4, ls="--")
ax.text(
    ax.get_xlim()[1] * 0.995,
    cut_h + 0.25,
    f"potong pada tinggi {cut_h:.2f} -> {K_FINAL} klaster",
    ha="right",
    color=RUST,
    fontsize=10,
)
ax.set_ylabel("Jarak penggabungan (Ward)")
ax.set_xlabel(f"{N} hari perdagangan (daun dendrogram)")
ax.set_title(
    f"Dendrogram Agglomerative Hierarchical Clustering, Linkage Ward (korelasi kofenetik = {coph:.3f})"
)
ax.grid(False)
savefig(fig, "G23_dendrogram", "Dendrogram agglomerative hierarchical clustering (Ward)")

# --- G24 Perbandingan metrik antar algoritma ------------------------------------------------
names_a = list(algo_res)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
for ax, key, ttl in zip(
    axes,
    ("silhouette", "dbi", "ch"),
    (
        "Silhouette (lebih besar lebih baik)",
        "Davies-Bouldin (lebih kecil lebih baik)",
        "Calinski-Harabasz (lebih besar lebih baik)",
    ),
):
    vals = [algo_res[n_][key] if algo_res[n_][key] is not None else 0 for n_ in names_a]
    cols = [EMERALD if n_ == "K-Means" else (GREY if n_ == "DBSCAN" else GOLD) for n_ in names_a]
    b = ax.barh(names_a[::-1], vals[::-1], color=cols[::-1], height=0.62)
    for bb, n_ in zip(b, names_a[::-1]):
        txt = f"{bb.get_width():.3f}" if key != "ch" else f"{bb.get_width():.1f}"
        if algo_res[n_][key] is None:
            txt = "tidak terdefinisi (1 klaster)"
        elif n_ == "DBSCAN":
            txt += "*"
        ax.text(
            bb.get_width() + max(vals) * 0.015,
            bb.get_y() + bb.get_height() / 2,
            txt,
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, max(vals) * 1.27)
    ax.set_title(ttl, fontsize=11.5)
    if ax is not axes[0]:
        ax.set_yticklabels([])
fig.suptitle(
    f"Perbandingan Metrik Internal antar Algoritma (K = {K_FINAL}; * DBSCAN dihitung tanpa titik noise)",
    fontsize=12.5,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
savefig(fig, "G24_perbandingan_algoritma", "Perbandingan metrik internal antar algoritma clustering")

# --- G25 Peta klaster tiap algoritma pada PCA ---------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True, sharey=True)
show = [
    "K-Means",
    "K-Medoids (PAM)",
    "Agglomerative (Ward)",
    "GMM (full)",
    "Agglomerative (Average)",
    "DBSCAN",
]
for ax, name in zip(axes.ravel(), show):
    lab = np.asarray(algo_labels[name])
    mask = lab >= 0
    lab_m = lab.copy()
    if mask.sum() and len(np.unique(lab[mask])) > 1:
        lab_m[mask] = match_labels(labels[mask], lab[mask])
    for k in np.unique(lab_m):
        m = lab_m == k
        if k < 0:
            ax.scatter(
                Z[m, 0], Z[m, 1], s=30, color="white", edgecolor=GREY, marker="x", lw=1.1, label="noise"
            )
            ax.scatter(Z[m, 0], Z[m, 1], s=30, color=GREY, marker="x", lw=1.1)
        else:
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                s=34,
                color=PALETTE[k % len(PALETTE)],
                marker=MARKERS[k % len(MARKERS)],
                edgecolor="white",
                lw=0.5,
            )
    v = algo_res[name]
    sil_txt = "-" if v["silhouette"] is None else f"{v['silhouette']:.3f}"
    ax.set_title(
        f"{name}\nsilhouette {sil_txt} | ARI thd K-Means {v['ari_vs_kmeans']:.2f} | noise {v['n_noise']}",
        fontsize=10.8,
    )
for ax in axes[1]:
    ax.set_xlabel("PC1")
for ax in axes[:, 0]:
    ax.set_ylabel("PC2")
fig.suptitle(
    "Hasil Enam Algoritma pada Proyeksi PCA yang Sama (warna disepadankan dengan klaster K-Means)",
    fontsize=12.5,
    fontweight="bold",
    y=1.0,
)
fig.tight_layout()
savefig(fig, "G25_peta_algoritma", "Peta klaster enam algoritma pada proyeksi PCA")

# =====================================================================================
# 12. ANALISIS LANJUTAN
# =====================================================================================
print("\n[12] ANALISIS LANJUTAN")

# --- 12a. Dimensi waktu: harga diwarnai klaster, lama bertahan, matriks transisi -----------
run_starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
run_lens = np.diff(np.r_[run_starts, N])
run_lab = labels[run_starts]
runs = {
    CODE[k]: {
        "n_runs": int((run_lab == k).sum()),
        "mean_len": float(run_lens[run_lab == k].mean()),
        "median_len": float(np.median(run_lens[run_lab == k])),
        "max_len": int(run_lens[run_lab == k].max()),
        "pct_single_day_runs": float((run_lens[run_lab == k] == 1).mean() * 100),
    }
    for k in range(K_FINAL)
}
trans_cnt = pd.crosstab(pd.Series(labels[:-1], name="dari"), pd.Series(labels[1:], name="ke")).reindex(
    index=range(K_FINAL), columns=range(K_FINAL), fill_value=0
)
trans_p = trans_cnt.div(trans_cnt.sum(1), axis=0)
marg = np.bincount(labels[1:], minlength=K_FINAL) / (N - 1)
chi2, chi_p, chi_dof, chi_exp = stats.chi2_contingency(trans_cnt.values)
# uji permutasi untuk jumlah "tetap di rezim yang sama" (lebih aman utk sel kecil)
stay_obs = int((labels[1:] == labels[:-1]).sum())
rngp = np.random.default_rng(RANDOM_STATE)
stay_perm = np.array(
    [(lambda p: int((p[1:] == p[:-1]).sum()))(rngp.permutation(labels)) for _ in range(5000)]
)
perm_p = float((np.sum(stay_perm >= stay_obs) + 1) / (len(stay_perm) + 1))
expected_runlen_iid = {CODE[k]: float(1 / (1 - sizes[k] / N)) for k in range(K_FINAL)}
month_share = pd.crosstab(data.Date.dt.strftime("%Y-%m"), labels, normalize="index") * 100
SUMMARY["temporal"] = r(
    {
        "n_switches": int((labels[1:] != labels[:-1]).sum()),
        "n_runs": int(len(run_lens)),
        "mean_run_length_all": float(run_lens.mean()),
        "runs": runs,
        "expected_run_length_if_independent": expected_runlen_iid,
        "transition_counts": trans_cnt.values.tolist(),
        "transition_prob": trans_p.values.tolist(),
        "marginal_next": marg.tolist(),
        "self_transition_prob": {CODE[k]: float(trans_p.values[k, k]) for k in range(K_FINAL)},
        "chi2_independence": {
            "chi2": chi2,
            "p": chi_p,
            "dof": int(chi_dof),
            "min_expected": float(chi_exp.min()),
        },
        "stay_same_regime": {
            "observed": stay_obs,
            "observed_pct": stay_obs / (N - 1) * 100,
            "perm_mean": float(stay_perm.mean()),
            "perm_mean_pct": float(stay_perm.mean() / (N - 1) * 100),
            "perm_p_one_sided": perm_p,
        },
        "eta2_time_index": eta_squared(np.arange(N), labels),
        "monthly_share_pct": {m: month_share.loc[m].tolist() for m in month_share.index},
    },
    4,
)

fig, (a1, a2) = plt.subplots(
    2, 1, figsize=(11.5, 6.0), sharex=True, gridspec_kw={"height_ratios": [4, 0.7], "hspace": 0.07}
)
a1.plot(data.Date, data.Close, color="#9CA3AF", lw=1.1, zorder=1)
for k in range(K_FINAL):
    m = labels == k
    a1.scatter(
        data.Date[m],
        data.Close[m],
        s=44,
        color=CL_COL[k],
        marker=CL_MRK[k],
        edgecolor="white",
        lw=0.5,
        zorder=3,
    )
a1.legend(handles=leg_handles, loc="upper left", ncol=2)
a1.set_ylabel("Harga penutupan (USD)")
a1.set_title(f"Harga Penutupan AAPL Diwarnai Menurut Rezim Hasil K-Means (K = {K_FINAL})")
dnum = mdates.date2num(data.Date)
for i in range(N):
    a2.bar(
        dnum[i],
        1,
        width=1.0 if i == N - 1 else max(dnum[i + 1] - dnum[i], 1) if i < N - 1 else 1,
        color=CL_COL[labels[i]],
        align="edge",
    )
a2.set_yticks([])
a2.set_ylabel("rezim", rotation=0, labelpad=22, va="center")
a2.grid(False)
a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
savefig(fig, "G26_harga_per_klaster", "Harga AAPL diwarnai menurut klaster dan pita rezim")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
annot = np.array(
    [
        [f"{trans_p.values[i, j] * 100:.0f}%\n(n={trans_cnt.values[i, j]})" for j in range(K_FINAL)]
        for i in range(K_FINAL)
    ]
)
sns.heatmap(
    trans_p * 100,
    annot=annot,
    fmt="",
    cmap=sns.light_palette(EMERALD, as_cmap=True),
    vmin=0,
    vmax=70,
    linewidths=1.5,
    linecolor="white",
    xticklabels=[f"{CODE[k]}\n{SHORT[k]}" for k in range(K_FINAL)],
    yticklabels=[f"{CODE[k]} {SHORT[k]}" for k in range(K_FINAL)],
    cbar_kws={"label": "Peluang transisi (%)"},
    ax=a1,
    annot_kws={"size": 10},
)
a1.grid(False)
a1.set_xlabel("Rezim hari berikutnya (t+1)")
a1.set_ylabel("Rezim hari ini (t)")
a1.set_title("Matriks Transisi Antar-Rezim")
a1.tick_params(axis="y", rotation=0)
xx = np.arange(K_FINAL)
b1 = a2.bar(xx - 0.2, [runs[c]["mean_len"] for c in CODE], 0.38, color=CL_COL, label="teramati")
b2 = a2.bar(
    xx + 0.2,
    [expected_runlen_iid[c] for c in CODE],
    0.38,
    color="#D1D5DB",
    label="harapan bila urutan acak",
)
for bb in list(b1) + list(b2):
    a2.text(
        bb.get_x() + bb.get_width() / 2,
        bb.get_height() + 0.03,
        f"{bb.get_height():.2f}",
        ha="center",
        fontsize=9,
    )
a2.set_xticks(xx)
a2.set_xticklabels([f"{CODE[k]}\n{SHORT[k]}" for k in range(K_FINAL)], fontsize=9)
a2.set_ylabel("Rata-rata lama bertahan (hari)")
a2.set_title("Lama Bertahan Tiap Rezim")
a2.legend(
    handles=[
        Patch(color=GREY, label="teramati (warna = klaster)"),
        Patch(color="#D1D5DB", label="harapan bila urutan acak"),
    ],
    loc="upper right",
    fontsize=9,
)
a2.set_ylim(0, max(max(r_["mean_len"] for r_ in runs.values()), max(expected_runlen_iid.values())) * 1.3)
fig.tight_layout()
savefig(fig, "G27_transisi_rezim", "Matriks transisi antar-rezim dan lama bertahan tiap rezim")

# --- 12b. Validasi temporal: fit pada periode awal, terapkan ke periode akhir --------------
n_tr = int(round(N * 0.70))
Xa, Xb = data[CORE].values[:n_tr], data[CORE].values[n_tr:]
early = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(K_FINAL, n_init=N_INIT_FINAL, random_state=RANDOM_STATE)),
    ]
).fit(Xa)
lab_a = early.named_steps["kmeans"].labels_
lab_b_pred = early.predict(Xb)  # penugasan ke centroid terdekat (bukan prediksi terawasi)
Xa_s, Xb_s = early.named_steps["scaler"].transform(Xa), early.named_steps["scaler"].transform(Xb)
late = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(K_FINAL, n_init=N_INIT_FINAL, random_state=RANDOM_STATE)),
    ]
).fit(Xb)
cen_late_in_early = early.named_steps["scaler"].transform(
    late.named_steps["scaler"].inverse_transform(late.named_steps["kmeans"].cluster_centers_)
)
Dm = cdist(early.named_steps["kmeans"].cluster_centers_, cen_late_in_early)
ri, ci_ = linear_sum_assignment(Dm)
cen_full_in_early = early.named_steps["scaler"].transform(scaler_f.inverse_transform(centers_std))
Dm2 = cdist(early.named_steps["kmeans"].cluster_centers_, cen_full_in_early)
ri2, ci2 = linear_sum_assignment(Dm2)
ks_shift = {
    c: {
        "ks": float(stats.ks_2samp(Xa[:, j], Xb[:, j]).statistic),
        "p": float(stats.ks_2samp(Xa[:, j], Xb[:, j]).pvalue),
        "mean_early": float(Xa[:, j].mean()),
        "mean_late": float(Xb[:, j].mean()),
    }
    for j, c in enumerate(CORE)
}
# walk-forward: 4 titik potong
wf = []
for frac in (0.5, 0.6, 0.7, 0.8):
    nt = int(round(N * frac))
    p_ = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(K_FINAL, n_init=N_INIT_FINAL, random_state=RANDOM_STATE)),
        ]
    ).fit(data[CORE].values[:nt])
    pred = p_.predict(data[CORE].values[nt:])
    wf.append(
        {
            "train_frac": frac,
            "n_train": nt,
            "n_test": N - nt,
            "train_end": data.Date[nt - 1],
            "ari_test_vs_full_model": adjusted_rand_score(labels[nt:], pred),
            "ari_train_vs_full_model": adjusted_rand_score(labels[:nt], p_.named_steps["kmeans"].labels_),
            "agreement_pct_test": float(
                (
                    match_labels(labels, np.r_[p_.named_steps["kmeans"].labels_, pred])[nt:] == labels[nt:]
                ).mean()
                * 100
            ),
            "silhouette_train": silhouette_score(
                p_.named_steps["scaler"].transform(data[CORE].values[:nt]), p_.named_steps["kmeans"].labels_
            ),
            "silhouette_test": silhouette_score(
                p_.named_steps["scaler"].transform(data[CORE].values[nt:]), pred
            )
            if len(np.unique(pred)) > 1
            else None,
            "n_clusters_used_in_test": int(len(np.unique(pred))),
        }
    )
SUMMARY["temporal_validation"] = r(
    {
        "split": {
            "n_train": n_tr,
            "n_test": N - n_tr,
            "train_start": data.Date[0],
            "train_end": data.Date[n_tr - 1],
            "test_start": data.Date[n_tr],
            "test_end": data.Date[N - 1],
        },
        "silhouette_train": silhouette_score(Xa_s, lab_a),
        "silhouette_test_assigned": silhouette_score(Xb_s, lab_b_pred),
        "dbi_train": davies_bouldin_score(Xa_s, lab_a),
        "dbi_test_assigned": davies_bouldin_score(Xb_s, lab_b_pred),
        "ari_test_early_model_vs_full_model": adjusted_rand_score(labels[n_tr:], lab_b_pred),
        "ari_train_early_model_vs_full_model": adjusted_rand_score(labels[:n_tr], lab_a),
        "agreement_pct_test": float(
            (match_labels(labels, np.r_[lab_a, lab_b_pred])[n_tr:] == labels[n_tr:]).mean() * 100
        ),
        "ari_test_early_model_vs_late_model": adjusted_rand_score(
            late.named_steps["kmeans"].labels_, lab_b_pred
        ),
        "centroid_shift_early_vs_late": {
            "mean": float(Dm[ri, ci_].mean()),
            "max": float(Dm[ri, ci_].max()),
        },
        "centroid_shift_early_vs_full": {
            "mean": float(Dm2[ri2, ci2].mean()),
            "max": float(Dm2[ri2, ci2].max()),
        },
        "share_train_pct": (
            np.bincount(match_labels(labels, np.r_[lab_a, lab_b_pred])[:n_tr], minlength=K_FINAL)[:K_FINAL]
            / n_tr
            * 100
        ).tolist(),
        "share_test_pct": (
            np.bincount(match_labels(labels, np.r_[lab_a, lab_b_pred])[n_tr:], minlength=K_FINAL)[:K_FINAL]
            / (N - n_tr)
            * 100
        ).tolist(),
        "feature_shift_ks": ks_shift,
        "walk_forward": wf,
    },
    4,
)
print(
    f"   validasi temporal: sil latih={silhouette_score(Xa_s, lab_a):.3f}, sil uji={silhouette_score(Xb_s, lab_b_pred):.3f}, "
    f"ARI (model awal vs model penuh, periode akhir)={adjusted_rand_score(labels[n_tr:], lab_b_pred):.3f}"
)

# --- 12c. Hubungan klaster dengan return hari berikutnya (DESKRIPTIF) ----------------------
nx = data.dropna(subset=["next_ret"])
nx_lab = nx.cluster.values
rngb = np.random.default_rng(RANDOM_STATE)
next_ret = {}
for k in range(K_FINAL):
    v = nx.loc[nx_lab == k, "next_ret"].values
    boot = np.array([rngb.choice(v, len(v), replace=True).mean() for _ in range(5000)])
    next_ret[CODE[k]] = {
        "n": int(len(v)),
        "mean": v.mean(),
        "median": np.median(v),
        "std": v.std(ddof=1),
        "pct_positive": float((v > 0).mean() * 100),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "same_day_mean": float(nx.loc[nx_lab == k, "log_ret"].mean()),
    }
SUMMARY["next_day_return"] = r(
    {
        "per_cluster": next_ret,
        "overall_mean": nx.next_ret.mean(),
        "overall_pct_positive": float((nx.next_ret > 0).mean() * 100),
        "kruskal_H": tests["next_ret"]["kruskal_H"],
        "kruskal_p": tests["next_ret"]["kruskal_p"],
        "anova_F": tests["next_ret"]["anova_F"],
        "anova_p": tests["next_ret"]["anova_p"],
        "eta2": tests["next_ret"]["eta2"],
    },
    4,
)
fig, ax = plt.subplots(figsize=(8.8, 4.3))
bp = boxplot_compat(
    ax,
    [nx.loc[nx_lab == k, "next_ret"] for k in range(K_FINAL)],
    [f"{CODE[k]}\n{SHORT[k]}\n(n={next_ret[CODE[k]]['n']})" for k in range(K_FINAL)],
    patch_artist=True,
    widths=0.55,
    medianprops=dict(color=BLACK, lw=1.6),
    flierprops=dict(marker="o", markersize=4, markerfacecolor=GREY),
)
for p, col in zip(bp["boxes"], CL_COL):
    p.set_facecolor(col)
    p.set_alpha(0.8)
for k in range(K_FINAL):
    v = next_ret[CODE[k]]
    ax.errorbar(
        k + 1.36,
        v["mean"],
        yerr=[[v["mean"] - v["ci95_low"]], [v["ci95_high"] - v["mean"]]],
        fmt="D",
        color=BLACK,
        ms=5,
        capsize=4,
        lw=1.2,
    )
ax.axhline(0, color=BLACK, lw=0.9)
ax.axhline(nx.next_ret.mean(), color=GOLD, lw=1.4, ls="--")
ax.legend(
    handles=[
        Line2D(
            [], [], color=GOLD, lw=1.4, ls="--", label=f"rata-rata seluruh hari = {nx.next_ret.mean():.2f}%"
        ),
        Line2D(
            [],
            [],
            marker="D",
            color=BLACK,
            lw=1.2,
            markersize=5,
            label="rata-rata klaster ± IK 95% (bootstrap)",
        ),
    ],
    loc="upper left",
    fontsize=9,
)
ax.set_xlim(0.5, K_FINAL + 0.65)
ax.set_ylabel("Log return hari berikutnya (%)")
ax.set_title(
    f"Return Hari Berikutnya menurut Rezim Hari Ini (Kruskal-Wallis p = {tests['next_ret']['kruskal_p']:.3f})"
)
savefig(fig, "G28_return_berikutnya", "Sebaran return hari berikutnya menurut rezim (deskriptif)")

# --- 12d. Bukti kelemahan (1): sensitivitas terhadap outlier ---------------------------------
out_sens = []
for n_out in (0, 1, 2, 3, 5):
    extra = np.zeros((n_out, X.shape[1]))
    if n_out:
        # hari ekstrem BUATAN: volume relatif 8 simpangan baku di atas rata-rata
        extra[:, CORE.index("rel_vol")] = 8.0 + np.arange(n_out) * 0.5
        extra[:, CORE.index("range_pct")] = 6.0
    Xo = np.vstack([X, extra])
    km_o = KMeans(K_FINAL, n_init=50, random_state=RANDOM_STATE).fit(Xo)
    pm_lab, pm_med, _ = pam_kmedoids(Xo, K_FINAL)
    mean_c, max_c = centroid_shift(centers_std, km_o.cluster_centers_)
    # pembanding K-Medoids: medoid & label dasar (tanpa outlier) berasal dari bagian 11 (pam_med, pam_lab)
    mean_m, max_m = centroid_shift(X[pam_med], Xo[pm_med]) if n_out else (0.0, 0.0)
    out_sens.append(
        {
            "n_injected": n_out,
            "kmeans_ari_original_points": adjusted_rand_score(labels, km_o.labels_[:N]),
            "kmeans_centroid_shift_mean": mean_c,
            "kmeans_centroid_shift_max": max_c,
            "kmeans_smallest_cluster": int(np.bincount(km_o.labels_).min()),
            "kmeans_outliers_form_own_cluster": bool(
                n_out > 0
                and len(set(km_o.labels_[N:])) == 1
                and np.sum(km_o.labels_[:N] == km_o.labels_[N]) <= 2
            ),
            "kmeans_n_days_changed": int((match_labels(labels, km_o.labels_[:N]) != labels).sum()),
            "kmedoids_ari_original_points": adjusted_rand_score(pam_lab, pm_lab[:N]),
            "kmedoids_medoid_shift_mean": mean_m,
            "kmedoids_medoid_shift_max": max_m,
            "kmedoids_n_days_changed": int((match_labels(pam_lab, pm_lab[:N]) != pam_lab).sum()),
            "kmedoids_smallest_cluster": int(np.bincount(pm_lab).min()),
        }
    )

# --- 12e. Bukti kelemahan (2): asumsi klaster sferis & berukuran serupa ----------------------
aniso = {}
for k in range(K_FINAL):
    ev = np.linalg.eigvalsh(np.cov(X[labels == k].T))
    aniso[CODE[k]] = {
        "eig_max_over_min": float(ev.max() / ev.min()),
        "trace": float(ev.sum()),
        "radius_rms": float(np.sqrt(((X[labels == k] - centers_std[k]) ** 2).sum(1).mean())),
    }
# Ilustrasi SINTETIS (bukan data AAPL): klaster memanjang dan klaster beda ukuran/varians
Xs1, ys1 = make_blobs(n_samples=600, centers=3, cluster_std=1.0, random_state=170)
Xs1 = Xs1 @ np.array([[0.6, -0.6], [-0.4, 0.8]])
Xs2, ys2 = make_blobs(
    n_samples=[500, 80, 40],
    centers=[[0, 0], [5.5, 0.5], [1.5, 5.5]],
    cluster_std=[1.7, 0.45, 0.4],
    random_state=RANDOM_STATE,
)
synth = {}
synth_plot = []
for nm, (Xs, ys) in {"anisotropic": (Xs1, ys1), "unequal_size_variance": (Xs2, ys2)}.items():
    Xs_ = StandardScaler().fit_transform(Xs)
    lk = KMeans(3, n_init=20, random_state=RANDOM_STATE).fit_predict(Xs_)
    lg = GaussianMixture(3, covariance_type="full", n_init=10, random_state=RANDOM_STATE).fit_predict(Xs_)
    synth[nm] = {
        "n": len(Xs_),
        "kmeans_ari_truth": adjusted_rand_score(ys, lk),
        "gmm_ari_truth": adjusted_rand_score(ys, lg),
        "kmeans_silhouette": silhouette_score(Xs_, lk),
        "truth_silhouette": silhouette_score(Xs_, ys),
        "true_sizes": np.bincount(ys).tolist(),
        "kmeans_sizes": sorted(np.bincount(lk).tolist(), reverse=True),
    }
    synth_plot.append((nm, Xs_, ys, match_labels(ys, lk), match_labels(ys, lg)))
fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4))
ttl_row = {
    "anisotropic": "Klaster memanjang (anisotropik)",
    "unequal_size_variance": "Klaster beda ukuran & varians",
}
for row, (nm, Xs_, ys, lk, lg) in enumerate(synth_plot):
    for col, (lab_, ttl) in enumerate(
        (
            (ys, "Kelompok sebenarnya"),
            (lk, f"K-Means (ARI = {synth[nm]['kmeans_ari_truth']:.2f})"),
            (lg, f"GMM full (ARI = {synth[nm]['gmm_ari_truth']:.2f})"),
        )
    ):
        ax = axes[row, col]
        for k in range(3):
            m = lab_ == k
            ax.scatter(
                Xs_[m, 0],
                Xs_[m, 1],
                s=11,
                color=[EMERALD, GOLD, RUST][k],
                marker=["o", "s", "^"][k],
                alpha=0.8,
                lw=0,
            )
        ax.set_title(f"{ttl_row[nm]}\n{ttl}" if col == 0 else ttl, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle(
    "Ilustrasi Data SINTETIS (bukan data AAPL): Kegagalan K-Means Saat Asumsi Klaster Sferis dan Seukuran Dilanggar",
    fontsize=12.5,
    fontweight="bold",
    y=1.0,
)
fig.tight_layout()
savefig(
    fig, "G29_asumsi_sferis", "Ilustrasi sintetis pelanggaran asumsi klaster sferis dan berukuran serupa"
)

# --- 12f. Bukti kelemahan (3): dampak jumlah fitur (dimensi) ---------------------------------
FEAT_ORDER = [
    "log_ret",
    "range_pct",
    "rel_vol",
    "dist_ma20",
    "volat_10",
    "gap_pct",
    "vol_chg",
    "mom_5",
    "rsi_14",
    "clv",
]
dim_real = []
for d_ in range(2, len(FEAT_ORDER) + 1):
    cols = FEAT_ORDER[:d_]
    Xd = StandardScaler().fit_transform(data[cols])
    kmd = KMeans(K_FINAL, n_init=20, random_state=RANDOM_STATE).fit(Xd)
    dd = pdist(Xd)
    dim_real.append(
        {
            "n_features": d_,
            "added": cols[-1],
            "silhouette": silhouette_score(Xd, kmd.labels_),
            "dbi": davies_bouldin_score(Xd, kmd.labels_),
            "ch": calinski_harabasz_score(Xd, kmd.labels_),
            "distance_contrast": float((dd.max() - dd.min()) / dd.min()),
            "cv_distance": float(dd.std() / dd.mean()),
            "ari_vs_final": adjusted_rand_score(labels, kmd.labels_),
        }
    )
dim_noise = []
for n_noise in (0, 2, 4, 8, 16, 32):
    sils, aris, cvs = [], [], []
    for rep in range(10):
        rn = np.random.default_rng(100 + rep)
        Xn = np.hstack([X, rn.standard_normal((N, n_noise))]) if n_noise else X
        kmn = KMeans(K_FINAL, n_init=20, random_state=RANDOM_STATE).fit(Xn)
        dd = pdist(Xn)
        sils.append(silhouette_score(Xn, kmn.labels_))
        aris.append(adjusted_rand_score(labels, kmn.labels_))
        cvs.append(dd.std() / dd.mean())
        if n_noise == 0:
            break
    dim_noise.append(
        {
            "n_noise": n_noise,
            "n_dims": len(CORE) + n_noise,
            "silhouette_mean": float(np.mean(sils)),
            "silhouette_std": float(np.std(sils)),
            "ari_vs_final_mean": float(np.mean(aris)),
            "cv_distance_mean": float(np.mean(cvs)),
        }
    )
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
dx = [d_["n_features"] for d_ in dim_real]
a1.plot(dx, [d_["silhouette"] for d_ in dim_real], "-o", color=EMERALD, lw=2, ms=7, markeredgecolor="white")
for d_ in dim_real:
    a1.text(
        d_["n_features"], d_["silhouette"] + 0.012, f"{d_['silhouette']:.3f}", ha="center", fontsize=8.8
    )
    a1.text(
        d_["n_features"],
        min(x_["silhouette"] for x_ in dim_real) - 0.045,
        "+" + d_["added"] if d_["n_features"] > 2 else "log_ret\n+range_pct",
        ha="center",
        fontsize=7.6,
        rotation=35,
        color=GREY,
    )
a1.axvline(6, color=GOLD, ls="--", lw=1.2)
a1.text(6.08, max(d_["silhouette"] for d_ in dim_real), "6 fitur inti", fontsize=9, color=GOLD, va="top")
a1.set_ylim(
    min(d_["silhouette"] for d_ in dim_real) - 0.075, max(d_["silhouette"] for d_ in dim_real) + 0.04
)
a1.set_xlabel("Jumlah fitur nyata yang dipakai")
a1.set_ylabel(f"Silhouette (K = {K_FINAL})")
a1.set_title("Fitur nyata ditambahkan satu per satu")
nxs = [d_["n_dims"] for d_ in dim_noise]
a2.errorbar(
    nxs,
    [d_["silhouette_mean"] for d_ in dim_noise],
    yerr=[d_["silhouette_std"] for d_ in dim_noise],
    fmt="-o",
    color=RUST,
    lw=2,
    ms=7,
    capsize=4,
    markeredgecolor="white",
    label="silhouette",
)
a2.plot(
    nxs,
    [d_["ari_vs_final_mean"] for d_ in dim_noise],
    "-s",
    color=EMERALD,
    lw=2,
    ms=6,
    markeredgecolor="white",
    label="ARI terhadap klaster final",
)
for d_ in dim_noise:
    a2.text(
        d_["n_dims"],
        d_["silhouette_mean"] + 0.035,
        f"{d_['silhouette_mean']:.3f}",
        ha="center",
        fontsize=8.8,
        color=RUST,
    )
    a2.text(
        d_["n_dims"],
        d_["ari_vs_final_mean"] + 0.035,
        f"{d_['ari_vs_final_mean']:.2f}",
        ha="center",
        fontsize=8.8,
        color=DEEP,
    )
a2.set_xscale("log", base=2)
a2.set_xticks(nxs)
a2.set_xticklabels(nxs)
a2.set_ylim(-0.02, 1.1)
a2.set_xlabel("Jumlah dimensi (6 fitur inti + fitur derau acak)")
a2.set_ylabel("Skor (skala sama, 0-1)")
a2.legend(loc="upper right")
a2.set_title("Fitur derau N(0,1) ditambahkan (rata-rata 10 ulangan)")
fig.tight_layout()
savefig(fig, "G30_skor_vs_dimensi", "Skor silhouette terhadap jumlah fitur dan dimensi derau")

SUMMARY["weakness"] = r(
    {
        "outlier_injection": out_sens,
        "cluster_anisotropy_real": aniso,
        "synthetic_shape_demo": synth,
        "ari_kmeans_vs_gmm_real": algo_res["GMM (full)"]["ari_vs_kmeans"],
        "dimension_real_features": dim_real,
        "dimension_noise_features": dim_noise,
        "feature_order": FEAT_ORDER,
    },
    4,
)

# --- 12g. Waktu komputasi vs ukuran data (data hasil RESAMPLING - hanya utk uji skalabilitas) -
print("   uji skalabilitas (data resampling) ...")
SIZES = [1_000, 5_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
BATCH, REPEATS = 4096, 5
rngs = np.random.default_rng(RANDOM_STATE)
scal_rows = []
for n_s in SIZES:
    Xbig = X[rngs.integers(0, N, n_s)] + rngs.normal(0, 0.05, (n_s, X.shape[1]))  # bootstrap + derau kecil
    tk, tm, ik, im_, itk, itm = [], [], [], [], [], []
    for rep in range(REPEATS):
        t0 = time.perf_counter()
        m1 = KMeans(K_FINAL, n_init=1, random_state=rep).fit(Xbig)
        tk.append(time.perf_counter() - t0)
        ik.append(m1.inertia_)
        itk.append(m1.n_iter_)
        t0 = time.perf_counter()
        m2 = MiniBatchKMeans(K_FINAL, n_init=1, batch_size=BATCH, random_state=rep).fit(Xbig)
        tm.append(time.perf_counter() - t0)
        im_.append(m2.inertia_)
        itm.append(m2.n_steps_)
    scal_rows.append(
        {
            "n": n_s,
            "kmeans_sec": float(np.median(tk)),
            "minibatch_sec": float(np.median(tm)),
            "ratio_minibatch_over_kmeans": float(np.median(tm) / np.median(tk)),
            "inertia_excess_minibatch_pct": float((np.median(im_) / np.median(ik) - 1) * 100),
            "kmeans_iter_median": float(np.median(itk)),
            "minibatch_steps_median": float(np.median(itm)),
            "kmeans_sec_per_iter": float(np.median(np.array(tk) / np.array(itk))),
        }
    )
ll = np.polyfit(
    np.log([r_["n"] for r_ in scal_rows[3:]]),
    np.log([r_["kmeans_sec_per_iter"] for r_ in scal_rows[3:]]),
    1,
)
ll2 = np.polyfit(
    np.log([r_["n"] for r_ in scal_rows[3:]]), np.log([r_["kmeans_sec"] for r_ in scal_rows[3:]]), 1
)
TIMING["scalability"] = r(
    {
        "rows": scal_rows,
        "loglog_slope_kmeans_sec_per_iter_n_ge_50k": float(ll[0]),
        "loglog_slope_kmeans_total_n_ge_50k": float(ll2[0]),
        "n_sizes_minibatch_faster": int(sum(r_["ratio_minibatch_over_kmeans"] < 1 for r_ in scal_rows)),
        "n_sizes": len(SIZES),
        "note": "data resampling bootstrap + derau N(0, 0.05); hanya untuk uji skalabilitas",
        "n_init": 1,
        "batch_size": BATCH,
        "repeats": REPEATS,
        "n_cpu": os.cpu_count(),
    },
    5,
)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
a1.plot(
    SIZES,
    [r_["kmeans_sec"] for r_ in scal_rows],
    "-o",
    color=EMERALD,
    lw=2,
    ms=7,
    markeredgecolor="white",
    label="KMeans (Lloyd)",
)
a1.plot(
    SIZES,
    [r_["minibatch_sec"] for r_ in scal_rows],
    "-s",
    color=GOLD,
    lw=2,
    ms=7,
    markeredgecolor="white",
    label=f"MiniBatchKMeans (batch {BATCH:,})".replace(",", "."),
)
a1.set_xscale("log")
a1.set_yscale("log")
a1.set_xlabel("Jumlah baris (skala log)")
a1.set_ylabel("Waktu fit (detik, skala log)")
a1.legend(loc="upper left")
a1.set_title(f"Waktu Fit terhadap Ukuran Data (median {REPEATS} ulangan)")
lbl = [f"{s // 1000}k" if s < 1_000_000 else "1 jt" for s in SIZES]
b = a2.bar(
    lbl,
    [r_["ratio_minibatch_over_kmeans"] for r_ in scal_rows],
    color=[EMERALD if r_["ratio_minibatch_over_kmeans"] < 1 else GOLD for r_ in scal_rows],
    width=0.6,
)
for bb, r_ in zip(b, scal_rows):
    a2.text(
        bb.get_x() + bb.get_width() / 2,
        bb.get_height() + 0.08,
        f"{r_['ratio_minibatch_over_kmeans']:.1f}x\n({r_['inertia_excess_minibatch_pct']:+.1f}%)",
        ha="center",
        fontsize=8.8,
    )
a2.axhline(1, color=BLACK, lw=1, ls=":")
a2.set_ylim(0, max(r_["ratio_minibatch_over_kmeans"] for r_ in scal_rows) * 1.3)
a2.set_xlabel("Jumlah baris")
a2.set_ylabel("Waktu MiniBatch / waktu KMeans")
a2.set_title("Rasio Waktu (di atas 1 = MiniBatch lebih lambat; kurung = selisih inertia)", fontsize=11.5)
fig.suptitle(
    "Uji Skalabilitas pada Data Hasil Resampling (bukan data AAPL asli)",
    fontsize=12.5,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()
savefig(fig, "G31_waktu_komputasi", "Waktu komputasi KMeans dan MiniBatchKMeans terhadap ukuran data")

# =====================================================================================
# 13. SIMPAN KELUARAN
# =====================================================================================
print("\n[13] SIMPAN KELUARAN")
data["pc1"], data["pc2"] = Z[:, 0], Z[:, 1]
export_cols = (
    ["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume", "Day_of_Week", "Month"]
    + ALL_FEATS
    + ["next_ret", "cluster_code", "cluster_name", "silhouette", "dist_to_centroid", "pc1", "pc2"]
)
out = data[export_cols].copy()
out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
num_cols = out.select_dtypes("number").columns.difference(["Volume"])
out[num_cols] = out[num_cols].round(4)
out.to_csv(os.path.join(OUT_DIR, "aapl_kmeans_clustered.csv"), index=False)

SUMMARY["samples"] = r(
    {
        "raw_head": [
            {
                "Date": d.strftime("%Y-%m-%d"),
                "Open": o,
                "High": h,
                "Low": l,
                "Close": c,
                "Adj_Close": a,
                "Volume": int(v),
                "Daily_Return_Pct": dr,
                "MA_7": m7,
                "Volatility_10D_Pct": vv,
                "Day_of_Week": dw,
            }
            for d, o, h, l, c, a, v, dr, m7, vv, dw in raw[
                [
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj_Close",
                    "Volume",
                    "Daily_Return_Pct",
                    "MA_7",
                    "Volatility_10D_Pct",
                    "Day_of_Week",
                ]
            ]
            .head(10)
            .itertuples(index=False)
        ],
        "clustered_head": json.loads(
            out[["Date", "Close", "Volume"] + CORE + ["cluster_code", "cluster_name", "silhouette"]]
            .head(12)
            .to_json(orient="records")
        ),
        "representative_days": json.loads(
            out.loc[
                [int(data.loc[data.cluster == k, "dist_to_centroid"].idxmin()) for k in range(K_FINAL)],
                ["Date", "Close", "Volume"] + CORE + ["cluster_code", "cluster_name"],
            ].to_json(orient="records")
        ),
    },
    4,
)
SUMMARY["meta"] = {
    "random_state": RANDOM_STATE,
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": sns.__version__,
    "n_figures": len(SUMMARY["figures"]),
    "outputs": ["aapl_kmeans_clustered.csv", "model_summary.json", "charts/*.png"],
    "note": "Kunci 'timing' berisi waktu komputasi yang wajar berbeda antar-run; seluruh angka lain deterministik (random_state terkunci).",
}
SUMMARY["timing"] = TIMING
with open(os.path.join(OUT_DIR, "model_summary.json"), "w", encoding="utf-8") as fh:
    json.dump(SUMMARY, fh, indent=2, ensure_ascii=False, default=str)
print(
    f"   {len(SUMMARY['figures'])} grafik | aapl_kmeans_clustered.csv ({len(out)} baris) | model_summary.json"
)
print("\nSELESAI - seluruh assert lolos.")
