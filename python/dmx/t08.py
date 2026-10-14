"""Studi Kasus 08 — K-Means: rezim pasar harian AAPL 2026 (dataset semi-sintetis).

Diringkas dari python/original/kmeans_aapl_stock.py: 10 fitur stasioner tanpa look-ahead
(enam fitur inti dipakai untuk klaster), buang 20 baris pemanasan → 116 hari, Pipeline
StandardScaler → KMeans(K=3, k-means++, n_init=100, random_state=42). Penamaan klaster
memakai aturan eksplisit laporan: return rata-rata tertinggi = Bullish/Reli; dari sisanya
volume relatif tertinggi = Aktif–Volatil; sisanya = Melemah/Konsolidasi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from . import RANDOM_STATE
from .common import cached, get

ALL_FEATS = ["log_ret", "range_pct", "gap_pct", "clv", "vol_chg", "rel_vol", "volat_10", "dist_ma20", "rsi_14", "mom_5"]
CORE = ["log_ret", "range_pct", "gap_pct", "rel_vol", "volat_10", "dist_ma20"]
FEAT_LABEL = {"log_ret": "Log return harian (%)", "range_pct": "Rentang intraday (%)", "gap_pct": "Gap pembukaan (%)",
              "clv": "Posisi close dalam rentang", "vol_chg": "Perubahan volume (log)",
              "rel_vol": "Volume relatif (× rata-rata 20 hari)", "volat_10": "Volatilitas 10 hari (%)",
              "dist_ma20": "Jarak harga ke MA20 (%)", "rsi_14": "RSI 14 hari", "mom_5": "Momentum 5 hari (%)"}
NAMES = ["Bullish / Reli", "Melemah / Konsolidasi", "Aktif – Volatil"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    C, O, H, L, V = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    f["log_ret"] = np.log(C / C.shift(1)) * 100
    f["range_pct"] = (H - L) / C * 100
    f["gap_pct"] = (O - C.shift(1)) / C.shift(1) * 100
    f["clv"] = (C - L) / (H - L)
    f["vol_chg"] = np.log(V / V.shift(1))
    f["rel_vol"] = V / V.shift(1).rolling(20).mean()
    f["volat_10"] = f["log_ret"].shift(1).rolling(10).std()
    f["dist_ma20"] = (C / C.shift(1).rolling(20).mean() - 1) * 100
    delta = C.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss)
    f["mom_5"] = np.log(C / C.shift(5)) * 100
    return f


def prepare(data_key: str):
    def build():
        raw = get(data_key).copy()
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw = raw.sort_values("Date").reset_index(drop=True)
        feats = build_features(raw)
        full = pd.concat([raw, feats], axis=1)
        full["next_ret"] = full["log_ret"].shift(-1)
        warm = feats[ALL_FEATS].isna().any(axis=1)
        data = full.loc[~warm].reset_index(drop=True)
        return {"raw": raw, "data": data, "n_warm": int(warm.sum())}
    return cached(f"{data_key}:t08", build)


def _X(data_key: str, scaler: str = "standard", features: list[str] | None = None):
    p = prepare(data_key)
    X = p["data"][features or CORE].values.astype(float)
    sc = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler(), "none": None}[scaler]
    return (sc.fit_transform(X) if sc is not None else X), p


def overview(data_key: str) -> dict:
    p = prepare(data_key)
    raw, d = p["raw"], p["data"]
    return {"n_raw": len(raw), "n_cols": raw.shape[1], "n_clean": len(d), "n_warmup": p["n_warm"],
            "date_min": str(raw["Date"].min().date()), "date_max": str(raw["Date"].max().date()),
            "clean_date_min": str(d["Date"].min().date()), "missing": raw.isna().sum()[lambda s: s > 0].to_dict(),
            "close": {"date": raw["Date"].dt.strftime("%Y-%m-%d").values, "close": raw["Close"].values,
                      "volume": raw["Volume"].values},
            "describe": d[ALL_FEATS].describe().T[["mean", "std", "min", "50%", "max"]].to_dict(orient="index"),
            "corr": {"cols": ALL_FEATS, "values": d[ALL_FEATS].corr().values}, "core": CORE, "labels": FEAT_LABEL,
            "raw_variance": d[CORE].var().to_dict()}


def k_metrics(data_key: str, init: str = "k-means++", n_init: int = 50, gap_B: int = 20, scaler: str = "standard") -> dict:
    X, _ = _X(data_key, scaler)
    rows = []
    inertia = {1: float(((X - X.mean(0)) ** 2).sum())}
    for k in range(2, 11):
        km = KMeans(k, init=init, n_init=n_init, random_state=RANDOM_STATE).fit(X)
        inertia[k] = float(km.inertia_)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_),
                     "dbi": davies_bouldin_score(X, km.labels_), "ch": calinski_harabasz_score(X, km.labels_),
                     "sizes": np.bincount(km.labels_).tolist()})
    rng = np.random.default_rng(RANDOM_STATE)
    lo, hi = X.min(0), X.max(0)
    refs = [rng.uniform(lo, hi, size=X.shape) for _ in range(gap_B)]
    gap = []
    for k in range(1, 11):
        wk = np.log(inertia[k])
        wb = np.array([np.log(KMeans(k, n_init=5, random_state=RANDOM_STATE).fit(r).inertia_) for r in refs])
        gap.append({"k": k, "gap": float(wb.mean() - wk), "sk": float(wb.std(ddof=0) * np.sqrt(1 + 1 / gap_B))})
    best_gap = next((g["k"] for i, g in enumerate(gap[:-1]) if g["gap"] >= gap[i + 1]["gap"] - gap[i + 1]["sk"]), 10)
    return {"rows": rows, "inertia_k1": inertia[1], "gap": gap, "gap_B": gap_B,
            "best": {"silhouette": max(rows, key=lambda r: r["silhouette"])["k"],
                     "dbi": min(rows, key=lambda r: r["dbi"])["k"], "ch": max(rows, key=lambda r: r["ch"])["k"],
                     "gap": best_gap}}


def _name_order(centers_orig: pd.DataFrame) -> list[int]:
    i_bull = int(centers_orig["log_ret"].idxmax())
    rest = [i for i in range(len(centers_orig)) if i != i_bull]
    i_act = max(rest, key=lambda i: centers_orig["rel_vol"][i])
    i_weak = [i for i in rest if i != i_act][0]
    return [i_bull, i_weak, i_act]


def fit(data_key: str, k: int = 3, init: str = "k-means++", n_init: int = 100, scaler: str = "standard",
        seed: int = RANDOM_STATE) -> dict:
    X, p = _X(data_key, scaler)
    d = p["data"]
    km = KMeans(k, init=init, n_init=n_init, max_iter=300, tol=1e-4, algorithm="lloyd", random_state=seed).fit(X)
    lab = km.labels_
    cent = d[CORE].groupby(lab).mean().reset_index(drop=True)
    if k == 3:
        order = _name_order(cent)
        remap = {old: new for new, old in enumerate(order)}
        lab = np.array([remap[v] for v in lab])
        names = NAMES
    else:
        order = list(np.argsort(-cent["log_ret"].values))
        remap = {old: new for new, old in enumerate(order)}
        lab = np.array([remap[v] for v in lab])
        names = [f"Klaster {i + 1}" for i in range(k)]
    centers_std = km.cluster_centers_[order]
    prof = d.assign(cluster=lab).groupby("cluster")[ALL_FEATS + ["next_ret"]].mean()
    pca = PCA(2, random_state=RANDOM_STATE).fit(X)
    sil = silhouette_samples(X, lab)
    return {
        "k": k, "names": names, "codes": [f"C{i + 1}" for i in range(k)], "labels": lab,
        "sizes": np.bincount(lab, minlength=k), "inertia": float(km.inertia_), "n_iter": int(km.n_iter_),
        "silhouette": float(sil.mean()), "silhouette_samples": sil,
        "dbi": davies_bouldin_score(X, lab), "ch": calinski_harabasz_score(X, lab),
        "bss_over_tss": float(1 - km.inertia_ / ((X - X.mean(0)) ** 2).sum()),
        "centers_std": centers_std, "profile": {int(c): r.to_dict() for c, r in prof.iterrows()},
        "pca": pca.transform(X), "pca_centers": pca.transform(centers_std), "pca_evr": pca.explained_variance_ratio_,
        "pca_loadings": pca.components_, "features": CORE, "feature_labels": FEAT_LABEL,
        "dates": d["Date"].dt.strftime("%Y-%m-%d").values, "close": d["Close"].values,
        "log_ret": d["log_ret"].values, "rel_vol": d["rel_vol"].values,
    }


def stability(data_key: str, k: int = 3, n_seeds: int = 20, init: str = "k-means++") -> dict:
    """Kestabilan lintas seed: ARI antar-solusi dan inertia untuk n_init=1 vs n_init=100."""
    X, _ = _X(data_key)
    out = {}
    for n_init in (1, 10, 100):
        labs, inert = [], []
        for s in range(n_seeds):
            km = KMeans(k, init=init, n_init=n_init, random_state=s).fit(X)
            labs.append(km.labels_)
            inert.append(km.inertia_)
        best = labs[int(np.argmin(inert))]
        pair = [adjusted_rand_score(labs[i], labs[j]) for i in range(n_seeds) for j in range(i + 1, n_seeds)]
        out[str(n_init)] = {"inertia": inert, "ari_vs_best": [adjusted_rand_score(best, l) for l in labs],
                            "pairwise_mean": float(np.mean(pair)), "pairwise_min": float(np.min(pair)),
                            "pct_best": float(np.mean(np.isclose(inert, np.min(inert), rtol=1e-9)) * 100)}
    per_k = []
    for kk in range(2, 8):
        labs = [KMeans(kk, n_init=20, random_state=s).fit(X).labels_ for s in range(6)]
        pair = [adjusted_rand_score(labs[i], labs[j]) for i in range(6) for j in range(i + 1, 6)]
        per_k.append({"k": kk, "pairwise_ari_mean": float(np.mean(pair)), "pairwise_ari_min": float(np.min(pair))})
    return {"by_n_init": out, "per_k": per_k, "n_seeds": n_seeds, "k": k}


def lloyd_setup(data_key: str, k: int = 3, init: str = "k-means++", seed: int = RANDOM_STATE) -> dict:
    """Data terstandar + pusat awal agar animasi Lloyd (assign → update) dihitung di browser;
    disertakan hasil scikit-learn dengan pusat awal yang sama sebagai verifikasi."""
    X, p = _X(data_key)
    rng = np.random.RandomState(seed)
    if init == "k-means++":
        C0, idx = kmeans_plusplus(X, k, random_state=seed)
    else:
        idx = rng.choice(len(X), k, replace=False)
        C0 = X[idx]
    sk = KMeans(k, init=C0, n_init=1, algorithm="lloyd", max_iter=300, tol=0).fit(X)
    pca = PCA(2, random_state=RANDOM_STATE).fit(X)
    return {"X": X, "init_centers": C0, "init_indices": np.asarray(idx).tolist(), "pca_mean": pca.mean_,
            "pca_components": pca.components_, "sk_labels": sk.labels_, "sk_centers": sk.cluster_centers_,
            "sk_inertia": float(sk.inertia_), "sk_n_iter": int(sk.n_iter_),
            "dates": p["data"]["Date"].dt.strftime("%Y-%m-%d").values}


def lloyd(X, C0, max_iter: int = 300):
    """Algoritma Lloyd versi NumPy (sama dengan lloyd_trace pada skrip laporan)."""
    X = np.asarray(X, float)
    C = np.array(C0, float)
    trace = []
    lab = None
    for _ in range(max_iter):
        D = cdist(X, C, "sqeuclidean")
        lab = D.argmin(1)
        trace.append(float(D[np.arange(len(X)), lab].sum()))
        Cn = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(len(C))])
        if np.allclose(Cn, C, atol=1e-12):
            break
        C = Cn
    return lab, C, trace


def manual_example(data_key: str) -> dict:
    """Contoh manual laporan: 10 hari pertama data bersih, fitur log_ret & range_pct
    (dibulatkan 2 desimal), K=3, pusat awal = 3 data acak (rng seed 42)."""
    p = prepare(data_key)
    sub = p["data"].loc[:9, ["Date", "log_ret", "range_pct"]]
    P = sub[["log_ret", "range_pct"]].round(2).values
    init_idx = sorted(np.random.default_rng(RANDOM_STATE).choice(10, 3, replace=False).tolist())
    C0 = P[init_idx].copy()
    sk = KMeans(3, init=C0, n_init=1, algorithm="lloyd", max_iter=300, tol=0).fit(P)
    return {"dates": sub["Date"].dt.strftime("%Y-%m-%d").tolist(), "points": P, "init_indices": init_idx,
            "init_centers": C0, "sk_labels": sk.labels_, "sk_centers": sk.cluster_centers_,
            "sk_inertia": float(sk.inertia_), "sk_n_iter": int(sk.n_iter_)}


def scaling_effect(data_key: str, k: int = 3) -> dict:
    rows = []
    ref = None
    for sc in ["standard", "none", "minmax", "robust"]:
        X, p = _X(data_key, sc)
        km = KMeans(k, n_init=50, random_state=RANDOM_STATE).fit(X)
        if ref is None:
            ref = km.labels_
        d = p["data"][CORE]
        eta = {}
        for c in CORE:
            v = d[c].values
            ssb = sum(len(v[km.labels_ == j]) * (v[km.labels_ == j].mean() - v.mean()) ** 2 for j in range(k))
            eta[c] = float(ssb / ((v - v.mean()) ** 2).sum())
        rows.append({"scaler": sc, "silhouette": silhouette_score(X, km.labels_),
                     "ari_vs_standard": adjusted_rand_score(ref, km.labels_), "eta2": eta,
                     "sizes": np.bincount(km.labels_).tolist()})
    return {"rows": rows}
