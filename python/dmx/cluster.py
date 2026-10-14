"""Alat clustering umum (K-Means) pada kolom numerik CSV apa pun."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from . import RANDOM_STATE
from .common import get


def _X(data_key: str, features: list[str], scaler: str):
    df = get(data_key)
    d = df.dropna(subset=features).reset_index(drop=True)
    X = d[features].values.astype(float)
    sc = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler(), "none": None}[scaler]
    return (sc.fit_transform(X) if sc is not None else X), d


def k_metrics(data_key: str, features: list[str], scaler: str = "standard", init: str = "k-means++",
              n_init: int = 10, k_max: int = 10) -> dict:
    X, d = _X(data_key, features, scaler)
    if len(X) > 5000:
        X = X[np.random.default_rng(RANDOM_STATE).choice(len(X), 5000, replace=False)]
    rows = []
    for k in range(2, min(k_max, len(X) - 1) + 1):
        km = KMeans(k, init=init, n_init=n_init, random_state=RANDOM_STATE).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_),
                     "dbi": davies_bouldin_score(X, km.labels_), "ch": calinski_harabasz_score(X, km.labels_)})
    return {"rows": rows, "n": len(X),
            "best": {"silhouette": max(rows, key=lambda r: r["silhouette"])["k"], "dbi": min(rows, key=lambda r: r["dbi"])["k"],
                     "ch": max(rows, key=lambda r: r["ch"])["k"]}}


def fit(data_key: str, features: list[str], k: int = 3, scaler: str = "standard", init: str = "k-means++",
        n_init: int = 10, label_col: str | None = None) -> dict:
    X, d = _X(data_key, features, scaler)
    km = KMeans(k, init=init, n_init=n_init, random_state=RANDOM_STATE).fit(X)
    lab = km.labels_
    pca = PCA(2, random_state=RANDOM_STATE).fit(X)
    prof = d[features].groupby(lab).mean()
    s = np.arange(len(X)) if len(X) <= 3000 else np.random.default_rng(RANDOM_STATE).choice(len(X), 3000, replace=False)
    out = {"k": k, "n": len(X), "labels": lab[s], "sizes": np.bincount(lab, minlength=k), "inertia": km.inertia_,
           "n_iter": int(km.n_iter_), "silhouette": float(silhouette_score(X[s], lab[s])) if k > 1 else None,
           "dbi": davies_bouldin_score(X, lab), "ch": calinski_harabasz_score(X, lab),
           "pca": pca.transform(X[s]), "pca_centers": pca.transform(km.cluster_centers_), "pca_evr": pca.explained_variance_ratio_,
           "profile": {int(c): r.to_dict() for c, r in prof.iterrows()}, "features": features,
           "hover": (d[label_col].astype(str).values[s] if label_col and label_col in d else None)}
    out["csv"] = d.assign(cluster=lab + 1).to_csv(index=False) if len(d) <= 20000 else None
    return out


def lloyd_setup(data_key: str, features: list[str], k: int = 3, scaler: str = "standard", init: str = "k-means++",
                seed: int = RANDOM_STATE, max_points: int = 1500) -> dict:
    X, _ = _X(data_key, features, scaler)
    if len(X) > max_points:
        X = X[np.random.default_rng(RANDOM_STATE).choice(len(X), max_points, replace=False)]
    if init == "k-means++":
        C0, idx = kmeans_plusplus(X, k, random_state=seed)
    else:
        idx = np.random.RandomState(seed).choice(len(X), k, replace=False)
        C0 = X[idx]
    sk = KMeans(k, init=C0, n_init=1, algorithm="lloyd", max_iter=300, tol=0).fit(X)
    pca = PCA(2, random_state=RANDOM_STATE).fit(X)
    return {"X": X, "init_centers": C0, "pca_mean": pca.mean_, "pca_components": pca.components_,
            "sk_labels": sk.labels_, "sk_inertia": float(sk.inertia_)}


def stability(data_key: str, features: list[str], k: int = 3, scaler: str = "standard", n_seeds: int = 10) -> dict:
    X, _ = _X(data_key, features, scaler)
    if len(X) > 3000:
        X = X[np.random.default_rng(RANDOM_STATE).choice(len(X), 3000, replace=False)]
    out = {}
    for n_init in (1, 10):
        labs, inert = [], []
        for s in range(n_seeds):
            km = KMeans(k, n_init=n_init, random_state=s).fit(X)
            labs.append(km.labels_)
            inert.append(km.inertia_)
        pair = [adjusted_rand_score(labs[i], labs[j]) for i in range(n_seeds) for j in range(i + 1, n_seeds)]
        out[str(n_init)] = {"pairwise_mean": float(np.mean(pair)), "pairwise_min": float(np.min(pair)),
                            "pct_best": float(np.mean(np.isclose(inert, np.min(inert), rtol=1e-9)) * 100)}
    return out
