"""Studi Kasus 07 — K-Nearest Neighbors: Instagram Reels "Viral Hit" vs "High Reach".

Diringkas dari python/original/knn_instagram_reels.py. Memakai berkas
instagram_reels_knn_clean.csv (fitur sudah direkayasa oleh skrip asli). Kolom pembentuk
target (views, likes, comments, engagement_rate_pct, virality_score) TIDAK dipakai.
Praproses dibungkus Pipeline: IQR capping (1,5×, di-fit pada data latih) → scaler,
biner passthrough, niche one-hot. Split terstratifikasi 75:25, random_state=42.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler

from . import RANDOM_STATE
from .common import cached, get

NUM_FEATS = ["video_duration_sec", "caption_word_count", "caption_char_len", "hashtag_count",
             "emoji_count", "mention_count", "post_age_days", "hour_sin", "hour_cos"]
BIN_FEATS = ["is_weekend", "has_question"]
CAT_FEATS = ["niche"]
FEATURES = NUM_FEATS + BIN_FEATS + CAT_FEATS
LEAKAGE_COLS = ["views", "likes", "comments", "engagement_rate_pct", "virality_score", "virality_tier"]
CLS = {0: "High Reach", 1: "Viral Hit"}


class IQRCapper(BaseEstimator, TransformerMixin):
    """Winsorization IQR; batas dihitung HANYA dari data latih (fit)."""

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


def make_prep(scaler: str = "standard"):
    sc = {"none": FunctionTransformer(feature_names_out="one-to-one"), "standard": StandardScaler(),
          "minmax": MinMaxScaler()}[scaler]
    return ColumnTransformer([
        ("num", Pipeline([("cap", IQRCapper(1.5)), ("scaler", sc)]), NUM_FEATS),
        ("bin", "passthrough", BIN_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS),
    ])


def make_knn(scaler="minmax", k=5, metric="cosine", weights="distance", p=2):
    kw = {"n_neighbors": k, "metric": metric, "weights": weights}
    if metric == "minkowski":
        kw["p"] = p
    return Pipeline([("prep", make_prep(scaler)), ("knn", KNeighborsClassifier(**kw))])


def data(data_key: str):
    def build():
        df = get(data_key).copy()
        df["target"] = (df["virality_tier"] == "Viral Hit").astype(int)
        X, y = df[FEATURES], df["target"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
        return df, X, y, Xtr, Xte, ytr, yte
    return cached(f"{data_key}:t07", build)


def overview(data_key: str) -> dict:
    df, X, y, Xtr, Xte, ytr, yte = data(data_key)
    return {"n": len(df), "n_train": len(ytr), "n_test": len(yte), "class_counts": {CLS[k]: int(v) for k, v in
            pd.Series(y).value_counts().sort_index().items()},
            "niche_viral_rate": df.groupby("niche")["target"].mean().to_dict(),
            "niche_counts": df["niche"].value_counts().to_dict(),
            "creator_counts": df["creator_handle"].value_counts().to_dict() if "creator_handle" in df else {},
            "corr_leak": {c: float(df[c].corr(df["target"])) for c in LEAKAGE_COLS[:-1]},
            "features": FEATURES, "leakage_cols": LEAKAGE_COLS,
            "describe": df[NUM_FEATS + BIN_FEATS].describe().T[["mean", "std", "min", "max"]].to_dict(orient="index")}


def _scores(model, X, y, Xtr, Xte, ytr, yte, n_repeats=5):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)
    # Seperti laporan: validasi silang dilakukan pada DATA LATIH saja (data uji disimpan).
    r = cross_validate(model, Xtr, ytr, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
    model.fit(Xtr, ytr)
    pp = model.predict_proba(Xte)[:, 1]
    yp = model.predict(Xte)
    return {"cv_acc": float(r["test_accuracy"].mean()), "cv_acc_std": float(r["test_accuracy"].std()),
            "cv_f1": float(r["test_f1"].mean()), "cv_auc": float(np.nanmean(r["test_roc_auc"])),
            "test": {"accuracy": accuracy_score(yte, yp), "precision": precision_score(yte, yp, zero_division=0),
                     "recall": recall_score(yte, yp), "f1": f1_score(yte, yp),
                     "balanced_acc": balanced_accuracy_score(yte, yp), "roc_auc": roc_auc_score(yte, pp),
                     "avg_precision": average_precision_score(yte, pp),
                     "confusion_matrix": confusion_matrix(yte, yp).tolist()}}


def evaluate(data_key: str, k: int = 5, metric: str = "cosine", weights: str = "distance",
             scaler: str = "minmax", p: int = 2, n_repeats: int = 5) -> dict:
    """Akurasi CV pada data latih (RepeatedStratifiedKFold 5×n_repeats, seperti GridSearch laporan)
    + metrik pada data uji."""
    df, X, y, Xtr, Xte, ytr, yte = data(data_key)
    out = _scores(make_knn(scaler, k, metric, weights, p), X, y, Xtr, Xte, ytr, yte, n_repeats)
    out["params"] = {"k": k, "metric": metric, "weights": weights, "scaler": scaler, "p": p}
    lr = Pipeline([("prep", make_prep("standard")),
                   ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    out["baseline_majority_test"] = float((yte == np.bincount(ytr).argmax()).mean())
    out["lr_reference"] = _scores(lr, X, y, Xtr, Xte, ytr, yte, n_repeats)
    return out


def k_curve(data_key: str, metric: str = "cosine", weights: str = "distance", scaler: str = "minmax",
            p: int = 2, n_repeats: int = 2) -> dict:
    df, X, y, Xtr, Xte, ytr, yte = data(data_key)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)
    ks = list(range(1, 50, 2))
    rows = []
    for k in ks:
        r = cross_validate(make_knn(scaler, k, metric, weights, p), Xtr, ytr, cv=cv, scoring="accuracy",
                           return_train_score=True)
        rows.append({"k": k, "cv": float(r["test_score"].mean()), "cv_std": float(r["test_score"].std()),
                     "train": float(r["train_score"].mean())})
    return {"rows": rows}


def neighbors(data_key: str, test_pos: int | None = None, k: int = 5, metric: str = "cosine",
              weights: str = "distance", scaler: str = "minmax", p: int = 2) -> dict:
    """Matriks fitur ter-transformasi (latih & satu titik uji) agar jarak, urutan, dan voting
    dapat dihitung ulang secara manual di browser; ditambah hasil scikit-learn sebagai pembanding."""
    df, X, y, Xtr, Xte, ytr, yte = data(data_key)
    model = make_knn(scaler, k, metric, weights, p).fit(Xtr, ytr)
    prep, knn = model.named_steps["prep"], model.named_steps["knn"]
    Ttr, Tte = prep.transform(Xtr), prep.transform(Xte)
    if test_pos is None:  # sama seperti laporan: data uji pertama yang tetangganya tidak seragam kelasnya
        nb = knn.kneighbors(Tte, return_distance=False)
        mixed = [i for i in range(len(Tte)) if len(set(ytr[nb[i]])) > 1]
        test_pos = mixed[0] if mixed else 0
    q = Tte[test_pos]
    d, idx = knn.kneighbors(q.reshape(1, -1))
    pca = PCA(2, random_state=RANDOM_STATE).fit(Ttr)
    return {
        "test_pos": int(test_pos), "row_index": int(Xte.index[test_pos]),
        "shortcode": str(df.loc[Xte.index[test_pos], "shortcode"]) if "shortcode" in df else None,
        "actual": int(yte[test_pos]), "feature_names": [str(f) for f in prep.get_feature_names_out()],
        "query": q, "train": Ttr, "train_y": ytr, "train_index": [int(i) for i in Xtr.index],
        "raw_query": {c: (Xte.iloc[test_pos][c] if isinstance(Xte.iloc[test_pos][c], str)
                          else float(Xte.iloc[test_pos][c])) for c in FEATURES},
        "params": {"k": k, "metric": metric, "weights": weights, "scaler": scaler, "p": p},
        "sk_neighbors": idx[0], "sk_distances": d[0], "sk_proba": model.predict_proba(Xte.iloc[[test_pos]])[0],
        "sk_pred": int(model.predict(Xte.iloc[[test_pos]])[0]),
        "pca_train": pca.transform(Ttr), "pca_query": pca.transform(q.reshape(1, -1))[0],
        "pca_evr": pca.explained_variance_ratio_, "n_test": len(yte),
    }


def manual_knn(query, train, train_y, k, metric="euclidean", weights="uniform", p=2) -> dict:
    """Versi Python dari langkah manual (jarak → urut → K terdekat → voting)."""
    a, B = np.asarray(query, float), np.asarray(train, float)
    if metric == "euclidean":
        dist = np.sqrt(((B - a) ** 2).sum(1))
    elif metric == "manhattan":
        dist = np.abs(B - a).sum(1)
    elif metric == "chebyshev":
        dist = np.abs(B - a).max(1)
    elif metric == "minkowski":
        dist = (np.abs(B - a) ** p).sum(1) ** (1 / p)
    elif metric == "cosine":
        dist = 1 - (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a))
    else:
        raise ValueError(metric)
    order = np.argsort(dist, kind="stable")[:k]
    w = 1 / np.maximum(dist[order], 1e-12) if weights == "distance" else np.ones(k)
    ty = np.asarray(train_y)
    votes = np.array([w[ty[order] == c].sum() for c in [0, 1]])
    return {"distances": dist, "neighbors": order, "weights": w, "votes": votes, "proba": votes / votes.sum(),
            "pred": int(np.argmax(votes))}


def curse(data_key: str, noise_dims: list[int] | None = None, n_repeats: int = 2) -> dict:
    """Tambahkan fitur derau U(0,1) lalu ukur kontras jarak (dmax−dmin)/dmin dan akurasi CV."""
    noise_dims = noise_dims or [0, 5, 10, 20, 50, 100]
    df, X, y, Xtr, Xte, ytr, yte = data(data_key)
    rng = np.random.default_rng(RANDOM_STATE)
    base = make_prep("minmax").fit_transform(X)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)
    rows = []
    for nd in noise_dims:
        Z = np.hstack([base, rng.uniform(size=(len(base), nd))]) if nd else base
        D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(D, np.nan)
        contrast = float(np.nanmean((np.nanmax(D, 1) - np.nanmin(D, 1)) / np.nanmin(D, 1)))
        acc_e = cross_validate(KNeighborsClassifier(5), Z, y, cv=cv)["test_score"].mean()
        acc_c = cross_validate(KNeighborsClassifier(5, metric="cosine", weights="distance"), Z, y, cv=cv)["test_score"].mean()
        rows.append({"noise_dims": nd, "total_dims": Z.shape[1], "contrast": contrast,
                     "acc_euclidean": float(acc_e), "acc_cosine": float(acc_c)})
    theory = []
    for dim in [2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        P = rng.uniform(size=(300, dim))
        d = np.sqrt(((P[1:] - P[0]) ** 2).sum(1))
        theory.append({"dims": dim, "contrast": float((d.max() - d.min()) / d.min())})
    return {"rows": rows, "theory": theory}


def timing(sizes: list[int] | None = None, n_features: int = 16, n_query: int = 200) -> dict:
    """Lazy learner: waktu fit KNN hampir konstan, waktu prediksi tumbuh dengan ukuran data latih."""
    sizes = sizes or [500, 1000, 2000, 5000, 10000, 20000]
    rng = np.random.default_rng(RANDOM_STATE)
    Q = rng.normal(size=(n_query, n_features))
    rows = []
    for n in sizes:
        Xs = rng.normal(size=(n, n_features))
        ys = (Xs[:, 0] + rng.normal(size=n) > 0).astype(int)
        r = {"n": n}
        for name, m in [("knn_brute", KNeighborsClassifier(5, algorithm="brute")),
                        ("knn_kdtree", KNeighborsClassifier(5, algorithm="kd_tree")),
                        ("logreg", LogisticRegression(max_iter=500))]:
            t0 = time.perf_counter()
            m.fit(Xs, ys)
            t1 = time.perf_counter()
            m.predict(Q)
            t2 = time.perf_counter()
            r[name + "_fit_ms"] = 1000 * (t1 - t0)
            r[name + "_pred_ms"] = 1000 * (t2 - t1)
        rows.append(r)
    return {"rows": rows, "n_query": n_query, "n_features": n_features,
            "note": "Waktu bergantung pada perangkat & browser; tidak deterministik."}
