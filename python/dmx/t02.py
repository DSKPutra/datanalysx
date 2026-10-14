"""Studi Kasus 02 — Regresi Linear vs Random Forest Regression: tingkat keparahan kegagalan agen AI.

Diringkas dari lampiran kode laporan studi kasus 02 (python/original/studikasus02_regresi_linear_rf.py).
Rekayasa fitur: difficulty_score (Easy=1, Medium=2, Hard=3), prompt_length dan context_length
(jumlah kata), target severity_score (Low=1 ... Critical=4). Split acak 80:20, random_state=42.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from . import RANDOM_STATE
from .common import cached, get, regression_metrics

FEATURES = ["difficulty_score", "prompt_length", "context_length"]
DIFFICULTY = {"Easy": 1, "Medium": 2, "Hard": 3}
SEVERITY = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}


def prepare(data_key: str) -> pd.DataFrame:
    def build():
        df = get(data_key).copy()
        df["difficulty_score"] = df["difficulty"].map(DIFFICULTY)
        df["severity_score"] = df["failure_severity"].map(SEVERITY)
        df["prompt_length"] = df["prompt"].astype(str).str.split().apply(len)
        df["context_length"] = df["context"].astype(str).str.split().apply(len)
        return df
    return cached(f"{data_key}:t02", build)


def design(data_key: str, features: list[str], extra_cats: list[str] | None = None):
    """Matriks fitur; extra_cats (mis. ['domain', 'failure_type']) di-one-hot-kan.
    One-hot bersifat deterministik per baris (tidak di-fit pada data), jadi tidak bocor."""
    df = prepare(data_key)
    X = df[features].astype(float)
    if extra_cats:
        X = pd.concat([X, pd.get_dummies(df[extra_cats].astype(str), prefix=extra_cats).astype(float)], axis=1)
    return df, X, df["severity_score"].values.astype(float)


def _split(data_key: str, features: list[str], extra_cats: list[str] | None = None):
    df, X, y = design(data_key, features, extra_cats)
    return df, train_test_split(X.values, y, test_size=0.2, random_state=RANDOM_STATE)


def _regressors(n_estimators=100, max_depth=None, min_samples_leaf=1):
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                               random_state=RANDOM_STATE),
        "kNN (k=5, terstandar)": make_pipeline(StandardScaler(), KNeighborsRegressor(5)),
        "Random Forest": RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf, random_state=RANDOM_STATE),
    }


def compare(data_key: str, n_estimators: int = 100, max_depth: int | None = None, min_samples_leaf: int = 1,
            extra_cats: list[str] | None = None, cv: bool = True) -> dict:
    """Bandingkan LR, Decision Tree, kNN regresi, dan Random Forest: hold-out 80:20
    dan (opsional) validasi silang 5-lipatan (KFold shuffle, random_state=42)."""
    df, X, y = design(data_key, FEATURES, extra_cats)
    Xtr, Xte, ytr, yte = train_test_split(X.values, y, test_size=0.2, random_state=RANDOM_STATE)
    rows = []
    preds = {}
    for name, m in _regressors(n_estimators, max_depth, min_samples_leaf).items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        preds[name] = p
        row = {"model": name, **regression_metrics(yte, p)}
        if cv:
            r = cross_validate(m, X.values, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
                               scoring=("r2", "neg_mean_absolute_error"))
            row.update({"cv_r2_mean": float(r["test_r2"].mean()), "cv_r2_std": float(r["test_r2"].std()),
                        "cv_mae_mean": float(-r["test_neg_mean_absolute_error"].mean())})
        rows.append(row)
    return {"rows": rows, "n_features": X.shape[1], "extra_cats": extra_cats or [], "actual": yte,
            "preds": preds}


def run(data_key: str, features: list[str] | None = None, n_estimators: int = 100,
        max_depth: int | None = None, min_samples_leaf: int = 1) -> dict:
    features = features or FEATURES
    df, (Xtr, Xte, ytr, yte) = _split(data_key, features)
    lin = LinearRegression().fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                               min_samples_leaf=min_samples_leaf, random_state=RANDOM_STATE).fit(Xtr, ytr)
    p_lin, p_rf = lin.predict(Xte), rf.predict(Xte)
    desc = df[FEATURES + ["severity_score"]].describe().T[["mean", "std", "min", "max"]]
    return {
        "shape": list(df.shape), "n_train": len(ytr), "n_test": len(yte),
        "missing": int(df[["difficulty", "prompt", "context", "failure_severity"]].isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "describe": {i: row.to_dict() for i, row in desc.iterrows()},
        "linear": {"metrics": regression_metrics(yte, p_lin), "coef": dict(zip(features, lin.coef_)),
                   "intercept": float(lin.intercept_)},
        "rf": {"metrics": regression_metrics(yte, p_rf),
               "importances": dict(zip(features, rf.feature_importances_)),
               "params": {"n_estimators": n_estimators, "max_depth": max_depth,
                          "min_samples_leaf": min_samples_leaf}},
        "test": {"actual": yte, "pred_linear": p_lin, "pred_rf": p_rf},
        "scatter_context": {"x": df["context_length"].values, "y": df["severity_score"].values},
        "severity_counts": df["failure_severity"].value_counts().to_dict(),
        "features": features,
    }


def manual_predict(data_key: str, test_index: int = 0, n_estimators: int = 100) -> dict:
    """Satu baris uji: prediksi LR dihitung dari persamaan b₀ + Σbⱼxⱼ, dan prediksi Random
    Forest sebagai rata-rata prediksi seluruh pohon; keduanya dicocokkan dengan predict()."""
    _, (Xtr, Xte, ytr, yte) = _split(data_key, FEATURES)
    lin = LinearRegression().fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE).fit(Xtr, ytr)
    i = int(np.clip(test_index, 0, len(yte) - 1))
    x = Xte[i]
    terms = lin.coef_ * x
    lin_manual = float(lin.intercept_ + terms.sum())
    trees = np.array([t.predict(x.reshape(1, -1))[0] for t in rf.estimators_])
    rf_manual = float(trees.mean())
    return {"test_index": i, "x": dict(zip(FEATURES, x.tolist())), "actual": float(yte[i]),
            "lin": {"intercept": float(lin.intercept_), "coef": lin.coef_.tolist(), "terms": terms.tolist(),
                    "manual": lin_manual, "sklearn": float(lin.predict(x.reshape(1, -1))[0])},
            "rf": {"tree_preds": trees.tolist(), "manual": rf_manual,
                   "sklearn": float(rf.predict(x.reshape(1, -1))[0])},
            "max_abs_diff": float(max(abs(lin_manual - lin.predict(x.reshape(1, -1))[0]),
                                      abs(rf_manual - rf.predict(x.reshape(1, -1))[0])))}


def feature_table(data_key: str, n_rows: int = 12) -> dict:
    """Contoh baris: teks asli → fitur angka (untuk tabel rekayasa fitur yang bisa diklik)."""
    df = prepare(data_key).head(n_rows)
    return {"rows": [{"task_id": str(r.task_id), "domain": str(r.domain), "difficulty": str(r.difficulty),
                      "prompt": str(r.prompt), "context": str(r.context), "failure_type": str(r.failure_type),
                      "failure_severity": str(r.failure_severity), "difficulty_score": int(r.difficulty_score),
                      "prompt_length": int(r.prompt_length), "context_length": int(r.context_length),
                      "severity_score": int(r.severity_score)} for r in df.itertuples()]}


def manual_metrics(data_key: str, n_rows: int = 8) -> dict:
    """Hitung MAE, RMSE, dan R² secara manual dari prediksi Linear Regression pada data uji,
    lalu bandingkan dengan fungsi metrik scikit-learn."""
    _, (Xtr, Xte, ytr, yte) = _split(data_key, FEATURES)
    lin = LinearRegression().fit(Xtr, ytr)
    p = lin.predict(Xte)
    e = yte - p
    mae = np.abs(e).sum() / len(e)
    rmse = np.sqrt((e ** 2).sum() / len(e))
    r2 = 1 - (e ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    rows = [{"x": Xte[i].tolist(), "y": float(yte[i]), "pred": float(p[i]), "abs_err": float(abs(e[i])),
             "sq_err": float(e[i] ** 2)} for i in range(min(n_rows, len(yte)))]
    sk = {"MAE": mean_absolute_error(yte, p), "RMSE": float(np.sqrt(mean_squared_error(yte, p))),
          "R2": r2_score(yte, p)}
    manual = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
    return {"rows": rows, "n_test": len(yte), "sum_abs": float(np.abs(e).sum()), "sum_sq": float((e ** 2).sum()),
            "sst": float(((yte - yte.mean()) ** 2).sum()), "mean_y": float(yte.mean()),
            "coef": lin.coef_.tolist(), "intercept": float(lin.intercept_),
            "manual": manual, "sklearn": sk,
            "max_abs_diff": max(abs(manual[k] - sk[k]) for k in manual)}
