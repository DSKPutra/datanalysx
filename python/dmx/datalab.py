"""Data Lab (klasifikasi) dan Regression Lab (regresi) generik.

Semua praproses berada di dalam Pipeline/ColumnTransformer sehingga imputasi, scaling,
dan encoding di-fit HANYA pada data latih di setiap lipatan (tanpa kebocoran data).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import RANDOM_STATE
from .common import get, regression_metrics


def _split_cols(df: pd.DataFrame, features: list[str]):
    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    cat = [c for c in features if c not in num and df[c].nunique() <= 50]
    return num, cat


def _prep(num, cat, scale=True):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    tr = [("num", Pipeline(steps), num)]
    if cat:
        tr.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(tr)


def classify(data_key: str, target: str, features: list[str] | None = None, cv_folds: int = 5,
             models: list[str] | None = None) -> dict:
    df = get(data_key).dropna(subset=[target]).copy()
    feats = features or [c for c in df.columns if c != target]
    num, cat = _split_cols(df, feats)
    dropped = [c for c in feats if c not in num + cat]
    y_raw = df[target].astype(str)
    classes = sorted(y_raw.unique())
    if len(classes) < 2:
        raise ValueError("Target harus memiliki minimal 2 kelas.")
    if len(classes) > 20:
        raise ValueError("Target memiliki terlalu banyak kelas (>20); pilih kolom kategorikal.")
    y = y_raw.map({c: i for i, c in enumerate(classes)}).values
    X = df[num + cat]
    all_models = {
        "Baseline (mayoritas)": (DummyClassifier(strategy="most_frequent"), True),
        "Regresi Logistik": (LogisticRegression(max_iter=2000), True),
        "Gaussian Naïve Bayes": (GaussianNB(), False),
        "KNN (k=5)": (KNeighborsClassifier(5), True),
        "Decision Tree (depth 5)": (DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), False),
        "Random Forest (200)": (RandomForestClassifier(200, random_state=RANDOM_STATE), False),
    }
    min_class = np.bincount(y).min()
    k = int(max(2, min(cv_folds, min_class)))
    cv = StratifiedKFold(k, shuffle=True, random_state=RANDOM_STATE)
    binary = len(classes) == 2
    scoring = ["accuracy", "balanced_accuracy", "f1_macro"] + (["roc_auc"] if binary else ["roc_auc_ovr"])
    rows = []
    for name in models or list(all_models):
        est, scale = all_models[name]
        pipe = Pipeline([("prep", _prep(num, cat, scale)), ("clf", est)])
        t0 = time.perf_counter()
        r = cross_validate(pipe, X, y, cv=cv, scoring=scoring, error_score=np.nan)
        auc_key = "test_roc_auc" if binary else "test_roc_auc_ovr"
        rows.append({"model": name, "accuracy": float(np.nanmean(r["test_accuracy"])),
                     "accuracy_std": float(np.nanstd(r["test_accuracy"])),
                     "balanced_accuracy": float(np.nanmean(r["test_balanced_accuracy"])),
                     "f1_macro": float(np.nanmean(r["test_f1_macro"])), "roc_auc": float(np.nanmean(r[auc_key])),
                     "seconds": time.perf_counter() - t0})
    rows.sort(key=lambda r: -r["accuracy"])
    return {"rows": rows, "classes": classes, "class_counts": np.bincount(y).tolist(), "n": len(y),
            "numeric": num, "categorical": cat, "dropped": dropped, "cv_folds": k}


def regress(data_key: str, target: str, features: list[str] | None = None, split: str = "random",
            date_col: str | None = None, models: list[str] | None = None) -> dict:
    """Regression Lab: bandingkan Linear, log-log, Ridge, Lasso, Decision Tree, Random Forest, kNN."""
    df = get(data_key).copy()
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=df[date_col].astype(str).str.match(r"^\d{1,2}/").all())
        df = df.sort_values(date_col).reset_index(drop=True)
    feats = [c for c in (features or [c for c in df.columns if c != target]) if c != date_col]
    num, cat = _split_cols(df, feats)
    df = df.dropna(subset=[target])
    X, y = df[num + cat], df[target].astype(float).values
    if split == "temporal":
        cut = int(len(df) * 0.8)
        tr, te = np.arange(cut), np.arange(cut, len(df))
        cv = TimeSeriesSplit(5)
    else:
        tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE)
        cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    all_models = {
        "Baseline (rata-rata)": (DummyRegressor(), True),
        "Linear Regression": (LinearRegression(), True),
        "Ridge (α=1)": (Ridge(1.0), True),
        "Lasso (α=0,01)": (Lasso(0.01, max_iter=20000), True),
        "Decision Tree (depth 6)": (DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE), False),
        "Random Forest (200)": (RandomForestRegressor(200, random_state=RANDOM_STATE), False),
        "kNN (k=5)": (KNeighborsRegressor(5), True),
    }
    can_log = bool((y > 0).all()) and all((df[c].dropna() > 0).all() for c in num)
    rows, preds = [], {}
    for name in models or list(all_models):
        est, scale = all_models[name]
        pipe = Pipeline([("prep", _prep(num, cat, scale)), ("reg", est)])
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict(X.iloc[te])
        preds[name] = p
        r = cross_validate(pipe, X, y, cv=cv, scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"))
        rows.append({"model": name, **regression_metrics(y[te], p), "cv_r2": float(r["test_r2"].mean()),
                     "cv_rmse": float(-r["test_neg_root_mean_squared_error"].mean()),
                     "cv_mae": float(-r["test_neg_mean_absolute_error"].mean())})
    if can_log and num:
        Xl = np.log(df[num].values.astype(float))
        m = LinearRegression().fit(Xl[tr], np.log(y[tr]))
        p = np.exp(m.predict(Xl[te]))
        preds["Log-log (OLS)"] = p
        rows.append({"model": "Log-log (OLS)", **regression_metrics(y[te], p), "cv_r2": None, "cv_rmse": None,
                     "cv_mae": None})
    # Panel asumsi klasik untuk OLS pada data latih (fitur numerik)
    assumptions = None
    if num:
        Xa = np.column_stack([np.ones(len(tr)), SimpleImputer(strategy="median").fit_transform(df[num].iloc[tr])])
        ya = y[tr]
        b, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
        res = ya - Xa @ b
        e2 = res ** 2
        g, *_ = np.linalg.lstsq(Xa, e2, rcond=None)
        r2_bp = 1 - ((e2 - Xa @ g) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum()
        lm = len(ya) * r2_bp
        sw = stats.shapiro(res[:5000])
        vifs = []
        for j in range(1, Xa.shape[1]):
            o = [i for i in range(Xa.shape[1]) if i != j]
            gg, *_ = np.linalg.lstsq(Xa[:, o], Xa[:, j], rcond=None)
            rr = Xa[:, j] - Xa[:, o] @ gg
            r2 = 1 - (rr @ rr) / ((Xa[:, j] - Xa[:, j].mean()) ** 2).sum()
            vifs.append(float(1 / (1 - r2)) if r2 < 0.9999999 else None)
        assumptions = {"breusch_pagan_lm": float(lm), "breusch_pagan_p": float(stats.chi2.sf(lm, Xa.shape[1] - 1)),
                       "shapiro_w": float(sw.statistic), "shapiro_p": float(sw.pvalue),
                       "durbin_watson": float((np.diff(res) ** 2).sum() / (res @ res)),
                       "vif": dict(zip(num, vifs)), "fitted": (Xa @ b)[:2000], "resid": res[:2000]}
    rows.sort(key=lambda r: -(r["R2"] if r["R2"] is not None else -1e9))
    return {"rows": rows, "n_train": len(tr), "n_test": len(te), "split": split, "numeric": num, "categorical": cat,
            "actual": y[te], "preds": preds, "assumptions": assumptions, "can_log": can_log}
