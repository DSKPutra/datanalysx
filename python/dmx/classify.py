"""Alat klasifikasi umum: satu algoritma dengan parameter yang dapat diatur, pada CSV apa pun.

Praproses (imputasi, scaling opsional, one-hot) berada di dalam Pipeline sehingga di-fit
hanya pada data latih. Mengembalikan metrik, confusion matrix, kurva ROC (biner), validasi
silang, serta muatan "perhitungan manual" untuk satu baris uji (Regresi Logistik biner,
Gaussian Naïve Bayes, KNN, Decision Tree) yang dapat diverifikasi di browser.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from . import RANDOM_STATE
from .common import get, roc_points

ALGOS = {"logreg": "Regresi Logistik", "gnb": "Gaussian Naïve Bayes", "knn": "K-Nearest Neighbors",
         "tree": "Decision Tree", "rf": "Random Forest"}


def _estimator(algo: str, p: dict):
    if algo == "logreg":
        return LogisticRegression(C=float(p.get("C", 1.0)), max_iter=3000, random_state=RANDOM_STATE)
    if algo == "gnb":
        return GaussianNB()
    if algo == "knn":
        return KNeighborsClassifier(n_neighbors=int(p.get("k", 5)), metric=p.get("metric", "euclidean"),
                                    weights=p.get("weights", "uniform"))
    if algo == "tree":
        md = p.get("max_depth")
        return DecisionTreeClassifier(criterion=p.get("criterion", "gini"), max_depth=int(md) if md else None,
                                      min_samples_leaf=int(p.get("min_samples_leaf", 1)), random_state=RANDOM_STATE)
    if algo == "rf":
        md = p.get("max_depth")
        return RandomForestClassifier(n_estimators=int(p.get("n_estimators", 200)), max_depth=int(md) if md else None,
                                      random_state=RANDOM_STATE)
    raise ValueError(f"Algoritma tidak dikenal: {algo}")


def _prep(num, cat, scaler):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scaler == "standard":
        steps.append(("sc", StandardScaler()))
    elif scaler == "minmax":
        steps.append(("sc", MinMaxScaler()))
    tr = [("num", Pipeline(steps), num)]
    if cat:
        tr.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(tr)


def _design(data_key, target, features, date_col):
    df = get(data_key).dropna(subset=[target]).copy()
    if date_col and date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce", dayfirst=df[date_col].astype(str).str.match(r"^\d{1,2}/").all())
        df = df.assign(__d=d).sort_values("__d").drop(columns="__d").reset_index(drop=True)
    feats = [c for c in features if c not in (target, date_col)]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    cat = [c for c in feats if c not in num and df[c].nunique() <= 50]
    y_raw = df[target].astype(str)
    classes = sorted(y_raw.unique())
    if not 2 <= len(classes) <= 20:
        raise ValueError("Target harus memiliki 2–20 kelas.")
    y = y_raw.map({c: i for i, c in enumerate(classes)}).values
    return df, num, cat, classes, y


def run(data_key: str, target: str, features: list[str], algo: str = "logreg", params: dict | None = None,
        scaler: str = "standard", split: str = "stratified", test_size: float = 0.25, date_col: str | None = None,
        manual_index: int = 0) -> dict:
    params = params or {}
    df, num, cat, classes, y = _design(data_key, target, features, date_col)
    X = df[num + cat]
    idx = np.arange(len(df))
    if split == "temporal":
        cut = int(len(df) * (1 - test_size))
        tr, te = idx[:cut], idx[cut:]
        cv = TimeSeriesSplit(5)
    else:
        tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
        cv = StratifiedKFold(int(max(2, min(5, np.bincount(y[tr]).min()))), shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([("prep", _prep(num, cat, scaler)), ("clf", _estimator(algo, params))])
    pipe.fit(X.iloc[tr], y[tr])
    pred = pipe.predict(X.iloc[te])
    proba = pipe.predict_proba(X.iloc[te])
    binary = len(classes) == 2
    out = {
        "algo": algo, "algo_name": ALGOS[algo], "params": params, "scaler": scaler, "split": split,
        "classes": classes, "n": len(df), "n_train": len(tr), "n_test": len(te), "numeric": num, "categorical": cat,
        "metrics": {"accuracy": accuracy_score(y[te], pred), "balanced_accuracy": balanced_accuracy_score(y[te], pred),
                    "precision_macro": precision_score(y[te], pred, average="macro", zero_division=0),
                    "recall_macro": recall_score(y[te], pred, average="macro", zero_division=0),
                    "f1_macro": f1_score(y[te], pred, average="macro", zero_division=0)},
        "confusion": confusion_matrix(y[te], pred, labels=range(len(classes))),
        "baseline": float((y[te] == np.bincount(y[tr]).argmax()).mean()),
    }
    try:
        out["metrics"]["roc_auc"] = (roc_auc_score(y[te], proba[:, 1]) if binary
                                     else roc_auc_score(y[te], proba, multi_class="ovr", labels=range(len(classes))))
    except ValueError:
        out["metrics"]["roc_auc"] = None
    if binary and len(set(y[te])) > 1:
        out["roc"] = roc_points(y[te], proba[:, 1])
    sc = cross_val_score(Pipeline([("prep", _prep(num, cat, scaler)), ("clf", _estimator(algo, params))]),
                         X.iloc[tr], y[tr], cv=cv)
    out["cv_acc"], out["cv_std"] = float(sc.mean()), float(sc.std())
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    names = [str(n).replace("num__", "").replace("cat__", "") for n in prep.get_feature_names_out()]
    if algo == "tree":
        out["rules"] = export_text(clf, feature_names=names, max_depth=5)
        out["importances"] = sorted(zip(names, clf.feature_importances_.tolist()), key=lambda t: -t[1])[:15]
    if algo == "rf":
        out["importances"] = sorted(zip(names, clf.feature_importances_.tolist()), key=lambda t: -t[1])[:15]
    if algo == "logreg" and binary:
        out["coef"] = sorted(zip(names, clf.coef_[0].tolist()), key=lambda t: -abs(t[1]))[:20]
    # ---- muatan perhitungan manual untuk satu baris uji
    i = int(np.clip(manual_index, 0, len(te) - 1))
    xq = prep.transform(X.iloc[[te[i]]])[0]
    man = {"index": i, "row_index": int(te[i]), "actual": int(y[te[i]]), "feature_names": names, "x": xq,
           "raw": {c: (str(v) if not isinstance(v, (int, float, np.number)) else float(v)) for c, v in X.iloc[te[i]].items()},
           "proba_sklearn": pipe.predict_proba(X.iloc[[te[i]]])[0]}
    if algo == "logreg" and binary:
        man.update({"kind": "logreg", "coef": clf.coef_[0], "intercept": float(clf.intercept_[0])})
    elif algo == "gnb":
        man.update({"kind": "gnb", "theta": clf.theta_, "var": clf.var_, "prior": clf.class_prior_})
    elif algo == "knn":
        Ttr = prep.transform(X.iloc[tr])
        cap = min(len(tr), 3000)
        man.update({"kind": "knn", "train": Ttr[:cap], "train_y": y[tr][:cap], "k": clf.n_neighbors,
                    "metric": clf.metric, "weights": clf.weights, "complete": cap == len(tr)})
    elif algo == "tree":
        t = clf.tree_
        node, path = 0, []
        while t.children_left[node] != -1:
            f, thr = t.feature[node], t.threshold[node]
            go_left = xq[f] <= thr
            path.append({"node": int(node), "feature": names[f], "threshold": float(thr), "value": float(xq[f]),
                         "go": "kiri (≤)" if go_left else "kanan (>)", "n": int(t.n_node_samples[node])})
            node = t.children_left[node] if go_left else t.children_right[node]
        v = t.value[node][0]
        man.update({"kind": "tree", "path": path, "leaf": int(node), "leaf_dist": (v / v.sum()).tolist(),
                    "leaf_n": int(t.n_node_samples[node])})
    else:
        man["kind"] = "none"
    out["manual"] = man
    return out


def leaderboard(data_key: str, target: str, features: list[str], split: str = "stratified",
                date_col: str | None = None) -> dict:
    rows = []
    for algo, sc in [("logreg", "standard"), ("gnb", "none"), ("knn", "standard"), ("tree", "none"), ("rf", "none")]:
        r = run(data_key, target, features, algo, {"max_depth": 5} if algo == "tree" else {}, sc, split, 0.25, date_col)
        rows.append({"algo": algo, "model": r["algo_name"], **r["metrics"], "cv_acc": r["cv_acc"], "cv_std": r["cv_std"],
                     "baseline": r["baseline"]})
    rows.sort(key=lambda r: -r["accuracy"])
    return {"rows": rows}
