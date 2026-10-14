"""Tugas 05 — Decision Tree: kategori topik pada Trending Topics 2026 (sintetis).

Diringkas dari Lampiran A laporan Tugas 05 (python/original/tugas05_decision_tree.py).
Skenario 1 = metadata (one-hot source/language/country/sentiment + engagement, trend, day,
weekday); Skenario 2 = bag-of-words biner dari headline; Skenario 3 = gabungan.
topic_subcategory dan short_text dibuang karena membocorkan label. Split berstrata 80:20,
StratifiedKFold(10/5, shuffle, random_state=42).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from . import RANDOM_STATE
from .common import cached, get

CATS = ["ai_and_tech", "economy", "entertainment", "global_events", "sports"]
CAT_LABEL = {"ai_and_tech": "AI & Tech", "economy": "Economy", "entertainment": "Entertainment",
             "global_events": "Global Events", "sports": "Sports"}


def prepare(data_key: str):
    def build():
        df = get(data_key).copy()
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = df["date"].dt.day
        df["weekday"] = df["date"].dt.weekday
        y = df["topic_category"].map({c: i for i, c in enumerate(CATS)}).values
        XA = pd.get_dummies(df[["source", "language", "country", "sentiment"]]).astype(int)
        for c in ["engagement_score", "trend_score", "day", "weekday"]:
            XA[c] = df[c]
        cv = CountVectorizer(binary=True, lowercase=True, token_pattern=r"(?u)\b[a-zA-Z0-9]{2,}\b")
        XB = pd.DataFrame(cv.fit_transform(df["headline"]).toarray(), columns=cv.get_feature_names_out())
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        return {"df": df, "y": y, "XA": XA, "XB": XB, "tr": tr, "te": te, "cv": cv}
    return cached(f"{data_key}:t05", build)


def _X(p, scenario: int) -> pd.DataFrame:
    if scenario == 1:
        return p["XA"]
    if scenario == 2:
        return p["XB"]
    return pd.concat([p["XA"], p["XB"].add_prefix("w_")], axis=1)


def entropy(counts) -> float:
    q = np.asarray(counts, float)
    q = q[q > 0] / q.sum()
    return float(-(q * np.log2(q)).sum())


def gini(counts) -> float:
    q = np.asarray(counts, float) / np.sum(counts)
    return float(1 - (q ** 2).sum())


def overview(data_key: str) -> dict:
    p = prepare(data_key)
    df = p["df"]
    leak = {"topic_subcategory_nested": bool(df.groupby("topic_subcategory")["topic_category"].nunique().max() == 1),
            "short_text_mentions_category": float(df["short_text"].str.contains("broader category").mean())}
    return {
        "shape": list(get(data_key).shape), "missing": int(get(data_key).isna().sum().sum()),
        "duplicates": int(df.duplicated(subset=[c for c in df.columns if c not in ["id", "day", "weekday"]]).sum()),
        "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date()),
        "class_counts": {CAT_LABEL[c]: int((df["topic_category"] == c).sum()) for c in CATS},
        "n_features": {"s1": p["XA"].shape[1], "s2": p["XB"].shape[1]},
        "headline_templates": df.groupby("topic_category")["headline"].nunique().rename(index=CAT_LABEL).to_dict(),
        "leakage": leak, "n_train": len(p["tr"]), "n_test": len(p["te"]),
    }


def fit_tree(data_key: str, scenario: int = 1, criterion: str = "gini", max_depth: int | None = 3,
             min_samples_leaf: int = 25, min_samples_split: int = 2, cv_folds: int = 10) -> dict:
    p = prepare(data_key)
    X, y, tr, te = _X(p, scenario), p["y"], p["tr"], p["te"]
    m = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                               min_samples_split=min_samples_split, random_state=RANDOM_STATE)
    m.fit(X.iloc[tr], y[tr])
    pred = m.predict(X.iloc[te])
    prec, rec, f1, _ = precision_recall_fscore_support(y[te], pred, labels=range(5), zero_division=0)
    skf = StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cvr = cross_validate(DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                                min_samples_leaf=min_samples_leaf,
                                                min_samples_split=min_samples_split, random_state=RANDOM_STATE),
                         X, y, cv=skf, scoring=["accuracy", "roc_auc_ovr"])
    imp = sorted(zip(X.columns, m.feature_importances_), key=lambda t: -t[1])[:12]
    t = m.tree_
    nodes = [{"id": i, "left": int(t.children_left[i]), "right": int(t.children_right[i]),
              "feature": (str(X.columns[t.feature[i]]) if t.children_left[i] != -1 else None),
              "threshold": float(t.threshold[i]), "n": int(t.n_node_samples[i]),
              "impurity": float(t.impurity[i]), "value": (t.value[i][0] / t.value[i][0].sum()).tolist()}
             for i in range(min(t.node_count, 127))]
    return {
        "scenario": scenario, "params": {"criterion": criterion, "max_depth": max_depth,
                                         "min_samples_leaf": min_samples_leaf, "min_samples_split": min_samples_split},
        "train_acc": accuracy_score(y[tr], m.predict(X.iloc[tr])), "test_acc": accuracy_score(y[te], pred),
        "majority_baseline": float(np.bincount(y[te]).max() / len(te)),
        "cv_acc_mean": float(cvr["test_accuracy"].mean()), "cv_acc_std": float(cvr["test_accuracy"].std()),
        "cv_auc_mean": float(cvr["test_roc_auc_ovr"].mean()), "cv_fold_acc": cvr["test_accuracy"],
        "n_nodes": int(t.node_count), "n_leaves": int(m.get_n_leaves()), "depth": int(m.get_depth()),
        "confusion": confusion_matrix(y[te], pred, labels=range(5)), "labels": [CAT_LABEL[c] for c in CATS],
        "per_class": {"precision": prec, "recall": rec, "f1": f1},
        "importances": [{"feature": f, "importance": float(v)} for f, v in imp if v > 0],
        "rules": export_text(m, feature_names=[str(c) for c in X.columns],
                             class_names=[CAT_LABEL[c] for c in CATS], max_depth=6),
        "nodes": nodes,
    }


def depth_curve(data_key: str, scenario: int = 1, criterion: str = "gini") -> dict:
    p = prepare(data_key)
    X, y, tr = _X(p, scenario), p["y"], p["tr"]
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    rows = []
    for d in depths:
        m = DecisionTreeClassifier(criterion=criterion, max_depth=d, random_state=RANDOM_STATE)
        r = cross_validate(m, X.iloc[tr], y[tr], cv=skf, scoring="accuracy", return_train_score=True)
        rows.append({"depth": d, "train": float(r["train_score"].mean()), "val": float(r["test_score"].mean())})
    return {"rows": rows, "majority": float(np.bincount(y).max() / len(y))}


def permutation_test(data_key: str, scenario: int = 1, n_perm: int = 50) -> dict:
    """Uji permutasi label (Ojala & Garriga, 2010) seperti laporan: pohon max_depth=5,
    min_samples_leaf=25, StratifiedKFold(5); p = (#acak ≥ asli + 1)/(n + 1)."""
    p = prepare(data_key)
    X, y = _X(p, scenario), p["y"]
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    m = DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=RANDOM_STATE)
    obs = float(cross_val_score(m, X, y, cv=skf).mean())
    rng = np.random.RandomState(RANDOM_STATE)
    null = np.array([cross_val_score(m, X, rng.permutation(y), cv=skf).mean() for _ in range(n_perm)])
    return {"observed": obs, "null": null, "null_mean": float(null.mean()), "null_std": float(null.std()),
            "p_value": float((np.sum(null >= obs) + 1) / (n_perm + 1)), "n_perm": n_perm}


def manual_split(data_key: str, word: str = "for") -> dict:
    """Entropy simpul akar dan information gain untuk pemisahan biner berdasarkan
    keberadaan satu kata pada judul; diverifikasi dengan impuritas pohon scikit-learn
    (criterion='entropy', max_depth=1) yang memakai logaritma basis 2."""
    p = prepare(data_key)
    y = p["y"]
    XB = p["XB"]
    if word not in XB.columns:
        raise ValueError(f"Kata '{word}' tidak ada dalam kosakata judul.")
    has = XB[word].values == 1
    root = np.bincount(y, minlength=5)
    left, right = np.bincount(y[has], minlength=5), np.bincount(y[~has], minlength=5)
    H, Hl, Hr = entropy(root), entropy(left), entropy(right)
    wl, wr = has.mean(), 1 - has.mean()
    Hafter = wl * Hl + wr * Hr
    si = -(wl * math.log2(wl) + wr * math.log2(wr)) if 0 < wl < 1 else 0.0
    sk = DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=RANDOM_STATE)
    sk.fit(XB[[word]].values, y)
    t = sk.tree_
    sk_after = (t.n_node_samples[1] * t.impurity[1] + t.n_node_samples[2] * t.impurity[2]) / t.n_node_samples[0]
    ig = H - Hafter
    return {
        "word": word, "labels": [CAT_LABEL[c] for c in CATS], "n": int(len(y)),
        "root_counts": root, "left_counts": left, "right_counts": right,
        "H_root": H, "H_left": Hl, "H_right": Hr, "w_left": float(wl), "w_right": float(wr),
        "H_after": float(Hafter), "info_gain": float(ig), "split_info": float(si),
        "gain_ratio": float(ig / si) if si > 0 else 0.0,
        "gini_root": gini(root), "gini_after": float(wl * gini(left) + wr * gini(right)),
        "sklearn_H_root": float(t.impurity[0]), "sklearn_H_after": float(sk_after),
        "max_abs_diff": float(max(abs(H - t.impurity[0]), abs(Hafter - sk_after))),
        "vocab_top": sorted(XB.columns, key=lambda w: -XB[w].sum())[:40],
    }


def predict_headlines(data_key: str, headlines: list[str], max_depth: int = 3) -> dict:
    """Uji ketangguhan: pohon Skenario 2 (entropy, kedalaman 3) menebak judul baru."""
    p = prepare(data_key)
    m = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=RANDOM_STATE)
    m.fit(p["XB"].iloc[p["tr"]], p["y"][p["tr"]])
    Xn = pd.DataFrame(p["cv"].transform(headlines).toarray(), columns=p["XB"].columns)
    proba = m.predict_proba(Xn)
    return {"rows": [{"headline": h, "pred": CAT_LABEL[CATS[int(np.argmax(pr))]], "confidence": float(pr.max()),
                      "known_words": [w for w in p["cv"].build_analyzer()(h) if w in p["XB"].columns]}
                     for h, pr in zip(headlines, proba)]}
