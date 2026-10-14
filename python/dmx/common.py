"""Utilitas bersama: registri data, konversi JSON, metrik, dan penipisan titik grafik."""
from __future__ import annotations

import io
import json
import math
from typing import Any

import numpy as np
import pandas as pd

# Registri DataFrame yang sudah dimuat (kunci -> DataFrame mentah).
DATA: dict[str, pd.DataFrame] = {}
# Sumber daya teks (leksikon, kamus slang) yang dimuat dari JavaScript.
RESOURCES: dict[str, str] = {}
# Cache hasil perantara yang mahal (mis. fitur yang sudah direkayasa).
CACHE: dict[str, Any] = {}


def load_csv(key: str, text: str, **read_kw) -> dict:
    """Parse teks CSV menjadi DataFrame dan simpan di registri."""
    df = pd.read_csv(io.StringIO(text), **read_kw)
    DATA[key] = df
    for k in [k for k in CACHE if k.startswith(key + ":")]:
        del CACHE[k]
    return {"key": key, "n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]),
            "columns": [str(c) for c in df.columns]}


def set_frame(key: str, df: pd.DataFrame) -> None:
    DATA[key] = df
    for k in [k for k in CACHE if k.startswith(key + ":")]:
        del CACHE[k]


def get(key: str) -> pd.DataFrame:
    if key not in DATA:
        raise KeyError(f"Data '{key}' belum dimuat.")
    return DATA[key]


def set_resource(name: str, text: str) -> dict:
    RESOURCES[name] = text
    for k in [k for k in CACHE if k.startswith("res:" + name)]:
        del CACHE[k]
    return {"name": name, "n_chars": len(text)}


def cached(key: str, fn):
    if key not in CACHE:
        CACHE[key] = fn()
    return CACHE[key]


def jsonable(obj: Any) -> Any:
    """Konversi rekursif numpy/pandas -> tipe Python murni; NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return jsonable(obj.tolist())
    return obj


def to_json(obj: Any) -> str:
    return json.dumps(jsonable(obj), ensure_ascii=False)


def thin(n: int, max_points: int = 1500) -> np.ndarray:
    """Indeks penipisan seragam agar grafik tetap ringan (titik pertama & terakhir dipertahankan)."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def regression_metrics(y, p) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mse = float(((y - p) ** 2).mean())
    sst = float(((y - y.mean()) ** 2).sum())
    return {"MAE": float(np.abs(y - p).mean()), "MSE": mse, "RMSE": mse ** 0.5,
            "R2": 1 - float(((y - p) ** 2).sum()) / sst if sst > 0 else float("nan")}


def binary_metrics(y, pred, proba=None) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                                 confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                                 precision_score, recall_score, roc_auc_score)
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if proba is not None and len(set(y.tolist())) > 1:
        proba = np.asarray(proba, float)
        out["roc_auc"] = roc_auc_score(y, proba)
        out["brier"] = brier_score_loss(y, proba)
        out["log_loss"] = log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1])
    return out


def roc_points(y, proba, max_points: int = 400) -> dict:
    from sklearn.metrics import precision_recall_curve, roc_curve
    fpr, tpr, _ = roc_curve(y, proba)
    idx = thin(len(fpr), max_points)
    prec, rec, _ = precision_recall_curve(y, proba)
    idx2 = thin(len(prec), max_points)
    return {"fpr": fpr[idx], "tpr": tpr[idx], "precision": prec[idx2], "recall": rec[idx2]}


def histogram(values, bins: int = 30) -> dict:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(v, bins=bins)
    return {"edges": edges, "counts": counts}


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse tanggal: coba ISO dulu, lalu format hari/bulan/tahun (dataset NVIDIA)."""
    if s.dtype.kind == "M":
        return s
    txt = s.astype(str)
    sample = txt.dropna().head(50)
    if sample.str.match(r"^\d{1,2}/\d{1,2}/\d{4}").all():
        return pd.to_datetime(txt, format="%d/%m/%Y", errors="coerce")
    out = pd.to_datetime(txt, errors="coerce", utc=False, format="mixed")
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_convert(None)
    except (AttributeError, TypeError):
        pass
    return out
