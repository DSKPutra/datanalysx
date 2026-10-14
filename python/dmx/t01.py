"""Tugas 01 — Regresi Linear: prediksi harga Close saham NVIDIA (1999–2026).

Diringkas dari lampiran kode laporan Tugas 01 (python/original/tugas01_regresi_nvidia.py).
Mode "same_day" mereplikasi laporan (fitur Open/High/Low hari yang sama, split acak 80:20,
random_state=42). Mode "next_day" adalah eksperimen tambahan aplikasi: memprediksi Close
hari berikutnya hanya dari informasi hari ini (peramalan sesungguhnya, tanpa look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from . import RANDOM_STATE
from .common import cached, get, parse_dates, regression_metrics, thin

FEATURES = ["open", "high", "low", "volume", "sma_50", "rsi_14"]
TARGET = "close"


def prepare(data_key: str) -> pd.DataFrame:
    def build():
        df = get(data_key).copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.drop_duplicates()
        df["date"] = parse_dates(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    return cached(f"{data_key}:t01", build)


LAG_COLS = ["open", "high", "low", "close"]
N_LAGS = 5


def lag_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """OHLC hari t, t−1, …, t−4 (informasi yang sudah diketahui saat penutupan hari t)
    untuk memprediksi Close hari t+1."""
    d = df.copy()
    names = []
    for lag in range(N_LAGS):
        for c in LAG_COLS:
            n = f"{c}_lag{lag}"
            d[n] = d[c].shift(lag)
            names.append(n)
    return d, names


def _design(df: pd.DataFrame, features: list[str], mode: str):
    if mode in ("next_day", "lag"):
        d, names = lag_features(df) if mode == "lag" else (df.copy(), features)
        d["y"] = d[TARGET].shift(-1)          # Close hari t+1
        d = d.dropna(subset=names + ["y"]).reset_index(drop=True)
        return d, d[names].values.astype(float), d["y"].values.astype(float)
    d = df.dropna(subset=features + [TARGET]).reset_index(drop=True)
    return d, d[features].values.astype(float), d[TARGET].values.astype(float)


def _split(d, X, y, split: str, test_size: float):
    idx = np.arange(len(y))
    if split == "temporal":
        cut = int(len(y) * (1 - test_size))
        return idx[:cut], idx[cut:]
    tr, te = train_test_split(idx, test_size=test_size, random_state=RANDOM_STATE)
    return tr, te


def run(data_key: str, features: list[str] | None = None, split: str = "random",
        mode: str = "same_day", test_size: float = 0.2) -> dict:
    features = features or FEATURES
    df = prepare(data_key)
    d, X, y = _design(df, features, mode)
    if mode == "lag":
        features = lag_features(df)[1]
    tr, te = _split(d, X, y, split, test_size)
    model = LinearRegression().fit(X[tr], y[tr])
    pred = model.predict(X[te])
    out = {
        "n_rows": len(df), "n_train": len(tr), "n_test": len(te),
        "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date()),
        "features": features, "mode": mode, "split": split,
        "metrics": regression_metrics(y[te], pred),
        "coef": dict(zip(features, model.coef_)), "intercept": float(model.intercept_),
    }
    # Baseline naif (persistence): Close besok = Close hari ini. Hanya bermakna untuk next_day.
    if mode in ("next_day", "lag"):
        base = d[TARGET].values[te]                  # Close hari t
        out["baseline_persistence"] = regression_metrics(y[te], base)
        dy, dp = y[te] - base, pred - base           # perubahan aktual vs perubahan yang diprediksi
        out["change_metrics"] = {
            "r2_change": float(1 - ((dy - dp) ** 2).sum() / ((dy - dy.mean()) ** 2).sum()),
            "skill_vs_persistence": float(1 - ((y[te] - pred) ** 2).mean() / ((y[te] - base) ** 2).mean()),
            "direction_accuracy": float((np.sign(dp) == np.sign(dy)).mean()),
        }
    order = np.argsort(d["date"].values[te])
    te_sorted = te[order]
    k = thin(len(te_sorted), 1200)
    out["scatter"] = {"actual": y[te_sorted][k], "pred": model.predict(X[te_sorted][k]),
                      "date": d["date"].dt.strftime("%Y-%m-%d").values[te_sorted][k]}
    corr = df[[c for c in FEATURES if c in df.columns] + [TARGET]].corr()
    out["corr"] = {"cols": list(corr.columns), "values": corr.values}
    s = thin(len(df), 1500)
    out["series"] = {"date": df["date"].dt.strftime("%Y-%m-%d").values[s], "close": df[TARGET].values[s],
                     "era": df["era"].values[s] if "era" in df.columns else None}
    if "era" in df.columns:
        g = df.groupby("era", sort=False)["date"].agg(["min", "max"])
        out["eras"] = [{"era": e, "start": str(r["min"].date()), "end": str(r["max"].date())} for e, r in g.iterrows()]
    if "key_event" in df.columns:
        ev = df.dropna(subset=["key_event"])
        ev = ev[ev["key_event"].ne(ev["key_event"].shift())]   # peristiwa berurutan yang sama digabung
        out["events"] = {"date": ev["date"].dt.strftime("%Y-%m-%d").values, "close": ev[TARGET].values,
                         "text": ev["key_event"].astype(str).values}
    if "stock_split" in df.columns:
        sp = df.dropna(subset=["stock_split"])
        out["splits"] = {"date": sp["date"].dt.strftime("%Y-%m-%d").values, "ratio": sp["stock_split"].values}
    out["quality"] = {"shape": [int(get(data_key).shape[0]), int(get(data_key).shape[1])],
                      "duplicates": int(get(data_key).duplicated().sum()),
                      "missing": get(data_key).isna().sum().to_dict()}
    return out


def manual_ols(data_key: str, features: list[str] | None = None, n_rows: int = 20) -> dict:
    """Perhitungan manual OLS b = (XᵀX)⁻¹Xᵀy pada subset kecil data (n_rows hari pertama),
    dibandingkan dengan sklearn LinearRegression pada subset yang sama.
    Volume diukur dalam juta lembar agar matriks XᵀX tidak sangat buruk kondisinya."""
    features = features or FEATURES
    df = prepare(data_key).dropna(subset=features + [TARGET]).head(n_rows)
    X = df[features].values.astype(float).copy()
    if "volume" in features:
        X[:, features.index("volume")] /= 1e6
    y = df[TARGET].values.astype(float)
    Xd = np.column_stack([np.ones(len(y)), X])
    XtX = Xd.T @ Xd
    Xty = Xd.T @ y
    beta = np.linalg.solve(XtX, Xty)
    sk = LinearRegression().fit(X, y)
    bsk = np.r_[sk.intercept_, sk.coef_]
    pred_m, pred_s = Xd @ beta, sk.predict(X)
    return {
        "features": ["intercept"] + [f + (" (juta)" if f == "volume" else "") for f in features],
        "n_rows": len(y), "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "X": Xd, "y": y, "XtX": XtX, "Xty": Xty, "cond_XtX": float(np.linalg.cond(XtX)),
        "beta_manual": beta, "beta_sklearn": bsk,
        "max_abs_diff_beta": float(np.max(np.abs(beta - bsk))),
        "pred_manual": pred_m, "pred_sklearn": pred_s,
        "max_abs_diff_pred": float(np.max(np.abs(pred_m - pred_s))),
    }
