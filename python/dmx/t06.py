"""Tugas 06 — Gaussian Naïve Bayes: arah IHSG hari berikutnya dari data makro multi-aset.

Diringkas dari python/original/naive_bayes_ihsg.py (tugas kelompok 4):
interpolasi linear harga aset, return harian (%), VIX_chg, rate_diff = BI_rate − US_rate,
CPI_chg, winsorization IQR 3× pada lima fitur return, target = IHSG_ret(t+1) > 0,
split temporal 80:20, GaussianNB tanpa scaling, TimeSeriesSplit 5 lipatan.

Mode outlier:
  "report"  — batas IQR dihitung dari SELURUH periode (persis skrip laporan);
  "train"   — batas IQR dihitung hanya dari data latih (tanpa kebocoran informasi masa depan);
  "none"    — tanpa penanganan outlier;
  "report_none" — tanpa penanganan outlier dengan baris data persis seperti laporan
                  (dipakai untuk mereproduksi angka "sebelum winsorization" 48,48%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB

from .common import binary_metrics, cached, get, histogram, roc_points

FEATURES = ["IHSG_ret", "SP500_ret", "OIL_ret", "GOLD_ret", "USDIDR_ret", "VIX", "VIX_chg", "rate_diff", "CPI_chg"]
RET_COLS = ["OIL_ret", "GOLD_ret", "USDIDR_ret", "SP500_ret", "IHSG_ret"]
PRICE_COLS = ["OIL", "GOLD", "USDIDR", "SP500", "IHSG", "VIX"]


def _base(data_key: str) -> pd.DataFrame:
    def build():
        df = get(data_key).copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df[PRICE_COLS] = df[PRICE_COLS].interpolate(method="linear", limit_direction="both")
        for c in ["OIL", "GOLD", "USDIDR", "SP500", "IHSG"]:
            df[f"{c}_ret"] = df[c].pct_change() * 100
        df["VIX_chg"] = df["VIX"].diff()
        df["rate_diff"] = df["BI_rate"] - df["US_rate"]
        df["CPI_chg"] = df["CPI"].diff()
        df["raw_USDIDR_ret"] = df["USDIDR_ret"]
        return df
    return cached(f"{data_key}:t06:base", build)


def build_model_frame(data_key: str, outlier_mode: str = "train"):
    def build():
        df = _base(data_key).copy()
        df["IHSG_ret_next"] = df["IHSG_ret"].shift(-1)
        if outlier_mode in ("report", "report_none"):
            # Persis skrip laporan: baris terakhir (IHSG besok belum diketahui) ikut dengan target 0.
            m = df.dropna(subset=FEATURES).reset_index(drop=True)
        else:
            m = df.dropna(subset=FEATURES + ["IHSG_ret_next"]).reset_index(drop=True)
        cut = int(len(m) * 0.8)
        bounds = {}
        for c in RET_COLS:
            if outlier_mode in ("none", "report_none"):
                continue
            src = df[c].dropna() if outlier_mode == "report" else m[c].iloc[:cut]
            q1, q3 = src.quantile(0.25), src.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            bounds[c] = {"lo": float(lo), "hi": float(hi), "n_capped": int(((m[c] < lo) | (m[c] > hi)).sum())}
            m[c] = m[c].clip(lower=lo, upper=hi)
        m["target"] = (m["IHSG_ret_next"] > 0).astype(int)   # tanda tidak berubah oleh winsorization
        return m, bounds
    return cached(f"{data_key}:t06:{outlier_mode}", build)


def _split(m: pd.DataFrame):
    cut = int(len(m) * 0.8)
    return m.iloc[:cut], m.iloc[cut:]


def run(data_key: str, outlier_mode: str = "train") -> dict:
    m, bounds = build_model_frame(data_key, outlier_mode)
    tr, te = _split(m)
    g = GaussianNB().fit(tr[FEATURES].values, tr["target"].values)
    yte = te["target"].values
    proba = g.predict_proba(te[FEATURES].values)[:, 1]
    pred = g.predict(te[FEATURES].values)
    met = binary_metrics(yte, pred, proba)
    met["average_precision"] = average_precision_score(yte, proba)
    majority = int(pd.Series(tr["target"]).mode()[0])
    frac, mp = calibration_curve(yte, proba, n_bins=10)
    # TimeSeriesSplit 5 lipatan
    X, y = m[FEATURES].values, m["target"].values
    folds = []
    for k, (a, b) in enumerate(TimeSeriesSplit(n_splits=5).split(X), start=1):
        gm = GaussianNB().fit(X[a], y[a])
        pp = gm.predict_proba(X[b])[:, 1]
        pr = gm.predict(X[b])
        folds.append({"fold": k, "n_train": len(a), "n_val": len(b), "accuracy": float((pr == y[b]).mean()),
                      "roc_auc": roc_auc_score(y[b], pp), "f1": f1_score(y[b], pr, zero_division=0),
                      "start": str(m["Date"].iloc[b[0]].date()), "end": str(m["Date"].iloc[b[-1]].date())})
    base = _base(data_key)
    feat_curves = {}
    for i, f in enumerate(FEATURES):
        lo = min(g.theta_[:, i] - 4 * np.sqrt(g.var_[:, i]))
        hi = max(g.theta_[:, i] + 4 * np.sqrt(g.var_[:, i]))
        xs = np.linspace(lo, hi, 120)
        feat_curves[f] = {"x": xs, "pdf0": _pdf(xs, g.theta_[0, i], g.var_[0, i]),
                          "pdf1": _pdf(xs, g.theta_[1, i], g.var_[1, i])}
    return {
        "outlier_mode": outlier_mode, "bounds": bounds, "n_model": len(m), "n_train": len(tr), "n_test": len(te),
        "train_period": [str(tr["Date"].min().date()), str(tr["Date"].max().date())],
        "test_period": [str(te["Date"].min().date()), str(te["Date"].max().date())],
        "class_balance": m["target"].value_counts(normalize=True).to_dict(),
        "metrics": met, "baseline_majority": float((yte == majority).mean()),
        "roc": roc_points(yte, proba), "calibration": {"mean_pred": mp, "frac_pos": frac},
        "cv": folds, "cv_auc_mean": float(np.mean([f["roc_auc"] for f in folds])),
        "cv_auc_std": float(np.std([f["roc_auc"] for f in folds], ddof=1)),
        "cv_acc_mean": float(np.mean([f["accuracy"] for f in folds])),
        "prior": g.class_prior_, "theta": g.theta_, "sigma": np.sqrt(g.var_), "features": FEATURES,
        "likelihood_curves": feat_curves,
        "usdidr_raw_hist": histogram(base["raw_USDIDR_ret"].dropna(), 60),
        "usdidr_clean_hist": histogram(m["USDIDR_ret"], 60),
        "usdidr_raw_extremes": base.loc[base["raw_USDIDR_ret"].abs() > 20, ["Date", "raw_USDIDR_ret"]]
        .assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "corr": {"cols": FEATURES + ["target"], "values": m[FEATURES + ["target"]].corr().values},
        "test_dates": te["Date"].dt.strftime("%Y-%m-%d").tolist(),
    }


def _pdf(x, mu, var):
    return 1.0 / np.sqrt(2 * np.pi * var) * np.exp(-((x - mu) ** 2) / (2 * var))


def model_params(data_key: str, date: str = "2023-02-21", outlier_mode: str = "report") -> dict:
    """Parameter model + vektor fitur satu hari uji untuk perhitungan manual (di TypeScript
    dan di Python), beserta predict_proba() sebagai pembanding."""
    m, _ = build_model_frame(data_key, outlier_mode)
    tr, te = _split(m)
    g = GaussianNB().fit(tr[FEATURES].values, tr["target"].values)
    hit = te.index[te["Date"].dt.strftime("%Y-%m-%d") == date]
    i = int(hit[0]) if len(hit) else int(te.index[0])
    row = m.loc[i]
    x = row[FEATURES].values.astype(float)
    return {"date": str(row["Date"].date()), "x": x, "features": FEATURES, "prior": g.class_prior_,
            "theta": g.theta_, "var": g.var_, "epsilon": float(g.epsilon_),
            "proba_sklearn": g.predict_proba(x.reshape(1, -1))[0], "actual": int(row["target"]),
            "ihsg_ret_next": float(row["IHSG_ret_next"]), "outlier_mode": outlier_mode}


def manual_posterior(params: dict) -> dict:
    """Prior → likelihood Gaussian per fitur → log-sum-exp → posterior → MAP."""
    x = np.asarray(params["x"], float)
    th = np.asarray(params["theta"], float)
    var = np.asarray(params["var"], float)
    logp = np.log(np.asarray(params["prior"], float))
    ll = np.array([np.sum(-0.5 * np.log(2 * np.pi * var[c]) - (x - th[c]) ** 2 / (2 * var[c])) for c in range(2)])
    joint = logp + ll
    mx = joint.max()
    post = np.exp(joint - mx) / np.exp(joint - mx).sum()
    return {"log_prior": logp, "sum_loglik": ll, "log_joint": joint, "posterior": post, "pred": int(np.argmax(post))}
