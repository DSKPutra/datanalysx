"""Studi Kasus 04 — Regresi Logistik: arah harga penutupan hari berikutnya dari data OHLC NASDAQ.

Diringkas dari python/original/logistic_regression_nasdaq.py (lampiran laporan studi kasus 04):
pembersihan OHLC, 10 fitur teknikal tanpa look-ahead, split temporal 80:20, StandardScaler
yang di-fit hanya pada data latih, LogisticRegression L2 (C=1, lbfgs, max_iter=2000, rs=42),
MLE tanpa regularisasi via IRLS, VIF, walk-forward 5 lipatan, penalaan C (TimeSeriesSplit),
threshold sweep, kalibrasi, permutation importance, bootstrap CI AUC, dan uji lintas-saham.

Catatan audit (ditampilkan di aplikasi): pada skrip asli, baris terakhir mendapat target 0
karena Close H+1 tidak tersedia (NaN > x bernilai False). Opsi `drop_last_unknown=True`
membuang baris tersebut; default False agar angka identik dengan laporan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, learning_curve
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import RANDOM_STATE
from .common import binary_metrics, cached, get, histogram, parse_dates, roc_points, thin

FEATURE_COLS = ["return_1d", "return_2d", "return_5d", "ma_gap_5_20", "vol_10d", "vol_20d",
                "hl_spread", "oc_spread", "rsi_14", "momentum_10"]


def load_and_clean(df: pd.DataFrame, start: str = "2010-01-01") -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["date"] = parse_dates(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset="date")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[(df["high"] >= df["low"]) & (df["high"] >= df["open"]) & (df["high"] >= df["close"])
            & (df["low"] <= df["open"]) & (df["low"] <= df["close"])]
    df = df[df["date"] >= start].reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame, drop_last_unknown: bool = False) -> pd.DataFrame:
    d = df.copy()
    d["return_1d"] = d["close"].pct_change()
    d["return_2d"] = d["close"].pct_change(2)
    d["return_5d"] = d["close"].pct_change(5)
    d["ma5"] = d["close"].rolling(5).mean()
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma_gap_5_20"] = (d["ma5"] - d["ma20"]) / d["ma20"]
    d["vol_10d"] = d["return_1d"].rolling(10).std()
    d["vol_20d"] = d["return_1d"].rolling(20).std()
    d["hl_spread"] = (d["high"] - d["low"]) / d["close"]
    d["oc_spread"] = (d["close"] - d["open"]) / d["open"]
    delta = d["close"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain, avg_loss = gain.rolling(14).mean(), loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi_14"] = 100 - (100 / (1 + rs))
    d["momentum_10"] = d["close"] - d["close"].shift(10)
    nxt = d["close"].shift(-1)
    d["target"] = (nxt > d["close"]).astype(int)
    if drop_last_unknown:
        d = d[nxt.notna()]
    d = d.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    return d


def features_for(data_key: str, start: str = "2010-01-01", drop_last_unknown: bool = False):
    def build():
        raw = get(data_key)
        clean = load_and_clean(raw, start)
        return clean, engineer_features(clean, drop_last_unknown)
    return cached(f"{data_key}:t04:{start}:{drop_last_unknown}", build)


def time_split(d: pd.DataFrame, test_size: float = 0.2):
    cut = int(len(d) * (1 - test_size))
    tr, te = d.iloc[:cut], d.iloc[cut:]
    return tr, te, tr[FEATURE_COLS].values, tr["target"].values, te[FEATURE_COLS].values, te["target"].values


def _model(penalty: str, C: float):
    # Regularisasi L2 adalah bawaan LogisticRegression; "tanpa regularisasi" = C tak hingga.
    # (Parameter `penalty` tidak dipakai agar kompatibel dengan scikit-learn 1.7–1.9+.)
    c = np.inf if penalty == "none" else C
    return LogisticRegression(C=c, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE)


def train_model(X, y, penalty="l2", C=1.0):
    scaler = StandardScaler().fit(X)          # fit HANYA pada data latih
    model = _model(penalty, C).fit(scaler.transform(X), y)
    return model, scaler


def run_main(data_key: str, penalty: str = "l2", C: float = 1.0, threshold: float = 0.5,
             start: str = "2010-01-01", drop_last_unknown: bool = False) -> dict:
    clean, d = features_for(data_key, start, drop_last_unknown)
    tr, te, Xtr, ytr, Xte, yte = time_split(d)
    model, scaler = train_model(Xtr, ytr, penalty, C)
    proba = model.predict_proba(scaler.transform(Xte))[:, 1]
    pred = (proba >= threshold).astype(int)
    m = binary_metrics(yte, pred, proba)
    thr = np.linspace(0.05, 0.95, 37)
    sweep = [{"threshold": float(t), "precision": precision_score(yte, (proba >= t).astype(int), zero_division=0),
              "recall": recall_score(yte, (proba >= t).astype(int), zero_division=0),
              "f1": f1_score(yte, (proba >= t).astype(int), zero_division=0),
              "pct_pred_up": float((proba >= t).mean())} for t in thr]
    best = max(sweep, key=lambda r: r["f1"])
    frac, mean_pred = calibration_curve(yte, proba, n_bins=10, strategy="quantile")
    s = thin(len(clean), 1500)
    cb = d["target"].value_counts().to_dict()
    return {
        "ticker": str(clean["ticker"].iloc[0]) if "ticker" in clean.columns else None,
        "n_clean": len(clean), "n_total": len(d), "n_train": len(tr), "n_test": len(te),
        "train_period": [str(tr["date"].min().date()), str(tr["date"].max().date())],
        "test_period": [str(te["date"].min().date()), str(te["date"].max().date())],
        "class_balance": {"up": int(cb.get(1, 0)), "down": int(cb.get(0, 0))},
        "penalty": penalty, "C": C, "threshold": threshold,
        "metrics": m, "coef": dict(zip(FEATURE_COLS, model.coef_[0])), "intercept": float(model.intercept_[0]),
        "roc": roc_points(yte, proba), "threshold_sweep": sweep, "f1_best": best,
        "calibration": {"mean_pred": mean_pred, "frac_pos": frac},
        "proba_hist": {"up": histogram(proba[yte == 1], 30), "down": histogram(proba[yte == 0], 30)},
        "series": {"date": clean["date"].dt.strftime("%Y-%m-%d").values[s], "close": clean["close"].values[s]},
        "corr": {"cols": FEATURE_COLS + ["target"], "values": d[FEATURE_COLS + ["target"]].corr().values},
        "last_row_target_unknown": not drop_last_unknown,
    }


def fit_mle_logit(X, y, max_iter: int = 100, tol: float = 1e-8, trace: bool = False):
    """MLE logit tanpa regularisasi, IRLS/Newton-Raphson (identik dengan skrip laporan)."""
    Xd = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xd.shape[1])
    steps = []
    for it in range(max_iter):
        p = 1 / (1 + np.exp(-(Xd @ beta)))
        w = np.clip(p * (1 - p), 1e-8, None)
        H = Xd.T @ (Xd * w[:, None])
        g = Xd.T @ (y - p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta_new = beta + step
        if trace:
            pc = np.clip(p, 1e-12, 1 - 1e-12)
            steps.append({"iter": it + 1, "loglik": float(np.sum(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
                          "max_step": float(np.max(np.abs(step)))})
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    p = np.clip(1 / (1 + np.exp(-(Xd @ beta))), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    w = np.clip(p * (1 - p), 1e-8, None)
    cov = np.linalg.inv(Xd.T @ (Xd * w[:, None]))
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pval = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return {"beta": beta, "se": se, "z": z, "p": pval, "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se, "loglik": ll, "trace": steps}


def vif(X) -> list[float]:
    Xc = np.column_stack([np.ones(len(X)), X])
    out = []
    for j in range(X.shape[1]):
        yj = X[:, j]
        Xo = np.delete(Xc, j + 1, axis=1)
        coef, *_ = np.linalg.lstsq(Xo, yj, rcond=None)
        ss_res = np.sum((yj - Xo @ coef) ** 2)
        ss_tot = np.sum((yj - yj.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        out.append(float(1 / (1 - r2)) if r2 < 0.9999 else float("inf"))
    return out


def inference(data_key: str, start: str = "2010-01-01", drop_last_unknown: bool = False) -> dict:
    _, d = features_for(data_key, start, drop_last_unknown)
    _, _, Xtr, ytr, _, _ = time_split(d)
    scaler = StandardScaler().fit(Xtr)
    Xs = scaler.transform(Xtr)
    mle = fit_mle_logit(Xs, ytr.astype(float), trace=True)
    p0 = np.clip(ytr.mean(), 1e-12, 1 - 1e-12)
    ll0 = float(np.sum(ytr * np.log(p0) + (1 - ytr) * np.log(1 - p0)))
    lr = 2 * (mle["loglik"] - ll0)
    names = ["intercept"] + FEATURE_COLS
    table = [{"feature": n, "coef": mle["beta"][i], "se": mle["se"][i], "z": mle["z"][i], "p_value": mle["p"][i],
              "odds_ratio": float(np.exp(mle["beta"][i])), "or_low": float(np.exp(mle["ci_low"][i])),
              "or_high": float(np.exp(mle["ci_high"][i]))} for i, n in enumerate(names)]
    return {"table": table, "loglik_model": mle["loglik"], "loglik_null": ll0, "lr_stat": float(lr),
            "lr_df": len(FEATURE_COLS), "lr_p": float(1 - stats.chi2.cdf(lr, len(FEATURE_COLS))),
            "mcfadden_r2": float(1 - mle["loglik"] / ll0), "irls_trace": mle["trace"],
            "vif": dict(zip(FEATURE_COLS, vif(Xs)))}


def walk_forward(data_key: str, n_splits: int = 5, start: str = "2010-01-01",
                 drop_last_unknown: bool = False) -> dict:
    _, d = features_for(data_key, start, drop_last_unknown)
    n = len(d)
    fs = n // (n_splits + 1)
    b = [fs * i for i in range(1, n_splits + 2)]
    rows = []
    for i in range(n_splits):
        trd, ted = d.iloc[:b[i]], d.iloc[b[i]:b[i + 1]]
        if trd["target"].nunique() < 2 or ted["target"].nunique() < 2 or len(ted) < 20:
            continue
        model, scaler = train_model(trd[FEATURE_COLS].values, trd["target"].values)
        proba = model.predict_proba(scaler.transform(ted[FEATURE_COLS].values))[:, 1]
        pred = (proba >= 0.5).astype(int)
        yv = ted["target"].values
        rows.append({"fold": i + 1, "n_train": len(trd), "n_test": len(ted),
                     "start": str(ted["date"].min().date()), "end": str(ted["date"].max().date()),
                     "accuracy": float((pred == yv).mean()), "roc_auc": roc_auc_score(yv, proba),
                     "f1": f1_score(yv, pred, zero_division=0)})
    aucs = np.array([r["roc_auc"] for r in rows])
    return {"folds": rows, "mean_auc": float(aucs.mean()), "std_auc": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0}


def tune_C(data_key: str, start: str = "2010-01-01", drop_last_unknown: bool = False) -> dict:
    _, d = features_for(data_key, start, drop_last_unknown)
    _, _, Xtr, ytr, _, _ = time_split(d)
    Xs = StandardScaler().fit_transform(Xtr)
    Cs = np.logspace(-3, 2, 12)
    g = GridSearchCV(LogisticRegression(solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE),
                     param_grid={"C": Cs}, scoring="roc_auc", cv=TimeSeriesSplit(n_splits=5), n_jobs=1).fit(Xs, ytr)
    return {"C": Cs, "mean": g.cv_results_["mean_test_score"], "std": g.cv_results_["std_test_score"],
            "best_C": float(g.best_params_["C"]), "best_auc": float(g.best_score_)}


def bootstrap_auc(data_key: str, n_boot: int = 1000, penalty: str = "l2", C: float = 1.0,
                  start: str = "2010-01-01", drop_last_unknown: bool = False) -> dict:
    _, d = features_for(data_key, start, drop_last_unknown)
    _, _, Xtr, ytr, Xte, yte = time_split(d)
    model, scaler = train_model(Xtr, ytr, penalty, C)
    proba = model.predict_proba(scaler.transform(Xte))[:, 1]
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(yte)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(yte[idx])) < 2:
            continue
        aucs.append(roc_auc_score(yte[idx], proba[idx]))
    aucs = np.array(aucs)
    t, p = stats.ttest_1samp(aucs, 0.5)
    return {"point": roc_auc_score(yte, proba), "lo": float(np.percentile(aucs, 2.5)),
            "hi": float(np.percentile(aucs, 97.5)), "n_boot": len(aucs), "hist": histogram(aucs, 40),
            "t_stat": float(t), "p_value": float(p)}


def permutation(data_key: str, n_repeats: int = 10, start: str = "2010-01-01",
                drop_last_unknown: bool = False) -> dict:
    _, d = features_for(data_key, start, drop_last_unknown)
    _, _, Xtr, ytr, Xte, yte = time_split(d)
    model, scaler = train_model(Xtr, ytr)
    perm = permutation_importance(model, scaler.transform(Xte), yte, n_repeats=n_repeats,
                                  random_state=RANDOM_STATE, scoring="roc_auc")
    return {"features": FEATURE_COLS, "mean": perm.importances_mean, "std": perm.importances_std,
            "coef_std": model.coef_[0]}


def learning(data_key: str, start: str = "2010-01-01", drop_last_unknown: bool = False) -> dict:
    _, d = features_for(data_key, start, drop_last_unknown)
    _, _, Xtr, ytr, _, _ = time_split(d)
    Xs = StandardScaler().fit(Xtr).transform(Xtr)
    sizes, tr_s, te_s = learning_curve(
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE),
        Xs, ytr, train_sizes=np.linspace(0.2, 1.0, 6), cv=TimeSeriesSplit(n_splits=5), scoring="roc_auc", n_jobs=1)
    return {"sizes": sizes, "train": tr_s.mean(1), "val": te_s.mean(1), "val_std": te_s.std(1)}


def cross_stock(data_keys: list[str], start: str = "2010-01-01") -> dict:
    rows = []
    for k in data_keys:
        try:
            clean = load_and_clean(get(k), start)
            if len(clean) < 500:
                rows.append({"key": k, "skipped": "kurang dari 500 baris setelah pembersihan"})
                continue
            d = engineer_features(clean)
            _, _, Xtr, ytr, Xte, yte = time_split(d)
            if len(set(ytr)) < 2 or len(set(yte)) < 2:
                continue
            model, scaler = train_model(Xtr, ytr)
            proba = model.predict_proba(scaler.transform(Xte))[:, 1]
            m = binary_metrics(yte, (proba >= 0.5).astype(int), proba)
            tk = str(clean["ticker"].iloc[0]) if "ticker" in clean.columns else k
            rows.append({"key": k, "ticker": tk, "n_obs": len(d), "accuracy": m["accuracy"],
                         "roc_auc": m["roc_auc"], "f1": m["f1"]})
        except Exception as e:  # berkas tidak valid tidak menghentikan analisis lain
            rows.append({"key": k, "skipped": str(e)})
    ok = [r for r in rows if "roc_auc" in r]
    out = {"rows": rows}
    if len(ok) >= 2:
        a = np.array([r["roc_auc"] for r in ok])
        t, p = stats.ttest_1samp(a, 0.5)
        out["summary"] = {"n": len(a), "mean_auc": float(a.mean()), "std_auc": float(a.std(ddof=1)),
                          "n_above": int((a > 0.5).sum()), "t_stat": float(t), "p_value": float(p)}
        if len(a) >= 5 and np.any(a != 0.5):
            w, pw = stats.wilcoxon(a - 0.5)
            out["summary"].update({"wilcoxon_stat": float(w), "wilcoxon_p": float(pw)})
    return out


def manual_proba(data_key: str, test_index: int = 0, penalty: str = "l2", C: float = 1.0,
                 start: str = "2010-01-01", drop_last_unknown: bool = False) -> dict:
    """Perhitungan manual P(naik) = σ(b₀ + Σ βⱼ·zⱼ) untuk satu hari uji, dengan zⱼ hasil
    standardisasi memakai mean & std data latih; dibandingkan dengan predict_proba()."""
    _, d = features_for(data_key, start, drop_last_unknown)
    tr, te, Xtr, ytr, Xte, yte = time_split(d)
    model, scaler = train_model(Xtr, ytr, penalty, C)
    i = int(np.clip(test_index, 0, len(te) - 1))
    x = Xte[i]
    z = (x - scaler.mean_) / scaler.scale_
    terms = model.coef_[0] * z
    logit = float(model.intercept_[0] + terms.sum())
    p_manual = 1 / (1 + np.exp(-logit))
    p_sk = float(model.predict_proba(scaler.transform(x.reshape(1, -1)))[0, 1])
    rows = [{"feature": f, "x": float(x[j]), "mean": float(scaler.mean_[j]), "std": float(scaler.scale_[j]),
             "z": float(z[j]), "beta": float(model.coef_[0][j]), "term": float(terms[j])}
            for j, f in enumerate(FEATURE_COLS)]
    return {"date": str(te["date"].iloc[i].date()), "test_index": i, "n_test": len(te),
            "actual": int(yte[i]), "rows": rows, "intercept": float(model.intercept_[0]), "logit": logit,
            "p_manual": float(p_manual), "p_sklearn": p_sk, "abs_diff": float(abs(p_manual - p_sk)),
            "pred": int(p_manual >= 0.5)}
