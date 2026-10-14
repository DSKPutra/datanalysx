"""Tugas 03 — Regresi Linear Sederhana & Berganda: R&D, capex, margin vs pendapatan
perusahaan semikonduktor global (2010–2026).

Diringkas dari Lampiran 1 laporan Tugas 03 (python/original/tugas03_regresi_semikonduktor.py):
kelas OLS dari nol (koefisien, uji t, uji F, R², VIF, Breusch–Pagan, Durbin–Watson),
hold-out 80:20 (random_state=42), dan validasi silang 10-lipatan KFold(shuffle, 42).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .common import cached, get, regression_metrics

R, RD, CX, OM = "revenue_usd_bn", "rd_spend_usd_bn", "capex_usd_bn", "operating_margin_pct"


class OLS:
    """OLS dengan aljabar matriks: b = (XᵀX)⁻¹Xᵀy (sama seperti laporan)."""

    def __init__(self, X, y, names):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.X = np.column_stack([np.ones(len(y)), X])
        self.y = y
        self.names = ["const"] + list(names)
        n, k = self.X.shape
        self.n, self.k = n, k
        self.XtXi = np.linalg.inv(self.X.T @ self.X)
        self.b = self.XtXi @ self.X.T @ y
        self.fit = self.X @ self.b
        self.res = y - self.fit
        self.sse = float(self.res @ self.res)
        self.sst = float(((y - y.mean()) ** 2).sum())
        self.ssr = self.sst - self.sse
        self.df = n - k
        self.s2 = self.sse / self.df
        self.se = np.sqrt(np.diag(self.XtXi) * self.s2)
        self.t = self.b / self.se
        self.p = 2 * stats.t.sf(np.abs(self.t), self.df)
        self.r2 = 1 - self.sse / self.sst
        self.adj = 1 - (1 - self.r2) * (n - 1) / self.df
        self.F = (self.ssr / (k - 1)) / self.s2
        self.Fp = float(stats.f.sf(self.F, k - 1, self.df))
        tc = stats.t.ppf(0.975, self.df)
        self.ci = np.c_[self.b - tc * self.se, self.b + tc * self.se]

    def predict(self, X):
        X = np.asarray(X, float)
        return np.column_stack([np.ones(len(X)), X]) @ self.b

    def dw(self):
        return float((np.diff(self.res) ** 2).sum() / self.sse)

    def bp(self):
        e2 = self.res ** 2
        g = np.linalg.lstsq(self.X, e2, rcond=None)[0]
        r = 1 - ((e2 - self.X @ g) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum()
        lm = self.n * r
        return float(lm), float(stats.chi2.sf(lm, self.k - 1))

    def vif(self):
        out = []
        for j in range(1, self.k):
            o = [i for i in range(self.k) if i != j]
            Z = self.X[:, o]
            g = np.linalg.lstsq(Z, self.X[:, j], rcond=None)[0]
            r = self.X[:, j] - Z @ g
            r2 = 1 - (r @ r) / ((self.X[:, j] - self.X[:, j].mean()) ** 2).sum()
            out.append(float(1 / (1 - r2)))
        return out

    def summary(self) -> dict:
        sh = stats.shapiro(self.res)
        lm, lmp = self.bp()
        return {
            "names": self.names, "coef": self.b, "se": self.se, "t": self.t, "p": self.p,
            "ci_low": self.ci[:, 0], "ci_high": self.ci[:, 1],
            "r2": self.r2, "adj_r2": self.adj, "multiple_r": float(np.sqrt(self.r2)),
            "F": float(self.F), "F_p": self.Fp, "df_model": self.k - 1, "df_resid": self.df,
            "sse": self.sse, "ssr": self.ssr, "sst": self.sst, "s": float(np.sqrt(self.s2)),
            "dw": self.dw(), "bp_lm": lm, "bp_p": lmp, "shapiro_w": float(sh.statistic),
            "shapiro_p": float(sh.pvalue), "vif": self.vif() if self.k > 2 else None, "n": self.n,
        }


def prepare(data_key: str) -> pd.DataFrame:
    def build():
        raw = get(data_key)
        c = raw[(raw[R] > 0) & (raw[RD] > 0) & (raw[CX] > 0)].copy().reset_index(drop=True)
        c["lR"], c["lRD"], c["lCX"] = np.log(c[R]), np.log(c[RD]), np.log(c[CX])
        return c
    return cached(f"{data_key}:t03", build)


def fit_models(data_key: str) -> dict:
    raw = get(data_key)
    c = prepare(data_key)
    m1 = OLS(c[[RD]], c[R], ["R&D"])
    m1l = OLS(c[["lRD"]], c["lR"], ["ln R&D"])
    m2 = OLS(c[[RD, CX, OM]], c[R], ["R&D", "Capex", "Margin"])
    m2l = OLS(c[["lRD", "lCX", OM]], c["lR"], ["ln R&D", "ln Capex", "Margin"])
    g = c.groupby("company_name")
    xd = c["lRD"] - g["lRD"].transform("mean")
    yd = c["lR"] - g["lR"].transform("mean")
    fe_b = float((xd @ yd) / (xd @ xd))
    r = yd - fe_b * xd
    fe_df = len(c) - c["company_name"].nunique() - 1
    s = c.sample(n=min(589, len(c)), random_state=0) if len(c) > 2000 else c
    return {
        "n_raw": len(raw), "n_clean": len(c), "n_dropped_zero": len(raw) - len(c),
        "n_companies": int(c["company_name"].nunique()),
        "describe": c[[R, RD, CX, OM]].describe().T[["mean", "std", "min", "50%", "max"]].to_dict(orient="index"),
        "skew": c[[R, RD, CX, OM]].skew().to_dict(),
        "corr_pearson": c[[R, RD, CX, OM]].corr().values, "corr_log": c[["lR", "lRD", "lCX", OM]].corr().values,
        "simple": m1.summary(), "simple_log": m1l.summary(),
        "multiple": m2.summary(), "multiple_log": m2l.summary(),
        "fixed_effect": {"slope": fe_b, "se": float(np.sqrt((r @ r) / fe_df / (xd @ xd))),
                         "r2_within": float(1 - (r @ r) / (yd @ yd))},
        "points": {"rd": s[RD].values, "capex": s[CX].values, "revenue": s[R].values,
                   "margin": s[OM].values, "segment": s["segment"].values,
                   "company": s["company_name"].values, "year": s["year"].values},
    }


def holdout(data_key: str) -> dict:
    c = prepare(data_key)
    y = c[R].values
    specs = {
        "Sederhana (linear)": (c[[RD]].values, False),
        "Sederhana (log-log)": (c[["lRD"]].values, True),
        "Berganda (linear)": (c[[RD, CX, OM]].values, False),
        "Berganda (log-log)": (c[["lRD", "lCX", OM]].values, True),
    }
    rows = []
    for name, (X, lg) in specs.items():
        a, b, ya, yb = train_test_split(X, y, test_size=0.2, random_state=42)
        m = OLS(a, np.log(ya) if lg else ya, range(a.shape[1]))
        p = m.predict(b)
        p = np.exp(p) if lg else p
        rows.append({"model": name, "n_train": len(a), "n_test": len(b), **regression_metrics(yb, p)})
    return {"rows": rows}


def _models():
    def tts(m):
        return TransformedTargetRegressor(regressor=m, transformer=StandardScaler())
    return {
        "Constant (rata-rata)": (DummyRegressor(), "s", False),
        "Linear – sederhana": (LinearRegression(), "s", False),
        "Linear – berganda": (LinearRegression(), "m", False),
        "Linear – berganda (log)": (LinearRegression(), "ml", True),
        "kNN (k=5)": (make_pipeline(StandardScaler(), KNeighborsRegressor(5)), "m", False),
        "SGD": (tts(make_pipeline(StandardScaler(), SGDRegressor(max_iter=5000, tol=1e-6, random_state=0))), "m", False),
        "SVM (RBF)": (tts(make_pipeline(StandardScaler(), SVR(C=3.0))), "m", False),
        "Random Forest": (RandomForestRegressor(200, random_state=0), "m", False),
    }


def cv_compare(data_key: str, models: list[str] | None = None, n_splits: int = 10) -> dict:
    c = prepare(data_key)
    y = c[R].values
    Xs = {"s": c[[RD]].values, "m": c[[RD, CX, OM]].values, "ml": c[["lRD", "lCX", OM]].values}
    kf = KFold(n_splits, shuffle=True, random_state=42)
    all_models = _models()
    names = models or list(all_models)
    rows, fold_mse = [], {}
    for name in names:
        m, xs, lg = all_models[name]
        X = Xs[xs]
        P = np.zeros(len(y))
        f_mse = []
        for tr, te in kf.split(X):
            m.fit(X[tr], np.log(y[tr]) if lg else y[tr])
            p = m.predict(X[te])
            p = np.exp(p) if lg else p
            P[te] = p
            f_mse.append(float(((y[te] - p) ** 2).mean()))
        fold_mse[name] = np.array(f_mse)
        ape = np.abs(y - P) / y * 100
        rows.append({"model": name, **regression_metrics(y, P), "median_ape": float(np.median(ape))})
    ntest = len(y) / n_splits
    ntrain = len(y) - ntest
    comp = []
    for a in names:
        row = []
        for b in names:
            if a == b:
                row.append(None)
                continue
            d = fold_mse[a] - fold_mse[b]
            se = np.sqrt((1 / n_splits + ntest / ntrain) * d.var(ddof=1)) or 1e-12
            row.append(float(stats.t.cdf(-d.mean() / se, n_splits - 1)))
        comp.append(row)
    return {"rows": rows, "comparison": {"names": names, "p_better": comp}}


def predict(data_key: str, rd: float, capex: float, margin: float) -> dict:
    c = prepare(data_key)
    m1 = OLS(c[[RD]], c[R], ["R&D"])
    m1l = OLS(c[["lRD"]], c["lR"], ["ln R&D"])
    m2 = OLS(c[[RD, CX, OM]], c[R], ["R&D", "Capex", "Margin"])
    m2l = OLS(c[["lRD", "lCX", OM]], c["lR"], ["ln R&D", "ln Capex", "Margin"])
    return {
        "Sederhana – linear": float(m1.predict([[rd]])[0]),
        "Sederhana – log-log": float(np.exp(m1l.predict([[np.log(rd)]])[0])),
        "Berganda – linear": float(m2.predict([[rd, capex, margin]])[0]),
        "Berganda – log-log": float(np.exp(m2l.predict([[np.log(rd), np.log(capex), margin]])[0])),
    }


def manual_simple(data_key: str, n_rows: int = 10, log: bool = False) -> dict:
    """Contoh manual rumus kuadrat terkecil regresi sederhana (materi Pertemuan 2):
    b = (nΣXY − ΣXΣY) / (nΣX² − (ΣX)²),  a = (ΣY − bΣX) / n."""
    c = prepare(data_key).head(n_rows)
    x = (np.log(c[RD]) if log else c[RD]).values.astype(float)
    y = (np.log(c[R]) if log else c[R]).values.astype(float)
    n = len(x)
    sx, sy, sxy, sxx = x.sum(), y.sum(), (x * y).sum(), (x * x).sum()
    b = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    a = (sy - b * sx) / n
    sk = LinearRegression().fit(x.reshape(-1, 1), y)
    return {"rows": [{"company": str(c["company_name"].iloc[i]), "year": int(c["year"].iloc[i]),
                      "x": float(x[i]), "y": float(y[i]), "xy": float(x[i] * y[i]), "x2": float(x[i] ** 2)}
                     for i in range(n)],
            "n": n, "sum_x": sx, "sum_y": sy, "sum_xy": sxy, "sum_x2": sxx,
            "b": float(b), "a": float(a), "b_sklearn": float(sk.coef_[0]), "a_sklearn": float(sk.intercept_),
            "max_abs_diff": float(max(abs(b - sk.coef_[0]), abs(a - sk.intercept_)))}


def _model_by_name(c: pd.DataFrame, which: str) -> OLS:
    if which == "simple":
        return OLS(c[[RD]], c[R], ["R&D"])
    if which == "simple_log":
        return OLS(c[["lRD"]], c["lR"], ["ln R&D"])
    if which == "multiple":
        return OLS(c[[RD, CX, OM]], c[R], ["R&D", "Capex", "Margin"])
    return OLS(c[["lRD", "lCX", OM]], c["lR"], ["ln R&D", "ln Capex", "Margin"])


def diagnostics(data_key: str, which: str = "simple") -> dict:
    """Residual vs Fitted, Q–Q normal, Scale–Location, leverage & jarak Cook untuk satu model."""
    c = prepare(data_key)
    m = _model_by_name(c, which)
    h = np.einsum("ij,jk,ik->i", m.X, m.XtXi, m.X)
    std_res = m.res / np.sqrt(m.s2 * (1 - h))
    cooks = m.res ** 2 / (m.k * m.s2) * h / (1 - h) ** 2
    order = np.argsort(std_res)
    theo = stats.norm.ppf((np.arange(1, m.n + 1) - 0.5) / m.n)
    top = np.argsort(cooks)[::-1][:8]
    return {
        "which": which, "fitted": m.fit, "resid": m.res, "std_resid": std_res,
        "sqrt_abs_std_resid": np.sqrt(np.abs(std_res)), "leverage": h, "cooks": cooks,
        "qq_theoretical": theo, "qq_sample": std_res[order],
        "cooks_threshold": 4 / m.n,
        "top_influential": [{"company": str(c["company_name"].iloc[i]), "year": int(c["year"].iloc[i]),
                             "cooks": float(cooks[i]), "leverage": float(h[i])} for i in top],
        "tests": m.summary(),
        "label": [f"{a} {b}" for a, b in zip(c["company_name"], c["year"])],
    }


def regression_band(data_key: str, log: bool = False, n_grid: int = 60) -> dict:
    """Garis regresi sederhana dengan confidence interval & prediction interval 95%."""
    c = prepare(data_key)
    x = (c["lRD"] if log else c[RD]).values
    y = (c["lR"] if log else c[R]).values
    m = OLS(x.reshape(-1, 1), y, ["x"])
    g = np.linspace(x.min(), x.max(), n_grid)
    G = np.column_stack([np.ones(n_grid), g])
    fit = G @ m.b
    se_mean = np.sqrt(np.einsum("ij,jk,ik->i", G, m.XtXi, G) * m.s2)
    tc = stats.t.ppf(0.975, m.df)
    se_pred = np.sqrt(m.s2 + se_mean ** 2)
    return {"x": x, "y": y, "segment": c["segment"].values, "grid": g, "fit": fit,
            "ci_low": fit - tc * se_mean, "ci_high": fit + tc * se_mean,
            "pi_low": fit - tc * se_pred, "pi_high": fit + tc * se_pred,
            "coef": m.b, "r2": m.r2, "log": log}


def manual_multiple(data_key: str, n_rows: int = 12) -> dict:
    """Regresi berganda manual b = (XᵀX)⁻¹XᵀY pada subset kecil, dicocokkan dengan LinearRegression."""
    c = prepare(data_key).head(n_rows)
    X = c[[RD, CX, OM]].values.astype(float)
    y = c[R].values.astype(float)
    Xd = np.column_stack([np.ones(len(y)), X])
    XtX = Xd.T @ Xd
    XtXi = np.linalg.inv(XtX)
    Xty = Xd.T @ y
    b = XtXi @ Xty
    sk = LinearRegression().fit(X, y)
    bsk = np.r_[sk.intercept_, sk.coef_]
    return {"names": ["b0", "R&D", "Capex", "Margin"], "X": Xd, "y": y, "XtX": XtX, "XtXi": XtXi, "Xty": Xty,
            "b_manual": b, "b_sklearn": bsk, "max_abs_diff": float(np.max(np.abs(b - bsk))),
            "rows": [{"company": str(r.company_name), "year": int(r.year)} for r in c.itertuples()]}


def feature_stats(data_key: str) -> dict:
    """Ringkasan bergaya widget Feature Statistics + Box plot per segmen (skala log)."""
    c = prepare(data_key)
    cols = {R: "Pendapatan", RD: "R&D", CX: "Capex", OM: "Margin operasi"}
    out = []
    for col, nm in cols.items():
        s = c[col]
        out.append({"feature": nm, "column": col, "mean": s.mean(), "median": s.median(), "std": s.std(),
                    "min": s.min(), "max": s.max(), "skew": s.skew(), "missing": int(s.isna().sum()),
                    "hist": np.histogram(s, bins=24)[0].tolist(), "hist_log": np.histogram(np.log(s.clip(lower=1e-9)), bins=24)[0].tolist() if col != OM else None})
    seg = {g: np.log(d[R]).tolist() for g, d in c.groupby("segment")}
    return {"features": out, "log_revenue_by_segment": seg}
