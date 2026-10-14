"""Profil data (EDA) dan praproses umum untuk Data Lab / Regression Lab."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import get, histogram


def profile(data_key: str, max_cat: int = 12) -> dict:
    df = get(data_key)
    cols = []
    for c in df.columns:
        s = df[c]
        info = {"name": str(c), "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean() * 100), "unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            v = s.dropna().astype(float)
            info["kind"] = "numerik"
            if len(v):
                q1, q3 = v.quantile(0.25), v.quantile(0.75)
                iqr = q3 - q1
                info.update({"mean": v.mean(), "std": v.std(), "min": v.min(), "q1": q1, "median": v.median(),
                             "q3": q3, "max": v.max(), "skew": v.skew(),
                             "n_outlier_iqr": int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()),
                             "lower_fence": q1 - 1.5 * iqr, "upper_fence": q3 + 1.5 * iqr,
                             "hist": histogram(v, 30)})
        else:
            info["kind"] = "kategorikal" if info["unique"] <= max(50, len(df) // 20) else "teks"
            vc = s.astype(str).value_counts().head(max_cat)
            info["top"] = [{"value": str(k), "count": int(n)} for k, n in vc.items()]
        cols.append(info)
    num = df.select_dtypes(include="number")
    corr = num.corr() if num.shape[1] >= 2 else None
    return {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]), "duplicates": int(df.duplicated().sum()),
            "missing_total": int(df.isna().sum().sum()), "columns": cols,
            "corr": None if corr is None else {"cols": [str(c) for c in corr.columns], "values": corr.values},
            "head": df.head(8).astype(str).to_dict(orient="records")}


def preprocess(data_key: str, impute: str = "median", outlier: str = "none", iqr_k: float = 1.5,
               encode: str = "onehot", scale: str = "none", out_key: str | None = None) -> dict:
    """Praproses demonstratif SELURUH data (untuk eksplorasi before–after).
    Untuk pemodelan, Data Lab memakai Pipeline yang di-fit hanya pada data latih."""
    df = get(data_key).copy()
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    cat = [c for c in df.columns if c not in num]
    before = {c: {"missing": int(df[c].isna().sum()), "mean": float(df[c].mean()), "std": float(df[c].std()),
                  "min": float(df[c].min()), "max": float(df[c].max()), "hist": histogram(df[c], 25)} for c in num}
    log = []
    if impute != "none":
        for c in num:
            n = int(df[c].isna().sum())
            if n == 0:
                continue
            if impute == "mean":
                df[c] = df[c].fillna(df[c].mean())
            elif impute == "median":
                df[c] = df[c].fillna(df[c].median())
            elif impute == "interpolate":
                df[c] = df[c].interpolate(limit_direction="both")
            log.append(f"{c}: {n} nilai hilang diisi ({impute})")
        for c in cat:
            n = int(df[c].isna().sum())
            if n:
                df[c] = df[c].fillna(df[c].mode().iloc[0] if df[c].notna().any() else "NA")
                log.append(f"{c}: {n} nilai hilang diisi modus")
    if outlier in ("winsorize", "drop"):
        mask = pd.Series(False, index=df.index)
        for c in num:
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            lo, hi = q1 - iqr_k * (q3 - q1), q3 + iqr_k * (q3 - q1)
            m = (df[c] < lo) | (df[c] > hi)
            if outlier == "winsorize":
                df[c] = df[c].clip(lo, hi)
                if m.sum():
                    log.append(f"{c}: {int(m.sum())} nilai di-winsorize ke [{lo:.4g}, {hi:.4g}]")
            mask |= m
        if outlier == "drop":
            log.append(f"{int(mask.sum())} baris dengan outlier IQR dibuang")
            df = df[~mask]
    if scale in ("standard", "minmax"):
        for c in num:
            s = df[c]
            df[c] = (s - s.mean()) / s.std(ddof=0) if scale == "standard" else (s - s.min()) / (s.max() - s.min())
        log.append(f"Scaling {scale} pada {len(num)} kolom numerik")
    if encode == "onehot" and cat:
        small = [c for c in cat if df[c].nunique() <= 30]
        df = pd.get_dummies(df, columns=small, dtype=int)
        log.append(f"One-hot: {', '.join(map(str, small))}" if small else "Tidak ada kolom kategorikal kecil")
    elif encode == "label" and cat:
        for c in cat:
            df[c] = df[c].astype("category").cat.codes
        log.append(f"Label encoding: {', '.join(map(str, cat))}")
    num_after = [c for c in num if c in df.columns]
    after = {c: {"missing": int(df[c].isna().sum()), "mean": float(df[c].mean()), "std": float(df[c].std()),
                 "min": float(df[c].min()), "max": float(df[c].max()), "hist": histogram(df[c], 25)} for c in num_after}
    if out_key:
        from .common import set_frame
        set_frame(out_key, df.reset_index(drop=True))
    return {"log": log, "before": before, "after": after, "shape_before": list(get(data_key).shape),
            "shape_after": list(df.shape), "head": df.head(8).astype(str).to_dict(orient="records"),
            "csv": df.to_csv(index=False) if len(df) <= 20000 else None}


def columns(data_key: str) -> dict:
    """Daftar kolom beserta jenisnya (untuk mengisi pilihan di UI)."""
    df = get(data_key)
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            kind = "numerik"
        else:
            avg_len = s.dropna().astype(str).str.len().mean() if s.notna().any() else 0
            kind = "teks" if (s.nunique() > max(50, len(df) // 20) and avg_len > 25) else "kategorikal"
        out.append({"name": str(c), "kind": kind, "unique": int(s.nunique()), "missing": int(s.isna().sum())})
    return {"n_rows": len(df), "columns": out}


def simple_pair(data_key: str, x: str, y: str, n_rows: int = 12, log: bool = False) -> dict:
    """Pasangan X–Y (n baris pertama tanpa nilai kosong) untuk perhitungan manual regresi sederhana,
    beserta koefisien LinearRegression scikit-learn sebagai pembanding."""
    from sklearn.linear_model import LinearRegression
    d = get(data_key)[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if log:
        d = d[(d[x] > 0) & (d[y] > 0)]
        d = np.log(d)
    d = d.head(n_rows)
    sk = LinearRegression().fit(d[[x]].values, d[y].values)
    return {"x": d[x].values, "y": d[y].values, "a_sklearn": float(sk.intercept_), "b_sklearn": float(sk.coef_[0]), "n": len(d)}
