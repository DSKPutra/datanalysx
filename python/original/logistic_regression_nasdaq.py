"""
==============================================================================
 LAMPIRAN PROGRAM: REGRESI LOGISTIK UNTUK PREDIKSI ARAH HARGA SAHAM
 Studi Kasus Utama: Saham AAPL (Apple Inc.) — NASDAQ Daily OHLC Dataset
 Uji Generalisasi: 20 Emiten Teknologi/Konsumer NASDAQ Berlikuiditas Tinggi
 Program Studi Magister Teknik Informatika - Universitas Pamulang
 Mata Kuliah: Data Mining
 Penulis   : Dea Saka Kurnia Putra
 NIM       : 261012000111
==============================================================================
 
Versi 2 — analisis diperluas dengan:
  1. Model dasar regresi logistik (L2-regularized) + confusion matrix + metrik.
  2. Inferensi statistik model MLE tanpa regularisasi: koefisien, galat baku,
     nilai-z, p-value, odds ratio, interval kepercayaan 95%, uji rasio
     likelihood (LR test) terhadap model null, dan McFadden pseudo-R^2.
  3. Uji multikolinearitas via Variance Inflation Factor (VIF).
  4. Validasi silang walk-forward (expanding window) 5-fold untuk menguji
     kestabilan/robustness model di luar satu kali pembagian data.
  5. Penalaan hyperparameter C via TimeSeriesSplit + kurva validasi.
  6. Optimasi ambang batas (threshold tuning) berbasis F1-score.
  7. Kurva kalibrasi probabilitas (reliability diagram) + Brier score.
  8. Permutation importance sebagai pembanding koefisien mentah.
  9. Interval kepercayaan ROC-AUC via bootstrap resampling (2.000 replikasi).
 10. Kurva belajar (learning curve): ukuran data latih vs performa.
 11. Uji generalisasi lintas-saham pada 20 emiten NASDAQ berlikuiditas tinggi,
     dengan uji hipotesis satu-sampel (t-test & Wilcoxon signed-rank) apakah
     rata-rata ROC-AUC berbeda signifikan dari 0,5 (baseline tebakan acak).
 
Seluruh angka pada laporan (Bab IV) diambil langsung dari keluaran program
ini (lihat outputs/model_summary.json), bukan estimasi manual.
==============================================================================
"""
 
import warnings
warnings.filterwarnings("ignore")
 
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, learning_curve
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, matthews_corrcoef, balanced_accuracy_score,
    log_loss, brier_score_loss
)
 
# ---------------------------------------------------------------------------
# 0. KONFIGURASI TEMA VISUAL (Hijau-Hitam Luxury)
# ---------------------------------------------------------------------------
INK_900, INK_800, INK_700 = "#080A09", "#0E1211", "#171B19"
PAPER_050, PAPER_100, PAPER_300 = "#F7F7F5", "#F4F4F2", "#D8D8D3"
GREEN_300, GREEN_400, GREEN_500 = "#6BE87F", "#2FDF4A", "#21D138"
GREEN_600, GREEN_700 = "#12A227", "#0B6B1C"
GREY_500, GREY_600 = "#7C807D", "#585C59"
RED = "#B3261E"
 
plt.rcParams.update({
    "figure.facecolor": PAPER_050, "axes.facecolor": PAPER_050, "axes.edgecolor": PAPER_300,
    "axes.labelcolor": INK_900, "text.color": INK_900, "xtick.color": GREY_600, "ytick.color": GREY_600,
    "font.family": "sans-serif", "axes.grid": True, "grid.color": PAPER_300, "grid.alpha": 0.6,
    "grid.linewidth": 0.6, "font.size": 11,
})
OUT_DIR = "charts"
RNG = np.random.default_rng(42)
 
FEATURE_COLS = [
    "return_1d", "return_2d", "return_5d", "ma_gap_5_20",
    "vol_10d", "vol_20d", "hl_spread", "oc_spread", "rsi_14", "momentum_10",
]
 
# ---------------------------------------------------------------------------
# 1. MEMUAT, MEMBERSIHKAN & REKAYASA FITUR (dipakai berulang utk multi-saham)
# ---------------------------------------------------------------------------
def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset="date")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[(df["high"] >= df["low"]) & (df["high"] >= df["open"]) & (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) & (df["low"] <= df["close"])]
    df = df[df["date"] >= "2010-01-01"].reset_index(drop=True)
    return df
 
 
def engineer_features(df):
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
    d["target"] = (d["close"].shift(-1) > d["close"]).astype(int)
    d = d.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    return d
 
 
def time_split(d, test_size=0.2):
    n = len(d)
    split_idx = int(n * (1 - test_size))
    train, test = d.iloc[:split_idx], d.iloc[split_idx:]
    return (train, test, train[FEATURE_COLS].values, train["target"].values,
            test[FEATURE_COLS].values, test["target"].values)
 
 
def train_model(X_train, y_train, C=1.0):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000, random_state=42)
    model.fit(X_train_s, y_train)
    return model, scaler
 
 
def evaluate(model, scaler, X_test, y_test):
    X_test_s = scaler.transform(X_test)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else np.nan,
        "mcc": matthews_corrcoef(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba, labels=[0, 1]),
        "brier": brier_score_loss(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return y_pred, y_proba, metrics, cm
 
 
def savefig(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
 
 
# ---------------------------------------------------------------------------
# 2. INFERENSI STATISTIK: MLE LOGIT TANPA REGULARISASI (IRLS manual)
#    -> koefisien, galat baku (SE), z, p-value, odds ratio, CI 95%,
#       log-likelihood, LR-test vs model null, McFadden pseudo-R^2
# ---------------------------------------------------------------------------
def fit_mle_logit(X, y, add_intercept=True, max_iter=100, tol=1e-8):
    Xd = np.column_stack([np.ones(len(X)), X]) if add_intercept else X
    n, k = Xd.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        z = Xd @ beta
        p = 1 / (1 + np.exp(-z))
        w = p * (1 - p)
        w = np.clip(w, 1e-8, None)
        W = Xd * w[:, None]
        hessian = Xd.T @ W                      # information matrix (approx.)
        grad = Xd.T @ (y - p)
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, grad, rcond=None)[0]
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    z_final = Xd @ beta
    p_final = 1 / (1 + np.exp(-z_final))
    p_final = np.clip(p_final, 1e-12, 1 - 1e-12)
    ll = np.sum(y * np.log(p_final) + (1 - y) * np.log(1 - p_final))
    w = np.clip(p_final * (1 - p_final), 1e-8, None)
    hessian = Xd.T @ (Xd * w[:, None])
    cov = np.linalg.inv(hessian)
    se = np.sqrt(np.diag(cov))
    zval = beta / se
    pval = 2 * (1 - stats.norm.cdf(np.abs(zval)))
    ci_low, ci_high = beta - 1.96 * se, beta + 1.96 * se
    return {
        "beta": beta, "se": se, "z": zval, "p": pval,
        "ci_low": ci_low, "ci_high": ci_high, "loglik": ll,
    }
 
 
def null_loglik(y):
    p0 = np.clip(y.mean(), 1e-12, 1 - 1e-12)
    return np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
 
 
def variance_inflation_factors(X, feature_names):
    """VIF_j = 1 / (1 - R^2) dari regresi OLS fitur j terhadap fitur lainnya."""
    vifs = []
    Xc = np.column_stack([np.ones(len(X)), X])
    for j in range(X.shape[1]):
        y_j = X[:, j]
        X_others = np.delete(Xc, j + 1, axis=1)
        coef, *_ = np.linalg.lstsq(X_others, y_j, rcond=None)
        y_hat = X_others @ coef
        ss_res = np.sum((y_j - y_hat) ** 2)
        ss_tot = np.sum((y_j - y_j.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif = 1 / (1 - r2) if r2 < 0.9999 else np.inf
        vifs.append(vif)
    return dict(zip(feature_names, vifs))
 
 
# ---------------------------------------------------------------------------
# 3. VALIDASI SILANG WALK-FORWARD (EXPANDING WINDOW), 5 LIPATAN
# ---------------------------------------------------------------------------
def walk_forward_cv(d, n_splits=5):
    n = len(d)
    fold_size = n // (n_splits + 1)
    results = []
    boundaries = [fold_size * i for i in range(1, n_splits + 2)]
    for i in range(n_splits):
        train_idx = slice(0, boundaries[i])
        test_idx = slice(boundaries[i], boundaries[i + 1])
        train_d, test_d = d.iloc[train_idx], d.iloc[test_idx]
        if train_d["target"].nunique() < 2 or test_d["target"].nunique() < 2 or len(test_d) < 20:
            continue
        Xtr, ytr = train_d[FEATURE_COLS].values, train_d["target"].values
        Xte, yte = test_d[FEATURE_COLS].values, test_d["target"].values
        model, scaler = train_model(Xtr, ytr)
        _, _, m, _ = evaluate(model, scaler, Xte, yte)
        results.append({
            "fold": i + 1, "n_train": len(train_d), "n_test": len(test_d),
            "accuracy": m["accuracy"], "roc_auc": m["roc_auc"], "f1": m["f1"],
        })
    return pd.DataFrame(results)
 
 
# ---------------------------------------------------------------------------
# 4. PENALAAN HYPERPARAMETER C VIA TimeSeriesSplit
# ---------------------------------------------------------------------------
def tune_hyperparameter(X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    Cs = np.logspace(-3, 2, 12)
    grid = GridSearchCV(
        LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, random_state=42),
        param_grid={"C": Cs}, scoring="roc_auc", cv=tscv, n_jobs=1,
    )
    grid.fit(X_s, y_train)
    return grid, Cs
 
 
# ---------------------------------------------------------------------------
# 5. OPTIMASI AMBANG BATAS (THRESHOLD TUNING)
# ---------------------------------------------------------------------------
def threshold_tuning(y_test, y_proba):
    thresholds = np.linspace(0.05, 0.95, 37)
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        })
    tdf = pd.DataFrame(rows)
    best = tdf.loc[tdf["f1"].idxmax()]
    return tdf, best
 
 
# ---------------------------------------------------------------------------
# 6. BOOTSTRAP CONFIDENCE INTERVAL UNTUK ROC-AUC
# ---------------------------------------------------------------------------
def bootstrap_auc_ci(y_test, y_proba, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_test)
    aucs = []
    y_test = np.asarray(y_test)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y_test[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_test[idx], y_proba[idx]))
    aucs = np.array(aucs)
    return aucs, np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
 
 
# ---------------------------------------------------------------------------
# 7. UJI GENERALISASI LINTAS-SAHAM (20 emiten)
# ---------------------------------------------------------------------------
BASKET_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "ADBE", "CSCO", "INTC", "AMD", "QCOM",
    "TXN", "AMAT", "PYPL", "SBUX", "GILD", "COST", "PEP", "CMCSA", "MU", "ISRG",
]
 
def multi_ticker_test(zip_path, tickers):
    import zipfile
    results = []
    with zipfile.ZipFile(zip_path) as z:
        for tkr in tickers:
            try:
                with z.open(f"{tkr}.csv") as f:
                    df = pd.read_csv(f)
            except KeyError:
                continue
            df.columns = [c.strip().lower() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df = df.drop_duplicates(subset="date").dropna(subset=["open", "high", "low", "close"])
            df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
            df = df[df["date"] >= "2010-01-01"].reset_index(drop=True)
            if len(df) < 500:
                continue
            d = engineer_features(df)
            if len(d) < 300:
                continue
            train, test, Xtr, ytr, Xte, yte = time_split(d)
            if len(set(ytr)) < 2 or len(set(yte)) < 2:
                continue
            model, scaler = train_model(Xtr, ytr)
            _, _, m, _ = evaluate(model, scaler, Xte, yte)
            results.append({"ticker": tkr, "n_obs": len(d), **m})
    return pd.DataFrame(results)
 
 
# ---------------------------------------------------------------------------
# VISUALISASI TAMBAHAN (ANALISIS LANJUTAN)
# ---------------------------------------------------------------------------
def chart_vif(vif_dict):
    names = list(vif_dict.keys())
    vals = list(vif_dict.values())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [RED if v > 10 else (GREEN_600 if v > 5 else GREEN_500) for v in vals]
    ax.barh(names, vals, color=colors, edgecolor=INK_900, linewidth=0.5)
    ax.axvline(5, color=GREY_600, linestyle="--", linewidth=1, label="VIF = 5 (waspada)")
    ax.axvline(10, color=RED, linestyle="--", linewidth=1, label="VIF = 10 (bermasalah)")
    ax.set_title("Variance Inflation Factor (VIF) per Fitur", fontsize=13, pad=12)
    ax.set_xlabel("VIF"); ax.legend(frameon=False, fontsize=8)
    savefig(fig, "14_vif_multikolinearitas")
 
 
def chart_walk_forward(wf_df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(wf_df["fold"], wf_df["roc_auc"], marker="o", color=GREEN_600, linewidth=2, label="ROC-AUC")
    ax.plot(wf_df["fold"], wf_df["accuracy"], marker="s", color=INK_900, linewidth=2, label="Akurasi")
    ax.axhline(0.5, color=GREY_500, linestyle="--", linewidth=1, label="Baseline Acak (0,5)")
    ax.set_xticks(wf_df["fold"])
    ax.set_xlabel("Lipatan (Fold) Walk-Forward"); ax.set_ylabel("Nilai Metrik")
    ax.set_title("Validasi Silang Walk-Forward (Expanding Window), 5 Lipatan", fontsize=13, pad=12)
    ax.legend(frameon=False)
    savefig(fig, "15_walk_forward_cv")
 
 
def chart_hyperparam(grid, Cs):
    means = grid.cv_results_["mean_test_score"]
    stds = grid.cv_results_["std_test_score"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(Cs, means, color=GREEN_600, marker="o", linewidth=2)
    ax.fill_between(Cs, means - stds, means + stds, color=GREEN_500, alpha=0.15)
    ax.set_xscale("log")
    ax.axvline(grid.best_params_["C"], color=RED, linestyle="--", linewidth=1,
               label=f"C optimal = {grid.best_params_['C']:.4g}")
    ax.set_xlabel("Nilai C (invers kekuatan regularisasi, skala log)")
    ax.set_ylabel("Rata-rata ROC-AUC (TimeSeriesSplit, 5-fold)")
    ax.set_title("Kurva Validasi Hyperparameter C", fontsize=13, pad=12)
    ax.legend(frameon=False)
    savefig(fig, "16_kurva_validasi_hyperparameter")
 
 
def chart_threshold(tdf, best):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tdf["threshold"], tdf["precision"], color=INK_900, linewidth=1.6, label="Presisi")
    ax.plot(tdf["threshold"], tdf["recall"], color=GREEN_600, linewidth=1.6, label="Recall")
    ax.plot(tdf["threshold"], tdf["f1"], color=RED, linewidth=2, label="F1-Score")
    ax.axvline(best["threshold"], color=GREY_600, linestyle="--", linewidth=1,
               label=f"Threshold optimal F1 = {best['threshold']:.2f}")
    ax.set_xlabel("Ambang Batas (Threshold)"); ax.set_ylabel("Nilai Metrik")
    ax.set_title("Optimasi Ambang Batas Klasifikasi", fontsize=13, pad=12)
    ax.legend(frameon=False)
    savefig(fig, "17_optimasi_threshold")
 
 
def chart_calibration(y_test, y_proba, brier):
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color=GREY_500, linestyle="--", linewidth=1, label="Kalibrasi Sempurna")
    ax.plot(mean_pred, frac_pos, marker="o", color=GREEN_600, linewidth=2, label=f"Model (Brier = {brier:.4f})")
    ax.set_xlabel("Rata-rata Probabilitas Prediksi"); ax.set_ylabel("Frekuensi Observasi Aktual (Naik)")
    ax.set_title("Kurva Kalibrasi (Reliability Diagram)", fontsize=13, pad=12)
    ax.legend(frameon=False)
    savefig(fig, "18_kurva_kalibrasi")
 
 
def chart_permutation_importance(perm, feature_cols):
    order = np.argsort(perm.importances_mean)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_cols)[order], perm.importances_mean[order],
            xerr=perm.importances_std[order], color=GREEN_500, edgecolor=INK_900, linewidth=0.4,
            error_kw={"ecolor": INK_700, "elinewidth": 1})
    ax.axvline(0, color=GREY_600, linewidth=0.8)
    ax.set_title("Permutation Importance (ΔROC-AUC, 30 pengulangan)", fontsize=13, pad=12)
    ax.set_xlabel("Penurunan Rata-rata ROC-AUC Saat Fitur Diacak")
    savefig(fig, "19_permutation_importance")
 
 
def chart_bootstrap_auc(aucs, lo, hi, point_auc):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(aucs, bins=50, color=GREEN_500, alpha=0.85, edgecolor=INK_900, linewidth=0.15)
    ax.axvline(0.5, color=INK_900, linestyle="--", linewidth=1.2, label="Baseline Acak (0,5)")
    ax.axvline(point_auc, color=RED, linewidth=1.5, label=f"ROC-AUC Titik = {point_auc:.3f}")
    ax.axvspan(lo, hi, color=GREY_500, alpha=0.2, label=f"CI 95% [{lo:.3f}, {hi:.3f}]")
    ax.set_title("Distribusi Bootstrap ROC-AUC (2.000 Replikasi)", fontsize=13, pad=12)
    ax.set_xlabel("ROC-AUC"); ax.legend(frameon=False, fontsize=9)
    savefig(fig, "20_bootstrap_auc")
 
 
def chart_learning_curve(train_sizes, train_scores, test_scores):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_sizes, train_scores.mean(axis=1), marker="o", color=INK_900, linewidth=2, label="Skor Data Latih")
    ax.plot(train_sizes, test_scores.mean(axis=1), marker="s", color=GREEN_600, linewidth=2, label="Skor Data Uji")
    ax.axhline(0.5, color=GREY_500, linestyle="--", linewidth=1, label="Baseline Acak (0,5)")
    ax.set_xlabel("Jumlah Sampel Data Latih"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Kurva Belajar (Learning Curve)", fontsize=13, pad=12)
    ax.legend(frameon=False)
    savefig(fig, "21_kurva_belajar")
 
 
def chart_multi_ticker(mt_df):
    mt_df = mt_df.sort_values("roc_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [GREEN_600 if v >= 0.5 else GREY_500 for v in mt_df["roc_auc"]]
    ax.barh(mt_df["ticker"], mt_df["roc_auc"], color=colors, edgecolor=INK_900, linewidth=0.4)
    ax.axvline(0.5, color=INK_900, linestyle="--", linewidth=1.2, label="Baseline Acak (0,5)")
    ax.axvline(mt_df["roc_auc"].mean(), color=RED, linewidth=1.5,
               label=f"Rata-rata = {mt_df['roc_auc'].mean():.3f}")
    ax.set_title("ROC-AUC Model per Emiten (Uji Generalisasi, n=20)", fontsize=13, pad=12)
    ax.set_xlabel("ROC-AUC"); ax.legend(frameon=False, fontsize=9)
    savefig(fig, "22_multi_ticker_auc")
 
 
def chart_multi_ticker_box(mt_df):
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(mt_df["roc_auc"], vert=True, widths=0.4, patch_artist=True,
                     medianprops={"color": INK_900, "linewidth": 2},
                     boxprops={"facecolor": GREEN_300, "edgecolor": INK_900},
                     whiskerprops={"color": INK_900}, capprops={"color": INK_900})
    ax.scatter(np.random.normal(1, 0.04, len(mt_df)), mt_df["roc_auc"], color=GREEN_700, alpha=0.7, zorder=5, s=22)
    ax.axhline(0.5, color=RED, linestyle="--", linewidth=1.2, label="Baseline Acak (0,5)")
    ax.set_xticks([1]); ax.set_xticklabels(["20 Emiten NASDAQ"])
    ax.set_ylabel("ROC-AUC"); ax.legend(frameon=False)
    ax.set_title("Sebaran ROC-AUC Lintas-Saham", fontsize=13, pad=12)
    savefig(fig, "23_multi_ticker_boxplot")
 
 
# ===========================================================================
# MAIN
# ===========================================================================
def main():
    zip_path = "/mnt/user-data/uploads/NASDAQ daily stock prices OHLC prices between 1962-01-02 till 2026-09-18.zip"
 
    print("[1/12] Memuat & membersihkan data AAPL ...")
    df = load_and_clean("data/AAPL.csv")
    d = engineer_features(df)
    train, test, X_train, y_train, X_test, y_test = time_split(d)
    print(f"      -> total={len(d)}, latih={len(train)}, uji={len(test)}")
 
    print("[2/12] Melatih model utama (L2, C=1.0) & evaluasi ...")
    model, scaler = train_model(X_train, y_train)
    y_pred, y_proba, metrics, cm = evaluate(model, scaler, X_test, y_test)
    print(classification_report(y_test, y_pred, target_names=["Turun/Tetap (0)", "Naik (1)"]))
 
    print("[3/12] Inferensi statistik MLE (tanpa regularisasi) ...")
    Xtr_s = scaler.transform(X_train)
    mle = fit_mle_logit(Xtr_s, y_train)
    ll_null = null_loglik(y_train)
    lr_stat = 2 * (mle["loglik"] - ll_null)
    df_lr = len(FEATURE_COLS)
    lr_pvalue = 1 - stats.chi2.cdf(lr_stat, df_lr)
    mcfadden_r2 = 1 - mle["loglik"] / ll_null
    mle_table = pd.DataFrame({
        "feature": ["intercept"] + FEATURE_COLS,
        "coef": mle["beta"], "se": mle["se"], "z": mle["z"], "p_value": mle["p"],
        "odds_ratio": np.exp(mle["beta"]),
        "ci_low": np.exp(mle["ci_low"]), "ci_high": np.exp(mle["ci_high"]),
    })
    print(mle_table.round(4).to_string(index=False))
    print(f"      LR-stat={lr_stat:.3f}, df={df_lr}, p={lr_pvalue:.4g}, McFadden R^2={mcfadden_r2:.4f}")
 
    print("[4/12] Menghitung VIF (multikolinearitas) ...")
    vif = variance_inflation_factors(Xtr_s, FEATURE_COLS)
    for k, v in vif.items():
        print(f"      {k:>14s}: VIF = {v:.2f}")
 
    print("[5/12] Validasi silang walk-forward (5-fold, expanding window) ...")
    wf_df = walk_forward_cv(d, n_splits=5)
    print(wf_df.round(4).to_string(index=False))
    print(f"      Rata-rata ROC-AUC={wf_df['roc_auc'].mean():.4f} (SD={wf_df['roc_auc'].std():.4f})")
 
    print("[6/12] Penalaan hyperparameter C (TimeSeriesSplit, 5-fold) ...")
    grid, Cs = tune_hyperparameter(X_train, y_train)
    print(f"      C optimal = {grid.best_params_['C']:.4g}, ROC-AUC CV = {grid.best_score_:.4f}")
 
    print("[7/12] Optimasi ambang batas (threshold tuning) ...")
    tdf, best_t = threshold_tuning(y_test, y_proba)
    print(f"      Threshold optimal (F1) = {best_t['threshold']:.2f}, F1 = {best_t['f1']:.4f}")
 
    print("[8/12] Kurva kalibrasi & Brier score ...")
    brier = metrics["brier"]
 
    print("[9/12] Permutation importance (30 repeats) ...")
    perm = permutation_importance(model, scaler.transform(X_test), y_test,
                                   n_repeats=30, random_state=42, scoring="roc_auc")
 
    print("[10/12] Bootstrap CI ROC-AUC (2000 replikasi) ...")
    aucs, ci_lo, ci_hi = bootstrap_auc_ci(y_test, y_proba, n_boot=2000)
    print(f"      ROC-AUC = {metrics['roc_auc']:.4f}, CI95% = [{ci_lo:.4f}, {ci_hi:.4f}]")
    t_stat_auc, p_auc = stats.ttest_1samp(aucs, 0.5)
    print(f"      Uji-t 1 sampel AUC vs 0,5: t={t_stat_auc:.3f}, p={p_auc:.4g}")
 
    print("[11/12] Kurva belajar (learning curve) ...")
    Xall_s = scaler.transform(X_train)
    train_sizes, train_scores, test_scores = learning_curve(
        LogisticRegression(penalty="l2", C=grid.best_params_["C"], solver="lbfgs", max_iter=2000, random_state=42),
        Xall_s, y_train, train_sizes=np.linspace(0.2, 1.0, 6),
        cv=TimeSeriesSplit(n_splits=5), scoring="roc_auc", n_jobs=1,
    )
 
    print("[12/12] Uji generalisasi lintas-saham (20 emiten) ...")
    mt_df = multi_ticker_test(zip_path, BASKET_TICKERS)
    print(mt_df[["ticker", "n_obs", "accuracy", "roc_auc", "f1"]].round(4).to_string(index=False))
    t_stat_mt, p_mt = stats.ttest_1samp(mt_df["roc_auc"].dropna(), 0.5)
    w_stat_mt, p_w_mt = stats.wilcoxon(mt_df["roc_auc"].dropna() - 0.5)
    print(f"      Rata-rata ROC-AUC lintas-saham = {mt_df['roc_auc'].mean():.4f} (SD={mt_df['roc_auc'].std():.4f})")
    print(f"      Uji-t 1 sampel vs 0,5: t={t_stat_mt:.3f}, p={p_mt:.4g}")
    print(f"      Uji Wilcoxon signed-rank vs 0,5: W={w_stat_mt:.3f}, p={p_w_mt:.4g}")
 
    # ------------------------------ VISUALISASI -----------------------------
    print("Membuat seluruh visualisasi (dasar + lanjutan) ...")
    # -- grafik dasar (lihat versi sebelumnya utk detail dokumentasi) --
    from math import isnan
    def basic_charts():
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(df["date"], df["close"], color=INK_900, linewidth=1.1)
        ax.fill_between(df["date"], df["close"], df["close"].min(), color=GREEN_500, alpha=0.08)
        ax.set_title("Harga Penutupan (Close) Saham AAPL, 2010–2026", fontsize=13, pad=12)
        ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga Penutupan (USD)")
        savefig(fig, "01_harga_penutupan_historis")
 
        tail = d.tail(500)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(tail["date"], tail["close"], color=INK_900, linewidth=1.2, label="Close")
        ax.plot(tail["date"], tail["ma5"], color=GREEN_500, linewidth=1.1, label="MA-5")
        ax.plot(tail["date"], tail["ma20"], color=GREY_500, linewidth=1.1, label="MA-20")
        ax.set_title("Harga Close vs. Moving Average (500 Hari Terakhir)", fontsize=13, pad=12)
        ax.legend(frameon=False); ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga (USD)")
        savefig(fig, "02_moving_average_overlay")
 
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(d["return_1d"] * 100, bins=80, color=GREEN_500, alpha=0.85, edgecolor=INK_900, linewidth=0.2)
        ax.axvline(0, color=INK_900, linewidth=1)
        ax.set_title("Distribusi Return Harian (%)", fontsize=13, pad=12)
        ax.set_xlabel("Return Harian (%)"); ax.set_ylabel("Frekuensi")
        savefig(fig, "03_distribusi_return_harian")
 
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(d["date"], d["vol_20d"] * 100, color=INK_900, linewidth=1.0)
        ax.fill_between(d["date"], d["vol_20d"] * 100, 0, color=GREEN_500, alpha=0.15)
        ax.set_title("Volatilitas Bergulir 20-Hari (%)", fontsize=13, pad=12)
        ax.set_xlabel("Tanggal"); ax.set_ylabel("Volatilitas (%)")
        savefig(fig, "04_volatilitas_20hari")
 
        counts = d["target"].value_counts().sort_index()
        labels = ["Turun/Tetap (0)", "Naik (1)"]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bars = ax.bar(labels, counts.values, color=[GREY_500, GREEN_500], edgecolor=INK_900, linewidth=0.6)
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v}\n({v/counts.sum()*100:.1f}%)", ha="center", va="bottom", fontsize=10)
        ax.set_title("Distribusi Kelas Target (Label)", fontsize=13, pad=12); ax.set_ylabel("Jumlah Observasi")
        savefig(fig, "05_distribusi_kelas_target")
 
        corr = d[FEATURE_COLS + ["target"]].corr()
        fig, ax = plt.subplots(figsize=(8, 6.5))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ink_green", [INK_900, PAPER_050, GREEN_500])
        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7,
                         color=INK_900 if abs(corr.values[i, j]) < 0.6 else PAPER_050)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Koefisien Korelasi")
        ax.set_title("Matriks Korelasi Antar-Fitur", fontsize=13, pad=12)
        savefig(fig, "06_matriks_korelasi_fitur")
 
        coefs = model.coef_[0]; order = np.argsort(coefs)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [GREEN_500 if c > 0 else INK_700 for c in coefs[order]]
        ax.barh(np.array(FEATURE_COLS)[order], coefs[order], color=colors, edgecolor=INK_900, linewidth=0.4)
        ax.axvline(0, color=GREY_600, linewidth=0.8)
        ax.set_title("Koefisien Regresi Logistik per Fitur (Model L2, Standardized)", fontsize=13, pad=12)
        ax.set_xlabel("Nilai Koefisien (β)")
        savefig(fig, "07_koefisien_model")
 
        fig, ax = plt.subplots(figsize=(5.5, 5))
        cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("ink_green2", [PAPER_050, GREEN_500])
        im = ax.imshow(cm, cmap=cmap2)
        lbl = ["Turun/Tetap (0)", "Naik (1)"]
        ax.set_xticks([0, 1]); ax.set_xticklabels(lbl); ax.set_yticks([0, 1]); ax.set_yticklabels(lbl)
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                         color=INK_900 if cm[i, j] < cm.max()*0.6 else PAPER_050, fontweight="bold")
        ax.set_title("Confusion Matrix — Data Uji", fontsize=13, pad=12)
        fig.colorbar(im, ax=ax, shrink=0.8)
        savefig(fig, "08_confusion_matrix")
 
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.plot(fpr, tpr, color=GREEN_500, linewidth=2.2, label=f"Model (AUC = {metrics['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], color=GREY_500, linestyle="--", linewidth=1, label="Baseline Acak (AUC = 0.5)")
        ax.fill_between(fpr, tpr, 0, color=GREEN_500, alpha=0.08)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("Kurva ROC", fontsize=13, pad=12); ax.legend(frameon=False, loc="lower right")
        savefig(fig, "09_kurva_roc")
 
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.plot(rec, prec, color=INK_900, linewidth=2.0)
        ax.fill_between(rec, prec, 0, color=GREEN_500, alpha=0.12)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Kurva Precision-Recall", fontsize=13, pad=12)
        savefig(fig, "10_kurva_precision_recall")
 
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(y_proba[y_test == 0], bins=40, alpha=0.7, label="Aktual: Turun/Tetap", color=GREY_500, edgecolor=INK_900, linewidth=0.2)
        ax.hist(y_proba[y_test == 1], bins=40, alpha=0.7, label="Aktual: Naik", color=GREEN_500, edgecolor=INK_900, linewidth=0.2)
        ax.axvline(0.5, color=INK_900, linestyle="--", linewidth=1)
        ax.set_title("Distribusi Probabilitas Prediksi per Kelas Aktual", fontsize=13, pad=12)
        ax.set_xlabel("Probabilitas Prediksi P(Naik)"); ax.set_ylabel("Frekuensi"); ax.legend(frameon=False)
        savefig(fig, "11_distribusi_probabilitas_prediksi")
 
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(test["date"], test["close"], color=INK_900, linewidth=1.1)
        correct = (test["target"].values == y_pred)
        ax.scatter(test["date"][~correct], test["close"][~correct], color=RED, s=14, zorder=5, label="Prediksi Salah")
        ax.scatter(test["date"][correct], test["close"][correct], color=GREEN_500, s=10, zorder=4, alpha=0.6, label="Prediksi Benar")
        ax.set_title("Ketepatan Prediksi Arah Harga pada Periode Data Uji", fontsize=13, pad=12)
        ax.legend(frameon=False); ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga Penutupan (USD)")
        savefig(fig, "12_ketepatan_prediksi_periode_uji")
 
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy"]
        names = ["Akurasi", "Presisi", "Recall", "F1-Score", "ROC-AUC", "Bal. Acc."]
        vals = [metrics[k] for k in keys]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(names, vals, color=GREEN_500, edgecolor=INK_900, linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.05); ax.set_title("Ringkasan Metrik Evaluasi Model", fontsize=13, pad=12)
        savefig(fig, "13_ringkasan_metrik_evaluasi")
 
        z = np.linspace(-10, 10, 400); sig = 1 / (1 + np.exp(-z))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(z, sig, color=GREEN_500, linewidth=2.4)
        ax.axhline(0.5, color=GREY_500, linestyle="--", linewidth=0.8)
        ax.axvline(0, color=GREY_500, linestyle="--", linewidth=0.8)
        ax.set_title("Fungsi Sigmoid (Logistic Function)", fontsize=13, pad=12)
        ax.set_xlabel("z = β₀ + β₁X₁ + ... + βₙXₙ"); ax.set_ylabel("σ(z) = P(Y=1|X)")
        savefig(fig, "00_fungsi_sigmoid")
 
    basic_charts()
    chart_vif(vif)
    chart_walk_forward(wf_df)
    chart_hyperparam(grid, Cs)
    chart_threshold(tdf, best_t)
    chart_calibration(y_test, y_proba, brier)
    chart_permutation_importance(perm, FEATURE_COLS)
    chart_bootstrap_auc(aucs, ci_lo, ci_hi, metrics["roc_auc"])
    chart_learning_curve(train_sizes, train_scores, test_scores)
    chart_multi_ticker(mt_df)
    chart_multi_ticker_box(mt_df)
    print("Selesai. 24 grafik tersimpan di folder 'charts/'.")
 
    # ------------------------------ SIMPAN RINGKASAN -------------------------
    summary = {
        "n_total": len(d), "n_train": len(train), "n_test": len(test),
        "date_min": str(d["date"].min().date()), "date_max": str(d["date"].max().date()),
        "feature_cols": FEATURE_COLS,
        "coefficients": dict(zip(FEATURE_COLS, model.coef_[0].round(4).tolist())),
        "intercept": float(model.intercept_[0]),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "confusion_matrix": cm.tolist(),
        "class_balance": d["target"].value_counts().to_dict(),
        "mle_table": mle_table.round(6).to_dict(orient="records"),
        "lr_test": {"lr_stat": float(lr_stat), "df": df_lr, "p_value": float(lr_pvalue),
                    "mcfadden_r2": float(mcfadden_r2), "loglik_model": float(mle["loglik"]),
                    "loglik_null": float(ll_null)},
        "vif": {k: float(v) for k, v in vif.items()},
        "walk_forward": wf_df.round(6).to_dict(orient="records"),
        "walk_forward_mean_auc": float(wf_df["roc_auc"].mean()),
        "walk_forward_std_auc": float(wf_df["roc_auc"].std()),
        "best_C": float(grid.best_params_["C"]),
        "best_C_cv_auc": float(grid.best_score_),
        "threshold_best": {k: float(v) for k, v in best_t.to_dict().items()},
        "brier_score": float(brier),
        "bootstrap_auc_ci": {"point": float(metrics["roc_auc"]), "lo": float(ci_lo), "hi": float(ci_hi),
                              "t_stat": float(t_stat_auc), "p_value": float(p_auc)},
        "permutation_importance": {f: {"mean": float(m), "std": float(s)} for f, m, s in
                                    zip(FEATURE_COLS, perm.importances_mean, perm.importances_std)},
        "multi_ticker": mt_df.round(6).to_dict(orient="records"),
        "multi_ticker_summary": {
            "n_tickers": int(len(mt_df)),
            "mean_auc": float(mt_df["roc_auc"].mean()), "std_auc": float(mt_df["roc_auc"].std()),
            "min_auc": float(mt_df["roc_auc"].min()), "max_auc": float(mt_df["roc_auc"].max()),
            "mean_accuracy": float(mt_df["accuracy"].mean()),
            "t_stat": float(t_stat_mt), "p_value": float(p_mt),
            "wilcoxon_stat": float(w_stat_mt), "wilcoxon_p": float(p_w_mt),
            "n_above_random": int((mt_df["roc_auc"] > 0.5).sum()),
        },
    }
    with open("outputs/model_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Ringkasan lengkap disimpan ke outputs/model_summary.json")
 
 
if __name__ == "__main__":
    main()
