"""Reproduksi angka "Hasil Laporan" dari dataset asli memakai modul browser (dmx).

Toleransi = pembulatan yang dipakai di laporan. Versi pustaka dikunci sama dengan Pyodide
0.29.5 (requirements-test.txt) karena scikit-learn ≥ 1.8 memberi hasil numerik berbeda
untuk LinearRegression dengan kolom Volume berorde 10^8 (lihat docs/verifikasi-angka.md).
"""
import pytest

from dmx import t01 as m01, t02 as m02, t03 as m03, t04 as m04, t05 as m05, t06 as m06, t07 as m07, t08 as m08


def test_t01_linear_regression_nvidia(t01):
    r = m01.run(t01)
    assert (r["n_train"], r["n_test"]) == (5663, 1416)
    assert r["metrics"]["MAE"] == pytest.approx(0.0047, abs=5e-5)
    assert r["metrics"]["RMSE"] == pytest.approx(0.0101, abs=5e-5)
    assert r["metrics"]["R2"] == pytest.approx(0.999994, abs=5e-7)
    assert r["intercept"] == pytest.approx(-0.0000818, abs=5e-7)
    assert r["coef"]["high"] == pytest.approx(0.475677, abs=5e-6)
    assert r["coef"]["low"] == pytest.approx(0.512563, abs=5e-6)


def test_t02_linear_vs_rf(t02):
    r = m02.run(t02)
    assert (r["n_train"], r["n_test"]) == (1200, 300)
    assert r["linear"]["metrics"]["MAE"] == pytest.approx(0.624560, abs=5e-7)
    assert r["linear"]["metrics"]["R2"] == pytest.approx(0.139403, abs=5e-7)
    assert r["linear"]["intercept"] == pytest.approx(1.637357, abs=5e-7)
    assert r["rf"]["metrics"]["MAE"] == pytest.approx(0.198289, abs=5e-7)
    assert r["rf"]["metrics"]["RMSE"] == pytest.approx(0.399665, abs=5e-7)
    assert r["rf"]["metrics"]["R2"] == pytest.approx(0.769824, abs=5e-7)


def test_t03_ols_and_cv(t03):
    f = m03.fit_models(t03)
    assert f["n_clean"] == 589 and f["n_companies"] == 39 and f["n_dropped_zero"] == 28
    assert f["simple"]["coef"][0] == pytest.approx(5.324, abs=5e-4)
    assert f["simple"]["coef"][1] == pytest.approx(4.726, abs=5e-4)
    assert f["simple"]["r2"] == pytest.approx(0.8234, abs=5e-5)
    assert f["simple_log"]["r2"] == pytest.approx(0.8776, abs=5e-5)
    assert f["multiple"]["r2"] == pytest.approx(0.9691, abs=5e-5)
    assert f["multiple"]["p"][3] == pytest.approx(0.358, abs=5e-4)
    assert f["multiple_log"]["r2"] == pytest.approx(0.9884, abs=5e-5)
    assert f["fixed_effect"]["slope"] == pytest.approx(0.9991, abs=5e-5)
    assert f["fixed_effect"]["r2_within"] == pytest.approx(0.9998, abs=5e-5)
    cv = {r["model"]: r for r in m03.cv_compare(t03)["rows"]}
    assert cv["Linear – berganda"]["RMSE"] == pytest.approx(3.837, abs=5e-4)
    # Random Forest: laporan 0,949; scikit-learn 1.7 (Pyodide) memberi 0,946 karena perbedaan
    # implementasi pohon antarversi. Dicatat di docs/verifikasi-angka.md.
    assert cv["Random Forest"]["MAE"] == pytest.approx(0.949, abs=5e-3)
    assert cv["Constant (rata-rata)"]["RMSE"] == pytest.approx(21.747, abs=5e-4)
    p = m03.predict(t03, 10, 8, 30)
    assert [round(v, 2) for v in p.values()] == [52.58, 65.23, 52.62, 68.73]


def test_t04_logistic_regression_aapl(aapl):
    r = m04.run_main(aapl)
    assert (r["n_total"], r["n_train"], r["n_test"]) == (4183, 3346, 837)
    assert r["metrics"]["confusion_matrix"] == [[94, 287], [120, 336]]
    assert r["metrics"]["roc_auc"] == pytest.approx(0.484, abs=5e-4)
    assert r["metrics"]["brier"] == pytest.approx(0.2511, abs=5e-5)
    assert r["f1_best"]["threshold"] == pytest.approx(0.05) and r["f1_best"]["f1"] == pytest.approx(0.7053, abs=5e-5)
    inf = m04.inference(aapl)
    assert inf["lr_p"] == pytest.approx(0.4632, abs=5e-5)
    assert inf["mcfadden_r2"] == pytest.approx(0.0021, abs=5e-5)
    assert m04.walk_forward(aapl)["mean_auc"] == pytest.approx(0.4925, abs=5e-5)
    b = m04.bootstrap_auc(aapl, n_boot=2000)
    assert (round(b["lo"], 4), round(b["hi"], 4)) == (0.4430, 0.5246)


def test_t05_decision_tree(t05):
    r1 = m05.fit_tree(t05, 1, "gini", 100, 2, 5)
    assert r1["n_nodes"] == 1289 and r1["depth"] == 23
    assert r1["train_acc"] == pytest.approx(0.776, abs=5e-4) and r1["test_acc"] == pytest.approx(0.200, abs=5e-4)
    r2 = m05.fit_tree(t05, 2, "entropy", 3, 1)
    assert r2["test_acc"] == 1.0 and r2["n_nodes"] == 9
    m = m05.manual_split(t05, "for")
    assert m["H_root"] == pytest.approx(2.3190, abs=5e-5) and m["info_gain"] == pytest.approx(0.9546, abs=5e-5)


def test_t06_naive_bayes(t06):
    r = m06.run(t06, "report")
    assert (r["n_model"], r["n_train"], r["n_test"]) == (4276, 3420, 856)
    assert r["metrics"]["accuracy"] == pytest.approx(0.5187, abs=5e-5)
    assert r["metrics"]["brier"] == pytest.approx(0.256, abs=5e-4)
    assert r["metrics"]["roc_auc"] == pytest.approx(0.5689, abs=5e-5)
    assert r["cv_auc_mean"] == pytest.approx(0.5994, abs=5e-5)
    r0 = m06.run(t06, "report_none")
    assert r0["metrics"]["accuracy"] == pytest.approx(0.4848, abs=5e-5)
    assert r0["metrics"]["brier"] == pytest.approx(0.461, abs=5e-4)


def test_t07_knn(t07):
    e = m07.evaluate(t07)
    assert e["cv_acc"] == pytest.approx(0.8846, abs=5e-5)
    assert e["test"]["accuracy"] == pytest.approx(0.7727, abs=5e-5)
    assert e["test"]["roc_auc"] == pytest.approx(0.6667, abs=5e-5)
    assert e["test"]["confusion_matrix"] == [[9, 3], [2, 8]]
    acc = {sc: m07.evaluate(t07, 5, "minkowski", "uniform", sc, n_repeats=10)["cv_acc"] for sc in ["none", "standard", "minmax"]}
    assert [round(acc[s], 4) for s in ["none", "standard", "minmax"]] == [0.7825, 0.7998, 0.8711]


def test_t08_kmeans(t08):
    f = m08.fit(t08)
    assert list(f["sizes"]) == [41, 52, 23]
    assert f["inertia"] == pytest.approx(456.4497, abs=5e-5)
    assert f["silhouette"] == pytest.approx(0.2104, abs=5e-5)
    assert f["dbi"] == pytest.approx(1.5521, abs=5e-5)
    assert f["ch"] == pytest.approx(29.6519, abs=5e-5)
