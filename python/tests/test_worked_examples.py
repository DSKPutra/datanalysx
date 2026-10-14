"""Setiap perhitungan manual harus identik dengan pustaka (toleransi < 1e-8)."""
import numpy as np

from dmx import t01 as m01, t02 as m02, t03 as m03, t05 as m05, t06 as m06, t07 as m07, t08 as m08, t09 as m09

TOL = 1e-8


def test_t01_normal_equation(t01):
    m = m01.manual_ols(t01, n_rows=20)
    assert m["max_abs_diff_pred"] < TOL


def test_t02_linear_and_forest_average(t02):
    assert m02.manual_predict(t02, 3)["max_abs_diff"] < TOL
    assert m02.manual_metrics(t02)["max_abs_diff"] < TOL


def test_t03_simple_and_multiple(t03):
    assert m03.manual_simple(t03)["max_abs_diff"] < TOL
    assert m03.manual_multiple(t03)["max_abs_diff"] < 1e-7  # inverse eksplisit (XᵀX)⁻¹ sesuai rumus materi


def test_t05_entropy_information_gain(t05):
    assert m05.manual_split(t05, "for")["max_abs_diff"] < TOL


def test_t06_gaussian_nb_posterior(t06):
    p = m06.model_params(t06, "2023-02-21", "report")
    post = m06.manual_posterior(p)["posterior"]
    assert np.max(np.abs(post - p["proba_sklearn"])) < TOL
    assert round(float(post[0]) * 100, 2) == 90.24


def test_t07_knn_votes(t07):
    n = m07.neighbors(t07)
    m = m07.manual_knn(n["query"], n["train"], n["train_y"], 5, "cosine", "distance")
    assert set(m["neighbors"]) == set(n["sk_neighbors"])
    assert np.max(np.abs(m["proba"] - n["sk_proba"])) < TOL
    assert n["row_index"] == 34


def test_t08_lloyd_manual(t08):
    e = m08.manual_example(t08)
    lab, C, trace = m08.lloyd(e["points"], e["init_centers"])
    assert np.array_equal(lab, e["sk_labels"])
    assert np.max(np.abs(C - e["sk_centers"])) < TOL
    assert abs(trace[-1] - e["sk_inertia"]) < TOL
    s = m08.lloyd_setup(t08)
    lab2, C2, _ = m08.lloyd(s["X"], s["init_centers"])
    assert np.array_equal(lab2, s["sk_labels"]) and np.max(np.abs(C2 - s["sk_centers"])) < TOL


def test_t09_tfidf_manual_matches_sklearn():
    docs = ["makan gratis bagus", "gizi anak tidak bagus", "makan siang anak sekolah",
            "program gizi makan gratis", "anggaran makan tidak jelas"]
    for smooth in (True, False):
        a, b = m09.tfidf_manual(docs, smooth_idf=smooth), m09.tfidf_sklearn(docs, smooth_idf=smooth)
        assert a["vocab"] == b["vocab"]
        assert np.max(np.abs(a["matrix"] - b["matrix"])) < TOL
        assert np.max(np.abs(a["idf"] - b["idf"])) < TOL


def test_t09_negation_kept():
    out = m09.preprocess("Program ini tidak bagus dan belum merata", keep_negation=True, do_stem=False, do_normalize=False)
    assert "tidak" in out["tokens"] and "belum" in out["tokens"]
    out2 = m09.preprocess("Program ini tidak bagus dan belum merata", keep_negation=False, do_stem=False, do_normalize=False)
    assert "tidak" not in out2["tokens"]


def test_t09_privacy_helpers():
    assert m09.pseudonym("@someone").startswith("user_") and "someone" not in m09.pseudonym("@someone")
    assert m09.censor("dasar bangsat") == "dasar b******"
