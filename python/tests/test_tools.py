"""Alat umum (klasifikasi & clustering) — muatan perhitungan manual harus identik dengan pustaka."""
import numpy as np
import pytest

from dmx import classify, cluster, eda
from dmx import t07 as m07, t08 as m08

FEATS = ["video_duration_sec", "caption_word_count", "hashtag_count", "post_hour", "niche"]


def test_classify_logreg_manual(t07):
    r = classify.run(t07, "virality_tier", FEATS, "logreg")
    m = r["manual"]
    z = m["intercept"] + float(np.dot(m["coef"], m["x"]))
    assert abs(1 / (1 + np.exp(-z)) - m["proba_sklearn"][1]) < 1e-8
    assert 0 <= r["metrics"]["accuracy"] <= 1 and r["n_test"] > 0


def test_classify_gnb_manual(t07):
    from dmx.t06 import manual_posterior
    m = classify.run(t07, "virality_tier", FEATS, "gnb", scaler="none")["manual"]
    post = manual_posterior({"x": m["x"], "theta": m["theta"], "var": m["var"], "prior": m["prior"]})["posterior"]
    assert np.max(np.abs(post - m["proba_sklearn"])) < 1e-8


def test_classify_knn_manual(t07):
    m = classify.run(t07, "virality_tier", FEATS, "knn", {"k": 5, "metric": "euclidean", "weights": "distance"})["manual"]
    r = m07.manual_knn(m["x"], m["train"], m["train_y"], m["k"], m["metric"], m["weights"])
    assert np.max(np.abs(r["proba"] - m["proba_sklearn"])) < 1e-8


def test_classify_tree_path(t07):
    m = classify.run(t07, "virality_tier", FEATS, "tree", {"max_depth": 3})["manual"]
    assert np.max(np.abs(np.array(m["leaf_dist"]) - m["proba_sklearn"])) < 1e-8


def test_cluster_matches_t08(t08):
    from dmx import common
    common.set_frame("t08_feats", m08.prepare(t08)["data"])
    f = cluster.fit("t08_feats", m08.CORE, 3, n_init=100)
    assert sorted(f["sizes"].tolist()) == [23, 41, 52]
    assert f["inertia"] == pytest.approx(456.4497, abs=5e-4)
    s = cluster.lloyd_setup("t08_feats", m08.CORE, 3)
    lab, C, tr = m08.lloyd(s["X"], s["init_centers"])
    assert np.array_equal(lab, s["sk_labels"]) and abs(tr[-1] - s["sk_inertia"]) < 1e-6


def test_simple_pair(t03):
    from dmx import t03 as m03
    p = eda.simple_pair(t03, "rd_spend_usd_bn", "revenue_usd_bn", 10)
    x, y = np.asarray(p["x"]), np.asarray(p["y"])
    n = len(x)
    b = (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x * x).sum() - x.sum() ** 2)
    assert abs(b - p["b_sklearn"]) < 1e-8
    assert eda.columns(t03)["n_rows"] == 617
