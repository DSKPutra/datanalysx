"""Bangkitkan fixture JSON dari scikit-learn untuk menguji implementasi manual TypeScript (Vitest).
Jalankan: python scripts/make_fixtures.py  (memakai versi pustaka requirements-test.txt)."""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from dmx import common, t01, t03, t04, t05, t06, t07, t08, t09  # noqa: E402
S = ROOT / "public/data/samples"
OUT = ROOT / "src/lib/manual/__fixtures__"
OUT.mkdir(parents=True, exist_ok=True)
for k, f in [("t01", "t01_nvidia_stock_1999_2026.csv"), ("t03", "t03_chip_companies_financials.csv"),
             ("t05", "t05_trending_topics_2026_synthetic.csv"), ("t06", "t06_indonesia_multiasset_macro_2010_2026.csv"),
             ("t07", "t07_instagram_reels_knn_clean.csv"), ("t08", "t08_aapl_2026_stock_data.csv")]:
    common.load_csv(k, (S / f).read_text())
dump = lambda name, obj: (OUT / f"{name}.json").write_text(common.to_json(obj))
m = t01.manual_ols("t01", n_rows=20)
dump("ols", {"X": m["X"], "y": m["y"], "pred_sklearn": m["pred_sklearn"], "beta_sklearn": m["beta_sklearn"]})
s = t03.manual_simple("t03")
dump("simple", {"x": [r["x"] for r in s["rows"]], "y": [r["y"] for r in s["rows"]], "a": s["a_sklearn"], "b": s["b_sklearn"]})
p = t06.model_params("t06", "2023-02-21", "report")
dump("gnb", {"x": p["x"], "theta": p["theta"], "var": p["var"], "prior": p["prior"], "proba": p["proba_sklearn"]})
n = t07.neighbors("t07")
dump("knn", {"query": n["query"], "train": n["train"], "y": n["train_y"], "proba": n["sk_proba"], "neighbors": n["sk_neighbors"]})
e = t08.manual_example("t08")
dump("kmeans", {"points": e["points"], "init": e["init_centers"], "labels": e["sk_labels"], "centers": e["sk_centers"], "inertia": e["sk_inertia"]})
ms = t05.manual_split("t05", "for")
dump("entropy", {"left": ms["left_counts"], "right": ms["right_counts"], "H_root": ms["sklearn_H_root"], "H_after": ms["sklearn_H_after"]})
docs = ["makan gratis bagus", "gizi anak tidak bagus", "makan siang anak sekolah", "program gizi makan gratis", "anggaran makan tidak jelas"]
dump("tfidf", {"docs": docs, "smooth": t09.tfidf_sklearn(docs, True), "classic": t09.tfidf_sklearn(docs, False)})
print("fixtures ditulis ke", OUT)
