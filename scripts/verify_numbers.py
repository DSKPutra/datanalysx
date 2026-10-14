"""Bangkitkan docs/verifikasi-angka.md: angka Hasil Laporan → sumber → hasil reproduksi dmx.
Jalankan dengan versi pustaka requirements-test.txt. Studi kasus 04 memerlukan .localdata/AAPL.csv."""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from dmx import common, t01, t02, t03, t04, t05, t06, t07, t08  # noqa: E402
S = ROOT / "public/data/samples"
for k, f in [("t01", "t01_nvidia_stock_1999_2026.csv"), ("t02", "t02_llm_agent_failure_benchmark.csv"), ("t03", "t03_chip_companies_financials.csv"),
             ("t05", "t05_trending_topics_2026_synthetic.csv"), ("t06", "t06_indonesia_multiasset_macro_2010_2026.csv"),
             ("t07", "t07_instagram_reels_knn_clean.csv"), ("t08", "t08_aapl_2026_stock_data.csv")]:
    common.load_csv(k, (S / f).read_text())
rows = []
def add(tugas, nama, lap, sumber, live, tol):
    ok = live is not None and abs(float(lap) - float(live)) <= tol
    rows.append((tugas, nama, lap, sumber, "–" if live is None else f"{live:.6g}", "✓ cocok" if ok else ("⚠ beda" if live is not None else "tidak direproduksi")))
r = t01.run("t01")
add("01", "MAE", 0.0047, "Tabel 3.3", r["metrics"]["MAE"], 5e-5); add("01", "RMSE", 0.0101, "Tabel 3.3", r["metrics"]["RMSE"], 5e-5)
add("01", "R²", 0.999994, "Tabel 3.3", r["metrics"]["R2"], 5e-7); add("01", "Intersep", -8.177877961923485e-05, "Lampiran B", r["intercept"], 1e-7)
add("01", "Koef. High", 0.475677, "Tabel 3.7", r["coef"]["high"], 5e-6); add("01", "Koef. Low", 0.512563, "Tabel 3.7", r["coef"]["low"], 5e-6)
add("01", "n latih", 5663, "Subbab 3.2", r["n_train"], 0); add("01", "n uji", 1416, "Subbab 3.2", r["n_test"], 0)
r = t02.run("t02")
for nm, key, v in [("LR MAE", "MAE", 0.624560), ("LR RMSE", "RMSE", 0.772797), ("LR R²", "R2", 0.139403)]: add("02", nm, v, "Tabel 3.4", r["linear"]["metrics"][key], 5e-7)
for nm, key, v in [("RF MAE", "MAE", 0.198289), ("RF RMSE", "RMSE", 0.399665), ("RF R²", "R2", 0.769824)]: add("02", nm, v, "Tabel 3.4", r["rf"]["metrics"][key], 5e-7)
add("02", "Intersep LR", 1.637357, "Tabel 3.2", r["linear"]["intercept"], 5e-7)
f = t03.fit_models("t03")
add("03", "n analisis", 589, "Tabel 3", f["n_clean"], 0); add("03", "a sederhana", 5.324, "Tabel 5", f["simple"]["coef"][0], 5e-4); add("03", "b sederhana", 4.726, "Tabel 5", f["simple"]["coef"][1], 5e-4)
add("03", "R² sederhana", 0.8234, "Tabel 5", f["simple"]["r2"], 5e-5); add("03", "R² log-log", 0.8776, "Subbab 4.2.2", f["simple_log"]["r2"], 5e-5)
add("03", "R² berganda", 0.9691, "Tabel 11", f["multiple"]["r2"], 5e-5); add("03", "p margin", 0.358, "Tabel 9", f["multiple"]["p"][3], 5e-4)
add("03", "R² berganda log", 0.9884, "Tabel 11", f["multiple_log"]["r2"], 5e-5); add("03", "Elastisitas efek tetap", 0.9991, "Tabel 16", f["fixed_effect"]["slope"], 5e-5)
cv = {x["model"]: x for x in t03.cv_compare("t03")["rows"]}
for nm, rm, ma, r2 in [("Constant (rata-rata)", 21.747, 12.915, -0.002), ("Linear – sederhana", 9.494, 5.243, 0.809), ("Linear – berganda", 3.837, 2.503, 0.969), ("Linear – berganda (log)", 5.226, 1.767, 0.942), ("kNN (k=5)", 7.534, 1.998, 0.880), ("SGD", 3.837, 2.505, 0.969), ("SVM (RBF)", 10.264, 2.256, 0.777), ("Random Forest", 4.912, 0.949, 0.949)]:
    add("03", f"CV RMSE {nm}", rm, "Tabel 13", cv[nm]["RMSE"], 5e-4); add("03", f"CV MAE {nm}", ma, "Tabel 13", cv[nm]["MAE"], 5e-4); add("03", f"CV R² {nm}", r2, "Tabel 13", cv[nm]["R2"], 5e-4)
for nm, v in zip(["Prediksi sederhana", "Prediksi sederhana log", "Prediksi berganda", "Prediksi berganda log"], [52.58, 65.23, 52.62, 68.73]):
    add("03", nm, v, "Tabel 19", list(t03.predict("t03", 10, 8, 30).values())[["Prediksi sederhana", "Prediksi sederhana log", "Prediksi berganda", "Prediksi berganda log"].index(nm)], 5e-3)
aapl = ROOT / ".localdata/AAPL.csv"
if aapl.exists():
    common.load_csv("aapl", aapl.read_text())
    r = t04.run_main("aapl"); inf = t04.inference("aapl"); wf = t04.walk_forward("aapl"); b = t04.bootstrap_auc("aapl", n_boot=2000); tc = t04.tune_C("aapl")
    add("04", "Observasi", 4183, "Subbab 3.3", r["n_total"], 0); add("04", "Baris bersih", 4184, "Subbab 3.2", r["n_clean"], 0)
    for nm, k, v in [("Akurasi", "accuracy", 0.5137), ("Balanced acc.", "balanced_accuracy", 0.4918), ("Presisi", "precision", 0.5393), ("Recall", "recall", 0.7368), ("F1", "f1", 0.6228), ("ROC-AUC", "roc_auc", 0.484), ("MCC", "mcc", -0.0188), ("Log-loss", "log_loss", 0.6955), ("Brier", "brier", 0.2511)]:
        add("04", nm, v, "Tabel 8", r["metrics"][k], 5e-4 if k == "roc_auc" else 5e-5)
    add("04", "LR-test p", 0.4632, "Subbab 4.4", inf["lr_p"], 5e-5); add("04", "McFadden R²", 0.0021, "Subbab 4.4", inf["mcfadden_r2"], 5e-5)
    add("04", "Walk-forward AUC", 0.4925, "Tabel 11", wf["mean_auc"], 5e-5); add("04", "C optimal", 0.001, "Subbab 4.7", tc["best_C"], 1e-9)
    add("04", "Bootstrap CI bawah", 0.4430, "Subbab 4.11", b["lo"], 5e-5); add("04", "Bootstrap CI atas", 0.5246, "Subbab 4.11", b["hi"], 5e-5)
    add("04", "F1 threshold 0,05", 0.7053, "Subbab 4.8", r["f1_best"]["f1"], 5e-5)
    keys = []
    for tk in ["AAPL","MSFT","AMZN","GOOGL","NVDA","ADBE","CSCO","INTC","AMD","QCOM","TXN","AMAT","PYPL","SBUX","GILD","COST","PEP","CMCSA","MU","ISRG"]:
        common.load_csv(tk, (ROOT / f".localdata/{tk}.csv").read_text()); keys.append(tk)
    cs = t04.cross_stock(keys)["summary"]
    add("04", "Lintas saham rata-rata AUC", 0.5156, "Subbab 4.13", cs["mean_auc"], 5e-5); add("04", "Lintas saham uji-t p", 0.0075, "Subbab 4.13", cs["p_value"], 5e-5)
    add("04", "Lintas saham Wilcoxon p", 0.0121, "Subbab 4.13", cs.get("wilcoxon_p"), 5e-5); add("04", "Lintas saham n > 0,5", 14, "Subbab 4.13", cs["n_above"], 0)
r1 = t05.fit_tree("t05", 1, "gini", 100, 2, 5); r2 = t05.fit_tree("t05", 2, "entropy", 3, 1); ms = t05.manual_split("t05", "for")
add("05", "S1 default simpul", 1289, "Tabel 14", r1["n_nodes"], 0); add("05", "S1 default akurasi latih", 0.776, "Tabel 14", r1["train_acc"], 5e-4)
add("05", "S1 default akurasi uji", 0.200, "Tabel 14", r1["test_acc"], 5e-4); add("05", "S2 akurasi uji", 1.0, "Tabel 14", r2["test_acc"], 0)
add("05", "Entropy akar", 2.3190, "Tabel 8", ms["H_root"], 5e-5); add("05", "IG kata for", 0.9546, "Subbab 4.3.2", ms["info_gain"], 5e-5)
r = t06.run("t06", "report"); r0 = t06.run("t06", "report_none")
for nm, k, v in [("Akurasi", "accuracy", 0.5187), ("Presisi", "precision", 0.4955), ("Recall", "recall", 0.6806), ("F1", "f1", 0.5735), ("ROC-AUC", "roc_auc", 0.5689), ("Brier", "brier", 0.2557)]:
    add("06", nm, v, "Tabel 4.5", r["metrics"][k], 5e-5)
add("06", "Baseline mayoritas", 0.4755, "Tabel 4.5", r["baseline_majority"], 5e-5); add("06", "CV AUC", 0.5994, "Subbab 4.9", r["cv_auc_mean"], 5e-5)
add("06", "Akurasi tanpa outlier handling", 0.4848, "Abstrak", r0["metrics"]["accuracy"], 5e-5); add("06", "Brier tanpa outlier handling", 0.461, "Abstrak", r0["metrics"]["brier"], 5e-4)
p = t06.model_params("t06"); add("06", "Posterior Turun 21-02-2023", 0.9024, "Tabel 4.4", float(t06.manual_posterior(p)["posterior"][0]), 5e-5)
e = t07.evaluate("t07"); summ = json.loads((ROOT / "public/data/studi-kasus-07.json").read_text())
add("07", "CV GridSearch", summ["gridsearch"]["best_cv_acc"], "model_summary.gridsearch", e["cv_acc"], 5e-5)
for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]: add("07", f"Uji {k}", summ["knn_test"][k], "model_summary.knn_test", e["test"][k], 5e-5)
f8 = t08.fit("t08"); s8 = json.loads((ROOT / "public/data/studi-kasus-08.json").read_text())["final_model"]
for k in ["inertia", "silhouette", "dbi", "ch"]: add("08", k, s8[k], "model_summary.final_model", f8[k], 5e-4)
for i, c in enumerate(["C1", "C2", "C3"]): add("08", f"Ukuran {c}", s8["sizes"][c], "model_summary.final_model", int(f8["sizes"][i]), 0)
lines = ["# Verifikasi angka “Hasil Laporan”", "",
         "Dibangkitkan oleh `scripts/verify_numbers.py` (Python 3.13, numpy 2.2.5, pandas 2.3.3, scikit-learn 1.7.0, scipy 1.14.1 — sama dengan Pyodide 0.29.5).",
         "Kolom *Laporan* = angka yang ditampilkan di tab Hasil Laporan (dari laporan / model_summary.json); *Reproduksi* = hasil menjalankan modul `python/dmx` pada dataset asli.", "",
         "| Studi kasus | Angka | Laporan | Sumber | Reproduksi | Status |", "|---|---|---|---|---|---|"]
lines += [f"| {a} | {b} | {c} | {d} | {e_} | {f_} |" for a, b, c, d, e_, f_ in rows]
n_ok = sum(1 for r in rows if r[5].startswith("✓"))
lines += ["", f"**Ringkasan:** {n_ok} dari {len(rows)} angka cocok dalam toleransi pembulatan.", "",
          "## Catatan perbedaan (dilaporkan, tidak diganti diam-diam)",
          "- **Studi kasus 01:** scikit-learn ≥ 1.8 menghasilkan solusi `LinearRegression` berbeda (MAE 1,34) karena kolom Volume berorde 10⁸; numpy `lstsq` dan scikit-learn 1.7.0 mereproduksi laporan. Aplikasi & CI dikunci ke 1.7.0.",
          "- **Studi kasus 02:** berkas data identik dengan *LLM Agent Failure Analysis Benchmark Dataset* (Apache 2.0), bukan *AI Economy Pulse* yang dikutip laporan.",
          "- **Studi kasus 03:** CV Random Forest: RMSE 4,906 / MAE 0,946 (laporan 4,912 / 0,949) — perbedaan implementasi pohon antarversi scikit-learn; R² dan tujuh model lain cocok.",
          "- **Studi kasus 04:** laporan menyebut 4.184 baris bersih; kode menghasilkan 4.203 baris bersih → 4.183 observasi (cocok). Baris terakhir pada skrip asli memperoleh target 0 walau Close H+1 tak diketahui; aplikasi menyediakan opsi membuangnya.",
          "- **Studi kasus 06:** batas IQR laporan dihitung dari seluruh periode (termasuk data uji) dan baris terakhir ikut dengan target 0; mode ‘benar’ (batas dari data latih) memberi akurasi 52,05%.",
          "- **Studi kasus 08:** pada pemilihan K, DBI terbaik live dapat berbeda dari laporan (K = 10) bila n_init diperkecil demi kecepatan browser.",
          "- **Studi kasus 09:** laporan belum tersedia — tidak ada angka Hasil Laporan.", "",
          "Studi kasus 04 direproduksi dengan data NASDAQ lokal (tidak diredistribusi). Studi kasus 07–08 dibandingkan dengan model_summary.json resmi."]
(ROOT / "docs/verifikasi-angka.md").write_text("\n".join(lines) + "\n")
print(f"{n_ok}/{len(rows)} cocok"); [print(r) for r in rows if not r[5].startswith("✓")]
