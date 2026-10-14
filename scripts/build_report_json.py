"""Transkripsi angka "Hasil Laporan" ke public/data/studi-kasus-XX.json.
Studi kasus 01–06: disalin dari tabel laporan (field "sumber" menunjuk tabel/subbab).
Studi kasus 07–08: diambil dari model_summary.json resmi. Studi kasus 09: laporan belum tersedia."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "public/data"

def w(name, obj):
    (D / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1))

w("studi-kasus-01.json", {
 "judul": "Penerapan Algoritma Linear Regression pada NVIDIA Stock Data (1999–2026)",
 "sumber": "Laporan studi kasus 01",
 "dataset": {"nama": "NVIDIA Stock Data (1999–2026)", "baris": 7079, "kolom": 16, "periode": "22 Jan 1999 – 11 Mar 2026",
   "duplikat": 0, "missing_struktural": {"stock_split": 7073, "key_event": 6826}, "sumber": "Tabel 2.4; Lampiran B"},
 "split": {"latih": 5663, "uji": 1416, "metode": "train_test_split acak 80:20, random_state=42", "sumber": "Subbab 3.2"},
 "metrik": {"MAE": 0.0047, "RMSE": 0.0101, "R2": 0.999994, "MAE_penuh": 0.004668613695335099,
   "RMSE_penuh": 0.01005357433891022, "R2_penuh": 0.9999935789811613, "sumber": "Tabel 3.3; Lampiran B"},
 "intercept": -8.177877961923485e-05,
 "koefisien": {"Open": 0.006963, "High": 0.475677, "Low": 0.512563, "Volume": -1.6e-12, "SMA 50": 0.005132, "RSI 14": 0.0000039, "sumber": "Tabel 3.7"},
 "korelasi_close": {"Open": 0.999985, "High": 0.999981, "Low": 0.999982, "Volume": 0.783967, "SMA 50": 0.998605, "RSI 14": 0.248561, "sumber": "Tabel 3.4"},
 "statistik": [["Open",2.95,3.84,0.03,20.81],["High",2.98,3.88,0.03,20.96],["Low",2.92,3.80,0.03,20.63],["Close",2.95,3.84,0.03,20.73],
   ["SMA 50",2.95,3.84,0.03,22.00],["RSI 14",56.36,10.15,38.00,78.00]],
 "statistik_sumber": "Tabel 2.5"})

w("studi-kasus-02.json", {
 "judul": "Pemodelan dan Perbandingan Algoritma Regresi Linear serta Non-Linear",
 "sumber": "Laporan studi kasus 02",
 "dataset": {"nama_di_laporan": "AI Economy Pulse Dataset Tahun 2022-2026", "baris": 1500, "kolom": 10, "missing": 0, "duplikat": 0, "sumber": "Tabel 2.1"},
 "statistik": [["difficulty_score",1.9,0.700233,1,3],["prompt_length",22.896,7.549604,10,59],["context_length",19.112667,8.379904,3,48],["severity_score",2.401333,0.844239,1,4]],
 "statistik_sumber": "Tabel 2.4",
 "split": {"latih": 1200, "uji": 300, "sumber": "Subbab 3.1"},
 "model": [{"model": "Linear Regression", "MAE": 0.624560, "RMSE": 0.772797, "R2": 0.139403},
           {"model": "Random Forest (n_estimators=100)", "MAE": 0.198289, "RMSE": 0.399665, "R2": 0.769824}],
 "model_sumber": "Tabel 3.4; Lampiran C",
 "persamaan": {"intercept": 1.637357, "difficulty_score": -0.078486, "prompt_length": 0.007658, "context_length": 0.038742, "sumber": "Tabel 3.2"}})

w("studi-kasus-03.json", {
 "judul": "Analisis Regresi Linear Sederhana dan Berganda — Industri Semikonduktor (2010–2026)",
 "sumber": "Laporan studi kasus 03",
 "csv_ditinjau": [
   ["chip_companies_financials.csv","Keuangan tahunan 40 perusahaan chip, 2010–2026",617,"Layak: banyak observasi, beberapa variabel numerik kontinu"],
   ["ai_chip_market.csv","Spesifikasi & pengiriman chip AI, 2020–2026",120,"Kurang layak: ASP hampir konstan; pendapatan = unit × ASP (identitas)"],
   ["chip_prices.csv","Harga bulanan DRAM, NAND, HBM3, H100, B200",405,"Cocok untuk tren deret waktu, bukan regresi antarvariabel"],
   ["fab_capacity.csv","Kapasitas wafer bulanan per pabrik",313,"Kurang layak: hanya satu variabel numerik utama"],
   ["export_controls.csv","Peristiwa kebijakan kontrol ekspor",34,"Terlalu kecil dan bersifat kategorikal"]],
 "csv_sumber": "Tabel 1",
 "pembersihan": {"awal": 617, "nol_dibuang": 28, "analisis": 589, "perusahaan": 39, "sumber": "Tabel 3; Subbab 3.3"},
 "statistik": [["Pendapatan (miliar USD)",15.03,21.75,0.10,7.56,250.68,4.55],["Belanja R&D (miliar USD)",2.05,4.18,0.01,1.03,62.67,8.71],
   ["Capex (miliar USD)",3.27,6.13,0.01,1.09,50.96,3.77],["Margin operasi (%)",31.30,10.86,3.30,31.30,64.20,-0.01]],
 "statistik_sumber": "Tabel 4",
 "model": [
   {"model":"Sederhana – linear","persamaan":"Pendapatan′ = 5,324 + 4,726·R&D","R2":0.8234,"sumber":"Tabel 5"},
   {"model":"Sederhana – log-log","persamaan":"ln(Pendapatan)′ = 2,072 + 0,915·ln(R&D)","R2":0.8776,"sumber":"Tabel 7"},
   {"model":"Berganda – linear","persamaan":"Pendapatan′ = 2,725 + 3,848·R&D + 1,477·Capex − 0,013·Margin","R2":0.9691,"catatan":"Margin tidak signifikan (p = 0,358)","sumber":"Tabel 9, Tabel 11"},
   {"model":"Berganda – log-log","persamaan":"ln(Pendapatan)′ = 1,969 + 0,697·ln R&D + 0,271·ln Capex + 0,0031·Margin","R2":0.9884,"catatan":"VIF 1,47 / 1,47 / 1,02","sumber":"Tabel 10, Tabel 11"}],
 "efek_tetap": {"elastisitas": 0.9991, "se": 0.0006, "R2_within": 0.9998, "R2_total": 0.99994, "sumber": "Tabel 16"},
 "holdout": [["Sederhana (linear)",100.81,10.04,5.65,0.8561],["Sederhana (log-log)",110.35,10.50,4.31,0.8425],["Berganda (linear)",8.98,3.00,2.16,0.9872],["Berganda (log-log)",10.27,3.20,1.13,0.9853]],
 "holdout_sumber": "Tabel 12 (MSE, RMSE, MAE, R²; latih 471 / uji 118)",
 "cv": [["Constant (rata-rata)",472.92,21.747,12.915,-0.002],["Linear – sederhana",90.13,9.494,5.243,0.809],["Linear – berganda",14.72,3.837,2.503,0.969],
   ["Linear – berganda (log)",27.31,5.226,1.767,0.942],["kNN (k=5)",56.76,7.534,1.998,0.880],["SGD",14.72,3.837,2.505,0.969],["SVM (RBF)",105.34,10.264,2.256,0.777],["Random Forest",24.12,4.912,0.949,0.949]],
 "cv_sumber": "Tabel 13 (validasi silang 10-lipatan)",
 "asumsi": [["Sederhana – linear","< 0,001","< 0,001",0.298],["Sederhana – log-log","0,975","< 0,001",0.090],["Berganda – linear","< 0,001","< 0,001",0.246],["Berganda – log-log","< 0,001","< 0,001",0.161]],
 "asumsi_sumber": "Tabel 18 (Breusch–Pagan p, Shapiro–Wilk p, Durbin–Watson)",
 "prediksi_contoh": {"input": {"R&D": 10, "Capex": 8, "Margin": 30}, "hasil": {"Sederhana – linear": 52.58, "Sederhana – log-log": 65.23, "Berganda – linear": 52.62, "Berganda – log-log": 68.73}, "sumber": "Tabel 19"}})

w("studi-kasus-04.json", {
 "judul": "Regresi Logistik — Prediksi Arah Harga Saham (NASDAQ Daily OHLC, AAPL)",
 "sumber": "Laporan studi kasus 04",
 "dataset": {"baris_bersih_menurut_laporan": 4184, "observasi": 4183, "periode": "2010-02-02 – 2026-09-18", "naik": 2222, "turun_tetap": 1961, "sumber": "Subbab 3.2–3.3, 4.1"},
 "split": {"latih": 3346, "uji": 837, "sumber": "Tabel 6"},
 "confusion": {"TN": 94, "FP": 287, "FN": 120, "TP": 336, "sumber": "Tabel 7"},
 "metrik": {"akurasi": 0.5137, "balanced_accuracy": 0.4918, "presisi": 0.5393, "recall": 0.7368, "f1": 0.6228, "roc_auc": 0.484, "mcc": -0.0188, "log_loss": 0.6955, "sumber": "Tabel 8"},
 "mle": [["intercept",0.1115,0.0347,3.215,0.0013,1.118,1.044,1.197],["return_1d",-0.0716,0.0634,-1.129,0.2591,0.931,0.822,1.054],["return_2d",0.0193,0.0580,0.333,0.7392,1.019,0.910,1.142],
   ["return_5d",0.0491,0.0532,0.922,0.3566,1.050,0.946,1.166],["ma_gap_5_20",0.0133,0.0879,0.151,0.8797,1.013,0.853,1.204],["vol_10d",-0.0661,0.0704,-0.939,0.3475,0.936,0.815,1.074],
   ["vol_20d",0.0252,0.0669,0.376,0.7068,1.025,0.899,1.169],["hl_spread",0.0048,0.0415,0.116,0.9080,1.005,0.926,1.090],["oc_spread",-0.0164,0.0530,-0.310,0.7563,0.984,0.887,1.091],
   ["rsi_14",0.0405,0.0783,0.517,0.6051,1.041,0.893,1.214],["momentum_10",-0.0708,0.0487,-1.454,0.1460,0.932,0.847,1.025]],
 "mle_sumber": "Tabel 9",
 "lr_test": {"loglik_model": -2309.23, "loglik_null": -2314.10, "LR": 9.744, "df": 10, "p": 0.4632, "mcfadden": 0.0021, "sumber": "Subbab 4.4"},
 "vif": {"return_1d":3.32,"return_2d":2.79,"return_5d":2.36,"ma_gap_5_20":6.41,"vol_10d":4.11,"vol_20d":3.72,"hl_spread":1.43,"oc_spread":2.33,"rsi_14":5.09,"momentum_10":1.96,"sumber":"Tabel 10"},
 "walk_forward": [[1,697,697,0.4878,0.4829,0.6073],[2,1394,697,0.5079,0.5203,0.6332],[3,2091,697,0.5265,0.4993,0.5854],[4,2788,697,0.5079,0.4784,0.6347],[5,3485,697,0.5079,0.4815,0.6243]],
 "walk_forward_ringkas": {"mean_auc": 0.4925, "sd": 0.0175, "sumber": "Tabel 11"},
 "tuning": {"C_optimal": 0.001, "auc_cv": 0.4984, "sumber": "Subbab 4.7"},
 "threshold": {"threshold": 0.05, "f1": 0.7053, "recall": 1.0, "presisi": 0.545, "sumber": "Subbab 4.8"},
 "brier": 0.2511,
 "bootstrap": {"lo": 0.4430, "hi": 0.5246, "t": -36.093, "replikasi": 2000, "sumber": "Subbab 4.11"},
 "lintas_saham": [["INTC",4183,0.5448,0.5584,0.5134],["NVDA",4183,0.5472,0.5532,0.6081],["TXN",4183,0.5257,0.5498,0.6018],["COST",4182,0.5317,0.5362,0.6323],["CSCO",4183,0.5460,0.5341,0.6339],
   ["GILD",4183,0.5113,0.5255,0.5690],["GOOGL",4183,0.5352,0.5236,0.6375],["MU",4182,0.5185,0.5210,0.5446],["ISRG",4183,0.5221,0.5195,0.6540],["AMAT",4183,0.5293,0.5162,0.6083],
   ["CMCSA",4183,0.4875,0.5127,0.6046],["QCOM",4183,0.5257,0.5120,0.6563],["AMD",4183,0.5006,0.5116,0.4336],["AMZN",4183,0.5221,0.5062,0.6716],["SBUX",4183,0.4922,0.4997,0.6104],
   ["PEP",4183,0.4791,0.4972,0.5978],["MSFT",4183,0.5090,0.4970,0.6353],["AAPL",4183,0.5137,0.4844,0.6228],["PYPL",2799,0.5018,0.4772,0.6098],["ADBE",4183,0.4994,0.4771,0.6295]],
 "lintas_saham_ringkas": {"mean_auc": 0.5156, "sd": 0.0234, "min": 0.4771, "max": 0.5584, "n_di_atas_05": 14, "t": 2.991, "p_t": 0.0075, "p_wilcoxon": 0.0121, "sumber": "Tabel 12; Subbab 4.13"}})

w("studi-kasus-05.json", {
 "judul": "Pemodelan Decision Tree untuk Klasifikasi Kategori Topik pada Trending Topics 2026 Dataset",
 "sumber": "Laporan studi kasus 05",
 "dataset": {"baris": 2500, "kolom": 14, "periode": "1–28 Februari 2026", "kelas": {"AI & Tech":507,"Economy":492,"Entertainment":511,"Global Events":544,"Sports":446}, "mayoritas": 0.218, "sumber": "Tabel 2, 3, 8"},
 "split": {"latih": 2000, "uji": 500, "sumber": "Subbab 3.5"},
 "skenario1": [["Constant (mayoritas)",0.500,0.218,0.078,0.047,0.218,0.002],["Tree (default Orange)",0.496,0.198,0.196,0.200,0.198,0.018],["Tree (Entropy, default)",0.491,0.183,0.182,0.184,0.183,0.028],
   ["Tree (tuning/pruning)",0.496,0.212,0.153,0.166,0.212,0.022],["Random Forest",0.499,0.198,0.196,0.198,0.198,0.011],["Logistic Regression",0.506,0.215,0.194,0.201,0.215,0.010]],
 "skenario1_sumber": "Tabel 12 (AUC, CA, F1, Presisi, Recall, SD CA)",
 "skenario2": [["Constant (mayoritas)",0.500,0.218,0.078,0.047,0.218,0.002],["Tree (default Orange)",1,1,1,1,1,0],["Tree (Entropy, default)",1,1,1,1,1,0],["Tree (tuning/pruning)",1,1,1,1,1,0],["Random Forest",1,1,1,1,1,0],["Logistic Regression",1,1,1,1,1,0]],
 "skenario2_sumber": "Tabel 13",
 "pohon": [["Skenario 1, default Orange",1289,645,23,0.776,0.200,0.195],["Skenario 1, hasil tuning",15,8,3,0.258,0.200,0.158],["Skenario 2, default Orange",9,5,4,1.0,1.0,1.0],["Skenario 2, hasil tuning",9,5,3,1.0,1.0,1.0]],
 "pohon_sumber": "Tabel 14 (simpul, daun, kedalaman, akurasi latih, akurasi uji, F1 makro uji)",
 "entropy_manual": {"H_akar": 2.3190, "for_ada": 0.9983, "for_tidak": 1.5842, "H_setelah": 1.3644, "IG": 0.9546, "sumber": "Tabel 8–9; Subbab 4.3"},
 "permutasi": {"akurasi_asli": 0.1928, "rerata_acak": 0.2061, "sd": 0.0081, "p": 0.970, "n": 200, "sumber": "Subbab 4.4.4"},
 "importance_s2": {"for": 0.412, "developments": 0.251, "react": 0.176, "professionals": 0.161, "sumber": "Gambar 51"},
 "judul_baru": [["Bitcoin price surges after new regulation","Economy","AI & Tech","Salah"],["Lakers beat Celtics in overtime thriller","Sports","AI & Tech","Salah"],
   ["New smartphone chip promises faster on-device AI","AI & Tech","AI & Tech","Benar"],["Fans react to latest Hollywood news on social media","Entertainment","Entertainment","Benar"],
   ["Global attention shifts towards elections developments","Global Events","Global Events","Benar"],["Inflation remains a key concern for young professionals","Economy","Economy","Benar"],
   ["Record engagement for recent cricket matches","Sports","Sports","Benar"],["Startup funding for climate technology grows","AI & Tech","Sports","Salah"]],
 "judul_baru_sumber": "Tabel 15"})

w("studi-kasus-06.json", {
 "judul": "Naïve Bayes — Klasifikasi Arah IHSG Berbasis Indikator Makro-Finansial Multi-Aset (2010–2026)",
 "sumber": "Laporan studi kasus 06 (proyek tim)",
 "dataset": {"observasi": 4277, "periode": "2010-01-01 – 2026-05-29", "siap_model": 4276, "naik": 0.5035, "sumber": "Subbab 3.2, 3.4"},
 "split": {"latih": 3420, "uji": 856, "periode_latih": "2010-01-04 – 2023-02-10", "periode_uji": "2023-02-13 – 2026-05-29", "sumber": "Tabel 3.3"},
 "outlier": [["OIL_ret",-8.004,8.124,52],["GOLD_ret",-3.457,3.586,40],["USDIDR_ret",-1.841,1.877,130],["SP500_ret",-3.067,3.260,68],["IHSG_ret",-3.331,3.433,56]],
 "outlier_sumber": "Tabel 3.2 (batas bawah %, batas atas %, jumlah di-cap)",
 "dampak_outlier": {"akurasi_sebelum": 0.4848, "akurasi_sesudah": 0.5187, "brier_sebelum": 0.461, "brier_sesudah": 0.256, "sumber": "Abstrak; Subbab 4.3"},
 "metrik": {"akurasi": 0.5187, "presisi": 0.4955, "recall": 0.6806, "f1": 0.5735, "roc_auc": 0.5689, "brier": 0.2557, "pr_auc": 0.5376, "baseline": 0.4755, "sumber": "Tabel 4.5"},
 "confusion": {"TN": 167, "FP": 282, "FN": 130, "TP": 277, "sumber": "Subbab 4.7"},
 "cv": {"auc_mean": 0.5994, "auc_sd": 0.0292, "acc_mean": 0.5289, "acc_sd": 0.0474, "sumber": "Subbab 4.9"},
 "prior": {"turun_tetap": 0.4895, "naik": 0.5105, "sumber": "Subbab 4.5"},
 "worked_example": {"tanggal": "2023-02-21", "posterior_turun": 0.9024, "posterior_naik": 0.0976, "log_joint": [-14.9841, -17.2085], "aktual_ihsg_besok": -0.9229, "sumber": "Tabel 4.1–4.4"},
 "korelasi": {"SP500_ret–VIX_chg": -0.81, "target–SP500_ret": 0.19, "target–VIX_chg": -0.15, "sumber": "Subbab 4.2"}})

w("studi-kasus-09.json", {"judul": "Text Mining — Analisis Sentimen dan Pemodelan Topik Komentar Publik terkait MBG",
  "sumber": None, "status": "Laporan studi kasus 09 belum tersedia; tidak ada angka Hasil Laporan."})
print("ok")
