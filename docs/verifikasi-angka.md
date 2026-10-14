# Verifikasi angka “Hasil Laporan”

Dibangkitkan oleh `scripts/verify_numbers.py` (Python 3.13, numpy 2.2.5, pandas 2.3.3, scikit-learn 1.7.0, scipy 1.14.1 — sama dengan Pyodide 0.29.5).
Kolom *Laporan* = angka yang ditampilkan di tab Hasil Laporan (dari laporan / model_summary.json); *Reproduksi* = hasil menjalankan modul `python/dmx` pada dataset asli.

| Studi kasus | Angka | Laporan | Sumber | Reproduksi | Status |
|---|---|---|---|---|---|
| 01 | MAE | 0.0047 | Tabel 3.3 | 0.00466861 | ✓ cocok |
| 01 | RMSE | 0.0101 | Tabel 3.3 | 0.0100536 | ✓ cocok |
| 01 | R² | 0.999994 | Tabel 3.3 | 0.999994 | ✓ cocok |
| 01 | Intersep | -8.177877961923485e-05 | Lampiran B | -8.17958e-05 | ✓ cocok |
| 01 | Koef. High | 0.475677 | Tabel 3.7 | 0.475677 | ✓ cocok |
| 01 | Koef. Low | 0.512563 | Tabel 3.7 | 0.512563 | ✓ cocok |
| 01 | n latih | 5663 | Subbab 3.2 | 5663 | ✓ cocok |
| 01 | n uji | 1416 | Subbab 3.2 | 1416 | ✓ cocok |
| 02 | LR MAE | 0.62456 | Tabel 3.4 | 0.62456 | ✓ cocok |
| 02 | LR RMSE | 0.772797 | Tabel 3.4 | 0.772797 | ✓ cocok |
| 02 | LR R² | 0.139403 | Tabel 3.4 | 0.139403 | ✓ cocok |
| 02 | RF MAE | 0.198289 | Tabel 3.4 | 0.198289 | ✓ cocok |
| 02 | RF RMSE | 0.399665 | Tabel 3.4 | 0.399665 | ✓ cocok |
| 02 | RF R² | 0.769824 | Tabel 3.4 | 0.769824 | ✓ cocok |
| 02 | Intersep LR | 1.637357 | Tabel 3.2 | 1.63736 | ✓ cocok |
| 03 | n analisis | 589 | Tabel 3 | 589 | ✓ cocok |
| 03 | a sederhana | 5.324 | Tabel 5 | 5.32374 | ✓ cocok |
| 03 | b sederhana | 4.726 | Tabel 5 | 4.72572 | ✓ cocok |
| 03 | R² sederhana | 0.8234 | Tabel 5 | 0.823432 | ✓ cocok |
| 03 | R² log-log | 0.8776 | Subbab 4.2.2 | 0.877607 | ✓ cocok |
| 03 | R² berganda | 0.9691 | Tabel 11 | 0.9691 | ✓ cocok |
| 03 | p margin | 0.358 | Tabel 9 | 0.357999 | ✓ cocok |
| 03 | R² berganda log | 0.9884 | Tabel 11 | 0.988408 | ✓ cocok |
| 03 | Elastisitas efek tetap | 0.9991 | Tabel 16 | 0.999115 | ✓ cocok |
| 03 | CV RMSE Constant (rata-rata) | 21.747 | Tabel 13 | 21.7468 | ✓ cocok |
| 03 | CV MAE Constant (rata-rata) | 12.915 | Tabel 13 | 12.9152 | ✓ cocok |
| 03 | CV R² Constant (rata-rata) | -0.002 | Tabel 13 | -0.00159661 | ✓ cocok |
| 03 | CV RMSE Linear – sederhana | 9.494 | Tabel 13 | 9.49366 | ✓ cocok |
| 03 | CV MAE Linear – sederhana | 5.243 | Tabel 13 | 5.24329 | ✓ cocok |
| 03 | CV R² Linear – sederhana | 0.809 | Tabel 13 | 0.809116 | ✓ cocok |
| 03 | CV RMSE Linear – berganda | 3.837 | Tabel 13 | 3.83675 | ✓ cocok |
| 03 | CV MAE Linear – berganda | 2.503 | Tabel 13 | 2.50344 | ✓ cocok |
| 03 | CV R² Linear – berganda | 0.969 | Tabel 13 | 0.968823 | ✓ cocok |
| 03 | CV RMSE Linear – berganda (log) | 5.226 | Tabel 13 | 5.22559 | ✓ cocok |
| 03 | CV MAE Linear – berganda (log) | 1.767 | Tabel 13 | 1.76744 | ✓ cocok |
| 03 | CV R² Linear – berganda (log) | 0.942 | Tabel 13 | 0.942168 | ✓ cocok |
| 03 | CV RMSE kNN (k=5) | 7.534 | Tabel 13 | 7.53381 | ✓ cocok |
| 03 | CV MAE kNN (k=5) | 1.998 | Tabel 13 | 1.99842 | ✓ cocok |
| 03 | CV R² kNN (k=5) | 0.88 | Tabel 13 | 0.879793 | ✓ cocok |
| 03 | CV RMSE SGD | 3.837 | Tabel 13 | 3.83665 | ✓ cocok |
| 03 | CV MAE SGD | 2.505 | Tabel 13 | 2.50481 | ✓ cocok |
| 03 | CV R² SGD | 0.969 | Tabel 13 | 0.968825 | ✓ cocok |
| 03 | CV RMSE SVM (RBF) | 10.264 | Tabel 13 | 10.2636 | ✓ cocok |
| 03 | CV MAE SVM (RBF) | 2.256 | Tabel 13 | 2.25581 | ✓ cocok |
| 03 | CV R² SVM (RBF) | 0.777 | Tabel 13 | 0.776899 | ✓ cocok |
| 03 | CV RMSE Random Forest | 4.912 | Tabel 13 | 4.90553 | ⚠ beda |
| 03 | CV MAE Random Forest | 0.949 | Tabel 13 | 0.945876 | ⚠ beda |
| 03 | CV R² Random Forest | 0.949 | Tabel 13 | 0.949035 | ✓ cocok |
| 03 | Prediksi sederhana | 52.58 | Tabel 19 | 52.581 | ✓ cocok |
| 03 | Prediksi sederhana log | 65.23 | Tabel 19 | 65.2327 | ✓ cocok |
| 03 | Prediksi berganda | 52.62 | Tabel 19 | 52.618 | ✓ cocok |
| 03 | Prediksi berganda log | 68.73 | Tabel 19 | 68.7273 | ✓ cocok |
| 04 | Observasi | 4183 | Subbab 3.3 | 4183 | ✓ cocok |
| 04 | Baris bersih | 4184 | Subbab 3.2 | 4203 | ⚠ beda |
| 04 | Akurasi | 0.5137 | Tabel 8 | 0.51374 | ✓ cocok |
| 04 | Balanced acc. | 0.4918 | Tabel 8 | 0.491781 | ✓ cocok |
| 04 | Presisi | 0.5393 | Tabel 8 | 0.539326 | ✓ cocok |
| 04 | Recall | 0.7368 | Tabel 8 | 0.736842 | ✓ cocok |
| 04 | F1 | 0.6228 | Tabel 8 | 0.622799 | ✓ cocok |
| 04 | ROC-AUC | 0.484 | Tabel 8 | 0.484425 | ✓ cocok |
| 04 | MCC | -0.0188 | Tabel 8 | -0.0187656 | ✓ cocok |
| 04 | Log-loss | 0.6955 | Tabel 8 | 0.695489 | ✓ cocok |
| 04 | Brier | 0.2511 | Tabel 8 | 0.251131 | ✓ cocok |
| 04 | LR-test p | 0.4632 | Subbab 4.4 | 0.463198 | ✓ cocok |
| 04 | McFadden R² | 0.0021 | Subbab 4.4 | 0.00210544 | ✓ cocok |
| 04 | Walk-forward AUC | 0.4925 | Tabel 11 | 0.4925 | ✓ cocok |
| 04 | C optimal | 0.001 | Subbab 4.7 | 0.001 | ✓ cocok |
| 04 | Bootstrap CI bawah | 0.443 | Subbab 4.11 | 0.443011 | ✓ cocok |
| 04 | Bootstrap CI atas | 0.5246 | Subbab 4.11 | 0.524611 | ✓ cocok |
| 04 | F1 threshold 0,05 | 0.7053 | Subbab 4.8 | 0.705336 | ✓ cocok |
| 04 | Lintas saham rata-rata AUC | 0.5156 | Subbab 4.13 | 0.515638 | ✓ cocok |
| 04 | Lintas saham uji-t p | 0.0075 | Subbab 4.13 | 0.00750439 | ✓ cocok |
| 04 | Lintas saham Wilcoxon p | 0.0121 | Subbab 4.13 | 0.0120792 | ✓ cocok |
| 04 | Lintas saham n > 0,5 | 14 | Subbab 4.13 | 14 | ✓ cocok |
| 05 | S1 default simpul | 1289 | Tabel 14 | 1289 | ✓ cocok |
| 05 | S1 default akurasi latih | 0.776 | Tabel 14 | 0.7765 | ✓ cocok |
| 05 | S1 default akurasi uji | 0.2 | Tabel 14 | 0.2 | ✓ cocok |
| 05 | S2 akurasi uji | 1.0 | Tabel 14 | 1 | ✓ cocok |
| 05 | Entropy akar | 2.319 | Tabel 8 | 2.31896 | ✓ cocok |
| 05 | IG kata for | 0.9546 | Subbab 4.3.2 | 0.954581 | ✓ cocok |
| 06 | Akurasi | 0.5187 | Tabel 4.5 | 0.518692 | ✓ cocok |
| 06 | Presisi | 0.4955 | Tabel 4.5 | 0.495528 | ✓ cocok |
| 06 | Recall | 0.6806 | Tabel 4.5 | 0.68059 | ✓ cocok |
| 06 | F1 | 0.5735 | Tabel 4.5 | 0.573499 | ✓ cocok |
| 06 | ROC-AUC | 0.5689 | Tabel 4.5 | 0.568854 | ✓ cocok |
| 06 | Brier | 0.2557 | Tabel 4.5 | 0.255701 | ✓ cocok |
| 06 | Baseline mayoritas | 0.4755 | Tabel 4.5 | 0.475467 | ✓ cocok |
| 06 | CV AUC | 0.5994 | Subbab 4.9 | 0.599392 | ✓ cocok |
| 06 | Akurasi tanpa outlier handling | 0.4848 | Abstrak | 0.484813 | ✓ cocok |
| 06 | Brier tanpa outlier handling | 0.461 | Abstrak | 0.460807 | ✓ cocok |
| 06 | Posterior Turun 21-02-2023 | 0.9024 | Tabel 4.4 | 0.902411 | ✓ cocok |
| 07 | CV GridSearch | 0.8846 | model_summary.gridsearch | 0.884615 | ✓ cocok |
| 07 | Uji accuracy | 0.7727 | model_summary.knn_test | 0.772727 | ✓ cocok |
| 07 | Uji precision | 0.7273 | model_summary.knn_test | 0.727273 | ✓ cocok |
| 07 | Uji recall | 0.8 | model_summary.knn_test | 0.8 | ✓ cocok |
| 07 | Uji f1 | 0.7619 | model_summary.knn_test | 0.761905 | ✓ cocok |
| 07 | Uji roc_auc | 0.6667 | model_summary.knn_test | 0.666667 | ✓ cocok |
| 08 | inertia | 456.4497 | model_summary.final_model | 456.45 | ✓ cocok |
| 08 | silhouette | 0.2104 | model_summary.final_model | 0.210383 | ✓ cocok |
| 08 | dbi | 1.5521 | model_summary.final_model | 1.55205 | ✓ cocok |
| 08 | ch | 29.6519 | model_summary.final_model | 29.6519 | ✓ cocok |
| 08 | Ukuran C1 | 41 | model_summary.final_model | 41 | ✓ cocok |
| 08 | Ukuran C2 | 52 | model_summary.final_model | 52 | ✓ cocok |
| 08 | Ukuran C3 | 23 | model_summary.final_model | 23 | ✓ cocok |

**Ringkasan:** 101 dari 104 angka cocok dalam toleransi pembulatan.

## Catatan perbedaan (dilaporkan, tidak diganti diam-diam)
- **Studi kasus 01:** scikit-learn ≥ 1.8 menghasilkan solusi `LinearRegression` berbeda (MAE 1,34) karena kolom Volume berorde 10⁸; numpy `lstsq` dan scikit-learn 1.7.0 mereproduksi laporan. Aplikasi & CI dikunci ke 1.7.0.
- **Studi kasus 02:** berkas data identik dengan *LLM Agent Failure Analysis Benchmark Dataset* (Apache 2.0), bukan *AI Economy Pulse* yang dikutip laporan.
- **Studi kasus 03:** CV Random Forest: RMSE 4,906 / MAE 0,946 (laporan 4,912 / 0,949) — perbedaan implementasi pohon antarversi scikit-learn; R² dan tujuh model lain cocok.
- **Studi kasus 04:** laporan menyebut 4.184 baris bersih; kode menghasilkan 4.203 baris bersih → 4.183 observasi (cocok). Baris terakhir pada skrip asli memperoleh target 0 walau Close H+1 tak diketahui; aplikasi menyediakan opsi membuangnya.
- **Studi kasus 06:** batas IQR laporan dihitung dari seluruh periode (termasuk data uji) dan baris terakhir ikut dengan target 0; mode ‘benar’ (batas dari data latih) memberi akurasi 52,05%.
- **Studi kasus 08:** pada pemilihan K, DBI terbaik live dapat berbeda dari laporan (K = 10) bila n_init diperkecil demi kecepatan browser.
- **Studi kasus 09:** laporan belum tersedia — tidak ada angka Hasil Laporan.

Studi kasus 04 direproduksi dengan data NASDAQ lokal (tidak diredistribusi). Studi kasus 07–08 dibandingkan dengan model_summary.json resmi.
