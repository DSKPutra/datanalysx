Buatkan aplikasi web Data Analysis bernama **DATANALYSX** yang merangkum seluruh tugas mata kuliah Data Mining saya (Tugas 01–09), lalu deploy ke GitHub, Netlify, dan Vercel. Kerjakan end-to-end sampai ketiga URL dapat dibuka.

═══════════════════════════════════════
A. IDENTITAS & SUMBER
═══════════════════════════════════════
Nama aplikasi : DATANALYSX
Tagline       : "From Raw Data to Insight — Data Mining Lab"
Pemilik       : Dea Saka Kurnia Putra (NIM 261012000111)
Program/Kelas : S2 TI Reg B 02 — kode kelas 01MKMM002
Mata Kuliah   : Data Mining — Magister Teknik Informatika, Universitas Pamulang
Dosen         : Dr. Ir. Agung Budi Susanto, M.M.
Tahun Ajaran  : 2026/2027

Sumber yang WAJIB dibaca seluruhnya sebelum mulai:
1. Laporan PDF yang saya lampirkan:
   • Pertemuan 01 — Algoritma Regresi (Linear Regression, saham NVIDIA)
   • Pertemuan 02 — Regresi Linear dan Non Linear (AI Economy Pulse)
   • Pertemuan 03 — Regresi Linear Sederhana dan Berganda (semikonduktor)
   • Pertemuan 04 — Logistic Regression (NASDAQ OHLC, AAPL)
   Laporan 01 & 02 sebagian besar berupa gambar: render halaman dan baca
   visual/OCR. Kode Python Colab ada di Lampiran A masing-masing laporan.
2. Dokumen di Project "UNPAM_Data Mining":
   laporan-regresi-logistik-summary.md (= Tugas 04),
   laporan-naive-bayes-summary.md (Tugas 06), laporan-knn-summary.md
   (Tugas 07), laporan-kmeans-summary.md (Tugas 08),
   prompt-tugas09-text-mining.md, dan laporan-text-mining-summary.md bila
   sudah ada (Tugas 09).
Semua angka, keputusan desain, dan temuan di aplikasi harus konsisten
dengan sumber tersebut. Angka di Bagian C di bawah dikutip dari laporan;
bila berbeda dengan sumber, sumber yang menang dan catat perbedaannya.

═══════════════════════════════════════
B. TUJUAN APLIKASI
═══════════════════════════════════════
1. Portofolio interaktif: setiap tugas tampil sebagai modul dengan
   ringkasan masalah, dataset, metodologi (CRISP-DM), hasil utama,
   grafik interaktif, dan kesimpulan kritis.
2. Lab interaktif: pengguna dapat mengunggah CSV sendiri (atau memakai
   dataset contoh) dan menjalankan algoritma tiap tugas langsung di
   browser, mengubah parameter, dan melihat hasilnya berubah.
3. Edukatif: setiap modul punya mode "perhitungan manual langkah demi
   langkah" yang hasilnya diverifikasi identik dengan implementasi
   library.
4. Benang merah: tampilkan perjalanan materi dari regresi (prediksi
   nilai) → klasifikasi (regresi logistik, NB, KNN) → clustering
   (K-Means) → text mining, termasuk pelajaran yang terbawa antar tugas
   (mis. kebocoran data di Tugas 01 → time-based split di Tugas 04).

═══════════════════════════════════════
C. MODUL (satu halaman/route per tugas: /tugas-01 … /tugas-09)
═══════════════════════════════════════
Setiap modul memiliki tab: Ringkasan · Data · Hasil Laporan · Lab Live ·
Perhitungan Manual · Kode Python · Catatan Kritis.

Tugas 01 — Linear Regression: Prediksi Harga Close Saham NVIDIA
 • Dataset: NVIDIA Stock Data (1999–2026), Kaggle (Shah, 2026);
   7.079 baris × 16 kolom, 22 Jan 1999 – 11 Mar 2026; 0 duplikat;
   missing hanya struktural (stock_split 7.073, key_event 6.826).
   Lima era bisnis (Pre-GPU → Generative AI), 6 stock split.
 • Fitur X: Open, High, Low, Volume, SMA 50, RSI 14 → target Close.
   Split acak 80:20 (random_state=42) → 5.663 latih / 1.416 uji.
 • Hasil laporan: MAE 0,0047; RMSE 0,0101; R² 0,999994. Persamaan:
   Close' = −0,0000818 + 0,006963·Open + 0,475677·High + 0,512563·Low
   − 1,6e−12·Volume + 0,005132·SMA50 + 0,0000039·RSI14.
   Korelasi Pearson ke Close: Open 0,999985; Low 0,999982; High
   0,999981; SMA50 0,998605; Volume 0,783967; RSI14 0,248561.
 • Lab: grafik harga Close dengan pita warna per era + marker key_event,
   heatmap korelasi, scatter aktual vs prediksi, bar koefisien, pilih
   fitur sendiri lalu latih ulang.
 • Fitur wajib "Uji Realisme" (sesuai keterbatasan & saran laporan):
   toggle (a) fitur hari yang sama [model laporan] vs (b) lag features
   (Close/OHLC hari t−1…t−5 untuk memprediksi Close t+1), dan
   (c) split acak vs split berbasis waktu. Tampilkan bagaimana R² yang
   hampir sempurna turun drastis dan bandingkan dengan baseline naif
   "Close besok = Close hari ini". Pesan: R² 0,999994 adalah estimasi
   dalam-hari, bukan peramalan.
 • Perhitungan manual: OLS b = (XᵀX)⁻¹Xᵀy pada subset kecil data,
   dicocokkan dengan LinearRegression (assert).

Tugas 02 — Regresi Linear vs Non-Linear (AI Economy Pulse 2022–2026)
 • Dataset: AI_Economy_Pulse_Dataset_2022_-_Present.csv, 1.500 baris ×
   10 kolom (task_id, domain [Coding, Mathematics, RAG/QA, Planning,
   Customer Support], difficulty, prompt, context, expected_answer,
   agent_answer, failure_type [12 kategori], failure_severity, notes);
   0 missing, 0 duplikat.
 • Rekayasa fitur: difficulty_score (Easy=1, Medium=2, Hard=3),
   prompt_length & context_length (jumlah kata) → target severity_score
   (Low=1, Medium=2, High=3, Critical=4). Split 80:20 → 1.200 / 300.
 • Hasil laporan:
     Linear Regression  MAE 0,624560  RMSE 0,772797  R² 0,139403
     Random Forest(100) MAE 0,198289  RMSE 0,399665  R² 0,769824
   Persamaan LR: severity' = 1,637357 − 0,078486·difficulty_score
   + 0,007658·prompt_length + 0,038742·context_length.
 • Lab: tabel feature engineering (teks → angka) yang bisa diklik per
   baris, scatter context_length vs severity, slider n_estimators &
   max_depth Random Forest, bandingkan dengan LR, Decision Tree, dan
   kNN regresi; bar R²; plot aktual vs prediksi berdampingan.
 • Perhitungan manual: satu prediksi LR dari persamaan + rata-rata
   prediksi pohon-pohon RF untuk satu baris uji (cocok dengan predict).
 • Catatan kritis: target ordinal 1–4 diperlakukan kontinu; domain &
   failure_type belum dipakai (tunjukkan efek menambahkannya di Lab);
   uji robust dengan cross-validation, bukan satu split.

Tugas 03 — Regresi Linear Sederhana & Berganda (Industri Semikonduktor)
 • Dataset: chip_companies_financials.csv — 617 baris, 40 perusahaan,
   2010–2026; 28 baris bernilai nol dibuang (Graphcore gugur) → 589
   baris, 39 perusahaan. Y = revenue_usd_bn; X = rd_spend_usd_bn,
   capex_usd_bn, operating_margin_pct. operating_income sengaja dibuang
   (kebocoran data). Tampilkan juga tabel 5 CSV yang ditinjau dan
   alasan kelayakannya.
 • Hasil laporan (OLS, seluruh data):
     Sederhana linear: Pendapatan' = 5,324 + 4,726·R&D; R² 0,8234
     Sederhana log-log: ln Y' = 2,072 + 0,915·ln R&D; R² 0,8776
     Berganda linear: 2,725 + 3,848·R&D + 1,477·Capex − 0,013·Margin;
       R² 0,9691 (margin tidak signifikan, p = 0,358)
     Berganda log-log: 1,969 + 0,697·ln R&D + 0,271·ln Capex
       + 0,0031·Margin; R² 0,9884; VIF < 1,5
     Efek tetap perusahaan: elastisitas R&D 0,9991; R² within 0,9998
 • CV 10-lipatan (RMSE / MAE / R²): Constant 21,747/12,915/−0,002;
   Linear sederhana 9,494/5,243/0,809; Linear berganda 3,837/2,503/
   0,969; Berganda log 5,226/1,767/0,942; kNN 7,534/1,998/0,880;
   SGD 3,837/2,505/0,969; SVM RBF 10,264/2,256/0,777; Random Forest
   4,912/0,949/0,949. Hold-out 80:20 → 471/118 baris.
 • Lab bergaya widget Orange (meniru laporan): Data Table, Feature
   Statistics, Distributions (asli vs log), Box/Violin per segmen,
   Correlations, Scatter Plot dengan garis regresi + CI/PI 95%,
   Linear Regression (koefisien, SE, t, p, CI), Test and Score
   (tabel + matriks "Model Comparison by MSE"), Predictions.
 • Diagnostik interaktif: Residual vs Fitted, Q–Q, Scale–Location,
   Cook's distance, uji Breusch–Pagan, Shapiro–Wilk, Durbin–Watson;
   toggle skala asli ↔ log-log untuk melihat heteroskedastisitas hilang.
 • Kalkulator prediksi: input R&D, Capex, Margin → hasil keempat model
   (contoh laporan: R&D 10, Capex 8, Margin 30% → 52,58 / 65,23 /
   52,62 / 68,73 miliar USD), dengan peringatan ekstrapolasi di luar
   rentang data.
 • Perhitungan manual: b = [nΣXY − ΣXΣY]/[nΣX² − (ΣX)²],
   a = [ΣY − bΣX]/n sesuai materi dosen, lalu matriks (XᵀX)⁻¹XᵀY untuk
   berganda — cocokkan dengan library.
 • Temuan kunci yang harus ditonjolkan: R&D dan capex adalah proporsi
   hampir tetap dari pendapatan tiap perusahaan, sehingga hubungannya
   definisional, bukan kausal; slope 0,915 adalah artefak percampuran
   perusahaan; data panel → galat baku optimistis.

Tugas 04 — Regresi Logistik (NASDAQ Daily OHLC, AAPL)
 • Dataset: NASDAQ Daily Stock Prices 1962–2026; AAPL dibatasi
   2010-01-01 s.d. 2026-09-18 → 4.184 baris bersih → 4.183 observasi.
   Time-based split 80:20 → 3.346 latih / 837 uji.
 • 10 fitur tanpa look-ahead: return_1d/2d/5d, ma_gap_5_20, vol_10d,
   vol_20d, hl_spread, oc_spread, rsi_14, momentum_10. Target = Close
   H+1 > Close H. StandardScaler di-fit pada data latih. L2, C = 1,0,
   lbfgs.
 • Hasil laporan: akurasi 51,37%; balanced accuracy 49,18%; presisi
   53,93%; recall 73,68%; F1 62,28%; ROC-AUC 0,484 (CI bootstrap
   [0,443; 0,525]); MCC −0,0188; log-loss 0,6955; CM TN 94, FP 287,
   FN 120, TP 336. MLE tanpa regularisasi: semua p > 0,14; LR-test
   p = 0,4632; McFadden R² 0,0021. Walk-forward AUC 0,4925 ± 0,0175;
   C optimal 0,001; Brier 0,2511.
 • Lab: confusion matrix interaktif, slider threshold (tunjukkan jebakan
   "F1 optimal" spurious di threshold 0,05), ROC & PR curve, kurva
   sigmoid dengan log-odds, tabel koefisien/odds ratio + p-value (IRLS),
   VIF, walk-forward CV, tuning C, kalibrasi, permutation importance,
   bootstrap CI AUC.
 • Uji lintas-saham 20 emiten: rata-rata AUC 0,5156 (SD 0,0234), uji-t
   p = 0,0075, Wilcoxon p = 0,0121, 14/20 di atas 0,5. Pesan kunci:
   AAPL ≈ acak (EMH bentuk lemah), sinyal agregat tipis tapi signifikan
   secara statistik, belum tentu ekonomis.
 • Perhitungan manual: z = b₀ + Σbᵢxᵢ → sigmoid → probabilitas →
   keputusan threshold untuk satu hari uji (cocok dengan predict_proba).

Tugas 05 — BELUM ADA LAPORAN
 • Saya akan melampirkan materi/laporan Tugas 05. Bila tidak ada, buat
   kartu modul "Segera hadir" tanpa isi dan tanpa angka apa pun. JANGAN
   mengarang topik atau hasil.

Tugas 06 — Naïve Bayes (tugas kelompok; tampilkan 5 anggota + NIM)
 • Arah IHSG hari berikutnya dari data makro multi-aset (4.277 obs.,
   2010 – Mei 2026), 9 fitur return, tanpa scaling.
 • Toggle "dengan/tanpa penanganan outlier" (winsorization IQR 3×):
   akurasi 48,48% → 51,87%, Brier 0,461 → 0,256; AUC 0,569; F1 0,5735.
 • Worked example Gaussian NB (21 Feb 2023): prior → likelihood per
   fitur → log-sum-exp → posterior → MAP, dibandingkan predict_proba.

Tugas 07 — KNN (Instagram Reels: Viral Hit vs High Reach)
 • Slider K, metrik jarak (Euclidean/Manhattan/cosine), bobot
   (uniform/distance), scaler (none/Standard/MinMax) → akurasi CV live.
 • Visualisasi 5 tetangga terdekat untuk satu titik uji + perhitungan
   manual; demo curse of dimensionality dan lazy learner.

Tugas 08 — K-Means (AAPL 2026; dataset sintetis-terinterpolasi — beri
label peringatan)
 • Slider K (2–10), k-means++ vs acak, n_init; Elbow, Silhouette, DBI,
   CH, Gap; scatter PCA; animasi iterasi (assign → update centroid).
 • Kestabilan lintas seed (ARI) sebagai alasan K = 3; profil rezim
   Bullish/Reli, Melemah/Konsolidasi, Aktif–Volatil; timeline rezim.

Tugas 09 — Text Mining (komentar publik terkait MBG)
 • Pipeline preprocessing interaktif (filtering, case folding,
   tokenizing, stopword removal dengan negasi dipertahankan, normalisasi
   slang, stemming Sastrawi), kalkulator TF-IDF manual (IDF klasik vs
   smooth, L2) dicocokkan dengan TfidfVectorizer.
 • Sentimen leksikon (InSet) + klasifikasi (MNB, LogReg, Linear SVM,
   KNN); LDA/NMF dengan slider jumlah topik; word cloud, tren mingguan,
   ko-okurensi. Etika: username disamarkan, kata kasar disensor.

Halaman tambahan:
 • Beranda: hero, kartu 9 modul (dikelompokkan: Regresi 01–03 ·
   Klasifikasi 04–07 · Clustering 08 · Text Mining 09), timeline
   perkuliahan.
 • "Regression Lab" gabungan untuk Tugas 01–03: unggah CSV → pilih
   target & fitur → bandingkan Linear, log-log, Ridge, Lasso, Decision
   Tree, Random Forest, kNN dengan MAE/RMSE/R², split acak vs berbasis
   waktu, plus panel uji asumsi klasik.
 • "Perbandingan Algoritma": kelebihan/kekurangan & kapan dipakai
   (Linear Regression, Random Forest Regression, Regresi Logistik, NB,
   KNN, K-Means, text mining), ditarik dari bagian kelebihan–kekurangan
   dan keterbatasan tiap laporan.
 • "Pelajaran Metodologi": kartu lintas tugas — kebocoran data (Tugas 01
   same-day OHLC, Tugas 03 operating_income, Tugas 04 time split),
   pilihan metrik mengubah peringkat model (Tugas 03 RMSE vs MAE),
   signifikansi statistik ≠ ekonomis (Tugas 04), outlier (Tugas 06),
   scaling (Tugas 07–08), data sintetis (Tugas 08).
 • "Data Lab" umum klasifikasi: unggah CSV → pilih target → jalankan
   beberapa algoritma → leaderboard metrik.
 • Tentang & Referensi (APA 7th): HANYA referensi yang ada di Daftar
   Pustaka laporan saya, dikelompokkan per tugas; jangan menambah
   referensi karangan.

═══════════════════════════════════════
D. ARSITEKTUR TEKNIS
═══════════════════════════════════════
Karena target deploy adalah hosting statis (Netlify & Vercel tanpa
server Python), semua komputasi berjalan di browser:
 • Frontend: React + Vite + TypeScript, Tailwind CSS, React Router.
 • Komputasi ML: Pyodide (Python di WebAssembly) dalam Web Worker,
   memakai numpy, pandas, scikit-learn, scipy yang tersedia di Pyodide;
   paket murni-Python lain (mis. Sastrawi) lewat micropip. Muat Pyodide
   secara lazy (hanya saat tab Lab/Manual dibuka), tampilkan indikator
   loading, dan cache-kan. Uji statistik regresi (t, F, VIF, BP, DW,
   Cook) diimplementasikan dengan NumPy/SciPy seperti kelas OLS di
   Lampiran Tugas 03 (tanpa statsmodels).
 • Grafik: Plotly.js atau Recharts (interaktif, tooltip, ekspor PNG).
 • Kode Python per modul diambil dari kode di lampiran laporan
   (Tugas 01–03 dari Lampiran A/1 PDF) dan skrip
   logistic_regression_nasdaq.py, naive_bayes_ihsg.py,
   knn_instagram_reels.py, kmeans_aapl_stock.py, textmining_mbg.py,
   lalu diringkas menjadi fungsi yang bisa dipanggil dari worker.
   Tombol "Lihat kode Python" di setiap modul.
 • Hasil laporan resmi disimpan sebagai JSON statis di
   /public/data/tugas-XX.json (dari model_summary.json bila ada; untuk
   Tugas 01–03 ditranskripsi dari tabel laporan, dengan field "sumber":
   "Laporan Pertemuan 0X, Tabel/Gambar …"). Ditampilkan sebagai "Hasil
   Laporan"; hasil yang dihitung ulang di browser diberi label "Hasil
   Live" — keduanya tidak boleh dicampur.
 • Dataset contoh: hanya sertakan yang lisensinya mengizinkan
   redistribusi (cek lisensi Kaggle tiap dataset: NVIDIA, NASDAQ OHLC,
   AAPL 2026, Instagram Reels; asal AI Economy Pulse dan
   chip_companies_financials perlu dicek dan ditanyakan ke saya). Data
   komentar MBG hanya versi tersamarkan atau sampel kecil. Bila tidak
   boleh atau ragu, sediakan tombol unggah saja dan tampilkan hasil
   laporan dari JSON.
 • Tidak ada backend, tidak ada API key, tidak ada pelacakan pengguna.
   Data unggahan tidak pernah meninggalkan browser (sebutkan di UI).
 • Responsif (mobile 375px sampai desktop), mode terang/gelap,
   aksesibel (kontras, label, navigasi keyboard).

═══════════════════════════════════════
E. DESAIN
═══════════════════════════════════════
 • Konsisten dengan laporan saya: hijau zamrud #0B4F3A / #0B7A53, emas
   #B8860B sebagai aksen tipis, hitam/abu gelap untuk mode gelap.
 • Font judul Cambria-like (mis. "Cormorant"/"Playfair" via Google
   Fonts), isi sans-serif (Inter/Calibri-like).
 • Logo UNPAM dan logo personal branding saya (@PUTRATALKS, versi latar
   transparan) — saya lampirkan file-nya.
 • Komponen "widget Orange" untuk Tugas 03 bergaya panel widget (judul,
   ikon, area kontrol kiri, area hasil kanan) namun tetap memakai
   palet DATANALYSX.
 • Warna kelas konsisten di semua grafik (positif/netral/negatif;
   naik/turun; klaster C1–C3; era NVIDIA; segmen semikonduktor), tetap
   terbedakan untuk buta warna.
 • Bahasa UI: Indonesia (opsional toggle English). Angka memakai format
   Indonesia (koma desimal) di UI.

═══════════════════════════════════════
F. REPOSITORI & DEPLOY
═══════════════════════════════════════
1. GitHub
   • Repo publik "datanalysx" di akun GitHub saya (tanyakan username
     bila belum diketahui).
   • Struktur rapi: /src, /public/data, /python (skrip per tugas),
     /docs, README.md (deskripsi, screenshot, cara menjalankan lokal,
     link demo, tabel 9 tugas), LICENSE (MIT, kecuali dataset),
     .gitignore.
   • GitHub Actions: lint + build + test pada setiap push/PR.
   • Aktifkan GitHub Pages juga (base path disesuaikan di Vite).
2. Netlify
   • netlify.toml (build: npm run build, publish: dist, redirect SPA
     /* → /index.html 200, header cache untuk aset Pyodide).
   • Hubungkan ke repo GitHub agar auto-deploy dari branch main.
   • Nama situs: datanalysx (atau varian terdekat bila sudah dipakai).
3. Vercel
   • vercel.json (rewrite SPA, framework Vite).
   • Proyek "datanalysx" terhubung ke repo yang sama, auto-deploy main.
4. Setelah deploy: buka ketiga URL, uji setiap modul (Pyodide termuat,
   grafik muncul, unggah CSV jalan, routing langsung ke /tugas-03 tidak
   404), lalu tulis ketiga URL di README.
Langkah yang memerlukan login/otorisasi akun saya (membuat repo,
menghubungkan Netlify/Vercel ke GitHub) harus meminta persetujuan saya
dulu; jangan memasukkan password atau token saya.

═══════════════════════════════════════
G. QUALITY CONTROL
═══════════════════════════════════════
1. Setiap angka "Hasil Laporan" cocok dengan laporan PDF /
   model_summary.json / ringkasan Project. Buat tabel verifikasi
   (angka → sumber halaman/tabel) di /docs/verifikasi-angka.md.
   Tidak ada angka yang dikarang.
2. Reproduksi: jalankan ulang kode Tugas 01–03 pada dataset aslinya
   (bila saya berikan) dan pastikan hasilnya sama dengan laporan
   (toleransi pembulatan); bila berbeda, laporkan, jangan diam-diam
   mengganti.
3. Setiap worked example lolos uji kesetaraan dengan library (toleransi
   < 1e-8) — unit test (Vitest untuk TS, pytest untuk fungsi Python).
4. random_state = 42 di semua modul; hasil live deterministik.
5. Tidak ada kebocoran data pada mode "benar" di Lab (scaler/TF-IDF
   di-fit hanya pada data latih via Pipeline; split temporal untuk data
   saham). Mode yang meniru laporan (mis. Tugas 01 split acak dengan
   OHLC hari yang sama) tetap tersedia, diberi label jelas.
6. Lighthouse: Performance ≥ 80 (setelah Pyodide di-cache),
   Accessibility ≥ 90; tidak ada error di console.
7. Uji di Chrome, Firefox, dan mobile.
8. Peringatan jujur ditampilkan: R² Tugas 01 bukan kemampuan
   peramalan; hubungan Tugas 03 definisional, bukan kausal; dataset
   AAPL 2026 sintetis; label sentimen otomatis ≠ ground truth; komentar
   YouTube ≠ opini publik Indonesia; hasil bukan saran investasi.

═══════════════════════════════════════
OUTPUT
═══════════════════════════════════════
(1) Link repo GitHub + GitHub Pages, (2) URL Netlify, (3) URL Vercel,
(4) source code lengkap (zip), (5) README dengan screenshot,
(6) /docs/verifikasi-angka.md, (7) ringkasan proyek disimpan ke Project
sebagai claude/datanalysx-summary.md agar bisa dilanjutkan di sesi
berikutnya.
