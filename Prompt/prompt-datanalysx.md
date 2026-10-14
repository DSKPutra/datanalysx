Buatkan aplikasi web Data Analysis bernama **DATANALYSX** yang merangkum seluruh tugas mata kuliah Data Mining saya (Tugas 01–09), lalu deploy ke GitHub, Netlify, dan Vercel. Kerjakan end-to-end sampai ketiga URL dapat dibuka.

═══════════════════════════════════════
A. IDENTITAS
═══════════════════════════════════════
Nama aplikasi : DATANALYSX
Tagline       : "From Raw Data to Insight — Data Mining Lab"
Pemilik       : Dea Saka Kurnia Putra (NIM 261012000111)
Mata Kuliah   : Data Mining — Magister Teknik Informatika, Universitas Pamulang
Dosen         : Dr. Ir. Agung Budi Susanto, M.M.
Tahun Ajaran  : 2026/2027
(Baca SELURUH dokumen di Project "UNPAM_Data Mining" sebelum mulai:
laporan-regresi-logistik-summary.md, laporan-naive-bayes-summary.md,
laporan-knn-summary.md, laporan-kmeans-summary.md,
prompt-tugas09-text-mining.md, serta laporan-text-mining-summary.md bila
sudah ada. Semua angka, keputusan desain, dan temuan di aplikasi harus
konsisten dengan dokumen tersebut.)

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
   langkah" (meniru subbab worked example di laporan saya) yang hasilnya
   diverifikasi identik dengan implementasi library.

═══════════════════════════════════════
C. MODUL (satu halaman/route per tugas)
═══════════════════════════════════════
Catatan: ringkasan Tugas 01–04 BELUM ada di Project. Saya akan
melampirkan materi/laporannya. Bila tidak ada, buat modul umum di bawah
ini dan tandai jelas sebagai "modul umum (belum terhubung ke laporan)".
JANGAN mengarang angka hasil untuk Tugas 01–04.

Tugas 01–04 (isi sesuai lampiran; fallback modul umum):
 • Pengantar Data Mining & CRISP-DM (penjelasan interaktif alur tahap)
 • Data Understanding & EDA: unggah CSV → profil kolom, tipe data,
   missing value, duplikat, statistik deskriptif, histogram, boxplot,
   korelasi (heatmap)
 • Data Preprocessing: imputasi (mean/median/interpolasi), deteksi
   outlier IQR & winsorization, encoding, scaling (Standard/MinMax),
   before–after
 • (modul ke-4 sesuai lampiran Tugas 04)

Tugas 05 — Regresi Logistik (AAPL/NASDAQ OHLC):
 • Prediksi arah harga hari berikutnya; fitur dari OHLC tanpa look-ahead
 • L2 vs tanpa regularisasi, koefisien/odds ratio + p-value (IRLS),
   VIF, walk-forward CV, tuning C, threshold slider (tunjukkan jebakan
   "F1 optimal" spurious di threshold rendah), kalibrasi, bootstrap CI
   AUC, uji lintas-saham
 • Pesan kunci: AUC AAPL ≈ 0,5 (EMH bentuk lemah), sinyal kecil tapi
   signifikan secara agregat lintas 20 saham

Tugas 06 — Naïve Bayes (tugas kelompok; tampilkan 5 anggota + NIM):
 • Arah IHSG hari berikutnya dari data makro multi-aset
 • Toggle "dengan/tanpa penanganan outlier" untuk menunjukkan dampaknya
   (akurasi 48,48% → 51,87%, Brier 0,461 → 0,256)
 • Worked example Gaussian NB: prior → likelihood per fitur → log-sum-exp
   → posterior → MAP, dibandingkan dengan predict_proba

Tugas 07 — KNN (Instagram Reels Viral Hit vs High Reach):
 • Slider K, pilihan metrik jarak (Euclidean/Manhattan/cosine), bobot
   (uniform/distance), scaler (none/Standard/MinMax) → akurasi CV live
 • Visualisasi tetangga terdekat untuk satu titik uji + perhitungan manual
 • Demo curse of dimensionality dan lazy learner (waktu fit vs prediksi)

Tugas 08 — K-Means (AAPL 2026, data sintetis-terinterpolasi; beri label
peringatan bahwa dataset ini sintetis):
 • Slider K (2–10), init k-means++ vs acak, n_init; Elbow, Silhouette,
   DBI, CH, Gap; scatter PCA berwarna klaster; animasi iterasi K-Means
   langkah demi langkah (assign → update centroid)
 • Kestabilan lintas seed (ARI) sebagai alasan memilih K = 3
 • Profil rezim: Bullish/Reli, Melemah/Konsolidasi, Aktif–Volatil;
   timeline rezim per hari

Tugas 09 — Text Mining (komentar publik terkait MBG):
 • Pipeline preprocessing interaktif: ketik/tempel komentar → tampilkan
   output tiap tahap (filtering, case folding, tokenizing, stopword
   removal dengan negasi dipertahankan, normalisasi slang, stemming
   Sastrawi)
 • Kalkulator TF-IDF manual (TF, DF, IDF klasik vs smooth, L2) yang
   dicocokkan dengan TfidfVectorizer
 • Sentimen leksikon (InSet) + klasifikasi (MNB, LogReg, Linear SVM, KNN);
   pemodelan topik LDA/NMF dengan slider jumlah topik; word cloud,
   tren sentimen mingguan, ko-okurensi
 • Etika: username wajib disamarkan, kata kasar disensor, bahasa netral

Halaman tambahan:
 • Beranda: hero, kartu 9 modul, timeline perkuliahan
 • "Perbandingan Algoritma": tabel kelebihan/kekurangan & kapan dipakai
   (Regresi Logistik, NB, KNN, K-Means, text mining), ditarik dari
   bagian kelebihan–kekurangan tiap laporan
 • "Data Lab" umum: unggah CSV → pilih target → jalankan beberapa
   algoritma sekaligus → leaderboard metrik
 • Tentang & Referensi (APA 7th, hanya referensi yang benar-benar ada di
   laporan saya; jangan menambah referensi karangan)

═══════════════════════════════════════
D. ARSITEKTUR TEKNIS
═══════════════════════════════════════
Karena target deploy adalah hosting statis (Netlify & Vercel tanpa
server Python), semua komputasi berjalan di browser:
 • Frontend: React + Vite + TypeScript, Tailwind CSS, React Router
 • Komputasi ML: Pyodide (Python di WebAssembly) dalam Web Worker,
   memakai numpy, pandas, scikit-learn, scipy yang tersedia di Pyodide;
   paket murni-Python lain (mis. Sastrawi) lewat micropip. Tampilkan
   indikator loading saat Pyodide pertama kali dimuat, dan cache-kan.
 • Grafik: Plotly.js atau Recharts (interaktif, tooltip, ekspor PNG)
 • Kode Python per modul diambil dari skrip tugas saya
   (logistic_regression_nasdaq.py, naive_bayes_ihsg.py,
   knn_instagram_reels.py, kmeans_aapl_stock.py, textmining_mbg.py) dan
   diringkas menjadi fungsi yang bisa dipanggil dari worker. Sertakan
   tombol "Lihat kode Python" di setiap modul.
 • Hasil laporan resmi disimpan sebagai JSON statis di /public/data
   (dari model_summary.json tiap tugas) dan ditampilkan sebagai "Hasil
   Laporan". Hasil yang dihitung ulang di browser diberi label "Hasil
   Live" — keduanya tidak boleh dicampur.
 • Dataset contoh: hanya sertakan yang lisensinya mengizinkan
   redistribusi (cek lisensi Kaggle tiap dataset); data komentar MBG
   hanya versi tersamarkan atau sampel kecil. Bila tidak boleh,
   sediakan tombol unggah saja.
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
 • Logo UNPAM dan logo personal branding saya (versi latar transparan)
   — saya lampirkan file-nya.
 • Warna kelas konsisten di semua grafik (positif/netral/negatif;
   naik/turun; klaster C1–C3), tetap terbedakan untuk buta warna.
 • Bahasa UI: Indonesia (opsional toggle English).

═══════════════════════════════════════
F. REPOSITORI & DEPLOY
═══════════════════════════════════════
1. GitHub
   • Repo publik "datanalysx" di akun GitHub saya (tanyakan username
     bila belum diketahui).
   • Struktur rapi: /src, /public/data, /python (skrip asli per tugas),
     /docs, README.md (deskripsi, screenshot, cara menjalankan lokal,
     link demo), LICENSE (MIT, kecuali dataset), .gitignore.
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
   grafik muncul, unggah CSV jalan, routing langsung ke /tugas-07 tidak
   404), lalu tulis ketiga URL di README.
Langkah yang memerlukan login/otorisasi akun saya (membuat repo,
menghubungkan Netlify/Vercel ke GitHub) harus meminta persetujuan saya
dulu; jangan memasukkan password atau token saya.

═══════════════════════════════════════
G. QUALITY CONTROL
═══════════════════════════════════════
1. Setiap angka "Hasil Laporan" cocok dengan model_summary.json /
   ringkasan Project. Tidak ada angka yang dikarang.
2. Setiap worked example lolos uji kesetaraan dengan library (assert /
   toleransi < 1e-8) — buat unit test (Vitest untuk TS, pytest untuk
   fungsi Python).
3. random_state = 42 di semua modul; hasil live deterministik.
4. Tidak ada kebocoran data pada modul klasifikasi (scaler/TF-IDF di-fit
   hanya pada data latih via Pipeline; split temporal untuk data saham).
5. Lighthouse: Performance ≥ 80 (setelah Pyodide di-cache),
   Accessibility ≥ 90; tidak ada error di console.
6. Uji di Chrome, Firefox, dan mobile.
7. Peringatan jujur ditampilkan: dataset AAPL 2026 sintetis, label
   sentimen otomatis ≠ ground truth, komentar YouTube ≠ opini publik
   Indonesia, hasil bukan saran investasi.

═══════════════════════════════════════
OUTPUT
═══════════════════════════════════════
(1) Link repo GitHub + GitHub Pages, (2) URL Netlify, (3) URL Vercel,
(4) source code lengkap (zip), (5) README dengan screenshot,
(6) ringkasan proyek disimpan ke Project sebagai
claude/datanalysx-summary.md agar bisa dilanjutkan di sesi berikutnya.
