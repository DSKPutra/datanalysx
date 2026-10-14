/* Metadata & narasi tiap modul. Seluruh angka di sini dikutip dari laporan (lihat docs/verifikasi-angka.md). */
import type { SampleInfo } from '../components/DataSource';

export type Group = 'Regresi' | 'Klasifikasi' | 'Clustering' | 'Text Mining';

export interface ModuleInfo {
  no: string;
  slug: string;
  title: string;
  short: string;
  algo: string;
  group: Group;
  pertemuan: string;
  type: 'individu' | 'kelompok';
  question: string;
  dataset: { name: string; detail: string; origin: string; license: string };
  sample: SampleInfo | null;
  crisp: [string, string][];
  findings: string[];
  critical: string[];
  warnings: string[];
  pros: string[];
  cons: string[];
  when: string;
  code: { label: string; path: string; note?: string }[];
  reportAvailable: boolean;
}

const dmx = (n: string, extra: { label: string; path: string; note?: string }[] = []) => [
  ...extra,
  { label: `Modul browser (dmx/t${n}.py)`, path: `code/dmx/t${n}.py`, note: 'Fungsi ringkas yang dijalankan Pyodide di browser — diturunkan dari skrip laporan.' },
];

export const MODULES: ModuleInfo[] = [
  {
    no: '01', slug: 'tugas-01', title: 'Linear Regression: Harga Close Saham NVIDIA', short: 'Regresi linear berganda memprediksi Close NVIDIA dari OHLC, volume, SMA 50, RSI 14.',
    algo: 'Linear Regression', group: 'Regresi', pertemuan: 'Pertemuan 01', type: 'individu',
    question: 'Seberapa akurat Linear Regression mengestimasi harga penutupan (Close) saham NVIDIA dari enam variabel numerik, dan apa makna koefisiennya?',
    dataset: { name: 'NVIDIA Stock Data (1999–2026)', detail: '7.079 baris × 16 kolom, 22 Jan 1999 – 11 Mar 2026; 0 duplikat; missing hanya struktural (stock_split 7.073, key_event 6.826); 5 era bisnis, 6 stock split.', origin: 'Kaggle — Shah (2026), alitaqishah/nvidia-stock-data-19992026-the-ai-mega-stock', license: 'CC0: Public Domain' },
    sample: { url: 'data/samples/t01_nvidia_stock_1999_2026.csv', label: 'NVIDIA Stock Data 1999–2026', license: 'CC0' },
    crisp: [
      ['Business understanding', 'Mengestimasi harga Close dan menafsirkan arah pengaruh tiap variabel.'],
      ['Data understanding', 'Struktur 16 kolom, cek missing (struktural), duplikat (0), statistik deskriptif, korelasi Pearson.'],
      ['Data preparation', 'drop_duplicates, parsing tanggal dd/mm/yyyy, urut kronologis; X = Open, High, Low, Volume, SMA 50, RSI 14; y = Close.'],
      ['Modeling', 'LinearRegression() scikit-learn pada 80% data latih (split acak, random_state = 42).'],
      ['Evaluation', 'MAE, RMSE, R² pada 1.416 baris uji; grafik aktual vs prediksi; koefisien.'],
      ['Deployment/insight', 'Persamaan regresi & interpretasi koefisien; keterbatasan: bukan peramalan.'],
    ],
    findings: ['MAE 0,0047; RMSE 0,0101; R² 0,999994 pada data uji.', 'Low (0,512563) dan High (0,475677) memiliki koefisien terbesar; Volume & RSI 14 nyaris nol.', 'Open berkorelasi 0,999985 dengan Close — Open, High, Low hari yang sama hampir menentukan Close.'],
    critical: [
      'R² 0,999994 adalah estimasi DALAM-HARI, bukan peramalan: Open, High, Low hari yang sama sudah membatasi Close (diakui sendiri di Subbab 4.3 laporan).',
      'Split acak pada deret waktu membuat model “melihat” hari-hari di sekitar data uji (kebocoran temporal ringan). Lab menyediakan split berbasis waktu & lag features sebagai uji realisme.',
      'Pada mode lag (OHLC t … t−4 → Close t+1), R² harga tetap tinggi karena harga bersifat random walk; ukuran yang jujur adalah R² perubahan harga dan skill terhadap baseline naif “Close besok = Close hari ini”.',
      'Volume berorde 10⁸ membuat XᵀX berkondisi sangat buruk; scikit-learn ≥ 1.8 menghasilkan solusi berbeda (MAE 1,34). Aplikasi memakai scikit-learn 1.7.0 (Pyodide) yang mereproduksi laporan.',
    ],
    warnings: ['Bukan saran investasi.'],
    pros: ['Sederhana, cepat, mudah direproduksi (scikit-learn).', 'Koefisien langsung dapat ditafsirkan (ceteris paribus).'],
    cons: ['Fitur hari yang sama → bukan peramalan.', 'Tanpa lag features, encoding kategori, validasi silang, atau pembanding algoritma.'],
    when: 'Hubungan kira-kira linear antara variabel numerik; butuh model yang mudah dijelaskan sebagai baseline.',
    code: dmx('01', [{ label: 'Skrip laporan (Lampiran A)', path: 'code/original/tugas01_regresi_nvidia.py', note: 'Kode Google Colab dari Lampiran A laporan Tugas 01 (penanda # %% Cell N).' }]),
    reportAvailable: true,
  },
  {
    no: '02', slug: 'tugas-02', title: 'Regresi Linear vs Non-Linear: Keparahan Kegagalan Agen AI', short: 'Linear Regression vs Random Forest Regression memprediksi severity_score kegagalan agen AI.',
    algo: 'Linear Regression & Random Forest Regression', group: 'Regresi', pertemuan: 'Pertemuan 02', type: 'individu',
    question: 'Algoritma mana — linear atau non-linear — yang lebih baik mengestimasi tingkat keparahan kegagalan agen AI dari karakteristik tugas?',
    dataset: { name: 'Disebut di laporan: “AI Economy Pulse Dataset Tahun 2022-2026”', detail: '1.500 baris × 10 kolom (task_id, domain, difficulty, prompt, context, expected_answer, agent_answer, failure_type, failure_severity, notes); 0 missing, 0 duplikat.', origin: 'Isi berkas identik byte-per-byte dengan “LLM Agent Failure Analysis Benchmark Dataset” (Kaggle sunil123kumar/ai-agent-failure-benchmark-dataset) — bukan dataset AI Economy Pulse (belbino) yang dikutip laporan.', license: 'Apache 2.0' },
    sample: { url: 'data/samples/t02_llm_agent_failure_benchmark.csv', label: 'LLM Agent Failure Analysis Benchmark (1.500 baris)', license: 'Apache 2.0' },
    crisp: [
      ['Business understanding', 'Mengestimasi keparahan kegagalan agen AI agar tugas berisiko dapat diantisipasi.'],
      ['Data understanding', '1.500 × 10, tanpa missing/duplikat; sebagian besar kolom teks/kategorikal.'],
      ['Data preparation', 'difficulty_score (Easy=1…Hard=3), prompt_length & context_length (jumlah kata), target severity_score (Low=1…Critical=4).'],
      ['Modeling', 'LinearRegression vs RandomForestRegressor(n_estimators=100, random_state=42); split acak 80:20.'],
      ['Evaluation', 'MAE, RMSE, R² pada 300 baris uji; scatter aktual vs prediksi.'],
      ['Deployment/insight', 'Pola non-linear (ambang & interaksi) lebih cocok ditangkap model berbasis pohon.'],
    ],
    findings: ['Linear Regression: MAE 0,624560; RMSE 0,772797; R² 0,139403.', 'Random Forest: MAE 0,198289; RMSE 0,399665; R² 0,769824.', 'severity′ = 1,637357 − 0,078486·difficulty + 0,007658·prompt_length + 0,038742·context_length.'],
    critical: [
      'Target ordinal 1–4 diperlakukan sebagai kontinu; prediksi 2,7 tidak bermakna sebagai kelas — klasifikasi ordinal lebih tepat (saran laporan).',
      'Hanya satu split (random_state = 42); Lab menambahkan validasi silang 5-lipatan sehingga stabilitas dapat dilihat.',
      'Menambahkan domain sedikit membantu; menambahkan failure_type membuat R² ≈ 1 karena keparahan praktis ditentukan oleh jenis kegagalan hasil evaluasi — itu kebocoran label, bukan model yang lebih pintar.',
      'Panjang teks (jumlah kata) hanyalah proksi kasar kompleksitas.',
    ],
    warnings: ['Nama dataset di laporan berbeda dengan asal berkas (lihat tab Data).'],
    pros: ['Random Forest menangkap pola ambang & interaksi tanpa spesifikasi manual.', 'Linear Regression transparan sebagai baseline.'],
    cons: ['Random Forest kurang dapat ditafsirkan dan tak dapat mengekstrapolasi.', 'Linear Regression gagal pada hubungan non-linear (R² 0,139).'],
    when: 'Random Forest Regression untuk hubungan non-linear/berinteraksi pada data tabular; linear sebagai pembanding wajib.',
    code: dmx('02', [{ label: 'Skrip laporan (Lampiran A)', path: 'code/original/tugas02_regresi_linear_rf.py' }]),
    reportAvailable: true,
  },
  {
    no: '03', slug: 'tugas-03', title: 'Regresi Linear Sederhana & Berganda: Industri Semikonduktor', short: 'R&D, capex, dan margin operasi terhadap pendapatan 39 perusahaan chip (2010–2026), bergaya widget Orange.',
    algo: 'Regresi Linear Sederhana & Berganda (OLS)', group: 'Regresi', pertemuan: 'Pertemuan 03', type: 'individu',
    question: 'Seberapa besar belanja R&D dan capex menjelaskan pendapatan perusahaan semikonduktor, dan apakah asumsi regresi terpenuhi?',
    dataset: { name: 'chip_companies_financials.csv — Global Semiconductor Industry 2010-2026', detail: '617 baris, 40 perusahaan; 28 baris bernilai nol dibuang (Graphcore gugur) → 589 baris, 39 perusahaan.', origin: 'Kaggle — sergionefedov/global-semiconductor-industry-2010-2026', license: 'CC0: Public Domain' },
    sample: { url: 'data/samples/t03_chip_companies_financials.csv', label: 'chip_companies_financials.csv', license: 'CC0' },
    crisp: [
      ['Business understanding', 'Mengukur hubungan R&D/capex dengan pendapatan dan kelayakannya untuk prediksi.'],
      ['Data understanding', 'Lima CSV ditinjau; distribusi menceng (skew pendapatan 4,55); korelasi Pearson & Spearman.'],
      ['Data preparation', 'Buang 28 baris bernilai nol; operating_income dikeluarkan (turunan pendapatan → kebocoran); transformasi log.'],
      ['Modeling', 'OLS sederhana & berganda (skala asli & log-log), model efek tetap perusahaan.'],
      ['Evaluation', 'Uji t/F, R², VIF, Breusch–Pagan, Shapiro–Wilk, Durbin–Watson, Cook; hold-out 80:20 & CV 10-lipatan (Test and Score).'],
      ['Deployment/insight', 'Kalkulator prediksi 4 model; rekomendasi log-log berganda.'],
    ],
    findings: ['Sederhana: Pendapatan′ = 5,324 + 4,726·R&D, R² 0,8234; log-log R² 0,8776.', 'Berganda linear R² 0,9691 (margin tidak signifikan, p = 0,358); log-log R² 0,9884, VIF < 1,5.', 'CV 10-lipatan: Linear berganda RMSE 3,837 terbaik; Random Forest MAE 0,949 terbaik — peringkat bergantung metrik.', 'Efek tetap: elastisitas R&D 0,9991; R² within 0,9998.'],
    critical: [
      'R&D dan capex tiap perusahaan hampir merupakan persentase TETAP dari pendapatannya (SD dalam-perusahaan ≤ 0,33 poin): hubungan bersifat definisional, bukan kausal.',
      'Slope gabungan 0,915 adalah artefak percampuran perusahaan dengan intensitas R&D berbeda, bukan “skala hasil menurun”.',
      'Data panel (perusahaan × tahun) → autokorelasi (DW 0,09–0,30); galat baku & p-value terlalu optimistis.',
      'Validasi silang acak menempatkan perusahaan yang sama di latih & uji; leave-one-company-out akan lebih jujur.',
      'Dengan scikit-learn 1.7 (Pyodide) CV Random Forest = RMSE 4,906 / MAE 0,946 (laporan 4,912 / 0,949) — perbedaan implementasi antarversi.',
    ],
    warnings: ['Hubungan definisional, bukan kausal.'],
    pros: ['Koefisien & elastisitas mudah ditafsirkan; uji statistik lengkap.', 'Log-log memulihkan homoskedastisitas (BP p = 0,975 pada model sederhana).'],
    cons: ['Sensitif pada titik berpengaruh (NVIDIA 2026, Cook 13,6).', 'Asumsi independensi dilanggar pada data panel.'],
    when: 'Hubungan linear (atau log-linear) antar variabel kontinu ketika interpretasi koefisien & inferensi dibutuhkan.',
    code: dmx('03', [{ label: 'Skrip laporan (Lampiran 1)', path: 'code/original/tugas03_regresi_semikonduktor.py' }]),
    reportAvailable: true,
  },
  {
    no: '04', slug: 'tugas-04', title: 'Regresi Logistik: Arah Harga Saham NASDAQ (AAPL)', short: 'Prediksi naik/turun hari berikutnya dari 10 fitur OHLC tanpa look-ahead; MLE, walk-forward, bootstrap, uji 20 saham.',
    algo: 'Logistic Regression', group: 'Klasifikasi', pertemuan: 'Pertemuan 04', type: 'individu',
    question: 'Apakah fitur teknikal turunan OHLC mampu memprediksi arah Close hari berikutnya lebih baik dari tebakan acak?',
    dataset: { name: 'NASDAQ Daily Stock Prices — OHLC 1962-01-02 s.d. 2026-09-18', detail: 'Studi kasus AAPL, dibatasi 2010-01-01 s.d. 2026-09-18 → 4.183 observasi; time-based split 80:20 → 3.346 latih / 837 uji. Uji generalisasi pada 20 emiten.', origin: 'Kaggle — svaningelgem/nasdaq-daily-stock-prices', license: 'Data files © Original Authors (tidak boleh diredistribusi)' },
    sample: null,
    crisp: [
      ['Business understanding', 'Menguji klaim “AI memprediksi arah saham” secara statistik yang jujur.'],
      ['Data understanding', 'OHLC harian AAPL; kelas 53,1% naik / 46,9% turun-tetap; fat tails & volatility clustering.'],
      ['Data preparation', 'Pembersihan logika OHLC; 10 fitur (return 1/2/5 hari, ma_gap_5_20, vol_10d, vol_20d, hl_spread, oc_spread, rsi_14, momentum_10); target Close H+1 > Close H; StandardScaler di-fit pada data latih.'],
      ['Modeling', 'LogisticRegression L2, C = 1, lbfgs; MLE tanpa regularisasi via IRLS untuk inferensi.'],
      ['Evaluation', 'Confusion matrix, akurasi, presisi, recall, F1, ROC-AUC, MCC, log-loss, Brier; VIF; walk-forward; tuning C; threshold; kalibrasi; permutation importance; bootstrap CI; lintas 20 saham.'],
      ['Deployment/insight', 'AAPL ≈ acak (EMH bentuk lemah); sinyal agregat tipis tapi signifikan secara statistik.'],
    ],
    findings: ['Akurasi 51,37%, balanced accuracy 49,18%, F1 62,28%, ROC-AUC 0,484 (CI bootstrap [0,443; 0,525]).', 'MLE: semua p fitur > 0,14; LR-test p = 0,4632; McFadden R² 0,0021.', 'Walk-forward AUC 0,4925 ± 0,0175; C optimal 0,001; Brier 0,2511.', 'Lintas 20 saham: rata-rata AUC 0,5156, uji-t p = 0,0075, Wilcoxon p = 0,0121, 14/20 di atas 0,5.'],
    critical: [
      '“F1 optimal” 0,7053 di threshold 0,05 adalah optimum semu: model memprediksi “naik” untuk hampir semua hari (recall 100%).',
      'Signifikansi statistik lintas saham (≈1,6 poin AUC) hampir pasti habis oleh biaya transaksi — tidak ekonomis.',
      'Audit kode: baris terakhir memperoleh target 0 walau Close H+1 tidak diketahui (NaN > x = False). Lab menyediakan opsi untuk membuangnya.',
      'Laporan menyebut 4.184 baris bersih; kode menghasilkan 4.203 baris bersih → 4.183 observasi (observasi cocok).',
    ],
    warnings: ['Bukan saran investasi.', 'Dataset NASDAQ tidak disertakan (hak cipta pemilik): unggah CSV per ticker (kolom ticker,date,open,high,low,close).'],
    pros: ['Probabilitas & odds ratio mudah ditafsirkan; inferensi formal (SE, z, LR-test).', 'Cepat, baseline kuat, regularisasi menstabilkan koefisien.'],
    cons: ['Asumsi log-odds linear; kurang menangkap interaksi.', 'Asumsi independensi dilanggar pada deret waktu.'],
    when: 'Klasifikasi biner yang butuh probabilitas terkalibrasi dan interpretasi; baseline sebelum model kompleks.',
    code: dmx('04', [{ label: 'logistic_regression_nasdaq.py', path: 'code/original/logistic_regression_nasdaq.py', note: 'Skrip lengkap dari Lampiran 1 laporan Tugas 04.' }]),
    reportAvailable: true,
  },
  {
    no: '05', slug: 'tugas-05', title: 'Decision Tree: Kategori Topik Trending 2026', short: 'Pohon keputusan pada metadata vs kata judul: 20% vs 100% — cermin data, bukan pencipta pola.',
    algo: 'Decision Tree (CART, Gini/Entropy)', group: 'Klasifikasi', pertemuan: 'Pertemuan 05', type: 'individu',
    question: 'Seberapa jauh pohon keputusan dapat dipercaya ketika data mungkin tidak menyimpan pola?',
    dataset: { name: 'trending_topics_2026_synthetic.csv (Global Trending Topics 2026 – Multi-Source Dataset)', detail: '2.500 baris × 14 kolom, 1–28 Feb 2026, 5 kelas seimbang (17,8–21,8%). Data SINTETIS.', origin: 'Laporan: “disediakan dosen”; berkas identik tersedia di Kaggle mmuneeb5522/crossplatform-trending-topics-2026multilanguage', license: 'CC0: Public Domain' },
    sample: { url: 'data/samples/t05_trending_topics_2026_synthetic.csv', label: 'Trending Topics 2026 (sintetis)', license: 'CC0' },
    crisp: [
      ['Business understanding', 'Klasifikasi otomatis kategori topik untuk platform konten.'],
      ['Data understanding', 'Tanpa missing/duplikat; atribut numerik berdistribusi seragam (indikasi sintetis); uji chi-kuadrat & Cramér’s V.'],
      ['Data preparation', 'topic_subcategory & short_text dibuang (bocor label); Skenario 1 = 17 fitur metadata; Skenario 2 = 52 kata biner judul.'],
      ['Modeling', 'Pohon default Orange vs hasil grid search; pembanding Random Forest & Regresi Logistik.'],
      ['Evaluation', 'Hold-out berstrata 80:20, CV 10-lipatan, ROC one-vs-rest, uji permutasi, pruning, uji judul baru.'],
      ['Deployment/insight', 'Aturan JIKA–MAKA 4 kata; peringatan kerapuhan di luar templat.'],
    ],
    findings: ['Skenario 1: pohon default 1.289 simpul, akurasi latih 77,6% vs uji 20,0% (< mayoritas 21,8%); uji permutasi p = 0,970.', 'Skenario 2: 9 simpul, kedalaman 3, akurasi & AUC 1,000 (judul dari 5 templat per kategori).', 'Entropy akar 2,3190 bit; IG kata “for” 0,9546 bit.', 'Judul baru di luar templat: 5/8 benar, semuanya dengan keyakinan 100%.'],
    critical: [
      'Akurasi 100% mengenali templat, bukan makna; uji di luar distribusi wajib.',
      'Feature importance (engagement 0,486) pada data tanpa sinyal hanyalah artefak overfitting.',
      'Dataset sintetis: hasil tidak boleh digeneralisasi ke data trending nyata.',
    ],
    warnings: ['Dataset sintetis.'],
    pros: ['Mudah dibaca (aturan JIKA–MAKA).', 'Tanpa scaling; seleksi atribut implisit; cepat.'],
    cons: ['Rentan overfitting & tidak stabil.', 'Bias ke atribut dengan banyak titik pisah; prediksi terlalu yakin.'],
    when: 'Saat keterbacaan aturan penting; gunakan pruning/ensemble dan uji permutasi bila sinyal diragukan.',
    code: dmx('05', [{ label: 'Skrip laporan (Lampiran A)', path: 'code/original/tugas05_decision_tree.py' }]),
    reportAvailable: true,
  },
  {
    no: '06', slug: 'tugas-06', title: 'Naïve Bayes: Arah IHSG dari Data Makro Multi-Aset', short: 'Gaussian NB memprediksi arah IHSG esok hari; dampak outlier USD/IDR +902% terhadap akurasi & kalibrasi.',
    algo: 'Gaussian Naïve Bayes', group: 'Klasifikasi', pertemuan: 'Pertemuan 06', type: 'kelompok',
    question: 'Dapatkah indikator makro-finansial hari ini memprediksi arah IHSG esok hari, dan seberapa sensitif Naïve Bayes terhadap outlier?',
    dataset: { name: 'Daily Multi-Asset Macroeconomic and Financial Time Series Dataset (Indonesia/Global Markets)', detail: '4.277 observasi harian, 2010-01-01 – 2026-05-29; OIL, GOLD, USDIDR, SP500, IHSG, VIX, CPI, BI_rate, US_rate.', origin: 'Kaggle — raphaelnazareth/indonesia-financial-time-series-dataset-2010-2026', license: 'CC BY-SA 4.0' },
    sample: { url: 'data/samples/t06_indonesia_multiasset_macro_2010_2026.csv', label: 'Indonesia Financial Time Series 2010–2026', license: 'CC BY-SA 4.0 (atribusi: Raphael Nazareth)' },
    crisp: [
      ['Business understanding', 'Klasifikasi arah IHSG hari berikutnya dari limpahan sentimen global & makro.'],
      ['Data understanding', 'Tren ternormalisasi, korelasi (SP500_ret–VIX_chg −0,81), anomali USDIDR +902,9%.'],
      ['Data preparation', 'Interpolasi linear, return harian %, VIX_chg, rate_diff, CPI_chg, winsorization IQR 3×; target IHSG_ret(t+1) > 0.'],
      ['Modeling', 'GaussianNB tanpa scaling (prior, μ, σ² per kelas).'],
      ['Evaluation', 'Split temporal 80:20, metrik, ROC/PR, kalibrasi, TimeSeriesSplit 5-lipatan, baseline mayoritas.'],
      ['Deployment/insight', 'Konsisten dengan EMH bentuk lemah; kualitas data > kompleksitas algoritma.'],
    ],
    findings: ['Winsorization: akurasi 48,48% → 51,87%; Brier 0,461 → 0,256.', 'Uji: ROC-AUC 0,569; F1 0,5735; baseline mayoritas 47,55%.', 'TimeSeriesSplit: AUC 0,5994 ± 0,0292.', 'Contoh 21-02-2023: P(Turun/Tetap) 90,24% = predict_proba.'],
    critical: [
      'Batas IQR di laporan dihitung dari SELURUH periode (termasuk data uji) — kebocoran ringan; Lab menyediakan mode “batas dari data latih saja”.',
      'Baris terakhir ikut dengan target 0 walau IHSG esok belum diketahui (replikasi laporan); mode “benar” membuangnya.',
      'Asumsi independensi dilanggar (korelasi −0,81) → probabilitas cenderung terlalu yakin.',
      'Kinerja tidak stasioner antarperiode (structural break).',
    ],
    warnings: ['Bukan saran investasi.'],
    pros: ['Sangat cepat (O(n·d)) dan parameter mudah ditafsirkan.', 'Tetap memberi sinyal di atas acak meski asumsi dilanggar.'],
    cons: ['Sangat sensitif terhadap outlier (varians membengkak).', 'Asumsi Gaussian & independensi sering tidak realistis pada return finansial.'],
    when: 'Baseline klasifikasi cepat, data kecil–besar, terutama teks (Multinomial) atau fitur yang mendekati independen.',
    code: dmx('06', [{ label: 'naive_bayes_ihsg.py', path: 'code/original/naive_bayes_ihsg.py', note: 'Skrip lengkap dari Lampiran A laporan Tugas 06 (Kelompok 4).' }]),
    reportAvailable: true,
  },
  {
    no: '07', slug: 'tugas-07', title: 'KNN: Instagram Reels “Viral Hit” vs “High Reach”', short: 'KNN bebas leakage dengan Pipeline; scaling, metrik jarak, K, lazy learner, curse of dimensionality.',
    algo: 'K-Nearest Neighbors', group: 'Klasifikasi', pertemuan: 'Pertemuan 07', type: 'individu',
    question: 'Dapatkah KNN membedakan Reels “Viral Hit” (virality_score ≥ 75) dari “High Reach” hanya dari informasi saat unggah?',
    dataset: { name: 'Instagram Reels: Virality & Engagement Trends 2026', detail: '88 unggahan, 9 kreator, 5 niche; 39 Viral Hit / 49 High Reach; split 66 latih / 22 uji.', origin: 'Kaggle — harishyadav0506/instagram-reels-virality-and-engagement-trends-2026', license: 'CC0: Public Domain' },
    sample: { url: 'data/samples/t07_instagram_reels_knn_clean.csv', label: 'instagram_reels_knn_clean.csv (88 baris)', license: 'CC0' },
    crisp: [
      ['Business understanding', 'Memperkirakan viralitas untuk kreator & pemasar.'],
      ['Data understanding', 'Kolom pembentuk target (views, likes, comments, engagement, virality_score) diidentifikasi sebagai leakage.'],
      ['Data preparation', '12 fitur (durasi, caption, hashtag, emoji, mention, usia, jam siklik, akhir pekan, tanya, niche); Pipeline: IQR capping → scaler; one-hot niche.'],
      ['Modeling', 'GridSearchCV 500 kombinasi (12.500 fit): MinMaxScaler, cosine, K = 5, distance.'],
      ['Evaluation', 'CV berulang 5×5 pada data latih; data uji terstratifikasi; pembanding LR, GNB, DT, RF.'],
      ['Deployment/insight', 'Niche fitur paling berpengaruh; KNN sebagai “cari unggahan serupa”.'],
    ],
    findings: ['Akurasi CV (latih) 88,46%; data uji 77,27%, presisi 72,73%, recall 80,00%, F1 0,762, ROC-AUC 0,667.', 'Scaling pada K = 5: tanpa 78,25% → Standard 79,98% → MinMax 87,11%.', 'Perhitungan manual (baris uji 34) identik dengan scikit-learn (selisih 1,1×10⁻¹⁶).', 'Kontras jarak 5,69 → 0,27 saat dimensi bertambah.'],
    critical: [
      'n = 88 dan 4 kreator mendominasi (83 unggahan): model cenderung mengenali identitas kreator/niche.',
      'Data uji hanya 22 baris → CI akurasi lebar; validasi GroupKFold per kreator disarankan.',
      'KNN lebih baik sebagai pengklasifikasi daripada pemeringkat probabilitas (AUC uji 0,667 < GNB).',
    ],
    warnings: [],
    pros: ['Transparan (bisa dihitung tangan), tanpa fase pelatihan.', 'Non-parametrik, menangkap batas keputusan kompleks.'],
    cons: ['Sensitif terhadap skala & metrik; biaya prediksi tumbuh dengan data.', 'Melemah pada dimensi tinggi (curse of dimensionality).'],
    when: 'Data kecil–menengah berdimensi rendah, setelah scaling yang tepat; pencarian kemiripan.',
    code: dmx('07', [{ label: 'knn_instagram_reels.py', path: 'code/original/knn_instagram_reels.py', note: 'Skrip asli Tugas 07.' }]),
    reportAvailable: true,
  },
  {
    no: '08', slug: 'tugas-08', title: 'K-Means: Rezim Pasar Harian AAPL 2026', short: 'Pengelompokan 116 hari perdagangan menjadi 3 rezim; pemilihan K berbasis kestabilan.',
    algo: 'K-Means (Lloyd, k-means++)', group: 'Clustering', pertemuan: 'Pertemuan 08', type: 'individu',
    question: 'Rezim pasar apa yang dapat diidentifikasi K-Means dari perilaku harian saham AAPL, dan seberapa andal?',
    dataset: { name: 'Apple Inc. (AAPL) Stock Price Dataset 2026', detail: '136 hari perdagangan (2 Jan – 20 Jul 2026) × 15 kolom; semi-sintetis menurut penyusunnya; 116 hari setelah pemanasan fitur.', origin: 'Kaggle — Sung (2026), crystalbaby/apple-inc-aapl-stock-price-dataset-2026', license: 'Apache 2.0' },
    sample: { url: 'data/samples/t08_aapl_2026_stock_data.csv', label: 'AAPL 2026 (semi-sintetis)', license: 'Apache 2.0' },
    crisp: [
      ['Business understanding', 'Segmentasi deskriptif hari perdagangan menjadi rezim pasar.'],
      ['Data understanding', 'Audit hari libur bursa, konsistensi OHLC, kolom turunan; data semi-sintetis.'],
      ['Data preparation', '10 fitur stasioner tanpa look-ahead (6 inti: log_ret, range_pct, gap_pct, rel_vol, volat_10, dist_ma20); StandardScaler.'],
      ['Modeling', 'K-Means K = 3, k-means++, n_init = 100.'],
      ['Evaluation', 'Elbow, Silhouette, DBI, CH, Gap; kestabilan ARI lintas seed; pembanding Ward, GMM, DBSCAN.'],
      ['Deployment/insight', 'Rezim Bullish/Reli, Melemah/Konsolidasi, Aktif–Volatil — deskriptif, bukan sinyal trading.'],
    ],
    findings: ['Tiga rezim: Bullish/Reli 41 hari (return +1,92%), Melemah/Konsolidasi 52 hari (−0,61%), Aktif–Volatil 23 hari (volume 1,39×).', 'Silhouette 0,210 (tertinggi), DBI 1,552, CH 29,65; inertia 456,45.', 'n_init = 1: hanya 12% percobaan k-means++ mencapai solusi terbaik; n_init = 100: 50/50 seed identik.', 'K = 4 ditolak: dua solusi berbeda isi (ARI 0,49) dengan WCSS terpaut 0,07%.'],
    critical: [
      'Seluruh silhouette < 0,25 → struktur klaster lemah; DBSCAN hanya menemukan 1 klaster.',
      'Rezim tidak berkaitan dengan return esok hari (Kruskal p = 0,512) dan tidak stabil antarperiode.',
      'Dataset semi-sintetis: hanya laboratorium metodologis.',
    ],
    warnings: ['Dataset AAPL 2026 bersifat sintetis/semi-sintetis.', 'Bukan saran investasi.'],
    pros: ['Sederhana, cepat, selalu konvergen, mudah ditafsirkan.'],
    cons: ['K harus ditentukan; peka inisialisasi, skala, outlier; mengasumsikan klaster bulat; selalu “menemukan” klaster.'],
    when: 'Segmentasi data numerik berskala setara dengan klaster kira-kira bulat; selalu periksa kestabilan.',
    code: dmx('08', [{ label: 'kmeans_aapl_stock.py', path: 'code/original/kmeans_aapl_stock.py', note: 'Skrip asli Tugas 08.' }]),
    reportAvailable: true,
  },
  {
    no: '09', slug: 'tugas-09', title: 'Text Mining: Komentar Publik terkait MBG', short: 'Preprocessing Bahasa Indonesia, TF-IDF, sentimen leksikon InSet, klasifikasi, LDA/NMF — laporan belum tersedia.',
    algo: 'Text Mining (TF-IDF, InSet, MNB/LR/SVM/KNN, LDA/NMF)', group: 'Text Mining', pertemuan: 'Pertemuan 09', type: 'individu',
    question: 'Bagaimana sentimen dan topik percakapan di kolom komentar terkait program Makan Bergizi Gratis?',
    dataset: { name: 'Indonesia MBG Sentiment Analysis (komentar-mbg.csv, MBG_sentiment_analysis.csv)', detail: 'komentar-mbg.csv: 15.000 baris × 4 kolom (author, comment, likes, published_at), 17 Feb – 24 Jun 2026; 3.496 duplikat penuh; 11.375 komentar unik.', origin: 'Kaggle — neevvv/indonesia-mbg-sentiment-analysis', license: '“Other (specified in description)” — tidak ada izin redistribusi eksplisit' },
    sample: null,
    crisp: [
      ['Business understanding', 'Memetakan sentimen & topik komentar (bukan opini seluruh masyarakat).'],
      ['Data understanding', 'Duplikat penuh vs duplikat teks, panjang komentar, emoji/URL/mention, volume mingguan, likes.'],
      ['Data preparation', 'Filtering → case folding → tokenizing → stopword (negasi dipertahankan) → normalisasi slang → stemming Sastrawi.'],
      ['Modeling', 'Label leksikon InSet; MNB, Logistic Regression, Linear SVM, KNN (Pipeline TF-IDF); LDA/NMF.'],
      ['Evaluation', 'Akurasi, F1-makro, confusion matrix, ROC-AUC; koherensi topik (UMass).'],
      ['Deployment/insight', 'Etika: pseudonim, sensor kata kasar, bahasa netral.'],
    ],
    findings: ['Laporan Tugas 09 belum tersedia — tidak ada angka Hasil Laporan.'],
    critical: [
      'Label sentimen otomatis (leksikon) ≠ ground truth; model yang dilatih pada label leksikon pada dasarnya meniru leksikon (sirkular).',
      'Komentar YouTube ≠ opini publik Indonesia; hanya ±10% komentar unik menyebut MBG secara eksplisit.',
      'Kata negasi harus dipertahankan saat stopword removal agar “tidak bagus” tidak berubah makna.',
    ],
    warnings: ['Label sentimen otomatis ≠ ground truth.', 'Komentar YouTube ≠ opini publik Indonesia.'],
    pros: ['Mengubah teks tak terstruktur menjadi fitur terukur; skala besar.', 'Leksikon tidak butuh data berlabel.'],
    cons: ['Sangat bergantung kualitas preprocessing & kamus; sarkasme, campur kode, dan typo sulit.', 'Label otomatis bias; topik perlu tafsir manusia.'],
    when: 'Analisis pola besar pada teks; validasi dengan sampel beranotasi manusia sebelum menarik kesimpulan.',
    code: dmx('09', []),
    reportAvailable: false,
  },
];

export const bySlug = (slug: string) => MODULES.find((m) => m.slug === slug);
export const GROUPS: { name: Group; range: string; blurb: string }[] = [
  { name: 'Regresi', range: '01–03', blurb: 'Memprediksi nilai numerik.' },
  { name: 'Klasifikasi', range: '04–07', blurb: 'Memprediksi kelas/kategori.' },
  { name: 'Clustering', range: '08', blurb: 'Menemukan kelompok tanpa label.' },
  { name: 'Text Mining', range: '09', blurb: 'Menggali makna dari teks.' },
];
