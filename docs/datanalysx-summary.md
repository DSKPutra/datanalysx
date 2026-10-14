# claude/datanalysx-summary.md — Ringkasan proyek DATANALYSX (status 14 Okt 2026)

**Aplikasi:** DATANALYSX — “From Raw Data to Insight — Data Mining Lab”. Lab analisis data & data mining berbasis browser oleh Dea Saka Kurnia Putra. Fokus: alat umum (Eksplorasi, Regresi, Klasifikasi, Clustering, Text Mining) untuk CSV apa pun; sembilan studi kasus sebagai contoh penerapan.

## Tautan
- Repo: https://github.com/DSKPutra/datanalysx (publik, branch `main`)
- GitHub Pages: https://dskputra.github.io/datanalysx/ (Actions `pages.yml`, BASE_PATH=/datanalysx/)
- Vercel: https://datanalysx.vercel.app (proyek `putratalks/datanalysx`; dideploy via CLI)
- Netlify: https://datanalysx.netlify.app (situs dibuat, ID e5e60491-…; deploy DIBLOKIR karena kredit akun tim “Putra Talks” habis)

## Keputusan penting
- Penomoran mengikuti berkas asli: 01 Regresi NVIDIA, 02 Linear vs RF, 03 Regresi sederhana/berganda (gaya widget Orange), 04 Regresi Logistik NASDAQ, 05 Decision Tree (laporan ADA), 06 Naïve Bayes (proyek tim), 07 KNN, 08 K-Means, 09 Text Mining (laporan BELUM ada → tanpa Hasil Laporan).
- Aplikasi berfokus pada alat analisis umum; contoh lama menjadi “Studi Kasus” (route `/studi-kasus-0X`, route lama dialihkan). Atribut institusi dihapus dari aplikasi dan repositori kerja; favicon netral + logo personal branding di footer.
- Pyodide 0.29.5 (scikit-learn 1.7.0). pytest dikunci ke versi yang sama (`requirements-test.txt`) karena sklearn ≥ 1.8 salah mereproduksi studi kasus 01.
- Dataset tidak diredistribusi: NASDAQ OHLC (© pemilik) dan komentar MBG (tanpa izin) → unggah saja. InSet & kamus slang dimuat runtime dari GitHub aslinya.
- “Hasil Laporan” = `public/data/studi-kasus-XX.json` (bersumber tabel), “Hasil Live” = Pyodide; tidak dicampur.

## Kualitas
- `docs/verifikasi-angka.md`: 101/104 angka cocok; selisih: T03 CV Random Forest (RMSE 4,906 / MAE 0,946 vs 4,912 / 0,949, beda versi sklearn), T04 “4.184 baris bersih” (kode: 4.203; observasi 4.183 cocok).
- pytest 24 lulus, Vitest 9 lulus, ESLint/tsc bersih, Lighthouse a11y 100, performa 81–99.

## Temuan audit (sudah ditampilkan di aplikasi)
- T02: berkas = LLM Agent Failure Analysis Benchmark (Apache 2.0), bukan AI Economy Pulse; failure_type membocorkan target (R²≈1).
- T04 & T06: baris terakhir mendapat target 0 walau H+1 tak diketahui; T06 batas IQR dihitung dari seluruh periode (mode “benar” disediakan).

## Sisa pekerjaan (perlu tindakan Anda)
1. Netlify: tambah kredit / pindah tim, lalu Project → Link repository → DSKPutra/datanalysx (build `npm run build`, publish `dist`; netlify.toml sudah ada).
2. Vercel: Account Settings → Authentication → Connect GitHub (Login Connection), lalu Project datanalysx → Settings → Git → Connect DSKPutra/datanalysx agar auto-deploy dari `main`.
3. Bila laporan studi kasus 09 selesai: isi `public/data/studi-kasus-09.json`, tambah `python/original/textmining_mbg.py`, referensinya ke `public/data/references.json`.

## Menjalankan ulang
`npm ci && npm run dev` · uji: `npm run lint && npm run typecheck && npm test && npm run build` · `pip install -r requirements-test.txt && python -m pytest python/tests -q` · regenerasi: `python scripts/build_report_json.py`, `python scripts/make_fixtures.py`, `python scripts/verify_numbers.py`.
