# DATANALYSX — ringkasan progres (2026-10-14)

## Selesai
- Kerangka Vite+React+TS+Tailwind (package.json, config, ESLint), git init (belum di-push).
- Skrip asli Tugas 01–08 diekstrak ke `python/original/` (T09 belum ada laporan/skrip).
- Paket Python `python/dmx/` (t01–t09, eda, datalab) untuk Pyodide.
- pytest `python/tests/` — 18/18 lulus dengan versi pustaka = Pyodide 0.29.5 (`requirements-test.txt`):
  reproduksi angka laporan T01–T08 + kesetaraan perhitungan manual (< 1e-8).
- Dataset contoh berlisensi di `public/data/samples/`; T04 (NASDAQ) & T09 (MBG) unggah-saja.
- `public/data/report/t07|t08_model_summary.json`; logo transparan di `public/img/`.
- Worker/klien Pyodide: `src/lib/pyodide/`.

## Temuan yang perlu dicatat di docs/verifikasi-angka.md
- scikit-learn ≥1.8 salah hitung LinearRegression T01 (Volume 1e8); 1.7.0 & numpy lstsq cocok laporan.
- T03 CV Random Forest MAE: laporan 0,949; sklearn 1.7 → 0,946.
- T04: laporan menyebut 4.184 baris bersih; kode menghasilkan 4.203 (observasi 4.183 cocok).
- T04 & T06 skrip asli: baris terakhir mendapat target 0 walau H+1 tak diketahui; T06 winsorize IQR pakai seluruh periode (opsi "train" disediakan).
- T02: CSV identik dengan "LLM Agent Failure Analysis Benchmark" (Apache 2.0), bukan AI Economy Pulse (belbino).
- T02: failure_type membuat R²≈1 (kebocoran).

## Belum dikerjakan
- Seluruh UI React (halaman Beranda, /tugas-01…09, Regression Lab, Data Lab, Perbandingan,
  Pelajaran Metodologi, Tentang & Referensi), pustaka manual TS + Vitest + fixtures,
  scripts/sync-assets.mjs & postbuild.mjs, index.html, sw.js, JSON Hasil Laporan T01–T06,
  netlify.toml, vercel.json, GitHub Actions, README, LICENSE, verifikasi-angka.md.
- Deploy (sudah disetujui): repo DSKPutra/datanalysx + Pages, Netlify, Vercel.
