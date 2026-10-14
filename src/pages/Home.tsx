import { Link } from 'react-router-dom';
import { GROUPS, MODULES } from '../content/modules';
import { useEffect } from 'react';

const JOURNEY = [
  ['01', 'Regresi pada harga NVIDIA', 'R² 0,999994 — tapi Open/High/Low hari yang sama membuatnya estimasi dalam-hari, bukan ramalan.'],
  ['02', 'Linear vs non-linear', 'Random Forest (R² 0,770) jauh di atas linear (0,139): hubungan berambang & berinteraksi.'],
  ['03', 'Inferensi & asumsi', 'Uji t/F, VIF, BP, DW; operating_income dibuang karena bocor; hubungan R&D–pendapatan ternyata definisional.'],
  ['04', 'Klasifikasi biner dengan disiplin waktu', 'Belajar dari Tugas 01: split berbasis waktu, walk-forward, bootstrap. AAPL ≈ acak (AUC 0,484).'],
  ['05', 'Pohon & sinyal palsu', '20% vs 100% dari kode yang sama: model adalah cermin data; uji permutasi p = 0,970.'],
  ['06', 'Probabilistik & outlier', 'Satu anomali USD/IDR +902% menurunkan akurasi ke 48,48%; winsorization → 51,87%.'],
  ['07', 'Jarak, skala, dimensi', 'MinMax + cosine menaikkan CV ke 88,46%; Pipeline mencegah kebocoran.'],
  ['08', 'Tanpa label: clustering', 'K = 3 dipilih karena kestabilan (ARI), bukan satu indeks; rezim hanya deskriptif.'],
  ['09', 'Teks', 'Preprocessing Bahasa Indonesia, negasi dipertahankan, label leksikon ≠ ground truth.'],
];

export default function Home() {
  useEffect(() => { document.title = 'DATANALYSX — From Raw Data to Insight'; }, []);
  return (
    <div className="space-y-12">
      <section className="relative overflow-hidden rounded-3xl p-8 md:p-12 text-white" style={{ background: 'linear-gradient(135deg, #062E22 0%, #0B4F3A 55%, #0B7A53 100%)' }}>
        <p className="text-sm font-semibold uppercase tracking-[0.2em]" style={{ color: '#E9C46A' }}>Data Mining · Magister Teknik Informatika · Universitas Pamulang</p>
        <h1 className="mt-3 text-5xl font-bold md:text-7xl">DATANALYSX</h1>
        <p className="mt-2 font-display text-2xl md:text-3xl">From Raw Data to Insight — Data Mining Lab</p>
        <p className="mt-4 max-w-2xl text-white/85">
          Portofolio interaktif sembilan tugas mata kuliah Data Mining: ringkasan laporan, grafik interaktif, lab yang menjalankan
          Python (scikit-learn) langsung di browser Anda, dan perhitungan manual langkah demi langkah yang diverifikasi identik dengan pustaka.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link to="/tugas-01" className="btn bg-white !text-[#0B4F3A] no-underline">Mulai dari Tugas 01</Link>
          <Link to="/regression-lab" className="btn border border-white/50 text-white no-underline">Regression Lab</Link>
          <Link to="/data-lab" className="btn border border-white/50 text-white no-underline">Data Lab</Link>
        </div>
        <p className="mt-6 text-sm text-white/75">Dea Saka Kurnia Putra · NIM 261012000111 · Dosen: Dr. Ir. Agung Budi Susanto, M.M. · T.A. 2026/2027</p>
      </section>

      {GROUPS.map((g) => (
        <section key={g.name} aria-labelledby={`g-${g.name}`}>
          <div className="mb-3 flex items-baseline gap-3">
            <h2 id={`g-${g.name}`} className="text-3xl font-bold">{g.name}</h2>
            <span className="text-sm muted">Tugas {g.range} · {g.blurb}</span>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {MODULES.filter((m) => m.group === g.name).map((m) => (
              <Link key={m.slug} to={`/${m.slug}`} className="card group block no-underline transition hover:-translate-y-0.5" style={{ color: 'var(--text)' }}>
                <div className="flex items-center justify-between">
                  <span className="font-display text-4xl font-bold" style={{ color: 'var(--brand-2)' }}>{m.no}</span>
                  <span className="chip" style={{ background: 'var(--surface-2)' }}>{m.algo.split(' (')[0]}</span>
                </div>
                <h3 className="mt-2 text-xl font-semibold leading-snug">{m.title}</h3>
                <p className="mt-1 text-sm muted">{m.short}</p>
                {!m.reportAvailable && <p className="mt-2 inline-block rounded px-1.5 text-xs font-semibold" style={{ background: 'var(--warn-bg)' }}>Laporan belum tersedia — lab tetap dapat dipakai</p>}
              </Link>
            ))}
          </div>
        </section>
      ))}

      <section aria-labelledby="journey">
        <h2 id="journey" className="mb-1 text-3xl font-bold">Benang merah perkuliahan</h2>
        <p className="mb-4 text-sm muted">Dari memprediksi nilai → memprediksi kelas → menemukan kelompok → menggali teks, dengan pelajaran yang terbawa antartugas.</p>
        <ol className="relative space-y-4 border-l-2 pl-6" style={{ borderColor: 'var(--brand-2)' }}>
          {JOURNEY.map(([no, t, d]) => (
            <li key={no} className="relative">
              <span className="on-brand absolute -left-[34px] top-0.5 flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold">{no}</span>
              <p className="font-semibold"><Link to={`/tugas-${no}`}>Pertemuan {Number(no)} — {t}</Link></p>
              <p className="text-sm muted">{d}</p>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
