import { Link } from 'react-router-dom';
import { GROUPS, MODULES } from '../content/modules';
import { useEffect, type JSX } from 'react';

type IconName = 'explore' | 'regression' | 'classification' | 'clustering' | 'text' | 'compass';

/** Ikon garis 24×24 (mengikuti currentColor). */
function Icon({ name }: { name: IconName }) {
  const paths: Record<IconName, JSX.Element> = {
    explore: <><circle cx="10.5" cy="10.5" r="6.5" /><path d="M15.5 15.5 21 21" /><path d="M8 12.5v-2M10.5 12.5V8M13 12.5V9.5" /></>,
    regression: <><path d="M3 21h18M3 21V3" /><path d="M5 17 19 6" /><circle cx="7" cy="14" r="1" /><circle cx="10" cy="13" r="1" /><circle cx="12" cy="9.5" r="1" /><circle cx="16" cy="9" r="1" /><circle cx="17.5" cy="5.5" r="1" /></>,
    classification: <><path d="M3 3h18v18H3z" /><path d="M3 17C9 15 13 9 21 7" /><circle cx="7" cy="8" r="1.2" /><circle cx="10" cy="6" r="1.2" /><path d="M14 16l2 2M16 16l-2 2M18 13l1.6 1.6M19.6 13 18 14.6" /></>,
    clustering: <><circle cx="7" cy="8" r="3.5" /><circle cx="16.5" cy="7" r="2.5" /><circle cx="13" cy="17" r="4" /><path d="M7 8h.01M16.5 7h.01M13 17h.01" /></>,
    text: <><path d="M6 3h9l4 4v14H6z" /><path d="M15 3v4h4" /><path d="M9 11h7M9 14h7M9 17h4" /></>,
    compass: <><circle cx="12" cy="12" r="9" /><path d="m15.5 8.5-2 5-5 2 2-5z" /></>,
  };
  return (
    <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {paths[name]}
    </svg>
  );
}

const TOOLS: [string, string, string, IconName][] = [
  ['/eksplorasi', 'Eksplorasi & praproses', 'Profil kolom, missing value, outlier, korelasi, serta praproses before–after untuk CSV apa pun.', 'explore'],
  ['/regresi', 'Regresi', 'Bandingkan 7 model (Linear, Ridge, Lasso, Tree, Random Forest, kNN, log-log), uji asumsi klasik, dan regresi manual.', 'regression'],
  ['/klasifikasi', 'Klasifikasi', 'Logistic Regression, Naïve Bayes, KNN, Decision Tree, Random Forest: metrik, ROC, leaderboard, dan perhitungan manual.', 'classification'],
  ['/clustering', 'Clustering', 'K-Means dengan pemilihan K (elbow, siluet, DBI), PCA, profil klaster, animasi Lloyd, dan uji kestabilan.', 'clustering'],
  ['/text-mining', 'Text Mining', 'Preprocessing Bahasa Indonesia, TF-IDF manual, sentimen leksikon, klasifikasi teks, dan topic modeling.', 'text'],
  ['/perbandingan', 'Panduan memilih algoritma', 'Kelebihan, kekurangan, dan kapan memakai tiap algoritma — dirangkum dari studi kasus.', 'compass'],
];

/** Titik sebar deterministik (LCG) untuk ilustrasi regresi di hero. */
const POINTS = (() => {
  let seed = 7;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  return Array.from({ length: 34 }, (_, i) => {
    const x = 30 + (i / 33) * 330 + (rnd() - 0.5) * 14;
    const y = 220 - (x - 30) * 0.5 + (rnd() - 0.5) * 70;
    return [Math.round(x), Math.round(Math.max(24, Math.min(236, y)))];
  });
})();

function HeroViz() {
  return (
    <div className="term hud-corners float-slow relative rounded-2xl p-4" aria-hidden>
      <div className="mb-2 flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.2em]" style={{ color: '#8fa3b8' }}>
        <span>model.fit(X, y)</span><span style={{ color: '#34e3a4' }}>● live</span>
      </div>
      <svg viewBox="0 0 390 260" className="w-full">
        <defs>
          <linearGradient id="hv-line" x1="0" x2="1"><stop offset="0" stopColor="#34e3a4" /><stop offset="1" stopColor="#22d3ee" /></linearGradient>
        </defs>
        {[60, 120, 180, 240].map((y) => <line key={y} x1="20" x2="380" y1={y} y2={y} stroke="rgba(148,163,184,.12)" />)}
        {[80, 160, 240, 320].map((x) => <line key={x} y1="16" y2="244" x1={x} x2={x} stroke="rgba(148,163,184,.12)" />)}
        {POINTS.map(([x, y], i) => (
          <circle key={i} className="viz-pt" cx={x} cy={y} r="4" fill={i % 5 === 0 ? '#a78bfa' : '#22d3ee'} fillOpacity=".85"
            style={{ animationDelay: `${0.03 * i}s`, transformBox: 'fill-box', transformOrigin: 'center' }} />
        ))}
        <line className="viz-line" x1="30" y1="220" x2="362" y2="54" stroke="url(#hv-line)" strokeWidth="3" strokeLinecap="round" />
      </svg>
      <div className="mt-2 grid grid-cols-3 gap-2 font-mono text-[11px]">
        {[['R²', '0,8231'], ['RMSE', '4,912'], ['n', '1.284']].map(([k, v]) => (
          <div key={k} className="rounded-lg border px-2 py-1.5" style={{ borderColor: 'rgba(148,163,184,.18)' }}>
            <div style={{ color: '#8fa3b8' }}>{k}</div><div className="text-sm font-semibold" style={{ color: '#e6edf6' }}>{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

const JOURNEY = [
  ['01', 'Regresi pada harga NVIDIA', 'R² 0,999994 — tapi Open/High/Low hari yang sama membuatnya estimasi dalam-hari, bukan ramalan.'],
  ['02', 'Linear vs non-linear', 'Random Forest (R² 0,770) jauh di atas linear (0,139): hubungan berambang & berinteraksi.'],
  ['03', 'Inferensi & asumsi', 'Uji t/F, VIF, BP, DW; operating_income dibuang karena bocor; hubungan R&D–pendapatan ternyata definisional.'],
  ['04', 'Klasifikasi biner dengan disiplin waktu', 'Belajar dari Studi Kasus 01: split berbasis waktu, walk-forward, bootstrap. AAPL ≈ acak (AUC 0,484).'],
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
      <section className="hero rounded-3xl p-7 md:p-12">
        <div className="relative grid grid-cols-1 items-center gap-10 lg:grid-cols-[1.25fr_1fr]">
          <div className="min-w-0">
            <p className="font-mono text-[11px] font-semibold uppercase tracking-[0.2em] sm:text-xs sm:tracking-[0.25em]" style={{ color: '#34e3a4' }}>// data analysis · data mining · machine learning</p>
            <h1 className="mt-4 text-5xl font-bold leading-[1.02] md:text-7xl">
              <span className="text-gradient">DATANALYSX</span>
            </h1>
            <p className="mt-3 font-display text-2xl font-medium md:text-3xl">From Raw Data to Insight.</p>
            <p className="hero-muted mt-4 max-w-xl">
              Unggah CSV Anda lalu eksplorasi, latih, dan bandingkan model regresi, klasifikasi, clustering, dan text mining dengan
              Python (scikit-learn) yang berjalan langsung di browser — lengkap dengan perhitungan manual yang diverifikasi identik dengan pustaka.
            </p>
            <p className="term mt-5 flex w-fit max-w-full items-center gap-2 overflow-hidden rounded-lg px-3 py-2 text-xs sm:text-sm" style={{ color: '#b4c2d3' }}>
              <span style={{ color: '#34e3a4' }}>$</span>
              <span className="min-w-0 truncate">pyodide · numpy · pandas · scikit-learn — 100% di perangkat Anda</span><span className="caret" aria-hidden />
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <Link to="/eksplorasi" className="btn-primary no-underline" style={{ background: 'linear-gradient(120deg, #34e3a4, #22d3ee)', color: '#03130d' }}>Mulai eksplorasi data →</Link>
              <a href="#studi-kasus" className="btn border no-underline" style={{ borderColor: 'rgba(148,163,184,.35)', color: '#e6edf6' }}>Lihat studi kasus</a>
            </div>
            <p className="mt-6 font-mono text-[11px] uppercase tracking-[0.18em]" style={{ color: '#8fa3b8' }}>Dibuat oleh Dea Saka Kurnia Putra · @PUTRATALKS</p>
          </div>
          <HeroViz />
        </div>
      </section>

      <section aria-label="Ringkasan kemampuan" className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {[['05', 'alat analisis'], ['09', 'studi kasus'], ['100%', 'komputasi di browser'], ['0 B', 'data dikirim ke server']].map(([v, k]) => (
          <div key={k} className="card hud-corners !p-4">
            <div className="text-gradient font-display text-3xl font-bold">{v}</div>
            <div className="mt-1 font-mono text-[11px] uppercase tracking-wider muted">{k}</div>
          </div>
        ))}
      </section>

      <section aria-labelledby="tools">
        <div className="mb-4 flex flex-wrap items-baseline gap-3">
          <h2 id="tools" className="text-3xl font-bold"><span className="section-bar" aria-hidden />Alat analisis</h2>
          <span className="font-mono text-xs uppercase tracking-wider muted">untuk data Anda sendiri — atau dataset contoh</span>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {TOOLS.map(([to, title, desc, icon], i) => (
            <Link key={to} to={to} className="card card-hover group block no-underline" style={{ color: 'var(--text)' }}>
              <div className="flex items-center justify-between">
                <span className="flex h-11 w-11 items-center justify-center rounded-xl border" style={{ borderColor: 'color-mix(in srgb, var(--brand-2) 40%, var(--line))', color: 'var(--brand-2)', background: 'color-mix(in srgb, var(--brand-2) 8%, transparent)' }}>
                  <Icon name={icon} />
                </span>
                <span className="font-mono text-[11px] muted">{String(i + 1).padStart(2, '0')}/{String(TOOLS.length).padStart(2, '0')}</span>
              </div>
              <h3 className="mt-4 text-xl font-semibold leading-snug">{title}</h3>
              <p className="mt-1 text-sm muted">{desc}</p>
              <p className="mt-3 font-mono text-xs font-semibold uppercase tracking-wider transition-transform group-hover:translate-x-1" style={{ color: 'var(--brand-2)' }}>Buka alat →</p>
            </Link>
          ))}
        </div>
      </section>

      <section id="studi-kasus" aria-labelledby="sk" className="space-y-8">
        <div>
          <h2 id="sk" className="text-3xl font-bold"><span className="section-bar" aria-hidden />Studi kasus</h2>
          <p className="text-sm muted">Contoh penerapan alat-alat di atas pada dataset nyata, lengkap dengan hasil laporan, kode, dan catatan kritis — sebagai ilustrasi, bukan patokan.</p>
        </div>
      {GROUPS.map((g) => (
        <div key={g.name} role="group" aria-labelledby={`g-${g.name}`}>
          <div className="mb-3 flex items-baseline gap-3">
            <h3 id={`g-${g.name}`} className="text-2xl font-bold">{g.name}</h3>
            <span className="font-mono text-[11px] uppercase tracking-wider muted">Studi Kasus {g.range} · {g.blurb}</span>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {MODULES.filter((m) => m.group === g.name).map((m) => (
              <Link key={m.slug} to={`/${m.slug}`} className="card card-hover group block no-underline" style={{ color: 'var(--text)' }}>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-gradient font-display text-4xl font-bold">{m.no}</span>
                  <span className="chip text-right" style={{ color: 'var(--brand-2)' }}>{m.algo.split(' (')[0]}</span>
                </div>
                <h4 className="mt-2 text-xl font-semibold leading-snug">{m.title}</h4>
                <p className="mt-1 text-sm muted">{m.short}</p>
                {!m.reportAvailable && <p className="mt-2 inline-block rounded px-1.5 text-xs font-semibold" style={{ background: 'var(--warn-bg)' }}>Laporan belum tersedia — lab tetap dapat dipakai</p>}
              </Link>
            ))}
          </div>
        </div>
      ))}
      </section>

      <section aria-labelledby="journey">
        <h2 id="journey" className="mb-1 text-3xl font-bold"><span className="section-bar" aria-hidden />Benang merah antar-studi kasus</h2>
        <p className="mb-4 text-sm muted">Dari memprediksi nilai → memprediksi kelas → menemukan kelompok → menggali teks, dengan pelajaran metodologi yang terbawa dari satu kasus ke kasus berikutnya.</p>
        <ol className="relative space-y-4 border-l pl-6" style={{ borderColor: 'color-mix(in srgb, var(--brand-2) 50%, var(--line))' }}>
          {JOURNEY.map(([no, t, d]) => (
            <li key={no} className="relative">
              <span className="on-brand absolute -left-[35px] top-0.5 flex h-[22px] w-[22px] items-center justify-center rounded-full font-mono text-[10px] font-bold" style={{ boxShadow: '0 0 12px var(--glow)' }}>{no}</span>
              <p className="font-semibold"><Link to={`/studi-kasus-${no}`}>Studi Kasus {no} — {t}</Link></p>
              <p className="text-sm muted">{d}</p>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
