import { useEffect } from 'react';
import { Section, Spinner } from '../components/ui';
import { MODULES } from '../content/modules';
import { useJson } from '../lib/useAsync';

export default function About() {
  useEffect(() => { document.title = 'Tentang & Referensi · DATANALYSX'; }, []);
  const { data: refs } = useJson<Record<string, string[]>>('data/references.json');
  return (
    <div className="space-y-5">
      <header><h1 className="text-4xl font-bold">Tentang & Referensi</h1></header>
      <Section title="Tentang DATANALYSX">
        <div className="prose-dmx">
          <p><strong>DATANALYSX</strong> adalah lab analisis data dan data mining berbasis browser oleh <strong>Dea Saka Kurnia Putra</strong>. Fokus utamanya adalah alat analisis umum — eksplorasi, regresi, klasifikasi, clustering, dan text mining — untuk CSV apa pun. Sembilan studi kasus disertakan sebagai contoh penerapan pada dataset nyata (Studi Kasus 06 dikerjakan sebagai proyek tim).</p>
          <p><strong>Arsitektur:</strong> React + Vite + TypeScript + Tailwind CSS. Komputasi berjalan di browser lewat Pyodide 0.29.5 (Python 3.13; numpy 2.2.5, pandas 2.3.3, scikit-learn 1.7.0, scipy 1.14.1) dalam Web Worker. Tidak ada backend, API key, maupun pelacakan; data unggahan tidak meninggalkan browser.</p>
          <p><strong>Pemisahan hasil:</strong> “Hasil Laporan” ditranskripsi dari laporan/model_summary.json (berkas JSON statis); “Hasil Live” dihitung ulang di browser. Keduanya tidak dicampur. Verifikasi angka: <code>docs/verifikasi-angka.md</code> di repositori.</p>
          <p><strong>Kualitas:</strong> 24 uji pytest mereproduksi angka laporan studi kasus, kesetaraan perhitungan manual, dan alat analisis umum; 9 uji Vitest memverifikasi implementasi manual TypeScript terhadap scikit-learn (toleransi &lt; 10⁻⁸). random_state = 42.</p>
          <p><strong>Lisensi:</strong> kode MIT. Dataset contoh mengikuti lisensinya: NVIDIA, semikonduktor, Trending Topics, Instagram Reels (CC0); LLM Agent Failure Benchmark & AAPL 2026 (Apache 2.0); Indonesia Financial Time Series (CC BY-SA 4.0). NASDAQ OHLC dan komentar MBG tidak diredistribusi.</p>
          <p><strong>Peringatan:</strong> hasil bukan saran investasi; dataset AAPL 2026 dan Trending Topics bersifat sintetis; label sentimen otomatis bukan ground truth.</p>
        </div>
      </Section>
      <Section title="Daftar pustaka studi kasus (APA 7th)">
        <p className="text-sm muted">Disalin dari daftar pustaka masing-masing laporan studi kasus. Studi Kasus 09 belum memiliki laporan sehingga belum ada daftar pustaka.</p>
        {!refs && <Spinner />}
        {refs && MODULES.filter((m) => refs[m.no]).map((m) => (
          <details key={m.no} className="rounded-xl border p-3" style={{ borderColor: 'var(--line)' }} open={m.no === '01'}>
            <summary className="cursor-pointer font-semibold">Studi Kasus {m.no} — {m.algo} ({refs[m.no].length} referensi)</summary>
            <ol className="mt-2 list-decimal space-y-1 pl-6 text-sm">{refs[m.no].map((r) => <li key={r} className="break-words">{r}</li>)}</ol>
          </details>
        ))}
      </Section>
    </div>
  );
}
