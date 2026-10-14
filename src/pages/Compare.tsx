import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { MODULES } from '../content/modules';

export default function Compare() {
  useEffect(() => { document.title = 'Perbandingan Algoritma · DATANALYSX'; }, []);
  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-4xl font-bold">Perbandingan Algoritma</h1>
        <p className="muted">Kelebihan, kekurangan, dan kapan dipakai — ditarik dari bagian kelebihan–kekurangan dan keterbatasan tiap laporan (Tugas 09: dari rancangan tugas karena laporannya belum ada).</p>
      </header>
      <div className="overflow-x-auto">
        <table className="tbl min-w-[900px]">
          <thead><tr><th>Algoritma (Tugas)</th><th>Kelompok</th><th>Kelebihan</th><th>Kekurangan</th><th>Kapan dipakai</th></tr></thead>
          <tbody>
            {MODULES.map((m) => (
              <tr key={m.slug}>
                <td><Link to={`/${m.slug}`}><strong>{m.algo}</strong></Link><div className="text-xs muted">Tugas {m.no}</div></td>
                <td>{m.group}</td>
                <td><ul className="list-disc pl-4">{m.pros.map((x) => <li key={x}>{x}</li>)}</ul></td>
                <td><ul className="list-disc pl-4">{m.cons.map((x) => <li key={x}>{x}</li>)}</ul></td>
                <td>{m.when}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <section className="card">
        <h2 className="mb-2 text-2xl font-semibold">Ringkas: memilih algoritma</h2>
        <ul className="prose-dmx">
          <li><strong>Target numerik:</strong> mulai dari regresi linear (interpretasi), periksa asumsi; bila hubungan non-linear/berinteraksi, coba Random Forest (Tugas 02–03).</li>
          <li><strong>Target biner dengan kebutuhan probabilitas:</strong> regresi logistik sebagai baseline yang dapat diuji secara statistik (Tugas 04).</li>
          <li><strong>Butuh aturan yang dapat dibaca:</strong> decision tree + pruning + uji permutasi (Tugas 05).</li>
          <li><strong>Baseline sangat cepat / teks:</strong> Naïve Bayes, tetapi tangani outlier (Tugas 06, 09).</li>
          <li><strong>Pencarian kemiripan, data kecil:</strong> KNN setelah scaling yang tepat (Tugas 07).</li>
          <li><strong>Tanpa label:</strong> K-Means dengan pemeriksaan kestabilan, bukan satu indeks (Tugas 08).</li>
        </ul>
      </section>
    </div>
  );
}
