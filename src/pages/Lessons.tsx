import { useEffect } from 'react';
import { Link } from 'react-router-dom';

const CARDS: { title: string; cases: string[]; body: string; takeaway: string }[] = [
  { title: 'Kebocoran data (data leakage)', cases: ['01', '03', '04', '05', '07'], body: 'Studi Kasus 01 memakai Open/High/Low hari yang sama untuk “memprediksi” Close (R² 0,999994). Studi Kasus 03 membuang operating_income karena = pendapatan × margin. Studi Kasus 04 belajar dari sini: split berbasis waktu, scaler di-fit pada data latih, walk-forward. Studi Kasus 05 membuang topic_subcategory & short_text; Studi Kasus 07 membuang views/likes/virality_score.', takeaway: 'Tanyakan selalu: apakah informasi ini sudah tersedia pada saat prediksi dibuat?' },
  { title: 'Pilihan metrik mengubah peringkat model', cases: ['03'], body: 'Pada validasi silang Studi Kasus 03, RMSE menjagokan Linear berganda (3,837) sedangkan MAE menjagokan Random Forest (0,949): RMSE didominasi perusahaan raksasa, MAE mencerminkan perusahaan tipikal.', takeaway: 'Pilih metrik sesuai tujuan bisnis sebelum memilih model.' },
  { title: 'Signifikansi statistik ≠ signifikansi ekonomis', cases: ['03', '04'], body: 'Rata-rata AUC 20 saham 0,5156 signifikan (p = 0,0075) tetapi ±1,6 poin AUC habis oleh biaya transaksi. Di Studi Kasus 03, margin “signifikan” pada log-log namun R² parsial hanya 7,1%.', takeaway: 'Laporkan ukuran efek & konteks, bukan hanya p-value.' },
  { title: 'Outlier dapat merusak model probabilistik', cases: ['06', '08'], body: 'Satu anomali USD/IDR +902,9% membuat Gaussian NB jatuh ke 48,48% (di bawah baseline) dan Brier 0,461; winsorization IQR 3× → 51,87% dan 0,256. Di Studi Kasus 08, outlier volume menggeser partisi K-Means.', takeaway: 'Deteksi, jelaskan, lalu tangani outlier — dengan batas yang dihitung dari data latih.' },
  { title: 'Skala fitur menentukan algoritma berbasis jarak', cases: ['07', '08'], body: 'KNN K = 5: tanpa scaling 78,25% → StandardScaler 79,98% → MinMaxScaler 87,11%. K-Means tanpa scaling dikuasai satu fitur (η² 0,81) dan nyaris tak berhubungan dengan partisi terstandar (ARI 0,13).', takeaway: 'Scaling adalah keputusan pemodelan, bukan langkah kosmetik.' },
  { title: 'Data sintetis & “pola” palsu', cases: ['05', '08'], body: 'Pohon pada metadata sintetis Studi Kasus 05 tidak lebih baik dari acak (uji permutasi p = 0,970) walau feature importance tampak meyakinkan; K-Means selalu mengembalikan klaster walau silhouette < 0,25.', takeaway: 'Bandingkan dengan baseline & uji permutasi/kestabilan sebelum mempercayai pola.' },
  { title: 'Evaluasi yang jujur berlapis', cases: ['04', '05', '07', '08'], body: 'Satu angka dari satu split rapuh. Laporan memakai walk-forward, bootstrap CI, uji permutasi, CV berulang, dan ARI lintas seed.', takeaway: 'Kuantifikasi ketidakpastian; reproduksibilitas (random_state = 42) wajib.' },
  { title: 'Label otomatis bukan kebenaran', cases: ['09'], body: 'Label sentimen dari leksikon InSet membuat klasifier meniru leksikon (sirkular); komentar YouTube bukan opini publik.', takeaway: 'Validasi dengan anotasi manusia dan nyatakan batas generalisasi.' },
];

export default function Lessons() {
  useEffect(() => { document.title = 'Pelajaran Metodologi · DATANALYSX'; }, []);
  return (
    <div className="space-y-5">
      <header><h1 className="text-4xl font-bold">Pelajaran Metodologi</h1><p className="muted">Pelajaran lintas studi kasus yang berlaku untuk analisis data apa pun. Angka dikutip dari laporan studi kasus.</p></header>
      <div className="grid gap-4 md:grid-cols-2">
        {CARDS.map((c) => (
          <article key={c.title} className="card space-y-2">
            <h2 className="text-2xl font-semibold">{c.title}</h2>
            <div className="flex flex-wrap gap-1">{c.cases.map((t) => <Link key={t} to={`/studi-kasus-${t}`} className="chip no-underline" style={{ background: 'var(--surface-2)' }}>Studi Kasus {t}</Link>)}</div>
            <p className="text-sm">{c.body}</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--brand-2)' }}>→ {c.takeaway}</p>
          </article>
        ))}
      </div>
    </div>
  );
}
