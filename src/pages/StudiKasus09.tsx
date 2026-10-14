import { useEffect, useMemo, useState } from 'react';
import DataSource from '../components/DataSource';
import Dataset, { type Column } from '../components/Dataset';
import ModulePage from '../components/ModulePage';
import Plot from '../components/Plot';
import { Callout, ErrorBox, LiveBadge, Section, Select, Slider, Spinner, Stat, StatGrid, Table, Toggle } from '../components/ui';
import { bySlug } from '../content/modules';
import { fmt, fmtFixed, int, pct } from '../lib/format';
import { tfidf } from '../lib/manual/tfidf';
import { ensurePackages, py, setResource } from '../lib/pyodide/client';
import { palette } from '../lib/theme';
import { useAsync } from '../lib/useAsync';

const m = bySlug('studi-kasus-09')!;
const LEX = [
  { name: 'inset_positive', url: 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv' },
  { name: 'inset_negative', url: 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv' },
  { name: 'kamus_alay', url: 'https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv' },
];
// Kalimat contoh ILUSTRATIF buatan penulis aplikasi — bukan data komentar MBG.
const EXAMPLES = [
  'Programnya bagus bgt, anak2 di sekolah jadi semangat makan siang 👍 #MBG',
  'Menunya gak bergizi sama sekali, nasi basi & lauk kurang!!! @dinas_terkait tolong dicek https://contoh.id',
  'Semoga anggaran makan bergizi gratis tidak dikorupsi, belum merata di desa kami.',
];

let resourcesPromise: Promise<void> | null = null;
function ensureTextResources() {
  resourcesPromise ??= (async () => {
    await ensurePackages(['Sastrawi']);
    for (const l of LEX) {
      const r = await fetch(l.url);
      if (!r.ok) throw new Error(`Gagal mengunduh leksikon ${l.name}`);
      await setResource(l.name, await r.text());
    }
  })().catch((e) => { resourcesPromise = null; throw e; });
  return resourcesPromise;
}

function useResources() {
  const s = useAsync<any>();
  useEffect(() => { s.run(async () => { await ensureTextResources(); return py('dmx.t09', 'resources_status'); }); }, [s.run]);
  return s;
}

function ResourceNote({ s }: { s: ReturnType<typeof useResources> }) {
  if (s.loading) return <Spinner text="Memasang Sastrawi (micropip) & mengunduh InSet + kamus slang dari GitHub …" />;
  if (s.error) return <ErrorBox error={s.error} />;
  if (!s.data) return null;
  return <p className="text-xs muted">Sumber daya: InSet {int(s.data.inset)} kata (Koto & Rahmaningtyas, 2017; github.com/fajri91/InSet) · kamus slang {int(s.data.kamus_alay)} entri (Salsabila dkk., 2018; github.com/nasalsabila/kamus-alay) · {s.data.stopwords} stopword Sastrawi. Dimuat langsung dari repositori sumber (tidak diredistribusi).</p>;
}

function Report() {
  return (
    <div className="space-y-5">
      <Callout kind="warn" title="Laporan Studi Kasus 09 belum tersedia">Belum ada laporan maupun berkas textmining_mbg.py/model_summary.json, sehingga tab ini sengaja kosong dari angka. Seluruh hasil pada tab Lab adalah Hasil Live yang dihitung ulang di browser dari data yang Anda unggah.</Callout>
      <Section title="Rancangan analisis">
        <ul className="prose-dmx">
          <li>Data: komentar-mbg.csv (15.000 × 4) dan MBG_sentiment_analysis.csv (11.375 × 6, tanpa label & tanpa stemming).</li>
          <li>Preprocessing teks Bahasa Indonesia standar, negasi dipertahankan, stemming Sastrawi.</li>
          <li>Label InSet, divalidasi sampel beranotasi (Cohen’s κ); klasifikasi MNB/LogReg/Linear SVM/KNN; LDA/NMF.</li>
          <li>Etika: username disamarkan, kata kasar disensor, bahasa netral.</li>
        </ul>
      </Section>
    </div>
  );
}

export function Playground() {
  const res = useResources();
  const [text, setText] = useState(EXAMPLES[1]);
  const [order, setOrder] = useState('slide');
  const [keepNeg, setKeepNeg] = useState(true);
  const [stem, setStem] = useState(true);
  const s = useAsync<any>();
  useEffect(() => { if (res.data) s.run(() => py('dmx.t09', 'preprocess', { text, order, keep_negation: keepNeg, do_stem: stem })); }, [res.data, text, order, keepNeg, stem, s.run]);
  return (
    <Section title="Pipeline preprocessing interaktif" right={<LiveBadge />}>
      <ResourceNote s={res} />
      <label className="block"><span className="label">Ketik / tempel komentar</span><textarea className="input h-24" value={text} onChange={(e) => setText(e.target.value)} /></label>
      <div className="flex flex-wrap gap-2">{EXAMPLES.map((e, i) => <button key={i} className="btn-ghost text-xs" onClick={() => setText(e)}>Contoh {i + 1}</button>)}</div>
      <p className="text-xs muted">Contoh adalah kalimat ilustratif buatan, bukan kutipan data.</p>
      <div className="grid gap-4 md:grid-cols-3">
        <Select label="Urutan tahap" value={order} onChange={setOrder} options={[{ value: 'slide', label: 'Klasik: stopword → normalisasi' }, { value: 'normalize_first', label: 'Normalisasi → stopword (negasi slang aman)' }]} />
        <Toggle label="Pertahankan kata negasi" checked={keepNeg} onChange={setKeepNeg} />
        <Toggle label="Stemming Sastrawi" checked={stem} onChange={setStem} />
      </div>
      {s.data && (
        <ol className="space-y-2">{s.data.steps.map((st: any, i: number) => (
          <li key={i} className="rounded-lg border p-2 text-sm" style={{ borderColor: 'var(--line)' }}>
            <span className="chip mr-2 on-brand">{i}. {st.stage}</span>
            {Array.isArray(st.output) ? <span className="font-mono">[{st.output.map((t: string) => `'${t}'`).join(', ')}]</span> : <span>{st.output}</span>}
          </li>))}</ol>
      )}
      <ErrorBox error={s.error} />
    </Section>
  );
}

export function CorpusLab({ generic = false }: { generic?: boolean }) {
  const res = useResources();
  const [key, setKey] = useState<string | null>(null);
  const [cols, setCols] = useState<Column[]>([]);
  const [textCol, setTextCol] = useState('');
  const [authorCol, setAuthorCol] = useState('');
  const [dateCol, setDateCol] = useState('');
  async function onData(k: string, preset?: { text?: string }) {
    const info = await py<{ columns: Column[] }>('dmx.eda', 'columns', { data_key: k });
    const names = info.columns.map((c) => c.name);
    const pick = (cands: string[]) => cands.find((c) => names.includes(c)) ?? '';
    setCols(info.columns);
    setTextCol(preset?.text && names.includes(preset.text) ? preset.text : pick(['comment', 'text', 'review', 'content', 'headline']) || (info.columns.find((c) => c.kind === 'teks')?.name ?? names[0]));
    setAuthorCol(pick(['author', 'user', 'username']));
    setDateCol(pick(['published_at', 'date', 'created_at', 'timestamp']));
    setKey(k);
  }
  const [subset, setSubset] = useState<'all' | 'mbg'>('all');
  const [order, setOrder] = useState('normalize_first');
  const [keepNeg, setKeepNeg] = useState(true);
  const [stem, setStem] = useState(true);
  const [nTopics, setNTopics] = useState(5);
  const [method, setMethod] = useState<'nmf' | 'lda'>('nmf');
  const [kw, setKw] = useState('gizi');
  const ov = useAsync<any>();
  const pre = useAsync<any>();
  const lab = useAsync<any>();
  const cls = useAsync<any>();
  const top = useAsync<any>();
  const co = useAsync<any>();
  const conc = useAsync<any>();
  useEffect(() => { if (key && res.data && textCol) ov.run(() => py('dmx.t09', 'load_corpus', { data_key: key, text_col: textCol, author_col: authorCol || null, date_col: dateCol || null })); }, [key, res.data, textCol, authorCol, dateCol, ov.run]);
  useEffect(() => {
    if (!ov.data) return;
    pre.run(async () => {
      const r = await py('dmx.t09', 'preprocess_corpus', { data_key: key, order, keep_negation: keepNeg, do_stem: stem, subset, max_docs: 20000 });
      const l = await py('dmx.t09', 'label_lexicon', { data_key: key });
      lab.setData(l);
      co.run(() => py('dmx.t09', 'cooccurrence', { data_key: key, top_n: 25 }));
      cls.run(() => py('dmx.t09', 'classify', { data_key: key }));
      return r;
    });
  }, [ov.data, key, order, keepNeg, stem, subset, pre.run, lab.setData, co.run, cls.run]);
  useEffect(() => { if (lab.data) top.run(() => py('dmx.t09', 'topics', { data_key: key, method, n_topics: nTopics })); }, [lab.data, key, method, nTopics, top.run]);
  const p = palette();
  const sentColor: Record<string, string> = { positif: p.up, netral: p.neutral, negatif: p.down };
  return (
    <div className="space-y-5">
      {generic ? (
        <Dataset dataKey="teks" tool="teks" onReady={(i) => onData(i.key, i.preset)} />
      ) : (<>
        <Callout kind="info" title="Dataset MBG tidak disertakan">Lisensi Kaggle neevvv/indonesia-mbg-sentiment-analysis tidak memberi izin redistribusi eksplisit dan data memuat nama akun. Unggah komentar-mbg.csv (kolom author, comment, likes, published_at) — nama akun langsung diganti pseudonim (hash) di browser.</Callout>
        <DataSource dataKey="t09" sample={null} onReady={(k) => onData(k)} uploadHint="Unggah komentar-mbg.csv atau CSV lain dengan kolom teks bernama comment/text." />
      </>)}
      {key && cols.length > 0 && (
        <div className="grid gap-4 md:grid-cols-3">
          <Select label="Kolom teks" value={textCol} onChange={setTextCol} options={cols.map((c) => ({ value: c.name, label: c.name }))} />
          <Select label="Kolom penulis (disamarkan)" value={authorCol} onChange={setAuthorCol} options={[{ value: '', label: '(tidak ada)' }, ...cols.map((c) => ({ value: c.name, label: c.name }))]} />
          <Select label="Kolom tanggal" value={dateCol} onChange={setDateCol} options={[{ value: '', label: '(tidak ada)' }, ...cols.map((c) => ({ value: c.name, label: c.name }))]} />
        </div>
      )}
      {generic && <p className="text-xs muted">Leksikon sentimen InSet dan kamus slang berbahasa Indonesia; teks berbahasa lain akan banyak berlabel netral.</p>}
      <ResourceNote s={res} />
      {ov.data && (
        <Section title="Pemahaman data (sebelum preprocessing)" right={<LiveBadge />}>
          <StatGrid>
            <Stat label="Baris × kolom" value={`${int(ov.data.n_rows)} × ${ov.data.n_cols}`} />
            <Stat label="Duplikat penuh" value={int(ov.data.n_full_duplicates)} /><Stat label="Duplikat teks" value={int(ov.data.n_text_duplicates)} />
            <Stat label="Komentar unik" value={int(ov.data.n_unique_texts)} /><Stat label="Author unik (pseudonim)" value={int(ov.data.n_unique_authors ?? 0)} />
            <Stat label="Menyebut MBG eksplisit" value={pct(ov.data.share_mbg, 1)} /><Stat label="Panjang (kata) min/median/maks" value={`${ov.data.n_words.min} / ${fmt(ov.data.n_words.median, 0)} / ${int(ov.data.n_words.max)}`} />
            <Stat label="Mengandung emoji" value={pct(ov.data.share_emoji, 1)} />
          </StatGrid>
          <div className="grid gap-4 lg:grid-cols-2">
            {ov.data.weekly && <Plot title="Volume komentar per minggu" height={260} data={[{ type: 'bar', x: ov.data.weekly.week, y: ov.data.weekly.count, marker: { color: p.brand2 } }]} />}
            {ov.data.likes && <Plot title={`Distribusi likes (log1p) · median ${fmt(ov.data.likes.median, 0)}, maks ${int(ov.data.likes.max)}`} height={260} data={[{ type: 'bar', y: ov.data.likes.hist_log1p, marker: { color: p.neutral } }]} />}
          </div>
          <p className="text-sm">{int(ov.data.n_texts_by_multiple_authors ?? 0)} teks identik diunggah oleh lebih dari satu akun (indikasi copy-paste; deskriptif saja).</p>
        </Section>
      )}
      {ov.data && (
        <Section title="Preprocessing korpus & label leksikon InSet">
          <div className="grid gap-4 md:grid-cols-4">
            <Select label="Cakupan" value={subset} onChange={setSubset} options={[{ value: 'all', label: 'Korpus penuh' }, { value: 'mbg', label: 'Hanya yang menyebut MBG' }]} />
            <Select label="Urutan" value={order} onChange={setOrder} options={[{ value: 'normalize_first', label: 'Normalisasi → stopword' }, { value: 'slide', label: 'Stopword → normalisasi (klasik)' }]} />
            <Toggle label="Pertahankan negasi" checked={keepNeg} onChange={setKeepNeg} />
            <Toggle label="Stemming" checked={stem} onChange={setStem} />
          </div>
          {pre.loading && <Spinner text="Membersihkan & men-stem korpus …" />}
          <ErrorBox error={pre.error} />
          {pre.data && <p className="text-sm">{int(pre.data.n_docs)} dokumen · kosakata mentah {int(pre.data.vocab_raw)} → bersih {int(pre.data.vocab_clean)} kata.</p>}
          {lab.data && (<>
            <div className="grid gap-4 lg:grid-cols-3">
              <Plot title="Distribusi label (otomatis)" height={260} data={[{ type: 'bar', x: ['negatif', 'netral', 'positif'], y: ['negatif', 'netral', 'positif'].map((k) => lab.data.counts[k] ?? 0), marker: { color: [p.down, p.neutral, p.up] } }]} />
              {lab.data.weekly && <Plot title="Tren sentimen mingguan" height={260} data={['negatif', 'netral', 'positif'].map((k) => ({ type: 'scatter', mode: 'lines', stackgroup: 'one', name: k, x: lab.data.weekly.week, y: lab.data.weekly[k], line: { color: sentColor[k] } }))} />}
              {pre.data && <Plot title="Hukum Zipf (log–log)" height={260} data={[{ type: 'scatter', mode: 'markers', x: pre.data.zipf.map((_: number, i: number) => i + 1), y: pre.data.zipf, marker: { size: 4, color: p.brand2 } }]} layout={{ xaxis: { type: 'log', title: { text: 'peringkat' } }, yaxis: { type: 'log', title: { text: 'frekuensi' } } }} />}
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              {(['positif', 'netral', 'negatif'] as const).map((s) => <WordCloud key={s} title={`Kata teratas — ${s}`} words={lab.data.top_terms[s]} color={sentColor[s]} />)}
            </div>
            <Callout kind="warn" title="Label otomatis ≠ ground truth">Label berasal dari leksikon InSet dengan pembalikan tanda setelah kata negasi; sarkasme, campur kode, dan konteks politik sering salah dibaca. Komentar YouTube tidak mewakili opini publik Indonesia.</Callout>
          </>)}
        </Section>
      )}
      {cls.data && (
        <Section title="Klasifikasi sentimen (Pipeline TF-IDF, duplikat dibuang sebelum split)">
          <Table head={['Model', 'Akurasi', 'Presisi makro', 'Recall makro', 'F1 makro', 'ROC-AUC makro']} numeric={[1, 2, 3, 4, 5]} rows={cls.data.rows.map((x: any) => [x.model, fmtFixed(x.accuracy, 3), fmtFixed(x.precision_macro, 3), fmtFixed(x.recall_macro, 3), fmtFixed(x.f1_macro, 3), x.roc_auc_macro === null ? '–' : fmtFixed(x.roc_auc_macro, 3)])} />
          <p className="text-sm">Matriks TF-IDF latih {cls.data.matrix_shape.join(' × ')} (sparsitas {pct(cls.data.sparsity, 2)}); latih {int(cls.data.n_train)} / uji {int(cls.data.n_test)}.</p>
          <Callout kind="warn" title="Sirkularitas">{cls.data.warning} Evaluasi utama semestinya pada sampel beranotasi manusia.</Callout>
        </Section>
      )}
      {lab.data && (
        <Section title="Pemodelan topik">
          <div className="grid gap-4 md:grid-cols-2">
            <Select label="Metode" value={method} onChange={setMethod} options={[{ value: 'nmf', label: 'NMF (TF-IDF)' }, { value: 'lda', label: 'LDA (Bag of Words)' }]} />
            <Slider label="Jumlah topik" value={nTopics} min={2} max={10} onChange={setNTopics} />
          </div>
          {top.loading && <Spinner />}
          {top.data && (<>
            <Table head={['Topik', 'Kata teratas', 'Porsi dokumen', 'UMass', 'Neg / Net / Pos']} numeric={[2, 3]} rows={top.data.top_words.map((w: string[], i: number) => [`T${i + 1}`, w.join(', '), pct(top.data.doc_share[i], 1), fmtFixed(top.data.umass[i], 1), top.data.sentiment_by_topic ? top.data.sentiment_by_topic[i].join(' / ') : ''])} />
            <p className="text-xs muted">Koherensi dihitung dengan UMass (bukan c_v, karena gensim tidak tersedia di browser). Nama topik perlu ditafsirkan manusia.</p>
          </>)}
        </Section>
      )}
      {co.data && (
        <Section title="Ko-okurensi kata (25 kata teratas)">
          <Plot height={460} data={[{ type: 'heatmap', z: co.data.matrix, x: co.data.terms, y: co.data.terms, colorscale: [[0, '#f5f5f2'], [1, '#0B4F3A']] }]} layout={{ margin: { l: 90, b: 90 } }} />
          <div className="flex flex-wrap items-end gap-2">
            <label><span className="label">Concordance kata kunci</span><input className="input" value={kw} onChange={(e) => setKw(e.target.value)} /></label>
            <button className="btn-ghost" onClick={() => conc.run(() => py('dmx.t09', 'concordance', { data_key: key, keyword: kw }))}>Cari</button>
          </div>
          {conc.data && <Table head={['Kiri', 'Kata', 'Kanan']} rows={conc.data.rows.map((r: any) => [<span className="block text-right">{r.left}</span>, <strong>{r.kw}</strong>, r.right])} />}
          <p className="text-xs muted">Kutipan otomatis disensor untuk kata kasar; nama akun tidak pernah ditampilkan.</p>
        </Section>
      )}
    </div>
  );
}

function WordCloud({ title, words, color }: { title: string; words: [string, number][]; color: string }) {
  const max = words[0]?.[1] ?? 1;
  return (
    <div className="card">
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      <p className="flex flex-wrap gap-x-2 gap-y-0.5 leading-tight" aria-label={`${title}: ${words.slice(0, 20).map((w) => w[0]).join(', ')}`}>
        {words.slice(0, 45).map(([w, c]) => <span key={w} style={{ fontSize: `${0.7 + 1.5 * Math.sqrt(c / max)}rem`, color, opacity: 0.55 + 0.45 * (c / max) }} title={`${w}: ${c}`}>{w}</span>)}
      </p>
    </div>
  );
}

function Lab() {
  return <div className="space-y-5"><Playground /><CorpusLab /></div>;
}

export function TfidfManual() {
  const [docsText, setDocsText] = useState('makan gratis bagus\ngizi anak tidak bagus\nmakan siang anak sekolah\nprogram gizi makan gratis\nanggaran makan tidak jelas');
  const [smooth, setSmooth] = useState(true);
  const docs = useMemo(() => docsText.split('\n').map((d) => d.trim()).filter(Boolean), [docsText]);
  const ts = useMemo(() => tfidf(docs, smooth), [docs, smooth]);
  const sk = useAsync<any>();
  useEffect(() => { if (docs.length) sk.run(() => py('dmx.t09', 'tfidf_sklearn', { docs, smooth_idf: smooth })); }, [docs, smooth, sk.run]);
  const diff = sk.data && sk.data.vocab.join() === ts.vocab.join() ? Math.max(...ts.matrix.flat().map((v, i) => Math.abs(v - sk.data.matrix.flat()[i]))) : null;
  const N = docs.length;
  return (
    <Section title="Kalkulator TF-IDF manual (dicocokkan dengan TfidfVectorizer)" right={<LiveBadge />}>
      <label className="block"><span className="label">Dokumen hasil preprocessing (satu per baris)</span><textarea className="input h-32 font-mono" value={docsText} onChange={(e) => setDocsText(e.target.value)} /></label>
      <p className="text-xs muted">Dokumen bawaan adalah contoh ilustratif buatan, bukan komentar asli.</p>
      <Toggle label="IDF smooth (bawaan scikit-learn)" checked={smooth} onChange={setSmooth} />
      <p className="font-mono text-sm">{smooth ? `IDF(t) = ln((1 + N) / (1 + df)) + 1, N = ${N}` : `IDF klasik ala scikit-learn (smooth_idf=False) = ln(N / df) + 1, N = ${N}`} · TF = frekuensi mentah · w = TF × IDF · normalisasi L2 per dokumen.</p>
      <Table head={['Term', 'DF', 'IDF', ...docs.map((_, i) => `TF d${i + 1}`)]} numeric={[1, 2, ...docs.map((_, i) => i + 3)]} rows={ts.vocab.map((t, j) => [t, ts.df[j], fmtFixed(ts.idf[j], 4), ...ts.tf.map((r) => r[j])])} />
      <Table caption="TF-IDF ternormalisasi L2" head={['Term', ...docs.map((_, i) => `d${i + 1}`)]} numeric={docs.map((_, i) => i + 1)} rows={ts.vocab.map((t, j) => [t, ...ts.matrix.map((r) => (r[j] ? fmtFixed(r[j], 4) : '0'))])} />
      <p className="text-xs muted">Panjang vektor sebelum normalisasi: {ts.norms.map((n) => fmtFixed(n, 4)).join(' · ')}. Catatan: IDF klasik di buku teks sering ditulis log(N/df) tanpa “+1”; scikit-learn menambahkan 1 agar term yang muncul di semua dokumen tidak berbobot nol.</p>
      {sk.loading && <Spinner />}
      {diff !== null && <Callout kind={diff < 1e-8 ? 'ok' : 'warn'} title="Verifikasi">Selisih maksimum TypeScript vs TfidfVectorizer = {diff.toExponential(2)} {diff < 1e-8 ? '✓ identik' : ''}</Callout>}
    </Section>
  );
}

export default function StudiKasus09() {
  return <ModulePage m={m} report={<Report />} lab={<Lab />} manual={<TfidfManual />} />;
}
