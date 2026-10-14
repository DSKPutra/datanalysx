import { useEffect, useState } from 'react';
import Plot from '../components/Plot';
import PyGate from '../components/PyGate';
import { ErrorBox, LiveBadge, Section, Select, Spinner, Table, Toggle } from '../components/ui';
import { fmt, fmtFixed, int } from '../lib/format';
import { py } from '../lib/pyodide/client';
import { palette } from '../lib/theme';
import { useAsync } from '../lib/useAsync';
import { Profile, SamplePicker } from './LabShared';

function Inner() {
  const [key, setKey] = useState<string | null>(null);
  const [ver, setVer] = useState(0);
  const prof = useAsync<any>();
  const prep = useAsync<any>();
  const run = useAsync<any>();
  const [target, setTarget] = useState('');
  const [feats, setFeats] = useState<string[]>([]);
  const [opt, setOpt] = useState({ impute: 'median', outlier: 'none', encode: 'onehot', scale: 'none' });
  const [cmpCol, setCmpCol] = useState('');
  useEffect(() => { if (key) prof.run(() => py('dmx.eda', 'profile', { data_key: key })); }, [key, ver, prof.run]);
  const cols: any[] = prof.data?.columns ?? [];
  const candidates = cols.filter((c) => c.unique >= 2 && c.unique <= 20).map((c) => c.name);
  useEffect(() => {
    if (!prof.data) return;
    const t = candidates.includes('virality_tier') ? 'virality_tier' : candidates.includes('topic_category') ? 'topic_category' : candidates[0] ?? '';
    setTarget(t);
    const leak = ['target', 'virality_score', 'views', 'likes', 'comments', 'engagement_rate_pct', 'split', 'topic_subcategory', 'short_text', 'id', 'post_id', 'shortcode'];
    setFeats(cols.filter((c) => c.name !== t && c.kind !== 'teks' && !leak.includes(c.name)).map((c) => c.name).slice(0, 12));
  }, [prof.data]);
  const p = palette();
  const before = prep.data ? Object.keys(prep.data.before) : [];
  const sel = cmpCol || before[0];
  return (
    <div className="space-y-5">
      <SamplePicker dataKey="datalab" initial={2} onReady={(k) => { setKey(k); setVer((v) => v + 1); run.setData(null); prep.setData(null); }} />
      {key && <Profile dataKey={key} version={ver} />}
      {key && (
        <Section title="Data preprocessing (before–after)" right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-4">
            <Select label="Imputasi" value={opt.impute} onChange={(v) => setOpt({ ...opt, impute: v })} options={[{ value: 'none', label: 'Tidak' }, { value: 'mean', label: 'Mean' }, { value: 'median', label: 'Median' }, { value: 'interpolate', label: 'Interpolasi linear' }]} />
            <Select label="Outlier (IQR 1,5×)" value={opt.outlier} onChange={(v) => setOpt({ ...opt, outlier: v })} options={[{ value: 'none', label: 'Biarkan' }, { value: 'winsorize', label: 'Winsorization' }, { value: 'drop', label: 'Buang baris' }]} />
            <Select label="Encoding" value={opt.encode} onChange={(v) => setOpt({ ...opt, encode: v })} options={[{ value: 'none', label: 'Tidak' }, { value: 'onehot', label: 'One-hot' }, { value: 'label', label: 'Label' }]} />
            <Select label="Scaling" value={opt.scale} onChange={(v) => setOpt({ ...opt, scale: v })} options={[{ value: 'none', label: 'Tidak' }, { value: 'standard', label: 'Standard (z-score)' }, { value: 'minmax', label: 'MinMax [0,1]' }]} />
          </div>
          <button className="btn-ghost" disabled={prep.loading} onClick={() => prep.run(() => py('dmx.eda', 'preprocess', { data_key: key, ...opt }))}>Terapkan (untuk eksplorasi)</button>
          {prep.loading && <Spinner />}
          {prep.data && (<>
            <ul className="prose-dmx text-sm">{prep.data.log.map((l: string) => <li key={l}>{l}</li>)}</ul>
            <p className="text-sm">Ukuran: {prep.data.shape_before.join(' × ')} → {prep.data.shape_after.join(' × ')}</p>
            {before.length > 0 && <Select label="Kolom" value={sel} onChange={setCmpCol} options={before.map((c) => ({ value: c, label: c }))} />}
            {sel && prep.data.after[sel] && <div className="grid gap-4 md:grid-cols-2">
              <Plot title={`Sebelum: ${sel}`} height={240} data={[{ type: 'bar', x: prep.data.before[sel].hist.edges.slice(0, -1), y: prep.data.before[sel].hist.counts, marker: { color: p.neutral } }]} />
              <Plot title={`Sesudah: ${sel}`} height={240} data={[{ type: 'bar', x: prep.data.after[sel].hist.edges.slice(0, -1), y: prep.data.after[sel].hist.counts, marker: { color: p.brand2 } }]} />
            </div>}
            {prep.data.csv && <a className="btn-ghost" download="datanalysx_preprocessed.csv" href={URL.createObjectURL(new Blob([prep.data.csv], { type: 'text/csv' }))}>Unduh CSV hasil praproses</a>}
            <p className="text-xs muted">Catatan: praproses di sini diterapkan ke SELURUH data untuk eksplorasi. Leaderboard di bawah memakai Pipeline yang di-fit pada data latih tiap lipatan (tanpa kebocoran).</p>
          </>)}
        </Section>
      )}
      {prof.data && (
        <Section title="Leaderboard klasifikasi" right={<LiveBadge />}>
          <Select label="Target (kategori, 2–20 kelas)" value={target} onChange={setTarget} options={candidates.map((c) => ({ value: c, label: c }))} />
          <fieldset><legend className="label">Fitur (hindari kolom pembentuk target!)</legend>
            <div className="grid grid-cols-2 gap-1 md:grid-cols-4">{cols.filter((c) => c.name !== target).map((c) => <Toggle key={c.name} label={`${c.name}${c.kind === 'teks' ? ' (teks, diabaikan)' : ''}`} checked={feats.includes(c.name)} onChange={(v) => setFeats(v ? [...feats, c.name] : feats.filter((x) => x !== c.name))} />)}</div>
          </fieldset>
          <button className="btn-primary" disabled={!target || !feats.length || run.loading} onClick={() => run.run(() => py('dmx.datalab', 'classify', { data_key: key, target, features: feats }))}>Jalankan semua algoritma</button>
          {run.loading && <Spinner text="Validasi silang berstrata …" />}
          <ErrorBox error={run.error} />
          {run.data && (<>
            <p className="text-sm">{int(run.data.n)} baris · kelas {run.data.classes.map((c: string, i: number) => `${c} (${run.data.class_counts[i]})`).join(', ')} · {run.data.cv_folds}-fold berstrata · numerik {run.data.numeric.length}, kategorikal {run.data.categorical.length}{run.data.dropped.length ? `, diabaikan ${run.data.dropped.join(', ')}` : ''}.</p>
            <Table head={['#', 'Model', 'Akurasi', 'Balanced acc.', 'F1 makro', 'ROC-AUC', 'Waktu (s)']} numeric={[2, 3, 4, 5, 6]} rows={run.data.rows.map((x: any, i: number) => [i + 1, x.model, `${fmtFixed(x.accuracy, 3)} ± ${fmtFixed(x.accuracy_std, 3)}`, fmtFixed(x.balanced_accuracy, 3), fmtFixed(x.f1_macro, 3), fmtFixed(x.roc_auc, 3), fmt(x.seconds, 2)])} />
            <Plot height={280} data={[{ type: 'bar', x: run.data.rows.map((x: any) => x.model), y: run.data.rows.map((x: any) => x.accuracy), error_y: { type: 'data', array: run.data.rows.map((x: any) => x.accuracy_std) }, marker: { color: run.data.rows.map((x: any) => (x.model.startsWith('Baseline') ? p.neutral : p.brand2)) } }]} layout={{ yaxis: { range: [0, 1], title: { text: 'Akurasi CV' } } }} />
          </>)}
        </Section>
      )}
    </div>
  );
}

export default function DataLab() {
  useEffect(() => { document.title = 'Data Lab · DATANALYSX'; }, []);
  return (
    <div className="space-y-5">
      <header><h1 className="text-4xl font-bold">Data Lab</h1><p className="muted">Unggah CSV → profil data (EDA) → praproses before–after → pilih target → jalankan Regresi Logistik, Naïve Bayes, KNN, Decision Tree, Random Forest sekaligus → leaderboard metrik.</p></header>
      <PyGate><Inner /></PyGate>
    </div>
  );
}
