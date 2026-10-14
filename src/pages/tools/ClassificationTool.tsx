import { useState } from 'react';
import Dataset, { type Column } from '../../components/Dataset';
import Plot from '../../components/Plot';
import { Callout, ErrorBox, LiveBadge, Section, Select, Slider, Spinner, Stat, StatGrid, Table, Toggle } from '../../components/ui';
import { fmt, fmtFixed, int, pct } from '../../lib/format';
import { gnbPosterior } from '../../lib/manual/gnb';
import { knnManual, type Metric } from '../../lib/manual/knn';
import { sigmoid } from '../../lib/manual/logistic';
import { py } from '../../lib/pyodide/client';
import { palette } from '../../lib/theme';
import { useAsync } from '../../lib/useAsync';
import ToolShell from './ToolShell';

const ALGOS = [
  { value: 'logreg', label: 'Regresi Logistik' }, { value: 'gnb', label: 'Gaussian Naïve Bayes' },
  { value: 'knn', label: 'K-Nearest Neighbors' }, { value: 'tree', label: 'Decision Tree' }, { value: 'rf', label: 'Random Forest' },
];

function ManualPanel({ r }: { r: any }) {
  const m = r.manual;
  const cls: string[] = r.classes;
  const sk: number[] = m.proba_sklearn;
  let body: JSX.Element | null = null;
  let mine: number[] | null = null;
  if (m.kind === 'logreg') {
    const terms = m.coef.map((b: number, i: number) => b * m.x[i]);
    const z = m.intercept + terms.reduce((a: number, b: number) => a + b, 0);
    const pr = sigmoid(z);
    mine = [1 - pr, pr];
    body = (<>
      <Table head={['Fitur (setelah praproses)', 'x', 'β', 'β·x']} numeric={[1, 2, 3]} rows={m.feature_names.map((f: string, i: number) => [f, fmt(m.x[i], 5), fmt(m.coef[i], 5), fmt(terms[i], 6)])} />
      <p className="font-mono text-sm">z = b₀ + Σβx = {fmt(m.intercept, 6)} + {fmt(z - m.intercept, 6)} = {fmt(z, 8)} → P({cls[1]}) = 1/(1+e^(−z)) = <strong>{fmtFixed(pr, 8)}</strong></p>
    </>);
  } else if (m.kind === 'gnb') {
    const g = gnbPosterior(m.x, m.theta, m.var, m.prior);
    mine = g.posterior;
    body = (<>
      <Table head={['Kelas', 'ln P(y)', 'Σ ln P(xᵢ|y)', 'ln gabungan', 'Posterior']} numeric={[1, 2, 3, 4]} rows={cls.map((c, k) => [c, fmtFixed(g.logPrior[k], 4), fmtFixed(g.sumLog[k], 4), fmtFixed(g.joint[k], 4), pct(g.posterior[k])])} />
      <p className="text-sm">Likelihood tiap fitur memakai kerapatan normal dengan μ dan σ² per kelas dari data latih; gabungan dalam ruang log lalu dinormalisasi dengan log-sum-exp.</p>
    </>);
  } else if (m.kind === 'knn' && m.complete) {
    const k = knnManual(m.x, m.train, m.train_y, m.k, m.metric as Metric, m.weights, 2, cls.length);
    mine = k.proba;
    body = (<>
      <Table head={['Peringkat', 'Baris latih', 'Kelas', 'Jarak', 'Bobot']} numeric={[0, 1, 3, 4]} rows={k.neighbors.map((i, j) => [j + 1, i, cls[m.train_y[i]], fmtFixed(k.distances[i], 6), fmtFixed(k.weights[j], 4)])} />
      <p className="text-sm">Suara: {cls.map((c, i) => `${c} = ${fmtFixed(k.votes[i], 4)}`).join(', ')}.</p>
    </>);
  } else if (m.kind === 'tree') {
    mine = m.leaf_dist;
    body = (<>
      <ol className="list-decimal space-y-1 pl-6 text-sm">{m.path.map((s: any) => <li key={s.node}>Simpul {s.node} (n = {int(s.n)}): {s.feature} = {fmt(s.value, 4)} ≤ {fmt(s.threshold, 4)}? → {s.go}</li>)}</ol>
      <p className="text-sm">Daun {m.leaf} (n = {int(m.leaf_n)}): proporsi kelas {cls.map((c, i) => `${c} ${pct(m.leaf_dist[i], 1)}`).join(', ')}.</p>
    </>);
  }
  if (!body || !mine) return <Callout kind="info" title="Perhitungan manual">Langkah manual tersedia untuk Regresi Logistik (target biner), Naïve Bayes, KNN, dan Decision Tree. Random Forest = rata-rata ratusan pohon.</Callout>;
  const diff = Math.max(...mine.map((v, i) => Math.abs(v - sk[i])));
  return (
    <Section title={`Perhitungan manual — baris uji ke-${m.index + 1} (baris data ${m.row_index + 1})`} right={<LiveBadge />}>
      {body}
      <p className="text-sm">Prediksi manual: <strong>{cls[mine.indexOf(Math.max(...mine))]}</strong> · aktual: {cls[m.actual]}</p>
      <Callout kind={diff < 1e-8 ? 'ok' : 'warn'} title="Verifikasi vs predict_proba()">scikit-learn = [{sk.map((v) => fmtFixed(v, 6)).join('; ')}] · selisih maks {diff.toExponential(2)} {diff < 1e-8 ? '✓ identik' : ''}</Callout>
    </Section>
  );
}

function Inner() {
  const [info, setInfo] = useState<{ key: string; columns: Column[] } | null>(null);
  const [target, setTarget] = useState('');
  const [feats, setFeats] = useState<string[]>([]);
  const [algo, setAlgo] = useState('logreg');
  const [scaler, setScaler] = useState('standard');
  const [split, setSplit] = useState('stratified');
  const [dateCol, setDateCol] = useState('');
  const [prm, setPrm] = useState<Record<string, any>>({ C: 1, k: 5, metric: 'euclidean', weights: 'uniform', criterion: 'gini', max_depth: 4, min_samples_leaf: 1, n_estimators: 200 });
  const [mi, setMi] = useState(0);
  const run = useAsync<any>();
  const lb = useAsync<any>();
  const cands = info ? info.columns.filter((c) => c.unique >= 2 && c.unique <= 20).map((c) => c.name) : [];
  const usable = info ? info.columns.filter((c) => c.kind !== 'teks') : [];
  function onReady(i: any) {
    setInfo(i);
    run.setData(null); lb.setData(null);
    const c = i.columns.filter((x: Column) => x.unique >= 2 && x.unique <= 20).map((x: Column) => x.name);
    const t = i.preset.target && c.includes(i.preset.target) ? i.preset.target : c[0] ?? '';
    setTarget(t);
    setFeats(i.preset.features ?? i.columns.filter((x: Column) => x.name !== t && x.kind !== 'teks').map((x: Column) => x.name).slice(0, 8));
    setDateCol(i.preset.date ?? '');
  }
  const go = (idx = mi) => info && run.run(() => py('dmx.classify', 'run', { data_key: info.key, target, features: feats.filter((f) => f !== target), algo, params: prm, scaler, split, date_col: dateCol || null, manual_index: idx }));
  const p = palette();
  const r = run.data;
  return (
    <div className="space-y-5">
      <Dataset dataKey="klasifikasi" tool="klasifikasi" onReady={onReady} />
      {info && (
        <Section title="Atur model" right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-4">
            <Select label="Target (2–20 kelas)" value={target} onChange={setTarget} options={cands.map((c) => ({ value: c, label: c }))} />
            <Select label="Algoritma" value={algo} onChange={setAlgo} options={ALGOS} />
            <Select label="Scaling numerik" value={scaler} onChange={setScaler} options={[{ value: 'none', label: 'Tanpa' }, { value: 'standard', label: 'StandardScaler' }, { value: 'minmax', label: 'MinMaxScaler' }]} />
            <Select label="Pembagian data" value={split} onChange={setSplit} options={[{ value: 'stratified', label: 'Acak berstrata 75:25' }, { value: 'temporal', label: 'Berbasis waktu 75:25' }]} />
          </div>
          {split === 'temporal' && <Select label="Kolom tanggal untuk urutan" value={dateCol} onChange={setDateCol} options={[{ value: '', label: '(urutan baris)' }, ...info.columns.map((c) => ({ value: c.name, label: c.name }))]} />}
          <div className="grid gap-4 md:grid-cols-4">
            {algo === 'logreg' && <Slider label="log₁₀ C (regularisasi L2)" value={Math.log10(prm.C)} min={-3} max={2} step={0.25} onChange={(v) => setPrm({ ...prm, C: 10 ** v })} format={(v) => `${fmt(v, 2)} (C = ${fmt(10 ** v, 3)})`} />}
            {algo === 'knn' && <>
              <Slider label="K" value={prm.k} min={1} max={31} step={2} onChange={(v) => setPrm({ ...prm, k: v })} />
              <Select label="Metrik jarak" value={prm.metric} onChange={(v) => setPrm({ ...prm, metric: v })} options={['euclidean', 'manhattan', 'cosine', 'chebyshev'].map((v) => ({ value: v, label: v }))} />
              <Select label="Bobot" value={prm.weights} onChange={(v) => setPrm({ ...prm, weights: v })} options={[{ value: 'uniform', label: 'uniform' }, { value: 'distance', label: 'distance (1/d)' }]} />
            </>}
            {(algo === 'tree' || algo === 'rf') && <Slider label="max_depth (0 = tanpa batas)" value={prm.max_depth} min={0} max={20} onChange={(v) => setPrm({ ...prm, max_depth: v })} />}
            {algo === 'tree' && <>
              <Select label="Kriteria" value={prm.criterion} onChange={(v) => setPrm({ ...prm, criterion: v })} options={[{ value: 'gini', label: 'Gini' }, { value: 'entropy', label: 'Entropy' }]} />
              <Slider label="min_samples_leaf" value={prm.min_samples_leaf} min={1} max={100} onChange={(v) => setPrm({ ...prm, min_samples_leaf: v })} />
            </>}
            {algo === 'rf' && <Slider label="n_estimators" value={prm.n_estimators} min={10} max={500} step={10} onChange={(v) => setPrm({ ...prm, n_estimators: v })} />}
          </div>
          <fieldset><legend className="label">Fitur — jangan sertakan kolom pembentuk target (kebocoran data)</legend>
            <div className="grid grid-cols-2 gap-1 md:grid-cols-4">{usable.filter((c) => c.name !== target).map((c) => <Toggle key={c.name} label={c.name} checked={feats.includes(c.name)} onChange={(v) => setFeats(v ? [...feats, c.name] : feats.filter((x) => x !== c.name))} />)}</div>
          </fieldset>
          <div className="flex flex-wrap gap-2">
            <button className="btn-primary" disabled={!target || !feats.length || run.loading} onClick={() => go()}>Latih & evaluasi</button>
            <button className="btn-ghost" disabled={!target || !feats.length || lb.loading} onClick={() => lb.run(() => py('dmx.classify', 'leaderboard', { data_key: info.key, target, features: feats.filter((f) => f !== target), split, date_col: dateCol || null }))}>Bandingkan semua algoritma</button>
          </div>
          {(run.loading || lb.loading) && <Spinner />}
          <ErrorBox error={run.error || lb.error} />
        </Section>
      )}
      {lb.data && (
        <Section title="Leaderboard (data uji 25%)">
          <Table head={['Model', 'Akurasi', 'Balanced', 'F1 makro', 'ROC-AUC', 'CV latih', 'Baseline mayoritas']} numeric={[1, 2, 3, 4, 5, 6]} rows={lb.data.rows.map((x: any) => [x.model, pct(x.accuracy, 1), pct(x.balanced_accuracy, 1), fmtFixed(x.f1_macro, 3), x.roc_auc === null ? '–' : fmtFixed(x.roc_auc, 3), `${pct(x.cv_acc, 1)} ± ${pct(x.cv_std, 1)}`, pct(x.baseline, 1)])} />
        </Section>
      )}
      {r && (
        <Section title={`Hasil: ${r.algo_name}`}>
          <p className="text-sm muted">{int(r.n)} baris · latih {int(r.n_train)} / uji {int(r.n_test)} · fitur numerik {r.numeric.length}, kategorikal {r.categorical.length} · praproses di dalam Pipeline (di-fit pada data latih).</p>
          <StatGrid>
            <Stat label="Akurasi uji" value={pct(r.metrics.accuracy, 1)} hint={`baseline ${pct(r.baseline, 1)}`} />
            <Stat label="Balanced acc." value={pct(r.metrics.balanced_accuracy, 1)} /><Stat label="F1 makro" value={fmtFixed(r.metrics.f1_macro, 3)} />
            <Stat label="ROC-AUC" value={r.metrics.roc_auc === null ? '–' : fmtFixed(r.metrics.roc_auc, 3)} />
            <Stat label="CV (latih)" value={`${pct(r.cv_acc, 1)} ± ${pct(r.cv_std, 1)}`} />
          </StatGrid>
          <div className="grid gap-5 lg:grid-cols-2">
            <Plot title="Confusion matrix (data uji)" height={340} data={[{ type: 'heatmap', z: r.confusion, x: r.classes, y: r.classes, colorscale: [[0, '#f5f5f2'], [1, '#0B7A53']], text: r.confusion, texttemplate: '%{text}', showscale: false }]} layout={{ yaxis: { autorange: 'reversed', title: { text: 'Aktual' } }, xaxis: { title: { text: 'Prediksi' } }, margin: { l: 110 } }} />
            {r.roc && <Plot title={`Kurva ROC (AUC ${fmtFixed(r.metrics.roc_auc, 3)})`} height={340} data={[{ type: 'scatter', mode: 'lines', x: r.roc.fpr, y: r.roc.tpr, line: { color: p.brand2 }, name: 'Model' }, { type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], line: { dash: 'dash', color: p.neutral }, name: 'Acak' }]} />}
            {(r.importances || r.coef) && <Plot title={r.coef ? 'Koefisien (skala praproses)' : 'Feature importance'} height={340} data={[{ type: 'bar', orientation: 'h', y: (r.coef ?? r.importances).map((x: any) => x[0]).reverse(), x: (r.coef ?? r.importances).map((x: any) => x[1]).reverse(), marker: { color: p.brand2 } }]} layout={{ margin: { l: 160 } }} />}
          </div>
          {r.rules && <details><summary className="cursor-pointer text-sm font-semibold">Aturan pohon (JIKA–MAKA)</summary><pre className="code-block mt-2 max-h-80 overflow-auto rounded-lg p-3 text-xs">{r.rules}</pre></details>}
          <Slider label="Baris uji untuk perhitungan manual" value={mi} min={0} max={r.n_test - 1} onChange={(v) => { setMi(v); go(v); }} format={(v) => String(v + 1)} />
        </Section>
      )}
      {r && <ManualPanel r={r} />}
    </div>
  );
}

export default function ClassificationTool() {
  return <ToolShell title="Klasifikasi" cases={['04', '05', '06', '07']}
    intro="Prediksi kelas/kategori pada CSV Anda dengan Regresi Logistik, Naïve Bayes, KNN, Decision Tree, atau Random Forest; atur parameter, lihat metrik, confusion matrix, ROC, leaderboard, dan perhitungan manual satu baris yang diverifikasi identik dengan scikit-learn."><Inner /></ToolShell>;
}
