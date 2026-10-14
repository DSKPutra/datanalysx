import { useEffect, useState } from 'react';
import DataSource from '../components/DataSource';
import ModulePage from '../components/ModulePage';
import Plot from '../components/Plot';
import { Callout, ErrorBox, LiveBadge, ReportBadge, Section, Select, Slider, Spinner, Stat, StatGrid, Table } from '../components/ui';
import { bySlug } from '../content/modules';
import { fmt, fmtFixed, int, pct } from '../lib/format';
import { knnManual, type Metric } from '../lib/manual/knn';
import { py } from '../lib/pyodide/client';
import { palette } from '../lib/theme';
import { useAsync, useJson } from '../lib/useAsync';

const m = bySlug('tugas-07')!;
const METRICS = [{ value: 'euclidean', label: 'Euclidean' }, { value: 'manhattan', label: 'Manhattan' }, { value: 'cosine', label: 'Cosine' }, { value: 'chebyshev', label: 'Chebyshev' }] as const;
const SCALERS = [{ value: 'none', label: 'Tanpa scaling' }, { value: 'standard', label: 'StandardScaler' }, { value: 'minmax', label: 'MinMaxScaler' }] as const;
const CLS = ['High Reach', 'Viral Hit'];

function Report() {
  const { data: r } = useJson<any>('data/tugas-07.json');
  if (!r) return <Spinner text="Memuat hasil laporan …" />;
  const p = palette();
  const t = r.knn_test;
  return (
    <div className="space-y-5">
      <ReportBadge source="model_summary.json Tugas 07" />
      <Section title={`KNN terbaik (${r.gridsearch.best_scaler}, ${r.gridsearch.best_metric}, K = ${r.gridsearch.best_k}, ${r.gridsearch.best_weights})`}>
        <StatGrid>
          <Stat label="CV (GridSearch)" value={pct(r.gridsearch.best_cv_acc)} hint={`${int(r.gridsearch.n_candidates)} kombinasi, ${int(r.gridsearch.n_fits)} fit`} />
          <Stat label="Akurasi uji" value={pct(t.accuracy)} /><Stat label="Presisi" value={pct(t.precision)} /><Stat label="Recall" value={pct(t.recall)} />
          <Stat label="F1" value={fmtFixed(t.f1, 3)} /><Stat label="ROC-AUC" value={fmtFixed(t.roc_auc, 3)} /><Stat label="TN/FP/FN/TP" value={`${t.tn}/${t.fp}/${t.fn}/${t.tp}`} />
        </StatGrid>
      </Section>
      <div className="grid gap-5 lg:grid-cols-2">
        <Section title="Pengaruh scaling (K = 5)">
          <Plot height={260} data={[{ type: 'bar', x: ['Tanpa', 'Standard', 'MinMax'], y: ['none', 'standard', 'minmax'].map((k) => r.scaling_comparison[k].cv_acc_k5), marker: { color: [p.neutral, p.c[0], p.brand2] }, text: ['none', 'standard', 'minmax'].map((k) => pct(r.scaling_comparison[k].cv_acc_k5)), textposition: 'outside' }]} layout={{ yaxis: { range: [0.6, 0.95] } }} />
        </Section>
        <Section title="Kurva K (validasi silang)">
          <Plot height={260} data={[
            { type: 'scatter', mode: 'lines', x: r.k_curve.k, y: r.k_curve.cv_acc_uniform, name: 'uniform', line: { color: p.neutral } },
            { type: 'scatter', mode: 'lines', x: r.k_curve.k, y: r.k_curve.cv_acc_distance, name: 'distance', line: { color: p.brand2 } }]} layout={{ xaxis: { title: { text: 'K' } } }} />
        </Section>
      </div>
      <Section title="Perbandingan model">
        <Table head={['Model', 'Akurasi uji', 'F1 uji', 'ROC-AUC uji', 'CV akurasi']} numeric={[1, 2, 3, 4]} rows={r.model_comparison.map((x: any) => [x.model, pct(x.accuracy), fmtFixed(x.f1, 3), fmtFixed(x.roc_auc, 3), pct(x.cv_acc)])} />
      </Section>
      <div className="grid gap-5 lg:grid-cols-2">
        <Section title="Curse of dimensionality">
          <Plot height={260} data={[{ type: 'scatter', mode: 'lines+markers', x: r.curse_of_dimensionality.dims_theory, y: r.curse_of_dimensionality.contrast_theory, line: { color: p.brand2 } }]} layout={{ xaxis: { type: 'log', title: { text: 'dimensi' } }, yaxis: { type: 'log', title: { text: 'kontras (dmax−dmin)/dmin' } } }} />
        </Section>
        <Section title="Permutation importance">
          <Plot height={260} data={[{ type: 'bar', orientation: 'h', y: r.permutation_importance.map((x: any) => x.feature).reverse(), x: r.permutation_importance.map((x: any) => x.mean).reverse(), marker: { color: p.brand2 } }]} layout={{ margin: { l: 130 } }} />
        </Section>
      </div>
    </div>
  );
}

function Lab() {
  const [key, setKey] = useState<string | null>(null);
  const [k, setK] = useState(5);
  const [metric, setMetric] = useState<string>('cosine');
  const [weights, setWeights] = useState<'uniform' | 'distance'>('distance');
  const [scaler, setScaler] = useState<string>('minmax');
  const ev = useAsync<any>();
  const kc = useAsync<any>();
  const nb = useAsync<any>();
  const curse = useAsync<any>();
  const timing = useAsync<any>();
  useEffect(() => { if (key) ev.run(() => py('dmx.t07', 'evaluate', { data_key: key, k, metric, weights, scaler })); }, [key, k, metric, weights, scaler, ev.run]);
  useEffect(() => { if (key) kc.run(() => py('dmx.t07', 'k_curve', { data_key: key, metric, weights, scaler })); }, [key, metric, weights, scaler, kc.run]);
  useEffect(() => { if (key) nb.run(() => py('dmx.t07', 'neighbors', { data_key: key, k, metric, weights, scaler })); }, [key, k, metric, weights, scaler, nb.run]);
  const r = ev.data;
  const p = palette();
  const n = nb.data;
  return (
    <div className="space-y-5">
      <DataSource dataKey="t07" sample={m.sample} onReady={setKey} />
      {key && (
        <Section title="Atur KNN → akurasi CV live" right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-4">
            <Slider label="K" value={k} min={1} max={31} step={2} onChange={setK} />
            <Select label="Metrik jarak" value={metric} onChange={setMetric} options={METRICS as any} />
            <Select label="Bobot" value={weights} onChange={setWeights} options={[{ value: 'uniform', label: 'uniform' }, { value: 'distance', label: 'distance (1/d)' }]} />
            <Select label="Scaler" value={scaler} onChange={setScaler} options={SCALERS as any} />
          </div>
          <ErrorBox error={ev.error} />
          {r && <StatGrid>
            <Stat label="Akurasi CV (latih, 5×5)" value={pct(r.cv_acc)} hint={`± ${pct(r.cv_acc_std)}`} />
            <Stat label="Akurasi uji (n=22)" value={pct(r.test.accuracy)} hint={`baseline ${pct(r.baseline_majority_test)}`} />
            <Stat label="F1 uji" value={fmtFixed(r.test.f1, 3)} /><Stat label="ROC-AUC uji" value={fmtFixed(r.test.roc_auc, 3)} />
            <Stat label="Pembanding: Regresi Logistik CV" value={pct(r.lr_reference.cv_acc)} />
          </StatGrid>}
          {kc.data && <Plot title="Kurva K: latih vs validasi silang" height={280} data={[
            { type: 'scatter', mode: 'lines', x: kc.data.rows.map((x: any) => x.k), y: kc.data.rows.map((x: any) => x.train), name: 'Latih', line: { color: p.text } },
            { type: 'scatter', mode: 'lines+markers', x: kc.data.rows.map((x: any) => x.k), y: kc.data.rows.map((x: any) => x.cv), name: 'CV', line: { color: p.brand2 } }]}
            layout={{ shapes: [{ type: 'line', x0: k, x1: k, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: p.gold } }], xaxis: { title: { text: 'K' } } }} />}
          {n && <Plot title={`K tetangga terdekat untuk satu data uji (proyeksi PCA; ruang asli ${n.feature_names.length} dimensi)`} height={380} data={[
            ...[0, 1].map((c) => ({ type: 'scatter', mode: 'markers', name: CLS[c], x: n.pca_train.filter((_: any, i: number) => n.train_y[i] === c).map((v: number[]) => v[0]), y: n.pca_train.filter((_: any, i: number) => n.train_y[i] === c).map((v: number[]) => v[1]), marker: { size: 8, color: c ? p.up : p.down, symbol: c ? 'circle' : 'square', opacity: 0.6 } })),
            { type: 'scatter', mode: 'markers', name: 'Tetangga', x: n.sk_neighbors.map((i: number) => n.pca_train[i][0]), y: n.sk_neighbors.map((i: number) => n.pca_train[i][1]), marker: { size: 16, color: 'rgba(0,0,0,0)', line: { width: 2.5, color: p.gold } } },
            { type: 'scatter', mode: 'markers', name: 'Data uji', x: [n.pca_query[0]], y: [n.pca_query[1]], marker: { size: 16, symbol: 'star', color: p.text } }]} />}
          <p className="text-xs muted">Catatan: jarak dihitung di ruang fitur asli; PCA 2D hanya ilustrasi sehingga tetangga tidak selalu tampak paling dekat.</p>
        </Section>
      )}
      {key && (
        <div className="grid gap-5 lg:grid-cols-2">
          <Section title="Curse of dimensionality">
            <button className="btn-ghost" disabled={curse.loading} onClick={() => curse.run(() => py('dmx.t07', 'curse', { data_key: key }))}>Tambah fitur derau 0–100</button>
            {curse.loading && <Spinner />}
            {curse.data && (<>
              <Plot height={260} data={[
                { type: 'scatter', mode: 'lines+markers', x: curse.data.rows.map((x: any) => x.noise_dims), y: curse.data.rows.map((x: any) => x.acc_euclidean), name: 'Euclidean K=5', line: { color: p.down } },
                { type: 'scatter', mode: 'lines+markers', x: curse.data.rows.map((x: any) => x.noise_dims), y: curse.data.rows.map((x: any) => x.acc_cosine), name: 'Cosine berbobot', line: { color: p.brand2 } }]} layout={{ xaxis: { title: { text: 'jumlah fitur derau' } }, yaxis: { title: { text: 'akurasi CV' } } }} />
              <Table head={['Dimensi', 'Kontras jarak']} numeric={[0, 1]} rows={curse.data.rows.map((x: any) => [x.total_dims, fmtFixed(x.contrast, 3)])} />
            </>)}
          </Section>
          <Section title="Lazy learner: waktu fit vs prediksi">
            <button className="btn-ghost" disabled={timing.loading} onClick={() => timing.run(() => py('dmx.t07', 'timing', {}))}>Ukur waktu (data sintetis 500–20.000 baris)</button>
            {timing.loading && <Spinner />}
            {timing.data && (<>
              <Plot height={260} data={[
                { type: 'scatter', mode: 'lines+markers', x: timing.data.rows.map((x: any) => x.n), y: timing.data.rows.map((x: any) => x.knn_brute_fit_ms), name: 'KNN fit', line: { color: p.brand2, dash: 'dot' } },
                { type: 'scatter', mode: 'lines+markers', x: timing.data.rows.map((x: any) => x.n), y: timing.data.rows.map((x: any) => x.knn_brute_pred_ms), name: 'KNN prediksi', line: { color: p.brand2 } },
                { type: 'scatter', mode: 'lines+markers', x: timing.data.rows.map((x: any) => x.n), y: timing.data.rows.map((x: any) => x.logreg_fit_ms), name: 'LogReg fit', line: { color: p.down, dash: 'dot' } },
                { type: 'scatter', mode: 'lines+markers', x: timing.data.rows.map((x: any) => x.n), y: timing.data.rows.map((x: any) => x.logreg_pred_ms), name: 'LogReg prediksi', line: { color: p.down } }]} layout={{ xaxis: { type: 'log', title: { text: 'n data latih' } }, yaxis: { type: 'log', title: { text: 'ms' } } }} />
              <p className="text-xs muted">{timing.data.note}</p>
            </>)}
          </Section>
        </div>
      )}
    </div>
  );
}

function Manual() {
  const [key, setKey] = useState<string | null>(null);
  const [pos, setPos] = useState<number | null>(null);
  const [metric, setMetric] = useState<Metric>('cosine');
  const [weights, setWeights] = useState<'uniform' | 'distance'>('distance');
  const [k, setK] = useState(5);
  const s = useAsync<any>();
  useEffect(() => { if (key) s.run(() => py('dmx.t07', 'neighbors', { data_key: key, test_pos: pos, k, metric, weights, scaler: 'minmax' })); }, [key, pos, k, metric, weights, s.run]);
  const r = s.data;
  const ts = r ? knnManual(r.query, r.train, r.train_y, k, metric, weights) : null;
  const diff = ts ? Math.max(...ts.proba.map((v, i) => Math.abs(v - r.sk_proba[i]))) : 1;
  return (
    <div className="space-y-5">
      <DataSource dataKey="t07" sample={m.sample} onReady={setKey} />
      {r && ts && (
        <Section title={`Perhitungan manual KNN — data uji ke-${r.test_pos + 1} (baris ${r.row_index}${r.shortcode ? `, ${r.shortcode}` : ''})`} right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-4">
            <Slider label="Data uji" value={r.test_pos} min={0} max={r.n_test - 1} onChange={setPos} format={(v) => `${v + 1}`} />
            <Slider label="K" value={k} min={1} max={15} step={2} onChange={setK} />
            <Select label="Metrik" value={metric} onChange={setMetric} options={METRICS as any} />
            <Select label="Bobot" value={weights} onChange={setWeights} options={[{ value: 'uniform', label: 'uniform' }, { value: 'distance', label: 'distance' }]} />
          </div>
          <p className="text-sm">Langkah 1 — titik uji ditransformasi oleh Pipeline (IQR capping + MinMaxScaler yang di-fit pada data latih; niche one-hot). Langkah 2 — hitung jarak ke {r.train.length} data latih dan urutkan. Langkah 3 — ambil {k} terdekat. Langkah 4 — voting{weights === 'distance' ? ' berbobot 1/d' : ''}.</p>
          <Table head={['Peringkat', 'Indeks latih', 'Kelas', 'Jarak', 'Bobot']} numeric={[0, 1, 3, 4]} rows={ts.neighbors.map((i, j) => [j + 1, r.train_index[i], CLS[r.train_y[i]], fmtFixed(ts.distances[i], 6), fmtFixed(ts.weights[j], 4)])} />
          <p className="text-sm">Suara: High Reach = {fmtFixed(ts.votes[0], 4)}, Viral Hit = {fmtFixed(ts.votes[1], 4)} → P = [{fmtFixed(ts.proba[0], 6)}, {fmtFixed(ts.proba[1], 6)}] → prediksi <strong>{CLS[ts.pred]}</strong> (aktual {CLS[r.actual]}).</p>
          <Callout kind={diff < 1e-8 ? 'ok' : 'warn'} title="Verifikasi vs scikit-learn">predict_proba() = [{fmtFixed(r.sk_proba[0], 6)}, {fmtFixed(r.sk_proba[1], 6)}], tetangga sama: {[...ts.neighbors].sort().join(',') === [...r.sk_neighbors].sort().join(',') ? 'ya' : 'tidak'} · selisih maks {diff.toExponential(2)}</Callout>
          <details><summary className="cursor-pointer text-sm font-semibold">Vektor fitur titik uji (ter-transformasi)</summary>
            <Table head={['Fitur', 'Nilai mentah', 'Nilai transformasi']} numeric={[2]} rows={r.feature_names.map((f: string, i: number) => [f, String(r.raw_query[f.replace(/^num__|^bin__/, '')] ?? ''), fmt(r.query[i], 4)])} />
          </details>
        </Section>
      )}
      {s.loading && <Spinner />}
    </div>
  );
}

export default function Tugas07() {
  return <ModulePage m={m} report={<Report />} lab={<Lab />} manual={<Manual />} />;
}
