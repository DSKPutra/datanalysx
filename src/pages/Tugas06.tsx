import { useEffect, useState } from 'react';
import DataSource from '../components/DataSource';
import ModulePage, { Summary } from '../components/ModulePage';
import Plot from '../components/Plot';
import { Callout, ErrorBox, LiveBadge, ReportBadge, Section, Select, Spinner, Stat, StatGrid, Table } from '../components/ui';
import { bySlug } from '../content/modules';
import { fmt, fmtFixed, int, pct } from '../lib/format';
import { gnbPosterior } from '../lib/manual/gnb';
import { py } from '../lib/pyodide/client';
import { palette } from '../lib/theme';
import { useAsync, useJson } from '../lib/useAsync';

const m = bySlug('tugas-06')!;
const MODES = [
  { value: 'report', label: 'Replikasi laporan (IQR dari seluruh periode)' },
  { value: 'report_none', label: 'Replikasi laporan TANPA penanganan outlier' },
  { value: 'train', label: 'Benar: IQR dari data latih saja (tanpa kebocoran)' },
  { value: 'none', label: 'Tanpa penanganan outlier (tanpa kebocoran)' },
];

function Members() {
  const { data: r } = useJson<any>('data/tugas-06.json');
  if (!r) return null;
  return (
    <Section title="Kelompok 4" right={<span className="text-xs muted">{r.anggota_sumber}</span>}>
      <Table head={['No', 'Nama', 'NIM']} rows={r.anggota.map((a: string[], i: number) => [i + 1, a[0], a[1]])} />
    </Section>
  );
}

function Report() {
  const { data: r } = useJson<any>('data/tugas-06.json');
  if (!r) return <Spinner text="Memuat hasil laporan …" />;
  const p = palette();
  return (
    <div className="space-y-5">
      <ReportBadge source={r.sumber} />
      <Section title="Dampak penanganan outlier" right={<span className="text-xs muted">{r.dampak_outlier.sumber}</span>}>
        <Plot height={260} data={[
          { type: 'bar', x: ['Akurasi', 'Brier score'], y: [r.dampak_outlier.akurasi_sebelum, r.dampak_outlier.brier_sebelum], name: 'Sebelum winsorization', marker: { color: p.down } },
          { type: 'bar', x: ['Akurasi', 'Brier score'], y: [r.dampak_outlier.akurasi_sesudah, r.dampak_outlier.brier_sesudah], name: 'Sesudah winsorization', marker: { color: p.up } }]} layout={{ barmode: 'group' }} />
        <Table head={['Fitur', 'Batas bawah %', 'Batas atas %', 'Di-cap']} numeric={[1, 2, 3]} rows={r.outlier} caption={r.outlier_sumber} />
      </Section>
      <Section title="Metrik data uji (856 hari)" right={<span className="text-xs muted">{r.metrik.sumber}</span>}>
        <StatGrid>
          <Stat label="Akurasi" value={pct(r.metrik.akurasi)} hint={`baseline ${pct(r.metrik.baseline)}`} /><Stat label="Presisi" value={pct(r.metrik.presisi)} />
          <Stat label="Recall" value={pct(r.metrik.recall)} /><Stat label="F1" value={fmtFixed(r.metrik.f1, 4)} />
          <Stat label="ROC-AUC" value={fmtFixed(r.metrik.roc_auc, 4)} /><Stat label="Brier" value={fmtFixed(r.metrik.brier, 4)} />
          <Stat label="CV AUC (5 lipatan)" value={`${fmtFixed(r.cv.auc_mean, 4)} ± ${fmtFixed(r.cv.auc_sd, 4)}`} /><Stat label="TN/FP/FN/TP" value={`${r.confusion.TN}/${r.confusion.FP}/${r.confusion.FN}/${r.confusion.TP}`} />
        </StatGrid>
      </Section>
      <Section title="Contoh manual 21-02-2023" right={<span className="text-xs muted">{r.worked_example.sumber}</span>}>
        <p>P(Turun/Tetap | x) = {pct(r.worked_example.posterior_turun)}, P(Naik | x) = {pct(r.worked_example.posterior_naik)} → prediksi Turun/Tetap; realisasi IHSG esok {fmt(r.worked_example.aktual_ihsg_besok, 4)}% (benar).</p>
      </Section>
    </div>
  );
}

function Lab() {
  const [key, setKey] = useState<string | null>(null);
  const [mode, setMode] = useState('report');
  const [feat, setFeat] = useState('SP500_ret');
  const run = useAsync<any>();
  const cmp = useAsync<any>();
  useEffect(() => { if (key) run.run(() => py('dmx.t06', 'run', { data_key: key, outlier_mode: mode })); }, [key, mode, run.run]);
  useEffect(() => { if (key) cmp.run(() => Promise.all(['report_none', 'report', 'none', 'train'].map((md) => py('dmx.t06', 'run', { data_key: key, outlier_mode: md }).then((x) => ({ md, x })))) as any); }, [key, cmp.run]);
  const r = run.data;
  const p = palette();
  return (
    <div className="space-y-5">
      <DataSource dataKey="t06" sample={m.sample} onReady={setKey} />
      {key && (
        <Section title="Gaussian Naïve Bayes" right={<LiveBadge />}>
          <Select label="Penanganan outlier" value={mode} onChange={setMode} options={MODES} />
          <ErrorBox error={run.error} />
          {r && (<>
            <StatGrid>
              <Stat label="Akurasi" value={pct(r.metrics.accuracy)} hint={`baseline ${pct(r.baseline_majority)}`} /><Stat label="Brier" value={fmtFixed(r.metrics.brier, 4)} />
              <Stat label="ROC-AUC" value={fmtFixed(r.metrics.roc_auc, 4)} /><Stat label="F1" value={fmtFixed(r.metrics.f1, 4)} />
              <Stat label="CV AUC" value={`${fmtFixed(r.cv_auc_mean, 4)} ± ${fmtFixed(r.cv_auc_std, 4)}`} /><Stat label="Latih / uji" value={`${int(r.n_train)} / ${int(r.n_test)}`} />
            </StatGrid>
            <div className="grid gap-5 lg:grid-cols-2">
              <Plot title="USDIDR_ret: mentah (log y) vs setelah perlakuan" height={300} data={[
                { type: 'bar', x: r.usdidr_raw_hist.edges.slice(0, -1), y: r.usdidr_raw_hist.counts, name: 'Mentah', marker: { color: p.down }, opacity: 0.6 },
                { type: 'bar', x: r.usdidr_clean_hist.edges.slice(0, -1), y: r.usdidr_clean_hist.counts, name: 'Setelah', marker: { color: p.up }, opacity: 0.8 }]} layout={{ barmode: 'overlay', yaxis: { type: 'log' }, xaxis: { title: { text: 'return harian (%)' } } }} />
              <div>
                <Select label="Kurva likelihood fitur" value={feat} onChange={setFeat} options={r.features.map((f: string) => ({ value: f, label: f }))} />
                <Plot height={260} data={[
                  { type: 'scatter', mode: 'lines', x: r.likelihood_curves[feat].x, y: r.likelihood_curves[feat].pdf0, name: 'Turun/Tetap', line: { color: p.down } },
                  { type: 'scatter', mode: 'lines', x: r.likelihood_curves[feat].x, y: r.likelihood_curves[feat].pdf1, name: 'Naik', line: { color: p.up, dash: 'dash' } }]} />
              </div>
              <Plot title="Kurva ROC" height={280} data={[{ type: 'scatter', mode: 'lines', x: r.roc.fpr, y: r.roc.tpr, line: { color: p.brand2 }, name: 'GNB' }, { type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], line: { dash: 'dash', color: p.neutral }, name: 'Acak' }]} />
              <Plot title="TimeSeriesSplit 5 lipatan" height={280} data={[
                { type: 'bar', x: r.cv.map((f: any) => `F${f.fold}`), y: r.cv.map((f: any) => f.roc_auc), name: 'ROC-AUC', marker: { color: p.brand2 } },
                { type: 'bar', x: r.cv.map((f: any) => `F${f.fold}`), y: r.cv.map((f: any) => f.accuracy), name: 'Akurasi', marker: { color: p.neutral } }]} layout={{ barmode: 'group', yaxis: { range: [0.4, 0.7] } }} />
            </div>
            {Object.keys(r.bounds).length > 0 && <Table caption="Batas winsorization 3×IQR (Hasil Live)" head={['Fitur', 'Bawah', 'Atas', 'Di-cap']} numeric={[1, 2, 3]} rows={Object.entries(r.bounds).map(([k, v]: any) => [k, fmt(v.lo, 3), fmt(v.hi, 3), v.n_capped])} />}
          </>)}
          {cmp.data && <Table caption="Perbandingan mode (Hasil Live)" head={['Mode', 'Akurasi', 'Brier', 'ROC-AUC', 'n uji']} numeric={[1, 2, 3, 4]} rows={(cmp.data as any[]).map(({ md, x }) => [MODES.find((o) => o.value === md)!.label, pct(x.metrics.accuracy), fmtFixed(x.metrics.brier, 4), fmtFixed(x.metrics.roc_auc, 4), x.n_test])} />}
        </Section>
      )}
    </div>
  );
}

function Manual() {
  const [key, setKey] = useState<string | null>(null);
  const [date, setDate] = useState('2023-02-21');
  const [mode, setMode] = useState('report');
  const dates = useAsync<any>();
  const s = useAsync<any>();
  useEffect(() => { if (key) dates.run(() => py('dmx.t06', 'run', { data_key: key, outlier_mode: mode })); }, [key, mode, dates.run]);
  useEffect(() => { if (key) s.run(() => py('dmx.t06', 'model_params', { data_key: key, date, outlier_mode: mode })); }, [key, date, mode, s.run]);
  const r = s.data;
  const ts = r ? gnbPosterior(r.x, r.theta, r.var, r.prior) : null;
  const L = ['Turun/Tetap (0)', 'Naik (1)'];
  return (
    <div className="space-y-5">
      <DataSource dataKey="t06" sample={m.sample} onReady={setKey} />
      {r && ts && (
        <Section title={`Worked example Gaussian NB — ${r.date}`} right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-2">
            <Select label="Tanggal data uji" value={date} onChange={setDate} options={(dates.data?.test_dates ?? [date]).map((d: string) => ({ value: d, label: d }))} />
            <Select label="Mode" value={mode} onChange={setMode} options={MODES.slice(0, 3)} />
          </div>
          <p className="text-sm"><strong>Langkah 1–2.</strong> Prior: P(Turun/Tetap) = {fmtFixed(r.prior[0], 4)}, P(Naik) = {fmtFixed(r.prior[1], 4)} → ln = {fmtFixed(ts.logPrior[0], 4)}, {fmtFixed(ts.logPrior[1], 4)}.</p>
          <p className="text-sm"><strong>Langkah 3.</strong> Likelihood P(xᵢ|y) = 1/√(2πσ²)·exp(−(x−μ)²/(2σ²)) per fitur (var termasuk var_smoothing scikit-learn):</p>
          <Table head={['Fitur', 'x', 'μ₀', 'σ₀', 'P(x|0)', 'μ₁', 'σ₁', 'P(x|1)']} numeric={[1, 2, 3, 4, 5, 6, 7]} rows={r.features.map((f: string, i: number) => [f, fmt(r.x[i], 4), fmt(ts.perFeature[0][i].mu, 4), fmt(ts.perFeature[0][i].sigma, 4), fmt(ts.perFeature[0][i].pdf, 4), fmt(ts.perFeature[1][i].mu, 4), fmt(ts.perFeature[1][i].sigma, 4), fmt(ts.perFeature[1][i].pdf, 4)])} />
          <Table caption="Langkah 4 — gabungan dalam ruang log" head={['Kelas', 'ln P(y)', 'Σ ln P(xᵢ|y)', 'ln gabungan']} numeric={[1, 2, 3]} rows={[0, 1].map((c) => [L[c], fmtFixed(ts.logPrior[c], 4), fmtFixed(ts.sumLog[c], 4), fmtFixed(ts.joint[c], 4)])} />
          <p className="text-sm"><strong>Langkah 5–6.</strong> Log-sum-exp → posterior P(Turun/Tetap|x) = <strong>{pct(ts.posterior[0])}</strong>, P(Naik|x) = <strong>{pct(ts.posterior[1])}</strong> → MAP: <strong>{L[ts.pred]}</strong>. Realisasi: target {r.actual} (IHSG esok {fmt(r.ihsg_ret_next, 4)}%).</p>
          <Callout kind={Math.max(...ts.posterior.map((v, i) => Math.abs(v - r.proba_sklearn[i]))) < 1e-8 ? 'ok' : 'warn'} title="Langkah 7 — verifikasi">predict_proba() = [{fmtFixed(r.proba_sklearn[0], 8)}, {fmtFixed(r.proba_sklearn[1], 8)}] · selisih maks TypeScript {Math.max(...ts.posterior.map((v, i) => Math.abs(v - r.proba_sklearn[i]))).toExponential(2)} ✓</Callout>
        </Section>
      )}
      {s.loading && <Spinner />}
    </div>
  );
}

export default function Tugas06() {
  return <ModulePage m={m} summary={<Summary m={m} extra={<Members />} />} report={<Report />} lab={<Lab />} manual={<Manual />} />;
}
