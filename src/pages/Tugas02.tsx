import { useEffect, useState } from 'react';
import DataSource from '../components/DataSource';
import ModulePage, { DataTab } from '../components/ModulePage';
import Plot from '../components/Plot';
import { Callout, ErrorBox, LiveBadge, ReportBadge, Section, Slider, Spinner, Table, Toggle } from '../components/ui';
import { bySlug } from '../content/modules';
import { fmt, fmtFixed } from '../lib/format';
import { py } from '../lib/pyodide/client';
import { palette } from '../lib/theme';
import { useAsync, useJson } from '../lib/useAsync';

const m = bySlug('tugas-02')!;

function Report() {
  const { data: r } = useJson<any>('data/tugas-02.json');
  if (!r) return <Spinner text="Memuat hasil laporan …" />;
  const p = palette();
  return (
    <div className="space-y-5">
      <ReportBadge source={r.sumber} />
      <Section title="Perbandingan model (300 baris uji)" right={<span className="text-xs muted">{r.model_sumber}</span>}>
        <Table head={['Model', 'MAE', 'RMSE', 'R²']} numeric={[1, 2, 3]} rows={r.model.map((x: any) => [x.model, fmtFixed(x.MAE, 6), fmtFixed(x.RMSE, 6), fmtFixed(x.R2, 6)])} />
        <Plot height={280} data={[{ type: 'bar', x: r.model.map((x: any) => x.model), y: r.model.map((x: any) => x.R2), marker: { color: [p.neutral, p.brand2] }, text: r.model.map((x: any) => fmtFixed(x.R2, 3)), textposition: 'outside' }]} layout={{ yaxis: { range: [0, 1], title: { text: 'R²' } } }} />
      </Section>
      <Section title="Persamaan Linear Regression" right={<span className="text-xs muted">{r.persamaan.sumber}</span>}>
        <p className="font-mono text-sm">severity′ = 1,637357 − 0,078486·difficulty_score + 0,007658·prompt_length + 0,038742·context_length</p>
      </Section>
      <Section title="Statistik deskriptif" right={<span className="text-xs muted">{r.statistik_sumber}</span>}>
        <Table head={['Variabel', 'Rata-rata', 'SD', 'Min', 'Maks']} numeric={[1, 2, 3, 4]} rows={r.statistik} />
      </Section>
    </div>
  );
}

function FeatureTable({ dataKey }: { dataKey: string }) {
  const s = useAsync<any>();
  const [sel, setSel] = useState(0);
  useEffect(() => { s.run(() => py('dmx.t02', 'feature_table', { data_key: dataKey, n_rows: 12 })); }, [dataKey, s.run]);
  if (!s.data) return <Spinner />;
  const row = s.data.rows[sel];
  return (
    <Section title="Rekayasa fitur: teks → angka (klik baris)">
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="overflow-x-auto">
          <table className="tbl">
            <thead><tr><th>task_id</th><th>difficulty</th><th className="num">prompt_len</th><th className="num">context_len</th><th>severity</th></tr></thead>
            <tbody>{s.data.rows.map((r: any, i: number) => (
              <tr key={r.task_id} onClick={() => setSel(i)} onKeyDown={(e) => e.key === 'Enter' && setSel(i)} tabIndex={0} className="cursor-pointer" style={i === sel ? { outline: '2px solid var(--brand-2)' } : {}}>
                <td>{r.task_id}</td><td>{r.difficulty} → {r.difficulty_score}</td><td className="num">{r.prompt_length}</td><td className="num">{r.context_length}</td><td>{r.failure_severity} → {r.severity_score}</td>
              </tr>))}</tbody>
          </table>
        </div>
        <div className="space-y-2 text-sm">
          <p><span className="label">Prompt ({row.prompt_length} kata)</span>{row.prompt}</p>
          <p><span className="label">Context ({row.context_length} kata)</span>{row.context}</p>
          <p><span className="label">Domain · failure_type</span>{row.domain} · {row.failure_type}</p>
        </div>
      </div>
    </Section>
  );
}

function Lab() {
  const [key, setKey] = useState<string | null>(null);
  const [nEst, setNEst] = useState(100);
  const [depth, setDepth] = useState(0);
  const [extra, setExtra] = useState<string[]>([]);
  const run = useAsync<any>();
  const base = useAsync<any>();
  useEffect(() => { if (key) base.run(() => py('dmx.t02', 'run', { data_key: key })); }, [key, base.run]);
  useEffect(() => {
    if (key) run.run(() => py('dmx.t02', 'compare', { data_key: key, n_estimators: nEst, max_depth: depth || null, extra_cats: extra.length ? extra : null }));
  }, [key, nEst, depth, extra, run.run]);
  const p = palette();
  const r = run.data;
  return (
    <div className="space-y-5">
      <DataSource dataKey="t02" sample={m.sample} onReady={setKey} />
      {key && <FeatureTable dataKey={key} />}
      {base.data && (
        <Section title="context_length vs severity_score">
          <Plot height={300} data={[{ type: 'scatter', mode: 'markers', x: base.data.scatter_context.x, y: base.data.scatter_context.y.map((v: number, i: number) => v + ((i * 7919) % 100) / 400 - 0.125), marker: { size: 5, color: p.brand2, opacity: 0.35 }, hovertemplate: 'context=%{x}<extra></extra>' }]} layout={{ xaxis: { title: { text: 'context_length (kata)' } }, yaxis: { title: { text: 'severity_score (dengan jitter)' }, tickvals: [1, 2, 3, 4] } }} />
        </Section>
      )}
      {key && (
        <Section title="Bandingkan model regresi" right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-3">
            <Slider label="n_estimators (Random Forest)" value={nEst} min={10} max={300} step={10} onChange={setNEst} />
            <Slider label="max_depth (0 = tanpa batas)" value={depth} min={0} max={20} onChange={setDepth} />
            <fieldset><legend className="label">Tambah fitur kategorikal (one-hot)</legend>
              {['domain', 'failure_type'].map((c) => <Toggle key={c} label={c} checked={extra.includes(c)} onChange={(v) => setExtra(v ? [...extra, c] : extra.filter((x) => x !== c))} />)}
            </fieldset>
          </div>
          {extra.includes('failure_type') && <Callout kind="warn" title="Kebocoran label">failure_type adalah hasil evaluasi yang hampir menentukan keparahan; R² ≈ 1 di sini bukan bukti model lebih baik.</Callout>}
          <ErrorBox error={run.error} />
          {run.loading && <Spinner />}
          {r && (
            <>
              <Table head={['Model', 'MAE', 'RMSE', 'R² uji', 'R² CV 5-lipatan (± SD)']} numeric={[1, 2, 3, 4]}
                rows={r.rows.map((x: any) => [x.model, fmt(x.MAE, 4), fmt(x.RMSE, 4), fmtFixed(x.R2, 4), `${fmtFixed(x.cv_r2_mean, 3)} ± ${fmtFixed(x.cv_r2_std, 3)}`])} />
              <div className="grid gap-4 md:grid-cols-2">
                <Plot title="R² uji per model" height={300} data={[{ type: 'bar', x: r.rows.map((x: any) => x.model), y: r.rows.map((x: any) => x.R2), marker: { color: p.brand2 } }]} layout={{ yaxis: { range: [Math.min(0, ...r.rows.map((x: any) => x.R2)), 1] } }} />
                <Plot title="Aktual vs prediksi: Linear vs Random Forest" height={300} data={[
                  { type: 'box', x: r.actual, y: r.preds['Linear Regression'], name: 'Linear', marker: { color: p.neutral } },
                  { type: 'box', x: r.actual, y: r.preds['Random Forest'], name: 'Random Forest', marker: { color: p.brand2 } }]}
                  layout={{ boxmode: 'group', xaxis: { title: { text: 'severity aktual' } }, yaxis: { title: { text: 'prediksi' } } }} />
              </div>
            </>
          )}
        </Section>
      )}
    </div>
  );
}

function Manual() {
  const [key, setKey] = useState<string | null>(null);
  const [idx, setIdx] = useState(0);
  const s = useAsync<any>();
  const mm = useAsync<any>();
  useEffect(() => { if (key) s.run(() => py('dmx.t02', 'manual_predict', { data_key: key, test_index: idx })); }, [key, idx, s.run]);
  useEffect(() => { if (key) mm.run(() => py('dmx.t02', 'manual_metrics', { data_key: key })); }, [key, mm.run]);
  const r = s.data;
  const names = ['difficulty_score', 'prompt_length', 'context_length'];
  return (
    <div className="space-y-5">
      <DataSource dataKey="t02" sample={m.sample} onReady={setKey} />
      {key && (
        <Section title="Satu prediksi dihitung tangan" right={<LiveBadge />}>
          <Slider label="Indeks baris uji" value={idx} min={0} max={299} onChange={setIdx} />
          {r && (
            <>
              <p className="text-sm">Fitur: {names.map((n) => `${n} = ${r.x[n]}`).join(', ')} · aktual = {r.actual}</p>
              <Table head={['Suku', 'Koefisien', 'x', 'Koefisien × x']} numeric={[1, 2, 3]} rows={[
                ['b₀ (intersep)', fmt(r.lin.intercept, 6), '1', fmt(r.lin.intercept, 6)],
                ...names.map((n, i) => [n, fmt(r.lin.coef[i], 6), r.x[n], fmt(r.lin.terms[i], 6)]),
                ['Σ = prediksi LR', '', '', fmt(r.lin.manual, 8)]]} />
              <p className="text-sm">Random Forest: rata-rata {r.rf.tree_preds.length} prediksi pohon = <strong>{fmt(r.rf.manual, 8)}</strong> (predict() = {fmt(r.rf.sklearn, 8)}).</p>
              <Plot height={240} title="Prediksi tiap pohon" data={[{ type: 'histogram', x: r.rf.tree_preds, marker: { color: palette().brand2 } }]} layout={{ xaxis: { title: { text: 'prediksi pohon' } }, shapes: [{ type: 'line', x0: r.rf.manual, x1: r.rf.manual, y0: 0, y1: 1, yref: 'paper', line: { color: palette().gold, width: 3 } }] }} />
              <Callout kind={r.max_abs_diff < 1e-8 ? 'ok' : 'warn'} title="Verifikasi">Selisih manual vs predict() = {r.max_abs_diff.toExponential(2)} {r.max_abs_diff < 1e-8 ? '✓ identik' : ''}</Callout>
            </>
          )}
        </Section>
      )}
      {mm.data && (
        <Section title="MAE, RMSE, R² dihitung manual">
          <p className="text-sm">MAE = Σ|y−ŷ|/n = {fmt(mm.data.sum_abs, 4)} / {mm.data.n_test}; RMSE = √(Σ(y−ŷ)²/n) = √({fmt(mm.data.sum_sq, 4)} / {mm.data.n_test}); R² = 1 − SSE/SST = 1 − {fmt(mm.data.sum_sq, 4)} / {fmt(mm.data.sst, 4)}</p>
          <Table head={['Metrik', 'Manual', 'scikit-learn']} numeric={[1, 2]} rows={['MAE', 'RMSE', 'R2'].map((k) => [k, fmtFixed(mm.data.manual[k], 8), fmtFixed(mm.data.sklearn[k], 8)])} />
        </Section>
      )}
    </div>
  );
}

export default function Tugas02() {
  return <ModulePage m={m} report={<Report />} lab={<Lab />} manual={<Manual />} data={<DataTab m={m}>
    <Callout kind="warn" title="Catatan asal dataset">Laporan menamai dataset “AI Economy Pulse Dataset 2022–2026” (Kaggle belbino). Kolom & isi berkas yang dipakai identik dengan “LLM Agent Failure Analysis Benchmark Dataset” (Kaggle sunil123kumar, Apache 2.0). Aplikasi menyertakan berkas tersebut dengan atribusi ini.</Callout>
  </DataTab>} />;
}
