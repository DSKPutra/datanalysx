import { useEffect, useState } from 'react';
import Dataset, { type Column } from '../../components/Dataset';
import Plot from '../../components/Plot';
import { Callout, ErrorBox, LiveBadge, Section, Select, Slider, Spinner, Stat, StatGrid, Table, Toggle } from '../../components/ui';
import { fmt, fmtFixed } from '../../lib/format';
import { simpleRegression } from '../../lib/manual/ols';
import { py } from '../../lib/pyodide/client';
import { palette } from '../../lib/theme';
import { useAsync } from '../../lib/useAsync';
import ToolShell from './ToolShell';

function Manual({ dataKey, nums, x0, y0 }: { dataKey: string; nums: string[]; x0?: string; y0?: string }) {
  const [x, setX] = useState(x0 && nums.includes(x0) ? x0 : nums[0]);
  const [y, setY] = useState(y0 && nums.includes(y0) ? y0 : nums[1] ?? nums[0]);
  const [n, setN] = useState(10);
  const [log, setLog] = useState(false);
  const s = useAsync<any>();
  useEffect(() => { if (x && y && x !== y) s.run(() => py('dmx.eda', 'simple_pair', { data_key: dataKey, x, y, n_rows: n, log })); }, [dataKey, x, y, n, log, s.run]);
  const r = s.data;
  const ts = r && r.n >= 2 ? simpleRegression(r.x, r.y) : null;
  const diff = ts ? Math.max(Math.abs(ts.a - r.a_sklearn), Math.abs(ts.b - r.b_sklearn)) : null;
  return (
    <Section title="Perhitungan manual: regresi linear sederhana" right={<LiveBadge />}>
      <div className="grid gap-4 md:grid-cols-4">
        <Select label="X (bebas)" value={x} onChange={setX} options={nums.map((c) => ({ value: c, label: c }))} />
        <Select label="Y (terikat)" value={y} onChange={setY} options={nums.map((c) => ({ value: c, label: c }))} />
        <Slider label="Jumlah baris pertama" value={n} min={3} max={30} onChange={setN} />
        <Toggle label="Skala ln (log-log)" checked={log} onChange={setLog} />
      </div>
      <ErrorBox error={s.error} />
      {ts && r && (<>
        <Table head={['i', 'X', 'Y', 'XY', 'X²']} numeric={[1, 2, 3, 4]} rows={[...r.x.map((xv: number, i: number) => [i + 1, fmt(xv, 4), fmt(r.y[i], 4), fmt(xv * r.y[i], 4), fmt(xv * xv, 4)]), ['Σ', fmt(ts.sx, 4), fmt(ts.sy, 4), fmt(ts.sxy, 4), fmt(ts.sxx, 4)]]} />
        <p className="font-mono text-sm">b = [nΣXY − ΣXΣY] / [nΣX² − (ΣX)²] = <strong>{fmt(ts.b, 8)}</strong> · a = [ΣY − bΣX] / n = <strong>{fmt(ts.a, 8)}</strong></p>
        <Callout kind={diff! < 1e-8 ? 'ok' : 'warn'} title="Verifikasi vs scikit-learn LinearRegression">a = {fmt(r.a_sklearn, 8)}, b = {fmt(r.b_sklearn, 8)} · selisih maks {diff!.toExponential(2)}</Callout>
      </>)}
    </Section>
  );
}

function Inner() {
  const [info, setInfo] = useState<{ key: string; columns: Column[]; preset: any } | null>(null);
  const [target, setTarget] = useState('');
  const [feats, setFeats] = useState<string[]>([]);
  const [split, setSplit] = useState<'random' | 'temporal'>('random');
  const [dateCol, setDateCol] = useState('');
  const run = useAsync<any>();
  const nums = info ? info.columns.filter((c) => c.kind === 'numerik').map((c) => c.name) : [];
  const cats = info ? info.columns.filter((c) => c.kind === 'kategorikal').map((c) => c.name) : [];
  const allCols = info ? info.columns.map((c) => c.name) : [];
  function onReady(i: { key: string; columns: Column[]; preset: any }) {
    setInfo(i);
    run.setData(null);
    const n = i.columns.filter((c) => c.kind === 'numerik').map((c) => c.name);
    const t = i.preset.target && n.includes(i.preset.target) ? i.preset.target : n[n.length - 1] ?? '';
    setTarget(t);
    setFeats(i.preset.features ?? n.filter((c) => c !== t).slice(0, 5));
    setDateCol(i.preset.date ?? '');
    setSplit(i.preset.date ? 'temporal' : 'random');
  }
  const p = palette();
  const r = run.data;
  return (
    <div className="space-y-5">
      <Dataset dataKey="regresi" tool="regresi" onReady={onReady} />
      {info && nums.length < 2 && <Callout kind="warn" title="Butuh minimal dua kolom numerik">Dataset ini tidak memiliki cukup kolom numerik untuk regresi.</Callout>}
      {info && nums.length >= 2 && (
        <Section title="Bandingkan model regresi" right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-3">
            <Select label="Target (numerik)" value={target} onChange={setTarget} options={nums.map((n) => ({ value: n, label: n }))} />
            <Select label="Pembagian data" value={split} onChange={setSplit} options={[{ value: 'random', label: 'Acak 80:20 + KFold 5' }, { value: 'temporal', label: 'Berbasis waktu 80:20 + TimeSeriesSplit' }]} />
            <Select label="Kolom tanggal (opsional)" value={dateCol} onChange={setDateCol} options={[{ value: '', label: '(tidak ada)' }, ...allCols.map((d) => ({ value: d, label: d }))]} />
          </div>
          <fieldset><legend className="label">Fitur (hindari kolom yang merupakan turunan target!)</legend>
            <div className="grid grid-cols-2 gap-1 md:grid-cols-4">{[...nums, ...cats].filter((c) => c !== target && c !== dateCol).map((c) => <Toggle key={c} label={c} checked={feats.includes(c)} onChange={(v) => setFeats(v ? [...feats, c] : feats.filter((x) => x !== c))} />)}</div>
          </fieldset>
          <button className="btn-primary" disabled={!feats.length || run.loading} onClick={() => run.run(() => py('dmx.datalab', 'regress', { data_key: info.key, target, features: feats.filter((f) => f !== target), split, date_col: dateCol || null }))}>Latih & bandingkan 7 model</button>
          {run.loading && <Spinner text="Melatih model + validasi silang …" />}
          <ErrorBox error={run.error} />
          {r && (<>
            <Table head={['Model', 'MAE uji', 'RMSE uji', 'R² uji', 'R² CV', 'RMSE CV']} numeric={[1, 2, 3, 4, 5]} rows={r.rows.map((x: any) => [x.model, fmt(x.MAE, 4), fmt(x.RMSE, 4), fmtFixed(x.R2, 4), x.cv_r2 === null ? '–' : fmtFixed(x.cv_r2, 4), x.cv_rmse === null ? '–' : fmt(x.cv_rmse, 4)])} />
            <div className="grid gap-4 lg:grid-cols-2">
              <Plot title="R² uji per model" height={300} data={[{ type: 'bar', x: r.rows.map((x: any) => x.model), y: r.rows.map((x: any) => x.R2), marker: { color: p.brand2 } }]} layout={{ yaxis: { range: [Math.min(0, ...r.rows.map((x: any) => x.R2)), 1] } }} />
              <Plot title="Aktual vs prediksi (4 model teratas)" height={300} data={r.rows.filter((x: any) => !x.model.startsWith('Baseline')).slice(0, 4).map((x: any, i: number) => ({ type: 'scatter', mode: 'markers', name: x.model, x: r.actual, y: r.preds[x.model], marker: { size: 4, color: p.c[i], opacity: 0.6 } }))} />
            </div>
            {r.assumptions && (
              <Section title="Uji asumsi klasik (OLS pada data latih)">
                <StatGrid>
                  <Stat label="Breusch–Pagan" value={`LM ${fmt(r.assumptions.breusch_pagan_lm, 2)}`} hint={`p ${fmtFixed(r.assumptions.breusch_pagan_p, 4)}${r.assumptions.breusch_pagan_p < 0.05 ? ' → heteroskedastis' : ''}`} />
                  <Stat label="Shapiro–Wilk" value={fmtFixed(r.assumptions.shapiro_w, 4)} hint={`p ${fmtFixed(r.assumptions.shapiro_p, 4)}`} />
                  <Stat label="Durbin–Watson" value={fmtFixed(r.assumptions.durbin_watson, 3)} hint="≈2 = tanpa autokorelasi" />
                  <Stat label="VIF maks" value={fmtFixed(Math.max(...(Object.values(r.assumptions.vif).filter((v) => v !== null) as number[])), 2)} hint="> 5 waspada" />
                </StatGrid>
                <Plot title="Residual vs fitted" height={260} data={[{ type: 'scatter', mode: 'markers', x: r.assumptions.fitted, y: r.assumptions.resid, marker: { size: 4, color: p.brand2 } }]} />
              </Section>
            )}
          </>)}
        </Section>
      )}
      {info && nums.length >= 2 && <Manual key={info.key + info.columns.length + (info.preset.x ?? '')} dataKey={info.key} nums={nums} x0={info.preset.x} y0={info.preset.y} />}
    </div>
  );
}

export default function RegressionTool() {
  return <ToolShell title="Regresi" cases={['01', '02', '03']}
    intro="Prediksi nilai numerik pada CSV Anda: Linear, log-log, Ridge, Lasso, Decision Tree, Random Forest, dan kNN; split acak atau berbasis waktu; panel uji asumsi klasik; serta perhitungan manual regresi sederhana yang diverifikasi dengan scikit-learn."><Inner /></ToolShell>;
}
