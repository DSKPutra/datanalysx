import { useEffect, useState } from 'react';
import Plot from '../components/Plot';
import PyGate from '../components/PyGate';
import { Callout, ErrorBox, LiveBadge, Section, Select, Spinner, Stat, StatGrid, Table, Toggle } from '../components/ui';
import { fmt, fmtFixed } from '../lib/format';
import { py } from '../lib/pyodide/client';
import { palette } from '../lib/theme';
import { useAsync } from '../lib/useAsync';
import { Profile, SamplePicker } from './LabShared';

function Inner() {
  const [key, setKey] = useState<string | null>(null);
  const [ver, setVer] = useState(0);
  const prof = useAsync<any>();
  const [target, setTarget] = useState('');
  const [feats, setFeats] = useState<string[]>([]);
  const [split, setSplit] = useState<'random' | 'temporal'>('random');
  const [dateCol, setDateCol] = useState('');
  const run = useAsync<any>();
  useEffect(() => { if (key) prof.run(() => py('dmx.eda', 'profile', { data_key: key })); }, [key, ver, prof.run]);
  const nums: string[] = prof.data ? prof.data.columns.filter((c: any) => c.kind === 'numerik').map((c: any) => c.name) : [];
  const cats: string[] = prof.data ? prof.data.columns.filter((c: any) => c.kind === 'kategorikal').map((c: any) => c.name) : [];
  const dates: string[] = prof.data ? prof.data.columns.filter((c: any) => /date|tanggal|time/i.test(c.name)).map((c: any) => c.name) : [];
  useEffect(() => {
    if (!prof.data) return;
    const t = nums.includes('revenue_usd_bn') ? 'revenue_usd_bn' : nums.includes('close') ? 'close' : nums[nums.length - 1];
    setTarget(t);
    setFeats(t === 'revenue_usd_bn' ? ['rd_spend_usd_bn', 'capex_usd_bn', 'operating_margin_pct'] : nums.filter((n) => n !== t).slice(0, 5));
    setDateCol(dates[0] ?? '');
  }, [prof.data]);
  const p = palette();
  const r = run.data;
  return (
    <div className="space-y-5">
      <SamplePicker dataKey="reglab" onReady={(k) => { setKey(k); setVer((v) => v + 1); run.setData(null); }} />
      {key && <Profile dataKey={key} version={ver} />}
      {prof.data && (
        <Section title="Bandingkan model regresi" right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-3">
            <Select label="Target (numerik)" value={target} onChange={setTarget} options={nums.map((n) => ({ value: n, label: n }))} />
            <Select label="Pembagian data" value={split} onChange={setSplit} options={[{ value: 'random', label: 'Acak 80:20 + KFold 5' }, { value: 'temporal', label: 'Berbasis waktu 80:20 + TimeSeriesSplit' }]} />
            <Select label="Kolom tanggal (urutan waktu)" value={dateCol} onChange={setDateCol} options={[{ value: '', label: '(tidak ada)' }, ...dates.map((d) => ({ value: d, label: d }))]} />
          </div>
          <fieldset><legend className="label">Fitur</legend>
            <div className="grid grid-cols-2 gap-1 md:grid-cols-4">{[...nums, ...cats].filter((c) => c !== target).map((c) => <Toggle key={c} label={c} checked={feats.includes(c)} onChange={(v) => setFeats(v ? [...feats, c] : feats.filter((x) => x !== c))} />)}</div>
          </fieldset>
          <button className="btn-primary" disabled={!feats.length || run.loading} onClick={() => run.run(() => py('dmx.datalab', 'regress', { data_key: key, target, features: feats, split, date_col: dateCol || null }))}>Latih & bandingkan</button>
          {split === 'temporal' && !dateCol && <Callout kind="warn" title="Tanpa kolom tanggal">Split berbasis waktu memakai urutan baris apa adanya.</Callout>}
          {run.loading && <Spinner text="Melatih 7 model + validasi silang …" />}
          <ErrorBox error={run.error} />
          {r && (<>
            <Table head={['Model', 'MAE uji', 'RMSE uji', 'R² uji', 'R² CV', 'RMSE CV']} numeric={[1, 2, 3, 4, 5]} rows={r.rows.map((x: any) => [x.model, fmt(x.MAE, 4), fmt(x.RMSE, 4), fmtFixed(x.R2, 4), x.cv_r2 === null ? '–' : fmtFixed(x.cv_r2, 4), x.cv_rmse === null ? '–' : fmt(x.cv_rmse, 4)])} />
            <div className="grid gap-4 lg:grid-cols-2">
              <Plot title="R² uji per model" height={300} data={[{ type: 'bar', x: r.rows.map((x: any) => x.model), y: r.rows.map((x: any) => x.R2), marker: { color: p.brand2 } }]} layout={{ yaxis: { range: [Math.min(0, ...r.rows.map((x: any) => x.R2)), 1] } }} />
              <Plot title="Aktual vs prediksi" height={300} data={r.rows.filter((x: any) => !x.model.startsWith('Baseline')).slice(0, 4).map((x: any) => [x.model, r.preds[x.model]]).map(([name, pr]: any, i: number) => ({ type: 'scatter', mode: 'markers', name, x: r.actual, y: pr, marker: { size: 4, color: p.c[i], opacity: 0.6 } }))} />
            </div>
            {r.assumptions && (
              <Section title="Panel uji asumsi klasik (OLS pada data latih)">
                <StatGrid>
                  <Stat label="Breusch–Pagan" value={`LM ${fmt(r.assumptions.breusch_pagan_lm, 2)}`} hint={`p ${fmtFixed(r.assumptions.breusch_pagan_p, 4)} ${r.assumptions.breusch_pagan_p < 0.05 ? '→ heteroskedastis' : ''}`} />
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
    </div>
  );
}

export default function RegressionLab() {
  useEffect(() => { document.title = 'Regression Lab · DATANALYSX'; }, []);
  return (
    <div className="space-y-5">
      <header><h1 className="text-4xl font-bold">Regression Lab</h1><p className="muted">Gabungan Tugas 01–03: unggah CSV, pilih target & fitur, bandingkan Linear, log-log, Ridge, Lasso, Decision Tree, Random Forest, kNN; split acak vs berbasis waktu; panel asumsi klasik. Semua praproses di dalam Pipeline (tanpa kebocoran).</p></header>
      <PyGate><Inner /></PyGate>
    </div>
  );
}
