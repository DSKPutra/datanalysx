import { useEffect, useMemo, useState, type ReactNode } from 'react';
import DataSource from '../components/DataSource';
import ModulePage, { DataTab } from '../components/ModulePage';
import Plot from '../components/Plot';
import { Callout, ErrorBox, LiveBadge, ReportBadge, Section, Select, Spinner, Stat, StatGrid, Table, Toggle } from '../components/ui';
import { bySlug } from '../content/modules';
import { fmt, fmtFixed, int } from '../lib/format';
import { simpleRegression, normalEquation } from '../lib/manual/ols';
import { py } from '../lib/pyodide/client';
import { palette } from '../lib/theme';
import { useAsync, useJson } from '../lib/useAsync';

const m = bySlug('tugas-03')!;
const pv = (v: number) => (v < 0.001 ? '< 0,001' : fmtFixed(v, 3));

/** Panel bergaya widget Orange: judul + ikon, kontrol kiri, hasil kanan. */
function Widget({ icon, title, controls, children }: { icon: string; title: string; controls?: ReactNode; children: ReactNode }) {
  return (
    <section className="overflow-hidden rounded-2xl border" style={{ borderColor: 'var(--line)', background: 'var(--surface)' }}>
      <header className="on-brand flex items-center gap-2 px-4 py-2 text-sm font-semibold">
        <span aria-hidden className="flex h-6 w-6 items-center justify-center rounded-full bg-white/20">{icon}</span>{title}
      </header>
      <div className="grid gap-4 p-4 lg:grid-cols-[220px_1fr]">
        <div className="space-y-3 border-b pb-3 lg:border-b-0 lg:border-r lg:pb-0 lg:pr-4" style={{ borderColor: 'var(--line)' }}>{controls ?? <p className="text-xs muted">Tanpa pengaturan.</p>}</div>
        <div className="min-w-0">{children}</div>
      </div>
    </section>
  );
}

function Report() {
  const { data: r } = useJson<any>('data/tugas-03.json');
  if (!r) return <Spinner text="Memuat hasil laporan …" />;
  return (
    <div className="space-y-5">
      <ReportBadge source={r.sumber} />
      <Section title="Empat model OLS (seluruh 589 baris)">
        <Table head={['Model', 'Persamaan', 'R²', 'Catatan', 'Sumber']} numeric={[2]} rows={r.model.map((x: any) => [x.model, x.persamaan, fmtFixed(x.R2, 4), x.catatan ?? '', x.sumber])} />
        <p className="text-sm">Efek tetap perusahaan: elastisitas R&D {fmtFixed(r.efek_tetap.elastisitas, 4)} (SE {fmtFixed(r.efek_tetap.se, 4)}), R² within {fmtFixed(r.efek_tetap.R2_within, 4)} — {r.efek_tetap.sumber}.</p>
      </Section>
      <Section title="Test and Score — validasi silang 10-lipatan" right={<span className="text-xs muted">{r.cv_sumber}</span>}>
        <Table head={['Model', 'MSE', 'RMSE', 'MAE', 'R²']} numeric={[1, 2, 3, 4]} rows={r.cv} />
      </Section>
      <div className="grid gap-5 lg:grid-cols-2">
        <Section title="Hold-out 80:20" right={<span className="text-xs muted">{r.holdout_sumber}</span>}>
          <Table head={['Model', 'MSE', 'RMSE', 'MAE', 'R²']} numeric={[1, 2, 3, 4]} rows={r.holdout} />
        </Section>
        <Section title="Uji asumsi klasik" right={<span className="text-xs muted">{r.asumsi_sumber}</span>}>
          <Table head={['Model', 'BP p', 'Shapiro p', 'DW']} numeric={[1, 2, 3]} rows={r.asumsi} />
        </Section>
      </div>
      <Section title="Lima CSV yang ditinjau" right={<span className="text-xs muted">{r.csv_sumber}</span>}>
        <Table head={['Berkas', 'Isi', 'Observasi', 'Kelayakan']} numeric={[2]} rows={r.csv_ditinjau} />
      </Section>
      <Section title="Statistik deskriptif (n = 589)" right={<span className="text-xs muted">{r.statistik_sumber}</span>}>
        <Table head={['Variabel', 'Rata-rata', 'SD', 'Min', 'Median', 'Maks', 'Skewness']} numeric={[1, 2, 3, 4, 5, 6]} rows={r.statistik} />
      </Section>
    </div>
  );
}

function Lab() {
  const [key, setKey] = useState<string | null>(null);
  const fit = useAsync<any>();
  const stats = useAsync<any>();
  const band = useAsync<any>();
  const cv = useAsync<any>();
  const diag = useAsync<any>();
  const pred = useAsync<any>();
  const [log, setLog] = useState(false);
  const [which, setWhich] = useState<'simple' | 'simple_log' | 'multiple' | 'multiple_log'>('simple');
  const [coefModel, setCoefModel] = useState<'simple' | 'simple_log' | 'multiple' | 'multiple_log'>('multiple_log');
  const [inp, setInp] = useState({ rd: 10, capex: 8, margin: 30 });
  useEffect(() => {
    if (!key) return;
    fit.run(() => py('dmx.t03', 'fit_models', { data_key: key }));
    stats.run(() => py('dmx.t03', 'feature_stats', { data_key: key }));
    cv.run(() => py('dmx.t03', 'cv_compare', { data_key: key }));
  }, [key, fit.run, stats.run, cv.run]);
  useEffect(() => { if (key) band.run(() => py('dmx.t03', 'regression_band', { data_key: key, log })); }, [key, log, band.run]);
  useEffect(() => { if (key) diag.run(() => py('dmx.t03', 'diagnostics', { data_key: key, which })); }, [key, which, diag.run]);
  useEffect(() => { if (key) pred.run(() => py('dmx.t03', 'predict', { data_key: key, ...inp })); }, [key, inp, pred.run]);
  const p = palette();
  const segColors = useMemo(() => {
    const segs = [...new Set((band.data?.segment ?? []) as string[])];
    return Object.fromEntries(segs.map((s, i) => [s, p.c[i % p.c.length]]));
  }, [band.data, p.c]);
  const f = fit.data;
  const modelOpts = [{ value: 'simple', label: 'Sederhana – linear' }, { value: 'simple_log', label: 'Sederhana – log-log' }, { value: 'multiple', label: 'Berganda – linear' }, { value: 'multiple_log', label: 'Berganda – log-log' }] as const;
  return (
    <div className="space-y-5">
      <DataSource dataKey="t03" sample={m.sample} onReady={setKey} />
      {key && <LiveBadge />}
      <ErrorBox error={fit.error || cv.error || diag.error} />
      {f && (
        <Widget icon="▦" title="Data Table" controls={<StatGrid><Stat label="Baris" value={int(f.n_clean)} hint={`${f.n_dropped_zero} baris nol dibuang`} /><Stat label="Perusahaan" value={f.n_companies} /></StatGrid>}>
          <Table head={['Perusahaan', 'Tahun', 'Segmen', 'Pendapatan', 'R&D', 'Capex', 'Margin %']} numeric={[3, 4, 5, 6]}
            rows={f.points.company.slice(0, 12).map((c: string, i: number) => [c, String(f.points.year[i]), f.points.segment[i], f.points.revenue[i], f.points.rd[i], f.points.capex[i], f.points.margin[i]])} />
        </Widget>
      )}
      {stats.data && (
        <Widget icon="Σ" title="Feature Statistics & Distributions" controls={<Toggle label="Skala logaritma" checked={log} onChange={setLog} />}>
          <Table head={['Fitur', 'Rata-rata', 'Median', 'SD', 'Min', 'Maks', 'Skew']} numeric={[1, 2, 3, 4, 5, 6]}
            rows={stats.data.features.map((x: any) => [x.feature, x.mean, x.median, x.std, x.min, x.max, x.skew])} />
          {f && <div className="grid gap-3 md:grid-cols-2">
            {[['revenue', 'Pendapatan'], ['rd', 'R&D'], ['capex', 'Capex'], ['margin', 'Margin']].map(([k, lab]) => (
              <Plot key={k} height={200} title={`${lab}${log && k !== 'margin' ? ' (ln)' : ''}`} data={[{ type: 'histogram', x: (f.points[k] as number[]).map((v) => (log && k !== 'margin' ? Math.log(v) : v)), marker: { color: p.brand2 }, nbinsx: 30 }]} layout={{ margin: { t: 32, b: 30, l: 40, r: 8 } }} />
            ))}
          </div>}
        </Widget>
      )}
      {stats.data && (
        <Widget icon="▮" title="Box / Violin Plot — ln(pendapatan) per segmen">
          <Plot height={380} data={Object.entries(stats.data.log_revenue_by_segment).map(([s, v], i) => ({ type: 'violin', y: v as number[], name: s, box: { visible: true }, meanline: { visible: true }, line: { color: p.c[i % p.c.length] }, points: false }))} layout={{ showlegend: false, yaxis: { title: { text: 'ln pendapatan' } } }} />
        </Widget>
      )}
      {f && (
        <Widget icon="ρ" title="Correlations (Pearson)" controls={<p className="text-xs muted">Skala asli vs skala log: korelasi Pendapatan–R&D naik dari 0,91 ke 0,94.</p>}>
          <div className="grid gap-3 md:grid-cols-2">
            {[['Skala asli', f.corr_pearson, ['Pendapatan', 'R&D', 'Capex', 'Margin']], ['Skala log', f.corr_log, ['ln Pendapatan', 'ln R&D', 'ln Capex', 'Margin']]].map(([t, z, lab]: any) => (
              <Plot key={t} height={300} title={t} data={[{ type: 'heatmap', z, x: lab, y: lab, zmin: -1, zmax: 1, colorscale: 'RdBu', reversescale: true, text: z.map((r: number[]) => r.map((v) => v.toFixed(2))), texttemplate: '%{text}' }]} layout={{ margin: { l: 90, t: 32, b: 60 } }} />
            ))}
          </div>
        </Widget>
      )}
      {band.data && (
        <Widget icon="⟋" title="Scatter Plot + garis regresi, CI & PI 95%" controls={<><Toggle label="Log-log" checked={log} onChange={setLog} /><p className="text-xs muted">R² = {fmtFixed(band.data.r2, 4)}<br />a = {fmt(band.data.coef[0], 4)}, b = {fmt(band.data.coef[1], 4)}</p></>}>
          <Plot height={420} data={[
            { type: 'scatter', mode: 'lines', x: band.data.grid, y: band.data.pi_high, line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
            { type: 'scatter', mode: 'lines', x: band.data.grid, y: band.data.pi_low, fill: 'tonexty', fillcolor: 'rgba(184,134,11,0.15)', line: { width: 0 }, name: 'PI 95%' },
            { type: 'scatter', mode: 'lines', x: band.data.grid, y: band.data.ci_high, line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
            { type: 'scatter', mode: 'lines', x: band.data.grid, y: band.data.ci_low, fill: 'tonexty', fillcolor: 'rgba(11,122,83,0.35)', line: { width: 0 }, name: 'CI 95%' },
            ...Object.keys(segColors).map((s) => ({ type: 'scatter', mode: 'markers', name: s, x: band.data.x.filter((_: number, i: number) => band.data.segment[i] === s), y: band.data.y.filter((_: number, i: number) => band.data.segment[i] === s), marker: { size: 6, color: segColors[s], opacity: 0.8 } })),
            { type: 'scatter', mode: 'lines', x: band.data.grid, y: band.data.fit, line: { color: p.text, width: 2 }, name: 'Garis regresi' }]}
            layout={{ xaxis: { title: { text: log ? 'ln R&D' : 'R&D (miliar USD)' } }, yaxis: { title: { text: log ? 'ln Pendapatan' : 'Pendapatan (miliar USD)' } }, legend: { font: { size: 10 } } }} />
        </Widget>
      )}
      {f && (
        <Widget icon="β" title="Linear Regression — koefisien" controls={<Select label="Model" value={coefModel} onChange={setCoefModel} options={modelOpts as any} />}>
          {(() => { const s = f[coefModel]; return (<>
            <Table head={['Parameter', 'Koefisien', 'SE', 't', 'p', 'CI 95%']} numeric={[1, 2, 3, 4]} rows={s.names.map((n: string, i: number) => [n, fmt(s.coef[i], 4), fmt(s.se[i], 4), fmt(s.t[i], 2), pv(s.p[i]), `[${fmt(s.ci_low[i], 3)} ; ${fmt(s.ci_high[i], 3)}]`])} />
            <p className="text-sm">R² = {fmtFixed(s.r2, 4)} · R² terkoreksi = {fmtFixed(s.adj_r2, 4)} · F = {fmt(s.F, 1)} (p {pv(s.F_p)}){s.vif ? ` · VIF = ${s.vif.map((v: number) => fmtFixed(v, 2)).join(' / ')}` : ''}</p>
          </>); })()}
        </Widget>
      )}
      <Widget icon="✓" title="Test and Score — validasi silang 10-lipatan" controls={<p className="text-xs muted">KFold(10, shuffle=True, random_state=42). Matriks: peluang model baris lebih baik daripada kolom (uji-t berpasangan terkoreksi Nadeau–Bengio, MSE).</p>}>
        {cv.loading && <Spinner text="Melatih 8 model × 10 lipatan …" />}
        {cv.data && (<>
          <Table head={['Model', 'RMSE', 'MAE', 'R²', 'Median galat %']} numeric={[1, 2, 3, 4]} rows={cv.data.rows.map((x: any) => [x.model, fmtFixed(x.RMSE, 3), fmtFixed(x.MAE, 3), fmtFixed(x.R2, 3), fmtFixed(x.median_ape, 1)])} />
          <Plot height={360} title="Model Comparison by MSE" data={[{ type: 'heatmap', z: cv.data.comparison.p_better, x: cv.data.comparison.names, y: cv.data.comparison.names, zmin: 0, zmax: 1, colorscale: [[0, '#f7f7f7'], [1, '#0B7A53']], text: cv.data.comparison.p_better.map((r: (number | null)[]) => r.map((v) => (v === null ? '' : v.toFixed(3)))), texttemplate: '%{text}' }]} layout={{ margin: { l: 150, b: 120 } }} />
          <Callout kind="info" title="Pelajaran">Peringkat berubah menurut metrik: RMSE memihak Linear berganda (galat perusahaan raksasa kecil), MAE memihak Random Forest & log-log (galat perusahaan tipikal kecil).</Callout>
        </>)}
      </Widget>
      <Widget icon="⚕" title="Diagnostik residual" controls={<Select label="Model" value={which} onChange={setWhich} options={modelOpts as any} />}>
        {diag.data && (<>
          <StatGrid>
            <Stat label="Breusch–Pagan" value={`LM ${fmt(diag.data.tests.bp_lm, 2)}`} hint={`p ${pv(diag.data.tests.bp_p)}`} />
            <Stat label="Shapiro–Wilk" value={fmtFixed(diag.data.tests.shapiro_w, 4)} hint={`p ${pv(diag.data.tests.shapiro_p)}`} />
            <Stat label="Durbin–Watson" value={fmtFixed(diag.data.tests.dw, 3)} />
            <Stat label="Cook maks." value={fmt(Math.max(...diag.data.cooks), 2)} hint={`${diag.data.top_influential[0].company} ${diag.data.top_influential[0].year}`} />
          </StatGrid>
          <div className="grid gap-3 md:grid-cols-2">
            <Plot height={260} title="Residual vs Fitted" data={[{ type: 'scatter', mode: 'markers', x: diag.data.fitted, y: diag.data.resid, text: diag.data.label, marker: { size: 4, color: p.brand2 } }]} layout={{ margin: { t: 32 } }} />
            <Plot height={260} title="Q–Q normal" data={[{ type: 'scatter', mode: 'markers', x: diag.data.qq_theoretical, y: diag.data.qq_sample, marker: { size: 4, color: p.brand2 } }, { type: 'scatter', mode: 'lines', x: [-3.5, 3.5], y: [-3.5, 3.5], line: { color: p.gold, dash: 'dash' } }]} layout={{ showlegend: false, margin: { t: 32 } }} />
            <Plot height={260} title="Scale–Location" data={[{ type: 'scatter', mode: 'markers', x: diag.data.fitted, y: diag.data.sqrt_abs_std_resid, marker: { size: 4, color: p.brand2 } }]} layout={{ margin: { t: 32 } }} />
            <Plot height={260} title="Jarak Cook" data={[{ type: 'bar', y: diag.data.cooks, text: diag.data.label, marker: { color: p.down } }]} layout={{ margin: { t: 32 }, shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: diag.data.cooks_threshold, y1: diag.data.cooks_threshold, line: { color: p.gold, dash: 'dash' } }], yaxis: { type: 'log' } }} />
          </div>
          <p className="text-xs muted">Ganti ke log-log untuk melihat heteroskedastisitas model sederhana menghilang (BP p 0,975 pada laporan).</p>
        </>)}
      </Widget>
      <Widget icon="ƒ" title="Predictions — kalkulator" controls={<div className="space-y-2">
        {([['rd', 'R&D (miliar USD)'], ['capex', 'Capex (miliar USD)'], ['margin', 'Margin operasi (%)']] as const).map(([k, l]) => (
          <label key={k} className="block"><span className="label">{l}</span><input type="number" className="input" value={inp[k]} min={0.01} step="any" onChange={(e) => setInp({ ...inp, [k]: Math.max(0.01, Number(e.target.value)) })} /></label>
        ))}</div>}>
        {pred.data && <Table head={['Model', 'Prediksi pendapatan (miliar USD)']} numeric={[1]} rows={Object.entries(pred.data).map(([k, v]) => [k, fmtFixed(v as number, 2)])} />}
        {(inp.rd > 62.7 || inp.capex > 51 || inp.rd < 0.01) && <Callout kind="warn" title="Ekstrapolasi">Masukan di luar rentang data (R&D 0,01–62,67; capex 0,01–50,96 miliar USD); prediksi tidak dapat diandalkan.</Callout>}
        <p className="text-xs muted">Contoh laporan (Tabel 19): R&D 10, capex 8, margin 30 → 52,58 / 65,23 / 52,62 / 68,73.</p>
      </Widget>
    </div>
  );
}

function Manual() {
  const [key, setKey] = useState<string | null>(null);
  const [log, setLog] = useState(false);
  const s = useAsync<any>();
  const mm = useAsync<any>();
  useEffect(() => { if (key) s.run(() => py('dmx.t03', 'manual_simple', { data_key: key, n_rows: 10, log })); }, [key, log, s.run]);
  useEffect(() => { if (key) mm.run(() => py('dmx.t03', 'manual_multiple', { data_key: key, n_rows: 12 })); }, [key, mm.run]);
  const r = s.data;
  const ts = r ? simpleRegression(r.rows.map((x: any) => x.x), r.rows.map((x: any) => x.y)) : null;
  const tm = mm.data ? normalEquation(mm.data.X, mm.data.y) : null;
  return (
    <div className="space-y-5">
      <DataSource dataKey="t03" sample={m.sample} onReady={setKey} />
      {r && ts && (
        <Section title="Regresi sederhana dengan rumus materi (10 observasi pertama)" right={<LiveBadge />}>
          <Toggle label="Skala log-log" checked={log} onChange={setLog} />
          <Table head={['Perusahaan', 'Tahun', 'X (R&D)', 'Y (Pendapatan)', 'XY', 'X²']} numeric={[2, 3, 4, 5]} rows={[...r.rows.map((x: any) => [x.company, String(x.year), fmt(x.x, 4), fmt(x.y, 4), fmt(x.xy, 4), fmt(x.x2, 4)]), ['Σ', '', fmt(ts.sx, 4), fmt(ts.sy, 4), fmt(ts.sxy, 4), fmt(ts.sxx, 4)]]} />
          <p className="font-mono text-sm">b = [nΣXY − ΣXΣY] / [nΣX² − (ΣX)²] = [{r.n}·{fmt(ts.sxy, 4)} − {fmt(ts.sx, 4)}·{fmt(ts.sy, 4)}] / [{r.n}·{fmt(ts.sxx, 4)} − {fmt(ts.sx, 4)}²] = <strong>{fmt(ts.b, 8)}</strong></p>
          <p className="font-mono text-sm">a = [ΣY − bΣX] / n = <strong>{fmt(ts.a, 8)}</strong></p>
          <Callout kind={Math.max(Math.abs(ts.b - r.b_sklearn), Math.abs(ts.a - r.a_sklearn)) < 1e-8 ? 'ok' : 'warn'} title="Verifikasi (TypeScript vs LinearRegression)">
            b = {fmt(r.b_sklearn, 8)}, a = {fmt(r.a_sklearn, 8)} · selisih maks {Math.max(Math.abs(ts.b - r.b_sklearn), Math.abs(ts.a - r.a_sklearn)).toExponential(2)} ✓
          </Callout>
        </Section>
      )}
      {mm.data && tm && (
        <Section title="Regresi berganda: b = (XᵀX)⁻¹XᵀY (12 observasi)">
          <Table caption="(XᵀX)⁻¹ (dari Python)" head={['', ...mm.data.names]} numeric={[1, 2, 3, 4]} rows={mm.data.XtXi.map((row: number[], i: number) => [mm.data.names[i], ...row.map((v) => fmt(v, 5))])} />
          <Table head={['Parameter', 'XᵀY', 'b (TypeScript, eliminasi Gauss)', 'b (Python inv)', 'b LinearRegression']} numeric={[1, 2, 3, 4]}
            rows={mm.data.names.map((n: string, i: number) => [n, fmt(mm.data.Xty[i], 4), fmt(tm.beta[i], 8), fmt(mm.data.b_manual[i], 8), fmt(mm.data.b_sklearn[i], 8)])} />
          <Callout kind="ok" title="Verifikasi">Selisih maks (XᵀX)⁻¹XᵀY vs LinearRegression = {mm.data.max_abs_diff.toExponential(2)}; TypeScript vs LinearRegression = {Math.max(...tm.beta.map((b, i) => Math.abs(b - mm.data.b_sklearn[i]))).toExponential(2)}.</Callout>
        </Section>
      )}
      {(s.loading || mm.loading) && <Spinner />}
    </div>
  );
}

export default function Tugas03() {
  return <ModulePage m={m} report={<Report />} lab={<Lab />} manual={<Manual />} data={<DataTab m={m}>
    <Callout kind="info" title="Mengapa operating_income dibuang?">operating_income = pendapatan × margin operasi (selisih maksimum hanya galat pembulatan 0,08 miliar USD). Memasukkannya sebagai prediktor membocorkan target dan membuat R² menyesatkan.</Callout>
  </DataTab>} />;
}
