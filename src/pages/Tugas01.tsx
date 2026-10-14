import { useEffect, useMemo, useState } from 'react';
import DataSource from '../components/DataSource';
import ModulePage, { DataTab } from '../components/ModulePage';
import Plot from '../components/Plot';
import { Callout, ErrorBox, LiveBadge, ReportBadge, Section, Select, Spinner, Stat, StatGrid, Table, Toggle } from '../components/ui';
import { bySlug } from '../content/modules';
import { fmt, fmtFixed, int } from '../lib/format';
import { py } from '../lib/pyodide/client';
import { palette } from '../lib/theme';
import { useAsync, useJson } from '../lib/useAsync';

const m = bySlug('tugas-01')!;
const FEATS = ['open', 'high', 'low', 'volume', 'sma_50', 'rsi_14'];
const ERA_COLORS = ['#999999', '#56B4E9', '#E69F00', '#009E73', '#CC79A7'];

function Report() {
  const { data: r } = useJson<any>('data/tugas-01.json');
  if (!r) return <Spinner text="Memuat hasil laporan …" />;
  const coef = Object.entries(r.koefisien).filter(([k]) => k !== 'sumber') as [string, number][];
  return (
    <div className="space-y-5">
      <ReportBadge source={r.sumber} />
      <Section title="Kinerja model (data uji 1.416 baris)" right={<span className="text-xs muted">{r.metrik.sumber}</span>}>
        <StatGrid>
          <Stat label="MAE" value={fmtFixed(r.metrik.MAE, 4)} hint={`penuh: ${r.metrik.MAE_penuh}`} />
          <Stat label="RMSE" value={fmtFixed(r.metrik.RMSE, 4)} />
          <Stat label="R²" value={fmtFixed(r.metrik.R2, 6)} />
          <Stat label="Latih / Uji" value={`${int(r.split.latih)} / ${int(r.split.uji)}`} hint={r.split.metode} />
        </StatGrid>
      </Section>
      <Section title="Persamaan regresi" right={<span className="text-xs muted">{r.koefisien.sumber}</span>}>
        <p className="font-mono text-sm">Close′ = −0,0000818 + 0,006963·Open + 0,475677·High + 0,512563·Low − 1,6×10⁻¹²·Volume + 0,005132·SMA50 + 0,0000039·RSI14</p>
        <Plot height={300} data={[{ type: 'bar', x: coef.map((c) => c[0]), y: coef.map((c) => c[1]), marker: { color: palette().brand2 } }]}
          layout={{ yaxis: { title: { text: 'Koefisien' } } }} ariaLabel="Diagram batang koefisien regresi" />
      </Section>
      <div className="grid gap-5 md:grid-cols-2">
        <Section title="Korelasi Pearson terhadap Close" right={<span className="text-xs muted">{r.korelasi_close.sumber}</span>}>
          <Table head={['Variabel', 'r']} numeric={[1]} rows={Object.entries(r.korelasi_close).filter(([k]) => k !== 'sumber').map(([k, v]) => [k, fmtFixed(v as number, 6)])} />
        </Section>
        <Section title="Statistik deskriptif" right={<span className="text-xs muted">{r.statistik_sumber}</span>}>
          <Table head={['Variabel', 'Rata-rata', 'SD', 'Min', 'Maks']} numeric={[1, 2, 3, 4]} rows={r.statistik} />
        </Section>
      </div>
    </div>
  );
}

function Lab() {
  const [key, setKey] = useState<string | null>(null);
  const [mode, setMode] = useState<'same_day' | 'lag'>('same_day');
  const [split, setSplit] = useState<'random' | 'temporal'>('random');
  const [feats, setFeats] = useState<string[]>(FEATS);
  const run = useAsync<any>();
  const cmp = useAsync<any[]>();
  useEffect(() => {
    if (key) run.run(() => py('dmx.t01', 'run', { data_key: key, mode, split, features: feats }));
  }, [key, mode, split, feats, run.run]);
  useEffect(() => {
    if (!key) return;
    cmp.run(() => Promise.all(([['same_day', 'random'], ['same_day', 'temporal'], ['lag', 'random'], ['lag', 'temporal']] as const)
      .map(([mo, sp]) => py('dmx.t01', 'run', { data_key: key, mode: mo, split: sp }).then((r) => ({ mo, sp, r })))) as any);
  }, [key, cmp.run]);
  const r = run.data;
  const p = palette();
  const series = useMemo(() => {
    if (!r) return [];
    const tr: any[] = [{ type: 'scatter', mode: 'lines', x: r.series.date, y: r.series.close, name: 'Close', line: { color: p.text, width: 1.2 } }];
    if (r.events) tr.push({ type: 'scatter', mode: 'markers', x: r.events.date, y: r.events.close, text: r.events.text, name: 'key_event', marker: { color: p.gold, size: 6, symbol: 'diamond' }, hovertemplate: '%{x}<br>%{text}<extra></extra>' });
    return tr;
  }, [r, p.text, p.gold]);
  const shapes = useMemo(() => (r?.eras ?? []).map((e: any, i: number) => ({ type: 'rect', xref: 'x', yref: 'paper', x0: e.start, x1: e.end, y0: 0, y1: 1, fillcolor: ERA_COLORS[i % 5], opacity: 0.13, line: { width: 0 } })), [r]);
  return (
    <div className="space-y-5">
      <DataSource dataKey="t01" sample={m.sample} onReady={(k) => setKey(k)} uploadHint="Unggah CSV dengan kolom date, open, high, low, close, volume, sma_50, rsi_14." />
      {key && (
        <Section title="Uji Realisme" right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-3">
            <Select label="Fitur" value={mode} onChange={setMode} options={[{ value: 'same_day', label: '(a) Hari yang sama — model laporan' }, { value: 'lag', label: '(b) Lag OHLC t…t−4 → Close t+1' }]} />
            <Select label="Pembagian data" value={split} onChange={setSplit} options={[{ value: 'random', label: 'Acak 80:20 (laporan)' }, { value: 'temporal', label: 'Berbasis waktu 80:20' }]} />
            {mode === 'same_day' && (
              <fieldset><legend className="label">Pilih fitur (latih ulang)</legend>
                <div className="grid grid-cols-2 gap-1">{FEATS.map((f) => <Toggle key={f} label={f} checked={feats.includes(f)} onChange={(c) => setFeats(c ? [...feats, f] : feats.filter((x) => x !== f).length ? feats.filter((x) => x !== f) : feats)} />)}</div>
              </fieldset>
            )}
          </div>
          {mode === 'same_day' && split === 'random' && <Callout kind="warn" title="Mode laporan">Fitur Open/High/Low hari yang sama + split acak: estimasi dalam-hari, bukan peramalan.</Callout>}
          <ErrorBox error={run.error} />
          {run.loading && <Spinner />}
          {r && (
            <>
              <StatGrid>
                <Stat label="MAE" value={fmt(r.metrics.MAE, 5)} />
                <Stat label="RMSE" value={fmt(r.metrics.RMSE, 5)} />
                <Stat label="R² (level harga)" value={fmtFixed(r.metrics.R2, 6)} />
                <Stat label="Latih / Uji" value={`${int(r.n_train)} / ${int(r.n_test)}`} />
                {r.baseline_persistence && <Stat label="Baseline naif MAE" value={fmt(r.baseline_persistence.MAE, 5)} hint="Close besok = Close hari ini" />}
                {r.change_metrics && <Stat label="R² perubahan harga" value={fmtFixed(r.change_metrics.r2_change, 4)} hint="ukuran peramalan yang jujur" />}
                {r.change_metrics && <Stat label="Skill vs naif" value={fmtFixed(r.change_metrics.skill_vs_persistence, 4)} hint="1 − MSE/MSE naif" />}
                {r.change_metrics && <Stat label="Akurasi arah" value={fmtFixed(r.change_metrics.direction_accuracy, 4)} />}
              </StatGrid>
              <Plot title="Harga Close NVIDIA dengan pita era & key_event" data={series} layout={{ shapes, yaxis: { type: 'log', title: { text: 'Close (USD, log)' } } }} height={360} />
              <div className="grid gap-5 lg:grid-cols-2">
                <Plot title="Aktual vs prediksi (data uji)" height={340} data={[
                  { type: 'scatter', mode: 'markers', x: r.scatter.actual, y: r.scatter.pred, text: r.scatter.date, marker: { size: 4, color: p.brand2, opacity: 0.6 }, name: 'Uji' },
                  { type: 'scatter', mode: 'lines', x: [Math.min(...r.scatter.actual), Math.max(...r.scatter.actual)], y: [Math.min(...r.scatter.actual), Math.max(...r.scatter.actual)], line: { color: p.gold, dash: 'dash' }, name: 'Ideal' }]}
                  layout={{ xaxis: { title: { text: 'Aktual' } }, yaxis: { title: { text: 'Prediksi' } } }} />
                <Plot title="Koefisien" height={340} data={[{ type: 'bar', orientation: 'h', y: Object.keys(r.coef), x: Object.values(r.coef), marker: { color: p.brand2 } }]} layout={{ margin: { l: 110 } }} />
              </div>
              <Plot title="Heatmap korelasi" height={360} data={[{ type: 'heatmap', z: r.corr.values, x: r.corr.cols, y: r.corr.cols, zmin: -1, zmax: 1, colorscale: 'RdBu', reversescale: true, text: r.corr.values.map((row: number[]) => row.map((v) => v.toFixed(3))), texttemplate: '%{text}' }]} />
            </>
          )}
          {cmp.data && (
            <Table caption="Ringkasan empat kombinasi (Hasil Live)" head={['Fitur', 'Split', 'R² level', 'MAE', 'R² perubahan', 'MAE naif']} numeric={[2, 3, 4, 5]}
              rows={cmp.data.map(({ mo, sp, r: x }: any) => [mo === 'lag' ? 'Lag t…t−4' : 'Hari sama', sp === 'random' ? 'Acak' : 'Waktu', fmtFixed(x.metrics.R2, 6), fmt(x.metrics.MAE, 5), x.change_metrics ? fmtFixed(x.change_metrics.r2_change, 4) : '–', x.baseline_persistence ? fmt(x.baseline_persistence.MAE, 5) : '–'])} />
          )}
          <Callout kind="info" title="Cara membaca">R² level harga tetap tinggi pada mode lag karena harga saham sangat berautokorelasi (harga besok ≈ harga hari ini). Ukuran yang jujur untuk peramalan adalah R² perubahan harga dan perbandingan dengan baseline naif — di sana keunggulan model menyusut drastis.</Callout>
        </Section>
      )}
    </div>
  );
}

function Manual() {
  const [key, setKey] = useState<string | null>(null);
  const [n, setN] = useState(20);
  const s = useAsync<any>();
  useEffect(() => { if (key) s.run(() => py('dmx.t01', 'manual_ols', { data_key: key, n_rows: n })); }, [key, n, s.run]);
  const r = s.data;
  return (
    <div className="space-y-5">
      <DataSource dataKey="t01" sample={m.sample} onReady={(k) => setKey(k)} />
      <Section title="OLS manual: b = (XᵀX)⁻¹Xᵀy pada subset kecil" right={<LiveBadge />}>
        <Select label="Jumlah baris" value={n} onChange={setN} options={[10, 20, 40].map((v) => ({ value: v, label: `${v} hari pertama` }))} />
        {s.loading && <Spinner />}
        <ErrorBox error={s.error} />
        {r && (
          <>
            <p className="text-sm">Langkah 1 — susun X (kolom 1 untuk intersep; Volume dalam juta lembar) dan y = Close. Langkah 2 — hitung XᵀX dan Xᵀy. Langkah 3 — selesaikan sistem persamaan normal.</p>
            <Table head={['', ...r.features]} numeric={r.features.map((_: string, i: number) => i + 1)} caption="Matriks XᵀX" rows={r.XtX.map((row: number[], i: number) => [r.features[i], ...row.map((v) => fmt(v, 4))])} />
            <Table head={['Parameter', 'Xᵀy', 'b manual', 'b LinearRegression', '|selisih|']} numeric={[1, 2, 3, 4]}
              rows={r.features.map((f: string, i: number) => [f, fmt(r.Xty[i], 4), fmt(r.beta_manual[i], 8), fmt(r.beta_sklearn[i], 8), fmt(Math.abs(r.beta_manual[i] - r.beta_sklearn[i]), 3)])} />
            <Callout kind={r.max_abs_diff_pred < 1e-8 ? 'ok' : 'warn'} title="Verifikasi">
              Selisih maksimum prediksi manual vs LinearRegression = {r.max_abs_diff_pred.toExponential(2)} (syarat &lt; 1×10⁻⁸) {r.max_abs_diff_pred < 1e-8 ? '✓ identik' : '✗'}. Bilangan kondisi XᵀX = {r.cond_XtX.toExponential(2)}.
            </Callout>
          </>
        )}
      </Section>
    </div>
  );
}

export default function Tugas01() {
  return <ModulePage m={m} report={<Report />} lab={<Lab />} manual={<Manual />} data={<DataTab m={m}>
    <Section title="Kualitas data (laporan)"><Table head={['Aspek', 'Temuan']} rows={[['Ukuran', '7.079 × 16'], ['Missing variabel numerik utama', '0'], ['Missing stock_split', '7.073 (struktural)'], ['Missing key_event', '6.826 (struktural)'], ['Duplikat', '0']]} /></Section>
  </DataTab>} />;
}
