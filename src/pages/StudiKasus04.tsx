import { useEffect, useState } from 'react';
import DataSource from '../components/DataSource';
import ModulePage from '../components/ModulePage';
import Plot from '../components/Plot';
import { Callout, ErrorBox, LiveBadge, ReportBadge, Section, Select, Slider, Spinner, Stat, StatGrid, Table, Toggle } from '../components/ui';
import { bySlug } from '../content/modules';
import { fmt, fmtFixed, int, pct } from '../lib/format';
import { logisticProba, sigmoid } from '../lib/manual/logistic';
import { py } from '../lib/pyodide/client';
import { palette } from '../lib/theme';
import { useAsync, useJson } from '../lib/useAsync';

const m = bySlug('studi-kasus-04')!;
const DEMO = { url: 'data/samples/t01_nvidia_stock_1999_2026.csv', label: 'DEMO PENGGANTI: OHLC NVIDIA (CC0) — bukan dataset laporan', license: 'CC0' };
const pv = (v: number) => (v < 0.0001 ? '< 0,0001' : fmtFixed(v, 4));

function Confusion({ cm, labels = ['Turun/Tetap (0)', 'Naik (1)'] }: { cm: number[][]; labels?: string[] }) {
  const max = Math.max(...cm.flat());
  const names = [['TN', 'FP'], ['FN', 'TP']];
  return (
    <div className="overflow-x-auto">
      <table className="tbl max-w-md text-center">
        <thead><tr><th>Aktual \ Prediksi</th>{labels.map((l) => <th key={l} className="text-center">{l}</th>)}</tr></thead>
        <tbody>{cm.map((row, i) => (
          <tr key={i}><th scope="row">{labels[i]}</th>{row.map((v, j) => (
            <td key={j} style={{ background: `color-mix(in srgb, var(--brand-2) ${Math.round((v / max) * 30)}%, var(--surface))` }}>
              <div className="text-lg font-bold">{int(v)}</div><div className="font-mono text-xs">{names[i][j]}</div>
            </td>))}</tr>))}</tbody>
      </table>
    </div>
  );
}

function Report() {
  const { data: r } = useJson<any>('data/studi-kasus-04.json');
  if (!r) return <Spinner text="Memuat hasil laporan …" />;
  const p = palette();
  const lt = [...r.lintas_saham].sort((a: any, b: any) => a[3] - b[3]);
  return (
    <div className="space-y-5">
      <ReportBadge source={r.sumber} />
      <Section title="Kinerja data uji AAPL (837 hari)" right={<span className="text-xs muted">{r.metrik.sumber}</span>}>
        <StatGrid>
          <Stat label="Akurasi" value={pct(r.metrik.akurasi)} /><Stat label="Balanced acc." value={pct(r.metrik.balanced_accuracy)} />
          <Stat label="Presisi" value={pct(r.metrik.presisi)} /><Stat label="Recall" value={pct(r.metrik.recall)} />
          <Stat label="F1" value={pct(r.metrik.f1)} /><Stat label="ROC-AUC" value={fmtFixed(r.metrik.roc_auc, 3)} hint={`CI bootstrap [${fmtFixed(r.bootstrap.lo, 3)}; ${fmtFixed(r.bootstrap.hi, 3)}]`} />
          <Stat label="MCC" value={fmtFixed(r.metrik.mcc, 4)} /><Stat label="Log-loss / Brier" value={`${fmtFixed(r.metrik.log_loss, 4)} / ${fmtFixed(r.brier, 4)}`} />
        </StatGrid>
        <Confusion cm={[[r.confusion.TN, r.confusion.FP], [r.confusion.FN, r.confusion.TP]]} />
      </Section>
      <Section title="Inferensi MLE tanpa regularisasi (IRLS)" right={<span className="text-xs muted">{r.mle_sumber}; {r.lr_test.sumber}</span>}>
        <Table head={['Fitur', 'β', 'SE', 'z', 'p-value', 'Odds ratio [CI 95%]']} numeric={[1, 2, 3, 4]} rows={r.mle.map((x: any[]) => [x[0], fmtFixed(x[1], 4), fmtFixed(x[2], 4), fmtFixed(x[3], 3), fmtFixed(x[4], 4), `${fmtFixed(x[5], 3)} [${fmtFixed(x[6], 3)}–${fmtFixed(x[7], 3)}]`])} />
        <p className="text-sm">LR = {fmtFixed(r.lr_test.LR, 3)} (df {r.lr_test.df}), p = {fmtFixed(r.lr_test.p, 4)}; McFadden R² = {fmtFixed(r.lr_test.mcfadden, 4)}. Tidak ada fitur yang signifikan (semua p &gt; 0,14).</p>
      </Section>
      <div className="grid gap-5 lg:grid-cols-2">
        <Section title="Walk-forward 5 lipatan" right={<span className="text-xs muted">{r.walk_forward_ringkas.sumber}</span>}>
          <Table head={['Lipatan', 'N latih', 'N uji', 'Akurasi', 'ROC-AUC', 'F1']} numeric={[1, 2, 3, 4, 5]} rows={r.walk_forward} />
          <p className="text-sm">Rata-rata AUC {fmtFixed(r.walk_forward_ringkas.mean_auc, 4)} ± {fmtFixed(r.walk_forward_ringkas.sd, 4)}; C optimal {r.tuning.C_optimal} (AUC CV {fmtFixed(r.tuning.auc_cv, 4)}).</p>
        </Section>
        <Section title="VIF" right={<span className="text-xs muted">{r.vif.sumber}</span>}>
          <Plot height={300} data={[{ type: 'bar', orientation: 'h', y: Object.keys(r.vif).filter((k) => k !== 'sumber'), x: Object.entries(r.vif).filter(([k]) => k !== 'sumber').map(([, v]) => v), marker: { color: Object.entries(r.vif).filter(([k]) => k !== 'sumber').map(([, v]) => ((v as number) > 5 ? p.gold : p.brand2)) } }]} layout={{ margin: { l: 110 }, shapes: [5, 10].map((x) => ({ type: 'line', x0: x, x1: x, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: x === 10 ? p.down : p.gold } })) }} />
        </Section>
      </div>
      <Section title="Uji generalisasi lintas 20 emiten" right={<span className="text-xs muted">{r.lintas_saham_ringkas.sumber}</span>}>
        <Plot height={420} data={[{ type: 'bar', orientation: 'h', y: lt.map((x: any) => x[0]), x: lt.map((x: any) => x[3]), marker: { color: lt.map((x: any) => (x[3] > 0.5 ? p.up : p.neutral)) }, text: lt.map((x: any) => fmtFixed(x[3], 4)), textposition: 'outside' }]} layout={{ xaxis: { range: [0.45, 0.58], title: { text: 'ROC-AUC' } }, shapes: [{ type: 'line', x0: 0.5, x1: 0.5, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: p.text } }] }} />
        <p className="text-sm">Rata-rata AUC {fmtFixed(r.lintas_saham_ringkas.mean_auc, 4)} (SD {fmtFixed(r.lintas_saham_ringkas.sd, 4)}), {r.lintas_saham_ringkas.n_di_atas_05}/20 di atas 0,5; uji-t p = {fmtFixed(r.lintas_saham_ringkas.p_t, 4)}, Wilcoxon p = {fmtFixed(r.lintas_saham_ringkas.p_wilcoxon, 4)}. Signifikan secara statistik, namun ≈1,6 poin AUC hampir pasti habis oleh biaya transaksi.</p>
      </Section>
      <Callout kind="warn" title="Jebakan threshold">“F1 optimal” {fmtFixed(r.threshold.f1, 4)} pada threshold {fmtFixed(r.threshold.threshold, 2)} (recall {fmtFixed(r.threshold.recall, 0)}, presisi {fmtFixed(r.threshold.presisi, 3)}) hanyalah strategi “selalu prediksi naik” — {r.threshold.sumber}.</Callout>
    </div>
  );
}

function Lab() {
  const [key, setKey] = useState<string | null>(null);
  const [multi, setMulti] = useState<string[]>([]);
  const [penalty, setPenalty] = useState<'l2' | 'none'>('l2');
  const [C, setC] = useState(1);
  const [thr, setThr] = useState(0.5);
  const [dropLast, setDropLast] = useState(false);
  const main = useAsync<any>();
  const inf = useAsync<any>();
  const wf = useAsync<any>();
  const tune = useAsync<any>();
  const boot = useAsync<any>();
  const perm = useAsync<any>();
  const cross = useAsync<any>();
  const opts = { drop_last_unknown: dropLast };
  useEffect(() => { if (key) main.run(() => py('dmx.t04', 'run_main', { data_key: key, penalty, C, threshold: thr, ...opts })); }, [key, penalty, C, thr, dropLast, main.run]);
  useEffect(() => { if (key) { inf.run(() => py('dmx.t04', 'inference', { data_key: key, ...opts })); wf.run(() => py('dmx.t04', 'walk_forward', { data_key: key, ...opts })); } }, [key, dropLast, inf.run, wf.run]);
  const r = main.data;
  const p = palette();
  return (
    <div className="space-y-5">
      <Callout kind="info" title="Dataset NASDAQ tidak disertakan">Lisensi “Data files © Original Authors”. Unggah satu atau beberapa CSV per ticker (kolom ticker, date, open, high, low, close) dari dataset Kaggle svaningelgem/nasdaq-daily-stock-prices — mis. AAPL.csv untuk mereplikasi laporan. Tombol dataset contoh memuat DEMO pengganti OHLC NVIDIA (CC0) agar lab tetap dapat dicoba.</Callout>
      <DataSource dataKey="t04" sample={DEMO} onReady={setKey} multiple onMulti={setMulti} />
      {key && (
        <Section title="Model utama" right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-4">
            <Select label="Regularisasi" value={penalty} onChange={setPenalty} options={[{ value: 'l2', label: 'L2 (laporan)' }, { value: 'none', label: 'Tanpa regularisasi' }]} />
            <Slider label="log₁₀ C" value={Math.log10(C)} min={-3} max={2} step={0.25} onChange={(v) => setC(10 ** v)} format={(v) => `${fmt(v, 2)} (C = ${fmt(10 ** v, 4)})`} />
            <Slider label="Threshold" value={thr} min={0.05} max={0.95} step={0.025} onChange={setThr} format={(v) => fmtFixed(v, 3)} />
            <Toggle label="Buang baris terakhir (target H+1 tak diketahui)" checked={dropLast} onChange={setDropLast} />
          </div>
          <ErrorBox error={main.error} />
          {main.loading && !r && <Spinner />}
          {r && (<>
            <p className="text-sm muted">{r.ticker ?? 'Data'} · {int(r.n_total)} observasi · latih {r.train_period.join(' – ')} ({int(r.n_train)}) · uji {r.test_period.join(' – ')} ({int(r.n_test)}) · split berbasis waktu; scaler di-fit pada data latih.</p>
            <StatGrid>
              <Stat label="Akurasi" value={pct(r.metrics.accuracy)} /><Stat label="Balanced acc." value={pct(r.metrics.balanced_accuracy)} />
              <Stat label="Presisi" value={pct(r.metrics.precision)} /><Stat label="Recall" value={pct(r.metrics.recall)} />
              <Stat label="F1" value={pct(r.metrics.f1)} /><Stat label="ROC-AUC" value={fmtFixed(r.metrics.roc_auc, 4)} />
              <Stat label="MCC" value={fmtFixed(r.metrics.mcc, 4)} /><Stat label="Brier" value={fmtFixed(r.metrics.brier, 4)} />
            </StatGrid>
            <div className="grid gap-5 lg:grid-cols-2">
              <div><h3 className="mb-2 font-semibold">Confusion matrix @ threshold {fmtFixed(thr, 3)}</h3><Confusion cm={r.metrics.confusion_matrix} /></div>
              <Plot title="Presisi, recall, F1 vs threshold" height={300} data={[
                { type: 'scatter', mode: 'lines', x: r.threshold_sweep.map((s: any) => s.threshold), y: r.threshold_sweep.map((s: any) => s.precision), name: 'Presisi', line: { color: p.text } },
                { type: 'scatter', mode: 'lines', x: r.threshold_sweep.map((s: any) => s.threshold), y: r.threshold_sweep.map((s: any) => s.recall), name: 'Recall', line: { color: p.up } },
                { type: 'scatter', mode: 'lines', x: r.threshold_sweep.map((s: any) => s.threshold), y: r.threshold_sweep.map((s: any) => s.f1), name: 'F1', line: { color: p.down, width: 3 } },
                { type: 'scatter', mode: 'lines', x: r.threshold_sweep.map((s: any) => s.threshold), y: r.threshold_sweep.map((s: any) => s.pct_pred_up), name: '% diprediksi naik', line: { color: p.gold, dash: 'dot' } }]}
                layout={{ shapes: [{ type: 'line', x0: thr, x1: thr, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: p.muted } }] }} />
            </div>
            {r.f1_best.threshold < 0.2 && <Callout kind="warn" title="F1 optimal semu">F1 maksimum {fmtFixed(r.f1_best.f1, 4)} terjadi pada threshold {fmtFixed(r.f1_best.threshold, 2)} dengan {pct(r.f1_best.pct_pred_up, 1)} hari diprediksi “naik” — setara strategi selalu-naik tanpa model.</Callout>}
            <div className="grid gap-5 lg:grid-cols-3">
              <Plot title="Kurva ROC" height={300} data={[{ type: 'scatter', mode: 'lines', x: r.roc.fpr, y: r.roc.tpr, name: `AUC ${fmtFixed(r.metrics.roc_auc, 3)}`, line: { color: p.brand2 } }, { type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], line: { dash: 'dash', color: p.neutral }, name: 'Acak' }]} layout={{ xaxis: { title: { text: 'FPR' } }, yaxis: { title: { text: 'TPR' } } }} />
              <Plot title="Kurva Precision–Recall" height={300} data={[{ type: 'scatter', mode: 'lines', x: r.roc.recall, y: r.roc.precision, line: { color: p.brand2 } }]} layout={{ showlegend: false, xaxis: { title: { text: 'Recall' } }, yaxis: { title: { text: 'Precision' } } }} />
              <Plot title="Kalibrasi (reliability)" height={300} data={[{ type: 'scatter', mode: 'lines+markers', x: r.calibration.mean_pred, y: r.calibration.frac_pos, line: { color: p.brand2 }, name: 'Model' }, { type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], line: { dash: 'dash', color: p.neutral }, name: 'Sempurna' }]} layout={{ xaxis: { range: [0.3, 0.7] }, yaxis: { range: [0.3, 0.7] } }} />
            </div>
            <div className="grid gap-5 lg:grid-cols-2">
              <Plot title="Sigmoid: log-odds → P(naik) dengan threshold" height={300} data={[
                { type: 'scatter', mode: 'lines', x: Array.from({ length: 121 }, (_, i) => -6 + i * 0.1), y: Array.from({ length: 121 }, (_, i) => sigmoid(-6 + i * 0.1)), line: { color: p.brand2, width: 3 }, name: 'σ(z)' }]}
                layout={{ showlegend: false, xaxis: { title: { text: 'z = log-odds' } }, yaxis: { title: { text: 'P(naik)' } }, shapes: [{ type: 'line', x0: -6, x1: 6, y0: thr, y1: thr, line: { dash: 'dash', color: p.gold } }, { type: 'line', x0: Math.log(thr / (1 - thr)), x1: Math.log(thr / (1 - thr)), y0: 0, y1: 1, line: { dash: 'dot', color: p.gold } }] }} />
              <Plot title="Distribusi P(naik) per kelas aktual" height={300} data={[
                { type: 'bar', x: r.proba_hist.up.edges.slice(0, -1), y: r.proba_hist.up.counts, name: 'Aktual naik', marker: { color: p.up }, opacity: 0.7 },
                { type: 'bar', x: r.proba_hist.down.edges.slice(0, -1), y: r.proba_hist.down.counts, name: 'Aktual turun/tetap', marker: { color: p.down }, opacity: 0.7 }]} layout={{ barmode: 'overlay' }} />
            </div>
          </>)}
        </Section>
      )}
      {inf.data && (
        <Section title="Koefisien, odds ratio & p-value (MLE/IRLS, tanpa regularisasi)">
          <Table head={['Fitur', 'β', 'SE', 'z', 'p', 'OR [CI 95%]', 'VIF']} numeric={[1, 2, 3, 4, 6]} rows={inf.data.table.map((x: any) => [x.feature, fmt(x.coef, 4), fmt(x.se, 4), fmt(x.z, 3), pv(x.p_value), `${fmtFixed(x.odds_ratio, 3)} [${fmtFixed(x.or_low, 3)}–${fmtFixed(x.or_high, 3)}]`, x.feature === 'intercept' ? '' : fmtFixed(inf.data.vif[x.feature], 2)])} />
          <p className="text-sm">LR-test = {fmtFixed(inf.data.lr_stat, 3)} (df {inf.data.lr_df}), p = {pv(inf.data.lr_p)}; McFadden R² = {fmtFixed(inf.data.mcfadden_r2, 4)}; IRLS konvergen dalam {inf.data.irls_trace.length} iterasi.</p>
        </Section>
      )}
      {wf.data && (
        <Section title="Validasi walk-forward (expanding window)">
          <Plot height={280} data={[{ type: 'scatter', mode: 'lines+markers', x: wf.data.folds.map((f: any) => `F${f.fold} (${f.start.slice(0, 7)})`), y: wf.data.folds.map((f: any) => f.roc_auc), name: 'ROC-AUC', line: { color: p.brand2 } }, { type: 'scatter', mode: 'lines+markers', x: wf.data.folds.map((f: any) => `F${f.fold} (${f.start.slice(0, 7)})`), y: wf.data.folds.map((f: any) => f.accuracy), name: 'Akurasi', line: { color: p.text } }]} layout={{ shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { dash: 'dash', color: p.neutral } }] }} />
          <p className="text-sm">Rata-rata AUC {fmtFixed(wf.data.mean_auc, 4)} ± {fmtFixed(wf.data.std_auc, 4)}.</p>
        </Section>
      )}
      {key && (
        <Section title="Analisis lanjutan (klik untuk menjalankan)">
          <div className="flex flex-wrap gap-2">
            <button className="btn-ghost" disabled={tune.loading} onClick={() => tune.run(() => py('dmx.t04', 'tune_C', { data_key: key, ...opts }))}>Tuning C (TimeSeriesSplit)</button>
            <button className="btn-ghost" disabled={boot.loading} onClick={() => boot.run(() => py('dmx.t04', 'bootstrap_auc', { data_key: key, n_boot: 1000, penalty, C, ...opts }))}>Bootstrap CI AUC (1.000×)</button>
            <button className="btn-ghost" disabled={perm.loading} onClick={() => perm.run(() => py('dmx.t04', 'permutation', { data_key: key, n_repeats: 10, ...opts }))}>Permutation importance</button>
          </div>
          {(tune.loading || boot.loading || perm.loading) && <Spinner />}
          <div className="grid gap-5 lg:grid-cols-3">
            {tune.data && <Plot title={`Kurva validasi C (terbaik ${fmt(tune.data.best_C, 4)})`} height={280} data={[{ type: 'scatter', mode: 'lines+markers', x: tune.data.C, y: tune.data.mean, error_y: { type: 'data', array: tune.data.std, visible: true }, line: { color: p.brand2 } }]} layout={{ xaxis: { type: 'log', title: { text: 'C' } }, yaxis: { title: { text: 'AUC CV' } } }} />}
            {boot.data && <Plot title={`Bootstrap AUC: [${fmtFixed(boot.data.lo, 3)}; ${fmtFixed(boot.data.hi, 3)}]`} height={280} data={[{ type: 'bar', x: boot.data.hist.edges.slice(0, -1), y: boot.data.hist.counts, marker: { color: p.brand2 } }]} layout={{ shapes: [0.5, boot.data.lo, boot.data.hi].map((x, i) => ({ type: 'line', x0: x, x1: x, y0: 0, y1: 1, yref: 'paper', line: { dash: i ? 'dot' : 'dash', color: i ? p.gold : p.text } })) }} />}
            {perm.data && <Plot title="Permutation importance (ΔAUC)" height={280} data={[{ type: 'bar', orientation: 'h', y: perm.data.features, x: perm.data.mean, error_x: { type: 'data', array: perm.data.std }, marker: { color: p.brand2 } }]} layout={{ margin: { l: 110 } }} />}
          </div>
        </Section>
      )}
      {multi.length > 1 && (
        <Section title={`Uji lintas-saham (${multi.length} berkas)`}>
          <button className="btn-primary" disabled={cross.loading} onClick={() => cross.run(() => py('dmx.t04', 'cross_stock', { data_keys: multi }))}>Jalankan pipeline identik pada semua berkas</button>
          {cross.loading && <Spinner />}
          {cross.data && (<>
            <Table head={['Ticker', 'N obs', 'Akurasi', 'ROC-AUC', 'F1']} numeric={[1, 2, 3, 4]} rows={cross.data.rows.filter((x: any) => x.roc_auc).sort((a: any, b: any) => b.roc_auc - a.roc_auc).map((x: any) => [x.ticker, int(x.n_obs), fmtFixed(x.accuracy, 4), fmtFixed(x.roc_auc, 4), fmtFixed(x.f1, 4)])} />
            {cross.data.summary && <p className="text-sm">Rata-rata AUC {fmtFixed(cross.data.summary.mean_auc, 4)} (SD {fmtFixed(cross.data.summary.std_auc, 4)}); {cross.data.summary.n_above}/{cross.data.summary.n} di atas 0,5; uji-t p = {pv(cross.data.summary.p_value)}{cross.data.summary.wilcoxon_p !== undefined && `; Wilcoxon p = ${pv(cross.data.summary.wilcoxon_p)}`}.</p>}
          </>)}
        </Section>
      )}
      <Callout kind="warn" title="Peringatan">Hasil bukan saran investasi.</Callout>
    </div>
  );
}

function Manual() {
  const [key, setKey] = useState<string | null>(null);
  const [idx, setIdx] = useState(0);
  const [thr, setThr] = useState(0.5);
  const s = useAsync<any>();
  useEffect(() => { if (key) s.run(() => py('dmx.t04', 'manual_proba', { data_key: key, test_index: idx })); }, [key, idx, s.run]);
  const r = s.data;
  const ts = r ? logisticProba(r.rows.map((x: any) => x.x), r.rows.map((x: any) => x.mean), r.rows.map((x: any) => x.std), r.rows.map((x: any) => x.beta), r.intercept) : null;
  return (
    <div className="space-y-5">
      <DataSource dataKey="t04" sample={DEMO} onReady={setKey} />
      {r && ts && (
        <Section title={`Satu hari uji: ${r.date}`} right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-2">
            <Slider label="Indeks hari uji" value={idx} min={0} max={r.n_test - 1} onChange={setIdx} />
            <Slider label="Threshold" value={thr} min={0.05} max={0.95} step={0.05} onChange={setThr} format={(v) => fmtFixed(v, 2)} />
          </div>
          <Table head={['Fitur', 'x', 'μ latih', 'σ latih', 'z = (x−μ)/σ', 'β', 'β·z']} numeric={[1, 2, 3, 4, 5, 6]} rows={r.rows.map((x: any, j: number) => [x.feature, fmt(x.x, 5), fmt(x.mean, 5), fmt(x.std, 5), fmt(ts.z[j], 5), fmt(x.beta, 5), fmt(ts.terms[j], 6)])} />
          <p className="font-mono text-sm">z = b₀ + Σβⱼzⱼ = {fmt(r.intercept, 6)} + ({fmt(ts.terms.reduce((a, b) => a + b, 0), 6)}) = <strong>{fmt(ts.logit, 8)}</strong></p>
          <p className="font-mono text-sm">P(naik) = 1 / (1 + e^(−z)) = <strong>{fmtFixed(ts.proba, 8)}</strong> → keputusan @ {fmtFixed(thr, 2)}: <strong>{ts.proba >= thr ? 'NAIK (1)' : 'TURUN/TETAP (0)'}</strong> · aktual: {r.actual ? 'naik' : 'turun/tetap'}</p>
          <Callout kind={Math.abs(ts.proba - r.p_sklearn) < 1e-8 ? 'ok' : 'warn'} title="Verifikasi">predict_proba() = {fmtFixed(r.p_sklearn, 8)} · selisih TypeScript = {Math.abs(ts.proba - r.p_sklearn).toExponential(2)} (syarat &lt; 1×10⁻⁸)</Callout>
        </Section>
      )}
      {s.loading && <Spinner />}
    </div>
  );
}

export default function StudiKasus04() {
  return <ModulePage m={m} report={<Report />} lab={<Lab />} manual={<Manual />} />;
}
