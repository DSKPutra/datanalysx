import { useEffect, useMemo, useRef, useState } from 'react';
import DataSource from '../components/DataSource';
import ModulePage from '../components/ModulePage';
import Plot from '../components/Plot';
import { Callout, ErrorBox, LiveBadge, ReportBadge, Section, Select, Slider, Spinner, Stat, StatGrid, Table } from '../components/ui';
import { bySlug } from '../content/modules';
import { fmt, fmtFixed, int, pct } from '../lib/format';
import { lloyd } from '../lib/manual/kmeans';
import { py } from '../lib/pyodide/client';
import { palette } from '../lib/theme';
import { useAsync, useJson } from '../lib/useAsync';

const m = bySlug('studi-kasus-08')!;
const SYM = ['triangle-up', 'triangle-down', 'diamond', 'circle', 'square', 'cross', 'x', 'star', 'hexagon', 'pentagon'];

function SyntheticWarning() {
  return <Callout kind="warn" title="Dataset AAPL 2026 bersifat sintetis/semi-sintetis">Menurut penyusunnya data dibangkitkan/terinterpolasi; rezim yang ditemukan adalah latihan metodologis, bukan fakta pasar dan bukan saran investasi.</Callout>;
}

function Report() {
  const { data: r } = useJson<any>('data/studi-kasus-08.json');
  if (!r) return <Spinner text="Memuat hasil laporan …" />;
  const p = palette();
  const ks = Object.keys(r.k_selection.per_k);
  const pk = r.k_selection.per_k;
  const fm = r.final_model;
  const nin = r.init_convergence.n_init_effect['k-means++'];
  return (
    <div className="space-y-5">
      <ReportBadge source="model_summary.json studi kasus 08" />
      <SyntheticWarning />
      <Section title="Model final: StandardScaler → K-Means (K = 3, k-means++, n_init = 100)">
        <StatGrid>
          {(['C1', 'C2', 'C3'] as const).map((c) => <Stat key={c} label={`${c} ${fm.names[c]}`} value={`${fm.sizes[c]} hari`} hint={`return rata-rata ${fmt(fm.profile[c].centroid_original.log_ret, 2)}% · vol rel ${fmt(fm.profile[c].centroid_original.rel_vol, 2)}×`} />)}
          <Stat label="Silhouette" value={fmtFixed(fm.silhouette, 4)} /><Stat label="DBI" value={fmtFixed(fm.dbi, 4)} /><Stat label="CH" value={fmtFixed(fm.ch, 2)} /><Stat label="Inertia" value={fmtFixed(fm.inertia, 4)} />
        </StatGrid>
      </Section>
      <Section title="Pemilihan K (n_init = 50)">
        <div className="grid gap-4 md:grid-cols-2">
          <Plot title="Elbow (inertia)" height={250} data={[{ type: 'scatter', mode: 'lines+markers', x: ks, y: ks.map((k) => pk[k].inertia), line: { color: p.brand2 } }]} />
          <Plot title="Silhouette" height={250} data={[{ type: 'scatter', mode: 'lines+markers', x: ks, y: ks.map((k) => pk[k].silhouette), line: { color: p.brand2 } }]} />
        </div>
        <Table head={['K', 'Silhouette', 'DBI', 'CH', 'Gap', 'ARI stabilitas (n_init=1) rata-rata']} numeric={[1, 2, 3, 4, 5]} rows={ks.map((k) => [k, fmtFixed(pk[k].silhouette, 3), fmtFixed(pk[k].dbi, 3), fmtFixed(pk[k].ch, 2), fmtFixed(pk[k].gap, 3), fmtFixed(pk[k].init_stability_ari_mean, 3)])} />
        <p className="text-sm">K terbaik per metode: {Object.entries(r.k_selection.best_k_by_method).map(([a, b]) => `${a} → ${b}`).join(' · ')}. K = 4 ditolak: dua solusi berbeda isi (ARI {fmtFixed(r.k4_alternative.rival.ari_vs_best_k4, 2)}) dengan WCSS terpaut {fmtFixed(r.k4_alternative.rival.excess_inertia_pct, 2)}%.</p>
      </Section>
      <Section title="Kestabilan inisialisasi (k-means++, K = 3)">
        <Plot height={260} data={[{ type: 'bar', x: Object.keys(nin), y: Object.values(nin).map((v: any) => v.pct_reach_best), marker: { color: p.brand2 }, text: Object.values(nin).map((v: any) => `${fmt(v.pct_reach_best, 1)}%`), textposition: 'outside' }]} layout={{ xaxis: { title: { text: 'n_init' }, type: 'category' }, yaxis: { title: { text: '% mencapai solusi terbaik' }, range: [0, 110] } }} />
      </Section>
      <Section title="Rezim tidak memprediksi return esok hari">
        <p>Kruskal–Wallis H = {fmtFixed(r.next_day_return.kruskal_H, 3)}, p = {fmtFixed(r.next_day_return.kruskal_p, 3)}; silhouette periode uji turun {fmtFixed(r.temporal_validation.silhouette_train, 2)} → {fmtFixed(r.temporal_validation.silhouette_test_assigned, 2)}.</p>
      </Section>
    </div>
  );
}

function Lab() {
  const [key, setKey] = useState<string | null>(null);
  const [k, setK] = useState(3);
  const [init, setInit] = useState<'k-means++' | 'random'>('k-means++');
  const [nInit, setNInit] = useState(100);
  const km = useAsync<any>();
  const fit = useAsync<any>();
  const stab = useAsync<any>();
  const setup = useAsync<any>();
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [seed, setSeed] = useState(42);
  useEffect(() => { if (key) km.run(() => py('dmx.t08', 'k_metrics', { data_key: key, init, n_init: 20, gap_B: 10 })); }, [key, init, km.run]);
  useEffect(() => { if (key) fit.run(() => py('dmx.t08', 'fit', { data_key: key, k, init, n_init: nInit })); }, [key, k, init, nInit, fit.run]);
  useEffect(() => { if (key) { setup.run(() => py('dmx.t08', 'lloyd_setup', { data_key: key, k, init, seed })); setStep(0); setPlaying(false); } }, [key, k, init, seed, setup.run]);
  const p = palette();
  const f = fit.data;
  const anim = useMemo(() => (setup.data ? lloyd(setup.data.X, setup.data.init_centers) : null), [setup.data]);
  const project = (v: number[]) => setup.data.pca_components.map((c: number[]) => c.reduce((s: number, w: number, i: number) => s + w * (v[i] - setup.data.pca_mean[i]), 0));
  const frames = anim ? anim.iterations.flatMap((it, i) => [{ phase: 'assign', i, it }, { phase: 'update', i, it }]) : [];
  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => setStep((s) => { if (s >= frames.length - 1) { setPlaying(false); return s; } return s + 1; }), 900);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [playing, frames.length]);
  const fr = frames[Math.min(step, frames.length - 1)];
  const animOk = anim && setup.data ? anim.labels.every((l, i) => l === setup.data.sk_labels[i]) : null;
  return (
    <div className="space-y-5">
      <SyntheticWarning />
      <DataSource dataKey="t08" sample={m.sample} onReady={setKey} />
      {key && (
        <Section title="Atur K-Means" right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-3">
            <Slider label="K" value={k} min={2} max={10} onChange={setK} />
            <Select label="Inisialisasi" value={init} onChange={setInit} options={[{ value: 'k-means++', label: 'k-means++' }, { value: 'random', label: 'Acak' }]} />
            <Select label="n_init" value={nInit} onChange={setNInit} options={[1, 5, 10, 50, 100].map((v) => ({ value: v, label: String(v) }))} />
          </div>
          <ErrorBox error={fit.error || km.error} />
          {km.data && (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {([['inertia', 'Elbow (inertia)'], ['silhouette', 'Silhouette ↑'], ['dbi', 'Davies–Bouldin ↓'], ['ch', 'Calinski–Harabasz ↑']] as const).map(([key2, t]) => (
                <Plot key={key2} title={t} height={220} data={[{ type: 'scatter', mode: 'lines+markers', x: km.data.rows.map((r: any) => r.k), y: km.data.rows.map((r: any) => r[key2]), line: { color: p.brand2 } }]} layout={{ margin: { t: 32, l: 48, r: 8, b: 30 }, shapes: [{ type: 'line', x0: k, x1: k, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: p.gold } }] }} />
              ))}
              <Plot title={`Gap statistic (B = ${km.data.gap_B})`} height={220} data={[{ type: 'scatter', mode: 'lines+markers', x: km.data.gap.map((g: any) => g.k), y: km.data.gap.map((g: any) => g.gap), error_y: { type: 'data', array: km.data.gap.map((g: any) => g.sk) }, line: { color: p.brand2 } }]} layout={{ margin: { t: 32, l: 48, r: 8, b: 30 } }} />
              <div className="text-sm"><p className="label">K terbaik per metode (Hasil Live)</p>{Object.entries(km.data.best).map(([a, b]) => <p key={a}>{a}: K = {String(b)}</p>)}<p className="text-xs muted">n_init = 20 & B = 10 agar cepat di browser (laporan: n_init 50, B 100).</p></div>
            </div>
          )}
          {f && (<>
            <StatGrid>
              {f.names.map((nm: string, i: number) => <Stat key={nm} label={`${f.codes[i]} ${nm}`} value={`${f.sizes[i]} hari`} hint={`return ${fmt(f.profile[i].log_ret, 2)}% · vol rel ${fmt(f.profile[i].rel_vol, 2)}×`} />)}
              <Stat label="Silhouette" value={fmtFixed(f.silhouette, 4)} /><Stat label="Inertia" value={fmtFixed(f.inertia, 3)} />
            </StatGrid>
            <div className="grid gap-5 lg:grid-cols-2">
              <Plot title={`Proyeksi PCA (PC1 ${pct(f.pca_evr[0], 0)}, PC2 ${pct(f.pca_evr[1], 0)})`} height={380} data={[
                ...f.names.map((nm: string, c: number) => ({ type: 'scatter', mode: 'markers', name: `${f.codes[c]} ${nm}`, x: f.pca.filter((_: any, i: number) => f.labels[i] === c).map((v: number[]) => v[0]), y: f.pca.filter((_: any, i: number) => f.labels[i] === c).map((v: number[]) => v[1]), text: f.dates.filter((_: any, i: number) => f.labels[i] === c), marker: { size: 8, color: p.c[c], symbol: SYM[c] } })),
                { type: 'scatter', mode: 'markers', name: 'Centroid', x: f.pca_centers.map((v: number[]) => v[0]), y: f.pca_centers.map((v: number[]) => v[1]), marker: { size: 18, symbol: 'x', color: p.text } }]} />
              <Plot title="Timeline rezim per hari (harga Close)" height={380} data={f.names.map((_nm: string, c: number) => ({ type: 'scatter', mode: 'markers', name: f.codes[c], x: f.dates.filter((_: any, i: number) => f.labels[i] === c), y: f.close.filter((_: any, i: number) => f.labels[i] === c), marker: { size: 8, color: p.c[c], symbol: SYM[c] } }))} layout={{ yaxis: { title: { text: 'Close (USD)' } } }} />
            </div>
            <Table caption="Profil rezim (rata-rata fitur, skala asli)" head={['Klaster', 'n', 'log_ret %', 'range %', 'gap %', 'rel_vol ×', 'volat_10 %', 'dist_ma20 %', 'return esok %']} numeric={[1, 2, 3, 4, 5, 6, 7, 8]}
              rows={f.names.map((nm: string, c: number) => [`${f.codes[c]} ${nm}`, f.sizes[c], ...['log_ret', 'range_pct', 'gap_pct', 'rel_vol', 'volat_10', 'dist_ma20', 'next_ret'].map((x) => fmt(f.profile[c][x], 3))])} />
          </>)}
        </Section>
      )}
      {setup.data && anim && fr && (
        <Section title="Animasi algoritma Lloyd: assign → update centroid">
          <div className="flex flex-wrap items-end gap-3">
            <button className="btn-primary" onClick={() => { if (step >= frames.length - 1) setStep(0); setPlaying(!playing); }}>{playing ? 'Jeda' : 'Putar'}</button>
            <button className="btn-ghost" onClick={() => setStep(Math.max(0, step - 1))}>◀</button>
            <button className="btn-ghost" onClick={() => setStep(Math.min(frames.length - 1, step + 1))}>▶</button>
            <button className="btn-ghost" onClick={() => setSeed(seed + 1)}>Pusat awal lain (seed {seed + 1})</button>
            <span className="text-sm">Iterasi {fr.i + 1}/{anim.iterations.length} · langkah <strong>{fr.phase === 'assign' ? 'ASSIGN' : 'UPDATE'}</strong> · SSE {fmt(fr.it.sse, 3)}{fr.it.moved !== null ? ` · ${fr.it.moved} hari pindah klaster` : ''}</span>
          </div>
          <Plot height={420} data={[
            ...Array.from({ length: k }, (_, c) => ({ type: 'scatter', mode: 'markers', name: `C${c + 1}`, x: setup.data.X.filter((_: any, i: number) => fr.it.labels[i] === c).map((v: number[]) => project(v)[0]), y: setup.data.X.filter((_: any, i: number) => fr.it.labels[i] === c).map((v: number[]) => project(v)[1]), marker: { size: 7, color: p.c[c], symbol: SYM[c], opacity: 0.75 } })),
            { type: 'scatter', mode: 'markers', name: 'Centroid', x: (fr.phase === 'assign' ? fr.it.centers : fr.it.newCenters).map((v) => project(v)[0]), y: (fr.phase === 'assign' ? fr.it.centers : fr.it.newCenters).map((v) => project(v)[1]), marker: { size: 20, symbol: 'x', color: p.text, line: { width: 2 } } }]}
            layout={{ xaxis: { title: { text: 'PC1' } }, yaxis: { title: { text: 'PC2' } } }} />
          <Callout kind={animOk ? 'ok' : 'warn'} title="Verifikasi">Animasi dihitung TypeScript di ruang 6 dimensi (proyeksi PCA hanya untuk tampilan). Label akhir {animOk ? 'identik' : 'BERBEDA'} dengan KMeans(init = pusat yang sama, n_init = 1) scikit-learn; inertia sklearn {fmt(setup.data.sk_inertia, 4)} vs TS {fmt(anim.sse, 4)}.</Callout>
        </Section>
      )}
      {key && (
        <Section title="Kestabilan lintas seed (ARI)">
          <button className="btn-ghost" disabled={stab.loading} onClick={() => stab.run(() => py('dmx.t08', 'stability', { data_key: key, k, n_seeds: 10, init }))}>Jalankan 10 seed × n_init 1/10/100</button>
          {stab.loading && <Spinner text="Menjalankan ratusan K-Means …" />}
          {stab.data && (<>
            <Table head={['n_init', 'ARI berpasangan rata-rata', 'ARI minimum', '% seed mencapai inertia terbaik']} numeric={[1, 2, 3]} rows={Object.entries(stab.data.by_n_init).map(([n, v]: any) => [n, fmtFixed(v.pairwise_mean, 3), fmtFixed(v.pairwise_min, 3), fmt(v.pct_best, 0)])} />
            <Plot title="ARI berpasangan per K (n_init = 20, 6 seed)" height={260} data={[{ type: 'bar', x: stab.data.per_k.map((x: any) => x.k), y: stab.data.per_k.map((x: any) => x.pairwise_ari_mean), marker: { color: stab.data.per_k.map((x: any) => (x.k === 3 ? p.brand2 : p.neutral)) } }]} layout={{ yaxis: { range: [0, 1.05] } }} />
            <p className="text-sm">K yang reproduktif (ARI ≈ 1 lintas seed) lebih dapat dipertanggungjawabkan daripada K yang dipilih satu indeks saja — alasan laporan memilih K = 3.</p>
          </>)}
        </Section>
      )}
    </div>
  );
}

function Manual() {
  const [key, setKey] = useState<string | null>(null);
  const s = useAsync<any>();
  const [it, setIt] = useState(0);
  useEffect(() => { if (key) s.run(() => py('dmx.t08', 'manual_example', { data_key: key })); }, [key, s.run]);
  const r = s.data;
  const res = useMemo(() => (r ? lloyd(r.points, r.init_centers) : null), [r]);
  const p = palette();
  if (!res || !r) return <div className="space-y-5"><DataSource dataKey="t08" sample={m.sample} onReady={setKey} />{s.loading && <Spinner />}</div>;
  const cur = res.iterations[Math.min(it, res.iterations.length - 1)];
  const dist = (a: number[], b: number[]) => Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2);
  const diffC = Math.max(...res.centers.flat().map((v, i) => Math.abs(v - r.sk_centers.flat()[i])));
  return (
    <div className="space-y-5">
      <DataSource dataKey="t08" sample={m.sample} onReady={setKey} />
      <Section title="Contoh manual laporan: 10 hari, fitur log_ret & range_pct (dibulatkan 2 desimal), K = 3" right={<LiveBadge />}>
        <p className="text-sm">Pusat awal = data ke-{r.init_indices.map((i: number) => i + 1).join(', ')} (rng seed 42). Pilih iterasi:</p>
        <div className="flex gap-2">{res.iterations.map((_, i) => <button key={i} className={i === it ? 'btn-primary' : 'btn-ghost'} onClick={() => setIt(i)}>Iterasi {i + 1}</button>)}</div>
        <Table head={['Hari', 'Tanggal', 'log_ret', 'range_pct', ...cur.centers.map((_, j) => `D ke C${j + 1}`), 'Klaster']} numeric={[2, 3, 4, 5, 6]}
          rows={r.points.map((pt: number[], i: number) => [i + 1, r.dates[i], fmt(pt[0], 2), fmt(pt[1], 2), ...cur.centers.map((c) => fmtFixed(dist(pt, c), 4)), `C${cur.labels[i] + 1}`])} />
        <Table head={['Klaster', 'Centroid dipakai', 'Centroid baru (rata-rata anggota)']} rows={cur.centers.map((c, j) => [`C${j + 1}`, `(${fmtFixed(c[0], 4)}; ${fmtFixed(c[1], 4)})`, `(${fmtFixed(cur.newCenters[j][0], 4)}; ${fmtFixed(cur.newCenters[j][1], 4)})`])} />
        <p className="text-sm">SSE iterasi ini = {fmtFixed(cur.sse, 6)}{cur.moved !== null ? ` · ${cur.moved} hari pindah klaster` : ''}.</p>
        <Plot height={320} data={[
          ...cur.centers.map((_, c) => ({ type: 'scatter', mode: 'markers+text', name: `C${c + 1}`, x: r.points.filter((_: any, i: number) => cur.labels[i] === c).map((v: number[]) => v[0]), y: r.points.filter((_: any, i: number) => cur.labels[i] === c).map((v: number[]) => v[1]), text: r.points.map((_: any, i: number) => i + 1).filter((_: any, i: number) => cur.labels[i] === c), textposition: 'top center', marker: { size: 11, color: p.c[c], symbol: SYM[c] } })),
          { type: 'scatter', mode: 'markers', name: 'Centroid lama', x: cur.centers.map((c) => c[0]), y: cur.centers.map((c) => c[1]), marker: { size: 16, symbol: 'x', color: p.text } },
          { type: 'scatter', mode: 'markers', name: 'Centroid baru', x: cur.newCenters.map((c) => c[0]), y: cur.newCenters.map((c) => c[1]), marker: { size: 16, symbol: 'star', color: p.gold } }]}
          layout={{ xaxis: { title: { text: 'log_ret (%)' } }, yaxis: { title: { text: 'range_pct (%)' } } }} />
        <Callout kind={diffC < 1e-8 ? 'ok' : 'warn'} title="Verifikasi vs scikit-learn KMeans(init=pusat yang sama, n_init=1)">Label {res.labels.every((l, i) => l === r.sk_labels[i]) ? 'identik' : 'berbeda'}; selisih centroid maks {diffC.toExponential(2)}; SSE TS {fmtFixed(res.sse, 6)} vs inertia sklearn {fmtFixed(r.sk_inertia, 6)}; {int(res.iterations.length)} iterasi (sklearn {r.sk_n_iter}).</Callout>
      </Section>
    </div>
  );
}

export default function StudiKasus08() {
  return <ModulePage m={m} report={<Report />} lab={<Lab />} manual={<Manual />} />;
}
