import { useEffect, useMemo, useRef, useState } from 'react';
import Dataset, { type Column } from '../../components/Dataset';
import Plot from '../../components/Plot';
import { Callout, ErrorBox, LiveBadge, Section, Select, Slider, Spinner, Stat, StatGrid, Table, Toggle } from '../../components/ui';
import { fmt, fmtFixed, int, pct } from '../../lib/format';
import { lloyd } from '../../lib/manual/kmeans';
import { py } from '../../lib/pyodide/client';
import { palette } from '../../lib/theme';
import { useAsync } from '../../lib/useAsync';
import ToolShell from './ToolShell';

const SYM = ['triangle-up', 'triangle-down', 'diamond', 'circle', 'square', 'cross', 'x', 'star', 'hexagon', 'pentagon'];

function Animation({ dataKey, features, k, scaler, init }: { dataKey: string; features: string[]; k: number; scaler: string; init: string }) {
  const [seed, setSeed] = useState(42);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const s = useAsync<any>();
  useEffect(() => { s.run(() => py('dmx.cluster', 'lloyd_setup', { data_key: dataKey, features, k, scaler, init, seed })); setStep(0); setPlaying(false); }, [dataKey, features, k, scaler, init, seed, s.run]);
  const anim = useMemo(() => (s.data ? lloyd(s.data.X, s.data.init_centers) : null), [s.data]);
  const frames = anim ? anim.iterations.flatMap((it, i) => [{ phase: 'assign', i, it }, { phase: 'update', i, it }]) : [];
  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => setStep((x) => { if (x >= frames.length - 1) { setPlaying(false); return x; } return x + 1; }), 900);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [playing, frames.length]);
  const p = palette();
  if (!s.data || !anim) return s.loading ? <Spinner /> : null;
  const proj = (v: number[]) => s.data.pca_components.map((c: number[]) => c.reduce((a: number, w: number, i: number) => a + w * (v[i] - s.data.pca_mean[i]), 0));
  const fr = frames[Math.min(step, frames.length - 1)];
  const ok = anim.labels.every((l, i) => l === s.data.sk_labels[i]);
  const C = fr.phase === 'assign' ? fr.it.centers : fr.it.newCenters;
  return (
    <Section title="Animasi algoritma Lloyd: assign → update centroid">
      <div className="flex flex-wrap items-end gap-3">
        <button className="btn-primary" onClick={() => { if (step >= frames.length - 1) setStep(0); setPlaying(!playing); }}>{playing ? 'Jeda' : 'Putar'}</button>
        <button className="btn-ghost" onClick={() => setStep(Math.max(0, step - 1))} aria-label="Langkah sebelumnya">◀</button>
        <button className="btn-ghost" onClick={() => setStep(Math.min(frames.length - 1, step + 1))} aria-label="Langkah berikutnya">▶</button>
        <button className="btn-ghost" onClick={() => setSeed(seed + 1)}>Pusat awal lain</button>
        <span className="text-sm">Iterasi {fr.i + 1}/{anim.iterations.length} · <strong>{fr.phase === 'assign' ? 'ASSIGN' : 'UPDATE'}</strong> · SSE {fmt(fr.it.sse, 3)}{fr.it.moved !== null ? ` · ${fr.it.moved} titik pindah` : ''}</span>
      </div>
      <Plot height={420} data={[
        ...Array.from({ length: k }, (_, c) => ({ type: 'scatter', mode: 'markers', name: `C${c + 1}`, x: s.data.X.filter((_: any, i: number) => fr.it.labels[i] === c).map((v: number[]) => proj(v)[0]), y: s.data.X.filter((_: any, i: number) => fr.it.labels[i] === c).map((v: number[]) => proj(v)[1]), marker: { size: 6, color: p.c[c % p.c.length], symbol: SYM[c % SYM.length], opacity: 0.7 } })),
        { type: 'scatter', mode: 'markers', name: 'Centroid', x: C.map((v) => proj(v)[0]), y: C.map((v) => proj(v)[1]), marker: { size: 20, symbol: 'x', color: p.text } }]} layout={{ xaxis: { title: { text: 'PC1' } }, yaxis: { title: { text: 'PC2' } } }} />
      <Callout kind={ok ? 'ok' : 'warn'} title="Verifikasi">Iterasi dihitung TypeScript di ruang {features.length} dimensi; label akhir {ok ? 'identik' : 'BERBEDA'} dengan KMeans scikit-learn (pusat awal sama, n_init = 1). Inertia sklearn {fmt(s.data.sk_inertia, 4)} vs TS {fmt(anim.sse, 4)}.</Callout>
    </Section>
  );
}

function Inner() {
  const [info, setInfo] = useState<{ key: string; columns: Column[] } | null>(null);
  const [feats, setFeats] = useState<string[]>([]);
  const [scaler, setScaler] = useState('standard');
  const [init, setInit] = useState('k-means++');
  const [nInit, setNInit] = useState(10);
  const [k, setK] = useState(3);
  const [labelCol, setLabelCol] = useState('');
  const km = useAsync<any>();
  const fit = useAsync<any>();
  const stab = useAsync<any>();
  const nums = info ? info.columns.filter((c) => c.kind === 'numerik').map((c) => c.name) : [];
  function onReady(i: any) {
    setInfo(i); km.setData(null); fit.setData(null); stab.setData(null);
    const n = i.columns.filter((c: Column) => c.kind === 'numerik').map((c: Column) => c.name);
    setFeats((i.preset.features ?? n.slice(0, 4)).filter((f: string) => n.includes(f)));
    setLabelCol('');
  }
  const ready = info && feats.length >= 2;
  useEffect(() => { if (ready) km.run(() => py('dmx.cluster', 'k_metrics', { data_key: info!.key, features: feats, scaler, init, n_init: Math.min(nInit, 10) })); }, [info, feats, scaler, init, nInit, km.run]);
  useEffect(() => { if (ready) fit.run(() => py('dmx.cluster', 'fit', { data_key: info!.key, features: feats, k, scaler, init, n_init: nInit, label_col: labelCol || null })); }, [info, feats, scaler, init, nInit, k, labelCol, fit.run]);
  const p = palette();
  const f = fit.data;
  return (
    <div className="space-y-5">
      <Dataset dataKey="clustering" tool="clustering" onReady={onReady} />
      {info && (
        <Section title="Atur K-Means" right={<LiveBadge />}>
          <fieldset><legend className="label">Fitur numerik (minimal 2)</legend>
            <div className="grid grid-cols-2 gap-1 md:grid-cols-4">{nums.map((c) => <Toggle key={c} label={c} checked={feats.includes(c)} onChange={(v) => setFeats(v ? [...feats, c] : feats.filter((x) => x !== c))} />)}</div>
          </fieldset>
          <div className="grid gap-4 md:grid-cols-5">
            <Slider label="K" value={k} min={2} max={10} onChange={setK} />
            <Select label="Scaling" value={scaler} onChange={setScaler} options={[{ value: 'standard', label: 'StandardScaler' }, { value: 'minmax', label: 'MinMaxScaler' }, { value: 'robust', label: 'RobustScaler' }, { value: 'none', label: 'Tanpa scaling' }]} />
            <Select label="Inisialisasi" value={init} onChange={setInit} options={[{ value: 'k-means++', label: 'k-means++' }, { value: 'random', label: 'Acak' }]} />
            <Select label="n_init" value={nInit} onChange={setNInit} options={[1, 5, 10, 50, 100].map((v) => ({ value: v, label: String(v) }))} />
            <Select label="Label hover (opsional)" value={labelCol} onChange={setLabelCol} options={[{ value: '', label: '(tidak ada)' }, ...info.columns.map((c) => ({ value: c.name, label: c.name }))]} />
          </div>
          {scaler === 'none' && <Callout kind="warn" title="Tanpa scaling">K-Means berbasis jarak Euclidean: fitur berskala besar akan mendominasi klaster.</Callout>}
          <ErrorBox error={km.error || fit.error} />
          {km.data && (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {([['inertia', 'Elbow (inertia)'], ['silhouette', 'Silhouette ↑'], ['dbi', 'Davies–Bouldin ↓'], ['ch', 'Calinski–Harabasz ↑']] as const).map(([key, t]) => (
                <Plot key={key} title={t} height={220} data={[{ type: 'scatter', mode: 'lines+markers', x: km.data.rows.map((r: any) => r.k), y: km.data.rows.map((r: any) => r[key]), line: { color: p.brand2 } }]} layout={{ margin: { t: 32, l: 48, r: 8, b: 30 }, shapes: [{ type: 'line', x0: k, x1: k, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dash', color: p.gold } }] }} />
              ))}
            </div>
          )}
          {km.data && <p className="text-sm">K terbaik menurut metode: silhouette {km.data.best.silhouette} · DBI {km.data.best.dbi} · CH {km.data.best.ch}. Metode sering tidak sepakat — periksa juga kestabilan dan interpretasi.</p>}
          {f && (<>
            <StatGrid>
              {Array.from({ length: f.k }, (_, c) => <Stat key={c} label={`Klaster C${c + 1}`} value={`${int(f.sizes[c])}`} hint={pct(f.sizes[c] / f.n, 1)} />)}
              <Stat label="Silhouette" value={fmtFixed(f.silhouette, 4)} hint="< 0,25 = struktur lemah" /><Stat label="Inertia" value={fmt(f.inertia, 3)} />
            </StatGrid>
            <Plot title={`Proyeksi PCA (PC1 ${pct(f.pca_evr[0], 0)}, PC2 ${pct(f.pca_evr[1] ?? 0, 0)})`} height={420} data={[
              ...Array.from({ length: f.k }, (_, c) => ({ type: 'scatter', mode: 'markers', name: `C${c + 1}`, x: f.pca.filter((_: any, i: number) => f.labels[i] === c).map((v: number[]) => v[0]), y: f.pca.filter((_: any, i: number) => f.labels[i] === c).map((v: number[]) => v[1]), text: f.hover ? f.hover.filter((_: any, i: number) => f.labels[i] === c) : undefined, marker: { size: 7, color: p.c[c % p.c.length], symbol: SYM[c % SYM.length] } })),
              { type: 'scatter', mode: 'markers', name: 'Centroid', x: f.pca_centers.map((v: number[]) => v[0]), y: f.pca_centers.map((v: number[]) => v[1]), marker: { size: 18, symbol: 'x', color: p.text } }]} />
            <Table caption="Profil klaster (rata-rata fitur, skala asli)" head={['Klaster', 'n', ...f.features]} numeric={f.features.map((_: string, i: number) => i + 2).concat(1)} rows={Array.from({ length: f.k }, (_, c) => [`C${c + 1}`, int(f.sizes[c]), ...f.features.map((x: string) => fmt(f.profile[c][x], 4))])} />
            {f.csv && <a className="btn-ghost" download="datanalysx_klaster.csv" href={`data:text/csv;charset=utf-8,${encodeURIComponent(f.csv)}`}>Unduh CSV + kolom cluster</a>}
          </>)}
        </Section>
      )}
      {ready && <Animation dataKey={info!.key} features={feats} k={k} scaler={scaler} init={init} />}
      {ready && (
        <Section title="Kestabilan lintas seed (ARI)">
          <button className="btn-ghost" disabled={stab.loading} onClick={() => stab.run(() => py('dmx.cluster', 'stability', { data_key: info!.key, features: feats, k, scaler }))}>Jalankan 10 seed (n_init 1 vs 10)</button>
          {stab.loading && <Spinner />}
          {stab.data && <Table head={['n_init', 'ARI berpasangan rata-rata', 'ARI minimum', '% seed mencapai inertia terbaik']} numeric={[1, 2, 3]} rows={Object.entries(stab.data).map(([n, v]: any) => [n, fmtFixed(v.pairwise_mean, 3), fmtFixed(v.pairwise_min, 3), fmt(v.pct_best, 0)])} />}
          <p className="text-xs muted">ARI ≈ 1 berarti partisi dapat direproduksi; K yang tidak stabil sebaiknya tidak dipilih walau indeksnya bagus.</p>
        </Section>
      )}
    </div>
  );
}

export default function ClusteringTool() {
  return <ToolShell title="Clustering (K-Means)" cases={['08']}
    intro="Kelompokkan baris data tanpa label: pilih fitur numerik dan scaling, bandingkan K dengan Elbow, Silhouette, DBI, CH, lihat proyeksi PCA, profil klaster, animasi algoritma Lloyd, dan kestabilan lintas seed; unduh hasil sebagai CSV."><Inner /></ToolShell>;
}
