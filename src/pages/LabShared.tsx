import { useEffect, useState } from 'react';
import DataSource, { type SampleInfo } from '../components/DataSource';
import Plot from '../components/Plot';
import { ErrorBox, Section, Select, Spinner, Stat, StatGrid, Table } from '../components/ui';
import { fmt, fmtFixed, int } from '../lib/format';
import { py } from '../lib/pyodide/client';
import { palette } from '../lib/theme';
import { useAsync } from '../lib/useAsync';

export const SAMPLES: (SampleInfo & { key: string })[] = [
  { key: 'semikonduktor', url: 'data/samples/t03_chip_companies_financials.csv', label: 'Semikonduktor (Tugas 03)', license: 'CC0' },
  { key: 'nvidia', url: 'data/samples/t01_nvidia_stock_1999_2026.csv', label: 'NVIDIA (Tugas 01)', license: 'CC0' },
  { key: 'reels', url: 'data/samples/t07_instagram_reels_knn_clean.csv', label: 'Instagram Reels (Tugas 07)', license: 'CC0' },
  { key: 'trending', url: 'data/samples/t05_trending_topics_2026_synthetic.csv', label: 'Trending Topics (Tugas 05)', license: 'CC0' },
  { key: 'macro', url: 'data/samples/t06_indonesia_multiasset_macro_2010_2026.csv', label: 'Makro Indonesia (Tugas 06)', license: 'CC BY-SA 4.0' },
  { key: 'aapl2026', url: 'data/samples/t08_aapl_2026_stock_data.csv', label: 'AAPL 2026 sintetis (Tugas 08)', license: 'Apache 2.0' },
];

export function SamplePicker({ dataKey, onReady, initial = 0 }: { dataKey: string; onReady: (k: string) => void; initial?: number }) {
  const [idx, setIdx] = useState(initial);
  const s = SAMPLES[idx];
  return (
    <div className="space-y-2">
      <Select label="Dataset contoh" value={idx} onChange={setIdx} options={SAMPLES.map((x, i) => ({ value: i, label: x.label }))} />
      <DataSource key={s.key} dataKey={dataKey} sample={s} onReady={(k) => onReady(k)} />
    </div>
  );
}

export function Profile({ dataKey, version }: { dataKey: string; version: number }) {
  const s = useAsync<any>();
  const [col, setCol] = useState<string>('');
  useEffect(() => { s.run(() => py('dmx.eda', 'profile', { data_key: dataKey })); }, [dataKey, version, s.run]);
  const r = s.data;
  const p = palette();
  if (s.loading && !r) return <Spinner />;
  if (!r) return <ErrorBox error={s.error} />;
  const nums = r.columns.filter((c: any) => c.kind === 'numerik');
  const sel = nums.find((c: any) => c.name === col) ?? nums[0];
  return (
    <Section title="Profil data (EDA)">
      <StatGrid><Stat label="Baris" value={int(r.n_rows)} /><Stat label="Kolom" value={r.n_cols} /><Stat label="Duplikat" value={int(r.duplicates)} /><Stat label="Total missing" value={int(r.missing_total)} /></StatGrid>
      <Table head={['Kolom', 'Tipe', 'Jenis', 'Missing', 'Unik', 'Rata-rata', 'SD', 'Min', 'Median', 'Maks', 'Outlier IQR']} numeric={[3, 4, 5, 6, 7, 8, 9, 10]}
        rows={r.columns.map((c: any) => [c.name, c.dtype, c.kind, `${c.missing} (${fmtFixed(c.missing_pct, 1)}%)`, int(c.unique), fmt(c.mean, 3), fmt(c.std, 3), fmt(c.min, 3), fmt(c.median, 3), fmt(c.max, 3), c.n_outlier_iqr ?? ''])} />
      {sel && (<>
        <Select label="Kolom numerik untuk histogram & boxplot" value={sel.name} onChange={setCol} options={nums.map((c: any) => ({ value: c.name, label: c.name }))} />
        <div className="grid gap-4 md:grid-cols-2">
          <Plot title={`Histogram ${sel.name}`} height={260} data={[{ type: 'bar', x: sel.hist.edges.slice(0, -1), y: sel.hist.counts, marker: { color: p.brand2 } }]} />
          <Plot title={`Boxplot ${sel.name} (pagar IQR 1,5×)`} height={260} data={[{ type: 'box', q1: [sel.q1], median: [sel.median], q3: [sel.q3], lowerfence: [Math.max(sel.min, sel.lower_fence)], upperfence: [Math.min(sel.max, sel.upper_fence)], mean: [sel.mean], name: sel.name, marker: { color: p.brand2 } }]} />
        </div>
      </>)}
      {r.corr && <Plot title="Heatmap korelasi Pearson" height={Math.min(620, 160 + 26 * r.corr.cols.length)} data={[{ type: 'heatmap', z: r.corr.values, x: r.corr.cols, y: r.corr.cols, zmin: -1, zmax: 1, colorscale: 'RdBu', reversescale: true }]} layout={{ margin: { l: 130, b: 130 } }} />}
    </Section>
  );
}
