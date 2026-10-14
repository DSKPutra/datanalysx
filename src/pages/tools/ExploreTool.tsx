import { useState } from 'react';
import Dataset from '../../components/Dataset';
import Plot from '../../components/Plot';
import { LiveBadge, Section, Select, Spinner } from '../../components/ui';
import { py } from '../../lib/pyodide/client';
import { palette } from '../../lib/theme';
import { useAsync } from '../../lib/useAsync';
import { Profile } from '../LabShared';
import ToolShell from './ToolShell';

function Inner() {
  const [key, setKey] = useState<string | null>(null);
  const [ver, setVer] = useState(0);
  const prep = useAsync<any>();
  const [opt, setOpt] = useState({ impute: 'median', outlier: 'none', encode: 'onehot', scale: 'none' });
  const [col, setCol] = useState('');
  const p = palette();
  const cols = prep.data ? Object.keys(prep.data.before) : [];
  const sel = col && cols.includes(col) ? col : cols[0];
  return (
    <div className="space-y-5">
      <Dataset dataKey="explore" tool="eksplorasi" onReady={(i) => { setKey(i.key); setVer((v) => v + 1); prep.setData(null); }} />
      {key && <Profile dataKey={key} version={ver} />}
      {key && (
        <Section title="Praproses data (before–after)" right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-4">
            <Select label="Imputasi nilai hilang" value={opt.impute} onChange={(v) => setOpt({ ...opt, impute: v })} options={[{ value: 'none', label: 'Tidak' }, { value: 'mean', label: 'Mean' }, { value: 'median', label: 'Median' }, { value: 'interpolate', label: 'Interpolasi linear' }]} />
            <Select label="Outlier (IQR 1,5×)" value={opt.outlier} onChange={(v) => setOpt({ ...opt, outlier: v })} options={[{ value: 'none', label: 'Biarkan' }, { value: 'winsorize', label: 'Winsorization' }, { value: 'drop', label: 'Buang baris' }]} />
            <Select label="Encoding kategori" value={opt.encode} onChange={(v) => setOpt({ ...opt, encode: v })} options={[{ value: 'none', label: 'Tidak' }, { value: 'onehot', label: 'One-hot' }, { value: 'label', label: 'Label' }]} />
            <Select label="Scaling" value={opt.scale} onChange={(v) => setOpt({ ...opt, scale: v })} options={[{ value: 'none', label: 'Tidak' }, { value: 'standard', label: 'Standard (z-score)' }, { value: 'minmax', label: 'MinMax [0,1]' }]} />
          </div>
          <button className="btn-primary" disabled={prep.loading} onClick={() => prep.run(() => py('dmx.eda', 'preprocess', { data_key: key, ...opt }))}>Terapkan</button>
          {prep.loading && <Spinner />}
          {prep.data && (<>
            <ul className="prose-dmx text-sm">{prep.data.log.map((l: string) => <li key={l}>{l}</li>)}</ul>
            <p className="text-sm">Ukuran: {prep.data.shape_before.join(' × ')} → {prep.data.shape_after.join(' × ')}</p>
            {sel && <Select label="Kolom" value={sel} onChange={setCol} options={cols.map((c) => ({ value: c, label: c }))} />}
            {sel && prep.data.after[sel] && <div className="grid gap-4 md:grid-cols-2">
              <Plot title={`Sebelum: ${sel}`} height={240} data={[{ type: 'bar', x: prep.data.before[sel].hist.edges.slice(0, -1), y: prep.data.before[sel].hist.counts, marker: { color: p.neutral } }]} />
              <Plot title={`Sesudah: ${sel}`} height={240} data={[{ type: 'bar', x: prep.data.after[sel].hist.edges.slice(0, -1), y: prep.data.after[sel].hist.counts, marker: { color: p.brand2 } }]} />
            </div>}
            {prep.data.csv && <a className="btn-ghost" download="datanalysx_praproses.csv" href={`data:text/csv;charset=utf-8,${encodeURIComponent(prep.data.csv)}`}>Unduh CSV hasil praproses</a>}
            <p className="text-xs muted">Praproses di sini diterapkan ke seluruh data untuk eksplorasi. Alat Regresi/Klasifikasi melakukan praproses di dalam Pipeline yang di-fit pada data latih saja.</p>
          </>)}
        </Section>
      )}
    </div>
  );
}

export default function ExploreTool() {
  return <ToolShell title="Eksplorasi & Praproses Data" cases={['01', '03', '06', '07']}
    intro="Profil kolom, tipe data, nilai hilang, duplikat, statistik deskriptif, histogram, boxplot, korelasi; lalu imputasi, penanganan outlier IQR, encoding, dan scaling dengan perbandingan sebelum–sesudah."><Inner /></ToolShell>;
}
