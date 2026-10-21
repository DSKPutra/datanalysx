import { useEffect, useRef, useState } from 'react';
import { loadDataText, loadDataUrl } from '../lib/pyodide/client';
import { useI18n } from '../lib/i18n';
import { Callout, Spinner } from './ui';

export interface SampleInfo { url: string; label: string; license: string }

/** Pilih dataset contoh (bila lisensinya mengizinkan) atau unggah CSV sendiri.
 *  Memanggil onReady(key) setelah data dimuat ke Python. */
export default function DataSource({ dataKey, sample, onReady, uploadHint, uploadTitle = 'Dataset contoh tidak disertakan', multiple = false, onMulti }: {
  dataKey: string; sample?: SampleInfo | null; onReady: (key: string, source: string) => void; uploadHint?: string; uploadTitle?: string;
  multiple?: boolean; onMulti?: (keys: string[]) => void;
}) {
  const { t } = useI18n();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [source, setSource] = useState<string | null>(null);
  const started = useRef(false);

  async function loadSample() {
    if (!sample) return;
    setBusy(true); setErr(null);
    try {
      await loadDataUrl(dataKey, import.meta.env.BASE_URL + sample.url);
      setSource(sample.label);
      onReady(dataKey, sample.label);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }
  useEffect(() => {
    if (sample && !started.current) { started.current = true; loadSample(); }
  }, []);

  async function onFiles(files: FileList | null) {
    if (!files?.length) return;
    setBusy(true); setErr(null);
    try {
      if (multiple && files.length > 1 && onMulti) {
        const keys: string[] = [];
        for (const [i, f] of Array.from(files).entries()) {
          const k = `${dataKey}_up${i}`;
          await loadDataText(k, await f.text(), f.name);
          keys.push(k);
        }
        setSource(`${files.length} berkas diunggah`);
        onMulti(keys);
        const first = keys[0];
        onReady(first, files[0].name);
      } else {
        const f = files[0];
        if (f.size > 60 * 1024 * 1024) throw new Error('Berkas terlalu besar (> 60 MB).');
        await loadDataText(dataKey, await f.text(), f.name);
        setSource(`${f.name} (unggahan)`);
        onReady(dataKey, f.name);
      }
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }

  return (
    <div data-export-skip className="card space-y-3">
      <div className="flex flex-wrap items-end gap-3">
        {sample && (
          <button className="btn-ghost" onClick={loadSample} disabled={busy}>Pakai dataset contoh</button>
        )}
        <label className="btn-primary cursor-pointer">
          Unggah CSV{multiple ? ' (boleh banyak)' : ''}
          <input type="file" accept=".csv,text/csv" multiple={multiple} className="sr-only" onChange={(e) => onFiles(e.target.files)} />
        </label>
        {busy && <Spinner text="Memuat data …" />}
        {source && !busy && <span className="text-sm">Data aktif: <strong data-export-dataset>{source}</strong></span>}
      </div>
      {sample && <p className="text-xs muted">Dataset contoh: {sample.label} — lisensi {sample.license}.</p>}
      {!sample && <Callout kind="info" title={uploadTitle}>{uploadHint ?? 'Lisensi dataset asli tidak mengizinkan redistribusi; silakan unggah berkas Anda sendiri.'}</Callout>}
      <p className="text-xs muted">🔒 {t('privacy')}</p>
      {err && <Callout kind="warn" title="Gagal memuat data">{err}</Callout>}
    </div>
  );
}
