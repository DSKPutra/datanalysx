import { useEffect, useState, type ReactNode } from 'react';
import { initPython, usePyStatus } from '../lib/pyodide/client';
import { Callout } from './ui';

/** Memuat Pyodide secara lazy saat komponen ini pertama kali tampil (tab Lab/Manual dibuka). */
export default function PyGate({ children }: { children: ReactNode }) {
  const st = usePyStatus();
  const [secs, setSecs] = useState(0);
  useEffect(() => { initPython().catch(() => undefined); }, []);
  useEffect(() => {
    if (st.status !== 'loading') return;
    const t = setInterval(() => setSecs(Math.round((Date.now() - (st.startedAt ?? Date.now())) / 1000)), 500);
    return () => clearInterval(t);
  }, [st.status, st.startedAt]);
  if (st.status === 'ready') return <>{children}</>;
  if (st.status === 'error') {
    return (
      <Callout kind="warn" title="Python gagal dimuat">
        <p>{st.message}</p>
        <p>Periksa koneksi internet (Pyodide diunduh dari cdn.jsdelivr.net pada kunjungan pertama), lalu muat ulang halaman.</p>
      </Callout>
    );
  }
  return (
    <div className="card space-y-3" role="status" aria-live="polite">
      <div className="flex items-center gap-3">
        <span className="inline-block h-6 w-6 animate-spin rounded-full border-[3px] border-t-transparent" style={{ borderColor: 'var(--brand-2)', borderTopColor: 'transparent' }} />
        <div>
          <p className="font-semibold">Menyiapkan Python di browser … {secs > 0 && <span className="muted">({secs} dtk)</span>}</p>
          <p className="text-sm muted">{st.message}</p>
        </div>
      </div>
      <p className="text-xs muted">
        Kunjungan pertama mengunduh ±30 MB (Pyodide 0.29.5 + numpy, pandas, scikit-learn, scipy) lalu disimpan di cache browser;
        kunjungan berikutnya jauh lebih cepat. Semua perhitungan berjalan di perangkat Anda.
      </p>
    </div>
  );
}
