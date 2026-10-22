import { useEffect, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import ExportPanel from '../../components/ExportPanel';
import PyGate from '../../components/PyGate';
import { MODULES } from '../../content/modules';

/** Kerangka halaman alat umum + tautan ke studi kasus terkait. */
export default function ToolShell({ title, intro, cases, children }: { title: string; intro: ReactNode; cases: string[]; children: ReactNode }) {
  useEffect(() => { document.title = `${title} · DATANALYSX`; }, [title]);
  const related = MODULES.filter((m) => cases.includes(m.no));
  return (
    <div data-export-root className="space-y-5">
      <header className="space-y-2">
        <span className="chip" style={{ color: 'var(--brand-2)' }}>// Alat analisis</span>
        <h1 className="text-4xl font-bold md:text-5xl"><span className="text-gradient">{title}</span></h1>
        <div className="max-w-3xl muted">{intro}</div>
        {related.length > 0 && (
          <p data-export-skip className="text-sm">Contoh penerapan: {related.map((m, i) => (
            <span key={m.no}>{i > 0 && ' · '}<Link to={`/${m.slug}`}>Studi kasus {m.no} — {m.algo.split(' (')[0]}</Link></span>
          ))}</p>
        )}
        <ExportPanel defaultTitle={`Analisis ${title}`} subtitle={typeof intro === 'string' ? intro : undefined} />
      </header>
      <PyGate>{children}</PyGate>
    </div>
  );
}
