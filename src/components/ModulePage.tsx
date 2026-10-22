import { useEffect, useState, type ReactNode } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { MODULES, type ModuleInfo } from '../content/modules';
import { useI18n, type Key } from '../lib/i18n';
import CodeViewer from './CodeViewer';
import ExportPanel from './ExportPanel';
import PyGate from './PyGate';
import { Callout, Section } from './ui';

const TABS: { id: string; key: Key }[] = [
  { id: 'ringkasan', key: 'tabSummary' }, { id: 'data', key: 'tabData' }, { id: 'laporan', key: 'tabReport' },
  { id: 'lab', key: 'tabLab' }, { id: 'manual', key: 'tabManual' }, { id: 'kode', key: 'tabCode' }, { id: 'kritis', key: 'tabNotes' },
];

export function Summary({ m, extra }: { m: ModuleInfo; extra?: ReactNode }) {
  return (
    <div className="space-y-5">
      <Section title="Pertanyaan penelitian"><p className="text-lg">{m.question}</p></Section>
      <Section title="Alur CRISP-DM">
        <ol className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {m.crisp.map(([k, v], i) => (
            <li key={k} className="rounded-xl border p-3" style={{ borderColor: 'var(--line)' }}>
              <span className="chip mb-1 on-brand">{i + 1}. {k}</span>
              <p className="text-sm">{v}</p>
            </li>
          ))}
        </ol>
      </Section>
      <Section title="Hasil utama (dari laporan)"><ul className="prose-dmx">{m.findings.map((f) => <li key={f}>{f}</li>)}</ul></Section>
      {extra}
    </div>
  );
}

export function DataTab({ m, children }: { m: ModuleInfo; children?: ReactNode }) {
  return (
    <div className="space-y-5">
      <Section title="Dataset">
        <dl className="grid gap-3 text-sm md:grid-cols-2">
          <div><dt className="label">Nama</dt><dd>{m.dataset.name}</dd></div>
          <div><dt className="label">Lisensi</dt><dd>{m.dataset.license}</dd></div>
          <div className="md:col-span-2"><dt className="label">Struktur</dt><dd>{m.dataset.detail}</dd></div>
          <div className="md:col-span-2"><dt className="label">Asal</dt><dd>{m.dataset.origin}</dd></div>
          <div className="md:col-span-2"><dt className="label">Di aplikasi</dt><dd>{m.sample ? <>Dataset contoh disertakan: <a href={import.meta.env.BASE_URL + m.sample.url} download>{m.sample.url.split('/').pop()}</a></> : 'Tidak disertakan — gunakan tombol unggah di tab Lab Live.'}</dd></div>
        </dl>
      </Section>
      {children}
    </div>
  );
}

export function Critical({ m, extra }: { m: ModuleInfo; extra?: ReactNode }) {
  return (
    <div className="space-y-5">
      {m.warnings.length > 0 && <Callout kind="warn" title="Peringatan">{m.warnings.map((w) => <p key={w}>{w}</p>)}</Callout>}
      <Section title="Catatan kritis"><ul className="prose-dmx">{m.critical.map((c) => <li key={c}>{c}</li>)}</ul></Section>
      <div className="grid gap-5 md:grid-cols-2">
        <Section title="Kelebihan"><ul className="prose-dmx">{m.pros.map((c) => <li key={c}>{c}</li>)}</ul></Section>
        <Section title="Kekurangan"><ul className="prose-dmx">{m.cons.map((c) => <li key={c}>{c}</li>)}</ul></Section>
      </div>
      <Section title="Kapan dipakai"><p>{m.when}</p></Section>
      {extra}
    </div>
  );
}

export default function ModulePage({ m, summary, data, report, lab, manual, critical }: {
  m: ModuleInfo; summary?: ReactNode; data?: ReactNode; report: ReactNode; lab: ReactNode; manual: ReactNode; critical?: ReactNode;
}) {
  const { t } = useI18n();
  const [params, setParams] = useSearchParams();
  const tab = TABS.some((x) => x.id === params.get('tab')) ? params.get('tab')! : 'ringkasan';
  const [visited, setVisited] = useState<Set<string>>(() => new Set([tab]));
  useEffect(() => { setVisited((v) => (v.has(tab) ? v : new Set(v).add(tab))); }, [tab]);
  useEffect(() => { document.title = `Studi Kasus ${m.no} — ${m.algo} · DATANALYSX`; }, [m]);
  const idx = MODULES.findIndex((x) => x.slug === m.slug);
  const prev = MODULES[idx - 1], next = MODULES[idx + 1];
  const content: Record<string, ReactNode> = {
    ringkasan: summary ?? <Summary m={m} />,
    data: data ?? <DataTab m={m} />,
    laporan: report,
    lab: <PyGate>{lab}</PyGate>,
    manual: <PyGate>{manual}</PyGate>,
    kode: <Section title="Kode Python"><CodeViewer files={m.code.length ? m.code : []} /></Section>,
    kritis: critical ?? <Critical m={m} />,
  };
  return (
    <article data-export-root className="space-y-5">
      <header className="space-y-2">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="chip" style={{ color: 'var(--brand-2)' }}>// Studi Kasus {m.no}</span>
          <span className="chip" style={{ background: 'var(--surface-2)' }}>{m.group}</span>
          {!m.reportAvailable && <span className="chip" style={{ background: 'var(--warn-bg)' }}>Laporan belum tersedia</span>}
        </div>
        <h1 className="text-4xl font-bold leading-tight md:text-5xl">{m.title}</h1>
        <p data-export-skip className="max-w-3xl muted">{m.short}</p>
        {m.tool && (
          <p data-export-skip className="max-w-3xl rounded-lg border-l-4 px-3 py-2 text-sm" style={{ borderColor: 'var(--gold)', background: 'var(--surface-2)' }}>
            Halaman ini adalah <strong>studi kasus</strong> — contoh penerapan pada satu dataset, bukan patokan. Untuk menganalisis data Anda sendiri, gunakan alat <Link to={m.tool.path}><strong>{m.tool.label}</strong></Link>.
          </p>
        )}
        <ExportPanel key={m.slug} defaultTitle={`Studi Kasus ${m.no}: ${m.title}`} subtitle={m.short} />
      </header>
      <div data-export-skip role="tablist" aria-label="Bagian modul" className="flex gap-1 overflow-x-auto border-b pb-px" style={{ borderColor: 'var(--line)' }}>
        {TABS.map((x) => (
          <button key={x.id} role="tab" id={`tab-${x.id}`} aria-selected={tab === x.id} aria-controls={`panel-${x.id}`}
            className={`nav-link !rounded-b-none !py-2 font-semibold${tab === x.id ? ' active' : ''}`}
            onClick={() => setParams({ tab: x.id }, { replace: true })}>{t(x.key)}</button>
        ))}
      </div>
      {TABS.map((x) => visited.has(x.id) && (
        <div key={x.id} role="tabpanel" id={`panel-${x.id}`} aria-labelledby={`tab-${x.id}`} hidden={tab !== x.id}>{content[x.id]}</div>
      ))}
      <nav data-export-skip className="flex justify-between gap-3 border-t pt-4 text-sm" style={{ borderColor: 'var(--line)' }} aria-label="Modul lain">
        {prev ? <Link to={`/${prev.slug}`}>← Studi Kasus {prev.no}: {prev.algo}</Link> : <span />}
        {next ? <Link to={`/${next.slug}`}>Studi Kasus {next.no}: {next.algo} →</Link> : <span />}
      </nav>
    </article>
  );
}
