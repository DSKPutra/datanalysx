import { useId, useState, type ReactNode } from 'react';
import { fmt } from '../lib/format';

export function Callout({ kind = 'info', title, children }: { kind?: 'info' | 'warn' | 'ok'; title?: string; children: ReactNode }) {
  const styles = {
    info: { borderColor: 'var(--brand-2)', background: 'color-mix(in srgb, var(--brand-2) 8%, var(--surface))' },
    warn: { borderColor: 'var(--warn-line)', background: 'var(--warn-bg)' },
    ok: { borderColor: 'var(--up)', background: 'color-mix(in srgb, var(--up) 10%, var(--surface))' },
  }[kind];
  return (
    <div role={kind === 'warn' ? 'note' : undefined} data-export-callout={kind} className="rounded-xl border-l-4 p-4 text-sm" style={styles}>
      {title && <p data-export-title className="mb-1 font-semibold">{kind === 'warn' ? '⚠ ' : ''}{title}</p>}
      <div className="prose-dmx">{children}</div>
    </div>
  );
}

export function ReportBadge({ source }: { source?: string | null }) {
  return (
    <span className="chip" style={{ background: 'color-mix(in srgb, var(--gold) 18%, transparent)', color: 'var(--text)' }}
      title={source ?? undefined}>📄 Hasil Laporan{source ? ` · ${source}` : ''}</span>
  );
}
export function LiveBadge() {
  return <span data-export-skip className="chip" style={{ background: 'color-mix(in srgb, var(--brand-2) 18%, transparent)', color: 'var(--text)' }}>⚡ Hasil Live (dihitung di browser)</span>;
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div data-export-stat className="rounded-xl border p-3" style={{ borderColor: 'var(--line)', background: 'var(--surface)' }}>
      <div data-export-label className="text-xs font-semibold uppercase tracking-wide muted">{label}</div>
      <div data-export-value className="mt-1 text-xl font-bold tabular-nums">{value}</div>
      {hint && <div data-export-hint className="mt-0.5 text-xs muted">{hint}</div>}
    </div>
  );
}
export function StatGrid({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">{children}</div>;
}

export function Section({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section data-export-section className="card space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-2xl font-semibold">{title}</h2>
        {right}
      </div>
      {children}
    </section>
  );
}

type Cell = ReactNode;
export function Table({ head, rows, numeric, caption }: { head: string[]; rows: Cell[][]; numeric?: number[]; caption?: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="tbl">
        {caption && <caption className="mb-1 text-left text-xs muted">{caption}</caption>}
        <thead><tr>{head.map((h, i) => <th key={i} scope="col" className={numeric?.includes(i) ? 'num' : ''}>{h}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{r.map((c, j) => <td key={j} className={numeric?.includes(j) ? 'num' : ''}>{typeof c === 'number' ? fmt(c) : c}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Field({ label, children }: { label: string; children: (id: string) => ReactNode }) {
  const id = useId();
  return <div><label htmlFor={id} className="label">{label}</label>{children(id)}</div>;
}

export function Select<T extends string | number>({ label, value, options, onChange }: {
  label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void;
}) {
  return (
    <Field label={label}>{(id) => (
      <select id={id} className="input" value={String(value)}
        onChange={(e) => {
          const o = options.find((x) => String(x.value) === e.target.value);
          if (o) onChange(o.value);
        }}>
        {options.map((o) => <option key={String(o.value)} value={String(o.value)}>{o.label}</option>)}
      </select>
    )}</Field>
  );
}

export function Slider({ label, value, min, max, step = 1, onChange, format }: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; format?: (v: number) => string;
}) {
  return (
    <Field label={`${label}: ${format ? format(value) : value}`}>{(id) => (
      <input id={id} type="range" className="w-full accent-[var(--brand-2)]" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} />
    )}</Field>
  );
}

export function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  const id = useId();
  return (
    <label htmlFor={id} className="flex cursor-pointer items-center gap-2 text-sm">
      <input id={id} type="checkbox" className="h-4 w-4 accent-[var(--brand-2)]" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}

export function ErrorBox({ error }: { error: string | null }) {
  if (!error) return null;
  return <Callout kind="warn" title="Terjadi kesalahan">{error}</Callout>;
}

export function Spinner({ text = 'Menghitung …' }: { text?: string }) {
  return <div data-export-skip role="status" className="flex items-center gap-2 text-sm muted"><span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />{text}</div>;
}

export function Collapse({ title, children, defaultOpen = false }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-xl border" style={{ borderColor: 'var(--line)' }}>
      <button className="flex w-full items-center justify-between px-4 py-2 text-left text-sm font-semibold" aria-expanded={open} onClick={() => setOpen(!open)}>
        {title}<span aria-hidden>{open ? '−' : '+'}</span>
      </button>
      {open && <div className="border-t px-4 py-3" style={{ borderColor: 'var(--line)' }}>{children}</div>}
    </div>
  );
}
