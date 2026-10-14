import { useState, type ReactNode } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { MODULES } from '../content/modules';
import { useI18n } from '../lib/i18n';
import { useTheme } from '../lib/theme';
import { usePyStatus } from '../lib/pyodide/client';

const BASE = import.meta.env.BASE_URL;

function PyIndicator() {
  const s = usePyStatus();
  const color = { idle: 'var(--neutral)', loading: 'var(--gold)', ready: 'var(--up)', error: 'var(--down)' }[s.status];
  const txt = { idle: 'Python: belum dimuat', loading: 'Python: memuat…', ready: 'Python: siap', error: 'Python: gagal' }[s.status];
  return <span className="hidden items-center gap-1.5 text-xs muted lg:inline-flex" title={s.message}><span className="h-2 w-2 rounded-full" style={{ background: color }} />{txt}</span>;
}

export default function Layout({ children }: { children: ReactNode }) {
  const { t, lang, setLang } = useI18n();
  const { dark, toggle } = useTheme();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  const nav = [
    { to: '/', label: t('home') },
    { to: '/regression-lab', label: t('regLab') },
    { to: '/data-lab', label: t('dataLab') },
    { to: '/perbandingan', label: t('compare') },
    { to: '/pelajaran', label: t('lessons') },
    { to: '/tentang', label: t('about') },
  ];
  const linkCls = ({ isActive }: { isActive: boolean }) =>
    `rounded-md px-2.5 py-1.5 text-sm font-medium ${isActive ? 'on-brand' : ''}`;
  const activeStyle = () => ({});
  return (
    <div className="flex min-h-screen flex-col">
      <a href="#main" className="skip-link">Lewati ke konten</a>
      <header className="sticky top-0 z-40 border-b backdrop-blur" style={{ borderColor: 'var(--line)', background: 'color-mix(in srgb, var(--bg) 88%, transparent)' }}>
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-2.5">
          <Link to="/" className="flex items-center gap-2 no-underline" style={{ color: 'var(--text)' }}>
            <img src={`${BASE}img/favicon.svg`} alt="" aria-hidden width={36} height={36} className="h-9 w-9" />
            <span className="leading-tight">
              <span className="block font-display text-xl font-bold tracking-wide" style={{ color: 'var(--brand)' }}>DATANALYSX</span>
              <span className="hidden text-[11px] muted sm:block">Data Mining Lab · UNPAM</span>
            </span>
          </Link>
          <nav className="ml-auto hidden items-center gap-1 lg:flex" aria-label="Navigasi utama">
            {nav.map((n) => <NavLink key={n.to} to={n.to} end className={linkCls} style={activeStyle}>{n.label}</NavLink>)}
            <details className="relative">
              <summary className="cursor-pointer list-none rounded-md px-2.5 py-1.5 text-sm font-medium">{t('modules')} ▾</summary>
              <div className="absolute right-0 mt-2 w-80 rounded-xl border p-2 shadow-lg" style={{ background: 'var(--surface)', borderColor: 'var(--line)' }}>
                {MODULES.map((m) => (
                  <Link key={m.slug} to={`/${m.slug}`} className="block rounded-md px-2 py-1.5 text-sm no-underline hover:opacity-80" style={{ color: 'var(--text)' }}>
                    <strong>Tugas {m.no}</strong> · {m.algo}
                  </Link>
                ))}
              </div>
            </details>
          </nav>
          <PyIndicator />
          <button className="btn-ghost !px-2.5" onClick={() => setLang(lang === 'id' ? 'en' : 'id')} aria-label={`${lang === 'id' ? 'EN' : 'ID'} — ganti bahasa / switch language`}>{lang === 'id' ? 'EN' : 'ID'}</button>
          <button className="btn-ghost !px-2.5" onClick={toggle} aria-label={t('theme')} title={t('theme')}>{dark ? '☀' : '☾'}</button>
          <button className="btn-ghost !px-2.5 lg:hidden" aria-expanded={open} aria-controls="mnav" onClick={() => setOpen(!open)}>☰<span className="sr-only">Menu</span></button>
        </div>
        {open && (
          <nav id="mnav" className="border-t px-4 py-3 lg:hidden" style={{ borderColor: 'var(--line)' }} aria-label="Navigasi seluler" onClick={() => setOpen(false)}>
            <div className="grid grid-cols-2 gap-1">
              {nav.map((n) => <NavLink key={n.to} to={n.to} end className={linkCls} style={activeStyle}>{n.label}</NavLink>)}
              {MODULES.map((m) => <NavLink key={m.slug} to={`/${m.slug}`} className={linkCls} style={activeStyle}>Tugas {m.no}</NavLink>)}
            </div>
          </nav>
        )}
      </header>
      <main id="main" key={loc.pathname} className="mx-auto w-full max-w-7xl flex-1 px-4 py-6" style={{ minHeight: '100vh' }}>{children}</main>
      <footer className="border-t" style={{ borderColor: 'var(--line)' }}>
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-6 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <span className="rounded-lg px-2 py-1" style={{ background: '#F6F6F3' }}>
              <img src={`${BASE}img/logo-dsk-light.png`} alt="Logo personal branding Dea Saka Kurnia Putra (@PUTRATALKS)" className="h-10 w-auto" width={143} height={40} />
            </span>
            <div className="text-xs muted">
              <p><strong style={{ color: 'var(--text)' }}>Dea Saka Kurnia Putra</strong> · NIM 261012000111 · S2 TI Reg B 02 (01MKMM002)</p>
              <p>Data Mining — Magister Teknik Informatika, Universitas Pamulang · Dosen: Dr. Ir. Agung Budi Susanto, M.M. · 2026/2027</p>
            </div>
          </div>
          <p className="max-w-md text-xs muted">🔒 {t('privacy')} Tanpa backend, tanpa API key, tanpa pelacakan. Kode MIT; dataset mengikuti lisensi masing-masing.</p>
        </div>
      </footer>
    </div>
  );
}
