import { useEffect, useState } from 'react';

export function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.dataset.theme === 'dark');
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    try { localStorage.setItem('dmx-theme', dark ? 'dark' : 'light'); } catch { /* abaikan */ }
    window.dispatchEvent(new Event('dmx-theme'));
  }, [dark]);
  return { dark, toggle: () => setDark((d) => !d) };
}

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
/** Palet ramah buta warna (Okabe–Ito) — dibaca dari token CSS agar konsisten di semua grafik. */
export function palette() {
  return {
    up: cssVar('--up'), down: cssVar('--down'), neutral: cssVar('--neutral'),
    c: [cssVar('--c1'), cssVar('--c2'), cssVar('--c3'), '#CC79A7', '#56B4E9', '#F0E442', '#999999', '#D55E00', '#0072B2', '#009E73'],
    brand: cssVar('--brand'), brand2: cssVar('--brand-2'), gold: cssVar('--gold'), text: cssVar('--text'),
    muted: cssVar('--muted'), line: cssVar('--line'), surface: cssVar('--surface'),
  };
}
