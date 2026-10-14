import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';

export type Lang = 'id' | 'en';
const dict = {
  home: ['Beranda', 'Home'],
  modules: ['Studi Kasus', 'Case Studies'],
  tools: ['Alat Analisis', 'Analysis Tools'],
  explore: ['Eksplorasi', 'Explore'],
  regression: ['Regresi', 'Regression'],
  classification: ['Klasifikasi', 'Classification'],
  clustering: ['Clustering', 'Clustering'],
  text: ['Text Mining', 'Text Mining'],
  guide: ['Panduan', 'Guide'],
  regLab: ['Regression Lab', 'Regression Lab'],
  dataLab: ['Data Lab', 'Data Lab'],
  compare: ['Perbandingan', 'Comparison'],
  lessons: ['Pelajaran Metodologi', 'Methodology Lessons'],
  about: ['Tentang & Referensi', 'About & References'],
  theme: ['Ganti tema', 'Toggle theme'],
  privacy: ['Data unggahan diproses di browser Anda dan tidak pernah dikirim ke server mana pun.', 'Uploaded data is processed in your browser and never sent to any server.'],
  tabSummary: ['Ringkasan', 'Summary'],
  tabData: ['Data', 'Data'],
  tabReport: ['Hasil Laporan', 'Report Results'],
  tabLab: ['Lab Live', 'Live Lab'],
  tabManual: ['Perhitungan Manual', 'Manual Calculation'],
  tabCode: ['Kode Python', 'Python Code'],
  tabNotes: ['Catatan Kritis', 'Critical Notes'],
  loadPython: ['Muat Python & jalankan', 'Load Python & run'],
  run: ['Jalankan', 'Run'],
  notInvest: ['Bukan saran investasi.', 'Not investment advice.'],
} as const;
export type Key = keyof typeof dict;

const Ctx = createContext<{ lang: Lang; setLang: (l: Lang) => void; t: (k: Key) => string }>({
  lang: 'id', setLang: () => {}, t: (k) => dict[k][0],
});

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(() => {
    try { return (localStorage.getItem('dmx-lang') as Lang) || 'id'; } catch { return 'id'; }
  });
  useEffect(() => {
    try { localStorage.setItem('dmx-lang', lang); } catch { /* abaikan */ }
    document.documentElement.lang = lang;
  }, [lang]);
  const t = (k: Key) => dict[k][lang === 'id' ? 0 : 1];
  return <Ctx.Provider value={{ lang, setLang, t }}>{children}</Ctx.Provider>;
}
export const useI18n = () => useContext(Ctx);
