/** Model laporan netral-format: dikumpulkan dari halaman, lalu dirender ke .docx atau .pptx. */
export type Block =
  | { kind: 'heading'; text: string; level: 1 | 2 }
  | { kind: 'para'; text: string }
  | { kind: 'bullets'; items: string[] }
  | { kind: 'note'; title?: string; text: string; tone: 'info' | 'warn' | 'ok' }
  | { kind: 'stats'; items: { label: string; value: string; hint?: string }[] }
  | { kind: 'table'; head: string[]; rows: string[][]; caption?: string }
  | { kind: 'image'; title?: string; dataUrl: string; width: number; height: number };

export interface ReportDoc {
  title: string;
  subtitle?: string;
  author: string;
  date: Date;
  source: string;
  blocks: Block[];
}

/** Nama berkas aman: huruf kecil, tanda hubung, tanpa karakter khusus. */
export function fileStem(...parts: string[]): string {
  return parts.join('-').normalize('NFKD').replace(/[̀-ͯ]/g, '')
    .toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 80) || 'laporan';
}

export function fmtDate(d: Date): string {
  return d.toLocaleDateString('id-ID', { day: 'numeric', month: 'long', year: 'numeric' });
}

/** Bagi array menjadi potongan berukuran n (dipakai untuk memecah tabel/butir ke beberapa slide). */
export function chunk<T>(xs: T[], n: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < xs.length; i += n) out.push(xs.slice(i, i + n));
  return out.length ? out : [[]];
}
