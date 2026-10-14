/* Format angka Indonesia (koma desimal, titik ribuan). */
export function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-4 || a >= 1e9)) return v.toExponential(Math.min(digits, 3)).replace('.', ',').replace('e', '×10^');
  return v.toLocaleString('id-ID', { minimumFractionDigits: 0, maximumFractionDigits: digits });
}
export function fmtFixed(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  return v.toLocaleString('id-ID', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
export function pct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  return `${(v * 100).toLocaleString('id-ID', { minimumFractionDigits: digits, maximumFractionDigits: digits })}%`;
}
export function int(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  return Math.round(v).toLocaleString('id-ID');
}
