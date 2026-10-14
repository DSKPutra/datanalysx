export function entropy(counts: number[]) {
  const n = counts.reduce((a, b) => a + b, 0);
  return -counts.filter((c) => c > 0).reduce((s, c) => s + (c / n) * Math.log2(c / n), 0);
}
export function gini(counts: number[]) {
  const n = counts.reduce((a, b) => a + b, 0);
  return 1 - counts.reduce((s, c) => s + (c / n) ** 2, 0);
}
/** Information gain pemisahan biner: H(akar) − Σ (nᵢ/n)·H(anakᵢ). */
export function infoGain(left: number[], right: number[]) {
  const root = left.map((v, i) => v + right[i]);
  const nl = left.reduce((a, b) => a + b, 0), nr = right.reduce((a, b) => a + b, 0), n = nl + nr;
  const H = entropy(root), Hl = entropy(left), Hr = entropy(right);
  const after = (nl / n) * Hl + (nr / n) * Hr;
  const wl = nl / n, wr = nr / n;
  const si = -(wl > 0 ? wl * Math.log2(wl) : 0) - (wr > 0 ? wr * Math.log2(wr) : 0);
  return { H, Hl, Hr, after, ig: H - after, splitInfo: si, gainRatio: si > 0 ? (H - after) / si : 0 };
}
