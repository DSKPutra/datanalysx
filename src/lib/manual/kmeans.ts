export interface LloydIter { centers: number[][]; labels: number[]; sse: number; newCenters: number[][]; moved: number | null }

const sq = (a: number[], b: number[]) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0);

/** Algoritma Lloyd: assign ke centroid terdekat → update centroid = rata-rata anggota; berhenti bila centroid tetap. */
export function lloyd(X: number[][], C0: number[][], maxIter = 300) {
  let C = C0.map((c) => [...c]);
  const iters: LloydIter[] = [];
  let prev: number[] | null = null;
  for (let it = 0; it < maxIter; it++) {
    const labels = X.map((x) => {
      let best = 0, bd = Infinity;
      C.forEach((c, j) => { const d = sq(x, c); if (d < bd) { bd = d; best = j; } });
      return best;
    });
    const sse = X.reduce((s, x, i) => s + sq(x, C[labels[i]]), 0);
    const Cn = C.map((c, j) => {
      const mem = X.filter((_, i) => labels[i] === j);
      return mem.length ? c.map((_, f) => mem.reduce((s, m) => s + m[f], 0) / mem.length) : c;
    });
    iters.push({ centers: C, labels, sse, newCenters: Cn, moved: prev ? labels.filter((l, i) => l !== prev![i]).length : null });
    const same = Cn.every((c, j) => c.every((v, f) => Math.abs(v - C[j][f]) <= 1e-12));
    prev = labels;
    C = Cn;
    if (same) break;
  }
  const last = iters[iters.length - 1];
  const finalSse = X.reduce((s, x, i) => s + sq(x, C[last.labels[i]]), 0);
  return { iterations: iters, labels: last.labels, centers: C, sse: finalSse };
}
