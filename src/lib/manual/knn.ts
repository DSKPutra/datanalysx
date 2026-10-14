export type Metric = 'euclidean' | 'manhattan' | 'chebyshev' | 'minkowski' | 'cosine';

export function distance(a: number[], b: number[], metric: Metric, p = 2): number {
  switch (metric) {
    case 'euclidean': return Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
    case 'manhattan': return a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0);
    case 'chebyshev': return Math.max(...a.map((v, i) => Math.abs(v - b[i])));
    case 'minkowski': return a.reduce((s, v, i) => s + Math.abs(v - b[i]) ** p, 0) ** (1 / p);
    case 'cosine': {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
      return 1 - dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
  }
}

/** Langkah manual KNN: jarak ke semua data latih → urut stabil → K terdekat → voting (berbobot 1/d). */
export function knnManual(q: number[], train: number[][], y: number[], k: number, metric: Metric,
  weights: 'uniform' | 'distance', p = 2, nClasses = Math.max(2, ...y.map((v) => v + 1))) {
  const d = train.map((r) => distance(q, r, metric, p));
  const order = d.map((v, i) => [v, i] as const).sort((A, B) => A[0] - B[0] || A[1] - B[1]).map((t) => t[1]);
  const nn = order.slice(0, k);
  const w = nn.map((i) => (weights === 'distance' ? 1 / Math.max(d[i], 1e-12) : 1));
  const votes: number[] = new Array(nClasses).fill(0);
  nn.forEach((i, j) => { votes[y[i]] += w[j]; });
  const s = votes.reduce((a, b) => a + b, 0);
  const proba = votes.map((v) => v / s);
  return { distances: d, order, neighbors: nn, weights: w, votes, proba, pred: proba.indexOf(Math.max(...proba)) };
}
