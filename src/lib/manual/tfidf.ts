/** TF-IDF manual ala scikit-learn: TF mentah, IDF smooth = ln((1+N)/(1+df)) + 1 atau klasik = ln(N/df) + 1, normalisasi L2. */
export function tfidf(docs: string[], smooth = true, l2 = true) {
  const toks = docs.map((d) => d.split(/\s+/).filter(Boolean));
  const vocab = [...new Set(toks.flat())].sort();
  const N = docs.length;
  const tf = toks.map((t) => vocab.map((w) => t.filter((x) => x === w).length));
  const df = vocab.map((_, j) => tf.filter((r) => r[j] > 0).length);
  const idf = df.map((d) => (smooth ? Math.log((1 + N) / (1 + d)) : Math.log(N / d)) + 1);
  const raw = tf.map((r) => r.map((v, j) => v * idf[j]));
  const norms = raw.map((r) => Math.sqrt(r.reduce((s, v) => s + v * v, 0)));
  const matrix = l2 ? raw.map((r, i) => r.map((v) => (norms[i] ? v / norms[i] : 0))) : raw;
  return { vocab, tf, df, idf, raw, norms, matrix };
}
