/** Gaussian Naïve Bayes manual: prior → log-likelihood → log-sum-exp → posterior → MAP. */
export function gaussianLogPdf(x: number, mu: number, v: number) {
  return -0.5 * Math.log(2 * Math.PI * v) - ((x - mu) ** 2) / (2 * v);
}
export function gnbPosterior(x: number[], theta: number[][], variance: number[][], prior: number[]) {
  const perFeature = theta.map((th, c) => x.map((xi, i) => {
    const lp = gaussianLogPdf(xi, th[i], variance[c][i]);
    return { mu: th[i], sigma: Math.sqrt(variance[c][i]), pdf: Math.exp(lp), logPdf: lp };
  }));
  const logPrior = prior.map(Math.log);
  const sumLog = perFeature.map((f) => f.reduce((s, r) => s + r.logPdf, 0));
  const joint = sumLog.map((s, c) => s + logPrior[c]);
  const m = Math.max(...joint);
  const ex = joint.map((j) => Math.exp(j - m));
  const z = ex.reduce((a, b) => a + b, 0);
  const posterior = ex.map((e) => e / z);
  return { perFeature, logPrior, sumLog, joint, posterior, pred: posterior.indexOf(Math.max(...posterior)) };
}
