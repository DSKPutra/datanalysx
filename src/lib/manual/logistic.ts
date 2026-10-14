export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** P(naik) = σ(b₀ + Σ βⱼ·(xⱼ − μⱼ)/σⱼ). */
export function logisticProba(x: number[], mean: number[], std: number[], beta: number[], intercept: number) {
  const z = x.map((v, j) => (v - mean[j]) / std[j]);
  const terms = z.map((v, j) => v * beta[j]);
  const logit = intercept + terms.reduce((a, b) => a + b, 0);
  return { z, terms, logit, proba: sigmoid(logit) };
}
