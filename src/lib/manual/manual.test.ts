/* Setiap implementasi manual TypeScript harus identik dengan scikit-learn (toleransi < 1e-8). */
import { describe, expect, it } from 'vitest';
import entropyFx from './__fixtures__/entropy.json';
import gnbFx from './__fixtures__/gnb.json';
import kmFx from './__fixtures__/kmeans.json';
import knnFx from './__fixtures__/knn.json';
import olsFx from './__fixtures__/ols.json';
import simpleFx from './__fixtures__/simple.json';
import tfFx from './__fixtures__/tfidf.json';
import { infoGain } from './entropy';
import { gnbPosterior } from './gnb';
import { lloyd } from './kmeans';
import { knnManual } from './knn';
import { normalEquation, simpleRegression } from './ols';
import { tfidf } from './tfidf';
import { logisticProba, sigmoid } from './logistic';

const TOL = 1e-8;
const maxDiff = (a: number[], b: number[]) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

describe('perhitungan manual = scikit-learn', () => {
  it('OLS persamaan normal (Studi Kasus 01)', () => {
    const { beta } = normalEquation(olsFx.X, olsFx.y);
    const pred = olsFx.X.map((r) => r.reduce((s, v, i) => s + v * beta[i], 0));
    expect(maxDiff(pred, olsFx.pred_sklearn)).toBeLessThan(TOL);
  });
  it('regresi sederhana rumus kuadrat terkecil (Studi Kasus 03)', () => {
    const r = simpleRegression(simpleFx.x, simpleFx.y);
    expect(Math.abs(r.a - simpleFx.a)).toBeLessThan(TOL);
    expect(Math.abs(r.b - simpleFx.b)).toBeLessThan(TOL);
  });
  it('sigmoid regresi logistik', () => {
    expect(sigmoid(0)).toBe(0.5);
    const r = logisticProba([1, 2], [0, 0], [1, 1], [0.5, -0.25], 0.1);
    expect(r.proba).toBeCloseTo(1 / (1 + Math.exp(-0.1)), 12);
  });
  it('entropy & information gain kata "for" (Studi Kasus 05)', () => {
    const r = infoGain(entropyFx.left, entropyFx.right);
    expect(Math.abs(r.H - entropyFx.H_root)).toBeLessThan(TOL);
    expect(Math.abs(r.after - entropyFx.H_after)).toBeLessThan(TOL);
  });
  it('posterior Gaussian NB 21-02-2023 (Studi Kasus 06)', () => {
    const r = gnbPosterior(gnbFx.x, gnbFx.theta, gnbFx.var, gnbFx.prior);
    expect(maxDiff(r.posterior, gnbFx.proba)).toBeLessThan(TOL);
    expect(r.posterior[0]).toBeCloseTo(0.9024, 4);
  });
  it('voting KNN cosine berbobot jarak (Studi Kasus 07)', () => {
    const r = knnManual(knnFx.query, knnFx.train, knnFx.y, 5, 'cosine', 'distance');
    expect([...r.neighbors].sort()).toEqual([...knnFx.neighbors].sort());
    expect(maxDiff(r.proba, knnFx.proba)).toBeLessThan(TOL);
  });
  it('algoritma Lloyd 10 hari (Studi Kasus 08)', () => {
    const r = lloyd(kmFx.points, kmFx.init);
    expect(r.labels).toEqual(kmFx.labels);
    expect(maxDiff(r.centers.flat(), kmFx.centers.flat())).toBeLessThan(TOL);
    expect(Math.abs(r.sse - kmFx.inertia)).toBeLessThan(TOL);
  });
  it('TF-IDF smooth & klasik + L2 (Studi Kasus 09)', () => {
    for (const [smooth, ref] of [[true, tfFx.smooth], [false, tfFx.classic]] as const) {
      const r = tfidf(tfFx.docs, smooth);
      expect(r.vocab).toEqual(ref.vocab);
      expect(maxDiff(r.matrix.flat(), ref.matrix.flat())).toBeLessThan(TOL);
      expect(maxDiff(r.idf, ref.idf)).toBeLessThan(TOL);
    }
  });
});
