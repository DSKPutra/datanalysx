import { matvec, solve, transpose, matmul } from './linalg';

/** Regresi sederhana dengan rumus kuadrat terkecil: b = (nΣXY − ΣXΣY)/(nΣX² − (ΣX)²), a = (ΣY − bΣX)/n. */
export function simpleRegression(x: number[], y: number[]) {
  const n = x.length;
  let sx = 0, sy = 0, sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i]; sxy += x[i] * y[i]; sxx += x[i] * x[i]; }
  const b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const a = (sy - b * sx) / n;
  return { n, sx, sy, sxy, sxx, a, b };
}

/** OLS berganda b = (XᵀX)⁻¹Xᵀy; X sudah memuat kolom 1 untuk intersep. */
export function normalEquation(X: number[][], y: number[]) {
  const Xt = transpose(X);
  const XtX = matmul(Xt, X);
  const Xty = matvec(Xt, y);
  return { XtX, Xty, beta: solve(XtX, Xty) };
}

export function regressionMetrics(y: number[], p: number[]) {
  const n = y.length;
  const mean = y.reduce((a, b) => a + b, 0) / n;
  let sae = 0, sse = 0, sst = 0;
  for (let i = 0; i < n; i++) { const e = y[i] - p[i]; sae += Math.abs(e); sse += e * e; sst += (y[i] - mean) ** 2; }
  return { MAE: sae / n, RMSE: Math.sqrt(sse / n), R2: 1 - sse / sst, sse, sst };
}
