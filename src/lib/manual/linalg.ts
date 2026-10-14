/* Aljabar linear kecil untuk perhitungan manual (eliminasi Gauss dengan pivot parsial). */
export type Mat = number[][];

export function transpose(A: Mat): Mat {
  return A[0].map((_, j) => A.map((r) => r[j]));
}
export function matmul(A: Mat, B: Mat): Mat {
  return A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
}
export function matvec(A: Mat, x: number[]): number[] {
  return A.map((r) => r.reduce((s, v, k) => s + v * x[k], 0));
}
export function solve(A: Mat, b: number[]): number[] {
  const n = A.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}
export function inverse(A: Mat): Mat {
  const n = A.length;
  const cols = Array.from({ length: n }, (_, j) => solve(A, Array.from({ length: n }, (_, i) => (i === j ? 1 : 0))));
  return transpose(cols);
}
