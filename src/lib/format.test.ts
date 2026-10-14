import { describe, expect, it } from 'vitest';
import { fmt, fmtFixed, int, pct } from './format';

describe('format Indonesia', () => {
  it('memakai koma desimal dan titik ribuan', () => {
    expect(fmtFixed(0.5187, 4)).toBe('0,5187');
    expect(int(7079)).toBe('7.079');
    expect(pct(0.5187)).toBe('51,87%');
    expect(fmt(null)).toBe('–');
  });
});
