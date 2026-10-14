import { useCallback, useEffect, useRef, useState } from 'react';

export function useAsync<T>() {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const token = useRef(0);
  const run = useCallback(async (fn: () => Promise<T>) => {
    const my = ++token.current;
    setLoading(true);
    setError(null);
    try {
      const r = await fn();
      if (my === token.current) setData(r);
      return r;
    } catch (e: any) {
      if (my === token.current) setError(String(e?.message ?? e));
      return null;
    } finally {
      if (my === token.current) setLoading(false);
    }
  }, []);
  return { data, error, loading, run, setData };
}

const jsonCache = new Map<string, Promise<any>>();
export function fetchJson<T>(url: string): Promise<T> {
  if (!jsonCache.has(url)) {
    jsonCache.set(url, fetch(import.meta.env.BASE_URL + url).then((r) => {
      if (!r.ok) throw new Error(`Gagal memuat ${url}`);
      return r.json();
    }));
  }
  return jsonCache.get(url)!;
}

export function useJson<T>(url: string) {
  const s = useAsync<T>();
  const { run } = s;
  useEffect(() => { run(() => fetchJson<T>(url)); }, [url, run]);
  return s;
}
