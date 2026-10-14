/* Klien RPC untuk worker Pyodide. Worker dibuat secara lazy pada panggilan pertama
   (hanya saat tab Lab/Manual dibuka), lalu dipakai bersama oleh seluruh halaman. */
import { useEffect, useState } from 'react';

export type PyStatus = 'idle' | 'loading' | 'ready' | 'error';
interface Progress {
  status: PyStatus;
  message: string;
  startedAt?: number;
  readyAt?: number;
}

type Pending = { resolve: (v: any) => void; reject: (e: Error) => void };

let worker: Worker | null = null;
let seq = 0;
const pending = new Map<number, Pending>();
let initPromise: Promise<void> | null = null;
const loadedData = new Map<string, string>(); // key -> sumber (url / nama berkas)
let progress: Progress = { status: 'idle', message: 'Python belum dimuat' };
const listeners = new Set<(p: Progress) => void>();

function setProgress(p: Partial<Progress>) {
  progress = { ...progress, ...p };
  listeners.forEach((l) => l(progress));
}

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (ev) => {
    const msg = ev.data;
    if (msg.type === 'progress') {
      setProgress({ message: msg.message });
      return;
    }
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    if (msg.ok) p.resolve(msg.result);
    else p.reject(new Error(msg.error));
  };
  worker.onerror = (e) => setProgress({ status: 'error', message: `Worker gagal: ${e.message}` });
  return worker;
}

function send<T>(payload: Record<string, unknown>): Promise<T> {
  const id = ++seq;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, ...payload });
  });
}

export function initPython(): Promise<void> {
  if (!initPromise) {
    setProgress({ status: 'loading', message: 'Memulai Python …', startedAt: Date.now() });
    initPromise = import('./modules')
      .then(({ pythonModules }) => send({ type: 'init', modules: pythonModules }))
      .then(() => setProgress({ status: 'ready', message: 'Python siap', readyAt: Date.now() }))
      .catch((e: Error) => {
        setProgress({ status: 'error', message: e.message });
        initPromise = null;
        throw e;
      });
  }
  return initPromise;
}

export async function ensurePackages(packages: string[]) {
  await initPython();
  await send({ type: 'ensurePackages', packages });
}

/** Muat CSV dari URL (dataset contoh) sekali saja per kunci. */
export async function loadDataUrl(key: string, url: string): Promise<void> {
  await initPython();
  if (loadedData.get(key) === url) return;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Gagal mengunduh ${url} (${res.status})`);
  const text = await res.text();
  await send({ type: 'loadCsv', key, text });
  loadedData.set(key, url);
}

/** Muat CSV dari berkas unggahan pengguna (tidak pernah meninggalkan browser). */
export async function loadDataText(key: string, text: string, source: string) {
  await initPython();
  const info = await send<{ n_rows: number; n_cols: number; columns: string[] }>({ type: 'loadCsv', key, text });
  loadedData.set(key, source);
  return info;
}

export function dataSource(key: string) {
  return loadedData.get(key);
}

export async function setResource(name: string, text: string) {
  await initPython();
  return send({ type: 'setResource', name, text });
}

export async function py<T = any>(module: string, fn: string, kwargs: Record<string, unknown> = {}): Promise<T> {
  await initPython();
  return send<T>({ type: 'call', module, fn, kwargs });
}

export function usePyStatus(): Progress {
  const [p, setP] = useState(progress);
  useEffect(() => {
    listeners.add(setP);
    return () => {
      listeners.delete(setP);
    };
  }, []);
  return p;
}
