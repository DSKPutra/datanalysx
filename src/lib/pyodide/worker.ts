/// <reference lib="webworker" />
/* Web Worker: memuat Pyodide (CDN jsDelivr), paket ilmiah, dan modul dmx, lalu menjalankan
   panggilan fungsi Python dari thread utama. Semua komputasi terjadi di perangkat pengguna. */

const PYODIDE_VERSION = '0.29.5';
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

type Req =
  | { id: number; type: 'init'; modules: Record<string, string>; extra?: string[] }
  | { id: number; type: 'loadCsv'; key: string; text: string }
  | { id: number; type: 'setResource'; name: string; text: string }
  | { id: number; type: 'ensurePackages'; packages: string[] }
  | { id: number; type: 'call'; module: string; fn: string; kwargs: Record<string, unknown> };

let pyodide: any = null;
const loadedExtra = new Set<string>();

function post(msg: unknown) {
  (self as unknown as Worker).postMessage(msg);
}

async function init(modules: Record<string, string>) {
  if (pyodide) return;
  post({ type: 'progress', stage: 'core', message: 'Memuat Python (Pyodide) …' });
  const mod = await import(/* @vite-ignore */ `${INDEX_URL}pyodide.mjs`);
  pyodide = await mod.loadPyodide({ indexURL: INDEX_URL });
  post({ type: 'progress', stage: 'packages', message: 'Memuat numpy, pandas, scikit-learn, scipy …' });
  await pyodide.loadPackage(['numpy', 'pandas', 'scikit-learn', 'scipy', 'micropip'], {
    messageCallback: (m: string) => post({ type: 'progress', stage: 'packages', message: m }),
  });
  post({ type: 'progress', stage: 'modules', message: 'Menyiapkan modul DATANALYSX …' });
  pyodide.FS.mkdirTree('/home/pyodide/dmx');
  for (const [name, src] of Object.entries(modules)) {
    pyodide.FS.writeFile(`/home/pyodide/dmx/${name}`, src);
  }
  await pyodide.runPythonAsync(
    `import sys, warnings\nwarnings.filterwarnings('ignore')\nsys.path.insert(0, '/home/pyodide')\nimport dmx, dmx.common`,
  );
}

async function ensurePackages(pkgs: string[]) {
  const todo = pkgs.filter((p) => !loadedExtra.has(p));
  if (!todo.length) return;
  post({ type: 'progress', stage: 'packages', message: `Memasang ${todo.join(', ')} (micropip) …` });
  const micropip = pyodide.pyimport('micropip');
  await micropip.install(todo);
  todo.forEach((p) => loadedExtra.add(p));
}

// Permintaan diproses berurutan: runPythonAsync bersifat async sehingga tanpa antrean
// variabel global argumen dapat tertimpa oleh permintaan berikutnya.
let queue: Promise<void> = Promise.resolve();
self.onmessage = (ev: MessageEvent<Req>) => {
  queue = queue.then(() => handle(ev.data));
};

async function handle(req: Req) {
  try {
    let result: unknown = null;
    if (req.type === 'init') {
      await init(req.modules);
      if (req.extra?.length) await ensurePackages(req.extra);
      result = { ok: true, version: PYODIDE_VERSION };
    } else if (req.type === 'ensurePackages') {
      await ensurePackages(req.packages);
      result = { ok: true };
    } else if (req.type === 'loadCsv') {
      pyodide.globals.set('__csv_text', req.text);
      pyodide.globals.set('__csv_key', req.key);
      const out = await pyodide.runPythonAsync(
        `import dmx.common as _c\n_c.to_json(_c.load_csv(__csv_key, __csv_text))`,
      );
      pyodide.globals.delete('__csv_text');
      result = JSON.parse(out);
    } else if (req.type === 'setResource') {
      pyodide.globals.set('__res_text', req.text);
      pyodide.globals.set('__res_name', req.name);
      const out = await pyodide.runPythonAsync(
        `import dmx.common as _c\n_c.to_json(_c.set_resource(__res_name, __res_text))`,
      );
      pyodide.globals.delete('__res_text');
      result = JSON.parse(out);
    } else if (req.type === 'call') {
      pyodide.globals.set('__kwargs_json', JSON.stringify(req.kwargs ?? {}));
      const code = `import json, importlib, dmx.common as _c\n_m = importlib.import_module(${JSON.stringify(
        req.module,
      )})\n_c.to_json(getattr(_m, ${JSON.stringify(req.fn)})(**json.loads(__kwargs_json)))`;
      const out = await pyodide.runPythonAsync(code);
      result = JSON.parse(out);
    }
    post({ id: req.id, ok: true, result });
  } catch (err: any) {
    let message = String(err?.message ?? err);
    const m = message.match(/(\w+Error: .*)\s*$/s);
    if (m) message = m[1].trim().split('\n').slice(-1)[0];
    post({ id: req.id, ok: false, error: message });
  }
}
