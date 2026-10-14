import { useEffect, useState } from 'react';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';

hljs.registerLanguage('python', python);

export default function CodeViewer({ files }: { files: { label: string; path: string; note?: string }[] }) {
  const [active, setActive] = useState(0);
  const [code, setCode] = useState<string>('');
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setCode(''); setErr(null);
    fetch(import.meta.env.BASE_URL + files[active].path)
      .then((r) => { if (!r.ok) throw new Error(r.statusText); return r.text(); })
      .then(setCode).catch((e) => setErr(String(e)));
  }, [active, files]);
  const html = code ? hljs.highlight(code, { language: 'python' }).value : '';
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Berkas kode">
        {files.map((f, i) => (
          <button key={f.path} role="tab" aria-selected={i === active} className={i === active ? 'btn-primary' : 'btn-ghost'} onClick={() => setActive(i)}>{f.label}</button>
        ))}
        <a className="btn-ghost" href={import.meta.env.BASE_URL + files[active].path} download>Unduh .py</a>
      </div>
      {files[active].note && <p className="text-sm muted">{files[active].note}</p>}
      {err && <p className="text-sm">Gagal memuat kode: {err}</p>}
      <pre className="code-block max-h-[70vh] overflow-auto rounded-xl p-4 text-xs leading-relaxed" tabIndex={0}>
        <code dangerouslySetInnerHTML={{ __html: html || 'Memuat …' }} />
      </pre>
    </div>
  );
}
