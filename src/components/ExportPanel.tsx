import { useId, useRef, useState } from 'react';
import { fileStem, type ReportDoc } from '../lib/export/model';
import { plotToImage } from './Plot';
import { Toggle } from './ui';

const NAME_KEY = 'dmx-export-name';
const readName = () => { try { return localStorage.getItem(NAME_KEY) ?? ''; } catch { return ''; } };
const saveName = (v: string) => { try { localStorage.setItem(NAME_KEY, v); } catch { /* abaikan */ } };

function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/** Ekspor isi halaman yang sedang tampil ke laporan Word (.docx) atau presentasi PowerPoint (.pptx). */
export default function ExportPanel({ defaultTitle, subtitle }: { defaultTitle: string; subtitle?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const id = useId();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(readName);
  const [title, setTitle] = useState(defaultTitle);
  const [charts, setCharts] = useState(true);
  const [busy, setBusy] = useState<'' | 'docx' | 'pptx'>('');
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  async function run(fmt: 'docx' | 'pptx') {
    const author = name.trim();
    if (!author) { setMsg({ ok: false, text: 'Isi nama penyusun terlebih dahulu.' }); return; }
    const root = ref.current?.closest('[data-export-root]');
    if (!root) return;
    saveName(author);
    setBusy(fmt); setMsg(null);
    try {
      const { collectBlocks } = await import('../lib/export/collect');
      const { blocks, dataset } = await collectBlocks(root, charts ? plotToImage : null);
      if (!blocks.length) throw new Error('Belum ada konten untuk diekspor.');
      const doc: ReportDoc = {
        title: title.trim() || defaultTitle, subtitle, author, date: new Date(),
        source: `${window.location.href}${dataset ? ` — data: ${dataset}` : ''}`, blocks,
      };
      const blob = fmt === 'docx'
        ? await (await import('../lib/export/docx')).toDocx(doc)
        : await (await import('../lib/export/pptx')).toPptx(doc);
      const file = `${fileStem('datanalysx', doc.title, author)}.${fmt}`;
      download(blob, file);
      setMsg({ ok: true, text: `${file} diunduh (${blocks.length} blok konten).` });
    } catch (e: any) {
      setMsg({ ok: false, text: `Gagal membuat berkas: ${e?.message ?? e}` });
    } finally { setBusy(''); }
  }

  return (
    <div ref={ref} data-export-skip className="max-w-3xl">
      <button className="btn-ghost" aria-expanded={open} aria-controls={`${id}-panel`} onClick={() => setOpen(!open)}>
        ⬇ Ekspor laporan (.docx / .pptx)
      </button>
      {open && (
        <div id={`${id}-panel`} className="card mt-2 space-y-3">
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <label htmlFor={`${id}-name`} className="label">Nama penyusun *</label>
              <input id={`${id}-name`} className="input" value={name} autoComplete="name" placeholder="mis. Dea Saka Kurnia Putra"
                onChange={(e) => setName(e.target.value)} />
            </div>
            <div>
              <label htmlFor={`${id}-title`} className="label">Judul laporan</label>
              <input id={`${id}-title`} className="input" value={title} onChange={(e) => setTitle(e.target.value)} />
            </div>
          </div>
          <Toggle label="Sertakan grafik (dirender ulang berlatar putih)" checked={charts} onChange={setCharts} />
          <p className="text-xs muted">Berisi konten yang sedang tampil: tab aktif serta hasil yang sudah dihitung (jalankan model terlebih dahulu). Berkas dibuat di browser Anda; tidak ada data yang dikirim ke server.</p>
          <div className="flex flex-wrap gap-2">
            <button className="btn-primary" disabled={!!busy} onClick={() => run('docx')}>{busy === 'docx' ? 'Menyusun .docx …' : 'Unduh laporan .docx'}</button>
            <button className="btn-primary" disabled={!!busy} onClick={() => run('pptx')}>{busy === 'pptx' ? 'Menyusun .pptx …' : 'Unduh slide .pptx'}</button>
          </div>
          {msg && <p role="status" className="text-sm" style={{ color: msg.ok ? 'var(--up)' : 'var(--down)' }}>{msg.text}</p>}
        </div>
      )}
    </div>
  );
}
