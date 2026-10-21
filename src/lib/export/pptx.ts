import PptxGenJS from 'pptxgenjs';
import { chunk, fmtDate, type Block, type ReportDoc } from './model';

const BRAND = '0B4F3A', BRAND2 = '0B7A53', GOLD = 'E9C46A', TEXT = '1F2A26', MUTED = '5B6660', LINE = 'DDE2DC';
const W = 13.333, H = 7.5, X = 0.6, CONTENT_W = W - 2 * X, TOP = 1.35, CONTENT_H = H - TOP - 0.7;
const ROWS_PER_SLIDE = 12, MAX_TABLE_SLIDES = 4, ITEMS_PER_SLIDE = 7, MAX_COLS = 8;
const TONE = { info: BRAND2, warn: 'B7791F', ok: '2F855A' } as const;

const cut = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

type Item = { text: string; note?: { title?: string; tone: keyof typeof TONE } };

/** Susun presentasi PowerPoint (.pptx, 16:9) dari model laporan. */
export async function toPptx(doc: ReportDoc): Promise<Blob> {
  const pptx = new PptxGenJS();
  pptx.layout = 'LAYOUT_WIDE';
  pptx.author = doc.author;
  pptx.title = doc.title;
  pptx.company = 'DATANALYSX';
  pptx.defineSlideMaster({
    title: 'DMX',
    background: { color: 'FFFFFF' },
    objects: [
      { rect: { x: 0, y: 0, w: W, h: 0.12, fill: { color: BRAND } } },
      { text: { text: `DATANALYSX · ${cut(doc.title, 70)} · ${doc.author}`, options: { x: X, y: H - 0.45, w: 10, h: 0.3, fontSize: 9, color: MUTED } } },
    ],
    slideNumber: { x: W - 1.1, y: H - 0.45, w: 0.6, h: 0.3, fontSize: 9, color: MUTED, align: 'right' },
  });

  // Slide judul
  const cover = pptx.addSlide();
  cover.background = { color: BRAND };
  cover.addText('DATANALYSX · LAPORAN ANALISIS', { x: X, y: 1.3, w: CONTENT_W, h: 0.4, fontSize: 14, bold: true, color: GOLD, charSpacing: 3 });
  cover.addText(doc.title, { x: X, y: 1.8, w: CONTENT_W, h: 1.6, fontSize: 40, bold: true, color: 'FFFFFF', fontFace: 'Georgia', valign: 'top', fit: 'shrink' });
  if (doc.subtitle) cover.addText(cut(doc.subtitle, 260), { x: X, y: 3.5, w: CONTENT_W * 0.8, h: 1.1, fontSize: 16, color: 'DDEFE6', valign: 'top' });
  cover.addText([
    { text: 'Disusun oleh: ', options: { bold: true } }, { text: doc.author, options: { breakLine: true } },
    { text: 'Tanggal: ', options: { bold: true } }, { text: fmtDate(doc.date), options: { breakLine: true } },
    { text: 'Sumber: ', options: { bold: true } }, { text: cut(doc.source, 120) },
  ], { x: X, y: 5.0, w: CONTENT_W, h: 1.4, fontSize: 14, color: 'FFFFFF', valign: 'top' });

  let section = doc.title, sub = '';
  const newSlide = (cont = false) => {
    const s = pptx.addSlide({ masterName: 'DMX' });
    s.addText(cut(section, 80) + (cont ? ' (lanjutan)' : ''), { x: X, y: 0.35, w: CONTENT_W, h: 0.6, fontSize: 26, bold: true, color: BRAND, fontFace: 'Georgia' });
    if (sub) s.addText(cut(sub, 120), { x: X, y: 0.92, w: CONTENT_W, h: 0.35, fontSize: 14, color: BRAND2 });
    return s;
  };

  let pending: Item[] = [];
  const flushText = () => {
    chunk(pending, ITEMS_PER_SLIDE).forEach((items, i) => {
      if (!items.length) return;
      const s = newSlide(i > 0);
      s.addText(items.map((it) => it.note
        ? { text: `${it.note.title ? `${it.note.title}: ` : ''}${cut(it.text, 320)}`, options: { bullet: { code: '25A0' }, color: TONE[it.note.tone], breakLine: true, paraSpaceAfter: 8 } }
        : { text: cut(it.text, 320), options: { bullet: true, breakLine: true, paraSpaceAfter: 8 } }),
      { x: X, y: TOP, w: CONTENT_W, h: CONTENT_H, fontSize: 16, color: TEXT, valign: 'top', fit: 'shrink' });
    });
    pending = [];
  };

  const addBlock = (b: Block) => {
    if (b.kind === 'para') { pending.push({ text: b.text }); return; }
    if (b.kind === 'bullets') { b.items.forEach((t) => pending.push({ text: t })); return; }
    if (b.kind === 'note') { pending.push({ text: b.text, note: { title: b.title, tone: b.tone } }); return; }
    flushText();
    if (b.kind === 'heading') {
      if (b.level === 1) { section = b.text; sub = ''; } else sub = b.text;
      return;
    }
    if (b.kind === 'stats') {
      chunk(b.items, 8).forEach((items, i) => {
        const s = newSlide(i > 0);
        const cols = Math.min(4, items.length), gap = 0.25, cw = (CONTENT_W - gap * (cols - 1)) / cols, ch = 2.1;
        items.forEach((it, j) => {
          const x = X + (j % cols) * (cw + gap), y = TOP + 0.2 + Math.floor(j / cols) * (ch + gap);
          s.addShape(pptx.ShapeType.roundRect, { x, y, w: cw, h: ch, fill: { color: 'F3F6F2' }, line: { color: LINE }, rectRadius: 0.1 });
          s.addText(cut(it.label.toUpperCase(), 40), { x: x + 0.15, y: y + 0.12, w: cw - 0.3, h: 0.4, fontSize: 11, bold: true, color: MUTED });
          s.addText(cut(it.value, 28), { x: x + 0.15, y: y + 0.55, w: cw - 0.3, h: 0.8, fontSize: 26, bold: true, color: BRAND, fit: 'shrink' });
          if (it.hint) s.addText(cut(it.hint, 90), { x: x + 0.15, y: y + 1.35, w: cw - 0.3, h: 0.6, fontSize: 10, color: MUTED, valign: 'top' });
        });
      });
      return;
    }
    if (b.kind === 'table') {
      const head = b.head.slice(0, MAX_COLS);
      const pages = chunk(b.rows, ROWS_PER_SLIDE).slice(0, MAX_TABLE_SLIDES);
      pages.forEach((rows, i) => {
        const s = newSlide(i > 0);
        const hdr = head.map((h) => ({ text: cut(h, 30), options: { bold: true, color: 'FFFFFF', fill: { color: BRAND } } }));
        const body = rows.map((r) => head.map((_, j) => ({ text: cut(r[j] ?? '', 40) })));
        s.addTable([hdr, ...body], { x: X, y: TOP, w: CONTENT_W, fontSize: head.length > 6 ? 10 : 12, color: TEXT, border: { type: 'solid', pt: 0.5, color: LINE }, autoPage: false });
        const shown = Math.min(b.rows.length, ROWS_PER_SLIDE * MAX_TABLE_SLIDES);
        const notes = [b.caption, i === pages.length - 1 && b.rows.length > shown ? `${b.rows.length - shown} baris lain tidak ditampilkan` : '', b.head.length > MAX_COLS ? `${b.head.length - MAX_COLS} kolom lain tidak ditampilkan` : ''].filter(Boolean).join(' · ');
        if (notes) s.addText(notes, { x: X, y: H - 0.85, w: CONTENT_W, h: 0.3, fontSize: 10, italic: true, color: MUTED });
      });
      return;
    }
    if (b.kind === 'image') {
      const s = newSlide();
      const boxH = CONTENT_H - (b.title ? 0.4 : 0);
      const scale = Math.min(CONTENT_W / b.width, boxH / b.height);
      const w = b.width * scale, h = b.height * scale;
      s.addImage({ data: b.dataUrl, x: X + (CONTENT_W - w) / 2, y: TOP, w, h });
      if (b.title) s.addText(cut(b.title, 120), { x: X, y: TOP + h + 0.05, w: CONTENT_W, h: 0.35, fontSize: 12, italic: true, color: MUTED, align: 'center' });
    }
  };
  doc.blocks.forEach(addBlock);
  flushText();

  const end = pptx.addSlide();
  end.background = { color: BRAND };
  end.addText('Terima kasih', { x: X, y: 2.6, w: CONTENT_W, h: 1, fontSize: 40, bold: true, color: 'FFFFFF', fontFace: 'Georgia' });
  end.addText(`${doc.author} · ${fmtDate(doc.date)} · dibuat dengan DATANALYSX`, { x: X, y: 3.7, w: CONTENT_W, h: 0.5, fontSize: 16, color: 'DDEFE6' });

  return (await pptx.write({ outputType: 'blob' })) as Blob;
}
