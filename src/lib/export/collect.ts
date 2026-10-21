import type { Block } from './model';

type PlotImage = (el: Element) => Promise<{ dataUrl: string; width: number; height: number; title?: string } | null>;
type Raw = Block | { kind: 'plot'; el: Element };

const SKIP_TAGS = new Set(['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'LABEL', 'FIELDSET', 'NAV', 'SCRIPT', 'STYLE', 'PRE', 'OPTION', 'SVG', 'CANVAS', 'H1', 'TEMPLATE']);

export const clean = (s: string | null | undefined) => (s ?? '').replace(/\s+/g, ' ').trim();

const BLOCK_TAGS = new Set(['DIV', 'P', 'LI', 'BR', 'TR', 'TD', 'TH', 'DT', 'DD', 'SECTION', 'HEADER', 'H2', 'H3', 'H4']);
/** Seperti textContent, tetapi memberi spasi di antara elemen blok (mis. "94" + "TN" → "94 TN"). */
function textOf(n: Node | null | undefined): string {
  if (!n) return '';
  if (n.nodeType === 3) return n.textContent ?? '';
  if (n.nodeType !== 1) return '';
  const inner = Array.from(n.childNodes).map(textOf).join('');
  return BLOCK_TAGS.has((n as Element).tagName.toUpperCase()) ? ` ${inner} ` : inner;
}
const txt = (n: Node | null | undefined) => clean(textOf(n));

function skipped(el: Element): boolean {
  if (SKIP_TAGS.has(el.tagName.toUpperCase())) return true;
  if (el.hasAttribute('data-export-skip') || el.hasAttribute('hidden') || el.getAttribute('aria-hidden') === 'true') return true;
  const role = el.getAttribute('role');
  return role === 'tablist' || role === 'status';
}

function tableBlock(t: HTMLTableElement): Block | null {
  const cells = (tr: Element) => Array.from(tr.children).filter((c) => c.tagName === 'TH' || c.tagName === 'TD').map((c) => txt(c));
  const trs = Array.from(t.querySelectorAll('tr'));
  if (!trs.length) return null;
  const headRow = t.querySelector('thead tr');
  const head = headRow ? cells(headRow) : cells(trs[0]);
  const body = trs.filter((tr) => tr !== (headRow ?? trs[0])).map(cells).filter((r) => r.some((c) => c));
  const caption = clean(t.querySelector('caption')?.textContent) || undefined;
  return { kind: 'table', head, rows: body, caption };
}

function walk(el: Element, depth: number, out: Raw[]): void {
  if (skipped(el)) return;
  const tag = el.tagName.toUpperCase();
  if (el.hasAttribute('data-export-plot')) { out.push({ kind: 'plot', el }); return; }
  if (el.hasAttribute('data-export-stat')) {
    const item = {
      label: clean(el.querySelector('[data-export-label]')?.textContent),
      value: clean(el.querySelector('[data-export-value]')?.textContent),
      hint: clean(el.querySelector('[data-export-hint]')?.textContent) || undefined,
    };
    const last = out[out.length - 1];
    if (last?.kind === 'stats') last.items.push(item); else out.push({ kind: 'stats', items: [item] });
    return;
  }
  if (el.hasAttribute('data-export-callout')) {
    const titleEl = el.querySelector('[data-export-title]');
    const title = clean(titleEl?.textContent).replace(/^⚠\s*/, '') || undefined;
    const body = el.querySelector('.prose-dmx') ?? el;
    const text = txt(body);
    if (text) out.push({ kind: 'note', title, text, tone: (el.getAttribute('data-export-callout') as 'info' | 'warn' | 'ok') || 'info' });
    return;
  }
  if (el.hasAttribute('data-export-section')) {
    const h = el.querySelector('h2, h3');
    const title = clean(h?.textContent);
    if (title) out.push({ kind: 'heading', text: title, level: depth === 0 ? 1 : 2 });
    for (const c of Array.from(el.children)) walk(c, depth + 1, out);
    return;
  }
  if (tag === 'H2' || tag === 'H3' || tag === 'H4' || tag === 'H5') {
    // Judul bagian sudah dicetak oleh <section>; judul lepas menjadi subjudul.
    if (el.parentElement?.closest('[data-export-section]') && el.parentElement.parentElement?.hasAttribute('data-export-section')) return;
    const text = txt(el);
    if (text) out.push({ kind: 'heading', text, level: 2 });
    return;
  }
  if (tag === 'TABLE') { const b = tableBlock(el as HTMLTableElement); if (b) out.push(b); return; }
  if (tag === 'UL' || tag === 'OL') {
    const items = Array.from(el.children).filter((c) => c.tagName === 'LI' && !skipped(c)).map((c) => txt(c)).filter(Boolean);
    if (items.length) out.push({ kind: 'bullets', items });
    return;
  }
  if (tag === 'DL') {
    const items: string[] = [];
    for (const dt of Array.from(el.querySelectorAll('dt'))) items.push(`${txt(dt)}: ${txt(dt.nextElementSibling)}`);
    if (items.length) out.push({ kind: 'bullets', items });
    return;
  }
  if (tag === 'P' || tag === 'FIGCAPTION' || tag === 'BLOCKQUOTE') {
    const text = txt(el);
    if (text) out.push({ kind: 'para', text });
    return;
  }
  for (const c of Array.from(el.children)) walk(c, depth, out);
}

/** Kumpulkan konten yang sedang tampil di halaman menjadi blok laporan. Grafik dirender ulang sebagai PNG. */
export async function collectBlocks(root: Element, plotImage: PlotImage | null): Promise<{ blocks: Block[]; dataset?: string; title?: string }> {
  const raw: Raw[] = [];
  for (const c of Array.from(root.children)) walk(c, 0, raw);
  const blocks: Block[] = [];
  for (const r of raw) {
    if (r.kind !== 'plot') { blocks.push(r); continue; }
    if (!plotImage) continue;
    const img = await plotImage(r.el).catch(() => null);
    if (img) blocks.push({ kind: 'image', ...img });
  }
  // Buang judul bagian yang tidak diikuti konten apa pun.
  const pruned = blocks.filter((b, i) => b.kind !== 'heading' || (i + 1 < blocks.length && !(blocks[i + 1].kind === 'heading' && (blocks[i + 1] as { level: number }).level <= b.level)));
  const dataset = clean(root.querySelector('[data-export-dataset]')?.textContent) || undefined;
  const title = clean(root.querySelector('h1')?.textContent) || undefined;
  return { blocks: pruned, dataset, title };
}
