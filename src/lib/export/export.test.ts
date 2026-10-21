// @vitest-environment jsdom
import JSZip from 'jszip';
import { describe, expect, it } from 'vitest';
import { collectBlocks } from './collect';
import { toDocx } from './docx';
import type { ReportDoc } from './model';
import { fileStem } from './model';
import { toPptx } from './pptx';

const PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

function page(html: string) {
  const root = document.createElement('div');
  root.setAttribute('data-export-root', '');
  root.innerHTML = html;
  return root;
}

describe('collectBlocks', () => {
  it('mengubah bagian, metrik, tabel, catatan, dan grafik menjadi blok; melewati kontrol & tab tersembunyi', async () => {
    const root = page(`
      <header><h1>Klasifikasi</h1><p>Intro alat.</p><div data-export-skip><button>Ekspor</button></div></header>
      <div data-export-skip><strong data-export-dataset>reels.csv</strong></div>
      <section data-export-section><div><h2>Hasil</h2><span data-export-skip>⚡ Live</span></div>
        <div><div data-export-stat><div data-export-label>Akurasi</div><div data-export-value>72,7%</div><div data-export-hint>baseline 54,5%</div></div>
             <div data-export-stat><div data-export-label>F1</div><div data-export-value>0,725</div></div></div>
        <table><caption>Uji</caption><thead><tr><th>Model</th><th>R²</th></tr></thead><tbody><tr><td>Linear</td><td>0,9</td></tr></tbody></table>
        <div data-export-callout="warn"><p data-export-title>⚠ Peringatan</p><div class="prose-dmx">Data sintetis.</div></div>
        <select><option>x</option></select>
        <div data-export-plot></div>
      </section>
      <div hidden><section data-export-section><div><h2>Tab lain</h2></div><p>tidak boleh ikut</p></section></div>
      <section data-export-section><div><h2>Kosong</h2></div></section>
    `);
    const { blocks, dataset, title } = await collectBlocks(root, async () => ({ dataUrl: PNG, width: 10, height: 5, title: 'Grafik' }));
    expect(title).toBe('Klasifikasi');
    expect(dataset).toBe('reels.csv');
    expect(blocks.map((b) => b.kind)).toEqual(['para', 'heading', 'stats', 'table', 'note', 'image']);
    expect(blocks[2]).toEqual({ kind: 'stats', items: [{ label: 'Akurasi', value: '72,7%', hint: 'baseline 54,5%' }, { label: 'F1', value: '0,725', hint: undefined }] });
    expect(blocks[3]).toEqual({ kind: 'table', head: ['Model', 'R²'], rows: [['Linear', '0,9']], caption: 'Uji' });
    expect(blocks[4]).toMatchObject({ kind: 'note', title: 'Peringatan', text: 'Data sintetis.', tone: 'warn' });
    expect(JSON.stringify(blocks)).not.toContain('tidak boleh ikut');
  });

  it('tanpa grafik bila opsi grafik dimatikan', async () => {
    const { blocks } = await collectBlocks(page('<p>a</p><div data-export-plot></div>'), null);
    expect(blocks).toEqual([{ kind: 'para', text: 'a' }]);
  });
});

const doc: ReportDoc = {
  title: 'Analisis Regresi', subtitle: 'Uji', author: 'Dea Saka Kurnia Putra', date: new Date(2026, 9, 21), source: 'https://contoh.test/regresi',
  blocks: [
    { kind: 'heading', text: 'Hasil', level: 1 },
    { kind: 'para', text: 'Paragraf.' },
    { kind: 'bullets', items: ['satu', 'dua'] },
    { kind: 'note', title: 'Catatan', text: 'Isi', tone: 'info' },
    { kind: 'stats', items: [{ label: 'R²', value: '0,99', hint: 'uji' }] },
    { kind: 'table', head: ['a', 'b'], rows: Array.from({ length: 30 }, (_, i) => [String(i), 'x']) },
    { kind: 'image', title: 'Grafik', dataUrl: PNG, width: 10, height: 5 },
  ],
};

async function unzip(blob: Blob) {
  // Blob jsdom tidak punya arrayBuffer(); baca lewat FileReader.
  const buf = await new Promise<ArrayBuffer>((res, rej) => { const fr = new FileReader(); fr.onload = () => res(fr.result as ArrayBuffer); fr.onerror = rej; fr.readAsArrayBuffer(blob); });
  return JSZip.loadAsync(buf);
}

describe('renderer', () => {
  it('.docx berisi judul, nama penyusun, tabel, dan gambar', async () => {
    const zip = await unzip(await toDocx(doc));
    const xml = await zip.file('word/document.xml')!.async('string');
    expect(xml).toContain('Analisis Regresi');
    expect(xml).toContain('Dea Saka Kurnia Putra');
    expect(xml).toContain('<w:tbl>');
    expect(Object.keys(zip.files).some((f) => f.startsWith('word/media/'))).toBe(true);
  });

  it('.pptx berisi slide judul, isi, tabel terpecah, gambar, dan penutup', async () => {
    const zip = await unzip(await toPptx(doc));
    const slides = Object.keys(zip.files).filter((f) => /^ppt\/slides\/slide\d+\.xml$/.test(f));
    // judul + teks + metrik + tabel (30 baris → 3 slide) + gambar + penutup
    expect(slides.length).toBe(8);
    const s1 = await zip.file('ppt/slides/slide1.xml')!.async('string');
    expect(s1).toContain('Dea Saka Kurnia Putra');
    expect(Object.keys(zip.files).some((f) => f.startsWith('ppt/media/'))).toBe(true);
  });

  it('nama berkas aman', () => {
    expect(fileStem('datanalysx', 'Studi Kasus 06: Naïve Bayes', 'Dea S.')).toBe('datanalysx-studi-kasus-06-naive-bayes-dea-s');
  });
});
