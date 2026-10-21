import {
  AlignmentType, BorderStyle, Document, Footer, Header, HeadingLevel, ImageRun, Packer, PageNumber,
  Paragraph, ShadingType, Table, TableCell, TableLayoutType, TableRow, TextRun, WidthType,
} from 'docx';
import { fmtDate, type Block, type ReportDoc } from './model';

const BRAND = '0B4F3A';
const MAX_ROWS = 100;
/** Lebar isi halaman A4 dengan margin 1 inci, dalam twip (DXA). */
const PAGE_W = 9026;
const TONE = { info: '0B7A53', warn: 'B7791F', ok: '2F855A' } as const;

function dataUrlBytes(dataUrl: string): Uint8Array {
  const bin = atob(dataUrl.split(',')[1] ?? '');
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const cell = (text: string, width: number, header = false) => new TableCell({
  width: { size: width, type: WidthType.DXA },
  children: [new Paragraph({ children: [new TextRun({ text, bold: header, size: 18, color: header ? 'FFFFFF' : undefined })] })],
  shading: header ? { type: ShadingType.CLEAR, color: 'auto', fill: BRAND } : undefined,
  margins: { top: 40, bottom: 40, left: 80, right: 80 },
});

function table(head: string[], rows: string[][]): Table {
  const width = Math.max(head.length, ...rows.map((r) => r.length), 1);
  const pad = (r: string[]) => [...r, ...Array(Math.max(0, width - r.length)).fill('')];
  // Lebar kolom eksplisit: tanpa ini sebagian pembaca (Quick Look, LibreOffice, Google Docs) meremas tabel.
  const colW = Math.floor(PAGE_W / width);
  return new Table({
    width: { size: colW * width, type: WidthType.DXA },
    columnWidths: Array(width).fill(colW),
    layout: TableLayoutType.FIXED,
    rows: [
      new TableRow({ tableHeader: true, children: pad(head).map((h) => cell(h, colW, true)) }),
      ...rows.map((r) => new TableRow({ children: pad(r).map((c) => cell(c, colW)) })),
    ],
  });
}

function blockToDocx(b: Block): (Paragraph | Table)[] {
  switch (b.kind) {
    case 'heading':
      return [new Paragraph({ text: b.text, heading: b.level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2 })];
    case 'para':
      return [new Paragraph({ text: b.text, spacing: { after: 120 } })];
    case 'bullets':
      return b.items.map((t) => new Paragraph({ text: t, bullet: { level: 0 } }));
    case 'note':
      return [new Paragraph({
        children: [...(b.title ? [new TextRun({ text: `${b.title}. `, bold: true })] : []), new TextRun(b.text)],
        border: { left: { style: BorderStyle.SINGLE, size: 18, color: TONE[b.tone], space: 8 } },
        shading: { type: ShadingType.CLEAR, color: 'auto', fill: 'F3F6F2' },
        spacing: { before: 80, after: 160 }, indent: { left: 160 },
      })];
    case 'stats':
      return [table(['Metrik', 'Nilai', 'Keterangan'], b.items.map((s) => [s.label, s.value, s.hint ?? ''])), new Paragraph({ text: '' })];
    case 'table': {
      const rows = b.rows.slice(0, MAX_ROWS);
      const out: (Paragraph | Table)[] = [];
      if (b.caption) out.push(new Paragraph({ children: [new TextRun({ text: b.caption, italics: true, size: 18 })] }));
      out.push(table(b.head, rows));
      out.push(new Paragraph({
        children: b.rows.length > MAX_ROWS ? [new TextRun({ text: `… ${b.rows.length - MAX_ROWS} baris lain tidak ditampilkan.`, italics: true, size: 18 })] : [],
        spacing: { after: 160 },
      }));
      return out;
    }
    case 'image': {
      const w = 600, h = Math.round((w * b.height) / b.width);
      return [
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: 'png', data: dataUrlBytes(b.dataUrl), transformation: { width: w, height: h } })] }),
        ...(b.title ? [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: b.title, italics: true, size: 18 })], spacing: { after: 160 } })] : []),
      ];
    }
  }
}

/** Susun dokumen Word (.docx) dari model laporan. */
export async function toDocx(doc: ReportDoc): Promise<Blob> {
  const meta = (label: string, value: string) => new Paragraph({ children: [new TextRun({ text: `${label}: `, bold: true }), new TextRun(value)], spacing: { after: 80 } });
  const cover = [
    new Paragraph({ children: [new TextRun({ text: 'DATANALYSX · LAPORAN ANALISIS', bold: true, color: 'B7791F', size: 20 })], spacing: { before: 2400, after: 240 } }),
    new Paragraph({ children: [new TextRun({ text: doc.title, bold: true, size: 52, color: BRAND })], spacing: { after: 240 } }),
    ...(doc.subtitle ? [new Paragraph({ children: [new TextRun({ text: doc.subtitle, size: 24, color: '4B5650' })], spacing: { after: 480 } })] : []),
    meta('Disusun oleh', doc.author),
    meta('Tanggal', fmtDate(doc.date)),
    meta('Sumber', doc.source),
    new Paragraph({ children: [new TextRun({ text: 'Dibuat otomatis oleh DATANALYSX dari hasil yang tampil di aplikasi; seluruh perhitungan berjalan di browser.', italics: true, size: 18, color: '4B5650' })], spacing: { before: 480 } }),
  ];
  const body = doc.blocks.flatMap(blockToDocx);
  const page = { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } };
  const document = new Document({
    creator: doc.author, title: doc.title, description: doc.subtitle,
    styles: {
      default: { document: { run: { font: 'Calibri', size: 22 } } },
      paragraphStyles: [
        { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 32, bold: true, color: BRAND }, paragraph: { spacing: { before: 360, after: 160 } } },
        { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 26, bold: true, color: '0B7A53' }, paragraph: { spacing: { before: 240, after: 120 } } },
      ],
    },
    sections: [{ properties: { page }, children: cover }, {
      properties: { page },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: `DATANALYSX — ${doc.title}`, size: 16, color: '808080' })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [`${doc.author} · Halaman `, PageNumber.CURRENT, ' dari ', PageNumber.TOTAL_PAGES], size: 16, color: '808080' })] })] }) },
      children: body,
    }],
  });
  return Packer.toBlob(document);
}
