import { useEffect, useRef, useState } from 'react';
import { palette } from '../lib/theme';

let plotlyPromise: Promise<any> | null = null;
const loadPlotly = () => (plotlyPromise ??= import('plotly.js-cartesian-dist-min').then((m: any) => m.default ?? m));

export interface PlotProps {
  data: any[];
  layout?: Record<string, any>;
  height?: number;
  title?: string;
  ariaLabel?: string;
  filename?: string;
}

type Colors = { text: string; muted: string; line: string };

/** Layout bersama untuk tampilan layar dan ekspor laporan (warna teks/sumbu mengikuti tema yang diberikan). */
function buildLayout({ layout = {}, height = 360, title }: PlotProps, p: Colors) {
  const axis = { gridcolor: p.line, zerolinecolor: p.line, linecolor: p.line, color: p.muted, automargin: true };
  return {
    title: title ? { text: title, font: { size: 14, color: p.text } } : undefined,
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, system-ui, sans-serif', color: p.text, size: 12 },
    margin: { l: 56, r: 16, t: title ? 40 : 16, b: 48 }, height,
    legend: { orientation: 'h', y: -0.22, font: { size: 11 } },
    separators: ',.',
    hoverlabel: { font: { family: 'Inter' } },
    ...layout,
    xaxis: { ...axis, ...(layout.xaxis ?? {}) },
    yaxis: { ...axis, ...(layout.yaxis ?? {}) },
  };
}

/** Props tiap grafik yang sedang terpasang — dibaca eksportir, termasuk grafik yang belum digulir ke layar. */
const mounted = new WeakMap<Element, PlotProps>();

/** Render grafik ke PNG berlatar putih (tema terang) untuk laporan .docx/.pptx. */
export async function plotToImage(el: Element, width = 1000): Promise<{ dataUrl: string; width: number; height: number; title?: string } | null> {
  const props = mounted.get(el);
  if (!props || !props.data.length) return null;
  const Plotly = await loadPlotly();
  const height = props.height ?? 360;
  const lay = { ...buildLayout(props, { text: '#1F2A26', muted: '#4B5650', line: '#DDE2DC' }), paper_bgcolor: '#FFFFFF', plot_bgcolor: '#FFFFFF', width, height };
  const dataUrl: string = await Plotly.toImage({ data: props.data, layout: lay }, { format: 'png', width, height, scale: 2 });
  return { dataUrl, width, height, title: props.title };
}

/** Grafik interaktif Plotly (tooltip, zoom, ekspor PNG); warna mengikuti tema terang/gelap. */
export default function Plot({ data, layout = {}, height = 360, title, ariaLabel, filename = 'datanalysx-grafik' }: PlotProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [themeTick, setThemeTick] = useState(0);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (el) mounted.set(el, { data, layout, height, title });
  }, [data, layout, height, title]);
  // Plotly (±470 kB gzip) baru dimuat saat grafik mendekati viewport.
  useEffect(() => {
    const el = ref.current;
    if (!el || visible) return;
    if (!('IntersectionObserver' in window)) { setVisible(true); return; }
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { setVisible(true); io.disconnect(); } }, { rootMargin: '200px' });
    io.observe(el);
    return () => io.disconnect();
  }, [visible]);
  useEffect(() => {
    const h = () => setThemeTick((t) => t + 1);
    window.addEventListener('dmx-theme', h);
    return () => window.removeEventListener('dmx-theme', h);
  }, []);
  useEffect(() => {
    if (!visible) return;
    let alive = true;
    const el = ref.current;
    loadPlotly().then((Plotly) => {
      if (!alive || !el) return;
      const lay = buildLayout({ data, layout, height, title }, palette());
      Plotly.react(el, data, lay, {
        responsive: true, displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        toImageButtonOptions: { format: 'png', filename, scale: 2 },
      });
    });
    return () => { alive = false; };
  }, [data, layout, height, title, filename, themeTick, visible]);
  useEffect(() => () => { if (ref.current) loadPlotly().then((P) => ref.current && P.purge(ref.current)); }, []);
  return <div ref={ref} data-export-plot role="img" aria-label={ariaLabel ?? title ?? 'Grafik interaktif'} style={{ minHeight: height, width: '100%' }} />;
}
