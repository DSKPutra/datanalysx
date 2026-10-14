import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { py } from '../lib/pyodide/client';
import DataSource, { type SampleInfo } from './DataSource';
import { Select } from './ui';

export interface Column { name: string; kind: 'numerik' | 'kategorikal' | 'teks'; unique: number; missing: number }
export interface Preset { target?: string; features?: string[]; date?: string; x?: string; y?: string; text?: string }

/** Dataset contoh = data studi kasus yang lisensinya mengizinkan redistribusi. */
export const SAMPLES: (SampleInfo & { key: string; caseNo: string; presets: Record<string, Preset> })[] = [
  { key: 'reels', caseNo: '07', url: 'data/samples/t07_instagram_reels_knn_clean.csv', label: 'Instagram Reels 2026 (studi kasus 07)', license: 'CC0',
    presets: { klasifikasi: { target: 'virality_tier', features: ['video_duration_sec', 'caption_word_count', 'caption_char_len', 'hashtag_count', 'emoji_count', 'mention_count', 'post_age_days', 'hour_sin', 'hour_cos', 'is_weekend', 'has_question', 'niche'] },
      clustering: { features: ['video_duration_sec', 'caption_word_count', 'hashtag_count', 'post_hour'] } } },
  { key: 'semikonduktor', caseNo: '03', url: 'data/samples/t03_chip_companies_financials.csv', label: 'Keuangan perusahaan semikonduktor (studi kasus 03)', license: 'CC0',
    presets: { regresi: { target: 'revenue_usd_bn', features: ['rd_spend_usd_bn', 'capex_usd_bn', 'operating_margin_pct'], x: 'rd_spend_usd_bn', y: 'revenue_usd_bn' },
      clustering: { features: ['rd_spend_usd_bn', 'capex_usd_bn', 'operating_margin_pct'] }, klasifikasi: { target: 'segment', features: ['rd_spend_usd_bn', 'capex_usd_bn', 'operating_margin_pct', 'revenue_usd_bn'] } } },
  { key: 'nvidia', caseNo: '01', url: 'data/samples/t01_nvidia_stock_1999_2026.csv', label: 'Saham NVIDIA 1999–2026 (studi kasus 01)', license: 'CC0',
    presets: { regresi: { target: 'close', features: ['open', 'high', 'low', 'volume', 'sma_50', 'rsi_14'], date: 'date', x: 'open', y: 'close' },
      klasifikasi: { target: 'era', features: ['rsi_14', 'volume', 'quarterly_revenue_usd_bn'], date: 'date' } } },
  { key: 'trending', caseNo: '05', url: 'data/samples/t05_trending_topics_2026_synthetic.csv', label: 'Trending Topics 2026 — sintetis (studi kasus 05)', license: 'CC0',
    presets: { klasifikasi: { target: 'topic_category', features: ['source', 'language', 'country', 'sentiment', 'engagement_score', 'trend_score'] },
      teks: { text: 'headline' } } },
  { key: 'llm', caseNo: '02', url: 'data/samples/t02_llm_agent_failure_benchmark.csv', label: 'LLM Agent Failure Benchmark (studi kasus 02)', license: 'Apache 2.0',
    presets: { klasifikasi: { target: 'failure_severity', features: ['domain', 'difficulty'] }, teks: { text: 'agent_answer' } } },
  { key: 'macro', caseNo: '06', url: 'data/samples/t06_indonesia_multiasset_macro_2010_2026.csv', label: 'Makro-finansial Indonesia 2010–2026 (studi kasus 06)', license: 'CC BY-SA 4.0',
    presets: { regresi: { target: 'IHSG', features: ['SP500', 'GOLD', 'OIL', 'USDIDR', 'VIX', 'BI_rate'], date: 'Date', x: 'SP500', y: 'IHSG' },
      clustering: { features: ['VIX', 'BI_rate', 'US_rate', 'CPI'] } } },
  { key: 'aapl2026', caseNo: '08', url: 'data/samples/t08_aapl_2026_stock_data.csv', label: 'AAPL 2026 — semi-sintetis (studi kasus 08)', license: 'Apache 2.0',
    presets: { clustering: { features: ['Daily_Return_Pct', 'Volatility_10D_Pct', 'Volume'] }, regresi: { target: 'Close', features: ['Open', 'High', 'Low', 'Volume'], date: 'Date', x: 'Open', y: 'Close' } } },
];

/** Panel pemilihan dataset untuk alat umum: unggah CSV sendiri atau pakai data studi kasus.
 *  Parameter URL ?sample=<kunci> memilih dataset contoh (dipakai tautan dari halaman studi kasus). */
export default function Dataset({ dataKey, tool, onReady }: {
  dataKey: string; tool: string; onReady: (info: { key: string; columns: Column[]; nRows: number; preset: Preset; source: string }) => void;
}) {
  const [params] = useSearchParams();
  const relevant = tool === 'eksplorasi' ? SAMPLES : SAMPLES.filter((s) => s.presets[tool]);
  const fromUrl = relevant.findIndex((s) => s.key === params.get('sample'));
  const [idx, setIdx] = useState(fromUrl >= 0 ? fromUrl : 0);
  const [mode, setMode] = useState<'sample' | 'upload'>(relevant.length > 0 ? 'sample' : 'upload');
  const s = relevant[idx];
  const [ver, setVer] = useState(0);
  useEffect(() => setVer((v) => v + 1), [idx, mode]);
  async function ready(key: string, source: string, preset: Preset) {
    const info = await py<{ n_rows: number; columns: Column[] }>('dmx.eda', 'columns', { data_key: key });
    onReady({ key, columns: info.columns, nRows: info.n_rows, preset, source });
  }
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-end gap-3">
        <div className="min-w-[220px]"><Select label="Sumber data" value={mode} onChange={setMode} options={[{ value: 'upload', label: 'Unggah CSV saya' }, ...(relevant.length ? [{ value: 'sample' as const, label: 'Contoh dari studi kasus' }] : [])]} /></div>
        {mode === 'sample' && relevant.length > 0 && <div className="min-w-[280px] flex-1"><Select label="Dataset contoh" value={idx} onChange={setIdx} options={relevant.map((x, i) => ({ value: i, label: x.label }))} /></div>}
      </div>
      <DataSource key={`${mode}-${ver}`} dataKey={dataKey} sample={mode === 'sample' ? s : null}
        uploadTitle="Unggah data Anda" uploadHint="Unggah CSV apa pun (baris = observasi, kolom = variabel). Data diproses di browser Anda."
        onReady={(k, src) => ready(k, src, mode === 'sample' ? s.presets[tool] ?? {} : {})} />
    </div>
  );
}
