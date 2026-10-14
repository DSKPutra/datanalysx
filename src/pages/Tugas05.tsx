import { useEffect, useState } from 'react';
import DataSource from '../components/DataSource';
import ModulePage from '../components/ModulePage';
import Plot from '../components/Plot';
import { Callout, ErrorBox, LiveBadge, ReportBadge, Section, Select, Slider, Spinner, Stat, StatGrid, Table } from '../components/ui';
import { bySlug } from '../content/modules';
import { fmt, fmtFixed, int, pct } from '../lib/format';
import { infoGain, gini } from '../lib/manual/entropy';
import { py } from '../lib/pyodide/client';
import { palette } from '../lib/theme';
import { useAsync, useJson } from '../lib/useAsync';

const m = bySlug('tugas-05')!;

function Report() {
  const { data: r } = useJson<any>('data/tugas-05.json');
  if (!r) return <Spinner text="Memuat hasil laporan …" />;
  return (
    <div className="space-y-5">
      <ReportBadge source={r.sumber} />
      <div className="grid gap-5 lg:grid-cols-2">
        <Section title="Skenario 1 — metadata" right={<span className="text-xs muted">{r.skenario1_sumber}</span>}>
          <Table head={['Model', 'AUC', 'CA', 'F1', 'Presisi', 'Recall', 'SD CA']} numeric={[1, 2, 3, 4, 5, 6]} rows={r.skenario1} />
        </Section>
        <Section title="Skenario 2 — kata pada judul" right={<span className="text-xs muted">{r.skenario2_sumber}</span>}>
          <Table head={['Model', 'AUC', 'CA', 'F1', 'Presisi', 'Recall', 'SD CA']} numeric={[1, 2, 3, 4, 5, 6]} rows={r.skenario2} />
        </Section>
      </div>
      <Section title="Kompleksitas pohon & overfitting" right={<span className="text-xs muted">{r.pohon_sumber}</span>}>
        <Table head={['Model', 'Simpul', 'Daun', 'Kedalaman', 'Akurasi latih', 'Akurasi uji', 'F1 makro uji']} numeric={[1, 2, 3, 4, 5, 6]} rows={r.pohon.map((x: any[]) => [x[0], int(x[1]), x[2], x[3], pct(x[4], 1), pct(x[5], 1), fmtFixed(x[6], 3)])} />
        <p className="text-sm">Uji permutasi Skenario 1: akurasi asli {fmtFixed(r.permutasi.akurasi_asli, 4)} vs rerata label acak {fmtFixed(r.permutasi.rerata_acak, 4)} → p = {fmtFixed(r.permutasi.p, 3)} ({r.permutasi.sumber}).</p>
      </Section>
      <Section title="Uji ketangguhan: judul baru" right={<span className="text-xs muted">{r.judul_baru_sumber}</span>}>
        <Table head={['Judul baru', 'Semestinya', 'Prediksi pohon', 'Hasil']} rows={r.judul_baru} />
        <p className="text-sm">Seluruh tebakan berkeyakinan 100% — pohon mengenali templat, bukan makna.</p>
      </Section>
    </div>
  );
}

function TreeDiagram({ nodes, labels }: { nodes: any[]; labels: string[] }) {
  const p = palette();
  const depthOf: Record<number, number> = { 0: 0 };
  nodes.forEach((n) => { if (n.left > 0) { depthOf[n.left] = depthOf[n.id] + 1; depthOf[n.right] = depthOf[n.id] + 1; } });
  const shown = nodes.filter((n) => (depthOf[n.id] ?? 99) <= 3);
  const byDepth: Record<number, any[]> = {};
  shown.forEach((n) => (byDepth[depthOf[n.id]] ??= []).push(n));
  return (
    <div className="space-y-2 overflow-x-auto">
      {Object.entries(byDepth).map(([d, ns]) => (
        <div key={d} className="flex justify-center gap-2">
          {ns.map((n) => {
            const k = n.value.indexOf(Math.max(...n.value));
            return (
              <div key={n.id} className="min-w-[120px] rounded-lg border p-2 text-center text-xs" style={{ borderColor: p.c[k], background: `color-mix(in srgb, ${p.c[k]} ${Math.round(Math.max(...n.value) * 45)}%, transparent)` }}>
                <div className="font-semibold">{labels[k]} {pct(Math.max(...n.value), 0)}</div>
                <div className="muted">n = {int(n.n)}</div>
                {n.feature && <div>{n.feature} ≤ {fmt(n.threshold, 2)}?</div>}
              </div>
            );
          })}
        </div>
      ))}
      {nodes.length > shown.length && <p className="text-center text-xs muted">Hanya kedalaman 0–3 ditampilkan; lihat aturan teks untuk detail.</p>}
    </div>
  );
}

function Lab() {
  const [key, setKey] = useState<string | null>(null);
  const [sc, setSc] = useState(1);
  const [crit, setCrit] = useState<'gini' | 'entropy'>('gini');
  const [depth, setDepth] = useState(3);
  const [leaf, setLeaf] = useState(25);
  const ov = useAsync<any>();
  const fit = useAsync<any>();
  const dc = useAsync<any>();
  const perm = useAsync<any>();
  const hl = useAsync<any>();
  const [headlines, setHeadlines] = useState('Lakers beat Celtics in overtime thriller\nInflation remains a key concern for young professionals\nStartup funding for climate technology grows');
  useEffect(() => { if (key) ov.run(() => py('dmx.t05', 'overview', { data_key: key })); }, [key, ov.run]);
  useEffect(() => { if (key) fit.run(() => py('dmx.t05', 'fit_tree', { data_key: key, scenario: sc, criterion: crit, max_depth: depth === 0 ? null : depth, min_samples_leaf: leaf })); }, [key, sc, crit, depth, leaf, fit.run]);
  useEffect(() => { if (key) dc.run(() => py('dmx.t05', 'depth_curve', { data_key: key, scenario: sc, criterion: crit })); }, [key, sc, crit, dc.run]);
  const r = fit.data;
  const p = palette();
  return (
    <div className="space-y-5">
      <DataSource dataKey="t05" sample={m.sample} onReady={setKey} />
      <Callout kind="warn" title="Dataset sintetis">Atribut dibangkitkan acak & judul dari templat; hasil tidak berlaku untuk data trending nyata.</Callout>
      {ov.data && <StatGrid><Stat label="Baris" value={int(ov.data.shape[0])} /><Stat label="Latih / uji" value={`${ov.data.n_train} / ${ov.data.n_test}`} /><Stat label="Fitur S1 / S2" value={`${ov.data.n_features.s1} / ${ov.data.n_features.s2}`} /><Stat label="Templat judul / kelas" value={Object.values(ov.data.headline_templates)[0] as number} /></StatGrid>}
      {key && (
        <Section title="Bangun pohon" right={<LiveBadge />}>
          <div className="grid gap-4 md:grid-cols-4">
            <Select label="Skenario" value={sc} onChange={setSc} options={[{ value: 1, label: '1 — metadata' }, { value: 2, label: '2 — kata judul' }, { value: 3, label: '3 — gabungan' }]} />
            <Select label="Kriteria" value={crit} onChange={setCrit} options={[{ value: 'gini', label: 'Gini' }, { value: 'entropy', label: 'Entropy' }]} />
            <Slider label="max_depth (0 = tanpa batas)" value={depth} min={0} max={25} onChange={setDepth} />
            <Slider label="min_samples_leaf" value={leaf} min={1} max={100} onChange={setLeaf} />
          </div>
          <ErrorBox error={fit.error} />
          {r && (<>
            <StatGrid>
              <Stat label="Akurasi latih" value={pct(r.train_acc, 1)} /><Stat label="Akurasi uji" value={pct(r.test_acc, 1)} hint={`mayoritas ${pct(r.majority_baseline, 1)}`} />
              <Stat label="CV 10-lipatan" value={`${pct(r.cv_acc_mean, 1)} ± ${pct(r.cv_acc_std, 1)}`} /><Stat label="AUC CV (OvR)" value={fmtFixed(r.cv_auc_mean, 3)} />
              <Stat label="Simpul / daun / kedalaman" value={`${int(r.n_nodes)} / ${int(r.n_leaves)} / ${r.depth}`} />
            </StatGrid>
            <TreeDiagram nodes={r.nodes} labels={r.labels} />
            <div className="grid gap-5 lg:grid-cols-2">
              <Plot title="Confusion matrix (data uji)" height={320} data={[{ type: 'heatmap', z: r.confusion, x: r.labels, y: r.labels, colorscale: [[0, '#f5f5f2'], [1, '#0B7A53']], text: r.confusion, texttemplate: '%{text}', showscale: false }]} layout={{ yaxis: { autorange: 'reversed', title: { text: 'Aktual' } }, xaxis: { title: { text: 'Prediksi' } }, margin: { l: 100 } }} />
              <Plot title="Feature importance (MDI)" height={320} data={[{ type: 'bar', orientation: 'h', y: r.importances.map((x: any) => x.feature).reverse(), x: r.importances.map((x: any) => x.importance).reverse(), marker: { color: p.brand2 } }]} layout={{ margin: { l: 140 } }} />
            </div>
            <details><summary className="cursor-pointer text-sm font-semibold">Aturan JIKA–MAKA (export_text)</summary><pre className="code-block mt-2 max-h-80 overflow-auto rounded-lg p-3 text-xs">{r.rules}</pre></details>
          </>)}
          {dc.data && <Plot title="Akurasi latih vs validasi terhadap kedalaman (bias–varians)" height={300} data={[
            { type: 'scatter', mode: 'lines+markers', x: dc.data.rows.map((x: any) => x.depth), y: dc.data.rows.map((x: any) => x.train), name: 'Latih', line: { color: p.text } },
            { type: 'scatter', mode: 'lines+markers', x: dc.data.rows.map((x: any) => x.depth), y: dc.data.rows.map((x: any) => x.val), name: 'Validasi (CV 5)', line: { color: p.brand2 } }]}
            layout={{ xaxis: { title: { text: 'max_depth' } }, shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: dc.data.majority, y1: dc.data.majority, line: { dash: 'dash', color: p.gold } }] }} />}
        </Section>
      )}
      {key && (
        <div className="grid gap-5 lg:grid-cols-2">
          <Section title="Uji permutasi (Skenario aktif)">
            <button className="btn-ghost" disabled={perm.loading} onClick={() => perm.run(() => py('dmx.t05', 'permutation_test', { data_key: key, scenario: sc, n_perm: 50 }))}>Jalankan 50 pengacakan label</button>
            {perm.loading && <Spinner text="Melatih 250 pohon …" />}
            {perm.data && (<>
              <Plot height={240} data={[{ type: 'histogram', x: perm.data.null, marker: { color: p.neutral }, name: 'Label acak' }]} layout={{ shapes: [{ type: 'line', x0: perm.data.observed, x1: perm.data.observed, y0: 0, y1: 1, yref: 'paper', line: { color: p.down, width: 3 } }], xaxis: { title: { text: 'Akurasi CV' } } }} />
              <p className="text-sm">Akurasi asli {fmtFixed(perm.data.observed, 4)} · rerata acak {fmtFixed(perm.data.null_mean, 4)} · p = {fmtFixed(perm.data.p_value, 3)}</p>
            </>)}
          </Section>
          <Section title="Uji judul baru (pohon Skenario 2)">
            <textarea className="input h-28" value={headlines} onChange={(e) => setHeadlines(e.target.value)} aria-label="Judul baru, satu per baris" />
            <button className="btn-ghost" onClick={() => hl.run(() => py('dmx.t05', 'predict_headlines', { data_key: key, headlines: headlines.split('\n').filter(Boolean) }))}>Tebak kategori</button>
            {hl.data && <Table head={['Judul', 'Prediksi', 'Keyakinan', 'Kata dikenal']} rows={hl.data.rows.map((x: any) => [x.headline, x.pred, pct(x.confidence, 0), x.known_words.join(', ') || '—'])} />}
          </Section>
        </div>
      )}
    </div>
  );
}

function Manual() {
  const [key, setKey] = useState<string | null>(null);
  const [word, setWord] = useState('for');
  const s = useAsync<any>();
  useEffect(() => { if (key) s.run(() => py('dmx.t05', 'manual_split', { data_key: key, word })); }, [key, word, s.run]);
  const r = s.data;
  const ts = r ? infoGain(r.left_counts, r.right_counts) : null;
  return (
    <div className="space-y-5">
      <DataSource dataKey="t05" sample={m.sample} onReady={setKey} />
      {r && ts && (
        <Section title={`Entropy & information gain untuk kata “${word}”`} right={<LiveBadge />}>
          <Select label="Kata pemisah" value={word} onChange={setWord} options={r.vocab_top.map((w: string) => ({ value: w, label: w }))} />
          <Table head={['Simpul', ...r.labels, 'n', 'Entropy (bit)']} numeric={[1, 2, 3, 4, 5, 6, 7]} rows={[
            ['Akar', ...r.root_counts, r.n, fmtFixed(ts.H, 4)],
            [`“${word}” ada`, ...r.left_counts, r.left_counts.reduce((a: number, b: number) => a + b, 0), fmtFixed(ts.Hl, 4)],
            [`“${word}” tidak ada`, ...r.right_counts, r.right_counts.reduce((a: number, b: number) => a + b, 0), fmtFixed(ts.Hr, 4)]]} />
          <p className="font-mono text-sm">H(akar) = −Σ pᵢ log₂ pᵢ = {fmtFixed(ts.H, 6)} bit (maks log₂5 = 2,3219)</p>
          <p className="font-mono text-sm">H(setelah) = {fmtFixed(r.w_left, 4)}·{fmtFixed(ts.Hl, 4)} + {fmtFixed(r.w_right, 4)}·{fmtFixed(ts.Hr, 4)} = {fmtFixed(ts.after, 6)}</p>
          <p className="font-mono text-sm">IG = {fmtFixed(ts.H, 6)} − {fmtFixed(ts.after, 6)} = <strong>{fmtFixed(ts.ig, 6)}</strong> bit · gain ratio = {fmtFixed(ts.gainRatio, 4)} · Gini akar = {fmtFixed(gini(r.root_counts), 4)}</p>
          <Callout kind={Math.max(Math.abs(ts.H - r.sklearn_H_root), Math.abs(ts.after - r.sklearn_H_after)) < 1e-8 ? 'ok' : 'warn'} title="Verifikasi vs DecisionTreeClassifier(criterion='entropy', max_depth=1)">
            Impuritas akar sklearn = {fmtFixed(r.sklearn_H_root, 8)}; impuritas tertimbang anak = {fmtFixed(r.sklearn_H_after, 8)}; selisih maks {Math.max(Math.abs(ts.H - r.sklearn_H_root), Math.abs(ts.after - r.sklearn_H_after)).toExponential(2)} ✓ (laporan: IG “for” 0,9546 bit).
          </Callout>
        </Section>
      )}
      {s.loading && <Spinner />}
      <ErrorBox error={s.error} />
    </div>
  );
}

export default function Tugas05() {
  return <ModulePage m={m} report={<Report />} lab={<Lab />} manual={<Manual />} />;
}
