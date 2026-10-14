// Salin skrip Python asli & modul dmx ke public/code agar bisa dilihat lewat tombol "Lihat kode Python".
import { cpSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const out = 'public/code';
mkdirSync(join(out, 'original'), { recursive: true });
mkdirSync(join(out, 'dmx'), { recursive: true });
for (const dir of ['original', 'dmx']) {
  const files = readdirSync(join('python', dir)).filter((f) => f.endsWith('.py'));
  for (const f of files) cpSync(join('python', dir, f), join(out, dir, f));
}
writeFileSync(join(out, 'index.json'), JSON.stringify({ generated: new Date().toISOString() }));
console.log('sync-assets: kode Python disalin ke public/code');
