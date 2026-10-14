// GitHub Pages tidak mendukung rewrite SPA: salin index.html -> 404.html.
import { copyFileSync, existsSync } from 'node:fs';
if (existsSync('dist/index.html')) copyFileSync('dist/index.html', 'dist/404.html');
console.log('postbuild: dist/404.html dibuat');
