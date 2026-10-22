/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: ['class', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        emerald: { 950: '#062E22', 900: '#0B4F3A', 700: '#0B7A53', 500: '#1FA374', 100: '#DDF1E8', 50: '#F1FAF5' },
        gold: { DEFAULT: '#B8860B', light: '#D9B24C' },
        ink: { DEFAULT: '#0F1512', soft: '#1A231F', line: '#2A3631' },
      },
      fontFamily: {
        display: ['"Space Grotesk"', '"Grotesk Fallback"', 'system-ui', 'sans-serif'],
        sans: ['Inter', '"Inter Fallback"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Mono Fallback"', 'Menlo', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
};
