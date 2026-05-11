/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono:    ['"JetBrains Mono"', 'monospace'],
        display: ['"Space Grotesk"', 'sans-serif'],
      },
      colors: {
        ink:    '#0a0a12',
        panel:  '#111120',
        border: '#1e1e38',
        muted:  '#3a3a5c',
        dim:    '#7070a0',
        text:   '#d0d0e8',
        bright: '#eeeeff',
        accent: '#5b6ef5',
        green:  '#00d68f',
        amber:  '#f5a623',
        red:    '#f03e3e',
      },
    },
  },
  plugins: [],
}
