/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50:  '#eff6ff',
          100: '#dbeafe',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          900: '#1e3a8a',
        },
        legal: {
          dark:  '#1a2744',
          mid:   '#1f4e79',
          light: '#4a90d9',
        },
      },
    },
  },
  plugins: [],
}
