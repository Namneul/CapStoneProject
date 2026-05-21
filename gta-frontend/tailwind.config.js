// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#0066FF',
        warning: '#FF8A00',
        background: '#F8F9FA'
      },
      boxShadow: {
        'soft': '0 10px 40px -10px rgba(0,0,0,0.05)',
        'floating': '0 20px 40px -20px rgba(0,102,255,0.2)'
      },
      fontFamily: {
        sans: ['Pretendard', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
