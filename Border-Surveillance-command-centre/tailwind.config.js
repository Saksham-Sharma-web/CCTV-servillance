/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Indian government-inspired palette
        navy: {
          DEFAULT: '#102A43',
          dark: '#0B1F33',
        },
        steel: {
          DEFAULT: '#3E6078',
          light: '#1F4E79',
        },
        background: {
          DEFAULT: '#F4F5F3',
          white: '#FFFFFF',
        },
        border: {
          DEFAULT: '#D6DADF',
          dark: '#C4CBD2',
        },
        text: {
          primary: '#17202A',
          secondary: '#5F6B76',
        },
        status: {
          critical: '#B42318',
          warning: '#B54708',
          success: '#18794E',
          info: '#175CD3',
        },
        accent: {
          saffron: '#E87817',
          green: '#138808',
        }
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
      },
      boxShadow: {
        'soft': '0 2px 4px 0 rgba(0, 0, 0, 0.05)',
        'medium': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
      }
    },
  },
  plugins: [],
}
