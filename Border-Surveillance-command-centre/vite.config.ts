import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Express backend (Supabase auth, stream token issuance)
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      },
      // Rust edge backend — plain HTTP port 4000 (no TLS cert issues).
      // Rewrites /edge-api/* → /api/* on the Rust server.
      // MJPEG streams are piped directly through — no buffering.
      '/edge-api': {
        target: 'http://localhost:4000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/edge-api/, '/api'),
      }
    }
  }
})
