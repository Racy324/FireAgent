import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8000',
      '/chat/stream': 'http://localhost:8000',
      '/sessions': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/papers': 'http://localhost:8000',
      '/ingest': 'http://localhost:8000',
      '/detect': 'http://localhost:8000',
    },
  },
})
