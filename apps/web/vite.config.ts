import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Khi dev (make dev), gọi /api được chuyển tiếp sang FastAPI ở cổng 8000.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/docs': 'http://localhost:8000',
      '/openapi.json': 'http://localhost:8000',
    },
  },
})
