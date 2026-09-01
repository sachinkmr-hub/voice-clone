import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * The dev server proxies to the FastAPI backend so the dashboard runs on :5173 without
 * CORS configuration or a second origin. In production the built assets are served by any static
 * host and VITE_API_BASE points at the API.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/v1': { target: 'http://127.0.0.1:8000', changeOrigin: true, ws: true },
      '/docs': 'http://127.0.0.1:8000',
    },
  },
  build: { outDir: 'dist', sourcemap: true, target: 'es2020' },
});
