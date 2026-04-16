import {defineConfig, loadEnv} from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({mode}) => {
  // Load .env files; also picks up VITE_BACKEND from process.env (Docker ARG → ENV)
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [react(), tailwindcss()],
    server: {port: 5173},
    build: {outDir: 'dist'},
    // __VITE_BACKEND__ is replaced at build time — no import.meta at runtime.
    // This lets Jest import api.ts without SyntaxError from import.meta.
    define: {
      __VITE_BACKEND__: JSON.stringify(env.VITE_BACKEND || ''),
    },
  };
});
