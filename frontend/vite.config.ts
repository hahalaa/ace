/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// No build option is customised: every Vite default is already what a static host wants.
// VITE_API_BASE_URL is inlined at build time, so one bundle talks to exactly one API.
export default defineConfig(({ command, mode }) => {
  if (command === 'build') {
    // loadEnv resolves .env files + VITE_-prefixed shell vars exactly as the build does.
    const env = loadEnv(mode, process.cwd(), 'VITE_')
    if ((env.VITE_API_BASE_URL ?? '').trim() === '') {
      // A warning, not a throw: `npm run build` is also the CI check, which has no API to point at.
      console.warn(
        '\n\x1b[33m[ace] warning: VITE_API_BASE_URL is not set.\x1b[0m\n' +
          '  This build will succeed, but the bundle has no API to call and every\n' +
          '  screen will render the "API base URL is not configured" panel.\n' +
          '  The value is inlined at BUILD time, setting it later does not fix an\n' +
          '  existing bundle; it needs a rebuild. See frontend/.env.example.\n',
      )
    }
  }

  return {
    plugins: [react()],
    test: {
      // node, not jsdom: component tests opt into jsdom where they need it.
      environment: 'node',
      include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    },
  }
})
