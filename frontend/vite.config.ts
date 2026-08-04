/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    // node, not jsdom: T4.1 tests the API client, which touches no DOM. A
    // component ticket (T4.2+) adds jsdom and testing-library when it needs them.
    environment: 'node',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  },
})
