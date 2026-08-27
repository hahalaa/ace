/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
//
// No build option is customised for the static-build/deploy path: every default
// is already the one a static host wants. Recorded here so the next reader does
// not re-derive it, and so a future change is a deliberate departure rather than
// an accident.
//
//   * `build.outDir` stays `dist`, the publish directory `render.yaml` names.
//   * `base` stays `/`, the app is served from a domain root. Hosting it under
//     a sub-path (GitHub Pages project sites) is the one case that needs a
//     `base` here, and it needs a rebuild, not a config toggle at deploy time.
//   * No SSR, no `ssr`/`build.ssrManifest`, the output is plain static files.
//   * **`VITE_API_BASE_URL` is inlined at build time**, by Vite's normal
//     `import.meta.env` substitution; nothing here changes that and nothing
//     should. `apiBaseUrl()` in `src/api/client.ts` compiles down to a string
//     literal, so one built bundle talks to exactly one API and pointing it at
//     another requires a **rebuild**. `frontend/.env.example` is the tracked
//     local value.
//
// The `test` block below is vitest's, not the build's.
//
// The one behavioural addition is the warning under `command === 'build'`: a
// build with no `VITE_API_BASE_URL` otherwise succeeds **completely silently**
// and ships a bundle whose every screen renders the ApiConfigError panel. It
// warns rather than throws on purpose, see the note on the check itself.
export default defineConfig(({ command, mode }) => {
  if (command === 'build') {
    // `loadEnv` resolves exactly as the build does, `.env` files plus any
    // VITE_-prefixed shell variable, shell winning, so this cannot disagree
    // with what actually gets inlined.
    const env = loadEnv(mode, process.cwd(), 'VITE_')
    if ((env.VITE_API_BASE_URL ?? '').trim() === '') {
      // Deliberately a warning, not a hard failure. `npm run build` is also the
      // CI build-check command, and that job has no API to point at; a
      // throw here would either break CI on every PR and fork or force a dummy
      // value into the workflow, which trains people to satisfy the check
      // rather than to set the real URL. The deploy-time defence is that this
      // is loud in the host's build log; the runtime defence is the
      // ApiConfigError panel, which names the file to fix.
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
      // node, not jsdom: the API client tests touch no DOM. Component tests
      // opt into jsdom and testing-library where they need them.
      environment: 'node',
      include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    },
  }
})
