/// <reference types="vite/client" />

/**
 * Environment contract for this app. Anything added here must also be
 * documented in `.env.example` — that file is the only place a default value
 * is written down, since none may be hardcoded in source.
 */
interface ImportMetaEnv {
  /**
   * Base URL of the ace API, e.g. `http://localhost:8000` — no trailing slash
   * required. Optional in the type because it is genuinely absent when nobody
   * created a `.env`; `apiBaseUrl()` turns that into a readable error.
   */
  readonly VITE_API_BASE_URL?: string;
}
