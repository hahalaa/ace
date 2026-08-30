/// <reference types="vite/client" />

// Environment contract; anything added here must also appear in .env.example.
interface ImportMetaEnv {
  // Base URL of the ace API. Optional because it is genuinely absent with no .env; apiBaseUrl() turns that into a readable error.
  readonly VITE_API_BASE_URL?: string;
}
