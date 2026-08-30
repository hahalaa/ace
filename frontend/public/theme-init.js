/*
 * Stamp <html data-theme> before first paint so there is no flash of the wrong
 * ground on each navigation. Must stay an external same-origin file, never an
 * inline <script>: the deployed CSP is `script-src 'self'` with no
 * 'unsafe-inline', which blocks inline scripts outright. A classic script cannot
 * import a module, so this mirrors resolveTheme() in src/theme.ts by hand:
 * stored choice wins, else dark; the OS prefers-color-scheme is not consulted.
 */
(function () {
  try {
    var stored = window.localStorage.getItem('ace-theme');
    var theme = stored === 'light' || stored === 'dark' ? stored : 'dark';
    document.documentElement.setAttribute('data-theme', theme);
  } catch (e) {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
})();
