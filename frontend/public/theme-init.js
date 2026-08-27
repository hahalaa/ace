/*
 * Pre-paint theme application. Runs before first paint to stamp
 * <html data-theme> so there is no flash of the wrong ground, and so a
 * navigation (every screen is a full document load, see App.tsx) restores the
 * visitor's stored choice before anything renders.
 *
 * This is a SEPARATE, same-origin classic script on purpose, NOT an inline
 * <script> in index.html. The deployed Content-Security-Policy is
 * `script-src 'self'` with no 'unsafe-inline' and no per-script hash (see
 * render.yaml): a same-origin file satisfies 'self' and runs; an inline script
 * would be silently BLOCKED, which is exactly the regression that shipped,
 * the pre-paint script never ran on Render, so navigations reset the theme to
 * the dark CSS default and the toggle's label inverted against it. Keeping this
 * logic in a file means a future edit to it can never re-break CSP (there is no
 * hash to forget to update).
 *
 * It mirrors resolveTheme() in src/theme.ts by hand, a classic script cannot
 * import a module, and this is the one place that logic is duplicated. Stored
 * choice wins; otherwise the dark default. The OS prefers-color-scheme is
 * deliberately NOT consulted: a first visit lands on the dark brand identity
 * regardless of OS, and only an explicit toggle choice moves off it. Kept tiny
 * and failure-tolerant on purpose.
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
