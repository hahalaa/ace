// Stamp <html data-theme> before first paint. Must stay an external file: the deployed CSP blocks inline scripts.
// Mirrors resolveTheme() in src/theme.ts by hand (a classic script cannot import a module): stored choice wins, else dark.
(function () {
  try {
    var stored = window.localStorage.getItem('ace-theme');
    var theme = stored === 'light' || stored === 'dark' ? stored : 'dark';
    document.documentElement.setAttribute('data-theme', theme);
  } catch (e) {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
})();
