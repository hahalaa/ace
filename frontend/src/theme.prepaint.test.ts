/// <reference types="node" />
/**
 * Root-cause regression guard for the theme-persistence bug.
 *
 * The bug: the pre-paint theme script lived INLINE in index.html, but the
 * deployed CSP (render.yaml) is `script-src 'self'` with no 'unsafe-inline' and
 * no hash. On Render the inline script was silently blocked, so `data-theme` was
 * never stamped, every navigation painted the dark CSS default (theme "reset"),
 * and the toggle read a stored 'light' and mounted with an inverted label. It
 * passed every local check because `vite preview` / dev send no CSP.
 *
 * These assertions read the actual files and pin the invariant that makes the
 * CSP and the pre-paint script agree: the theme logic is an EXTERNAL same-origin
 * script (covered by 'self'), never inline. A future edit that moves it back
 * inline, or a CSP that would block a same-origin script, fails here.
 *
 * @vitest-environment node
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const read = (rel: string) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8');

const indexHtml = read('../index.html');
const themeInit = read('../public/theme-init.js');
const renderYaml = read('../../render.yaml');

describe('pre-paint theme script survives the deployed CSP', () => {
  it('index.html references the external pre-paint script before the module bundle', () => {
    expect(indexHtml).toContain('<script src="/theme-init.js"></script>');
    const initAt = indexHtml.indexOf('/theme-init.js');
    const moduleAt = indexHtml.indexOf('/src/main.tsx');
    // Render-blocking and ordered first, so the attribute is set before render.
    expect(initAt).toBeGreaterThan(-1);
    expect(moduleAt).toBeGreaterThan(-1);
    expect(initAt).toBeLessThan(moduleAt);
  });

  it('index.html contains NO inline <script> with a body (they would be CSP-blocked)', () => {
    // Every <script> in the source HTML must carry a src; none may inline a body.
    const scripts = indexHtml.match(/<script\b[^>]*>[\s\S]*?<\/script>/gi) ?? [];
    for (const tag of scripts) {
      const hasSrc = /\bsrc\s*=/.test(tag);
      const body = tag.replace(/<script\b[^>]*>/i, '').replace(/<\/script>/i, '');
      const inlinesCode = body.trim().length > 0;
      expect(hasSrc && !inlinesCode).toBe(true);
    }
  });

  it('the external script actually applies the theme before paint', () => {
    expect(themeInit).toContain("getItem('ace-theme')");
    expect(themeInit).toContain("setAttribute('data-theme'");
  });

  it("the CSP is script-src 'self' with no 'unsafe-inline', so external is load-bearing", () => {
    // If this ever relaxed to 'unsafe-inline', an inline script would work again
    // and this whole guard would be moot, assert it has NOT, so the external
    // file stays genuinely required rather than incidental.
    expect(renderYaml).toMatch(/script-src 'self'/);
    const cspBlock = renderYaml.slice(
      renderYaml.indexOf('Content-Security-Policy'),
      renderYaml.indexOf('X-Content-Type-Options'),
    );
    expect(cspBlock).not.toContain('unsafe-inline');
  });
});
