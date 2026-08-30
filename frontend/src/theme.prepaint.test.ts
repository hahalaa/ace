/// <reference types="node" />
// @vitest-environment node
// Regression guard: the pre-paint theme script must stay an external same-origin file, and the CSP must have no 'unsafe-inline'.

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
    expect(renderYaml).toMatch(/script-src 'self'/);
    const cspBlock = renderYaml.slice(
      renderYaml.indexOf('Content-Security-Policy'),
      renderYaml.indexOf('X-Content-Type-Options'),
    );
    expect(cspBlock).not.toContain('unsafe-inline');
  });
});
