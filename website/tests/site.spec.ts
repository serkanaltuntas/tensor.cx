import { expect, test } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test('homepage leads to installation and has accessible semantics', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/tensor.cx/);
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Tensor compute.');
  const accessibility = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
  expect(accessibility.violations).toEqual([]);
  await page.getByRole('link', { name: 'Start building' }).click();
  await expect(page).toHaveURL('/docs/installation/');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Installation');
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute(
    'href', 'https://tensor.cx/docs/installation/',
  );
});

test('built search returns a useful result and theme changes persist', async ({ page }) => {
  await page.goto('/docs/');
  await page.getByRole('button', { name: 'Search documentation', exact: true }).click();
  await page.getByRole('textbox', { name: 'Search documentation' }).fill('RMSNorm');
  const result = page.locator('.pagefind-ui__result-link').first();
  await expect(result).toBeVisible();
  await result.click();
  await expect(page).toHaveURL(/\/docs\/(operations|backends|examples|experimental|performance)\//);
  await page.getByRole('combobox', { name: 'Select theme' }).selectOption('dark');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  const accessibility = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
  expect(accessibility.violations).toEqual([]);
});

test('all local navigation targets and fragments resolve', async ({ page, request, baseURL }) => {
  const origin = new URL(baseURL!).origin;
  const pending = ['/'];
  const visited = new Set<string>();
  const fragments = new Map<string, Set<string>>();
  while (pending.length) {
    const path = pending.shift()!;
    if (visited.has(path)) continue;
    visited.add(path);
    const response = await page.goto(path);
    expect(response?.status(), path).toBe(200);
    const links = await page.locator('a[href]').evaluateAll((anchors) =>
      anchors.map((anchor) => (anchor as HTMLAnchorElement).href));
    for (const href of links) {
      const url = new URL(href);
      if (url.origin !== origin) continue;
      if (!visited.has(url.pathname)) pending.push(url.pathname);
      if (url.hash) {
        const ids = fragments.get(url.pathname) ?? new Set<string>();
        ids.add(decodeURIComponent(url.hash.slice(1)));
        fragments.set(url.pathname, ids);
      }
    }
  }
  expect(visited.size).toBeGreaterThanOrEqual(10);
  for (const [path, ids] of fragments) {
    await page.goto(path);
    for (const id of ids) {
      expect(await page.evaluate((value) => Boolean(document.getElementById(value)), id), `${path}#${id}`).toBe(true);
    }
  }
  expect((await request.get('/sitemap-index.xml')).ok()).toBe(true);
  expect((await request.get('/favicon.svg')).ok()).toBe(true);
  const missing = await page.goto('/a-page-that-does-not-exist/');
  expect(missing?.status()).toBe(404);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Page not found');
});

test('mobile pages fit and the documentation menu works', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  for (const path of ['/', '/docs/installation/', '/docs/backends/', '/docs/first-tensor/']) {
    await page.goto(path);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), path).toBe(true);
  }
  await page.getByRole('button', { name: 'Menu', exact: true }).click();
  await page.getByRole('link', { name: 'Backends & support', exact: true }).click();
  await expect(page).toHaveURL('/docs/backends/');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Backends & support');
});
