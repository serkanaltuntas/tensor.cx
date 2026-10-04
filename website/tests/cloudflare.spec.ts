import { expect, test } from '@playwright/test';

test('static hosting normalizes documentation URLs without losing query parameters', async ({ request }) => {
  const response = await request.get('/docs/installation?from=test', { maxRedirects: 0 });
  expect(response.status()).toBe(307);
  const target = new URL(response.headers().location, response.url());
  expect(target.origin).toBe(new URL(response.url()).origin);
  expect(target.pathname).toBe('/docs/installation/');
  expect(target.search).toBe('?from=test');
  expect((await request.get(target.href)).status()).toBe(200);
});
