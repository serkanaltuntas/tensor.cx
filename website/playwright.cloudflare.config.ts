import { defineConfig } from '@playwright/test';
import base from './playwright.config';

export default defineConfig({
  ...base,
  testIgnore: [],
  use: { ...base.use, baseURL: 'http://127.0.0.1:8787' },
  webServer: {
    command: 'npm run preview:cloudflare',
    url: 'http://127.0.0.1:8787',
    reuseExistingServer: false,
  },
});
