# tensor.cx website

The product homepage and curated user documentation for `tensor.cx`, built with
Astro and Starlight. Independent tensor.cx identity; English content.
The website is part of the source repository, not a separate Git repository or
a Python runtime dependency.

## Develop and verify

Use Node.js 24 (see `.nvmrc`) and npm. Run from `website/`:

```bash
npm ci
npm run dev
```

Before committing:

```bash
npm run check
npm run build
npx playwright install chromium
npm test
npm run test:cloudflare
npm run deploy:check
```

Tests exercise the production build: navigation and fragments, Pagefind search,
theme persistence, mobile layout/menu, and WCAG accessibility checks on the
homepage and a documentation page. The Cloudflare suite runs the same checks
against local Workers Static Assets, including trailing-slash routing and 404
status codes. `deploy:check` rebuilds and packages a dry run; it uploads nothing
and does not change hosting or DNS. Browser/system dependencies may need
`npx playwright install --with-deps chromium` on a fresh Linux machine.

From the repository root, with the runtime already installed:

```bash
uv run --no-sync python website/src/snippets/first-tensor.py
uv run --no-sync python website/src/snippets/operations.py
```

CI runs website checks separately from native builds and executes the shared
snippets with the CPU runtime. Dependency versions and `package-lock.json` are
committed. Build output, dependencies, and browser reports stay ignored.

## Content ownership

- `src/pages/index.astro`: product homepage.
- `src/content/docs/docs/`: curated user guides, served under `/docs/`.
- `src/snippets/`: runnable Python examples imported into pages as source text.
- `src/styles/docs.css`: Starlight theme adjustments.
- `public/`: explicitly reviewed public assets only.

`../PROJECT.md` remains the phase ledger; `../docs/` remains the source for
engineering decisions and validation evidence. Link to those records instead
of copying them wholesale. Do not add a loader for the repository's docs tree,
private notes, raw reports, or build directories. Update the support matrix
when validated runtime capabilities change. Autograd is not implemented.

## Cloudflare deployment preparation

Build command: `npm ci && npm run build` with project root `website/`.
Publish directory: `website/dist/` (or `dist/` relative to that root).
Serve directory indexes and use `404.html` for missing pages. `npm run preview`
is for local verification, not a production server. Search is generated during
the build and works in the preview; the dev server does not build its index.

`wrangler.jsonc` targets the `tensorcx-website` Worker with static assets only,
as described in the [Cloudflare Astro guide](https://developers.cloudflare.com/workers/framework-guides/web-apps/astro/).
No server adapter, Worker handler, R2 bucket, database, or runtime API is required.
There are no bindings or `Env` types to generate. Canonical URLs and the sitemap
target `https://tensor.cx`; both `tensor.cx` and `www.tensor.cx` serve the site.
HTML paths use trailing slashes; missing pages return `404.html` with HTTP 404.
This Worker's `workers.dev` and version preview URLs are disabled. The config
enables logs and sampled traces for any future Worker execution; static asset
requests do not invoke a user Worker. No external fonts or analytics are used.

Preparation and CI perform local checks only. No deployment has been performed
as part of this setup. Account identifiers and credentials are supplied outside
the repository through the operator's authenticated environment; never put them
in this config, a tracked `.env` file, or the built assets.

Before the first authorized deployment:

1. Confirm the intended Cloudflare account owns `tensor.cx` and the Worker name
   is free or belongs to this site. Remove legacy forwarding rules.
2. Preserve Google verification TXT and domain registration/nameservers. Review
   existing apex/www DNS records: binding the custom domains may require
   replacing placeholder records. A dry run does not validate remote DNS or TLS.
3. Complete repository publication checks and resolve source-link visibility.
   The installation page currently discloses private repository access; update
   it when that changes. No PyPI release is claimed.
4. Run the checks above. Only when publishing is explicitly authorized, run
   `npm run build` followed by the project-local `wrangler deploy` through the
   operator's credential wrapper. CI contains no deployment or secret setup.
5. Verify HTTPS, both hostnames, installation pages, search, canonical URLs and
   a missing page on the live site, and confirm the Google TXT record remains.

For manual local inspection, build first and run `npm run preview:cloudflare`.
This binds only to `127.0.0.1:8787` and uses no remote resources.
