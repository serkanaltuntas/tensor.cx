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
```

Tests exercise the production build: navigation and fragments, Pagefind search,
theme persistence, mobile layout/menu, and WCAG accessibility checks on the
homepage and a documentation page. Browser/system dependencies may need
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

## Static deployment

Build command: `npm ci && npm run build` with project root `website/`.
Publish directory: `website/dist/` (or `dist/` relative to that root).
Serve directory indexes and use `404.html` for missing pages. `npm run preview`
is for local verification, not a production server. Search is generated during
the build and works in the preview; the dev server does not build its index.

Canonical URLs and the sitemap target `https://tensor.cx`. No backend, external
font service, analytics, or runtime API is required. This change creates and
validates the site only; it does not provision hosting, alter DNS, or publish it.
Before launch, connect a static host and HTTPS, verify the domain's routing,
complete the repository's publication checks, and make source links accessible.
The installation page currently discloses private repository access; update it
when that changes. No PyPI release is claimed.
