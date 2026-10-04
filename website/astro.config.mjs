import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://tensor.cx',
  output: 'static',
  trailingSlash: 'always',
  integrations: [
    starlight({
      title: 'Cortex Runtime',
      description: 'A compact Python-first tensor runtime with a C++20 core, CPU references, and Metal and CUDA backends.',
      favicon: '/favicon.svg',
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/serkanaltuntas/cortex-runtime' }],
      customCss: ['./src/styles/docs.css'],
      sidebar: [
        { label: 'Start here', items: [
          { label: 'Introduction', slug: 'docs' },
          { label: 'Installation', slug: 'docs/installation' },
          { label: 'Your first tensor', slug: 'docs/first-tensor' },
        ] },
        { label: 'Use the runtime', items: [
          { label: 'Tensor operations', slug: 'docs/operations' },
          { label: 'Backends & support', slug: 'docs/backends' },
          { label: 'Examples', slug: 'docs/examples' },
          { label: 'Experimental kernels', slug: 'docs/experimental' },
        ] },
        { label: 'Project', items: [
          { label: 'Performance', slug: 'docs/performance' },
          { label: 'Contributing', slug: 'docs/contributing' },
        ] },
      ],
    }),
  ],
});
