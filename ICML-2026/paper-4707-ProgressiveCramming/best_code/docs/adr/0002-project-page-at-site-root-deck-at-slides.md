# Project page at the Pages site root; video deck stays at /slides

Today the GitHub Pages root is just an HTML redirect to the reveal.js deck under
`/slides` (see `.github/workflows/deploy-presentation.yml`). We will serve the new
**project page at the site root** and keep the deck at `/slides`, replacing the
redirect with the real landing page. This is what the poster's "Project Page" QR
code already points at (the repo root URL), and a landing page is the conventional
destination for a paper's project link.

## Consequences

- The deploy workflow must stage the project page into `_site/` root (instead of
  writing the redirect) while still copying `presentation/` into `_site/slides/`,
  and must trigger on the project-page source paths as well as `presentation/**`.
- The project page links *out* to `/slides` for the video deck, so the deck stays
  discoverable.
