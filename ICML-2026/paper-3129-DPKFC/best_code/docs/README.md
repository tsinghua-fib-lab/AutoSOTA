# DP-KFC, project page

Static, dependency-free project page (one `index.html` + `assets/style.css` + `assets/main.js`).
Dark theme, animated hero, scroll-reveal sections, scroll-progress bar, copy-able BibTeX.
Only external resources are Google Fonts and a MathJax CDN script (both optional, the page
works without them, just less pretty / equations show as TeX source).

## Figures

The page pulls these from `docs/static/images/` (already converted from the paper PDFs with
`pdftocairo -png -r 220 -singlefile <fig>.pdf <fig>`; the source PDFs are kept alongside the PNGs
and can be deleted if you want a leaner repo). Any missing image is replaced by a small inline
placeholder note, so the layout never breaks.

| file (PNG) | paper figure | shown in |
|---|---|---|
| `snr_motivation.png` | Fig. 1, layer-wise SNR | *Problem* |
| `spectrum_ablation_simple_full.png` | Fig. 2, eigenspectrum alignment | *Key idea* |
| `cov_tracking_synthetic_pink_eps1.0_summary.png` | Fig. 3, covariance tracking | *Why it works* |
| `vision_combined_results.png` | Fig. 4, vision privacy-utility | *Results* |
| `nlp_combined_results.png` | Fig. 5, NLP privacy-utility | *Results* |
| `static/og.png` | 1200×630 social-card image (optional) | `<meta>` only |

Still to add: the camera-ready PDF at `static/dp_kfc.pdf` (the "Paper (PDF)" badge points there),
and an arXiv ID, replace the `#` in the arXiv badge in `index.html`.

To re-export from PDFs later: `pdftocairo -png -r 220 -singlefile fig.pdf fig` (or
`pdftoppm -png -r 220 fig.pdf fig`). If you have `pngquant`, `pngquant --quality=70-92 --ext .png *.png`
shrinks them ~3×.

## Deploy on GitHub Pages

1. Commit this `docs/` folder to the default branch.
2. Repo → **Settings → Pages** → *Build and deployment* → Source = **Deploy from a branch**,
   Branch = `main` (or whatever your default is), Folder = **`/docs`** → Save.
3. After a minute it's live at `https://<user-or-org>.github.io/DP-KFC/`
   (for this repo: `https://molinamarcvdb.github.io/DP-KFC/`, adjust if the repo lives under a
   different account/org).
4. Push = redeploy. The `.nojekyll` file tells Pages to serve the folder as-is (no Jekyll build).

Custom domain (optional): add a `docs/CNAME` file containing your domain and set the DNS record,
then enable "Enforce HTTPS" in the Pages settings.

### Alternative hosts
The same files deploy unchanged on Netlify / Vercel / Cloudflare Pages (point them at the `docs/`
directory as the publish root, no build command).

## Editing

Everything is in `index.html`; styling tokens (colors, accent gradient, max width) are CSS
variables at the top of `assets/style.css`. The accent palette is pink → violet → cyan on
near-black; change `--accent`, `--accent-2`, `--accent-3` to retheme.
