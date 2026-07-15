# Changelog

All notable changes to TSL are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com) and the project adheres to
[Semantic Versioning](https://semver.org).

## [0.3.0] - 2026-06-03

### Documentation

- Align README and docs with the code, link the docs site (#24)

### Refactor

- **tsl-py:** Rename import package `tsl_py` to `tensorsl` — the import name now
  matches the `tensorsl` PyPI distribution; `import tsl_py` no longer works

## [0.2.0] - 2026-06-03

- Add R wrapper `tensorsl` with S3 interface and ggplot2 interpretability layer

## [0.1.3] - 2026-06-02

### Features

- **plot:** Complete plotting overhaul (#22)
## [0.1.1] - 2026-06-02

### Refactor

- Solve boosting OLS with a pure-Rust SVD, dropping OpenBLAS (#19)
## [0.1.0] - 2026-06-01

### Bug Fixes

- Handle non-contiguous arrays in fit and predict (#6)
- **tsl-py:** Use a package-local README for the sdist (#17)
- **tsl-py:** Keep package metadata free of direct-URL dependencies (#18)

### Documentation

- Polish readme
- Documentation overhaul (README, CLAUDE.md, CONTRIBUTING) (#4)
- Enlarge readme logo and example figures (#5)
- MkDocs documentation site + GitHub Pages deployment (#7)

