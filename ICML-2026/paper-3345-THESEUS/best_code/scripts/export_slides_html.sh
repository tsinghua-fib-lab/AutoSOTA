#!/usr/bin/env bash
cd "$(dirname "$0")"
npx --yes @marp-team/marp-cli@2.0.0 ../docs/repo-overview-slides.md --html -o ../docs/repo-overview-slides.html
