#!/bin/bash
cd /repo
git config --global --add safe.directory /repo 2>/dev/null
rm -f .git/index.lock
git add -A
git commit -q -m 'baseline' --allow-empty 2>&1
git tag -f _baseline 2>&1
HEAD=$(git rev-parse HEAD)
echo "BASELINE_HEAD=$HEAD" > /repo/_baseline_commit.txt
echo "GIT_SETUP_DONE" >> /repo/_baseline_commit.txt
