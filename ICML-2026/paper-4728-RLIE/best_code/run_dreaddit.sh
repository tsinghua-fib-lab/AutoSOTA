#!/bin/bash
# RLIE Dreaddit reproduction script
# Requires: RLIE_BASE_URL and RLIE_API_KEY environment variables
set -e
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy no_proxy NO_PROXY
export RLIE_BASE_URL="${RLIE_BASE_URL:-https://api.deepseek.com}"
export RLIE_API_KEY="${RLIE_API_KEY}"
export PYTHONUNBUFFERED=1
cd /repo
exec python -m rlie.main --config configs/dreaddit_prod.yaml
