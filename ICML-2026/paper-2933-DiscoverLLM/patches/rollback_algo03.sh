#!/bin/bash
set -e
cd /repo
cp /repo/patches/hierarchize_criteria.yaml.orig /repo/discoverllm/core/prompts/hierarchize_criteria.yaml
echo "ALGO-03 rolled back"
