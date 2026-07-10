#!/bin/bash
set -e
cd /repo
cp /repo/patches/assistant_simulator.py.bak /repo/discoverllm/pipeline/assistant_simulator.py
echo "CODE-02 rolled back"
