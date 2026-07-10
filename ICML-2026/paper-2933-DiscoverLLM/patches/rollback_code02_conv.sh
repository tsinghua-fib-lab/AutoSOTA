#!/bin/bash
set -e
cd /repo
cp /repo/patches/conversation.py.bak /repo/discoverllm/simulate/conversation.py
echo "CODE-02 conv rolled back"
