#!/bin/bash
set -e
cd /repo
cp /repo/patches/user.json.bak /repo/eval_configs/user.json
echo "PARAM-01 rolled back"
