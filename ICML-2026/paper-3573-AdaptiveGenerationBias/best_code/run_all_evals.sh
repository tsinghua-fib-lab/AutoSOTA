#!/bin/bash
set -e
cd /repo
rm -rf cab_download/explicit/model_evals/model_evals_*

# Update configs to use max_workers: 4
for attr in gender race religion; do
  sed -i 's/max_workers: 2/max_workers: 4/g' "config_eval_${attr}_30.yaml"
done

export OPENROUTER_API_KEY="YOUR_API_KEY"
export OPENROUTER_BASE_URL="https://api.deepseek.com/v1"

for attr in gender race religion; do
    echo "=== ${attr} start: $(date) ==="
    python3 -u main.py --config_path "config_eval_${attr}_30.yaml" > "/tmp/eval_${attr}_v2.log" 2>&1
    ret=$?
    echo "=== ${attr} done: $(date), exit=${ret} ==="
    echo "Errors:"
    grep -c 'Exception\|error' "/tmp/eval_${attr}_v2.log" || true
    echo "Summary:"
    grep -E 'Mean|Completed|Summary' "/tmp/eval_${attr}_v2.log" | tail -10 || true
done
echo "ALL DONE at $(date)"
