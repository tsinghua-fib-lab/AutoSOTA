#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/bin:$PATH

ITER="$1"
IDEA_ID="$2"
TITLE="$3"
shift 3

cd /repo/src/static_offline

echo "=== Running iter=$ITER: $TITLE ==="
echo "Args: $@"

python3 main.py --dataset Amazon "$@" > /dev/null 2>&1
actual_log=$(ls -t logs/SASRec_Amazon_*.log 2>/dev/null | head -1)

echo "Parsing metrics from: $actual_log"

python3 << 'PYEOF'
import re, json, sys, os

log_file = os.popen('ls -t logs/SASRec_Amazon_*.log 2>/dev/null | head -1').read().strip()

with open(log_file, 'r') as f:
    content = f.read()

lines = content.replace('\r', '\n').split('\n')

def parse_table_section(lines, section_label):
    metrics = {}
    in_table = False
    for line in lines:
        line_s = line.strip()
        if section_label in line_s:
            in_table = True
            continue
        if in_table:
            if '====' in line_s or line_s.startswith('Avg HD'):
                break
            # Skip separator lines
            if line_s.startswith('---') or all(c in '- |' for c in line_s):
                continue
            parts = [p.strip() for p in line_s.split('|')]
            if len(parts) >= 2:
                metric_name = parts[0]
                try:
                    global_val = float(parts[1])
                    metrics[metric_name] = global_val
                except (ValueError, IndexError):
                    continue
    return metrics

baseline_metrics = parse_table_section(lines, 'Baseline Stratified')
coral_metrics = parse_table_section(lines, 'CORAL Stratified')

trigger_rate = 0.0
for line in lines:
    if 'Intervention Trigger Rate' in line:
        m = re.search(r'([\d.]+)%', line)
        if m:
            trigger_rate = float(m.group(1)) / 100.0
        break

sat_improvement = 0.0
for line in lines:
    if 'Avg Improvement' in line:
        m = re.search(r'([\d.-]+)', line)
        if m:
            sat_improvement = float(m.group(1))
        break

output = {
    'baseline': baseline_metrics,
    'coral': coral_metrics,
    'trigger_rate': trigger_rate,
    'sat_improvement': sat_improvement
}

print(json.dumps(output))
PYEOF
