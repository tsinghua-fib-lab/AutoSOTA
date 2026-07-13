import glob, re, json, sys

log = sorted(glob.glob('logs/SASRec_Amazon_*.log'))[-1]
with open(log) as f:
    content = f.read()

lines = content.replace('\r', '\n').split('\n')

def parse_table(section_label):
    metrics = {}
    in_table = False
    for line in lines:
        s = line.strip()
        if section_label in s:
            in_table = True
            continue
        if in_table:
            if s.startswith('===') or s.startswith('Avg HD') or s == '':
                break
            if s.startswith('---') or all(c in '- |' for c in s):
                continue
            parts = [p.strip() for p in s.split('|')]
            if len(parts) >= 2:
                try:
                    metrics[parts[0]] = float(parts[1])
                except (ValueError, IndexError):
                    pass
    return metrics

baseline = parse_table('Baseline Stratified Risk')
coral = parse_table('CORAL Stratified Risk')

trigger_rate = 0.0
for line in lines:
    if 'Intervention Trigger Rate' in line:
        m = re.search(r'([\d.]+)\s*%', line)
        if m:
            trigger_rate = float(m.group(1)) / 100.0
        break

sat_imp = 0.0
for line in lines:
    if 'Avg Improvement' in line:
        m = re.search(r'(-?\d+\.\d+)', line)
        if m:
            sat_imp = float(m.group(1))
        break

result = {
    'baseline_R10': baseline.get('Recall@10', 0),
    'baseline_M10': baseline.get('MRR@10', 0),
    'baseline_TCC10': baseline.get('TCC@10', 0),
    'baseline_SatPost': baseline.get('Sat(Post)', 0),
    'coral_R10': coral.get('Recall@10', 0),
    'coral_M10': coral.get('MRR@10', 0),
    'coral_TCC10': coral.get('TCC@10', 0),
    'coral_SatPost': coral.get('Sat(Post)', 0),
    'trigger_rate': trigger_rate,
    'sat_improvement': sat_imp
}

print(json.dumps(result))
