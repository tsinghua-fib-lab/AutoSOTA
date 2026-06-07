"""Record iteration results."""
import subprocess, json, os, sys
from datetime import datetime

GIT_BIN = '/repo/git_bin'
GIT_ENV = {**os.environ, 'LD_LIBRARY_PATH': '/repo:' + os.environ.get('LD_LIBRARY_PATH', '')}

def git(*args):
    r = subprocess.run([GIT_BIN] + list(args), cwd='/repo', env=GIT_ENV, capture_output=True, text=True, timeout=30)
    return r.stdout.strip(), r.stderr.strip(), r.returncode

if len(sys.argv) < 2:
    print("Usage: record_score.py <scores_jsonl>")
    sys.exit(1)

scores_path = sys.argv[1]
iter_num = sys.argv[2] if len(sys.argv) > 2 else "1"
idea_id = sys.argv[3] if len(sys.argv) > 3 else "IDEA-001"
title = sys.argv[4] if len(sys.argv) > 4 else "Soft Sigmoid"
status = sys.argv[5] if len(sys.argv) > 5 else "success"
primary = float(sys.argv[6]) if len(sys.argv) > 6 else 27.12
metrics_json = sys.argv[7] if len(sys.argv) > 7 else '{}'
notes = sys.argv[8] if len(sys.argv) > 8 else ''
is_best = sys.argv[9] if len(sys.argv) > 9 else 'true'

git('config', '--global', '--add', 'safe.directory', '/repo')
git('config', 'user.name', 'optimizer')
git('config', 'user.email', 'opt@local')
git('add', '-A')
msg = f"iter-{iter_num}: {title} [{status}]"
git('commit', '-q', '-m', msg, '--allow-empty')
commit_hash, _, _ = git('rev-parse', 'HEAD')

if status == 'success' and is_best.lower() == 'true':
    git('tag', '-f', '_best')

record = {
    'iter': iter_num,
    'idea_id': idea_id,
    'title': title,
    'status': status,
    'primary_metric': primary,
    'metrics': json.loads(metrics_json),
    'commit_hash': commit_hash,
    'notes': notes,
    'timestamp': datetime.now().isoformat()
}

os.makedirs(os.path.dirname(scores_path), exist_ok=True)
with open(scores_path, 'a') as f:
    f.write(json.dumps(record) + '\n')

print(f"[record] iter={iter_num} primary={primary} hash={commit_hash[:10]} best_tag={is_best}")
