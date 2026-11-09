"""Print compact status without dumping carriage-return progress bars."""
import json
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for name in ['width_trial', 'physics_trial', 'pseudo_trial', 'independent_noise_trial', 'real_event_wavegram_gpu2',
             'five_fold_optimized', 'campaign12h', 'campaign10h', 'campaign14h', 'campaign_final']:
    path = ROOT / 'reports' / (name+'_status.json')
    if not path.exists():
        continue
    status = json.loads(path.read_text())
    print(name+': '+status['stage'])
    if status.get('deadline_china'):
        print('  Deadline: '+status['deadline_china'])
        remaining = dt.datetime.fromisoformat(status['deadline_china'])-dt.datetime.now(dt.timezone.utc)
        print('  Remaining wall-clock budget: '+str(round(max(0,remaining.total_seconds())/3600,2))+' hours')
    for gpu, worker in status.get('gpu_workers', {}).items():
        print('  GPU '+gpu+': '+worker['state'])
    for tag, job in status['jobs'].items():
        if job['state'] != 'running':
            continue
        print('  '+tag+' (PID '+str(job.get('pid'))+')')
        log = Path(job['log'])
        if log.exists():
            text = log.read_text(errors='replace')
            lines = [line for line in text.replace('\r', '\n').splitlines() if line]
            scores = [line for line in lines if 'valid_score:' in line or 'valid_metric=' in line]
            for line in scores[-4:]:
                print('    '+line[:300])
            if lines:
                print('    '+lines[-1][:300])
