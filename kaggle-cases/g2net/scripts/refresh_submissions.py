"""Refresh existing submission records without uploading additional files."""
import argparse
import datetime
import json
import time
from pathlib import Path

from kaggle import api

from submit_checked import COMPETITION, ROOT, summarize


def refresh():
    entries = {str(s.ref): s for s in (api.competition_submissions(COMPETITION) or [])}
    records = []
    pending = False
    for path in sorted((ROOT / 'reports/submissions').glob('*.json')):
        record = json.loads(path.read_text())
        ref = (record.get('submission') or {}).get('ref')
        selected = entries.get(str(ref))
        if selected is None:
            matches = [s for s in entries.values() if s.description == record['message']]
            selected = matches[0] if len(matches) == 1 else None
        if selected is not None:
            record['submission'] = summarize(selected)
            record['last_checked_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            path.write_text(json.dumps(record, indent=2))
            score = record['submission']
            pending |= (score['public_score'] is None and score['private_score'] is None
                        and getattr(selected.status, 'name', '') not in ['ERROR', 'FAILED', 'CANCELED', 'CANCELLED'])
        else:
            pending = True
        records.append(record)
    scored = [r for r in records if r.get('submission')]
    status_path = ROOT / 'reports/project_status.json'
    status = json.loads(status_path.read_text())
    for field in ['public', 'private']:
        scores = [r['submission'][field+'_score'] for r in scored if r['submission'][field+'_score'] is not None]
        status['best_submitted_'+field+'_auc'] = max(scores) if scores else None
    status['kaggle_submissions'] = len(records)
    status['scores_last_checked_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    status['champion_exceeded'] = any(
        (r['submission']['public_score'] or 0) > status['target_public_auc_gt']
        and (r['submission']['private_score'] or 0) > status['target_private_auc_gt'] for r in scored)
    for field in ['public', 'private']:
        status['champion_'+field+'_exceeded'] = any(
            (r['submission'][field+'_score'] or 0) > status['target_'+field+'_auc_gt'] for r in scored)
    status['champion_both_exceeded'] = status['champion_exceeded']
    ranked = [(path, record) for path, record in zip(sorted((ROOT / 'reports/submissions').glob('*.json')), records)
              if record.get('submission') and (record['submission'].get('private_score') is not None
                                              or record['submission'].get('public_score') is not None)]
    if ranked:
        path, best = max(ranked, key=lambda item: (
            item[1]['submission']['private_score'] if item[1]['submission']['private_score'] is not None else -1,
            item[1]['submission']['public_score'] if item[1]['submission']['public_score'] is not None else -1))
        status['best_submission_ref'] = best['submission']['ref']
        status['best_submission_report'] = str(path.relative_to(ROOT))
    for field in ['public', 'private']:
        score = status['best_submitted_'+field+'_auc']
        status[field+'_auc_gap_to_champion'] = (round(status['target_'+field+'_auc_gt']-score, 8)
                                                if score is not None else None)
    status_path.write_text(json.dumps(status, indent=2))
    return records, pending


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-seconds', type=int, default=0)
    args = parser.parse_args()
    deadline = time.monotonic()+args.wait_seconds
    while True:
        records, pending = refresh()
        if not pending or time.monotonic() >= deadline:
            print(json.dumps(records, indent=2), flush=True)
            break
        time.sleep(15)


if __name__ == '__main__':
    main()
