"""Validate a full-data prediction artifact and submit using configured Kaggle credentials."""
import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from kaggle import api

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = 'g2net-gravitational-wave-detection'


def summarize(submission):
    names = ['ref', 'date', 'description', 'status', 'public_score', 'private_score', 'error_description']
    result = {name: str(getattr(submission, name, '')) for name in names}
    for name in ['public_score', 'private_score']:
        value = getattr(submission, name, None)
        try:
            score = float(value)
            result[name] = score if np.isfinite(score) else None
        except (TypeError, ValueError):
            result[name] = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--message', required=True)
    parser.add_argument('--wait-seconds', type=int, default=600)
    args = parser.parse_args()
    evidence = json.loads((args.predictions / 'metrics.json').read_text())
    if evidence.get('fixture') or evidence.get('data_scope') != 'full official training data':
        raise ValueError('Only verified full-data predictions can be submitted by this entry point')
    expected = pd.read_csv(ROOT / 'data/packed/test.csv', dtype={'id': str})
    submission_path = args.predictions / 'submission.csv'
    table = pd.read_csv(submission_path, dtype={'id': str})
    if list(table.columns) != ['id', 'target'] or not table.id.equals(expected.id) or table.id.duplicated().any():
        raise ValueError('Submission columns or IDs differ from the official template')
    if not np.isfinite(table.target).all() or not table.target.between(0, 1).all():
        raise ValueError('Invalid probability values')
    digest = hashlib.sha256('\n'.join(expected.id).encode()).hexdigest()
    if evidence.get('test_id_sha256') != digest or not np.isfinite(evidence['competition_validation_auc']):
        raise ValueError('Prediction evidence does not match submission IDs or held-out evaluation')
    limits = api.competition_get_submission_limits(COMPETITION)
    if limits.num_allowed_now <= 0:
        raise RuntimeError('Kaggle has no remaining submission allowance')
    before = {str(s.ref) for s in (api.competition_submissions(COMPETITION) or [])}
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    response = api.competition_submit(str(submission_path), args.message, COMPETITION, quiet=True)
    record = {'submitted_at': started, 'file': str(submission_path), 'message': args.message,
              'file_sha256': hashlib.sha256(submission_path.read_bytes()).hexdigest(),
              'validation_auc': evidence['competition_validation_auc'], 'samples': len(table),
              'response_message': response.message, 'submission': None}
    directory = ROOT / 'reports/submissions'
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    report_path.write_text(json.dumps(record, indent=2))
    deadline = time.monotonic()+args.wait_seconds
    while time.monotonic() < deadline:
        entries = api.competition_submissions(COMPETITION) or []
        matches = [s for s in entries if str(s.ref) not in before and s.description == args.message]
        if matches:
            selected = matches[0]
            record['submission'] = summarize(selected)
            report_path.write_text(json.dumps(record, indent=2))
            scores = record['submission']
            if (scores['public_score'] is not None or scores['private_score'] is not None
                    or getattr(selected.status, 'name', '') in ['ERROR', 'FAILED', 'CANCELED', 'CANCELLED']):
                break
        time.sleep(15)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
