#!/usr/bin/env python3
"""Create separate code, score, report and integration archives with checksums."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    for name, folder in [('code.tar.gz', 'code'), ('scores_217.tar.gz', 'scores'),
                         ('reports.tar.gz', 'reports'), ('integration_and_drafts.tar.gz', None)]:
        with tarfile.open(artifacts / name, 'w:gz', compresslevel=6) as tar:
            folders = [folder] if folder else ['integration', 'drafts']
            files = [p for f in folders for p in (ROOT / f).rglob('*') if p.is_file()]
            if not folder:
                files += [ROOT / 'START_HERE.md', ROOT / 'package_release.py']
            for path in sorted(files):
                if '__pycache__' in path.parts or path.suffix == '.pyc':
                    continue
                if path.is_symlink():
                    raise ValueError('Public archives must not contain external symlinks: %s' % path)
                tar.add(path, arcname=path.relative_to(ROOT), recursive=False)
    files = sorted(artifacts.glob('*.tar.gz')) + [ROOT / 'weights/model_best.pth']
    entries = [{'file': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files]
    (artifacts / 'SHA256SUMS').write_text(''.join('%s  %s\n' % (e['sha256'], e['file']) for e in entries))
    (artifacts / 'manifest.json').write_text(json.dumps(entries, indent=2) + '\n')
    for e in entries:
        print('%s: %.2f MB' % (e['file'], e['bytes'] / 1000000))


if __name__ == '__main__':
    main()
