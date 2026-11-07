"""Apply the frozen AutoSOTA research overlay to a clean upstream checkout."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
target = Path(sys.argv[1]).resolve()
manifest = json.loads((root / 'results/source_manifest.json').read_text())
def git(*args):
    return subprocess.check_output(['git', '-C', str(target), *args], text=True).strip()
if git('rev-parse', 'HEAD') != manifest['upstream_commit']:
    raise SystemExit('Checkout does not match the required upstream revision.')
if git('status', '--porcelain'):
    raise SystemExit('Use a clean upstream checkout to preserve existing work.')
for name, expected in manifest['overlay_sha256'].items():
    if hashlib.sha256((root / 'overlay' / name).read_bytes()).hexdigest() != expected:
        raise SystemExit('Overlay integrity mismatch: ' + name)
for name in manifest['overlay_sha256']:
    destination = target / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / 'overlay' / name, destination)
print('AutoSOTA source overlay applied. Review git diff before installation.')
