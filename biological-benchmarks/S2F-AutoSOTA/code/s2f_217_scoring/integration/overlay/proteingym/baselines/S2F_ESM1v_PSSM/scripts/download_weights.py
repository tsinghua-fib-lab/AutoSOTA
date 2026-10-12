#!/usr/bin/env python3
"""Download required weights from their publishers and verify every SHA256."""
import argparse
import json
from pathlib import Path
import urllib.request

from score import sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--checkpoint-url', help='Public URL for the exact local checkpoint; can also be set in publication.json')
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    a.destination.mkdir(parents=True, exist_ok=True)
    publication = json.loads((ROOT / 'publication.json').read_text())
    for w in json.loads((ROOT / 'reference/weights.json').read_text()):
        path = a.destination / w['filename']
        if path.exists() and path.stat().st_size == w['bytes'] and sha256(path) == w['sha256']:
            print('Verified %s' % path.name)
            continue
        if a.verify_only:
            raise ValueError('Missing or invalid weight: %s' % path)
        url = w['url'] or a.checkpoint_url or publication['checkpoint_url']
        if not url:
            raise ValueError('Checkpoint URL is pending publication. Supply --checkpoint-url or place the verified model_best.pth in the destination.')
        tmp = path.with_suffix(path.suffix + '.partial')
        print('Downloading %s' % path.name, flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp, 'wb') as f:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                f.write(block)
        if tmp.stat().st_size != w['bytes'] or sha256(tmp) != w['sha256']:
            raise ValueError('Downloaded weight checksum mismatch: %s' % tmp)
        tmp.replace(path)


if __name__ == '__main__':
    main()
