"""Relocate historical absolute paths in runtime metadata, preserving evidence."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    old=json.loads((ROOT/'DELIVERY.json').read_text())['runtime_source_root']
    changed=[]
    for base in ['experiments','data']:
        for p in (ROOT/base).rglob('*.json'):
            original=p.read_text()
            updated=original.replace(old,str(ROOT))
            if updated!=original:
                json.loads(updated)
                p.write_text(updated)
                changed.append(str(p.relative_to(ROOT)))
    print(json.dumps({'changed_files':len(changed),'root':str(ROOT),
                      'note':'Run verify_package.py before relocation. Archived evidence in snapshot remains unchanged. Training-only paths may refer to arrays or old weights intentionally excluded from this delivery.'},indent=2))


if __name__=='__main__':main()
