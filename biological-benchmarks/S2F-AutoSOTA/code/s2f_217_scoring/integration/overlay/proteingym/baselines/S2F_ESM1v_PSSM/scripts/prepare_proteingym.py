#!/usr/bin/env python3
"""Build a reviewable ProteinGym overlay and patch without publishing anything."""
import argparse
import difflib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'S2F_ESM1v_PSSM'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proteingym-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    upstream = a.proteingym_root.resolve()
    out = a.output.resolve()
    overlay = out / 'overlay'
    baseline = overlay / 'proteingym/baselines' / MODEL
    baseline.mkdir(parents=True, exist_ok=True)
    for name in ('scripts', 'reference', 'vendor', 'environment'):
        shutil.copytree(ROOT / name, baseline / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('README.md', 'MODEL_CARD.md', 'THIRD_PARTY_NOTICES.md', 'LICENSE', 'publication.json'):
        shutil.copy2(ROOT / name, baseline / name)
    (baseline / 'score.py').write_text('import runpy\nfrom pathlib import Path\nrunpy.run_path(str(Path(__file__).resolve().parent / "scripts/score.py"), run_name="__main__")\n')
    pub = json.loads((ROOT / 'publication.json').read_text())
    config = json.loads((upstream / 'config.json').read_text())
    entry = json.loads((ROOT / 'reference/proteingym_config.json').read_text())['model_list_zero_shot_substitutions_DMS'][MODEL]
    config['model_list_zero_shot_substitutions_DMS'][MODEL] = entry
    (overlay / 'config.json').write_text(json.dumps(config, indent=4) + '\n')
    constants = json.loads((upstream / 'proteingym/constants.json').read_text())
    constants['clean_names'][MODEL] = pub['display_name'] or 'S2F-ESM1v-PSSM (provisional)'
    constants['model_details'][MODEL] = 'S2F + five ESM-1v models + MSA/PSSM; inference weights selected using ProteinGym subset labels; see model card'
    constants['model_references'][MODEL] = pub['code_url'] or 'PENDING_PUBLIC_CODE_URL'
    (overlay / 'proteingym/constants.json').write_text(json.dumps(constants, indent=4) + '\n')
    bash = overlay / 'scripts/scoring_DMS_zero_shot/scoring_S2F_ESM1v_PSSM.sh'
    bash.parent.mkdir(parents=True, exist_ok=True)
    bash.write_text('#!/bin/sh\nset -eu\nPG_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)\nexec sh "$PG_ROOT/proteingym/baselines/S2F_ESM1v_PSSM/scripts/score_all.sh"\n')
    patch = []
    for f in sorted(overlay.rglob('*')):
        if not f.is_file():
            continue
        relative = f.relative_to(overlay).as_posix()
        old = upstream / relative
        before = old.read_text().splitlines(keepends=True) if old.is_file() else []
        after = f.read_text().splitlines(keepends=True)
        if before == after:
            continue
        patch.append('diff --git a/%s b/%s\n' % (relative, relative))
        if not old.exists():
            patch.append('new file mode 100644\n')
        for line in difflib.unified_diff(before, after,
                     fromfile='a/' + relative if old.exists() else '/dev/null', tofile='b/' + relative):
            patch.append(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n')
    (out / 'proteingym.patch').write_text(''.join(patch))
    print('Prepared overlay and patch in %s' % out)


if __name__ == '__main__':
    main()
