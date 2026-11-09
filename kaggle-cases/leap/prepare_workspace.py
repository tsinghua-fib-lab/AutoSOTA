"""Stage autosota's frozen best solution without changing the source assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

BUNDLE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def manifest():
    return json.loads((BUNDLE / 'assets.json').read_text())


def selected_assets(profile):
    return [a for a in manifest()['assets'] if profile in a['profiles']]


def check_assets(source, profile, verify=False):
    source = Path(source).resolve()
    failures = []
    for item in selected_assets(profile):
        p = source / item['path']
        if not p.is_file():
            failures.append('missing: ' + item['path'])
        elif p.stat().st_size != item['bytes']:
            failures.append('size mismatch: ' + item['path'])
        elif verify and item.get('sha256') and sha256(p) != item['sha256']:
            failures.append('SHA256 mismatch: ' + item['path'])
    if failures:
        raise RuntimeError('\n'.join(failures))
    return selected_assets(profile)


def stage(source, workspace, profile='cached', link_mode='auto', verify=False):
    source, workspace = Path(source).resolve(), Path(workspace).resolve()
    if workspace == source or workspace == BUNDLE or BUNDLE in workspace.parents or workspace in source.parents:
        # A dedicated generated child of the source is safe; never overwrite an asset root.
        raise ValueError('Use a separate workspace outside the source asset tree and bundle directory.')
    items = check_assets(source, profile, verify)
    marker = workspace / 'AUTOSOTA_WORKSPACE.json'
    if workspace.exists() and any(workspace.iterdir()) and not marker.is_file():
        raise ValueError('Refusing to modify a nonempty directory without an autosota workspace marker.')
    if marker.exists() and json.loads(marker.read_text())['bundle_id'] != manifest()['bundle_id']:
        raise ValueError('Workspace belongs to a different bundle.')
    workspace.mkdir(parents=True, exist_ok=True)
    source_prefix = manifest()['original_project_root']
    code = BUNDLE / 'solution_code'
    for path in sorted(code.rglob('*')):
        if path.is_file():
            dest = workspace / path.relative_to(code)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    templates = BUNDLE / 'runtime_templates'
    for path in sorted(templates.rglob('*')):
        if not path.is_file():
            continue
        dest = workspace / path.relative_to(templates)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.json':
            # Replace paths only; all learned numeric coefficients stay unchanged.
            dest.write_text(path.read_text().replace(source_prefix, str(workspace)))
        else:
            shutil.copyfile(path, dest)
    imported = []
    for item in items:
        src, dest = source / item['path'], workspace / item['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if item.get('relocate_json_paths'):
            dest.write_text(src.read_text().replace(source_prefix, str(workspace)))
            imported.append({'path': item['path'], 'mode': 'external recipe copied with paths rebased'})
            continue
        if dest.exists():
            if dest.stat().st_size != item['bytes']:
                raise ValueError('Existing workspace asset differs: ' + str(dest))
            imported.append({'path': item['path'], 'mode': 'existing'})
            continue
        mode = link_mode
        if link_mode in ('auto', 'hardlink'):
            try:
                os.link(src, dest)
                mode = 'hardlink'
            except OSError:
                if link_mode == 'hardlink':
                    raise
                shutil.copyfile(src, dest)
                mode = 'copy'
        else:
            shutil.copyfile(src, dest)
        imported.append({'path': item['path'], 'mode': mode})
    marker.write_text(json.dumps({'bundle_id': manifest()['bundle_id'],
                                 'source_assets': str(source), 'workspace': str(workspace),
                                 'profile': profile, 'imported': imported,
                                 'linked_assets_are_read_only_inputs': True}, indent=2))
    return workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-source', type=Path, required=True)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--profile', choices=['cached', 'inference', 'training'], default='cached')
    parser.add_argument('--link-mode', choices=['auto', 'hardlink', 'copy'], default='auto')
    parser.add_argument('--verify-assets', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        items = check_assets(args.assets_source, args.profile, args.verify_assets)
        print(json.dumps({'assets_ok': True, 'profile': args.profile, 'files': len(items),
                          'logical_GiB': sum(i['bytes'] for i in items) / 2**30}, indent=2))
        return
    if args.workspace is None:
        parser.error('--workspace is required unless --check-only is used')
    root = stage(args.assets_source, args.workspace, args.profile, args.link_mode, args.verify_assets)
    print('Prepared:', root)
    print('Asset hardlinks are inputs only. Use fresh output directories for new training.')


if __name__ == '__main__':
    main()
