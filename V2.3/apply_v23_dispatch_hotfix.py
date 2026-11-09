#!/usr/bin/env python3
"""Apply the V2.3.1 dispatch hotfix to an already-applied V2.3 SmallTile worktree.

The script is deliberately hash-guarded and transactional. It updates only
csrc/flash_attn/flash_api.cpp and the existing .V23_SOURCE_MANIFEST.json entry,
so the original V2+V2.2 backup remains usable by RUN_V23_RESTORE.bat.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys

BUNDLE = Path(__file__).resolve().parent
HOTFIX_SOURCE = BUNDLE / 'source_overlay_v23/csrc/flash_attn/flash_api.cpp'
TARGET_REL = 'csrc/flash_attn/flash_api.cpp'

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def norm(data: bytes) -> bytes:
    return data.replace(b'\r\n', b'\n')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path.cwd())
    root = ap.parse_args().root.resolve()
    manifest_path = root / '.V23_SOURCE_MANIFEST.json'
    backup_dir = root / '.v23_backup_source'
    target = root / TARGET_REL
    backup = backup_dir / TARGET_REL

    if not manifest_path.is_file() or not backup_dir.is_dir():
        print('ERROR: an applied V2.3 worktree with .V23_SOURCE_MANIFEST.json and .v23_backup_source is required.', file=sys.stderr)
        return 2
    if not target.is_file() or not backup.is_file() or not HOTFIX_SOURCE.is_file():
        print('ERROR: missing target, original backup, or hotfix source.', file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') == 'restored':
        print('ERROR: this V2.3 overlay has already been restored; apply to a fresh V2.2 copy instead.', file=sys.stderr)
        return 3
    entry = next((e for e in manifest.get('files', []) if e.get('path') == TARGET_REL), None)
    if entry is None:
        print('ERROR: flash_api.cpp is not tracked by the current V2.3 manifest.', file=sys.stderr)
        return 3

    current = target.read_bytes()
    original_backup = backup.read_bytes()
    if sha(original_backup) != entry.get('original_sha256'):
        print('ERROR: original flash_api.cpp backup does not match the manifest; refusing to patch.', file=sys.stderr)
        return 4
    if sha(current) != entry.get('patched_sha256'):
        print('ERROR: current flash_api.cpp differs from the applied V2.3 candidate; refusing to overwrite local changes.', file=sys.stderr)
        return 4

    # Validate every V2.3 file before changing anything, so unrelated edits are not hidden.
    errors = []
    for e in manifest.get('files', []):
        p = root / e['path']
        if not p.is_file() or sha(p.read_bytes()) != e.get('patched_sha256'):
            errors.append('modified/missing source: ' + e['path'])
        b = backup_dir / e['path']
        if not b.is_file() or sha(b.read_bytes()) != e.get('original_sha256'):
            errors.append('missing/invalid backup: ' + e['path'])
    for e in manifest.get('created', []):
        p = root / e['path']
        if not p.is_file() or sha(p.read_bytes()) != e.get('sha256'):
            errors.append('modified/missing added file: ' + e['path'])
    if errors:
        print('ERROR: worktree is not an untouched V2.3 apply; no files changed:', file=sys.stderr)
        for item in errors:
            print(' - ' + item, file=sys.stderr)
        return 5

    raw_hotfix = HOTFIX_SOURCE.read_bytes()
    style = b'\r\n' if b'\r\n' in current else b'\n'
    patched = norm(raw_hotfix).replace(b'\n', style)
    patched_hash = sha(patched)
    old_hash = sha(current)

    # Prepare updated manifest in memory and stage both writes before atomic replacement.
    entry['patched_sha256'] = patched_hash
    manifest['version'] = '2.3.1-smalltile-dispatch-hotfix'
    manifest.setdefault('hotfixes', []).append({
        'name': 'dispatch-preprocessor-scope-fix',
        'path': TARGET_REL,
        'previous_sha256': old_hash,
        'patched_sha256': patched_hash,
        'newline': 'CRLF' if style == b'\r\n' else 'LF'
    })
    manifest_tmp = manifest_path.with_name(manifest_path.name + '.hotfixtmp')
    target_tmp = target.with_name(target.name + '.hotfixtmp')
    try:
        target_tmp.write_bytes(patched)
        manifest_tmp.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        os.replace(target_tmp, target)
        os.replace(manifest_tmp, manifest_path)
    except Exception:
        try:
            target.write_bytes(current)
        except Exception:
            pass
        target_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
        raise

    print('V2.3.1 dispatch hotfix applied.')
    print('Changed: ' + TARGET_REL)
    print('Updated rollback manifest; original V2+V2.2 backup preserved.')
    print('Next: RUN_V23_BUILD.bat, then RUN_V23_TESTS.bat, then RUN_V23_TUNE.bat.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
