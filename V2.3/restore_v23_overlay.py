#!/usr/bin/env python3
"""Hash-guarded rollback of the V2.3 overlay back to the V2+V2.2 source state."""
from pathlib import Path
import argparse, hashlib, json, os, sys

def sha_file(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=Path.cwd())
    root = ap.parse_args().root.resolve()
    mp = root / '.V23_SOURCE_MANIFEST.json'; backup = root / '.v23_backup_source'
    if not mp.is_file() or not backup.is_dir():
        print('ERROR: V2.3 manifest/backup not found', file=sys.stderr); return 2
    man = json.loads(mp.read_text(encoding='utf-8')); errors = []
    for e in man.get('files', []):
        p = root / e['path']; b = backup / e['path']
        if not p.is_file(): errors.append(f"missing: {e['path']}")
        elif sha_file(p) != e['patched_sha256']: errors.append(f"modified after apply: {e['path']}")
        if not b.is_file() or sha_file(b) != e['original_sha256']: errors.append(f"backup mismatch: {e['path']}")
    for e in man.get('created', []):
        p = root / e['path']
        if p.exists() and sha_file(p) != e['sha256']: errors.append(f"modified after apply: {e['path']}")
    if errors:
        print('REFUSING ROLLBACK; no files changed:', file=sys.stderr)
        for e in errors: print(' - ' + e, file=sys.stderr)
        return 3
    for e in man['files']:
        tmp = (root / e['path']).with_name(Path(e['path']).name + '.v23restoretmp')
        tmp.write_bytes((backup / e['path']).read_bytes()); os.replace(tmp, root / e['path'])
    for e in man.get('created', []):
        (root / e['path']).unlink(missing_ok=True)
    man['status'] = 'restored'; mp.write_text(json.dumps(man, indent=2) + '\n', encoding='utf-8')
    print('Restored V2+V2.2 sources; V2.3 tile files removed. Rebuild before using the old binary.')
    return 0

if __name__ == '__main__': raise SystemExit(main())
