#!/usr/bin/env python3
"""Apply the V2.3 overlay to a project tree that is EXACTLY V2 + V2.2 (hash-checked, transactional).

Modified files must match the V2.2-applied baseline (line endings normalised), the new files
must not exist yet. Nothing is written until every check has passed. Backups go to
.v23_backup_source; the manifest is .V23_SOURCE_MANIFEST.json.
"""
from pathlib import Path
import argparse, hashlib, json, os, shutil, sys

BUNDLE = Path(__file__).resolve().parent
OVERLAY = BUNDLE / 'source_overlay_v23'
EXPECTED = {   # SHA256 of the LF-normalised V2 + V2.2 baseline
    "setup.py": "6d21029ae9138c57fd4f765932a32204ff9f8ca6b57b518a333b4f80cf29fc56",
    "csrc/flash_attn/flash_api.cpp": "c273867fa865c374f11637a9ba738024312926341ba27387e01fa451fb8a99ab",
    "csrc/flash_attn/src/flash_fwd_kernel.h": "34b2e238303e1e2760b1265d0fc960ea7306256ef7ad38f81c2b8299ea1f24fe",
    "csrc/flash_attn/src/flash_fwd_launch_template.h": "4ce51dce9a11f6ac6b47877c4c38b533f669f9d2570bfb1c5e8ff5ad01d44c8a",
    "csrc/flash_attn/src/kernel_traits.h": "1d60b5185606312a45108590a515e8aa3091d416526397bd86d85f63cce2e18a",
}
NEW_FILES = ["csrc/flash_attn/src/flash_fwd_tiles.h"] + [
    f"csrc/flash_attn/src/flash_fwd_hdim{d}_fp16_m{m}{c}_sm75.cu"
    for m in (64, 32) for d in (32, 64, 96, 128) for c in ("", "_causal")]


def sha(b): return hashlib.sha256(b).hexdigest()
def norm(b): return b.replace(b'\r\n', b'\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=Path.cwd())
    root = ap.parse_args().root.resolve()
    if not (root / 'setup.py').is_file():
        print('ERROR: project root must contain setup.py', file=sys.stderr); return 2
    if (root / 'flash_attn_v22.py').exists() is False and not (root / '.V22_SOURCE_MANIFEST.json').is_file():
        print('WARNING: .V22_SOURCE_MANIFEST.json not found; the hash check below decides whether this is V2.2.', file=sys.stderr)
    backup = root / '.v23_backup_source'; manifest_path = root / '.V23_SOURCE_MANIFEST.json'
    if backup.exists() or manifest_path.exists():
        print('ERROR: V2.3 backup/manifest already exists. Use a fresh V2.2 copy or restore first.', file=sys.stderr); return 2
    originals, patched, styles = {}, {}, {}
    for rel, want in EXPECTED.items():
        p = root / rel; src = OVERLAY / rel
        if not p.is_file() or not src.is_file():
            print(f'ERROR: missing required file: {rel}', file=sys.stderr); return 2
        raw = p.read_bytes(); got = sha(norm(raw))
        if got != want:
            print(f'ERROR: {rel} is not the exact V2+V2.2 baseline (normalised SHA256={got}, expected={want}).', file=sys.stderr); return 3
        style = b'\r\n' if b'\r\n' in raw else b'\n'
        originals[rel] = raw; patched[rel] = norm(src.read_bytes()).replace(b'\n', style)
        styles[rel] = 'CRLF' if style == b'\r\n' else 'LF'
    new_style = b'\r\n' if styles["setup.py"] == 'CRLF' else b'\n'
    created = {}
    for rel in NEW_FILES:
        if (root / rel).exists():
            print(f'ERROR: {rel} already exists; refusing to overwrite.', file=sys.stderr); return 4
        src = OVERLAY / rel
        if not src.is_file():
            print(f'ERROR: overlay is missing {rel}', file=sys.stderr); return 2
        created[rel] = norm(src.read_bytes()).replace(b'\n', new_style)
    backup.mkdir(); temp, written_new = [], []
    entries, new_entries = [], []
    try:
        for rel, raw in originals.items():
            dst = backup / rel; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(raw)
        for rel, out in patched.items():
            p = root / rel; tmp = p.with_name(p.name + '.v23tmp'); tmp.write_bytes(out); temp.append(tmp); os.replace(tmp, p)
            entries.append({'path': rel, 'original_sha256': sha(originals[rel]), 'patched_sha256': sha(out), 'newline': styles[rel]})
        for rel, out in created.items():
            p = root / rel; p.write_bytes(out); written_new.append(p)
            new_entries.append({'path': rel, 'sha256': sha(out)})
        mt = manifest_path.with_name(manifest_path.name + '.tmp')
        mt.write_text(json.dumps({'version': '2.3.1-smalltile-dispatch-hotfix', 'baseline': 'V2 freeze + V2.2 overlay',
                                  'files': entries, 'created': new_entries}, indent=2) + '\n', encoding='utf-8')
        os.replace(mt, manifest_path)
    except Exception:
        for rel, raw in originals.items():
            try: (root / rel).write_bytes(raw)
            except Exception: pass
        for p in written_new + temp:
            try: p.unlink(missing_ok=True)
            except Exception: pass
        shutil.rmtree(backup, ignore_errors=True)
        raise
    print('V2.3 overlay applied.'); print(f'Backup: {backup}'); print(f'Manifest: {manifest_path}')
    print('Next: RUN_V23_BUILD.bat (FLASH_TURING_SMALLM=2), then RUN_V23_TESTS.bat, then RUN_V23_TUNE.bat.')
    return 0

if __name__ == '__main__': raise SystemExit(main())
