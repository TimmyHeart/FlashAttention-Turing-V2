#!/usr/bin/env python3
"""Safely apply the V2.2 overlay to an exact V2 ABCD source tree."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, sys
BUNDLE=Path(__file__).resolve().parent
OVERLAY=BUNDLE/'source_overlay_v22'
EXPECTED={
    "flash_attention_interface.py": "056843766adf18439a341696e27edd45f3fac1788a4af56b45f13254e56c4945",
    "csrc/flash_attn/flash_api.cpp": "5feb7ef3205f15141a42796cc448856f9f98c63e1fc5bd55908615afe8caaed1"
}
TARGETS=['flash_attention_interface.py','csrc/flash_attn/flash_api.cpp']

def sha(b): return hashlib.sha256(b).hexdigest()
def norm(b): return b.replace(b'\r\n',b'\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path.cwd())
    a=ap.parse_args(); root=a.root.resolve()
    if not (root/'setup.py').is_file():
        print('ERROR: project root must contain setup.py',file=sys.stderr); return 2
    backup=root/'.v22_backup_source'
    manifest_path=root/'.V22_SOURCE_MANIFEST.json'
    if backup.exists() or manifest_path.exists():
        print('ERROR: V2.2 backup/manifest already exists. Use a fresh V2 copy or inspect/restore first.',file=sys.stderr); return 2
    originals={}; patched={}; styles={}
    for rel in TARGETS:
        p=root/rel; src=OVERLAY/rel
        if not p.is_file() or not src.is_file():
            print(f'ERROR: missing required file: {rel}',file=sys.stderr); return 2
        raw=p.read_bytes(); n=norm(raw)
        actual=sha(n)
        if actual != EXPECTED[rel]:
            print(f'ERROR: {rel} is not the exact frozen V2 baseline. normalized SHA256={actual} expected={EXPECTED[rel]}',file=sys.stderr); return 3
        style=b'\r\n' if b'\r\n' in raw else b'\n'
        out=norm(src.read_bytes()).replace(b'\n',style)
        originals[rel]=raw; patched[rel]=out; styles[rel]='CRLF' if style==b'\r\n' else 'LF'
    # All validation and transformations complete before the first write.
    backup.mkdir()
    entries=[]
    temp_paths=[]
    try:
        for rel,raw in originals.items():
            dst=backup/rel; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(raw)
        for rel,out in patched.items():
            p=root/rel; tmp=p.with_name(p.name+'.v22tmp'); tmp.write_bytes(out); temp_paths.append(tmp); os.replace(tmp,p)
            entries.append({'path':rel,'original_sha256':sha(originals[rel]),'patched_sha256':sha(out),'newline':styles[rel]})
        manifest={'version':'2.2','baseline':'FlashAttention Turing V2 ABCD freeze 2026-11-09','files':entries}
        manifest_tmp=manifest_path.with_name(manifest_path.name+'.tmp')
        manifest_tmp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        os.replace(manifest_tmp, manifest_path)
    except Exception:
        # Best-effort transactional rollback if any write/replace fails.
        for rel,raw in originals.items():
            try: (root/rel).write_bytes(raw)
            except Exception: pass
        for tmp in temp_paths:
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
        try: manifest_path.unlink(missing_ok=True)
        except Exception: pass
        try: manifest_path.with_name(manifest_path.name+'.tmp').unlink(missing_ok=True)
        except Exception: pass
        shutil.rmtree(backup,ignore_errors=True)
        raise
    print('V2.2 overlay applied successfully.')
    print(f'Backup: {backup}')
    print(f'Manifest: {manifest_path}')
    for rel,out in patched.items(): print(f'{rel} SHA256={sha(out)}')
    print('Next: rebuild with RUN_V22_BUILD.bat, then RUN_V22_TESTS.bat.')
    return 0
if __name__=='__main__': raise SystemExit(main())
