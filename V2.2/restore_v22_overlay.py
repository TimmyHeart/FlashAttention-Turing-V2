#!/usr/bin/env python3
"""Hash-guarded byte-exact rollback of the V2.2 source overlay."""
from pathlib import Path
import argparse, hashlib, json, os, sys

def sha_file(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path.cwd()); a=ap.parse_args()
    root=a.root.resolve(); manifest_path=root/'.V22_SOURCE_MANIFEST.json'; backup=root/'.v22_backup_source'
    if not manifest_path.is_file() or not backup.is_dir():
        print('ERROR: V2.2 manifest/backup not found',file=sys.stderr); return 2
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    errors=[]
    for ent in manifest.get('files',[]):
        p=root/ent['path']
        if not p.is_file(): errors.append(f"missing patched file: {ent['path']}")
        elif sha_file(p)!=ent['patched_sha256']: errors.append(f"modified after apply: {ent['path']}")
        b=backup/ent['path']
        if not b.is_file() or sha_file(b)!=ent['original_sha256']: errors.append(f"backup mismatch: {ent['path']}")
    if errors:
        print('REFUSING ROLLBACK; no files changed:',file=sys.stderr)
        for e in errors: print(' - '+e,file=sys.stderr)
        return 3
    # Stage every original byte sequence before replacing any target.
    staged=[]; patched_bytes={}; replaced=[]
    try:
        for ent in manifest['files']:
            dst=root/ent['path']; src=backup/ent['path']
            patched_bytes[ent['path']]=dst.read_bytes()
            tmp=dst.with_name(dst.name+'.v22restoretmp')
            tmp.write_bytes(src.read_bytes()); staged.append((ent['path'],tmp,dst))
        for rel,tmp,dst in staged:
            os.replace(tmp,dst); replaced.append(rel)
        manifest['status']='restored'
        manifest_tmp=manifest_path.with_name(manifest_path.name+'.tmp')
        manifest_tmp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        os.replace(manifest_tmp,manifest_path)
    except Exception:
        # If a replacement fails, put already-replaced files back to the V2.2 state.
        for rel in replaced:
            try: (root/rel).write_bytes(patched_bytes[rel])
            except Exception: pass
        for _,tmp,_ in staged:
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
        try: manifest_path.with_name(manifest_path.name+'.tmp').unlink(missing_ok=True)
        except Exception: pass
        raise
    print('V2 source files restored byte-for-byte. Backup retained at:',backup)
    print('Rebuild the V2 extension before running the original V2 binary.')
    return 0
if __name__=='__main__': raise SystemExit(main())
