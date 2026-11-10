#!/usr/bin/env python3
"""Transactional V2.3.1 -> V2.3.2 upgrade for an already-applied SmallTile worktree.

Validates every current V2.3.1 patched file and every original V2+V2.2 backup before
writing. Updates only the files needed by D-aware dispatch and the D96 M64 N128 variant.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys

BUNDLE = Path(__file__).resolve().parent
OVERLAY = BUNDLE / "source_overlay_v23"
MODIFIED = [
    "setup.py",
    "csrc/flash_attn/flash_api.cpp",
    "csrc/flash_attn/src/flash_fwd_launch_template.h",
]
CREATED_MODIFIED = ["csrc/flash_attn/src/flash_fwd_tiles.h"]
NEW_FILES = [
    "csrc/flash_attn/src/flash_fwd_hdim96_fp16_m64_n128_sm75.cu",
    "csrc/flash_attn/src/flash_fwd_hdim96_fp16_m64_n128_causal_sm75.cu",
]

def sha(b): return hashlib.sha256(b).hexdigest()
def norm(b): return b.replace(b"\r\n", b"\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    root = ap.parse_args().root.resolve()
    mp = root / ".V23_SOURCE_MANIFEST.json"
    backup = root / ".v23_backup_source"
    if not mp.is_file() or not backup.is_dir():
        print("ERROR: existing applied V2.3 worktree with backup + manifest is required.", file=sys.stderr); return 2
    man = json.loads(mp.read_text(encoding="utf-8"))
    if man.get("status") == "restored" or man.get("version", "").startswith("2.3.2"):
        print("ERROR: this tree is restored or already upgraded; refusing to apply twice.", file=sys.stderr); return 3
    entries = {e["path"]: e for e in man.get("files", [])}
    created = {e["path"]: e for e in man.get("created", [])}
    needed_files = set(MODIFIED)
    needed_created = set(CREATED_MODIFIED)
    errors = []
    # Verify the complete V2.3.1 tree and baseline backups, not only the files to be changed.
    for e in man.get("files", []):
        p = root / e["path"]
        b = backup / e["path"]
        if not p.is_file() or sha(p.read_bytes()) != e.get("patched_sha256"):
            errors.append("modified/missing V2.3 source: " + e["path"])
        if not b.is_file() or sha(b.read_bytes()) != e.get("original_sha256"):
            errors.append("missing/invalid V2+V2.2 backup: " + e["path"])
    for e in man.get("created", []):
        p = root / e["path"]
        if not p.is_file() or sha(p.read_bytes()) != e.get("sha256"):
            errors.append("modified/missing added V2.3 file: " + e["path"])
    for rel in needed_files:
        if rel not in entries: errors.append("manifest does not track source file: " + rel)
    for rel in needed_created:
        if rel not in created: errors.append("manifest does not track created source: " + rel)
    for rel in MODIFIED + CREATED_MODIFIED + NEW_FILES:
        if not (OVERLAY / rel).is_file(): errors.append("bundle overlay missing: " + rel)
    for rel in NEW_FILES:
        if (root / rel).exists(): errors.append("new source already exists: " + rel)
    if errors:
        print("ERROR: V2.3.1 tree did not pass the integrity gate; no changes made:", file=sys.stderr)
        for e in errors: print(" - " + e, file=sys.stderr)
        return 4

    staged = {}
    for rel in MODIFIED + CREATED_MODIFIED:
        current = (root / rel).read_bytes()
        style = b"\r\n" if b"\r\n" in current else b"\n"
        staged[rel] = norm((OVERLAY / rel).read_bytes()).replace(b"\n", style)
    new_style = b"\r\n" if b"\r\n" in (root / "setup.py").read_bytes() else b"\n"
    for rel in NEW_FILES:
        staged[rel] = norm((OVERLAY / rel).read_bytes()).replace(b"\n", new_style)

    old_bytes = {rel: (root / rel).read_bytes() for rel in staged if (root / rel).exists()}
    old_manifest = mp.read_bytes()
    try:
        # Stage all source files first; replace atomically one by one after complete validation.
        for rel, data in staged.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".v232tmp")
            tmp.write_bytes(data)
            os.replace(tmp, p)
        for rel in MODIFIED:
            entries[rel]["patched_sha256"] = sha(staged[rel])
            entries[rel]["v23_2_sha256"] = sha(staged[rel])
        for rel in CREATED_MODIFIED:
            created[rel]["sha256"] = sha(staged[rel])
            created[rel]["v23_2_sha256"] = sha(staged[rel])
        for rel in NEW_FILES:
            man.setdefault("created", []).append({"path": rel, "sha256": sha(staged[rel])})
        man["version"] = "2.3.2-daware-d96-m64-n128"
        man.setdefault("upgrades", []).append({
            "from": "2.3.1-smalltile-dispatch-hotfix",
            "to": "2.3.2-daware-d96-m64-n128",
            "modified": MODIFIED + CREATED_MODIFIED,
            "created": NEW_FILES,
            "note": "V2.3.1 full-worktree hash gate passed before transaction"
        })
        mt = mp.with_name(mp.name + ".v232tmp")
        mt.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        os.replace(mt, mp)
    except Exception:
        for rel, data in old_bytes.items():
            try: (root / rel).write_bytes(data)
            except Exception: pass
        for rel in NEW_FILES:
            try: (root / rel).unlink(missing_ok=True)
            except Exception: pass
        try: mp.write_bytes(old_manifest)
        except Exception: pass
        raise
    print("V2.3.2 upgrade applied to existing V2.3.1 worktree.")
    print("Updated: setup.py, flash_api.cpp, flash_fwd_launch_template.h, flash_fwd_tiles.h")
    print("Added: D96 M64 N128 causal/non-causal instantiations")
    print("Original V2+V2.2 backup preserved; rollback manifest updated.")
    return 0

if __name__ == "__main__": raise SystemExit(main())
