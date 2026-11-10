from __future__ import annotations
import argparse
from pathlib import Path

MARK = "COMFYUI_TURING_FA_CUDAGRAPH_V2"

def main() -> int:
    p = argparse.ArgumentParser(description="Verify ComfyUI Turing FlashAttention CUDA Graph V2 source patch")
    p.add_argument("comfy_root")
    root = Path(p.parse_args().comfy_root).resolve()
    files = {
        "attention": root / "comfy/ldm/modules/attention.py",
        "wan": root / "comfy/ldm/wan/model.py",
        "adapter": root / "comfy/ldm/modules/flash_attn_turing_interface.py",
        "cli": root / "comfy/cli_args.py",
    }
    missing = [str(v) for v in files.values() if not v.is_file()]
    if missing:
        print("FAIL: missing source files:")
        print("\n".join(missing))
        return 1
    data = {k: v.read_text(encoding="utf-8") for k, v in files.items()}
    checks = {
        "V2 marker in attention dispatcher": MARK in data["attention"],
        "V2 marker in Wan graph-owner path": MARK in data["wan"],
        "V2 graph adapter present": MARK in data["adapter"] and "def flash_attn_graph_func" in data["adapter"],
        "CLI graph option present": "--use-turing-flash-attention" in data["cli"],
        "Dispatcher calls graph adapter": "_turing_flash_attn_graph_func(" in data["attention"],
        "Wan owner is injected": "transformer_options[\"_turing_graph_owner\"] = self" in data["wan"],
    }
    failed = False
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
        failed |= not ok
    if failed:
        return 1
    print("Source verification PASS. This does not replace GPU graph-capture runtime testing.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
