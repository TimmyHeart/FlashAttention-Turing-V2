#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil, zipfile, hashlib, compileall
from datetime import datetime
from pathlib import Path

MARK = "COMFYUI_TURING_FA_CUDAGRAPH_V2"

ATTN_REPLACEMENT = r'''_TURING_FA_GRAPH_SKIP_LOGGED = set()

# COMFYUI_TURING_FA_CUDAGRAPH_V2: graph-captured native FA path for Wan.
@wrap_attn
def attention_turing_flash(q, k, v, heads, mask=None, attn_precision=None,
                           skip_reshape=False, skip_output_reshape=False, **kwargs):
    """Dispatch SM75 FlashAttention, with an inference CUDA Graph path for Wan.

    --use-turing-flash-attention captures/replays the native attention op for
    Wan calls carrying _turing_graph_owner. Other calls safely use SDPA.
    --force-turing-flash-attention remains the explicit eager experiment path.
    """
    global _TURING_FA_LOGGED, _TURING_FA_ERROR_LOGGED

    if (mask is not None or kwargs.get("low_precision_attention", True) is False
            or attn_precision == torch.float32 or kwargs.get("is_causal", False)):
        return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                 skip_reshape=skip_reshape,
                                 skip_output_reshape=skip_output_reshape, **kwargs)

    eligible, shape_info = _turing_fa_shape_eligible(q, k, v, heads, skip_reshape)
    if not eligible:
        return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                 skip_reshape=skip_reshape,
                                 skip_output_reshape=skip_output_reshape, **kwargs)

    batch, q_heads, k_heads, q_len, k_len, head_dim = shape_info
    capture_active = _turing_fa_capture_active(q.device)
    graph_only = bool(args.use_turing_flash_attention) and not bool(args.force_turing_flash_attention)
    transformer_options = kwargs.get("transformer_options") or {}
    graph_owner = transformer_options.get("_turing_graph_owner")

    # Frozen V2 graph measurements favor D32/D64 from S=128 and D96/D128
    # from S=512. Use the shorter side for cross-attention. GQA is held back
    # until this exact extension path has a dedicated GQA graph validation.
    if graph_only and q_heads != k_heads:
        return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                 skip_reshape=skip_reshape,
                                 skip_output_reshape=skip_output_reshape, **kwargs)
    if graph_only and (head_dim in (96, 128) and min(q_len, k_len) < 512):
        return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                 skip_reshape=skip_reshape,
                                 skip_output_reshape=skip_output_reshape, **kwargs)

    if skip_reshape:
        q_flash, k_flash, v_flash = (t.transpose(1, 2) for t in (q, k, v))
    else:
        q_flash, k_flash, v_flash = _reshape_qkv_to_heads(
            q, k, v, batch, heads, head_dim,
            kwargs.get("enable_gqa", False), expand_kv=False
        )

    scale = kwargs.get("scale", None)
    if scale is None:
        scale = head_dim ** -0.5
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                 skip_reshape=skip_reshape,
                                 skip_output_reshape=skip_output_reshape, **kwargs)
    if not (scale > 0.0 and scale < float("inf")):
        return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                 skip_reshape=skip_reshape,
                                 skip_output_reshape=skip_output_reshape, **kwargs)

    graph_used = False
    try:
        if graph_only and not capture_active:
            if graph_owner is None or args.disable_cuda_graphs or getattr(args, "disable_comfy_compiler", False):
                reason = ("Wan graph owner missing" if graph_owner is None else
                          "ComfyUI CUDA Graphs disabled by CLI")
                if reason not in _TURING_FA_GRAPH_SKIP_LOGGED:
                    logging.info("FlashAttention Turing CUDA Graph unavailable; SDPA fallback: %s", reason)
                    _TURING_FA_GRAPH_SKIP_LOGGED.add(reason)
                return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                         skip_reshape=skip_reshape,
                                         skip_output_reshape=skip_output_reshape, **kwargs)
            graph_out = _turing_flash_attn_graph_func(
                q_flash, k_flash, v_flash, scale, False, graph_owner
            )
            if graph_out is None:
                return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                         skip_reshape=skip_reshape,
                                         skip_output_reshape=skip_output_reshape, **kwargs)
            out = graph_out.transpose(1, 2)
            graph_used = True
        else:
            # Eager force mode, or an enclosing CUDA Graph capture.
            if graph_only and graph_owner is None:
                return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                         skip_reshape=skip_reshape,
                                         skip_output_reshape=skip_output_reshape, **kwargs)
            out = flash_attn_turing_wrapper(
                q_flash, k_flash, v_flash, dropout_p=0.0, causal=False, softmax_scale=scale
            ).transpose(1, 2)
            graph_used = bool(graph_only and capture_active)

        if not skip_output_reshape:
            out = out.transpose(1, 2).reshape(batch, -1, q_heads * head_dim)
        if not _TURING_FA_LOGGED:
            logging.info(
                "FlashAttention Turing active: SM75 FP16 D%d, Q=%d K=%d, cuda_graph=%s, GQA=%s",
                head_dim, q_len, k_len, graph_used, q_heads != k_heads,
            )
            _TURING_FA_LOGGED = True
        return out
    except Exception as e:
        if not _TURING_FA_ERROR_LOGGED:
            logging.warning("FlashAttention Turing failed; falling back to PyTorch SDPA: %s", e)
            _TURING_FA_ERROR_LOGGED = True
        return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                 skip_reshape=skip_reshape,
                                 skip_output_reshape=skip_output_reshape, **kwargs)
'''

WAN_INSERT = '''\n        # COMFYUI_TURING_FA_CUDAGRAPH_V2: identify Wan calls so the attention\n        # dispatcher can reuse shape-specific CUDA Graphs across transformer blocks.\n        if getattr(comfy.model_management.args, "use_turing_flash_attention", False) and not getattr(comfy.model_management.args, "force_turing_flash_attention", False):\n            transformer_options["_turing_graph_owner"] = self\n'''


def once(text, old, new, label):
    n=text.count(old)
    if n != 1: raise RuntimeError(f"Expected exactly one {label} anchor; found {n}")
    return text.replace(old,new,1)


def apply(root: Path, package: Path):
    attn=root/"comfy/ldm/modules/attention.py"
    wan=root/"comfy/ldm/wan/model.py"
    interface=root/"comfy/ldm/modules/flash_attn_turing_interface.py"
    src=package/"flash_attn_turing_interface_v2.py"
    for p in (attn,wan,interface,src):
        if not p.is_file(): raise FileNotFoundError(p)
    a=attn.read_text(encoding="utf-8")
    w=wan.read_text(encoding="utf-8")
    i=interface.read_text(encoding="utf-8")
    if MARK in a and MARK in w and MARK in i:
        print("V2 CUDA Graph patch already applied; no changes made.")
        return
    if "COMFYUI_TURING_FA_PATCH_V1" not in a or "import flash_attn_turing as flash_attn_gpu" not in i:
        raise RuntimeError("FlashAttention Turing V1 is not installed. Apply V1 first.")
    if MARK in a or MARK in w or MARK in i:
        raise RuntimeError("Partial V2 marker found. Restore the latest V2 backups before retrying.")

    # Replace exactly the V1 dispatcher function, not surrounding ComfyUI code.
    start = a.find("@wrap_attn\ndef attention_turing_flash(")
    if start < 0: raise RuntimeError("V1 attention_turing_flash function was not found")
    # In the applied V1 tree the dispatcher ends immediately before the
    # optimized_attention assignment that selects the backend.
    end = a.find("\n\noptimized_attention = attention_basic", start)
    if end < 0: raise RuntimeError("Could not locate end of V1 dispatcher function")
    a_new = a[:start] + ATTN_REPLACEMENT + a[end:]
    # Add graph runner import next to the existing V1 adapter import.
    import_anchor = "from .flash_attn_turing_interface import flash_attn_func as _turing_flash_attn_func\n"
    a_new = once(a_new, import_anchor, import_anchor + "    from .flash_attn_turing_interface import flash_attn_graph_func as _turing_flash_attn_graph_func\n", "Turing graph adapter import")
    old_startup = 'Using FlashAttention Turing (CUDA Graph-gated; PyTorch SDPA fallback outside capture)'
    new_startup = 'Using FlashAttention Turing (Wan attention CUDA Graph capture/replay; SDPA fallback for unsupported calls)'
    if a_new.count(old_startup) != 1:
        raise RuntimeError("Expected exactly one V1 Turing startup message")
    a_new = a_new.replace(old_startup, new_startup, 1)
    i_new = src.read_text(encoding="utf-8")
    # Keep a V1 provenance marker in the replaced adapter and mark the graph extension.
    i_new = i_new.replace('"""Autograd adapter and bounded CUDA-Graph runner', '"""COMFYUI_TURING_FA_PATCH_V1 + COMFYUI_TURING_FA_CUDAGRAPH_V2: Autograd adapter and bounded CUDA-Graph runner', 1)
    # Set explicit marker as a module-level constant for idempotence.
    i_new = i_new.replace('import flash_attn_turing as flash_attn_gpu\n', 'import flash_attn_turing as flash_attn_gpu\n\n' + MARK + ' = True\n', 1)
    w_anchor = '''        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        patches_replace = transformer_options.get("patches_replace", {})
'''
    # Only patch the base WanModel.forward_orig (the Wan2.1 path used by
    # the current workflow), not the similarly shaped subclasses below it.
    class_start = w.find("class WanModel(")
    class_end = w.find("class VaceWanModel(", class_start)
    if class_start < 0 or class_end < 0:
        raise RuntimeError("Could not isolate WanModel class in model.py")
    wan_segment = w[class_start:class_end]
    wan_segment_new = once(wan_segment, w_anchor, w_anchor.replace(
        '        patches_replace = transformer_options.get("patches_replace", {})',
        WAN_INSERT.strip("\n") + '\n        patches_replace = transformer_options.get("patches_replace", {})',
    ), "WanModel.forward_orig graph-owner injection")
    w_new = w[:class_start] + wan_segment_new + w[class_end:]
    # Compile every prospective Python file before touching the installation.
    for code, name in ((a_new,str(attn)),(w_new,str(wan)),(i_new,str(interface))):
        compile(code,name,"exec")
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    backups=[]
    for p in (attn,wan,interface):
        b=p.with_name(p.name+f".bak.turing_graph_v2_{stamp}")
        shutil.copy2(p,b); backups.append(b)
    try:
        attn.write_text(a_new,encoding="utf-8",newline="\n")
        wan.write_text(w_new,encoding="utf-8",newline="\n")
        interface.write_text(i_new,encoding="utf-8",newline="\n")
    except Exception:
        for p,b in zip((attn,wan,interface),backups): shutil.copy2(b,p)
        raise
    print("FlashAttention Turing CUDA Graph V2 applied successfully.")
    print(f"ComfyUI root: {root}")
    for b in backups: print(f"Backup: {b}")
    print("Scope: Wan self/cross attention CUDA Graphs; fallback to SDPA on unsupported calls.")


def find_root(arg: Path):
    for c in (arg, *arg.parents):
        if (c/"comfy/cli_args.py").is_file() and (c/"comfy/ldm/wan/model.py").is_file():
            return c
    raise FileNotFoundError("Could not find ComfyUI root containing comfy/cli_args.py and comfy/ldm/wan/model.py")


def main():
    p=argparse.ArgumentParser(description="Apply Wan FlashAttention Turing CUDA Graph V2")
    p.add_argument("comfy_root")
    args=p.parse_args()
    root=find_root(Path(args.comfy_root).resolve())
    apply(root,Path(__file__).resolve().parent)

if __name__ == "__main__": main()
