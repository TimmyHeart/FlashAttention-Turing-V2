#!/usr/bin/env python3
"""V2.3 forward-tile correctness: every compiled tile (kBlockM 128/64/32) and the automatic
selector must match an FP32 reference, including ragged / Sq != Sk / causal / GQA / varlen / graph.

Run from the project root after building with FLASH_TURING_SMALLM=1 or 2:
    py -3.12 validation\\test_v23_tiles.py
"""
from __future__ import annotations
import sys, traceback
import torch
import flash_attn_turing as ext
import flash_attention_interface as fi

ATOL = RTOL = 4e-3


def ref_attn(q, k, v, causal, scale):
    # q (B,Sq,H,D), k/v (B,Sk,Hk,D) -> fp32 (B,Sq,H,D); fully-masked rows -> 0 (kernel convention)
    B, Sq, H, D = q.shape
    Sk, Hk = k.shape[1], k.shape[2]
    kf = k.float().repeat_interleave(H // Hk, dim=2)
    vf = v.float().repeat_interleave(H // Hk, dim=2)
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kf) * scale
    if causal:
        row = torch.arange(Sq, device=q.device).view(1, 1, Sq, 1)
        col = torch.arange(Sk, device=q.device).view(1, 1, 1, Sk)
        s = s.masked_fill(col > row + (Sk - Sq), float("-inf"))
    p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
    return torch.einsum("bhqk,bkhd->bqhd", p, vf)


def modes_available():
    lvl = ext.fwd_tile_support()
    modes = [128] + ([64] if lvl >= 1 else []) + ([32] if lvl >= 2 else [])
    return modes + [0]  # 0 = automatic selector


def check(name, out, ref, fails, stats):
    err = (out.float() - ref).abs()
    tol = ATOL + RTOL * ref.abs()
    ok = bool((err <= tol).all())
    stats["worst"] = max(stats["worst"], float(err.max()))
    if not ok:
        fails.append(f"{name}: max_err={float(err.max()):.5f}")
    return ok


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    dev = torch.device("cuda")
    lvl = ext.fwd_tile_support()
    print(f"fwd_tile_support={lvl}  modes={modes_available()}")
    if lvl == 0:
        print("WARNING: extension was built without FLASH_TURING_SMALLM; only kBlockM=128 is exercised.")
    shapes = [  # (B, Sq, Sk, H, Hk)
        (1, 1, 1, 4, 4), (1, 17, 17, 4, 4), (1, 33, 33, 4, 4), (1, 64, 64, 4, 4), (1, 65, 65, 4, 4),
        (1, 127, 127, 4, 4), (1, 128, 128, 4, 4), (1, 129, 129, 4, 4), (1, 255, 255, 4, 4),
        (1, 256, 256, 4, 4), (1, 512, 512, 4, 4), (1, 513, 513, 4, 4),
        (2, 100, 200, 6, 2), (1, 64, 200, 4, 4), (1, 200, 64, 4, 4), (3, 129, 257, 8, 8),
        (1, 96, 96, 4, 4), (1, 192, 320, 4, 4), (5, 130, 130, 2, 1),
    ]
    fails, stats, n = [], {"worst": 0.0}, 0
    bitwise_diff_cases = 0
    torch.manual_seed(1234)
    for D in (32, 64, 96, 128):
        scale = D ** -0.5
        for causal in (False, True):
            for (B, Sq, Sk, H, Hk) in shapes:
                q = torch.randn(B, Sq, H, D, device=dev, dtype=torch.float16)
                k = torch.randn(B, Sk, Hk, D, device=dev, dtype=torch.float16)
                v = torch.randn(B, Sk, Hk, D, device=dev, dtype=torch.float16)
                ref = ref_attn(q, k, v, causal, scale)
                outs = {}
                for m in modes_available():
                    ext.set_fwd_tile_mode(m)
                    with torch.no_grad():
                        o = fi.flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
                    torch.cuda.synchronize()
                    outs[m] = o
                    n += 1
                    check(f"fixed D{D} causal={causal} B{B} Sq{Sq} Sk{Sk} H{H}/{Hk} tile={m}", o, ref, fails, stats)
                base = outs[128]
                for m, o in outs.items():
                    if m not in (128, 0) and not torch.equal(o, base):
                        bitwise_diff_cases += 1
    # varlen
    for D in (32, 64, 96, 128):
        scale = D ** -0.5
        for causal in (False, True):
            lens = [17, 33, 64, 129]
            cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), device=dev, dtype=torch.int32)
            H = 4
            q = torch.randn(sum(lens), H, D, device=dev, dtype=torch.float16)
            k = torch.randn(sum(lens), H, D, device=dev, dtype=torch.float16)
            v = torch.randn(sum(lens), H, D, device=dev, dtype=torch.float16)
            ref = torch.cat([ref_attn(q[cu[i]:cu[i+1]][None], k[cu[i]:cu[i+1]][None], v[cu[i]:cu[i+1]][None], causal, scale)[0]
                             for i in range(len(lens))], dim=0)
            for m in modes_available():
                ext.set_fwd_tile_mode(m)
                with torch.no_grad():
                    o = fi.flash_attn_varlen_func(q, k, v, cu, cu, max(lens), max(lens), softmax_scale=scale, causal=causal)
                torch.cuda.synchronize(); n += 1
                check(f"varlen D{D} causal={causal} tile={m}", o, ref, fails, stats)
    # CUDA Graph capture with a forced tile, then replay
    for D in (32, 64, 96, 128):
        scale = D ** -0.5
        q = torch.randn(1, 256, 4, D, device=dev, dtype=torch.float16)
        k = torch.randn(1, 256, 4, D, device=dev, dtype=torch.float16)
        v = torch.randn(1, 256, 4, D, device=dev, dtype=torch.float16)
        ref = ref_attn(q, k, v, True, scale)
        for m in modes_available():
            ext.set_fwd_tile_mode(m)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad():
                fi.flash_attn_func(q, k, v, softmax_scale=scale, causal=True)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g):
                o = fi.flash_attn_func(q, k, v, softmax_scale=scale, causal=True)
            g.replay(); torch.cuda.synchronize(); n += 1
            check(f"graph D{D} tile={m}", o, ref, fails, stats)
    ext.set_fwd_tile_mode(0)
    print(f"cases run: {n}   worst abs err vs fp32 ref: {stats['worst']:.5f}")
    print(f"informational: {bitwise_diff_cases} small-tile outputs differ bitwise from kBlockM=128 (expected 0 or tiny)")
    if fails:
        print(f"FAILED {len(fails)} cases:")
        for f in fails[:40]:
            print("  -", f)
        return 1
    print("ALL V2.3 TILE TESTS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
