#!/usr/bin/env python3
"""V2.3.2 tile tuner: M128/M64/M32 plus the D96 M64 N128 experiment.

The automatic policy is shape-aware (D + causal + B/H) for the two measured
CMP 40HX profiles. This script reports per-cell regret against the best forced
candidate, including the optional D96 M64 N128 variant.
"""
from __future__ import annotations
import argparse, csv, statistics
from pathlib import Path
import torch
import torch.nn.functional as F
import flash_attn_turing as ext
import flash_attention_interface as fi


def graph_us_per_op(fn, n_ops, replays, rounds, dev):
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize(dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_ops): fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(dev)
    vals = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays): g.replay()
        end.record(); torch.cuda.synchronize(dev)
        vals.append(start.elapsed_time(end) * 1000.0 / (replays * n_ops))
    return statistics.median(vals)


def tile_label(code: int) -> str:
    return "M64N128" if code == 65 else f"M{code}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[32, 64, 96, 128])
    ap.add_argument("--seqs", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--causal", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-ops", type=int, default=16)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--output", type=Path, default=Path("v23_tiles.csv"))
    a = ap.parse_args()
    dev = torch.device("cuda", torch.cuda.current_device())
    lvl = ext.fwd_tile_support()
    has_d96_n128 = bool(ext.fwd_tile_n128_support())
    base_tiles = [128] + ([64] if lvl >= 1 else []) + ([32] if lvl >= 2 else [])
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    print(f"# gpu={torch.cuda.get_device_name(dev)} SMs={sms} tiles={base_tiles} "
          f"D96_M64_N128={int(has_d96_n128)} B={a.batch} H={a.heads} n_ops={a.n_ops}")
    rows, cells = [], {}
    for d in a.dims:
        tiles = list(base_tiles)
        if d == 96 and has_d96_n128:
            tiles.append(65)
        for s in a.seqs:
            for c in a.causal:
                causal = bool(c)
                q, k, v = (torch.randn(a.batch, s, a.heads, d, device=dev, dtype=torch.float16)
                           for _ in range(3))
                qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
                sdpa = lambda: F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal)
                t_sdpa = graph_us_per_op(sdpa, a.n_ops, a.replays, a.rounds, dev)
                scale = d ** -0.5
                times = {}
                for t in tiles:
                    ext.set_fwd_tile_mode(t)
                    fn = lambda: fi.flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
                    with torch.no_grad():
                        times[t] = graph_us_per_op(fn, a.n_ops, a.replays, a.rounds, dev)
                    tile_m, tile_n = (64, 128) if t == 65 else (t, 128 if d in (32, 64) else 64)
                    rows.append(dict(d=d, b=a.batch, h=a.heads, s=s, causal=c,
                                     tile=t, tile_label=tile_label(t), tile_m=tile_m, tile_n=tile_n,
                                     us_per_op=round(times[t], 3), sdpa_us_per_op=round(t_sdpa, 3),
                                     speedup_vs_sdpa=round(t_sdpa / times[t], 3)))
                ext.set_fwd_tile_mode(0)
                cells[(d, s, c)] = (times, t_sdpa)
                best = min(times, key=times.get)
                pick = ext.plan_fwd_tile_for_shape(a.batch, s, a.heads, d, causal)
                picked_us = times.get(pick)
                if picked_us is None:
                    # 65 is not an automatic option yet; if it ever appears, the benchmark
                    # candidate list above will contain it for D96.
                    picked_us = times[128]
                regret = picked_us - times[best]
                print(f"D{d:<3} S{s:<4} causal={c}  sdpa={t_sdpa:7.2f}us  " +
                      "  ".join(f"{tile_label(t)}={times[t]:7.2f}us" for t in tiles) +
                      f"   best={tile_label(best)} ({t_sdpa/times[best]:.2f}x) "
                      f"auto={tile_label(pick)} regret={regret:.2f}us", flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    if len(base_tiles) > 1:
        print("\n# shape-aware automatic selector regret (lower is better)")
        total_regret = 0.0
        by_d = {d: 0.0 for d in a.dims}
        for (d, s, c), (times, _) in cells.items():
            pick = ext.plan_fwd_tile_for_shape(a.batch, s, a.heads, d, bool(c))
            chosen = times.get(pick, times[128])
            regret = chosen - min(times.values())
            total_regret += regret
            by_d[d] += regret
        print(f"total_regret={total_regret:.2f}us")
        for d, value in by_d.items():
            print(f"D{d:<3} regret={value:.2f}us")
        print("# The D96 M64 N128 candidate is measured independently; it is not selected automatically yet.")
    print(f"# wrote {a.output}")


if __name__ == "__main__":
    main()
