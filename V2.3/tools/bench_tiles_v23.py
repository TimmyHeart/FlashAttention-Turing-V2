#!/usr/bin/env python3
"""V2.3 forward-tile tuner. CUDA-Graph per-op latency for every compiled kBlockM, vs PyTorch SDPA.

Prints a table, writes a CSV, reports which tile is fastest per (D, S, causal) cell and the
fixed-cost constant of the automatic selector that minimises total regret. No CUPTI needed.
    py -3.12 tools\\bench_tiles_v23.py --output v23_tiles.csv
"""
from __future__ import annotations
import argparse, csv, statistics
from pathlib import Path
import torch
import torch.nn.functional as F
import flash_attn_turing as ext
import flash_attention_interface as fi


def graph_us_per_op(fn, n_ops, replays, rounds, dev):
    side = torch.cuda.Stream(device=dev); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize(dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_ops): fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(dev)
    vals = []
    for _ in range(rounds):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(replays): g.replay()
        e.record(); torch.cuda.synchronize(dev)
        vals.append(s.elapsed_time(e) * 1000.0 / (replays * n_ops))
    return statistics.median(vals)


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
    tiles = [128] + ([64] if lvl >= 1 else []) + ([32] if lvl >= 2 else [])
    print(f"# gpu={torch.cuda.get_device_name(dev)} SMs={torch.cuda.get_device_properties(dev).multi_processor_count} "
          f"tiles={tiles} B={a.batch} H={a.heads} n_ops={a.n_ops}")
    rows, cells = [], {}
    for d in a.dims:
        for s in a.seqs:
            for c in a.causal:
                causal = bool(c)
                q, k, v = (torch.randn(a.batch, s, a.heads, d, device=dev, dtype=torch.float16) for _ in range(3))
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
                    rows.append(dict(d=d, b=a.batch, h=a.heads, s=s, causal=c, tile=t,
                                     us_per_op=round(times[t], 3), sdpa_us_per_op=round(t_sdpa, 3),
                                     speedup_vs_sdpa=round(t_sdpa / times[t], 3)))
                ext.set_fwd_tile_mode(0)
                cells[(d, s, c)] = (times, t_sdpa)
                best = min(times, key=times.get)
                print(f"D{d:<3} S{s:<4} causal={c}  sdpa={t_sdpa:7.2f}us  " +
                      "  ".join(f"M{t}={times[t]:7.2f}us" for t in tiles) +
                      f"   best=M{best} ({t_sdpa/times[best]:.2f}x vs SDPA; baseline M128 {t_sdpa/times[128]:.2f}x)", flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if len(tiles) > 1:
        print("\n# automatic selector calibration (total regret in us over all cells; lower is better)")
        best_cost, best_regret = None, None
        for fixed in (0, 16, 32, 48, 64, 96, 128, 192):
            ext.set_fwd_tile_fixed_cost(fixed); ext.set_fwd_tile_mode(0)
            regret = 0.0
            for (d, s, c), (times, _) in cells.items():
                pick = ext.plan_fwd_tile(a.batch, s, a.heads)
                regret += times.get(pick, times[128]) - min(times.values())
            print(f"fixed_cost={fixed:<4} regret={regret:8.2f}")
            if best_regret is None or regret < best_regret:
                best_cost, best_regret = fixed, regret
        ext.set_fwd_tile_fixed_cost(48)
        print(f"# lowest-regret fixed_cost on this grid: {best_cost} (default 48). "
              f"Per-D regret is not modelled; if one D disagrees, force a tile for it.")
    print(f"# wrote {a.output}")


if __name__ == "__main__":
    main()
