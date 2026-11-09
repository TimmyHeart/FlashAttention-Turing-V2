"""
Where does the D32 time go?  (run on the Turing machine, after building)

For D32 and D64 (and optionally D96/D128) this reports, per forward call:
  * wall time (CUDA events, same method as the V1 benchmark)
  * number of CUDA kernels launched and the SUM of their GPU execution time
  * "overhead" = wall - kernel_sum   (launch gaps / CPU-side ATen dispatch)
  * the individual kernel names, so you can see the pad/zero/copy kernels

Interpretation
  If D32's kernel_sum is close to D64's but wall is ~4x larger, the V1 slowdown is
  padding-path overhead (extra allocs/fills/copies), not the attention kernel, and
  a native D32 kernel will only help via removing that path.
  If D32's kernel_sum itself is much larger than D64's, the compute is the problem.

Usage:  python profile_d32_overhead.py [--seqlen 256] [--heads 4] [--batch 1] [--dims 32 64]
"""
import argparse
import statistics
from collections import Counter

import torch
from torch.profiler import profile, ProfilerActivity

from flash_attention_interface import flash_attn_func


def wall_ms(fn, warmup=10, iters=50, rounds=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / iters)
    return statistics.median(out)


def kernel_stats(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    names, total_us = Counter(), 0.0
    for ev in prof.events():
        # device-side kernel / memset / memcpy records
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            names[ev.name] += 1
            total_us += ev.device_time if hasattr(ev, "device_time") else ev.cuda_time
    n_kernels = sum(names.values()) / iters
    return n_kernels, total_us / iters / 1000.0, names, iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--dims", type=int, nargs="+", default=[32, 64, 96, 128])
    ap.add_argument("--causal", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    print(f"B={args.batch} H={args.heads} S={args.seqlen} causal={args.causal} fp16  ({torch.cuda.get_device_name()})")
    print(f"{'D':>4} {'wall ms':>9} {'kern sum ms':>12} {'overhead ms':>12} {'#kernels':>9}")
    detail = {}
    for d in args.dims:
        q = torch.randn(args.batch, args.seqlen, args.heads, d, device=dev, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        fn = lambda: flash_attn_func(q, k, v, causal=args.causal)
        w = wall_ms(fn)
        nk, ksum, names, iters = kernel_stats(fn)
        print(f"{d:>4} {w:>9.4f} {ksum:>12.4f} {w - ksum:>12.4f} {nk:>9.1f}")
        detail[d] = (names, iters)
        # SDPA reference
        qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
        sd = lambda: torch.nn.functional.scaled_dot_product_attention(qs, ks, vs, is_causal=args.causal)
        if d == args.dims[0]:
            print(f"     (SDPA D={d}: wall {wall_ms(sd):.4f} ms)")

    print("\nKernels per call:")
    for d, (names, iters) in detail.items():
        print(f"  D={d}")
        for n, c in names.most_common():
            print(f"    {c / iters:4.1f} x {n[:100]}")


if __name__ == "__main__":
    main()
