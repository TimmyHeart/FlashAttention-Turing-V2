"""
A/B benchmark + quick correctness gate for the V2 speed variants (run on the Turing GPU).

Per shape and head dim it prints, for FlashAttn-Turing and torch SDPA:
  event ms  : CUDA-event time of back-to-back calls   (includes CPU launch gaps - what V1 reported)
  graph ms  : same call replayed from a CUDA graph    (GPU time only - no Python/ATen/launch overhead)
  kernels   : GPU ops per call (memsets + kernels), from torch.profiler
If 'event' >> 'graph' the call is host/launch bound; if they match, the kernel itself is the limit.

Correctness gate (--check, on by default): forward output and, with --bwd, dq/dk/dv versus an
fp32 reference, using the same tolerance style as a fp16 attention test (max abs err).

Typical use (build each variant into its own venv/site-packages or rebuild between runs):
  python bench_v2.py --tag base      > base.txt
  python bench_v2.py --tag sepv      > sepv.txt        # built with FLASH_TURING_SEPV=1
  python bench_v2.py --tag sepv_fastsm > both.txt      # built with both flags
"""
import argparse
import statistics
from collections import Counter

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from flash_attention_interface import flash_attn_func
import flash_attn_turing as flash_attn_gpu


def event_ms(fn, warmup=10, iters=50, rounds=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    res = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        res.append(s.elapsed_time(e) / iters)
    return statistics.median(res)


def graph_ms(fn, warmup=10, iters=50, rounds=5):
    for _ in range(warmup):          # warm-up also performs one-time setup (smem attribute, allocator)
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    res = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters // 10 or 1):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        res.append(s.elapsed_time(e) / (10 * (iters // 10 or 1)))
    return statistics.median(res)


def gpu_ops_per_call(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    names = Counter(ev.name for ev in prof.events()
                    if ev.device_type == torch.autograd.DeviceType.CUDA)
    return sum(names.values()) / iters


def ref_attn(q, k, v, causal, scale):
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))       # B,H,S,D
    hq, hk = qf.shape[1], kf.shape[1]
    if hq != hk:
        kf = kf.repeat_interleave(hq // hk, 1)
        vf = vf.repeat_interleave(hq // hk, 1)
    s = (qf @ kf.transpose(-1, -2)) * scale
    if causal:
        sq, sk = s.shape[-2:]
        qi = torch.arange(sq, device=s.device)[:, None]
        ki = torch.arange(sk, device=s.device)[None, :]
        allowed = (ki - qi) <= (sk - sq)  # this fork uses bottom-right causal alignment
        s = s.masked_fill(~allowed[None, None, :, :], float("-inf"))
        has_any = allowed.any(dim=-1)[None, None, :, None]
        # Avoid softmax(all -inf) -> NaN for query rows with no legal key.
        s = torch.where(has_any, s, torch.zeros_like(s))
        p = torch.softmax(s, -1).masked_fill(~allowed[None, None, :, :], 0.0)
    else:
        p = torch.softmax(s, -1)
    return (p @ vf).transpose(1, 2)


def check(d, b, sq, sk, h, hk, causal, bwd):
    torch.manual_seed(0)
    mk = lambda s, hh: torch.randn(b, s, hh, d, device="cuda", dtype=torch.float16, requires_grad=bwd)
    q, k, v = mk(sq, h), mk(sk, hk), mk(sk, hk)
    scale = d ** -0.5
    out = flash_attn_func(q, k, v, causal=causal)
    ref = ref_attn(q, k, v, causal, scale)
    fwd_err = (out.float() - ref).abs().max().item()
    msg = f"fwd max|err|={fwd_err:.2e}"
    ok = fwd_err < 2e-2
    if bwd:
        g = torch.randn_like(out)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
        rq, rk, rv = (x.detach().float().requires_grad_() for x in (q, k, v))
        rr = ref_attn(rq, rk, rv, causal, scale)
        gq, gk, gv = torch.autograd.grad(rr, (rq, rk, rv), g.float())
        e = [(a.float() - r).abs().max().item() for a, r in ((dq, gq), (dk, gk), (dv, gv))]
        msg += f" | dq/dk/dv max|err|={e[0]:.2e}/{e[1]:.2e}/{e[2]:.2e}"
        ok = ok and max(e) < 5e-2
    return ok, msg


def benchmark_backward(d, b, s, heads, causal):
    """Time the extension backward API separately from forward and report SDPA autograd."""
    q, k, v = (torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)
               for h in (heads, heads, heads))
    scale = d ** -0.5
    # Prepare saved forward tensors outside the timed region.
    out, lse = flash_attn_gpu.fwd(q, k, v, scale, causal)
    dout = torch.randn_like(out)
    fa_bwd = lambda: flash_attn_gpu.bwd(q, k, v, out, lse, dout, scale, causal)
    fa_ms = event_ms(fa_bwd)
    fa_ops = gpu_ops_per_call(fa_bwd)

    # SDPA backward is measured on a retained prebuilt autograd graph.
    qg, kg, vg = (x.detach().requires_grad_(True) for x in (q, k, v))
    sd_out = F.scaled_dot_product_attention(
        qg.transpose(1, 2), kg.transpose(1, 2), vg.transpose(1, 2),
        is_causal=causal,
    )
    sd_grad = dout.transpose(1, 2).contiguous()
    sd_bwd = lambda: torch.autograd.grad(
        sd_out, (qg, kg, vg), grad_outputs=sd_grad, retain_graph=True,
    )
    sd_ms = event_ms(sd_bwd)
    return fa_ms, sd_ms, fa_ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--dims", type=int, nargs="+", default=[32, 64, 96, 128])
    ap.add_argument("--seqs", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--bwd", action="store_true", help="also check backward")
    args = ap.parse_args()

    print(f"# tag={args.tag}  torch={torch.__version__}  gpu={torch.cuda.get_device_name()}")
    if not args.no_check:
        bad = 0
        for d in args.dims:
            for (sq, sk) in ((130, 130), (257, 257), (64, 200), (200, 64)):
                for causal in (False, True):
                    for (h, hk) in ((4, 4), (8, 2)):
                        ok, msg = check(d, 2, sq, sk, h, hk, causal, args.bwd)
                        if not ok:
                            bad += 1
                            print(f"CHECK FAIL d={d} sq={sq} sk={sk} causal={causal} h={h}/{hk}: {msg}")
        print(f"# correctness gate: {'PASS' if bad == 0 else f'{bad} FAIL'}")

    print("d,b,s,causal,fa_event_ms,fa_graph_ms,sdpa_event_ms,sdpa_graph_ms,fa_gpu_ops,fa_vs_sdpa_graph")
    for d in args.dims:
        for b in args.batches:
            for s in args.seqs:
                for causal in (False, True):
                    q, k, v = (torch.randn(b, s, args.heads, d, device="cuda", dtype=torch.float16)
                               for _ in range(3))
                    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
                    fa = lambda: flash_attn_func(q, k, v, causal=causal)
                    sd = lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
                    fe, fg = event_ms(fa), graph_ms(fa)
                    se, sg = event_ms(sd), graph_ms(sd)
                    ops = gpu_ops_per_call(fa)
                    print(f"{d},{b},{s},{int(causal)},{fe:.4f},{fg:.4f},{se:.4f},{sg:.4f},{ops:.1f},{fg / sg:.2f}")

    if args.bwd:
        print("# Backward-only timings; forward outputs/LSE are prepared before timing.")
        print("d,b,s,causal,fa_bwd_event_ms,sdpa_bwd_event_ms,fa_bwd_gpu_ops")
        for d in args.dims:
            for b in args.batches:
                for s in args.seqs:
                    for causal in (False, True):
                        fa_ms, sd_ms, ops = benchmark_backward(d, b, s, args.heads, causal)
                        print(f"{d},{b},{s},{int(causal)},{fa_ms:.4f},{sd_ms:.4f},{ops:.1f}")


if __name__ == "__main__":
    main()
