#!/usr/bin/env python3
"""Comprehensive FlashAttention Turing validation for D32/D64/D96/D128.

Run from the repository root after building/installing the patched extension:
    py -3.12 test_flashattn_all_head_dims_v1.py

Options:
    --no-bench       Run correctness/stability tests only.
    --bench-only     Skip correctness tests and run the D-dimension benchmark.
    --replays N      CUDA Graph replays per case (default: 2).
    --bench-iters N  Timed iterations per benchmark round (default: 20).
    --bench-rounds N Number of benchmark rounds (default: 3).

The test uses FP32 reference attention for correctness and PyTorch SDPA for
performance comparison. It does not modify source files, install packages,
or change the PyTorch/CUDA environment.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

try:
    from flash_attention_interface import flash_attn_func, flash_attn_varlen_func
except Exception as exc:  # pragma: no cover - exercised on the target machine
    print("[FATAL] Cannot import FlashAttention Python interface:", repr(exc))
    raise


SUPPORTED_DIMS = (32, 64, 96, 128)
HEAD_CONFIGS = ((4, 4, "MHA"), (4, 2, "GQA"), (4, 1, "MQA"))
DTYPE = torch.float16
DEVICE = "cuda"
FORWARD_ATOL = 3.0e-3
FORWARD_RTOL = 3.0e-3
GRAD_ATOL = 8.0e-3
GRAD_RTOL = 8.0e-3
GRAPH_GRAD_ATOL = 1.0e-2
GRAPH_GRAD_RTOL = 1.0e-2


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""


RESULTS: list[Result] = []
BENCH_ROWS: list[dict[str, object]] = []


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def record(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    RESULTS.append(Result(name, status, detail))
    print(f"[{status}] {name}" + (f" | {detail}" if detail else ""))


def record_skip(name: str, detail: str) -> None:
    RESULTS.append(Result(name, "SKIP", detail))
    print(f"[SKIP] {name} | {detail}")


def make_qkv(
    batch: int,
    sq: int,
    sk: int,
    hq: int,
    hk: int,
    head_dim: int,
    requires_grad: bool,
):
    q = torch.randn(batch, sq, hq, head_dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, sk, hk, head_dim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, sk, hk, head_dim, device=DEVICE, dtype=DTYPE)
    if requires_grad:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    return q, k, v


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """FP32 reference. Fixed-length tensors use [B, S, H, D]."""
    hq, hk = q.shape[2], k.shape[2]
    if hq != hk:
        if hq % hk:
            raise ValueError(f"Hq={hq} must be divisible by Hk={hk}")
        ratio = hq // hk
        k = k.repeat_interleave(ratio, dim=2)
        v = v.repeat_interleave(ratio, dim=2)

    scale = (1.0 / math.sqrt(q.shape[-1])) if softmax_scale is None else softmax_scale
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2)
    vf = v.float().transpose(1, 2)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    if causal:
        if q.shape[1] != k.shape[1]:
            raise ValueError("Causal reference currently requires equal Q/K sequence lengths")
        mask = torch.ones((q.shape[1], k.shape[1]), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vf).transpose(1, 2).to(q.dtype)


def tensor_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = FORWARD_ATOL,
    rtol: float = FORWARD_RTOL,
) -> tuple[bool, str, float]:
    if actual.shape != expected.shape:
        return False, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}", float("inf")
    if not torch.isfinite(actual).all().item():
        return False, "actual contains NaN/Inf", float("inf")
    if not torch.isfinite(expected).all().item():
        return False, "reference contains NaN/Inf", float("inf")
    diff = (actual.float() - expected.float()).abs()
    max_err = diff.max().item() if diff.numel() else 0.0
    mean_err = diff.mean().item() if diff.numel() else 0.0
    rms_err = torch.sqrt(torch.mean(diff * diff)).item() if diff.numel() else 0.0
    ok = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    return ok, f"max={max_err:.7g}, mean={mean_err:.7g}, rms={rms_err:.7g}", max_err


def grad_metrics(actual: torch.Tensor, expected: torch.Tensor, *, graph: bool = False):
    atol = GRAPH_GRAD_ATOL if graph else GRAD_ATOL
    rtol = GRAPH_GRAD_RTOL if graph else GRAD_RTOL
    ok, info, max_err = tensor_metrics(actual, expected, atol=atol, rtol=rtol)
    return ok, info + f", atol={atol:g}, rtol={rtol:g}", max_err


def test_fixed_case(head_dim: int, causal: bool, hq: int, hk: int, label: str, seq_len: int = 129):
    name = f"FIXED D{head_dim} {label} S={seq_len} causal={causal} forward+backward"
    try:
        q, k, v = make_qkv(1, seq_len, seq_len, hq, hk, head_dim, True)
        out = flash_attn_func(q, k, v, causal=causal)

        qr = q.detach().clone().requires_grad_(True)
        kr = k.detach().clone().requires_grad_(True)
        vr = v.detach().clone().requires_grad_(True)
        ref = reference_attention(qr, kr, vr, causal)
        ok_fwd, info_fwd, _ = tensor_metrics(out, ref)

        grad_out = torch.randn_like(out)
        actual_grads = torch.autograd.grad(out, (q, k, v), grad_outputs=grad_out)
        ref_grads = torch.autograd.grad(ref, (qr, kr, vr), grad_outputs=grad_out)
        grad_ok = True
        grad_details = []
        for label_grad, actual, expected in zip(("dQ", "dK", "dV"), actual_grads, ref_grads):
            ok, info, _ = grad_metrics(actual, expected)
            grad_ok &= ok
            grad_details.append(f"{label_grad}[{info}]")
        record(name, ok_fwd and grad_ok, f"forward[{info_fwd}]; " + "; ".join(grad_details))
    except Exception:
        record(name, False, traceback.format_exc().strip().replace("\n", " | "))


def test_custom_scale(head_dim: int, causal: bool):
    scale = 0.73 / math.sqrt(head_dim)
    name = f"CUSTOM SCALE D{head_dim} causal={causal} scale=0.73/sqrt(D)"
    try:
        q, k, v = make_qkv(1, 96, 96, 4, 4, head_dim, True)
        out = flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
        qr = q.detach().clone().requires_grad_(True)
        kr = k.detach().clone().requires_grad_(True)
        vr = v.detach().clone().requires_grad_(True)
        ref = reference_attention(qr, kr, vr, causal, softmax_scale=scale)
        ok_fwd, info_fwd, _ = tensor_metrics(out, ref)
        grad_out = torch.randn_like(out)
        actual = torch.autograd.grad(out, (q, k, v), grad_outputs=grad_out)
        expected = torch.autograd.grad(ref, (qr, kr, vr), grad_outputs=grad_out)
        grad_ok = True
        details = []
        for tag, a, e in zip(("dQ", "dK", "dV"), actual, expected):
            ok, info, _ = grad_metrics(a, e)
            grad_ok &= ok
            details.append(f"{tag}[{info}]")
        record(name, ok_fwd and grad_ok, f"forward[{info_fwd}]; " + "; ".join(details))
    except Exception:
        record(name, False, traceback.format_exc().strip().replace("\n", " | "))


def varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: tuple[int, ...],
    causal: bool,
):
    parts = []
    start = 0
    for n in lengths:
        q_i = q[start : start + n].unsqueeze(0)
        k_i = k[start : start + n].unsqueeze(0)
        v_i = v[start : start + n].unsqueeze(0)
        parts.append(reference_attention(q_i, k_i, v_i, causal).squeeze(0))
        start += n
    return torch.cat(parts, dim=0)


def test_varlen_case(head_dim: int, causal: bool, hq: int, hk: int, label: str):
    lengths = (17, 33, 64)
    cu_vals = [0]
    for n in lengths:
        cu_vals.append(cu_vals[-1] + n)
    cu = torch.tensor(cu_vals, device=DEVICE, dtype=torch.int32)
    total = cu_vals[-1]
    name = f"VARLEN D{head_dim} {label} lengths={lengths} causal={causal} forward+backward"
    try:
        q = torch.randn(total, hq, head_dim, device=DEVICE, dtype=DTYPE, requires_grad=True)
        k = torch.randn(total, hk, head_dim, device=DEVICE, dtype=DTYPE, requires_grad=True)
        v = torch.randn(total, hk, head_dim, device=DEVICE, dtype=DTYPE, requires_grad=True)
        out = flash_attn_varlen_func(q, k, v, cu, cu, max(lengths), max(lengths), causal=causal)

        qr = q.detach().clone().requires_grad_(True)
        kr = k.detach().clone().requires_grad_(True)
        vr = v.detach().clone().requires_grad_(True)
        ref = varlen_reference(qr, kr, vr, lengths, causal)
        ok_fwd, info_fwd, _ = tensor_metrics(out, ref)
        grad_out = torch.randn_like(out)
        actual = torch.autograd.grad(out, (q, k, v), grad_outputs=grad_out)
        expected = torch.autograd.grad(ref, (qr, kr, vr), grad_outputs=grad_out)
        grad_ok = True
        details = []
        for tag, a, e in zip(("dQ", "dK", "dV"), actual, expected):
            ok, info, _ = grad_metrics(a, e)
            grad_ok &= ok
            details.append(f"{tag}[{info}]")
        record(name, ok_fwd and grad_ok, f"forward[{info_fwd}]; " + "; ".join(details))
    except Exception:
        record(name, False, traceback.format_exc().strip().replace("\n", " | "))


def test_non_default_stream(head_dim: int, causal: bool):
    name = f"NON-DEFAULT STREAM D{head_dim} causal={causal} forward+backward"
    try:
        q, k, v = make_qkv(1, 128, 128, 4, 2, head_dim, True)
        grad_out = torch.randn_like(q)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            out = flash_attn_func(q, k, v, causal=causal)
            grads = torch.autograd.grad(out, (q, k, v), grad_outputs=grad_out)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        qr = q.detach().clone().requires_grad_(True)
        kr = k.detach().clone().requires_grad_(True)
        vr = v.detach().clone().requires_grad_(True)
        ref = reference_attention(qr, kr, vr, causal)
        ok_fwd, info_fwd, _ = tensor_metrics(out, ref)
        expected = torch.autograd.grad(ref, (qr, kr, vr), grad_outputs=grad_out)
        grad_ok = True
        details = []
        for tag, a, e in zip(("dQ", "dK", "dV"), grads, expected):
            ok, info, _ = grad_metrics(a, e)
            grad_ok &= ok
            details.append(f"{tag}[{info}]")
        record(name, ok_fwd and grad_ok, f"forward[{info_fwd}]; " + "; ".join(details))
    except Exception:
        record(name, False, traceback.format_exc().strip().replace("\n", " | "))


def warmup_on_side_stream(fn: Callable[[], object], rounds: int = 3):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(rounds):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()


def test_cudagraph(head_dim: int, causal: bool, replays: int):
    name = f"CUDA GRAPH D{head_dim} causal={causal} forward+backward ({replays} replay)"
    try:
        q, k, v = make_qkv(1, 128, 128, 4, 4, head_dim, True)
        grad_out = torch.randn_like(q)

        def warmup_step():
            warm_out = flash_attn_func(q, k, v, causal=causal)
            torch.autograd.grad(warm_out, (q, k, v), grad_outputs=grad_out)

        warmup_on_side_stream(warmup_step, rounds=3)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = flash_attn_func(q, k, v, causal=causal)
            graph_grads = torch.autograd.grad(graph_out, (q, k, v), grad_outputs=grad_out)
        torch.cuda.synchronize()

        worst_out = 0.0
        worst_grads = [0.0, 0.0, 0.0]
        for replay in range(replays):
            q_new, k_new, v_new = make_qkv(1, 128, 128, 4, 4, head_dim, False)
            grad_new = torch.randn_like(grad_out)
            with torch.no_grad():
                q.copy_(q_new)
                k.copy_(k_new)
                v.copy_(v_new)
                grad_out.copy_(grad_new)
            graph.replay()
            torch.cuda.synchronize()

            captured_out = graph_out.clone()
            captured_grads = tuple(g.clone() for g in graph_grads)
            qr = q.detach().clone().requires_grad_(True)
            kr = k.detach().clone().requires_grad_(True)
            vr = v.detach().clone().requires_grad_(True)
            ref = reference_attention(qr, kr, vr, causal)
            expected_grads = torch.autograd.grad(ref, (qr, kr, vr), grad_outputs=grad_out)
            ok_out, info_out, out_err = tensor_metrics(captured_out, ref)
            if not ok_out:
                record(name, False, f"replay={replay + 1}; output[{info_out}]")
                return
            worst_out = max(worst_out, out_err)
            for idx, (actual, expected) in enumerate(zip(captured_grads, expected_grads)):
                ok, info, max_err = grad_metrics(actual, expected, graph=True)
                if not ok:
                    record(name, False, f"replay={replay + 1}; gradient[{idx}][{info}]")
                    return
                worst_grads[idx] = max(worst_grads[idx], max_err)
        record(
            name,
            True,
            f"worst_out={worst_out:.7g}; worst_dQ={worst_grads[0]:.7g}; "
            f"worst_dK={worst_grads[1]:.7g}; worst_dV={worst_grads[2]:.7g}",
        )
    except Exception:
        record(name, False, traceback.format_exc().strip().replace("\n", " | "))


def benchmark_latency_ms(fn: Callable[[], torch.Tensor], warmup: int, iterations: int, rounds: int) -> float:
    for _ in range(warmup):
        result = fn()
        del result
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            result = fn()
            del result
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples)


def peak_extra_allocated_mib(fn: Callable[[], torch.Tensor]) -> float:
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    extra = max(0, peak - baseline) / (1024.0 ** 2)
    del result
    torch.cuda.synchronize()
    return extra


def run_benchmarks(bench_iters: int, bench_rounds: int):
    print("\n=== FORWARD BENCHMARK: FLASHATTN vs PYTORCH SDPA ===")
    print("Latency is median CUDA-event time; throughput counts query tokens (B*S).")
    print("Peak extra allocated is a one-call max_memory_allocated delta; it is not reserved VRAM.")
    for head_dim in SUPPORTED_DIMS:
        for seq_len in (128, 256, 512):
            q, k, v = make_qkv(1, seq_len, seq_len, 4, 4, head_dim, False)
            for causal in (False, True):
                flash_fn = lambda: flash_attn_func(q, k, v, causal=causal)
                sdpa_fn = lambda: F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=causal,
                ).transpose(1, 2)
                name = f"D{head_dim} S={seq_len} causal={causal}"
                try:
                    flash_ms = benchmark_latency_ms(flash_fn, 5, bench_iters, bench_rounds)
                    sdpa_ms = benchmark_latency_ms(sdpa_fn, 5, bench_iters, bench_rounds)
                    flash_mem = peak_extra_allocated_mib(flash_fn)
                    sdpa_mem = peak_extra_allocated_mib(sdpa_fn)
                    query_tokens = q.shape[0] * q.shape[1]
                    flash_tps = query_tokens * 1000.0 / flash_ms
                    sdpa_tps = query_tokens * 1000.0 / sdpa_ms
                    ratio = sdpa_ms / flash_ms if flash_ms > 0 else float("nan")
                    row = {
                        "head_dim": head_dim,
                        "seq_len": seq_len,
                        "causal": causal,
                        "flash_ms": flash_ms,
                        "sdpa_ms": sdpa_ms,
                        "flash_query_tokens_s": flash_tps,
                        "sdpa_query_tokens_s": sdpa_tps,
                        "sdpa_over_flash_speed_ratio": ratio,
                        "flash_peak_extra_allocated_mib": flash_mem,
                        "sdpa_peak_extra_allocated_mib": sdpa_mem,
                    }
                    BENCH_ROWS.append(row)
                    print(
                        f"{name}: FA={flash_ms:.5f} ms ({flash_tps:,.0f} query tok/s, +{flash_mem:.2f} MiB) | "
                        f"SDPA={sdpa_ms:.5f} ms ({sdpa_tps:,.0f} query tok/s, +{sdpa_mem:.2f} MiB) | "
                        f"SDPA/FA={ratio:.3f}x"
                    )
                except Exception as exc:
                    print(f"[BENCH FAIL] {name}: {type(exc).__name__}: {exc}")


def write_benchmark_csv(path: Path):
    if not BENCH_ROWS:
        return
    fieldnames = list(BENCH_ROWS[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(BENCH_ROWS)
    print(f"Benchmark CSV: {path}")


def print_environment():
    print("=" * 88)
    print("FLASHATTENTION TURING — ALL HEAD DIMENSIONS VALIDATION V1")
    print("=" * 88)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"VRAM total: {props.total_memory / (1024 ** 3):.2f} GiB")
    free_bytes, _ = torch.cuda.mem_get_info()
    print(f"VRAM free before tests: {free_bytes / (1024 ** 3):.2f} GiB")
    print(f"FlashAttention callable: {flash_attn_func}")
    print(f"Head dimensions: {SUPPORTED_DIMS}")
    print(f"Correctness thresholds: forward atol/rtol={FORWARD_ATOL:g}; gradients atol/rtol={GRAD_ATOL:g}")
    if (props.major, props.minor) != (7, 5):
        print("[WARN] This test pack is configured for Turing SM75; detected another architecture.")
    if not torch.__version__.startswith("2.11.0") or torch.version.cuda != "13.0":
        print("[WARN] Expected PyTorch 2.11.0 and CUDA runtime 13.0 for the validated project environment.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Comprehensive D32/D64/D96/D128 test and benchmark")
    parser.add_argument("--no-bench", action="store_true", help="skip the optional SDPA benchmark")
    parser.add_argument("--bench-only", action="store_true", help="skip correctness/stability tests")
    parser.add_argument("--replays", type=int, default=2, help="CUDA Graph replays per head_dim/causal case")
    parser.add_argument("--bench-iters", type=int, default=20, help="timed iterations per benchmark round")
    parser.add_argument("--bench-rounds", type=int, default=3, help="benchmark rounds; median is reported")
    args = parser.parse_args()
    if args.replays < 1 or args.bench_iters < 1 or args.bench_rounds < 1:
        parser.error("--replays, --bench-iters and --bench-rounds must be >= 1")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path.cwd() / f"flashattn_all_head_dims_v1_{stamp}.log"
    original_stdout = sys.stdout
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            sys.stdout = Tee(original_stdout, log_file)
            print_environment()
            torch.manual_seed(20261109)
            torch.cuda.manual_seed_all(20261109)

            if not args.bench_only:
                print("\n=== FIXED-LENGTH FORWARD + BACKWARD: ALL DIMS / MHA-GQA-MQA ===")
                for d in SUPPORTED_DIMS:
                    for causal in (False, True):
                        for hq, hk, label in HEAD_CONFIGS:
                            test_fixed_case(d, causal, hq, hk, label, seq_len=129)

                print("\n=== VARLEN FORWARD + BACKWARD: ALL DIMS / MHA-GQA-MQA ===")
                for d in SUPPORTED_DIMS:
                    for causal in (False, True):
                        for hq, hk, label in HEAD_CONFIGS:
                            test_varlen_case(d, causal, hq, hk, label)

                print("\n=== CUSTOM SOFTMAX SCALE: ALL DIMS ===")
                for d in SUPPORTED_DIMS:
                    for causal in (False, True):
                        test_custom_scale(d, causal)

                print("\n=== NON-DEFAULT CUDA STREAM: ALL DIMS / GQA ===")
                for d in SUPPORTED_DIMS:
                    for causal in (False, True):
                        test_non_default_stream(d, causal)

                print("\n=== CUDA GRAPH FORWARD + BACKWARD: ALL DIMS ===")
                for d in SUPPORTED_DIMS:
                    for causal in (False, True):
                        test_cudagraph(d, causal, args.replays)

                torch.cuda.synchronize()
                passed = sum(r.status == "PASS" for r in RESULTS)
                failed = sum(r.status == "FAIL" for r in RESULTS)
                skipped = sum(r.status == "SKIP" for r in RESULTS)
                print("\n" + "=" * 88)
                print(f"CORRECTNESS SUMMARY: PASS={passed} | FAIL={failed} | SKIP={skipped} | TOTAL={len(RESULTS)}")
                print("=" * 88)

            if not args.no_bench:
                run_benchmarks(args.bench_iters, args.bench_rounds)
                csv_path = Path.cwd() / f"flashattn_all_head_dims_benchmark_{stamp}.csv"
                write_benchmark_csv(csv_path)

            torch.cuda.synchronize()
            free_after, _ = torch.cuda.mem_get_info()
            print("\n=== FINAL MEMORY ===")
            print(f"Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.1f} MiB")
            print(f"Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.1f} MiB")
            print(f"VRAM free after run: {free_after / (1024 ** 3):.2f} GiB")
            print(f"Full log: {log_path}")

            failures = [r for r in RESULTS if r.status == "FAIL"]
            print("\nFINAL SUMMARY")
            print(f"Correctness PASS={sum(r.status == 'PASS' for r in RESULTS)}")
            print(f"Correctness FAIL={len(failures)}")
            print(f"Correctness SKIP={sum(r.status == 'SKIP' for r in RESULTS)}")
            if failures:
                print("Failed cases:")
                for item in failures:
                    print(f" - {item.name}: {item.detail}")
            print("Benchmark rows:", len(BENCH_ROWS))
            return 1 if failures else 0
    except Exception:
        sys.stdout = original_stdout
        print("[FATAL] Test suite aborted:")
        traceback.print_exc()
        print(f"Partial log, if created: {log_path}")
        return 2
    finally:
        sys.stdout = original_stdout


if __name__ == "__main__":
    raise SystemExit(main())
