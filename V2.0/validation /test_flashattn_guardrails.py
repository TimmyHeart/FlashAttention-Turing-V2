#!/usr/bin/env python3
"""Supplemental CUDA tests for the A+B+C+D guardrail branch.

Run after building/installing the extension in the target Windows environment:
    py -3.12 test_flashattn_guardrails.py

Covers cases absent from the original 72-case suite: Q/K length mismatch,
causal Q>K zero rows, B>1, non-contiguous views with stride(-1)==1, and
unsupported head-dimension rejection.
"""
from __future__ import annotations

import math
import traceback

import torch

from flash_attention_interface import flash_attn_func
import flash_attn_turing as flash_attn_gpu


ATOL = 8e-3
RTOL = 8e-3
RESULTS: list[tuple[str, bool, str]] = []


def reference_attention(q, k, v, causal: bool):
    """FP32 attention reference matching this fork's bottom-right causal mask."""
    hq, hk = q.shape[2], k.shape[2]
    if hq % hk:
        raise ValueError(f"Hq={hq} must be divisible by Hk={hk}")
    if hq != hk:
        ratio = hq // hk
        k = k.repeat_interleave(ratio, dim=2)
        v = v.repeat_interleave(ratio, dim=2)

    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2)
    vf = v.float().transpose(1, 2)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(q.shape[-1])

    if causal:
        sq, sk = q.shape[1], k.shape[1]
        qi = torch.arange(sq, device=q.device)[:, None]
        ki = torch.arange(sk, device=q.device)[None, :]
        allowed = (ki - qi) <= (sk - sq)
        scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
        has_any = allowed.any(dim=-1)[None, None, :, None]
        scores = torch.where(has_any, scores, torch.zeros_like(scores))
        probs = torch.softmax(scores, dim=-1)
        probs = probs.masked_fill(~allowed[None, None, :, :], 0.0)
    else:
        probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vf).transpose(1, 2).to(q.dtype)


def compare(actual, expected, label: str):
    if not torch.isfinite(actual).all():
        raise AssertionError(f"{label}: actual contains NaN/Inf")
    if not torch.isfinite(expected).all():
        raise AssertionError(f"{label}: reference contains NaN/Inf")
    diff = (actual.float() - expected.float()).abs()
    max_err = diff.max().item() if diff.numel() else 0.0
    if not torch.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        raise AssertionError(f"{label}: max_abs={max_err:.7g}")
    return max_err


def run_fwd_bwd_case(name: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    out = flash_attn_func(q, k, v, causal=causal)

    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    ref = reference_attention(qr, kr, vr, causal)
    fwd_err = compare(out, ref, name + " forward")

    grad_out = torch.randn_like(out)
    got = torch.autograd.grad(out, (q, k, v), grad_outputs=grad_out)
    want = torch.autograd.grad(ref, (qr, kr, vr), grad_outputs=grad_out)
    grad_errs = [compare(a, b, f"{name} {tag}") for tag, a, b in zip(("dQ", "dK", "dV"), got, want)]
    return f"forward_max={fwd_err:.6g}; grad_max={max(grad_errs):.6g}"


def record(name: str, fn):
    try:
        detail = fn()
        RESULTS.append((name, True, detail))
        print(f"[PASS] {name} | {detail}")
    except Exception:
        detail = traceback.format_exc().strip().replace("\n", " | ")
        RESULTS.append((name, False, detail))
        print(f"[FAIL] {name} | {detail}")


def main() -> int:
    if not torch.cuda.is_available():
        print("[FATAL] CUDA is not available; run this test on the SM75 target.")
        return 2

    device = "cuda"
    dtype = torch.float16
    torch.manual_seed(20261109)
    torch.cuda.manual_seed_all(20261109)

    # Q longer than K: verifies early-returned query blocks and zero-initialized output.
    for causal in (False, True):
        name = f"B2 D32 Q200 K64 causal={causal}"
        def case_q_gt_k(causal=causal):
            q = torch.randn(2, 200, 4, 32, device=device, dtype=dtype)
            k = torch.randn(2, 64, 4, 32, device=device, dtype=dtype)
            v = torch.randn_like(k)
            detail = run_fwd_bwd_case(name, q, k, v, causal)
            if causal:
                # Under bottom-right causal alignment, the first 128 query rows have no legal key.
                # Those rows must be explicitly zero, not uninitialized output memory.
                out = flash_attn_func(q, k, v, causal=True)
                if not torch.equal(out[:, :128], torch.zeros_like(out[:, :128])):
                    raise AssertionError("fully masked leading query rows are not exactly zero")
            return detail
        record(name, case_q_gt_k)

    # K longer than Q: checks the opposite length direction and its bottom-right mask offset.
    for causal in (False, True):
        name = f"B2 D32 Q64 K200 causal={causal}"
        def case_k_gt_q(causal=causal):
            q = torch.randn(2, 64, 4, 32, device=device, dtype=dtype)
            k = torch.randn(2, 200, 4, 32, device=device, dtype=dtype)
            v = torch.randn_like(k)
            return run_fwd_bwd_case(name, q, k, v, causal)
        record(name, case_k_gt_q)

    # These views are non-contiguous but retain stride(-1)==1, the case the old wrapper missed.
    name = "non-contiguous inputs with inner stride 1"
    def case_noncontiguous():
        qb = torch.randn(2, 4, 129, 32, device=device, dtype=dtype)
        kb = torch.randn(2, 4, 129, 32, device=device, dtype=dtype)
        vb = torch.randn(2, 4, 129, 32, device=device, dtype=dtype)
        q, k, v = (x.transpose(1, 2) for x in (qb, kb, vb))
        assert q.stride(-1) == 1 and not q.is_contiguous()
        assert k.stride(-1) == 1 and not k.is_contiguous()
        assert v.stride(-1) == 1 and not v.is_contiguous()
        return run_fwd_bwd_case(name, q, k, v, causal=False)
    record(name, case_noncontiguous)

    name = "reject unsupported head dimension 48"
    def case_unsupported_dim():
        q = torch.randn(1, 16, 2, 48, device=device, dtype=dtype)
        try:
            flash_attn_gpu.fwd(q, q, q, 1.0 / math.sqrt(48), False)
        except RuntimeError as exc:
            if "Unsupported head dimension" not in str(exc):
                raise AssertionError(f"unexpected error: {exc}") from exc
            return "rejected with expected TORCH_CHECK"
        raise AssertionError("head dimension 48 was not rejected")
    record(name, case_unsupported_dim)

    torch.cuda.synchronize()
    passed = sum(ok for _, ok, _ in RESULTS)
    failed = len(RESULTS) - passed
    print(f"\nGUARDRAIL SUMMARY: PASS={passed} FAIL={failed} TOTAL={len(RESULTS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
