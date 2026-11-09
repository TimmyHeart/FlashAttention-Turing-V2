#!/usr/bin/env python3
"""V2.2 GPU validation: inference dispatch, reusable output/LSE, and CUDA Graph."""
from __future__ import annotations
import torch
import flash_attn_turing as ext
import flash_attention_interface as fi
from flash_attn_v22 import FlashAttnStaticGraph


def check(name, fn, passed, failed):
    try:
        fn()
        torch.cuda.synchronize()
        passed.append(name)
        print(f"[PASS] {name}")
    except Exception as exc:
        failed.append((name, f"{type(exc).__name__}: {exc}"))
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")


def close(a, b, name, atol=3e-3, rtol=3e-3):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=name)


def make_qkv(d=32, sq=130, sk=130, b=2, hq=4, hk=4, requires_grad=False):
    q = torch.randn(b, sq, hq, d, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
    k = torch.randn(b, sk, hk, d, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
    v = torch.randn(b, sk, hk, d, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
    return q, k, v


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    print(f"GPU={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
    print(f"PyTorch={torch.__version__} CUDA={torch.version.cuda}")
    print("Import order: torch -> flash_attn_turing -> flash_attention_interface")
    print(f"Fastpath enabled={fi._FASTPATH}; fwd_into={'available' if fi._ext_fwd_into else 'missing'}")
    if fi._ext_fwd_into is None:
        raise SystemExit("fwd_into is missing: rebuild the V2.2 C++ extension first")

    passed, failed = [], []

    # Public inference fast path must preserve the V2 Function.forward result.
    for d in (32, 64, 96, 128):
        for causal in (False, True):
            def run_fast(d=d, causal=causal):
                q, k, v = make_qkv(d=d, requires_grad=False)
                fast = fi.flash_attn_func(q, k, v, causal=causal)
                legacy = fi.FlashAttnFunc.apply(q, k, v, None, causal, False)
                close(fast, legacy, f"public fastpath vs V2 Function D{d} causal={causal}", 0, 0)
                assert not fast.requires_grad
            check(f"public fastpath equals V2 Function D{d} causal={causal}", run_fast, passed, failed)

    # Grad mode with requires_grad must retain the V2 autograd.Function route.
    def grad_path():
        q, k, v = make_qkv(d=64, requires_grad=True)
        out = fi.flash_attn_func(q, k, v, causal=True)
        assert out.requires_grad and out.grad_fn is not None
        out.float().sum().backward()
        assert all(t.grad is not None for t in (q, k, v))
    check("autograd path retained and backward populates q/k/v gradients", grad_path, passed, failed)

    # no_grad with requires_grad inputs should bypass saving/recording exactly as V2 did.
    def no_grad_path():
        q, k, v = make_qkv(d=32, requires_grad=True)
        with torch.no_grad():
            fast = fi.flash_attn_func(q, k, v, causal=False)
            legacy = fi.FlashAttnFunc.apply(q, k, v, None, False, False)
        close(fast, legacy, "no_grad fastpath vs V2", 0, 0)
        assert not fast.requires_grad
    check("no_grad + requires_grad inputs uses inference path", no_grad_path, passed, failed)

    # Varlen fast path must match the old V2 autograd Function forward path.
    for d in (32, 64, 96, 128):
        for causal in (False, True):
            def run_varlen(d=d, causal=causal):
                lengths = [17, 33, 64]
                cu = torch.tensor([0, 17, 50, 114], device="cuda", dtype=torch.int32)
                q = torch.randn(114, 4, d, device="cuda", dtype=torch.float16)
                k = torch.randn(114, 4, d, device="cuda", dtype=torch.float16)
                v = torch.randn_like(k)
                fast = fi.flash_attn_varlen_func(q, k, v, cu, cu, 64, 64, causal=causal)
                legacy = fi.FlashAttnVarlenFunc.apply(q, k, v, cu, cu, 64, 64, None, causal, False)
                close(fast, legacy, f"varlen fastpath D{d} causal={causal}", 0, 0)
            check(f"varlen fastpath equals V2 D{d} causal={causal}", run_varlen, passed, failed)

    # fwd_into vs allocating V2 fwd, including GQA and Q>K zero-fill coverage.
    cases = [
        (32, 130, 130, 4, 4, False), (32, 130, 130, 4, 4, True),
        (64, 130, 130, 4, 2, False), (64, 130, 130, 4, 2, True),
        (96, 130, 130, 4, 4, False), (96, 130, 130, 4, 4, True),
        (128, 130, 130, 4, 4, False), (128, 130, 130, 4, 4, True),
        (32, 200, 64, 4, 4, True), (32, 64, 200, 4, 4, True),
    ]
    for d, sq, sk, hq, hk, causal in cases:
        def run_into(d=d, sq=sq, sk=sk, hq=hq, hk=hk, causal=causal):
            q, k, v = make_qkv(d, sq, sk, b=2, hq=hq, hk=hk)
            scale = d ** -0.5
            ref, ref_lse = ext.fwd(q, k, v, scale, causal)
            out, lse = fi.allocate_fwd_buffers(q)
            result = fi.flash_attn_func_into(q, k, v, out, lse, scale, causal)
            assert result.data_ptr() == out.data_ptr()
            close(out, ref, "fwd_into output")
            close(lse, ref_lse, "fwd_into LSE", 1e-4, 1e-4)
        check(f"fwd_into vs fwd D{d} B2 Q{sq} K{sk} GQA={hq != hk} causal={causal}", run_into, passed, failed)

    # Negative tests: alias on output and LSE must be rejected by C++.
    def reject_out_alias():
        q, k, v = make_qkv(32, 128, 128, b=1)
        l = torch.empty((1, 4, 128), device="cuda", dtype=torch.float32)
        try:
            fi.flash_attn_func_into(q, k, v, q, l)
        except RuntimeError as exc:
            assert "share storage" in str(exc)
            return
        raise AssertionError("out aliasing q was accepted")
    check("reject output/input storage alias", reject_out_alias, passed, failed)

    def reject_lse_alias():
        q, k, v = make_qkv(32, 128, 128, b=1)
        # Reinterpret q storage as float32 and take a contiguous prefix of matching shape.
        l = q.view(torch.float32).reshape(-1)[:512].view(1, 4, 128)
        o = torch.empty_like(q)
        try:
            fi.flash_attn_func_into(q, k, v, o, l)
        except RuntimeError as exc:
            assert "share storage" in str(exc)
            return
        raise AssertionError("lse aliasing q was accepted")
    check("reject LSE/input storage alias", reject_lse_alias, passed, failed)

    def reject_lse_output_alias():
        q, k, v = make_qkv(32, 128, 128, b=1)
        shared = torch.empty((q.numel() // 2,), device="cuda", dtype=torch.float32)
        out = shared.view(torch.float16).view_as(q)
        lse = shared.view(1, q.shape[2], q.shape[1])
        try:
            fi.flash_attn_func_into(q, k, v, out, lse)
        except RuntimeError as exc:
            assert "share storage" in str(exc)
            return
        raise AssertionError("out/lse sharing storage was accepted")
    check("reject output/LSE shared storage", reject_lse_output_alias, passed, failed)

    def reject_noncontiguous():
        q, k, v = make_qkv(32, 128, 128, b=1)
        q_nc = q.transpose(1, 2)
        out, lse = fi.allocate_fwd_buffers(q)
        try:
            fi.flash_attn_func_into(q_nc, k, v, out, lse)
        except ValueError as exc:
            assert "contiguous" in str(exc)
            return
        raise AssertionError("fwd_into silently accepted non-contiguous input")
    check("reject non-contiguous fwd_into input without hidden copy", reject_noncontiguous, passed, failed)

    # One-op static graph, input-copy replay, direct persistent-buffer update, and Q>K.
    def graph_tests():
        q, k, v = make_qkv(32, 128, 128, b=1)
        runner = FlashAttnStaticGraph(q, k, v, causal=False)
        got = runner.replay()
        ref = fi.flash_attn_func(q, k, v, causal=False)
        close(got, ref, "static graph initial replay")
        q2, k2, v2 = make_qkv(32, 128, 128, b=1)
        got = runner.copy_inputs_and_replay(q2, k2, v2)
        close(got, fi.flash_attn_func(q2, k2, v2), "copy-input graph replay")
        q3, k3, v3 = make_qkv(32, 128, 128, b=1)
        runner.q.copy_(q3); runner.k.copy_(k3); runner.v.copy_(v3)
        got = runner.replay()
        close(got, fi.flash_attn_func(q3, k3, v3), "direct persistent input update")
        q4, k4, v4 = make_qkv(32, 129, 64, b=1)
        runner_qk = FlashAttnStaticGraph(q4, k4, v4, causal=True)
        got_qk = runner_qk.replay()
        ref_qk = fi.flash_attn_func(q4, k4, v4, causal=True)
        close(got_qk, ref_qk, "causal Q>K graph zero-fill")
    check("CUDA Graph capture/replay, input update/copy, causal Q>K", graph_tests, passed, failed)

    print(f"V2.2 HOSTPATH SUMMARY: PASS={len(passed)} FAIL={len(failed)} TOTAL={len(passed)+len(failed)}")
    if failed:
        for name, error in failed:
            print(f"FAILED: {name}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
