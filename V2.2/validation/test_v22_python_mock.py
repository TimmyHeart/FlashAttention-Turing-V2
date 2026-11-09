#!/usr/bin/env python3
"""CPU-only mock tests for V2.2 Python dispatch; not a CUDA correctness test."""
from __future__ import annotations
import importlib
import sys
import types
import torch

calls = {"fwd": 0, "varlen_fwd": 0, "fwd_into": 0}


def fwd(q, k, v, scale, causal):
    calls["fwd"] += 1
    out = q + k + v
    lse = torch.zeros((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32, device=q.device)
    return out, lse


def varlen_fwd(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal):
    calls["varlen_fwd"] += 1
    out = q + k + v
    lse = torch.zeros((cu_q.numel() - 1, q.shape[1], max_q), dtype=torch.float32, device=q.device)
    return out, lse


def bwd(q, k, v, out, lse, dout, scale, causal):
    return dout, dout, dout


def fwd_into(q, k, v, out, lse, scale, causal):
    calls["fwd_into"] += 1
    out.copy_(q + k + v)
    lse.zero_()


fake = types.ModuleType("flash_attn_turing")
fake.fwd = fwd
fake.varlen_fwd = varlen_fwd
fake.bwd = bwd
fake.fwd_into = fwd_into
sys.modules["flash_attn_turing"] = fake
sys.modules.pop("flash_attention_interface", None)
fi = importlib.import_module("flash_attention_interface")

passed = 0

def check(name, cond):
    global passed
    if not cond:
        raise AssertionError(name)
    passed += 1
    print(f"[PASS] {name}")


q = torch.randn(2, 8, 4, 32)
k = torch.randn_like(q)
v = torch.randn_like(q)
# Public inference fast path must execute the raw extension, not Function.apply.
c0 = calls["fwd"]
a = fi.flash_attn_func(q, k, v)
b = fi.FlashAttnFunc.apply(q, k, v, None, False, False)
check("fixed-length fastpath matches V2 Function", torch.equal(a, b))
check("fixed-length fastpath calls extension directly", calls["fwd"] == c0 + 2)

# Gradient-enabled path must retain the original autograd Function.
qg = q.clone().requires_grad_(True)
kg = k.clone().requires_grad_(True)
vg = v.clone().requires_grad_(True)
out = fi.flash_attn_func(qg, kg, vg)
check("grad path records autograd", out.requires_grad and out.grad_fn is not None)
out.sum().backward()
check("grad path calls backward extension", qg.grad is not None and kg.grad is not None and vg.grad is not None)

# no_grad with requires_grad inputs is safe to dispatch without autograd.
with torch.no_grad():
    ng = fi.flash_attn_func(qg, kg, vg)
check("no_grad fastpath returns no-grad output", not ng.requires_grad)

# Varlen fast path must match the old Function's forward path.
cu = torch.tensor([0, 3, 8], dtype=torch.int32)
qv = torch.randn(8, 4, 32); kv = torch.randn_like(qv); vv = torch.randn_like(qv)
a = fi.flash_attn_varlen_func(qv, kv, vv, cu, cu, 5, 5)
b = fi.FlashAttnVarlenFunc.apply(qv, kv, vv, cu, cu, 5, 5, None, False, False)
check("varlen fastpath matches V2 Function", torch.equal(a, b))

# Kill switch must route through Function.apply-compatible code.
fi._FASTPATH = False
c0 = calls["fwd"]
_ = fi.flash_attn_func(q, k, v)
check("fastpath kill switch uses legacy Function path", calls["fwd"] == c0 + 1)

print(f"V2.2 PYTHON MOCK SUMMARY: PASS={passed} FAIL=0 TOTAL={passed}")
