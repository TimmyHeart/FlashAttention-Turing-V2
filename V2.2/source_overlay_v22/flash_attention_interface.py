"""Python interface for FlashAttention Turing extension.

V2.2 (host-overhead optimization candidate): flash_attn_func / flash_attn_varlen_func skip
torch.autograd.Function.apply when no gradient will be recorded (grad mode off, or none of
q/k/v requires grad). The result is identical; only the per-call Python/autograd cost is
removed. Set FLASH_TURING_DISABLE_FASTPATH=1 to restore the V2 behaviour exactly.
"""

import os
from typing import Optional, Tuple

import torch
import flash_attn_turing as flash_attn_gpu

# Bound once: avoids a module attribute lookup per call on the hot path.
_ext_fwd = flash_attn_gpu.fwd
_ext_varlen_fwd = flash_attn_gpu.varlen_fwd
_ext_fwd_into = getattr(flash_attn_gpu, "fwd_into", None)  # added by V2.2
_is_grad_enabled = torch.is_grad_enabled
_FASTPATH = os.environ.get("FLASH_TURING_DISABLE_FASTPATH", "0") != "1"


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and not x.is_contiguous() else x


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    out, lse = _ext_fwd(q, k, v, softmax_scale, causal)
    return out, lse


def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, q, k, v, out, lse = [maybe_contiguous(x) for x in (dout, q, k, v, out, lse)]
    dq, dk, dv = flash_attn_gpu.bwd(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        causal,
    )
    return dq, dk, dv


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, lse = _ext_varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
    )
    return out, lse


def _flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, q, k, v, out, lse = [maybe_contiguous(x) for x in (dout, q, k, v, out, lse)]
    dq, dk, dv = flash_attn_gpu.varlen_bwd(
        q,
        k,
        v,
        out,
        lse,
        dout,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
    )
    return dq, dk, dv


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal)
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, k, v))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        return dq, dk, dv, None, None, None


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        q, k, v = (
            qkv.select(dim=2, index=0).contiguous(),
            qkv.select(dim=2, index=1).contiguous(),
            qkv.select(dim=2, index=2).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal)
        is_grad = is_grad_enabled and qkv.requires_grad
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        dqkv[:, :, 0] = dq
        dqkv[:, :, 1] = dk
        dqkv[:, :, 2] = dv
        return dqkv, None, None, None


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        k, v = (
            kv.select(dim=2, index=0).contiguous(),
            kv.select(dim=2, index=1).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal)
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, kv))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        dkv[:, :, 0] = dk
        dkv[:, :, 1] = dv
        return dq, dkv, None, None, None


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
        )
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, k, v))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.softmax_scale,
            ctx.causal,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


class FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        q, k, v = (
            qkv.select(dim=1, index=0).contiguous(),
            qkv.select(dim=1, index=1).contiguous(),
            qkv.select(dim=1, index=2).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            softmax_scale,
            causal,
        )
        is_grad = is_grad_enabled and qkv.requires_grad
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
            ctx.max_seqlen = max_seqlen
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.softmax_scale,
            ctx.causal,
        )
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        dqkv[:, 0] = dq
        dqkv[:, 1] = dk
        dqkv[:, 2] = dv
        return dqkv, None, None, None, None, None


class FlashAttnVarlenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        k, v = (
            kv.select(dim=1, index=0).contiguous(),
            kv.select(dim=1, index=1).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
        )
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, kv))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.softmax_scale,
            ctx.causal,
        )
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        dkv[:, 0] = dk
        dkv[:, 1] = dv
        return dq, dkv, None, None, None, None, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim)
        k: (batch_size, seqlen_k, nheads_k, headdim)
        v: (batch_size, seqlen_k, nheads_k, headdim)
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (batch_size, seqlen_q, nheads, headdim).
    """
    grad_mode = _is_grad_enabled()
    if _FASTPATH and not (grad_mode and (q.requires_grad or k.requires_grad or v.requires_grad)):
        # Inference / no-grad fast path: no autograd.Function, no ctx, nothing saved.
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        return _flash_attn_forward(q, k, v, softmax_scale, causal)[0]
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        grad_mode,
    )


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim)
        kv: (batch_size, seqlen_k, 2, nheads_k, headdim)
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (batch_size, seqlen_q, nheads, headdim).
    """
    return FlashAttnKVPackedFunc.apply(
        q,
        kv,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (total_q, nheads, headdim).
    """
    grad_mode = _is_grad_enabled()
    if _FASTPATH and not (grad_mode and (q.requires_grad or k.requires_grad or v.requires_grad)):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        return _flash_attn_varlen_forward(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal
        )[0]
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        grad_mode,
    )


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (total, nheads, headdim).
    """
    return FlashAttnVarlenQKVPackedFunc.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        kv: (total_k, 2, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (total_q, nheads, headdim).
    """
    return FlashAttnVarlenKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )



def _validate_fwd_into_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Strict validation for the hot-path inference API; never makes hidden copies."""
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must be rank-4 [batch, sequence, heads, head_dim]")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype != torch.float16:
            raise TypeError(f"{name} must have dtype torch.float16")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous; V2.2 fwd_into never copies implicitly")
        if tensor.requires_grad:
            raise ValueError(f"{name} must not require gradients; fwd_into is inference-only")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must have the same batch size")
    if k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
        raise ValueError("k and v must have matching sequence/head dimensions")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, and v must have the same head dimension")
    if q.shape[2] <= 0 or k.shape[2] <= 0 or q.shape[2] % k.shape[2] != 0:
        raise ValueError("query/key-value head counts must be positive and q_heads divisible by kv_heads")


def allocate_fwd_buffers(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate reusable output/LSE buffers for fixed-length FP16 inference."""
    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        raise ValueError("q must be rank-4 [batch, sequence, heads, head_dim]")
    if not q.is_cuda or q.dtype != torch.float16 or not q.is_contiguous():
        raise ValueError("q must be contiguous CUDA float16")
    b, sq, h, _ = q.shape
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    lse = torch.empty((b, h, sq), device=q.device, dtype=torch.float32)
    return out, lse


def flash_attn_func_into(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Run inference directly into caller-owned FP16 output and FP32 LSE buffers.

    Unlike flash_attn_func, this API is deliberately strict: q/k/v must be contiguous,
    all buffers are caller-owned, and no autograd graph or implicit contiguous() copy is made.
    """
    if _ext_fwd_into is None:
        raise RuntimeError("flash_attn_func_into requires the V2.2 native extension build (fwd_into is missing)")
    _validate_fwd_into_inputs(q, k, v)
    if not isinstance(out, torch.Tensor) or not isinstance(lse, torch.Tensor):
        raise TypeError("out and lse must be torch.Tensor instances")
    if out.device != q.device or out.dtype != torch.float16 or not out.is_contiguous():
        raise ValueError("out must be contiguous CUDA float16 on q.device")
    if tuple(out.shape) != tuple(q.shape):
        raise ValueError("out must have the same shape as q")
    expected_lse = (q.shape[0], q.shape[2], q.shape[1])
    if lse.device != q.device or lse.dtype != torch.float32 or not lse.is_contiguous():
        raise ValueError("lse must be contiguous CUDA float32 on q.device")
    if tuple(lse.shape) != expected_lse:
        raise ValueError(f"lse must have shape {expected_lse}")
    if out.requires_grad or lse.requires_grad:
        raise ValueError("out/lse buffers must not require gradients")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    _ext_fwd_into(q, k, v, out, lse, float(softmax_scale), bool(causal))
    return out
