"""Autograd adapter and bounded CUDA-Graph runner for FlashAttention Turing.

The graph runner is intentionally limited to inference-time Wan attention calls.
It captures the native SM75 FlashAttention forward op, not the whole Wan block.
Inputs are staged into persistent buffers before replay so graph pointer arguments
remain stable when Q/K/V are newly allocated by each transformer block.
"""

from __future__ import annotations

import logging
import threading
import warnings
import weakref
from typing import Optional, Tuple

import torch
import flash_attn_turing as flash_attn_gpu

_LOG = logging.getLogger("ComfyUI")
_GRAPH_LOCK = threading.RLock()
_GRAPH_SESSIONS = weakref.WeakKeyDictionary()
_GRAPH_SKIP_LOGGED = set()


def _skip_graph(reason: str):
    if reason not in _GRAPH_SKIP_LOGGED:
        _LOG.info("FlashAttention Turing CUDA Graph skipped; using SDPA fallback: %s", reason)
        _GRAPH_SKIP_LOGGED.add(reason)
    return None


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and not x.is_contiguous() else x


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, lse = flash_attn_gpu.fwd(q, k, v, softmax_scale, causal)
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
    dout, q, k, v, out, lse = [
        maybe_contiguous(x) for x in (dout, q, k, v, out, lse)
    ]
    return flash_attn_gpu.bwd(q, k, v, out, lse, dout, softmax_scale, causal)


class _FlashAttnTuringFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, causal, is_grad_enabled):
        scale = q.shape[-1] ** (-0.5) if softmax_scale is None else float(softmax_scale)
        out, lse = _flash_attn_forward(q, k, v, scale, bool(causal))
        if is_grad_enabled and any(t.requires_grad for t in (q, k, v)):
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = scale
            ctx.causal = bool(causal)
        else:
            ctx.save_for_backward()
        return out

    @staticmethod
    def backward(ctx, dout):
        saved = ctx.saved_tensors
        if not saved:
            raise RuntimeError("FlashAttention Turing backward was requested without saved forward tensors")
        q, k, v, out, lse = saved
        dq, dk, dv = _flash_attn_backward(
            dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal
        )
        return dq, dk, dv, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Compute attention for Q/K/V shaped (B, sequence, heads, head_dim)."""
    return _FlashAttnTuringFunction.apply(
        q, k, v, softmax_scale, causal, torch.is_grad_enabled()
    )


class _GraphRecord:
    __slots__ = (
        "graph", "static_q", "static_k", "static_v", "static_out",
        "captured_out", "captured_lse", "warmup_stream", "key", "replays", "stream_id",
    )

    def __init__(self, graph, static_q, static_k, static_v, static_out,
                 captured_out, captured_lse, warmup_stream, key, stream_id, replays=1):
        self.graph = graph
        self.static_q = static_q
        self.static_k = static_k
        self.static_v = static_v
        self.static_out = static_out
        # Keep graph-private allocations alive for the graph's full lifetime.
        self.captured_out = captured_out
        self.captured_lse = captured_lse
        self.warmup_stream = warmup_stream
        self.key = key
        self.replays = replays
        self.stream_id = int(stream_id)

    def reset(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.graph.reset()
        except Exception:
            pass


class _GraphSession:
    def __init__(self, shape_family):
        self.shape_family = shape_family
        self.records = {}
        self.failed = set()
        self.logged = set()

    def clear(self):
        if self.records:
            # A cached output may still be consumed by downstream kernels.
            # Synchronize only on rare shape changes/cache eviction before
            # releasing graph-private storage.
            try:
                first = next(iter(self.records.values()))
                torch.cuda.synchronize(first.static_q.device)
            except Exception:
                pass
        for record in self.records.values():
            record.reset()
        self.records.clear()
        self.failed.clear()
        self.logged.clear()


def _tensor_family(q: torch.Tensor):
    return (
        q.device.type,
        q.device.index,
        tuple(q.shape),
        q.dtype,
    )


def _graph_key(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               scale: float, causal: bool):
    return (
        tuple(q.shape), tuple(k.shape), tuple(v.shape),
        q.dtype, k.dtype, v.dtype,
        q.device.type, q.device.index,
        float(scale), bool(causal),
    )


def _get_session(owner, q):
    family = _tensor_family(q)
    session = _GRAPH_SESSIONS.get(owner)
    if session is None or session.shape_family != family:
        if session is not None:
            session.clear()
        session = _GraphSession(family)
        _GRAPH_SESSIONS[owner] = session
    return session


def _capture_attention_graph(q, k, v, scale: float, causal: bool, key):
    """Capture one native FA call with static Q/K/V and output storage."""
    device = q.device
    static_q = torch.empty(q.shape, dtype=q.dtype, device=device)
    static_k = torch.empty(k.shape, dtype=k.dtype, device=device)
    static_v = torch.empty(v.shape, dtype=v.dtype, device=device)

    static_q.copy_(q)
    static_k.copy_(k)
    static_v.copy_(v)

    current_stream = torch.cuda.current_stream(device)
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        for _ in range(2):
            warm_out, warm_lse = flash_attn_gpu.fwd(
                static_q, static_k, static_v, float(scale), bool(causal)
            )
            del warm_out, warm_lse
    current_stream.wait_stream(warmup_stream)

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(current_stream)
    graph = torch.cuda.CUDAGraph()
    captured_out = None
    captured_lse = None
    try:
        with torch.cuda.graph(
            graph,
            stream=capture_stream,
            capture_error_mode="global",
        ):
            captured_out, captured_lse = flash_attn_gpu.fwd(
                static_q, static_k, static_v, float(scale), bool(causal)
            )
            # captured_out is graph-owned persistent storage (the standard
            # PyTorch CUDA Graph output pattern). Retain it and return it
            # directly; an extra copy would add a full output-sized kernel.
        current_stream.wait_stream(capture_stream)
        graph.replay()
    except Exception:
        try:
            graph.reset()
        except Exception:
            pass
        raise

    return _GraphRecord(
        graph, static_q, static_k, static_v, captured_out,
        captured_out, captured_lse, warmup_stream, key, current_stream.cuda_stream,
    )


def flash_attn_graph_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    owner,
):
    """Run/replay a CUDA Graph for a Wan attention shape.

    Returns None when graph capture is not safe or has failed, allowing the
    caller to fall back to SDPA. Graphs are cached by attention tensor shape;
    the same shape graph can be reused by every Wan block because the native
    attention op has no layer-specific weights.
    """
    if owner is None:
        return _skip_graph("Wan graph owner was not attached to transformer_options")
    try:
        if torch.compiler.is_compiling():
            return _skip_graph("TorchDynamo compilation is active")
    except AttributeError:
        pass
    if torch.is_grad_enabled():
        return _skip_graph("autograd is enabled; graph path is inference-only")
    if q.device.type != "cuda" or not (q.is_cuda and k.is_cuda and v.is_cuda):
        return _skip_graph("Q/K/V are not all CUDA tensors")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        return _skip_graph("Q/K/V dtype is not FP16")
    if not (q.device == k.device == v.device):
        return _skip_graph("Q/K/V are on different CUDA devices")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return _skip_graph("Q/K/V do not have rank 4 (B,S,H,D)")
    if torch.cuda.is_current_stream_capturing():
        # Nested capture is not supported here; the ordinary adapter can be
        # used by an enclosing graph instead.
        return _skip_graph("an outer CUDA Graph capture is already active")

    key = _graph_key(q, k, v, softmax_scale, causal)
    with _GRAPH_LOCK:
        try:
            session = _get_session(owner, q)
            if key in session.failed:
                return None
            record = session.records.get(key)
            if record is None:
                # WAN T2V/I2V uses a small, stable set of attention signatures
                # (normally self-attention and text cross-attention). Bound
                # retained graph memory if a workflow keeps changing layouts.
                if len(session.records) >= 6:
                    session.clear()
                    session = _GraphSession(_tensor_family(q))
                    _GRAPH_SESSIONS[owner] = session
                record = _capture_attention_graph(q, k, v, softmax_scale, causal, key)
                session.records[key] = record
                if key not in session.logged:
                    _LOG.info(
                        "FlashAttention Turing CUDA Graph captured and first replayed: Q=%s K=%s Hq=%d Hkv=%d D=%d",
                        tuple(q.shape), tuple(k.shape), q.shape[-2], k.shape[-2], q.shape[-1],
                    )
                    session.logged.add(key)
            else:
                # Reusing one static output buffer across different CUDA
                # streams could race with downstream consumers. Keep this
                # graph pinned to the stream that first captured it; other
                # streams fall back to SDPA safely.
                current_stream = torch.cuda.current_stream(q.device)
                if int(current_stream.cuda_stream) != record.stream_id:
                    return _skip_graph("current CUDA stream changed after graph capture")
                record.static_q.copy_(q)
                record.static_k.copy_(k)
                record.static_v.copy_(v)
                record.graph.replay()
                if record.replays == 1:
                    _LOG.info(
                        "FlashAttention Turing CUDA Graph reuse confirmed: Q=%s K=%s",
                        tuple(q.shape), tuple(k.shape),
                    )
                record.replays += 1
            return record.static_out
        except Exception as exc:
            try:
                session.failed.add(key)
            except Exception:
                pass
            _LOG.warning(
                "FlashAttention Turing CUDA Graph failed for Q=%s K=%s D=%d; falling back to SDPA: %s",
                tuple(q.shape), tuple(k.shape), q.shape[-1], exc,
            )
            return None
