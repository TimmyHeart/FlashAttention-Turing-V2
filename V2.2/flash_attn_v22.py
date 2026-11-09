"""Static-buffer CUDA Graph helper for FlashAttention Turing V2.2 inference."""
from __future__ import annotations
from typing import Optional
import torch
from flash_attention_interface import allocate_fwd_buffers, flash_attn_func_into


def _validate_source(name: str, src: torch.Tensor, dst: torch.Tensor) -> None:
    if not isinstance(src, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if src.shape != dst.shape or src.dtype != dst.dtype or src.device != dst.device:
        raise ValueError(f"{name} must match captured shape/dtype/device")
    if not src.is_cuda or not src.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA tensor")
    if src.requires_grad:
        raise ValueError("Static graph input buffers are inference-only")


class FlashAttnStaticGraph:
    """One fixed-shape attention graph with long-lived input/output buffers.

    Best use: upstream layers write directly into runner.q/k/v, then call replay().
    copy_inputs_and_replay() is provided for convenience and includes three input copies.
    The runner is not reentrant; use one instance per concurrent request/stream.
    """
    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        warmup_iters: int = 3,
    ) -> None:
        if warmup_iters < 1:
            raise ValueError("warmup_iters must be >= 1")
        # Validate original inputs before allocating/copying; never silently normalize
        # source tensors or accept a requires_grad tensor into an inference runner.
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                raise ValueError(f"{name} must be a rank-4 torch.Tensor")
            if not tensor.is_cuda or tensor.dtype != torch.float16 or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous CUDA float16")
            if tensor.requires_grad:
                raise ValueError(f"{name} must not require gradients")
        if q.device != k.device or q.device != v.device:
            raise ValueError("q/k/v devices must match")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("q/k/v batch sizes must match")
        if k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
            raise ValueError("k/v sequence and head dimensions must match")
        if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
            raise ValueError("q/k/v head dimensions must match")
        if q.shape[2] <= 0 or k.shape[2] <= 0 or q.shape[2] % k.shape[2]:
            raise ValueError("invalid GQA/MQA head ratio")
        # Allocate matching persistent buffers only after the source metadata is valid.
        self.out, self.lse = allocate_fwd_buffers(q)
        self.q = torch.empty_like(q, memory_format=torch.contiguous_format)
        self.k = torch.empty_like(k, memory_format=torch.contiguous_format)
        self.v = torch.empty_like(v, memory_format=torch.contiguous_format)
        self.q.copy_(q)
        self.k.copy_(k)
        self.v.copy_(v)
        self.device = q.device
        self.softmax_scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
        self.causal = bool(causal)
        self._validate_inputs()

        with torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            warmup = torch.cuda.Stream(device=self.device)
            warmup.wait_stream(current)
            with torch.cuda.stream(warmup):
                for _ in range(warmup_iters):
                    flash_attn_func_into(
                        self.q, self.k, self.v, self.out, self.lse,
                        self.softmax_scale, self.causal,
                    )
            current.wait_stream(warmup)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                flash_attn_func_into(
                    self.q, self.k, self.v, self.out, self.lse,
                    self.softmax_scale, self.causal,
                )
            torch.cuda.synchronize(self.device)

    def _validate_inputs(self) -> None:
        # Reuse the API-level checks without issuing a kernel by validating metadata here.
        if self.q.ndim != 4 or self.k.ndim != 4 or self.v.ndim != 4:
            raise ValueError("q/k/v must be rank-4")
        for name, tensor in (("q", self.q), ("k", self.k), ("v", self.v)):
            if not tensor.is_cuda or tensor.dtype != torch.float16 or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous CUDA float16")
        if self.q.device != self.k.device or self.q.device != self.v.device:
            raise ValueError("q/k/v devices must match")
        if self.q.shape[0] != self.k.shape[0] or self.q.shape[0] != self.v.shape[0]:
            raise ValueError("q/k/v batch sizes must match")
        if self.k.shape[1] != self.v.shape[1] or self.k.shape[2] != self.v.shape[2]:
            raise ValueError("k/v sequence and head dimensions must match")
        if self.q.shape[-1] != self.k.shape[-1] or self.q.shape[-1] != self.v.shape[-1]:
            raise ValueError("q/k/v head dimensions must match")
        if self.q.shape[2] <= 0 or self.k.shape[2] <= 0 or self.q.shape[2] % self.k.shape[2]:
            raise ValueError("invalid GQA/MQA head ratio")

    def replay(self) -> torch.Tensor:
        """Replay using the current contents of persistent q/k/v buffers."""
        self.graph.replay()
        return self.out

    def copy_inputs_and_replay(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Copy new input data into captured buffers, then replay (includes 3 copies)."""
        for name, src, dst in (("q", q, self.q), ("k", k, self.k), ("v", v, self.v)):
            _validate_source(name, src, dst)
        self.q.copy_(q)
        self.k.copy_(k)
        self.v.copy_(v)
        return self.replay()
