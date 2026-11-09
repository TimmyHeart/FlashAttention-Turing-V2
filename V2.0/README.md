# FlashAttention Turing V2

An SM75-focused FlashAttention implementation derived from [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing) and [Dao-AILab/FlashAttention](https://github.com/Dao-AILab/flash-attention) , with an integrated native-D32 path and additional Turing-specific kernel and host-side changes.

V2 is a **frozen experimental/performance candidate**. It has been built and tested on a Windows x64 system with an NVIDIA CMP 40HX (Turing, compute capability 7.5). The benchmark results below describe that system only; they should not be treated as universal performance claims for every Turing GPU or software stack.

## Features

### Supported and tested

- Forward and backward attention.
- FP16 inputs on NVIDIA Turing / SM75.
- Head dimensions 32, 64, 96, and 128.
- Causal and non-causal attention.
- MHA, GQA, and MQA.
- Fixed-length and variable-length sequences.
- Custom `softmax_scale`.
- Non-default CUDA streams and CUDA Graph capture/replay.
- A tested non-contiguous input case with inner stride 1.
- Supplemental validation for batch size 2 and different query/key sequence lengths on D32.

The frozen V2 build selects the **native D32 SM75** implementation. A fused D32 pad/unpad path (D32 → D64 → D32) remains in the source as a fallback, but it is not the active D32 route in the benchmarked V2 build.

### Not supported or not verified in this version

- Dropout is not exposed by the current Python interface.
- Local/sliding-window attention masks are not exposed.
- KV-cache attention is not exposed.
- BF16 and other dtypes were not validated by this V2 freeze; the published results are for FP16.
- GPUs other than the CMP 40HX were not runtime-tested as part of this freeze, even though the compiled target is SM75.

## V2 implementation

The frozen build integrates four work streams:

- **A — fused D32 pad/unpad:** retained as a compatibility/fallback route.
- **B — native D32:** native SM75 forward and backward kernels for head dimension 32; selected in the frozen build.
- **C — kernel/host optimizations:** SEPV (separate V shared-memory buffer and synchronization changes) and FASTSM (exp2-domain softmax) are enabled together. Host-side launch and initialization changes are also included.
- **D — guardrails:** argument validation, supported-head-dimension checks, CUDA error checks, and protection for output rows that may otherwise be left unwritten by early-return paths.

The exact build defines are:

```text
FLASH_TURING_NATIVE_D32=1
FLASH_TURING_SEPV=1
FLASH_TURING_FASTSM=1
```

SEPV and FASTSM were enabled together in the frozen binary. Their individual performance and stability contributions have **not** been isolated by ablation tests.

## Performance

Benchmarks were run on one NVIDIA CMP 40HX (SM75, 8 GB VRAM) using PyTorch 2.11.0+cu130 and CUDA 13.0. Results compare the V2 extension with PyTorch `scaled_dot_product_attention` (SDPA) on the same machine.

### Forward pass — CUDA Graph replay

The ratios below are `SDPA graph latency / V2 graph latency`; values greater than 1 mean V2 was faster. The matched profile used batch size 1, 4 heads, FP16, causal and non-causal cases, and sequence lengths 128, 256, and 512. Each table cell is the average of the causal and non-causal ratios derived from the recorded latencies.

| Head dimension | S=128 | S=256 | S=512 |
|---|---:|---:|---:|
| D32 | **1.82x** | **2.01x** | **2.10x** |
| D64 | **1.43x** | **1.54x** | **1.66x** |
| D96 | 0.79x | 0.87x | **1.24x** |
| D128 | 0.73x | 0.78x | **1.13x** |

<img width="680" height="331" alt="image" src="https://github.com/user-attachments/assets/0825e45b-e133-46b0-9db5-1268641bf6ba" />

In the expanded profile (batch sizes 1/2/4, 8 heads, sequence lengths 128/256/512/1024), the best recorded B=1, H=8 graph result was **2.59x at D32, S=512**. D96 and D128 also improved relative to SDPA at longer sequence lengths, while they did not lead at the shortest matched lengths.

### Forward pass — ordinary CUDA-event timing

CUDA-event medians for the matched profile were:

| Head dimension | V2 FlashAttention | PyTorch SDPA |
|---|---:|---:|
| D32 | 26.15 µs | 20.45 µs |
| D64 | 26.85 µs | 20.65 µs |
| D96 | 27.80 µs | 18.30 µs |
| D128 | 30.10 µs | 19.45 µs |

In this ordinary event-timing profile, V2 was slower than SDPA for all four dimensions. The graph-replay advantage therefore does **not** imply lower latency for ordinary eager calls. The event measurements include GPU idle gaps caused by CPU launch pacing; they are not isolated kernel-only timings.

**Practical interpretation:** the current measurements support testing V2 first for D32/D64 workloads that use CUDA Graphs. They do not support enabling V2 globally for ordinary eager workloads. D96/D128 should be evaluated at the actual sequence lengths and execution mode used by the application.

### Backward pass

Backward correctness passed for the tested fixed-length and variable-length cases. Backward timing was also collected, but the current comparison is not an equivalent API-to-API measurement: the FlashAttention side calls the extension's direct backward entry point, while the SDPA side calls `torch.autograd.grad` on a retained graph. For that reason, this README does not claim a backward speedup ratio. A fair comparison requires matching the full autograd boundary and graph-reuse policy on both sides.

### Benchmark limitations

- The V1 and V2 numbers currently available were collected with different benchmark harnesses/runs. The historical V1/V2 table is directional only, not a controlled same-harness A/B result.
- CUPTI reported `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED (42)` on the CMP 40HX. Kernel/operation counts were therefore unavailable; no kernel-count or achieved-compute-throughput claim is made.
- Occupancy was not measured. The compiler reported no register spills for the native D32 translation units, but D32 causal forward used up to approximately 255 registers in the captured build output. This remains a tuning target, not a proven performance bottleneck.
- Peak VRAM, long-duration soak behavior, and Compute Sanitizer results are not part of this freeze.

Raw logs and derived tables are preserved in the V2 freeze package.

## How to use FlashAttention

The Python module defines fixed-length and variable-length functions, including packed QKV and packed KV entry points:

```python
import torch  # Import torch before the extension on the tested Windows setup.

from flash_attention_interface import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
)

# Fixed-length example: (batch, sequence, heads, head_dim)
q = torch.randn(1, 256, 4, 32, device="cuda", dtype=torch.float16)
k = torch.randn(1, 256, 4, 32, device="cuda", dtype=torch.float16)
v = torch.randn(1, 256, 4, 32, device="cuda", dtype=torch.float16)

out = flash_attn_func(q, k, v, causal=False)
```

The main operation is scaled dot-product attention:

```text
softmax(Q @ K^T * softmax_scale) @ V
```

The interface is intentionally smaller than the full upstream FlashAttention API. Consult `flash_attention_interface.py` for exact signatures and tensor layouts. The current functions do not expose dropout, local-window masks, or KV-cache arguments.

## Requirements

### Exact environment validated for this V2 freeze

- Windows x64.
- Python 3.12.0.
- PyTorch 2.11.0+cu130.
- CUDA runtime/toolkit 13.0; NVCC 13.0.88.
- MSVC toolset 14.29.30133.
- NVIDIA GPU with compute capability 7.5 (SM75).
- Runtime validation performed on NVIDIA CMP 40HX, 8 GB VRAM, driver 596.36.
- CUTLASS submodule commit: `df18f5e4f5de76bed8be1de8e4c245f2f5ec3020`.

These are the versions tested for the published V2 result, not a claim that every combination of Windows, CUDA, PyTorch, or compiler versions is supported.

## Build notes

V2 is currently distributed as source/patch material, not as a portable prebuilt wheel. The frozen Windows build script contains machine-specific Visual Studio and CUDA paths; edit those paths for another machine before building.

The reproducible flow is:

1. Prepare a clean working copy of the exact V1 source baseline. Do not apply V2 onto the frozen V1 archive or a locally modified source tree.
2. Extract the V2 ABCD bundle separately.
3. From the V1 project root (the directory containing `setup.py`), apply the overlay:

   ```bat
   py -3.12 C:\path\to\V2_bundle\apply_merged_abcd.py
   ```

   The script verifies expected V1 source hashes, creates `.backup_abcd_v1`, and refuses to overwrite unexpected local modifications.
4. From a fresh CMD, run the V2 build script:

   ```bat
   C:\path\to\V2_bundle\BUILD_ABCD_FULL_SM75.bat
   ```

5. Run correctness before benchmarking:

   ```bat
   C:\path\to\V2_bundle\RUN_TESTS_ABCD.bat
   ```

6. After correctness passes, run the benchmark script:

   ```bat
   C:\path\to\V2_bundle\RUN_BENCHMARK_ABCD_COMPARISON.bat
   ```

The build script used for the recorded result sets the three V2 macros shown above and builds for `sm_75`. Its absolute toolchain paths are specific to the original test machine and must be adjusted when used elsewhere.

## Validation

The frozen binary passed **78/78** checks on the CMP 40HX:

| Test group | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Fixed-length forward + backward | 24 | 0 | 0 |
| Variable-length forward + backward | 24 | 0 | 0 |
| Custom softmax scale | 8 | 0 | 0 |
| Non-default CUDA stream | 8 | 0 | 0 |
| CUDA Graph forward + backward | 8 | 0 | 0 |
| Supplemental guardrails | 6 | 0 | 0 |
| **Total** | **78** | **0** | **0** |

Coverage includes D32/D64/D96/D128; MHA/GQA/MQA; causal/non-causal; fixed and variable sequence lengths; forward and backward; custom scale; CUDA Graph replay; non-default stream; batch size 2 with Q>K and Q<K on D32; a non-contiguous inner-stride-1 case; and rejection of unsupported head dimension 48.

This is a finite validation set, not a proof of correctness for every possible tensor shape or workload.

## Current status and next priorities

V2 is frozen as an experimental candidate. V1 remains the immutable known-good fallback. The next engineering priorities are:

1. Run V1 and V2 with the exact same benchmark harness, timing parameters, shapes, and GPU session.
2. Isolate the three D32 routes: V1 padded D64, A fused pad/unpad, and B native D32.
3. Benchmark SEPV and FASTSM independently to identify their individual effects.
4. Reduce host/API launch overhead, especially for short sequences and eager execution.
5. Tune D96/D128 for short sequences.
6. Measure occupancy and investigate D32 causal-forward register pressure.
7. Add sustained-load, peak-VRAM, longer-sequence, and SM75-compatible sanitizer testing.
8. Find a profiler/tracing method that works on CMP 40HX; the current CUPTI path does not report kernel activities on this device.

Do not overwrite the V1 or V2 freeze. Kernel changes and new performance claims should be published as a new version with fresh build, correctness, and benchmark evidence.

## Acknowledgements

This work is derived from the Turing FlashAttention implementation by [ssiu](https://github.com/ssiu/flash-attention-turing) and incorporates the V2 changes documented in this repository. Please retain the upstream attribution and applicable licenses when redistributing source.
