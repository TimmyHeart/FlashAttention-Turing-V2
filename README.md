# FlashAttention Turing V2.3

An SM75-focused FlashAttention implementation derived from [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing) and [Dao-AILab/FlashAttention](https://github.com/Dao-AILab/flash-attention), with an integrated native-D32 path, host-side fast path dispatch (V2.2), and adaptive GPU tile size selection (V2.3).

**V2.3** is a kernel-tiling optimization update built on top of the host-dispatch improvements of V2.2. It addresses the SRAM memory constraint bottleneck on Turing (SM75) hardware for larger head dimensions (D96/D128) at short sequence lengths ($S \le 256$) by introducing **adaptive tile size selection ($M \in \{128, 64, 32\}$)**.

With V2.3, FlashAttention-Turing now **outperforms PyTorch SDPA across 100% of tested configurations**, achieving peak speedups of up to **3.07x vs SDPA** (at D32, S=128, B=1, H=4) and resolving the previous D96/D128 regression.

The original **V2.2 report is retained in [Part II](#part-ii--v22-report-retained)** and the **V2 freeze report is retained in [Part III](#part-iii--v2-frozen-report-retained)**.

## Contents

- [Part I — V2.3](#part-i--v23)
  - [What changed in V2.3](#what-changed-in-v23)
  - [V2.3 results at a glance](#v23-results-at-a-glance)
  - [V2.3 tile-tuning benchmark](#v23-tile-tuning-benchmark)
  - [V2.3 validation](#v23-validation)
  - [Status of the V2 priorities](#status-of-the-v2-priorities)
- [Part II — V2.2 report (retained)](#part-ii--v22-report-retained)
- [Part III — V2 frozen report (retained)](#part-iii--v2-frozen-report-retained)

---

# Part I — V2.3

## What changed in V2.3

- **Adaptive Tile Size Selection ($M \in \{128, 64, 32\}$).** V2.0 and V2.2 used a fixed sequence block tile size ($M=128$). V2.3 enables dynamic/configurable tiling per sequence length and head dimension, reducing shared-memory pressure on SM75 GPUs.
- **D96 & D128 Short-Sequence Fix.** By switching to smaller tile sizes ($M=64$ or $M=32$) for short sequence lengths ($S \le 256$) on D96 and D128, V2.3 converts previous performance regressions (0.73x–0.89x in V2.0/V2.2) into speedups (**1.10x–1.92x vs SDPA**).
- **Inherited V2.2 Host Dispatch.** V2.3 retains the low-overhead C++ `fwd_into` inference dispatch introduced in V2.2, maintaining ~13.7 µs median CPU host time.
- **Automatic Selector Calibration.** Evaluated total regret across fixed cost metrics to establish optimal tile selection thresholds per workload grid.

## V2.3 results at a glance

Measured on one NVIDIA CMP 40HX (Turing, SM75), FP16, grid across $D \in \{32, 64, 96, 128\}$, $S \in \{128, 256, 512, 1024\}$, causal and non-causal.

- **Peak Speedup ($B=1, H=4$):** **3.07x vs SDPA** at D32, S=128 (causal=1) using $M=32$ tile size (3.64 µs vs 11.17 µs SDPA).
- **Peak Speedup ($B=2, H=8$):** **2.52x vs SDPA** at D32, S=512 (causal=0) using $M=64$ tile size (25.37 µs vs 64.01 µs SDPA).
- **Short Sequence D96/D128 Resolution:**
  - **D96 S128 (B=1, H=4):** **1.34x vs SDPA** with $M=32$ (baseline M128 was 0.77x).
  - **D128 S128 (B=1, H=4):** **1.17x vs SDPA** with $M=32$ (baseline M128 was 0.75x).
- **Correctness:** 100% pass (**72/72** suite passed across all head dimensions, MHA/GQA/MQA, fixed and varlen, custom scales, streams, and CUDA Graphs).

---

## V2.3 tile-tuning benchmark

### Environment

| Item | Value |
|---|---|
| Date of run | 2026-11-09 |
| GPU | NVIDIA CMP 40HX (SM75, 34 SMs, 8192 MiB VRAM) |
| Driver | 596.36 |
| Python / PyTorch / CUDA | 3.12.0 / 2.11.0+cu130 / 13.0 |
| Grid Tiles Tested | `tiles=[128, 64, 32]` |

---

### Profile 1: $B=1, H=4$ (Single-Batch / LLM Generation Workload)

Captured with CUDA Graph execution (`n_ops=16`). Latencies in microseconds (µs).

| Configuration | SDPA Latency | Baseline M128 | Tile M64 | Tile M32 | Best Tile | Best Speedup vs SDPA | Baseline M128 Ratio |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| **D32 S128 Causal=0** | 10.96 µs | 5.75 µs | 4.35 µs | **4.17 µs** | **M32** | **2.63x** | 1.91x |
| **D32 S128 Causal=1** | 11.17 µs | 6.48 µs | 3.72 µs | **3.64 µs** | **M32** | **3.07x** | 1.72x |
| **D32 S256 Causal=0** | 14.20 µs | 6.92 µs | 5.15 µs | **4.99 µs** | **M32** | **2.84x** | 2.05x |
| **D32 S256 Causal=1** | 14.34 µs | 7.34 µs | 5.30 µs | **5.24 µs** | **M32** | **2.74x** | 1.95x |
| **D32 S512 Causal=0** | 26.45 µs | 12.24 µs | **8.88 µs** | 10.72 µs | **M64** | **2.98x** | 2.16x |
| **D32 S512 Causal=1** | 25.48 µs | 11.89 µs | **8.88 µs** | 9.55 µs | **M64** | **2.87x** | 2.14x |
| **D64 S128 Causal=0** | 9.20 µs | 6.26 µs | **4.42 µs** | 4.92 µs | **M64** | **2.08x** | 1.47x |
| **D64 S128 Causal=1** | 9.49 µs | 6.98 µs | 4.87 µs | **4.85 µs** | **M32** | **1.96x** | 1.36x |
| **D64 S256 Causal=0** | 15.78 µs | 9.95 µs | **6.86 µs** | 8.86 µs | **M64** | **2.30x** | 1.59x |
| **D64 S256 Causal=1** | 15.94 µs | 10.74 µs | **7.28 µs** | 7.70 µs | **M64** | **2.19x** | 1.48x |
| **D64 S512 Causal=0** | 30.08 µs | 17.34 µs | **12.65 µs** | 27.21 µs | **M64** | **2.38x** | 1.73x |
| **D64 S512 Causal=1** | 28.79 µs | 18.21 µs | **12.47 µs** | 20.48 µs | **M64** | **2.31x** | 1.58x |
| **D96 S128 Causal=0** | 6.85 µs | 8.89 µs | 5.72 µs | **5.13 µs** | **M32** | **1.34x** | 0.77x *(loss)* |
| **D96 S128 Causal=1** | 6.91 µs | 8.56 µs | **5.73 µs** | 6.60 µs | **M64** | **1.21x** | 0.81x *(loss)* |
| **D96 S256 Causal=0** | 12.43 µs | 14.59 µs | 9.17 µs | **8.63 µs** | **M32** | **1.44x** | 0.85x *(loss)* |
| **D96 S256 Causal=1** | 11.89 µs | 13.41 µs | **9.17 µs** | 11.47 µs | **M64** | **1.30x** | 0.89x *(loss)* |
| **D96 S512 Causal=0** | 31.53 µs | 26.14 µs | **17.69 µs** | 23.42 µs | **M64** | **1.78x** | 1.21x |
| **D96 S512 Causal=1** | 30.08 µs | 23.04 µs | **17.23 µs** | 24.54 µs | **M64** | **1.75x** | 1.31x |
| **D128 S128 Causal=0** | 7.44 µs | 9.95 µs | 6.71 µs | **6.36 µs** | **M32** | **1.17x** | 0.75x *(loss)* |
| **D128 S128 Causal=1** | 7.49 µs | 10.18 µs | **6.82 µs** | 6.85 µs | **M64** | **1.10x** | 0.74x *(loss)* |
| **D128 S256 Causal=0** | 13.52 µs | 16.14 µs | **11.01 µs** | 11.18 µs | **M64** | **1.23x** | 0.84x *(loss)* |
| **D128 S256 Causal=1** | 12.71 µs | 17.46 µs | **10.97 µs** | 11.21 µs | **M64** | **1.16x** | 0.73x *(loss)* |
| **D128 S512 Causal=0** | 35.67 µs | 28.74 µs | **21.41 µs** | 37.74 µs | **M64** | **1.67x** | 1.24x |
| **D128 S512 Causal=1** | 33.83 µs | 32.20 µs | **20.16 µs** | 28.65 µs | **M64** | **1.68x** | 1.05x |

---

### Profile 2: $B=2, H=8$ (Multi-Batch / Parallel Attention Workload)

Captured with CUDA Graph execution (`n_ops=16`). Latencies in microseconds (µs).

| Configuration | SDPA Latency | Baseline M128 | Tile M64 | Tile M32 | Best Tile | Best Speedup vs SDPA |
|---|---:|---:|---:|---:|:---:|:---:|
| **D32 S128 Causal=0** | 11.41 µs | 5.86 µs | **4.74 µs** | 6.00 µs | **M64** | **2.41x** |
| **D32 S128 Causal=1** | 11.45 µs | 6.55 µs | **5.15 µs** | 5.41 µs | **M64** | **2.22x** |
| **D32 S256 Causal=0** | 18.91 µs | **7.65 µs** | 7.87 µs | 11.14 µs | **M128** | **2.47x** |
| **D32 S256 Causal=1** | 16.95 µs | 7.71 µs | **6.93 µs** | 9.46 µs | **M64** | **2.44x** |
| **D32 S512 Causal=0** | 64.01 µs | 25.50 µs | **25.37 µs** | 32.19 µs | **M64** | **2.52x** |
| **D32 S512 Causal=1** | 45.07 µs | 21.59 µs | **20.32 µs** | 24.00 µs | **M64** | **2.22x** |
| **D32 S1024 Causal=0** | 211.15 µs | 95.59 µs | **91.96 µs** | 117.93 µs | **M64** | **2.30x** |
| **D32 S1024 Causal=1** | 132.35 µs | 61.66 µs | **59.59 µs** | 72.04 µs | **M64** | **2.22x** |
| **D64 S128 Causal=0** | 10.36 µs | 6.48 µs | **5.03 µs** | 9.82 µs | **M64** | **2.06x** |
| **D64 S128 Causal=1** | 9.91 µs | 7.32 µs | **5.37 µs** | 9.16 µs | **M64** | **1.85x** |
| **D64 S256 Causal=0** | 21.97 µs | **11.22 µs** | 14.67 µs | 32.33 µs | **M128** | **1.96x** |
| **D64 S256 Causal=1** | 19.25 µs | **11.56 µs** | 14.13 µs | 23.60 µs | **M128** | **1.66x** |
| **D64 S512 Causal=0** | 79.99 µs | **39.41 µs** | 48.84 µs | 102.57 µs | **M128** | **2.03x** |
| **D64 S512 Causal=1** | 53.27 µs | **34.22 µs** | 36.36 µs | 62.82 µs | **M128** | **1.56x** |
| **D64 S1024 Causal=0** | 267.53 µs | **147.52 µs** | 180.81 µs | 362.92 µs | **M128** | **1.81x** |
| **D64 S1024 Causal=1** | 164.75 µs | **103.19 µs** | 111.85 µs | 189.16 µs | **M128** | **1.60x** |
| **D96 S128 Causal=0** | 10.16 µs | 9.16 µs | **6.70 µs** | 8.54 µs | **M64** | **1.52x** |
| **D96 S128 Causal=1** | 9.52 µs | 8.84 µs | **6.41 µs** | 8.45 µs | **M64** | **1.49x** |
| **D96 S256 Causal=0** | 33.42 µs | **17.36 µs** | 19.99 µs | 25.14 µs | **M128** | **1.92x** |
| **D96 S256 Causal=1** | 27.83 µs | **14.78 µs** | 17.19 µs | 24.00 µs | **M128** | **1.88x** |
| **D96 S512 Causal=0** | 124.70 µs | **66.76 µs** | 76.97 µs | 90.10 µs | **M128** | **1.87x** |
| **D96 S512 Causal=1** | 82.41 µs | **50.92 µs** | 52.93 µs | 70.87 µs | **M128** | **1.62x** |
| **D96 S1024 Causal=0** | 461.50 µs | **238.23 µs** | 272.24 µs | 330.95 µs | **M128** | **1.94x** |
| **D96 S1024 Causal=1** | 265.36 µs | **145.65 µs** | 159.97 µs | 221.97 µs | **M128** | **1.82x** |
| **D128 S128 Causal=0** | 11.42 µs | 10.35 µs | **8.06 µs** | 13.52 µs | **M64** | **1.42x** |
| **D128 S128 Causal=1** | 10.09 µs | 10.73 µs | **7.79 µs** | 13.62 µs | **M64** | **1.30x** |
| **D128 S256 Causal=0** | 38.16 µs | **19.89 µs** | 25.66 µs | 43.48 µs | **M128** | **1.92x** |
| **D128 S256 Causal=1** | 30.98 µs | **20.01 µs** | 21.36 µs | 36.08 µs | **M128** | **1.55x** |
| **D128 S512 Causal=0** | 141.63 µs | **75.60 µs** | 92.02 µs | 152.76 µs | **M128** | **1.87x** |
| **D128 S512 Causal=1** | 93.08 µs | 67.01 µs | **63.05 µs** | 99.61 µs | **M64** | **1.48x** |
| **D128 S1024 Causal=0** | 535.96 µs | **270.90 µs** | 330.45 µs | 559.91 µs | **M128** | **1.98x** |
| **D128 S1024 Causal=1** | 304.58 µs | 192.06 µs | **189.09 µs** | 304.26 µs | **M64** | **1.61x** |

---

## V2.3 validation

The full correctness suite executed on the NVIDIA CMP 40HX and achieved **72/72 PASS**:

| Test group | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Fixed-length forward + backward (MHA/GQA/MQA, causal/non-causal) | 24 | 0 | 0 |
| Variable-length forward + backward (MHA/GQA/MQA, causal/non-causal) | 24 | 0 | 0 |
| Custom softmax scale | 8 | 0 | 0 |
| Non-default CUDA stream | 8 | 0 | 0 |
| CUDA Graph forward + backward | 8 | 0 | 0 |
| **Total** | **72** | **0** | **0** |

- **Tolerances:** Forward `atol/rtol=0.003`; Gradients `atol/rtol=0.008`.
- **Worst observed differences:** Forward `0.00195`, dQ `0.00195`, dK `0.00391`, dV `0.00586`.

---

## Status of the V2 priorities

Tracking engineering priorities across releases:

| # | V2 priority | V2.2 status | V2.3 status |
|---:|---|---|---|
| 1 | Same-harness comparison | Partly addressed (V2 vs V2.2) | Fully addressed across tile sizes $M \in \{128, 64, 32\}$ |
| 2 | Isolate D32 routes | Not addressed | Native D32 route benchmarked across all tile sizes |
| 3 | Benchmark SEPV / FASTSM | Not addressed | Not addressed |
| 4 | Reduce Host Launch Overhead | **Addressed** (~13.7 µs host time) | **Retained** (V2.2 fast-path inherited) |
| 5 | Tune D96/D128 for short sequence | Not addressed | **FULLY RESOLVED** (M64/M32 adaptive tiling beats SDPA) |
| 6 | Measure Occupancy & Register pressure | Not addressed | Not addressed |
| 7 | Sustained-load / Sanitizer testing | Not addressed | Not addressed |

---

# Part II — V2.2 report (retained)

*The text below is the original V2.2 report, kept as written for historical reference.*

## What changed in V2.2

- **Inference dispatch (fast path).** The public forward call has a V2.2 inference dispatch that avoids the retained V2 `torch.autograd.Function` wrapper. In the harness this is the `v22_public_fastpath` candidate; the legacy path is kept as `v2_function_apply` for A/B comparison.
- **Forward-into entry point (`fwd_into`).** The extension exposes a forward variant that writes into a caller-provided output buffer.
- **Kernels unchanged.** No kernel, build-define, or numerical change was introduced in V2.2.

## V2.2 results at a glance

Measured on one NVIDIA CMP 40HX, FP16, B=1, H=4, D ∈ {32, 64, 96, 128}, S ∈ {128, 256, 512}, causal and non-causal.

- **Eager forward host overhead:** the V2.2 fast path has a median CPU-side cost of **13.7 µs per call**, versus **30.5 µs** for the retained V2 path (about 55% lower) and **25.7 µs** for PyTorch SDPA (about 47% lower).
- **Eager CUDA-event time:** the median is **13.7 µs** for V2.2, versus **27.0 µs** for V2 and **24.1 µs** for SDPA.
- **CUDA Graph replay:** V2.2 is the same as V2 within ±0.02x on every cell. The best graph result is **2.17x vs SDPA at D32, S=512** (16 ops per graph).

### Eager forward — host overhead and event time

Medians over 24 configurations (D32/64/96/128 × S128/256/512 × causal/non-causal). Lower is better.

| Candidate | host µs (A/B run) | host µs (full run) | event µs (A/B run) | event µs (full run) |
|---|---:|---:|---:|---:|
| PyTorch SDPA | 25.70 | 25.52 | 24.09 | 23.03 |
| V2 (`v2_function_apply`) | 30.53 | 30.36 | 27.04 | 25.01 |
| **V2.2 public fast path** | **13.71** | **13.69** | **13.70** | **14.60** |
| Direct ext forward (allocating) | 12.27 | 12.22 | 13.44 | 12.60 |
| V2.2 checked forward-into | 14.70 | 13.53 | 14.48 | 16.02 |
| Direct forward-into (raw) | 4.84 | 4.73 | 13.61 | 12.11 |

### CUDA Graph replay — V2.2 vs V2

Ratio = `SDPA graph latency / candidate graph latency` (16 operations per graph, V2 in parentheses):

| Head dimension | S=128 | S=256 | S=512 |
|---|---:|---:|---:|
| D32 | **1.81x** (1.81x) | **2.02x** (2.00x) | **2.17x** (2.17x) |
| D64 | **1.41x** (1.41x) | **1.54x** (1.54x) | **1.66x** (1.66x) |
| D96 | 0.79x (0.79x) | 0.87x (0.87x) | **1.25x** (1.26x) |
| D128 | 0.73x (0.73x) | 0.80x (0.80x) | **1.14x** (1.14x) |

---

# Part III — V2 frozen report (retained)

*The text below is the original V2 freeze report, kept as written for historical reference.*

## Features

- Forward and backward attention for NVIDIA Turing / SM75.
- Supported Head Dimensions: 32, 64, 96, 128.
- Supported Attention Variants: MHA, GQA, MQA.

## V2 implementation build defines

```text
FLASH_TURING_NATIVE_D32=1
FLASH_TURING_SEPV=1
FLASH_TURING_FASTSM=1
```

## Performance (V2 Baseline CUDA Graph Replay)

Ratio = `SDPA graph latency / V2 graph latency`:

| Head dimension | S=128 | S=256 | S=512 |
|---|---:|---:|---:|
| D32 | **1.82x** | **2.01x** | **2.10x** |
| D64 | **1.43x** | **1.54x** | **1.66x** |
| D96 | 0.79x | 0.87x | **1.24x** |
| D128 | 0.73x | 0.78x | **1.13x** |

---

## How to use FlashAttention

```python
import torch

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

---

## Acknowledgements

This work is derived from the Turing FlashAttention implementation by [ssiu](https://github.com/ssiu/flash-attention-turing) and incorporates the V2, V2.2, and V2.3 changes documented in this repository. Please retain the upstream attribution and applicable licenses when redistributing source.
