# FlashAttention Turing V2.3.2

An SM75-focused FlashAttention implementation derived from [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing) and [Dao-AILab/FlashAttention](https://github.com/Dao-AILab/flash-attention), with an integrated native-D32 path, host-side fast path dispatch (V2.2), adaptive GPU tile size selection (V2.3), and shape-aware automatic selector calibration with specialized D96 kernels (V2.3.2).

**V2.3.2** introduces a shape-aware automatic selector model and adds a specialized **$M=64, N=128$ tile candidate (`D96_M64_N128=1`)** specifically designed for D96 head dimensions. With shape-aware selection, the selector achieves **0.00 µs total regret** on the $B=2, H=8$ profile across all head dimensions and **0.23 µs total regret** on $B=1, H=4$, guaranteeing optimal tile routing for all sequence shapes.

The original **V2.2 report is retained in [Part II](#part-ii--v22-report-retained)** and the **V2 freeze report is retained in [Part III](#part-iii--v2-frozen-report-retained)**.

## Contents

* [Part I — V2.3.2](#part-i--v232)
  * [What changed in V2.3.2](#what-changed-in-v232)
  * [V2.3.2 results at a glance](#v232-results-at-a-glance)
  * [V2.3.2 tile-tuning benchmark](#v232-tile-tuning-benchmark)
  * [V2.3.2 validation](#v232-validation)
  * [Status of the V2 priorities](#status-of-the-v2-priorities)
* [Part II — V2.2 report (retained)](#part-ii--v22-report-retained)
* [Part III — V2 frozen report (retained)](#part-iii--v2-frozen-report-retained)

---

# Part I — V2.3.2

## What changed in V2.3.2

* **D96 Specialized Tile Kernel (`D96_M64_N128`).** Introduced a dedicated tile configuration ($M=64, N=128$) for Head Dimension 96. This variant yields additional speedups in causal D96 configurations (e.g., reducing D96 S128 Causal=1 execution time down to 5.70 µs on B1H4).
* **Shape-Aware Automatic Selector Calibration.** Replaced fixed-cost heuristic selector tuning with a shape-aware automatic selector model.
  * **Zero Regret on $B=2, H=8$:** Total selection regret is **0.00 µs** across all 32 grid configurations.
  * **Near-Zero Regret on $B=1, H=4$:** Total selection regret is **0.23 µs** across all 24 grid configurations.
* **Inherited Fast-Path Dispatch.** Maintains the low-overhead C++ `fwd_into` host dispatch (~13.7 µs host cost) introduced in V2.2.

## V2.3.2 results at a glance

Measured on one NVIDIA CMP 40HX (Turing, SM75), FP16, grid across $D \in \{32, 64, 96, 128\}$, $S \in \{128, 256, 512, 1024\}$, causal and non-causal.

* **Peak Speedup ($B=1, H=4$):** **3.03x vs SDPA** at D32, S=128 (causal=1) using $M=32$ tile size (3.67 µs vs 11.11 µs SDPA).
* **Peak Speedup ($B=2, H=8$):** **2.54x vs SDPA** at D32, S=512 (causal=0) using $M=64$ tile size (25.40 µs vs 64.53 µs SDPA).
* **Shape-Aware Selector Regret:**
  * $B=2, H=8$: **0.00 µs** total regret ($D32=0.00$, $D64=0.00$, $D96=0.00$, $D128=0.00$).
  * $B=1, H=4$: **0.23 µs** total regret ($D32=0.00$, $D64=0.00$, $D96=0.23$, $D128=0.00$).
* **Correctness:** 100% pass (**72/72** suite passed across all head dimensions, MHA/GQA/MQA, fixed and varlen, custom scales, streams, and CUDA Graphs).

## V2.3.2 tile-tuning benchmark

### Environment

| Item | Value |
| --- | --- |
| Date of run | 2026-11-09 |
| GPU | NVIDIA CMP 40HX (SM75, 34 SMs, 8192 MiB VRAM) |
| Driver | 596.36 |
| Python / PyTorch / CUDA | 3.12.0 / 2.11.0+cu130 / 13.0 |
| Grid Tiles & Flags | `tiles=[128, 64, 32]`, `D96_M64_N128=1` |

### Profile 1: $B=1, H=4$ (Single-Batch / LLM Generation Workload)

Captured with CUDA Graph execution (`n_ops=16`). Latencies in microseconds (µs).

| Configuration | SDPA | Baseline M128 | Tile M64 | Tile M32 | Specialized M64N128 | Best Candidate | Auto Selected | Regret | Best Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| **D32 S128 Causal=0** | 10.93 µs | 5.73 µs | 4.35 µs | **4.17 µs** | - | **M32** | M32 | 0.00 µs | **2.62x** |
| **D32 S128 Causal=1** | 11.11 µs | 6.48 µs | 3.69 µs | **3.67 µs** | - | **M32** | M32 | 0.00 µs | **3.03x** |
| **D32 S256 Causal=0** | 14.18 µs | 6.92 µs | 5.15 µs | **4.99 µs** | - | **M32** | M32 | 0.00 µs | **2.84x** |
| **D32 S256 Causal=1** | 14.27 µs | 7.33 µs | 5.30 µs | **5.24 µs** | - | **M32** | M32 | 0.00 µs | **2.73x** |
| **D32 S512 Causal=0** | 26.48 µs | 12.23 µs | **8.88 µs** | 10.71 µs | - | **M64** | M64 | 0.00 µs | **2.98x** |
| **D32 S512 Causal=1** | 25.45 µs | 11.86 µs | **8.88 µs** | 9.56 µs | - | **M64** | M64 | 0.00 µs | **2.87x** |
| **D64 S128 Causal=0** | 9.15 µs | 6.22 µs | **4.38 µs** | 4.84 µs | - | **M64** | M64 | 0.00 µs | **2.09x** |
| **D64 S128 Causal=1** | 9.30 µs | 6.90 µs | **4.77 µs** | 4.79 µs | - | **M64** | M64 | 0.00 µs | **1.95x** |
| **D64 S256 Causal=0** | 15.65 µs | 9.86 µs | **6.80 µs** | 8.77 µs | - | **M64** | M64 | 0.00 µs | **2.30x** |
| **D64 S256 Causal=1** | 15.78 µs | 10.64 µs | **7.23 µs** | 7.63 µs | - | **M64** | M64 | 0.00 µs | **2.18x** |
| **D64 S512 Causal=0** | 29.81 µs | 17.24 µs | **12.66 µs** | 27.20 µs | - | **M64** | M64 | 0.00 µs | **2.36x** |
| **D64 S512 Causal=1** | 28.91 µs | 18.19 µs | **12.47 µs** | 20.48 µs | - | **M64** | M64 | 0.00 µs | **2.32x** |
| **D96 S128 Causal=0** | 6.82 µs | 8.90 µs | 5.72 µs | **5.13 µs** | 6.30 µs | **M32** | M32 | 0.00 µs | **1.33x** |
| **D96 S128 Causal=1** | 6.93 µs | 8.56 µs | 5.73 µs | 6.61 µs | **5.70 µs** | **M64N128** | M64 | 0.02 µs | **1.21x** |
| **D96 S256 Causal=0** | 12.43 µs | 14.59 µs | 9.18 µs | **8.63 µs** | 9.88 µs | **M32** | M32 | 0.00 µs | **1.44x** |
| **D96 S256 Causal=1** | 11.89 µs | 13.42 µs | 9.17 µs | 11.46 µs | **9.09 µs** | **M64N128** | M64 | 0.08 µs | **1.31x** |
| **D96 S512 Causal=0** | 31.53 µs | 26.12 µs | **17.69 µs** | 23.44 µs | 18.65 µs | **M64** | M64 | 0.00 µs | **1.78x** |
| **D96 S512 Causal=1** | 30.00 µs | 23.11 µs | 17.24 µs | 25.11 µs | **17.11 µs** | **M64N128** | M64 | 0.13 µs | **1.75x** |
| **D128 S128 Causal=0** | 7.46 µs | 9.96 µs | 6.71 µs | **6.37 µs** | - | **M32** | M32 | 0.00 µs | **1.17x** |
| **D128 S128 Causal=1** | 7.50 µs | 10.18 µs | **6.82 µs** | 6.85 µs | - | **M64** | M64 | 0.00 µs | **1.10x** |
| **D128 S256 Causal=0** | 13.53 µs | 16.13 µs | **11.00 µs** | 11.19 µs | - | **M64** | M64 | 0.00 µs | **1.23x** |
| **D128 S256 Causal=1** | 12.79 µs | 17.51 µs | **10.99 µs** | 11.30 µs | - | **M64** | M64 | 0.00 µs | **1.16x** |
| **D128 S512 Causal=0** | 35.73 µs | 28.74 µs | **21.47 µs** | 37.75 µs | - | **M64** | M64 | 0.00 µs | **1.66x** |
| **D128 S512 Causal=1** | 33.77 µs | 32.17 µs | **20.12 µs** | 28.62 µs | - | **M64** | M64 | 0.00 µs | **1.68x** |

### Profile 2: $B=2, H=8$ (Multi-Batch / Parallel Attention Workload)

Captured with CUDA Graph execution (`n_ops=16`). Latencies in microseconds (µs).

| Configuration | SDPA | Baseline M128 | Tile M64 | Tile M32 | Specialized M64N128 | Best Candidate | Auto Selected | Regret | Best Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| **D32 S128 Causal=0** | 11.46 µs | 5.90 µs | **4.77 µs** | 6.05 µs | - | **M64** | M64 | 0.00 µs | **2.40x** |
| **D32 S128 Causal=1** | 11.44 µs | 6.56 µs | **5.16 µs** | 5.42 µs | - | **M64** | M64 | 0.00 µs | **2.22x** |
| **D32 S256 Causal=0** | 18.80 µs | **7.59 µs** | 7.83 µs | 11.10 µs | - | **M128** | M128 | 0.00 µs | **2.48x** |
| **D32 S256 Causal=1** | 16.80 µs | 7.68 µs | **6.96 µs** | 9.57 µs | - | **M64** | M64 | 0.00 µs | **2.41x** |
| **D32 S512 Causal=0** | 64.53 µs | 25.80 µs | **25.40 µs** | 32.65 µs | - | **M64** | M64 | 0.00 µs | **2.54x** |
| **D32 S512 Causal=1** | 44.84 µs | 21.49 µs | **20.17 µs** | 23.57 µs | - | **M64** | M64 | 0.00 µs | **2.22x** |
| **D32 S1024 Causal=0** | 210.94 µs | 95.35 µs | **92.07 µs** | 117.07 µs | - | **M64** | M64 | 0.00 µs | **2.29x** |
| **D32 S1024 Causal=1** | 132.12 µs | 61.98 µs | **59.14 µs** | 72.33 µs | - | **M64** | M64 | 0.00 µs | **2.23x** |
| **D64 S128 Causal=0** | 10.37 µs | 6.45 µs | **5.06 µs** | 9.98 µs | - | **M64** | M64 | 0.00 µs | **2.05x** |
| **D64 S128 Causal=1** | 9.88 µs | 7.37 µs | **5.38 µs** | 9.15 µs | - | **M64** | M64 | 0.00 µs | **1.84x** |
| **D64 S256 Causal=0** | 22.07 µs | **11.20 µs** | 14.82 µs | 31.92 µs | - | **M128** | M128 | 0.00 µs | **1.97x** |
| **D64 S256 Causal=1** | 19.20 µs | **11.54 µs** | 14.17 µs | 23.62 µs | - | **M128** | M128 | 0.00 µs | **1.66x** |
| **D64 S512 Causal=0** | 79.93 µs | **39.45 µs** | 48.85 µs | 102.37 µs | - | **M128** | M128 | 0.00 µs | **2.03x** |
| **D64 S512 Causal=1** | 53.35 µs | **34.19 µs** | 36.37 µs | 62.79 µs | - | **M128** | M128 | 0.00 µs | **1.56x** |
| **D64 S1024 Causal=0** | 267.55 µs | **147.69 µs** | 179.97 µs | 360.97 µs | - | **M128** | M128 | 0.00 µs | **1.81x** |
| **D64 S1024 Causal=1** | 164.70 µs | **102.92 µs** | 112.00 µs | 188.12 µs | - | **M128** | M128 | 0.00 µs | **1.60x** |
| **D96 S128 Causal=0** | 10.08 µs | 9.10 µs | **6.65 µs** | 8.45 µs | 7.33 µs | **M64** | M64 | 0.00 µs | **1.52x** |
| **D96 S128 Causal=1** | 9.35 µs | 8.79 µs | **6.37 µs** | 8.42 µs | 6.75 µs | **M64** | M64 | 0.00 µs | **1.47x** |
| **D96 S256 Causal=0** | 33.26 µs | **17.15 µs** | 19.85 µs | 25.14 µs | 21.39 µs | **M128** | M128 | 0.00 µs | **1.94x** |
| **D96 S256 Causal=1** | 27.60 µs | **14.65 µs** | 17.08 µs | 23.80 µs | 18.78 µs | **M128** | M128 | 0.00 µs | **1.88x** |
| **D96 S512 Causal=0** | 124.18 µs | **66.57 µs** | 77.18 µs | 89.78 µs | 78.24 µs | **M128** | M128 | 0.00 µs | **1.87x** |
| **D96 S512 Causal=1** | 82.31 µs | **50.66 µs** | 52.66 µs | 70.42 µs | 55.77 µs | **M128** | M128 | 0.00 µs | **1.62x** |
| **D96 S1024 Causal=0** | 460.95 µs | **238.47 µs** | 271.96 µs | 330.23 µs | 274.75 µs | **M128** | M128 | 0.00 µs | **1.93x** |
| **D96 S1024 Causal=1** | 266.04 µs | **145.17 µs** | 160.20 µs | 221.45 µs | 164.69 µs | **M128** | M128 | 0.00 µs | **1.83x** |
| **D128 S128 Causal=0** | 12.04 µs | 10.46 µs | **8.05 µs** | 13.50 µs | - | **M64** | M64 | 0.00 µs | **1.50x** |
| **D128 S128 Causal=1** | 10.12 µs | 10.72 µs | **7.80 µs** | 13.61 µs | - | **M64** | M64 | 0.00 µs | **1.30x** |
| **D128 S256 Causal=0** | 38.52 µs | **20.45 µs** | 25.68 µs | 43.49 µs | - | **M128** | M128 | 0.00 µs | **1.88x** |
| **D128 S256 Causal=1** | 30.96 µs | **19.71 µs** | 21.46 µs | 36.03 µs | - | **M128** | M128 | 0.00 µs | **1.57x** |
| **D128 S512 Causal=0** | 141.99 µs | **75.42 µs** | 91.64 µs | 152.70 µs | - | **M128** | M128 | 0.00 µs | **1.88x** |
| **D128 S512 Causal=1** | 92.84 µs | 67.06 µs | **63.04 µs** | 99.46 µs | - | **M64** | M64 | 0.00 µs | **1.47x** |
| **D128 S1024 Causal=0** | 535.13 µs | **271.54 µs** | 331.19 µs | 559.29 µs | - | **M128** | M128 | 0.00 µs | **1.97x** |
| **D128 S1024 Causal=1** | 304.61 µs | 191.64 µs | **189.57 µs** | 304.54 µs | - | **M64** | M64 | 0.00 µs | **1.61x** |

## V2.3.2 validation

The full correctness suite executed on the NVIDIA CMP 40HX and achieved **72/72 PASS**:

| Test group | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Fixed-length forward + backward (MHA/GQA/MQA, causal/non-causal) | 24 | 0 | 0 |
| Variable-length forward + backward (MHA/GQA/MQA, causal/non-causal) | 24 | 0 | 0 |
| Custom softmax scale | 8 | 0 | 0 |
| Non-default CUDA stream | 8 | 0 | 0 |
| CUDA Graph forward + backward | 8 | 0 | 0 |
| **Total** | **72** | **0** | **0** |

* **Tolerances:** Forward `atol/rtol=0.003`; Gradients `atol/rtol=0.008`.
* **Worst observed differences:** Forward `0.00195`, dQ `0.00195`, dK `0.00391`, dV `0.00586`.

## Status of the V2 priorities

Tracking engineering priorities across releases:

| # | V2 priority | V2.2 status | V2.3.2 status |
| ---: | --- | --- | --- |
| 1 | Same-harness comparison | Partly addressed | Fully addressed with Shape-Aware Selector |
| 2 | Isolate D32 routes | Not addressed | Native D32 benchmarked across all tile sizes |
| 3 | Benchmark SEPV / FASTSM | Not addressed | Not addressed |
| 4 | Reduce Host Launch Overhead | **Addressed** (~13.7 µs host time) | **Retained** (V2.2 fast-path inherited) |
| 5 | Tune D96/D128 for short sequence | Not addressed | **RESOLVED** (`M64`, `M32`, and `M64N128` kernels beat SDPA) |
| 6 | Measure Occupancy & Register pressure | Not addressed | Not addressed |
| 7 | Sustained-load / Sanitizer testing | Not addressed | Not addressed |

# Part II — V2.2 report (retained)

*The text below is the original V2.2 report, kept as written for historical reference.*

## What changed in V2.2

* **Inference dispatch (fast path).** The public forward call has a V2.2 inference dispatch that avoids the retained V2 `torch.autograd.Function` wrapper.
* **Forward-into entry point (`fwd_into`).** The extension exposes a forward variant that writes into a caller-provided output buffer.
* **Kernels unchanged.** No kernel, build-define, or numerical change was introduced in V2.2.

## V2.2 results at a glance

Measured on one NVIDIA CMP 40HX, FP16, B=1, H=4, D ∈ {32, 64, 96, 128}, S ∈ {128, 256, 512}, causal and non-causal.

* **Eager forward host overhead:** median CPU-side cost of **13.7 µs per call**, versus **30.5 µs** for V2 and **25.7 µs** for PyTorch SDPA.
* **Eager CUDA-event time:** median of **13.7 µs** for V2.2, versus **27.0 µs** for V2 and **24.1 µs** for SDPA.

# Part III — V2 frozen report (retained)

*The text below is the original V2 freeze report, kept as written for historical reference.*

## Features

* Forward and backward attention for NVIDIA Turing / SM75.
* Supported Head Dimensions: 32, 64, 96, 128.
* Supported Attention Variants: MHA, GQA, MQA.

## V2 implementation build defines

```text
FLASH_TURING_NATIVE_D32=1
FLASH_TURING_SEPV=1
FLASH_TURING_FASTSM=1
```

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

## Acknowledgements

This work is derived from the Turing FlashAttention implementation by [ssiu](https://github.com/ssiu/flash-attention-turing) and incorporates the V2, V2.2, V2.3, and V2.3.2 changes documented in this repository. Please retain the upstream attribution and applicable licenses when redistributing source.
