# FlashAttention Turing V2.2

An SM75-focused FlashAttention implementation derived from [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing) and [Dao-AILab/FlashAttention](https://github.com/Dao-AILab/flash-attention), with an integrated native-D32 path and additional Turing-specific kernel and host-side changes.

**V2.2** is a host-side/dispatch update on top of the frozen V2 build. It targets the CPU launch overhead of forward-only (inference) calls in eager mode. The attention kernels are **unchanged** from V2: both the V2 and V2.2 paths call the same `ext.fwd` / `mha_fwd` kernel, so CUDA Graph kernel timings are expected to match V2 and the A/B below confirms that.

V2.2 is an **experimental/performance candidate**. It has been built and tested on a Windows x64 system with an NVIDIA CMP 40HX (Turing, compute capability 7.5). The benchmark results describe that system only; they should not be treated as universal performance claims for every Turing GPU or software stack.

The complete, original **V2 frozen report is retained unchanged in [Part II](#part-ii--v2-frozen-report-retained)**. V2 remains the frozen record of the kernel work; V1 remains the immutable known-good fallback.

## Contents

- [Part I — V2.2](#part-i--v22)
  - [What changed in V2.2](#what-changed-in-v22)
  - [V2.2 results at a glance](#v22-results-at-a-glance)
  - [V2.2 benchmark](#v22-benchmark)
  - [V2.2 validation](#v22-validation)
  - [How to use](#how-to-use-flashattention)
  - [Status of the V2 priorities](#status-of-the-v2-priorities)
- [Part II — V2 frozen report (retained)](#part-ii--v2-frozen-report-retained)

---

# Part I — V2.2

## What changed in V2.2

- **Inference dispatch (fast path).** The public forward call has a V2.2 inference dispatch that avoids the retained V2 `torch.autograd.Function` wrapper. In the harness this is the `v22_public_fastpath` candidate; the legacy path is kept as `v2_function_apply` for A/B comparison.
- **Forward-into entry point (`fwd_into`).** The extension exposes a forward variant that writes into a caller-provided output buffer. A checked variant and a raw (unchecked) variant were measured in eager mode, and a forward-into variant was also captured in CUDA Graphs.
- **Kernels unchanged.** No kernel, build-define, or numerical change is claimed for V2.2. The V2 build defines (`FLASH_TURING_NATIVE_D32=1`, `FLASH_TURING_SEPV=1`, `FLASH_TURING_FASTSM=1`) and the supported feature set are as described in Part II.

<!-- TODO(author): state the exact conditions under which the fast path is taken (e.g. no autograd graph required) and the exact fwd_into signature, once confirmed against flash_attention_interface.py. -->

## V2.2 results at a glance

Measured on one NVIDIA CMP 40HX, FP16, B=1, H=4, D ∈ {32, 64, 96, 128}, S ∈ {128, 256, 512}, causal and non-causal.

- **Eager forward host overhead:** the V2.2 fast path has a median CPU-side cost of **13.7 µs per call**, versus **30.5 µs** for the retained V2 path (about 55% lower) and **25.7 µs** for PyTorch SDPA (about 47% lower).
- **Eager CUDA-event time:** the median is **13.7 µs** for V2.2, versus **27.0 µs** for V2 and **24.1 µs** for SDPA. In V2 this metric was where V2 lost to SDPA; the V2.2 fast path is below SDPA in 24/24 configurations in the final A/B run (23/24 in the second run).
- **CUDA Graph replay:** V2.2 is the same as V2 within ±0.02x on every cell, as expected for an unchanged kernel. The best graph result is **2.17x vs SDPA at D32, S=512** (16 ops per graph).
- **D96/D128 at short sequence lengths** are still slower than SDPA in graph replay (0.73x–0.89x for S ≤ 256). V2.2 does not change this.
- **Correctness:** the all-head-dimension suite passed **72/72** (see [V2.2 validation](#v22-validation)).

## V2.2 benchmark

### Environment

| Item | Value |
|---|---|
| Date of run | 2026-11-09 |
| GPU | NVIDIA CMP 40HX, compute capability 7.5, 8192 MiB |
| Driver | 596.36 |
| Python | 3.12.0 |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| Extension | `flash_attn_turing` (cp312, win_amd64), `fwd_into` available, fast path enabled |

The V2.2 logs do not re-record the NVCC, MSVC, or CUTLASS versions; the V2 freeze values in Part II are the last recorded ones.

### Method

- Batch size 1, 4 heads, FP16. Grid: 4 head dimensions × 3 sequence lengths (128/256/512) × 2 causal settings.
- Modes: `eager` (one call per measurement), `graph` (CUDA Graph replay with 1, 4, or 16 operations captured per graph), and `graph_copy` (inputs copied into static graph buffers before each replay).
- Per-row columns recorded: `host_us` (CPU-side time), `event_us` (CUDA-event time), `e2e_us` (end-to-end time recorded by the harness), and per-operation versions of each for graph modes. The `max_abs_error` column is 0.0 in every row of every run.
- Two full runs are reported: the **final A/B run** (`v2_v22_final_ab`, which includes the retained V2 path in graph mode) and a **host-overhead full run** (`v22_host_overhead_full_full`). A smaller quick run (`v22_host_overhead_quick_full`) is also preserved. All three used the same machine and software stack.
- Summary statistics below are medians over the 24 eager configurations unless stated otherwise. Each configuration was recorded once per run, so no per-configuration variance is available.

Candidates:

| Candidate | Meaning |
|---|---|
| `sdpa` | PyTorch `scaled_dot_product_attention` |
| `v2_function_apply` | Retained V2 legacy `autograd.Function` path |
| `v22_public_fastpath` | V2.2 inference dispatch through the public interface |
| `direct_ext_fwd_alloc` | Direct extension forward call, allocating the output |
| `direct_fwd_into_raw` | Direct forward-into call with no wrapper (lower bound on host cost) |
| `v22_checked_fwd_into` | Forward-into with the V2.2 checks |
| `v22_fwd_into` | Forward-into inside a CUDA Graph |
| `v22_fwd_into_copy_replay` | Forward-into graph with an input copy before each replay |

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

Observations:

- The V2.2 fast path is below both SDPA and V2 in host time in 24/24 configurations in the A/B run (23/24 vs SDPA and 24/24 vs V2 in the full run). Its host time is flat across head dimension and sequence length (about 13.7 µs median in every head-dimension group).
- Median host-time ratios (A/B run): SDPA/V2.2 = 1.86x and V2/V2.2 = 2.21x.
- The fast path is close to the direct allocating extension call (about 12.3 µs), which suggests that most of the remaining host cost is output allocation and the extension call itself rather than Python-side dispatch. The raw forward-into call, which avoids both, is about 4.8 µs.

CUDA-event median by sequence length (A/B run, all head dimensions and both causal settings):

| S | SDPA | V2 | V2.2 fast path | Direct forward-into (raw) |
|---|---:|---:|---:|---:|
| 128 | 23.36 | 26.64 | **12.53** | 8.48 |
| 256 | 23.79 | 26.93 | **14.62** | 14.12 |
| 512 | 30.91 | 28.60 | **25.08** | 21.15 |

The eager event-time advantage **shrinks as sequence length grows**: at S=512 V2.2 is within about 4 µs of the raw direct call, which indicates the call is mostly kernel-bound there and not launch-bound. As in the V2 report, eager event measurements include GPU idle gaps caused by CPU launch pacing; they are not isolated kernel-only timings. Lower host overhead therefore shows up as lower event time mainly when the GPU would otherwise be waiting for the CPU.

A few individual rows are noisy: for example, the V2.2 fast path measured 22.8 µs host time for D32, S=256, non-causal in the A/B run against about 13.7 µs typical, and the full run has a few similar outliers. Medians are used to limit their influence.

### CUDA Graph replay — V2.2 vs V2

Ratio = `SDPA graph latency / candidate graph latency`, using per-operation CUDA-event time and averaging the causal and non-causal ratios, the same convention as the V2 table in Part II. Values greater than 1 mean the candidate was faster than SDPA. Data: final A/B run, B=1, H=4.

**16 operations per graph** (V2 value in parentheses):

| Head dimension | S=128 | S=256 | S=512 |
|---|---:|---:|---:|
| D32 | **1.81x** (1.81x) | **2.02x** (2.00x) | **2.17x** (2.17x) |
| D64 | **1.41x** (1.41x) | **1.54x** (1.54x) | **1.66x** (1.66x) |
| D96 | 0.79x (0.79x) | 0.87x (0.87x) | **1.25x** (1.26x) |
| D128 | 0.73x (0.73x) | 0.80x (0.80x) | **1.14x** (1.14x) |

**1 operation per graph** (V2 value in parentheses):

| Head dimension | S=128 | S=256 | S=512 |
|---|---:|---:|---:|
| D32 | **1.59x** (1.58x) | **1.82x** (1.81x) | **1.97x** (1.98x) |
| D64 | **1.32x** (1.30x) | **1.46x** (1.46x) | **1.59x** (1.58x) |
| D96 | 0.82x (0.82x) | 0.89x (0.89x) | **1.24x** (1.24x) |
| D128 | 0.78x (0.78x) | 0.80x (0.81x) | **1.12x** (1.12x) |

- V2.2 and V2 agree within 0.02x in every cell. This is the expected result for an unchanged kernel, and it is evidence that the V2.2 dispatch does not change graph-replay performance.
- The 1-operation ratios are lower than the 16-operation ratios because the fixed per-replay cost is a larger share of the (smaller) V2.2 time when only one operation is captured; per-operation time falls for both V2.2 and SDPA as more operations are captured.
- The numbers differ slightly from the V2 table in Part II (for example D32, S=512: 2.10x there against 1.97x–2.17x here depending on graph size) because they come from different runs and graph configurations. Treat the V2 table in Part II as the V2 freeze record and this table as the same-harness V2 vs V2.2 comparison.
- The best recorded graph ratio in these runs is **2.17x** (D32, S=512, 16 ops per graph); the full run gives 2.17x–2.18x for the same cell.

### Graph replay with input copy

`v22_fwd_into_copy_replay` copies the inputs into the graph's static buffers before each replay. This adds a fixed cost: the median host time was about 35.6 µs per replay at 1 operation per graph and 38.3 µs at 16, against about 7.1 µs and 9.2 µs for plain replay. For short sequences this copy dominates, so graph replay with changing inputs benefits much less than the plain-replay numbers above suggest. Plain replay measures a graph whose input buffers are reused as they are.

### Practical interpretation

- **Eager, forward-only, D32–D128, short to medium sequences:** in the V2 report the eager path was slower than PyTorch SDPA; in this harness the V2.2 fast path is faster than SDPA on this system, mostly because of lower per-call host overhead. This is a reasonable candidate to test in eager inference workloads at B=1, H=4.
- **CUDA Graph workloads:** unchanged from V2. D32/D64 are faster than SDPA at all measured lengths; D96/D128 are slower at S ≤ 256 and faster at S=512.
- **Longer sequences in eager mode:** the advantage narrows and the kernel dominates; re-measure at the application's own shapes.
- **Training / backward:** V2.2 does not claim any change. The retained V2 autograd path and backward kernels are as described in Part II.

### Benchmark limitations

- Only B=1, H=4, FP16, and S ∈ {128, 256, 512} were measured. Larger batch sizes, more heads, and longer sequences were not part of the V2.2 runs.
- One GPU, one driver, one software stack, one session per run. Each configuration was recorded once per run.
- Forward only. No backward timing was collected for V2.2, and the backward comparison caveat in Part II still applies.
- V1 was not included in the V2.2 harness, so there is still no same-harness V1 vs V2 vs V2.2 comparison. The V2 vs V2.2 comparison is same-harness.
- The `max_abs_error` column is 0.0 for all rows. The harness logs do not state the reference it is computed against, so it is not used here as a standalone correctness claim; correctness is covered by the validation suite below.
- CUPTI is still unavailable on the CMP 40HX (`CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED (42)` in V2), so no kernel counts, occupancy, or achieved-throughput claims are made. Peak VRAM, long-duration soak behavior, and Compute Sanitizer results are not part of V2.2.

Raw logs and CSVs for the V2.2 runs (`v2_v22_final_ab`, `v22_host_overhead_full_full`, `v22_host_overhead_quick_full`) and the benchmark environment record are preserved alongside this README.

## V2.2 validation

The all-head-dimension correctness suite was run against the V2.2 build on the CMP 40HX and passed **72/72**:

| Test group | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Fixed-length forward + backward (MHA/GQA/MQA, causal/non-causal) | 24 | 0 | 0 |
| Variable-length forward + backward (MHA/GQA/MQA, causal/non-causal) | 24 | 0 | 0 |
| Custom softmax scale | 8 | 0 | 0 |
| Non-default CUDA stream | 8 | 0 | 0 |
| CUDA Graph forward + backward | 8 | 0 | 0 |
| **Total** | **72** | **0** | **0** |

- Head dimensions 32, 64, 96, and 128 are covered in every group. Fixed-length cases use S=129; variable-length cases use sequence lengths (17, 33, 64).
- Tolerances: forward atol/rtol 0.003; gradients atol/rtol 0.008.
- Worst observed absolute differences across the suite: forward 0.00195, dQ 0.00195, dK 0.00391, dV 0.00586. The dV maximum is about 73% of its tolerance, so the margin there is smaller than for the other outputs.

This 72-case suite does **not** include the 6 supplemental guardrail checks that are part of the V2 78/78 record (batch size 2 with Q>K and Q<K on D32, a non-contiguous inner-stride-1 case, and rejection of unsupported head dimension 48). Those results are recorded for V2 in Part II; a V2.2 re-run of them is not part of the logs in this release. The suite calls the public `flash_attn_func`; the log does not state which dispatch route (fast path or legacy) each case exercised.

This is a finite validation set, not a proof of correctness for every possible tensor shape or workload.

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

The interface is intentionally smaller than the full upstream FlashAttention API. Consult `flash_attention_interface.py` for exact signatures, tensor layouts, and the V2.2 `fwd_into` entry point. The current functions do not expose dropout, local-window masks, or KV-cache arguments.

Build and install steps for the V2.2 sources follow the V2 flow described in [Build notes](#build-notes). The V2.2 build is validated on the environment listed in the benchmark section above.

<!-- TODO(author): add the V2.2-specific overlay/build script names if they differ from the V2 ABCD bundle scripts. -->

## Status of the V2 priorities

The V2 report (Part II) listed eight next priorities. Status after V2.2:

| # | V2 priority | V2.2 status |
|---:|---|---|
| 1 | Same-harness, same-session comparison | Partly addressed: V2 vs V2.2 is same-harness. V1 is not in the V2.2 logs. |
| 2 | Isolate the three D32 routes (padded D64, fused pad/unpad, native D32) | Not addressed in V2.2 logs. |
| 3 | Benchmark SEPV and FASTSM independently | Not addressed. |
| 4 | Reduce host/API launch overhead, especially for short sequences and eager execution | **Addressed for forward-only eager calls** (about 55% lower host time than V2 at the median). |
| 5 | Tune D96/D128 for short sequences | Not addressed; graph ratios are unchanged (kernel unchanged). |
| 6 | Measure occupancy and investigate D32 causal-forward register pressure | Not addressed. |
| 7 | Sustained-load, peak-VRAM, longer-sequence, and sanitizer testing | Not addressed. |
| 8 | Find a profiler that works on CMP 40HX | Not addressed; CUPTI data is still unavailable. |

New versions should keep the same rule as before: do not overwrite the V1 or V2 freezes. Kernel changes and new performance claims should be published as a new version with fresh build, correctness, and benchmark evidence.

---

# Part II — V2 frozen report (retained)

*The text below is the original V2 freeze report, kept as written. "This version" and "the frozen V2 build" in this part refer to V2. Where Part I gives newer V2.2 information, Part I takes precedence for V2.2 and this part remains the record of V2.*

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

> **Note added with V2.2:** the eager-mode result above applies to the V2 path. The V2.2 inference fast path in Part I addresses this eager-mode overhead; see the V2.2 benchmark for same-harness numbers.

### Backward pass

Backward correctness passed for the tested fixed-length and variable-length cases. Backward timing was also collected, but the current comparison is not an equivalent API-to-API measurement: the FlashAttention side calls the extension's direct backward entry point, while the SDPA side calls `torch.autograd.grad` on a retained graph. For that reason, this README does not claim a backward speedup ratio. A fair comparison requires matching the full autograd boundary and graph-reuse policy on both sides.

### Benchmark limitations

- The V1 and V2 numbers currently available were collected with different benchmark harnesses/runs. The historical V1/V2 table is directional only, not a controlled same-harness A/B result.
- CUPTI reported `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED (42)` on the CMP 40HX. Kernel/operation counts were therefore unavailable; no kernel-count or achieved-compute-throughput claim is made.
- Occupancy was not measured. The compiler reported no register spills for the native D32 translation units, but D32 causal forward used up to approximately 255 registers in the captured build output. This remains a tuning target, not a proven performance bottleneck.
- Peak VRAM, long-duration soak behavior, and Compute Sanitizer results are not part of this freeze.

Raw logs and derived tables are preserved in the V2 freeze package.

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

## Current status and next priorities (as of the V2 freeze)

V2 is frozen as an experimental candidate. V1 remains the immutable known-good fallback. The next engineering priorities were:

1. Run V1 and V2 with the exact same benchmark harness, timing parameters, shapes, and GPU session.
2. Isolate the three D32 routes: V1 padded D64, A fused pad/unpad, and B native D32.
3. Benchmark SEPV and FASTSM independently to identify their individual effects.
4. Reduce host/API launch overhead, especially for short sequences and eager execution.
5. Tune D96/D128 for short sequences.
6. Measure occupancy and investigate D32 causal-forward register pressure.
7. Add sustained-load, peak-VRAM, longer-sequence, and SM75-compatible sanitizer testing.
8. Find a profiler/tracing method that works on CMP 40HX; the current CUPTI path does not report kernel activities on this device.

Do not overwrite the V1 or V2 freeze. Kernel changes and new performance claims should be published as a new version with fresh build, correctness, and benchmark evidence. See [Status of the V2 priorities](#status-of-the-v2-priorities) in Part I for what V2.2 has and has not addressed.

---

## Acknowledgements

This work is derived from the Turing FlashAttention implementation by [ssiu](https://github.com/ssiu/flash-attention-turing) and incorporates the V2 and V2.2 changes documented in this repository. Please retain the upstream attribution and applicable licenses when redistributing source.
