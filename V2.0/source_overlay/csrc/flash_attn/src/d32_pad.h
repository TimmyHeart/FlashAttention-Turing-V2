#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Fused D32 <-> D64 helpers (fp16 only). One kernel launch each, replacing the
// zeros() + narrow().copy_() / narrow-copy-back sequence of the V1 padding path.
//   pad:   src [rows, 32] -> dst [rows, 64]   (upper 32 channels written as 0)
//   unpad: src [rows, 64] -> dst [rows, 32]
// All pointers must be 16-byte aligned and densely packed (contiguous).
void launch_pad_d32_to_d64(const void* src, void* dst, int64_t rows, cudaStream_t stream);
void launch_unpad_d64_to_d32(const void* src, void* dst, int64_t rows, cudaStream_t stream);
