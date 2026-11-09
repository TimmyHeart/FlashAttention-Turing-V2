#include "d32_pad.h"

namespace {

constexpr int kThreads = 256;

// Row of 32 halves = 4 x uint4 (16 B). Row of 64 halves = 8 x uint4.
__global__ void pad_d32_to_d64_kernel(const uint4* __restrict__ src,
                                      uint4* __restrict__ dst,
                                      int64_t rows) {
    const int64_t n = rows * 4;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const int64_t r = i >> 2;
        const int c = static_cast<int>(i & 3);
        dst[r * 8 + c] = src[i];
        dst[r * 8 + 4 + c] = make_uint4(0u, 0u, 0u, 0u);
    }
}

__global__ void unpad_d64_to_d32_kernel(const uint4* __restrict__ src,
                                        uint4* __restrict__ dst,
                                        int64_t rows) {
    const int64_t n = rows * 4;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const int64_t r = i >> 2;
        const int c = static_cast<int>(i & 3);
        dst[i] = src[r * 8 + c];
    }
}

inline int num_blocks(int64_t rows) {
    const int64_t n = rows * 4;
    int64_t b = (n + kThreads - 1) / kThreads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;  // grid-stride loop handles the rest
    return static_cast<int>(b);
}

}  // namespace

void launch_pad_d32_to_d64(const void* src, void* dst, int64_t rows, cudaStream_t stream) {
    if (rows <= 0) return;
    pad_d32_to_d64_kernel<<<num_blocks(rows), kThreads, 0, stream>>>(
        static_cast<const uint4*>(src), static_cast<uint4*>(dst), rows);
}

void launch_unpad_d64_to_d32(const void* src, void* dst, int64_t rows, cudaStream_t stream) {
    if (rows <= 0) return;
    unpad_d64_to_d32_kernel<<<num_blocks(rows), kThreads, 0, stream>>>(
        static_cast<const uint4*>(src), static_cast<uint4*>(dst), rows);
}
