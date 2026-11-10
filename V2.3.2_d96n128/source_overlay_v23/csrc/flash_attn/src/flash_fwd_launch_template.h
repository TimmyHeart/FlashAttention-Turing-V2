
#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include "flash.h"
#include "flash_fwd_kernel.h"
#include <cutlass/numeric_types.h>
#include "static_switch.h"
#include "flash_fwd_tiles.h"
#include <atomic>
using half_t = cutlass::half_t;

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
__global__ __launch_bounds__(Kernel_traits::kNThreads)
// for some reason changing this into params struc is 10% slower for hdim = 128
void flash_fwd_kernel(half_t* __restrict__ q,
                          half_t* __restrict__ k,
                          half_t* __restrict__ v,
                          half_t* __restrict__ o,
                          float* __restrict__ l,
                          int* __restrict__ cu_seqlens_q,
                          int* __restrict__ cu_seqlens_k,
                          int batch_size,
                          int seqlen_q,
                          int seqlen_k,
                          int num_heads,
                          int num_heads_k,
                          int h_h_k_ratio,
                          int head_dim,
                          float softmax_scale,
                          int is_casual)
{
    compute_attn<Kernel_traits, Is_causal, Is_even_MN>(q,
                                           k,
                                           v,
                                           o,
                                           l,
                                           cu_seqlens_q,
                                           cu_seqlens_k,
                                           batch_size,
                                           seqlen_q,
                                           seqlen_k,
                                           num_heads,
                                           num_heads_k,
                                           h_h_k_ratio,
                                           head_dim,
                                           softmax_scale,
                                           is_casual);
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params) {
    auto stream = c10::cuda::getCurrentCUDAStream().stream();


    //auto kernel = flash_fwd_kernel<Kernel_traits, Is_causal>;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;

    const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;

    const bool is_even_MN  = params.cu_seqlens_q == nullptr &&
                             params.cu_seqlens_k == nullptr &&
                             params.seqlen_q % kBlockM == 0 &&
                             params.seqlen_k % kBlockN == 0;

    dim3 dimGrid(num_m_block, params.b, params.h);
    dim3 dimBlock(Kernel_traits::kNThreads);

    // Exact dynamic smem: sQ (kBlockM x D) + sK (kBlockN x D) [+ sV (kBlockN x D) with SEPV].
    // (V1 always requested 64 KB, which pinned occupancy at 1 CTA/SM even for small tiles.)
#ifdef FLASH_TURING_SEPV
    constexpr int kSmemTiles = kBlockM + 2 * kBlockN;
#else
    constexpr int kSmemTiles = kBlockM + kBlockN;
#endif
    constexpr int maxbytes = kSmemTiles * Kernel_traits::kHeadDim * int(sizeof(half_t));
    static_assert(maxbytes <= 65536, "forward tile does not fit in 64 KB of shared memory on SM75");

    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));

    BOOL_SWITCH(is_even_MN, Is_even_MN, [&] {
        // cudaFuncSetAttribute is a host-side driver call; do it once per (kernel, device), not per launch.
        static std::atomic<bool> smem_attr_set[16];
        if (dev < 0 || dev >= 16 || !smem_attr_set[dev].load(std::memory_order_relaxed)) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                flash_fwd_kernel<Kernel_traits, Is_causal, Is_even_MN>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
            if (dev >= 0 && dev < 16) smem_attr_set[dev].store(true, std::memory_order_relaxed);
        }
        flash_fwd_kernel<Kernel_traits, Is_causal, Is_even_MN><<<dimGrid, dimBlock, maxbytes, stream>>>(params.q_ptr,
                                                                                    params.k_ptr,
                                                                                    params.v_ptr,
                                                                                    params.o_ptr,
                                                                                    params.l_ptr,
                                                                                    params.cu_seqlens_q,
                                                                                    params.cu_seqlens_k,
                                                                                    params.b,
                                                                                    params.seqlen_q,
                                                                                    params.seqlen_k,
                                                                                    params.h,
                                                                                    params.h_k,
                                                                                    params.h_h_k_ratio,
                                                                                    params.d,
                                                                                    params.softmax_scale,
                                                                                    params.is_causal);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

    });

}



// V2.3: smaller-M variants (kBlockM = 16 * kNWarps) of the same kernel, for grids that would
// otherwise leave most SMs idle. Instantiated from flash_fwd_hdim<D>_fp16_m<M>[_causal]_sm75.cu.
template<int Headdim, int kBlockM, int kBlockN, int kNWarps, bool Is_causal>
void run_mha_fwd_tile_impl_n(Flash_fwd_params &params) {
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}

template<int Headdim, int kBlockM, int kNWarps, bool Is_causal>
void run_mha_fwd_tile_impl(Flash_fwd_params &params) {
    constexpr static int kBlockN = Fwd_tile_n<Headdim>::value;
    run_mha_fwd_tile_impl_n<Headdim, kBlockM, kBlockN, kNWarps, Is_causal>(params);
}


template<bool Is_causal>
void run_mha_fwd_hdim32(Flash_fwd_params &params) {
    constexpr static int Headdim = 32;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 128;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}


template<bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);


}


template<bool Is_causal>
void run_mha_fwd_hdim96(Flash_fwd_params &params) {
    constexpr static int Headdim = 96;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);


}


template<bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params) {
    constexpr static int Headdim = 64;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 128;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);


}
