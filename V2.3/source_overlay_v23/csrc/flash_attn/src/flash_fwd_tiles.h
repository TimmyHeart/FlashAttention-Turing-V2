#pragma once
// V2.3 forward-tile selection support (see README_V23.md).
//
// The V2/V2.2 forward kernels always use kBlockM = 128 (8 warps x 16 rows). With the
// shapes that matter for short-sequence inference (B*H small, S <= 512) that launches only
// ceil(S/128) * B * H CTAs, far fewer than the SM count, so most SMs sit idle.
// V2.3 additionally compiles kBlockM = 64 (4 warps) and kBlockM = 32 (2 warps) variants
// of the SAME kernel body; the host picks one per launch.
#include "flash.h"

// kBlockN used by the existing 128-row kernels, kept for the smaller-M variants.
template <int Headdim>
struct Fwd_tile_n {
    static constexpr int value = (Headdim == 96 || Headdim == 128) ? 64 : 128;
};

// Explicitly specialised in flash_fwd_hdim<D>_fp16_m<M>[_causal]_sm75.cu.
// kBlockM is the number of query rows per CTA (64 or 32).
template <int Headdim, bool Is_causal, int kBlockM>
void run_mha_fwd_tile_(Flash_fwd_params &params);
