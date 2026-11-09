#include "flash_fwd_launch_template.h"

#include <cutlass/numeric_types.h>

using half_t = cutlass::half_t;

template<>
void run_mha_fwd_tile_<32, true, 32>(Flash_fwd_params &params) {
    run_mha_fwd_tile_impl<32, 32, 2, true>(params);
}
