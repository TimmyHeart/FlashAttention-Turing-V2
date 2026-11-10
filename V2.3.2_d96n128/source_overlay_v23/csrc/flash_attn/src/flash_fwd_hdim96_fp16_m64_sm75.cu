#include "flash_fwd_launch_template.h"

#include <cutlass/numeric_types.h>

using half_t = cutlass::half_t;

template<>
void run_mha_fwd_tile_<96, false, 64>(Flash_fwd_params &params) {
    run_mha_fwd_tile_impl<96, 64, 4, false>(params);
}
