// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

  // Native D32 kernels are opt-in (build with FLASH_TURING_NATIVE_D32=1, see setup.py).
  // Without the flag, D32 keeps using the V1 zero-pad-to-D64 path in flash_api.cpp
  // and no D32 kernel is referenced.
#ifdef FLASH_TURING_NATIVE_D32
  #define HEADDIM_SWITCH(HEADDIM, ...)        \
    [&] {                                     \
      if (HEADDIM == 32) {                   \
        constexpr static int kHeadDim = 32;  \
        return __VA_ARGS__();                 \
      } else if (HEADDIM == 64) {            \
        constexpr static int kHeadDim = 64;  \
        return __VA_ARGS__();                 \
      } else if (HEADDIM == 96) {            \
        constexpr static int kHeadDim = 96;  \
        return __VA_ARGS__();                 \
      } else if (HEADDIM == 128) {           \
        constexpr static int kHeadDim = 128;   \
        return __VA_ARGS__();                 \
      }                                       \
    }()
#else
  #define HEADDIM_SWITCH(HEADDIM, ...)        \
    [&] {                                     \
      if (HEADDIM == 64) {                   \
        constexpr static int kHeadDim = 64;  \
        return __VA_ARGS__();                 \
      } else if (HEADDIM == 96) {            \
        constexpr static int kHeadDim = 96;  \
        return __VA_ARGS__();                 \
      } else if (HEADDIM == 128) {           \
        constexpr static int kHeadDim = 128;   \
        return __VA_ARGS__();                 \
      }                                       \
    }()
#endif
