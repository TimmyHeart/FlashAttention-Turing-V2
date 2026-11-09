from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys
from pathlib import Path
import subprocess

this_dir = os.path.dirname(os.path.abspath(__file__))


if os.path.isdir(".git"):
    subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)

nvcc_flags = ["-std=c++17",
              "--expt-relaxed-constexpr",
              "-arch=sm_75",
              "-O3",
              "--use_fast_math",
              "--ptxas-options=-v",
              "-lineinfo"]

if sys.platform == "win32":
    from distutils import ccompiler
    compiler = ccompiler.new_compiler()
    if compiler.compiler_type == "msvc":
        nvcc_flags[0] = "-std=c++17"
        nvcc_flags.extend(["-Xcompiler", "/Zc:preprocessor"])
# Opt-in native D32 kernels (default: V1 zero-pad-to-D64 path).
#   set FLASH_TURING_NATIVE_D32=1   (cmd: set FLASH_TURING_NATIVE_D32=1)
native_d32 = os.environ.get("FLASH_TURING_NATIVE_D32", "0") == "1"

# V2 forward-kernel variants (all opt-in, default = V1 kernel math/schedule). See README_V2.md.
#   set FLASH_TURING_SEPV=1     separate V smem buffer + V staged in registers
#   set FLASH_TURING_FASTSM=1   exp2-domain softmax, no separate S*=scale pass (needs softmax_scale > 0)
v2_macros = []
if os.environ.get("FLASH_TURING_SEPV", "0") == "1":
    v2_macros.append(("FLASH_TURING_SEPV", "1"))
if os.environ.get("FLASH_TURING_FASTSM", "0") == "1":
    v2_macros.append(("FLASH_TURING_FASTSM", "1"))
# V2.3 forward tile variants (opt-in; default = V2/V2.2 kernels only)
#   set FLASH_TURING_SMALLM=1   also compile kBlockM=64 (4 warps) forward kernels
#   set FLASH_TURING_SMALLM=2   also compile kBlockM=32 (2 warps) forward kernels
#   set FLASH_TURING_CAUSAL_REVERSE=0   disable heavy-first causal CTA order (default on when SMALLM is set)
smallm = os.environ.get("FLASH_TURING_SMALLM", "0")
if smallm not in ("0", "1", "2"):
    raise SystemExit("FLASH_TURING_SMALLM must be 0, 1 or 2")
if smallm != "0":
    v2_macros.append(("FLASH_TURING_SMALLM", smallm))
    if os.environ.get("FLASH_TURING_CAUSAL_REVERSE", "1") == "1":
        v2_macros.append(("FLASH_TURING_CAUSAL_REVERSE", "1"))
smallm_sources = []
for _m in ([64] if smallm == "1" else [64, 32] if smallm == "2" else []):
    for _d in (32, 64, 96, 128):
        # D32 tiles need the native-D32 build; the padded fallback launches the D64 tiles.
        if _d == 32 and not native_d32:
            continue
        for _c in ("", "_causal"):
            smallm_sources.append(f"csrc/flash_attn/src/flash_fwd_hdim{_d}_fp16_m{_m}{_c}_sm75.cu")

native_d32_sources = [
    "csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm75.cu",
    "csrc/flash_attn/src/flash_fwd_hdim32_fp16_causal_sm75.cu",
    "csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm75.cu",
    "csrc/flash_attn/src/flash_bwd_hdim32_fp16_causal_sm75.cu",
] if native_d32 else []

setup(
    name="flash_attn_turing",
    ext_modules=[
        CUDAExtension(
            name="flash_attn_turing",
            sources=["csrc/flash_attn/flash_api.cpp",
                     "csrc/flash_attn/src/d32_pad_sm75.cu",
                     "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm75.cu",
                     "csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm75.cu",
                     "csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm75.cu",
                     "csrc/flash_attn/src/flash_fwd_hdim96_fp16_causal_sm75.cu",
                     "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm75.cu",
                     "csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm75.cu",
                     "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm75.cu",
                     "csrc/flash_attn/src/flash_bwd_hdim64_fp16_causal_sm75.cu",
                     "csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm75.cu",
                     "csrc/flash_attn/src/flash_bwd_hdim96_fp16_causal_sm75.cu",
                     "csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm75.cu",
                     "csrc/flash_attn/src/flash_bwd_hdim128_fp16_causal_sm75.cu"
                     ] + native_d32_sources + smallm_sources,
            define_macros=([("FLASH_TURING_NATIVE_D32", "1")] if native_d32 else []) + v2_macros,
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_attn",
                Path(this_dir) / "csrc" / "flash_attn" / "src",
                Path(this_dir) / "csrc" / "cutlass" / "include",
                Path(this_dir) / "csrc" / "cutlass" / "tools/util/include"
                ],
            extra_compile_args={'nvcc': nvcc_flags}
        )
    ],
    install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension}
)

