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
                     ] + native_d32_sources,
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

