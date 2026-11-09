#include <torch/extension.h>
#include <cmath>
#include <atomic>
#include <climits>
#include <c10/cuda/CUDAException.h>
#include "flash.h"
#include "static_switch.h"
#include "d32_pad.h"
#include "flash_fwd_tiles.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

// D32 strategy. Default (V1 behaviour): D32 is zero-padded to a physical D64 and
// run through the D64 kernels. Build with FLASH_TURING_NATIVE_D32=1 (see setup.py)
// to compile the native D32 kernels and skip padding entirely.
#ifdef FLASH_TURING_NATIVE_D32
static constexpr bool kD32UsePadding = false;
#else
static constexpr bool kD32UsePadding = true;
#endif

// Fail closed before dispatch: HEADDIM_SWITCH has no fallback branch.
static inline void check_supported_head_size(const int64_t head_size) {
    TORCH_CHECK(head_size == 32 || head_size == 64 || head_size == 96 || head_size == 128,
                "Unsupported head dimension ", head_size,
                ". Supported dimensions are 32, 64, 96, and 128.");
}

static inline void check_fp16_cuda_contiguous(
        const at::Tensor& tensor, const char* name, const at::Tensor& reference) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same CUDA device as q");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must have dtype float16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static inline void check_fp32_cuda_contiguous(
        const at::Tensor& tensor, const char* name, const at::Tensor& reference) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same CUDA device as q");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must have dtype float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static inline void check_cuda_int32_contiguous(
        const at::Tensor& tensor, const char* name, const at::Tensor& reference) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same CUDA device as q");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must have dtype int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static inline void check_softmax_scale(const float softmax_scale) {
#ifdef FLASH_TURING_FASTSM
    TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0f,
                "FLASH_TURING_FASTSM requires a finite softmax_scale > 0");
#endif
}

// D32 compatibility path: materialize a contiguous D64 tensor with the upper
// 32 channels zero-filled. The original logical softmax scale is kept by the
// caller; only the storage/kernel head dimension is widened.
static inline at::Tensor pad_head_dim_32_to_64(const at::Tensor& input) {
    TORCH_CHECK(input.dim() >= 1 && input.size(-1) == 32,
                "D32 padding helper expects last dimension == 32");
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf,
                "D32 padding helper expects a CUDA fp16 tensor");
    at::Tensor src = input.is_contiguous() ? input : input.contiguous();
    if (reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 != 0) {
        src = src.clone(at::MemoryFormat::Contiguous);
    }
    std::vector<int64_t> padded_sizes(src.sizes().begin(), src.sizes().end());
    padded_sizes.back() = 64;
    at::Tensor padded = torch::empty(padded_sizes, src.options());  // fully written by the kernel below
    c10::cuda::CUDAGuard device_guard(src.device());
    launch_pad_d32_to_d64(src.data_ptr(), padded.data_ptr(), src.numel() / 32,
                          c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return padded;
}

static inline void copy_head_dim_64_to_32(at::Tensor& output,
                                           const at::Tensor& padded) {
    TORCH_CHECK(output.size(-1) == 32 && padded.size(-1) == 64,
                "D32 copy-back expects output D32 and kernel tensor D64");
    TORCH_CHECK(output.is_cuda() && padded.is_cuda() &&
                output.scalar_type() == at::kHalf && padded.scalar_type() == at::kHalf,
                "D32 copy-back expects CUDA fp16 tensors");
    if (output.is_contiguous() && padded.is_contiguous() &&
        reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(padded.data_ptr()) % 16 == 0) {
        c10::cuda::CUDAGuard device_guard(output.device());
        launch_unpad_d64_to_d32(padded.data_ptr(), output.data_ptr(), output.numel() / 32,
                                c10::cuda::getCurrentCUDAStream().stream());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        output.copy_(padded.narrow(padded.dim() - 1, 0, 32));  // generic fallback
    }
}


void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      at::Tensor l,
                      //void *softmax_lse_d,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      float softmax_scale,
                      bool is_causal) {

    // Reset the parameters
    params = {};

    // Set the pointers and strides.
    params.q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
    params.k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
    params.v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
    params.o_ptr = reinterpret_cast<half_t*>(out.data_ptr());


    // Softmax sum
    params.l_ptr = reinterpret_cast<float*>(l.data_ptr());


    // All stride are in elements, not bytes.
    // params.q_row_stride = q.stride(-3);
    // params.k_row_stride = k.stride(-3);
    // params.v_row_stride = v.stride(-3);
    // params.q_head_stride = q.stride(-2);
    // params.k_head_stride = k.stride(-2);
    // params.v_head_stride = v.stride(-2);

    // params.o_row_stride = out.stride(-3);
    // params.o_head_stride = out.stride(-2);

    // if (cu_seqlens_q_d == nullptr) {
    //     params.q_batch_stride = q.stride(0);
    //     params.k_batch_stride = k.stride(0);
    //     params.v_batch_stride = v.stride(0);
    //     params.o_batch_stride = out.stride(0);
    // }



    // Set the dimensions.
    params.b = b;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.d = d;
    params.softmax_scale = softmax_scale;
    params.is_causal = is_causal;
    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
}


void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor l,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      at::Tensor do_o,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      //void *softmax_lse_d,
                      float softmax_scale,
                      bool is_causal) {


    set_params_fprop(params,
                     b,
                     seqlen_q,
                     seqlen_k,
                     h,
                     h_k,
                     d,
                     q, k, v, out, l,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     softmax_scale,
                     is_causal
                     );


    params.do_o_ptr = reinterpret_cast<float*>(do_o.data_ptr());
    params.do_ptr = reinterpret_cast<half_t*>(dout.data_ptr());

    params.dq_ptr = reinterpret_cast<half_t*>(dq.data_ptr());
    params.dk_ptr = reinterpret_cast<half_t*>(dk.data_ptr());
    params.dv_ptr = reinterpret_cast<half_t*>(dv.data_ptr());

    // params.do_row_stride = dout.stride(-3);
    // params.do_head_stride = dout.stride(-2);

    // params.dq_row_stride = dq.stride(-3);
    // params.dk_row_stride = dk.stride(-3);
    // params.dv_row_stride = dv.stride(-3);
    // params.dq_head_stride = dq.stride(-2);
    // params.dk_head_stride = dk.stride(-2);
    // params.dv_head_stride = dv.stride(-2);                

    // if (cu_seqlens_q_d == nullptr) {
    //     params.do_batch_stride = dout.stride(0);
    //     params.dq_batch_stride = dq.stride(0);
    //     params.dk_batch_stride = dk.stride(0);
    //     params.dv_batch_stride = dv.stride(0);
    // }


}


// ---------------------------------------------------------------------------------------------
// V2.3 forward tile selection.
// FLASH_TURING_SMALLM=1 compiles the kBlockM=64 (4 warp) kernels, =2 also kBlockM=32 (2 warp).
// With neither defined, run_mha_fwd launches exactly the V2/V2.2 kBlockM=128 kernels.
// Mode 0 = automatic (cost model below); 128/64/32 force one tile (for A/B and the tuner).
// ---------------------------------------------------------------------------------------------
#ifdef FLASH_TURING_SMALLM
static constexpr int kFwdTileCompiled = FLASH_TURING_SMALLM;
#else
static constexpr int kFwdTileCompiled = 0;
#endif
static std::atomic<int> g_fwd_tile_mode{0};
// Per-row-block fixed cost (in units of query rows) used by the automatic model: the K/V loop
// has latency that does not shrink with kBlockM, so a smaller tile is not proportionally faster.
static std::atomic<int> g_fwd_tile_fixed_cost{48};

static inline int device_sm_count() {
    static std::atomic<int> cache[16];
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 0 && dev < 16) {
        const int cached = cache[dev].load(std::memory_order_relaxed);
        if (cached > 0) return cached;
    }
    int sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    if (sms <= 0) sms = 1;
    if (dev >= 0 && dev < 16) cache[dev].store(sms, std::memory_order_relaxed);
    return sms;
}

static inline int plan_fwd_tile(const int b, const int seqlen_q, const int h) {
    if (kFwdTileCompiled <= 0) return 128;
    const int mode = g_fwd_tile_mode.load(std::memory_order_relaxed);
    if (mode != 0) return mode;
    const long long sms = device_sm_count();
    const long long bh = static_cast<long long>(b) * static_cast<long long>(h);
    const long long fixed = g_fwd_tile_fixed_cost.load(std::memory_order_relaxed);
    int best = 128;
    long long best_cost = LLONG_MAX;
    const int candidates[3] = {128, 64, 32};
    for (int t : candidates) {
        if (t == 64 && kFwdTileCompiled < 1) continue;
        if (t == 32 && kFwdTileCompiled < 2) continue;
        const long long ctas = ((static_cast<long long>(seqlen_q) + t - 1) / t) * bh;
        const long long waves = (ctas + sms - 1) / sms;
        const long long cost = waves * (t + fixed);
        if (cost < best_cost) { best_cost = cost; best = t; }
    }
    return best;
}

static void set_fwd_tile_mode(const int mode) {
    TORCH_CHECK(mode == 0 || mode == 128 || mode == 64 || mode == 32,
                "fwd tile mode must be 0 (auto), 128, 64 or 32");
    TORCH_CHECK(mode != 64 || kFwdTileCompiled >= 1,
                "kBlockM=64 kernels were not compiled (build with FLASH_TURING_SMALLM=1 or 2)");
    TORCH_CHECK(mode != 32 || kFwdTileCompiled >= 2,
                "kBlockM=32 kernels were not compiled (build with FLASH_TURING_SMALLM=2)");
    g_fwd_tile_mode.store(mode, std::memory_order_relaxed);
}

static void set_fwd_tile_fixed_cost(const int cost) {
    TORCH_CHECK(cost >= 0 && cost <= 4096, "fixed cost must be in [0, 4096]");
    g_fwd_tile_fixed_cost.store(cost, std::memory_order_relaxed);
}

// Keep preprocessor conditionals outside HEADDIM_SWITCH / BOOL_SWITCH macro arguments.
// MSVC does not treat #if directives embedded in a macro argument as directives;
// it parses the tokens as part of the lambda body and reports C2065/C2121.
template <int Headdim, bool Is_causal>
static inline void dispatch_mha_fwd_tile(Flash_fwd_params &params, const int tile) {
    (void)tile;  // Also keeps the V2.2-only build warning-clean when SMALLM is not defined.
#ifdef FLASH_TURING_SMALLM
    if (tile == 64) {
        run_mha_fwd_tile_<Headdim, Is_causal, 64>(params);
        return;
    }
#if FLASH_TURING_SMALLM >= 2
    if (tile == 32) {
        run_mha_fwd_tile_<Headdim, Is_causal, 32>(params);
        return;
    }
#endif
#endif
    run_mha_fwd_<Headdim, Is_causal>(params);
}

void run_mha_fwd(Flash_fwd_params &params){
    const int tile = plan_fwd_tile(params.b, params.seqlen_q, params.h);
    HEADDIM_SWITCH(params.d, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            dispatch_mha_fwd_tile<kHeadDim, Is_causal>(params, tile);
        });
    });
}

void run_mha_bwd(Flash_bwd_params &params){
    HEADDIM_SWITCH(params.d, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_bwd_<kHeadDim, Is_causal>(params);
        });
    });
}


std::vector<at::Tensor>
mha_fwd(at::Tensor q,
        at::Tensor k,
        at::Tensor v,
//             int batch_size,
//             int seq_len,
//             int num_heads,
//             int head_dim,
        const float softmax_scale,
        bool is_causal)
{
    auto device = q.device();

    const auto sizes = q.sizes();

    int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    int head_size = sizes[3];

    int seqlen_k = k.size(1);
    int num_heads_k = k.size(2);

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be rank-4 tensors");
    TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size, "k/v batch size must match q");
    TORCH_CHECK(v.size(1) == seqlen_k, "k and v seqlen_k must match");
    TORCH_CHECK(v.size(2) == num_heads_k, "k and v num_heads must match");
    TORCH_CHECK(k.size(3) == head_size && v.size(3) == head_size, "q/k/v head_dim must match");
    TORCH_CHECK(num_heads % num_heads_k == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");
    check_supported_head_size(head_size);
    check_fp16_cuda_contiguous(q, "q", q);
    check_fp16_cuda_contiguous(k, "k", q);
    check_fp16_cuda_contiguous(v, "v", q);
    c10::cuda::CUDAGuard device_guard(q.device());
    check_softmax_scale(softmax_scale);

    const bool use_d32_padding = (kD32UsePadding && head_size == 32);
    const int kernel_head_size = use_d32_padding ? 64 : head_size;
    at::Tensor q_kernel = use_d32_padding ? pad_head_dim_32_to_64(q) : q;
    at::Tensor k_kernel = use_d32_padding ? pad_head_dim_32_to_64(k) : k;
    at::Tensor v_kernel = use_d32_padding ? pad_head_dim_32_to_64(v) : v;

    // V2: o / l are fully written by the kernel except when whole query rows never run:
    // causal with seqlen_q > seqlen_k (blocks that exit early), or empty sequences.
    // Only then do we pay for the zero-fill memsets (2 fewer GPU ops per call otherwise).
    const bool fwd_needs_zero_fill = (seqlen_q == 0 || seqlen_k == 0 ||
                                      (is_causal && seqlen_q > seqlen_k));
    auto fwd_alloc = [&](at::IntArrayRef shape, at::ScalarType dt) {
        return fwd_needs_zero_fill
            ? torch::zeros(shape, q.options().dtype(dt))
            : torch::empty(shape, q.options().dtype(dt));
    };

    at::Tensor o = use_d32_padding
        ? torch::empty(q.sizes(), q.options().dtype(torch::kFloat16))  // fully overwritten by copy-back
        : fwd_alloc(q.sizes(), torch::kFloat16);
    at::Tensor o_kernel = use_d32_padding
        ? fwd_alloc(q_kernel.sizes(), torch::kFloat16)
        : o;

    std::vector<int64_t> size = {batch_size, num_heads, seqlen_q};
    at::Tensor l = fwd_alloc(size, torch::kFloat32);

    TORCH_CHECK(o.is_cuda(), "Tensor o is not on CUDA");

//    half_t* q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
//    half_t* k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
//    half_t* v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
//    half_t* o_ptr = reinterpret_cast<half_t*>(o.data_ptr());
//
//    float* l_ptr = reinterpret_cast<float*>(l.data_ptr());

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q,
                     seqlen_k,
                     num_heads,
                     num_heads_k,
                     kernel_head_size,
                     q_kernel, k_kernel, v_kernel, o_kernel, l,
                     nullptr,
                     nullptr,
                     softmax_scale,
                     is_causal
                     );

    // std::cout << "Q ptr: " << q.data_ptr() << "\n";
    // std::cout << "K ptr: " << k.data_ptr() << "\n";
    // std::cout << "V ptr: " << v.data_ptr() << "\n";
    // std::cout << "O ptr: " << o.data_ptr() << "\n";

    run_mha_fwd(params);

    if (use_d32_padding) {
        copy_head_dim_64_to_32(o, o_kernel);
    }

    return {o, l};

}



// V2.2: inference-only Forward into caller-owned buffers.
// The original V2 mha_fwd remains unchanged; this additive entry point avoids
// allocating output and LSE on every call and provides stable pointers for CUDA Graphs.
void mha_fwd_into(at::Tensor q,
                  at::Tensor k,
                  at::Tensor v,
                  at::Tensor o,
                  at::Tensor l,
                  const float softmax_scale,
                  bool is_causal)
{
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, v must be rank-4 tensors");

    const int batch_size = q.size(0);
    const int seqlen_q = q.size(1);
    const int num_heads = q.size(2);
    const int head_size = q.size(3);
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);

    TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size,
                "k/v batch size must match q");
    TORCH_CHECK(v.size(1) == seqlen_k, "k and v seqlen_k must match");
    TORCH_CHECK(v.size(2) == num_heads_k, "k and v num_heads must match");
    TORCH_CHECK(k.size(3) == head_size && v.size(3) == head_size,
                "q/k/v head_dim must match");
    TORCH_CHECK(num_heads > 0 && num_heads_k > 0 && num_heads % num_heads_k == 0,
                "num_heads_q and num_heads_k must be positive and num_heads_q divisible by num_heads_k");
    check_supported_head_size(head_size);
    TORCH_CHECK(!(kD32UsePadding && head_size == 32),
                "fwd_into requires FLASH_TURING_NATIVE_D32=1 for head dimension 32");

    // This entry point is deliberately strict: no hidden contiguous() copies and no autograd.
    check_fp16_cuda_contiguous(q, "q", q);
    check_fp16_cuda_contiguous(k, "k", q);
    check_fp16_cuda_contiguous(v, "v", q);
    check_fp16_cuda_contiguous(o, "out", q);
    check_fp32_cuda_contiguous(l, "lse", q);
    TORCH_CHECK(o.sizes() == q.sizes(), "out must have the same shape as q");
    TORCH_CHECK(l.dim() == 3 && l.size(0) == batch_size &&
                l.size(1) == num_heads && l.size(2) == seqlen_q,
                "lse must have shape [batch, nheads, seqlen_q]");
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad() &&
                !o.requires_grad() && !l.requires_grad(),
                "fwd_into is inference-only and does not support tensors requiring gradients");

    // Reject any shared storage, not just identical data pointers. The output and
    // LSE are independently written by the kernel and must not overlap inputs/each other.
    TORCH_CHECK(!o.is_alias_of(q) && !o.is_alias_of(k) && !o.is_alias_of(v),
                "out must not share storage with q/k/v");
    TORCH_CHECK(!l.is_alias_of(q) && !l.is_alias_of(k) && !l.is_alias_of(v) &&
                !l.is_alias_of(o),
                "lse must not share storage with q/k/v/out");

    c10::cuda::CUDAGuard device_guard(q.device());
    check_softmax_scale(softmax_scale);

    // Match V2's initialized-output behavior for empty inputs and causal Q>K.
    // Some causal query rows are not visited by the attention grid in that case.
    const bool fwd_needs_zero_fill = (seqlen_q == 0 || seqlen_k == 0 ||
                                      (is_causal && seqlen_q > seqlen_k));
    if (fwd_needs_zero_fill) {
        o.zero_();
        l.zero_();
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q,
                     seqlen_k,
                     num_heads,
                     num_heads_k,
                     head_size,
                     q, k, v, o, l,
                     nullptr,
                     nullptr,
                     softmax_scale,
                     is_causal);
    run_mha_fwd(params);
}



std::vector<at::Tensor>
mha_bwd(at::Tensor q,
        at::Tensor k,
        at::Tensor v,
        at::Tensor out,
        at::Tensor l,
        at::Tensor dout,
//        int batch_size,
//        int seq_len,
//        int num_heads,
//        int head_dim,
        const float softmax_scale,
        bool is_causal)
{

    const auto sizes = q.sizes();

    int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    int head_size = sizes[3];

    int seqlen_k = k.size(1);
    int num_heads_k = k.size(2);

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be rank-4 tensors");
    TORCH_CHECK(out.dim() == 4 && dout.dim() == 4, "out and dout must be rank-4 tensors");
    TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size, "k/v batch size must match q");
    TORCH_CHECK(v.size(1) == seqlen_k, "k and v seqlen_k must match");
    TORCH_CHECK(v.size(2) == num_heads_k, "k and v num_heads must match");
    TORCH_CHECK(k.size(3) == head_size && v.size(3) == head_size, "q/k/v head_dim must match");
    TORCH_CHECK(out.sizes() == q.sizes() && dout.sizes() == q.sizes(), "out and dout must match q shape");
    TORCH_CHECK(num_heads % num_heads_k == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");
    check_supported_head_size(head_size);
    check_fp16_cuda_contiguous(q, "q", q);
    check_fp16_cuda_contiguous(k, "k", q);
    check_fp16_cuda_contiguous(v, "v", q);
    check_fp16_cuda_contiguous(out, "out", q);
    check_fp16_cuda_contiguous(dout, "dout", q);
    check_fp32_cuda_contiguous(l, "lse", q);
    c10::cuda::CUDAGuard device_guard(q.device());
    check_softmax_scale(softmax_scale);

    const bool use_d32_padding = (kD32UsePadding && head_size == 32);
    const int kernel_head_size = use_d32_padding ? 64 : head_size;
    at::Tensor q_kernel = use_d32_padding ? pad_head_dim_32_to_64(q) : q;
    at::Tensor k_kernel = use_d32_padding ? pad_head_dim_32_to_64(k) : k;
    at::Tensor v_kernel = use_d32_padding ? pad_head_dim_32_to_64(v) : v;
    at::Tensor out_kernel = use_d32_padding ? pad_head_dim_32_to_64(out) : out;
    at::Tensor dout_kernel = use_d32_padding ? pad_head_dim_32_to_64(dout) : dout;

    at::Tensor dq = use_d32_padding
        ? torch::empty(q.sizes(), q.options().dtype(torch::kFloat16))  // fully overwritten by copy-back
        : torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));
    at::Tensor dk = use_d32_padding
        ? torch::empty(k.sizes(), k.options().dtype(torch::kFloat16))  // fully overwritten by copy-back
        : torch::zeros(k.sizes(), k.options().dtype(torch::kFloat16));
    at::Tensor dv = use_d32_padding
        ? torch::empty(v.sizes(), v.options().dtype(torch::kFloat16))  // fully overwritten by copy-back
        : torch::zeros(v.sizes(), v.options().dtype(torch::kFloat16));

    at::Tensor dq_kernel = use_d32_padding
        ? torch::zeros(q_kernel.sizes(), q.options().dtype(torch::kFloat16))
        : dq;
    at::Tensor dk_kernel = use_d32_padding
        ? torch::zeros(k_kernel.sizes(), k.options().dtype(torch::kFloat16))
        : dk;
    at::Tensor dv_kernel = use_d32_padding
        ? torch::zeros(v_kernel.sizes(), v.options().dtype(torch::kFloat16))
        : dv;

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads != num_heads_k) {
        dk_expanded = torch::zeros({batch_size, seqlen_k, num_heads, kernel_head_size}, k.options().dtype(torch::kFloat16));
        dv_expanded = torch::zeros({batch_size, seqlen_k, num_heads, kernel_head_size}, v.options().dtype(torch::kFloat16));
    } else {
        dk_expanded = dk_kernel;
        dv_expanded = dv_kernel;
    }

    at::Tensor do_o = torch::zeros(l.sizes(), l.options());

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                    seqlen_q,
                    seqlen_k,
                    num_heads,
                    num_heads_k,
                    kernel_head_size,
                    q_kernel,
                    k_kernel,
                    v_kernel,
                    out_kernel,
                    l,
                    dout_kernel,
                    dq_kernel,
                    dk_expanded,
                    dv_expanded,
                    do_o,
                    nullptr,
                    nullptr,
                    softmax_scale,
                    is_causal);

    run_mha_bwd(params);

    if (num_heads != num_heads_k) {
        torch::sum_out(
            dk_kernel,
            torch::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, kernel_head_size}),
            {3}
        );
        torch::sum_out(
            dv_kernel,
            torch::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, kernel_head_size}),
            {3}
        );
    }


    if (use_d32_padding) {
        copy_head_dim_64_to_32(dq, dq_kernel);
        copy_head_dim_64_to_32(dk, dk_kernel);
        copy_head_dim_64_to_32(dv, dv_kernel);
    }

    return {dq, dk, dv};

}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor q,
               at::Tensor k,
               at::Tensor v,
               at::Tensor &cu_seqlens_q,
               at::Tensor &cu_seqlens_k,
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float softmax_scale,
               bool is_causal)
{
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, v must be rank-3 packed tensors");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens_q/cu_seqlens_k must be CUDA tensors");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 && cu_seqlens_k.scalar_type() == torch::kInt32,
                "cu_seqlens_q/cu_seqlens_k must be int32 tensors");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "cu_seqlens_q/cu_seqlens_k must be contiguous");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens_q/cu_seqlens_k must be rank-1");
    TORCH_CHECK(cu_seqlens_q.numel() >= 2 && cu_seqlens_k.numel() >= 2,
                "cu_seqlens_q/cu_seqlens_k must have at least 2 elements");
    const int batch_size = cu_seqlens_q.numel() - 1;
    TORCH_CHECK(cu_seqlens_k.numel() == batch_size + 1,
                "cu_seqlens_k must have shape [batch_size + 1] with cumulative offsets");
    TORCH_CHECK(k.size(0) == v.size(0), "k and v total tokens must match");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v num_heads must match");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v head_dim must match");
    TORCH_CHECK(q.size(2) == k.size(2), "q/k/v head_dim must match");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    const int num_heads = q.size(1);
    const int num_heads_k = k.size(1);
    const int head_size = q.size(2);
    check_supported_head_size(head_size);
    check_fp16_cuda_contiguous(q, "q", q);
    check_fp16_cuda_contiguous(k, "k", q);
    check_fp16_cuda_contiguous(v, "v", q);
    check_cuda_int32_contiguous(cu_seqlens_q, "cu_seqlens_q", q);
    check_cuda_int32_contiguous(cu_seqlens_k, "cu_seqlens_k", q);
    c10::cuda::CUDAGuard device_guard(q.device());
    check_softmax_scale(softmax_scale);

    const bool use_d32_padding = (kD32UsePadding && head_size == 32);
    const int kernel_head_size = use_d32_padding ? 64 : head_size;
    at::Tensor q_kernel = use_d32_padding ? pad_head_dim_32_to_64(q) : q;
    at::Tensor k_kernel = use_d32_padding ? pad_head_dim_32_to_64(k) : k;
    at::Tensor v_kernel = use_d32_padding ? pad_head_dim_32_to_64(v) : v;

    at::Tensor out = torch::zeros_like(q);
    at::Tensor out_kernel = use_d32_padding
        ? torch::zeros(q_kernel.sizes(), q.options().dtype(torch::kFloat16))
        : out;
    at::Tensor l = torch::zeros({batch_size, num_heads, max_seqlen_q}, q.options().dtype(torch::kFloat32));


    // at::Tensor o = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));

    // std::vector<int64_t> size = {batch_size, num_heads, seqlen_q};
    // at::Tensor l = torch::zeros(size, q.options().dtype(torch::kFloat32).device(device));

    // TORCH_CHECK(o.is_cuda(), "Tensor o is not on CUDA");


    Flash_fwd_params params;
    set_params_fprop(
        params,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        kernel_head_size,
        q_kernel, k_kernel, v_kernel, out_kernel, l,
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        softmax_scale,
        is_causal
    );

    run_mha_fwd(params);

    if (use_d32_padding) {
        copy_head_dim_64_to_32(out, out_kernel);
    }

    return {out, l};
}

std::vector<at::Tensor>
mha_varlen_bwd(at::Tensor q,
               at::Tensor k,
               at::Tensor v,
               at::Tensor out,
               at::Tensor l,
               at::Tensor dout,
               at::Tensor cu_seqlens_q,
               at::Tensor cu_seqlens_k,
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float softmax_scale,
               bool is_causal)
{
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, v must be rank-3 packed tensors");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens_q/cu_seqlens_k must be CUDA tensors");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 && cu_seqlens_k.scalar_type() == torch::kInt32,
                "cu_seqlens_q/cu_seqlens_k must be int32 tensors");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "cu_seqlens_q/cu_seqlens_k must be contiguous");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens_q/cu_seqlens_k must be rank-1");
    TORCH_CHECK(cu_seqlens_q.numel() >= 2 && cu_seqlens_k.numel() >= 2,
                "cu_seqlens_q/cu_seqlens_k must have at least 2 elements");
    const int64_t batch_size = cu_seqlens_q.numel() - 1;
    TORCH_CHECK(cu_seqlens_k.numel() == batch_size + 1,
                "cu_seqlens_k must have shape [batch_size + 1] with cumulative offsets");
    TORCH_CHECK(k.size(0) == v.size(0), "k and v total tokens must match");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v num_heads must match");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v head_dim must match");
    TORCH_CHECK(q.size(2) == k.size(2), "q/k/v head_dim must match");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
    TORCH_CHECK(dout.sizes() == q.sizes(), "dout must match q shape");
    TORCH_CHECK(l.dim() == 3, "l must be rank-3 for varlen_bwd");

    const int64_t num_heads = q.size(1);
    const int64_t num_heads_k = k.size(1);
    const int64_t head_size = q.size(2);
    const int64_t total_k = k.size(0);
    TORCH_CHECK(l.size(0) == batch_size && l.size(1) == num_heads && l.size(2) == max_seqlen_q,
                "l must have shape [batch_size, nheads_q, max_seqlen_q]");
    check_supported_head_size(head_size);
    check_fp16_cuda_contiguous(q, "q", q);
    check_fp16_cuda_contiguous(k, "k", q);
    check_fp16_cuda_contiguous(v, "v", q);
    check_fp16_cuda_contiguous(out, "out", q);
    check_fp16_cuda_contiguous(dout, "dout", q);
    check_fp32_cuda_contiguous(l, "lse", q);
    check_cuda_int32_contiguous(cu_seqlens_q, "cu_seqlens_q", q);
    check_cuda_int32_contiguous(cu_seqlens_k, "cu_seqlens_k", q);
    c10::cuda::CUDAGuard device_guard(q.device());
    check_softmax_scale(softmax_scale);

    const bool use_d32_padding = (kD32UsePadding && head_size == 32);
    const int64_t kernel_head_size = use_d32_padding ? 64 : head_size;
    at::Tensor q_kernel = use_d32_padding ? pad_head_dim_32_to_64(q) : q;
    at::Tensor k_kernel = use_d32_padding ? pad_head_dim_32_to_64(k) : k;
    at::Tensor v_kernel = use_d32_padding ? pad_head_dim_32_to_64(v) : v;
    at::Tensor out_kernel = use_d32_padding ? pad_head_dim_32_to_64(out) : out;
    at::Tensor dout_kernel = use_d32_padding ? pad_head_dim_32_to_64(dout) : dout;

    at::Tensor dq = use_d32_padding ? torch::empty_like(q) : torch::zeros_like(q);
    at::Tensor dk = use_d32_padding ? torch::empty_like(k) : torch::zeros_like(k);
    at::Tensor dv = use_d32_padding ? torch::empty_like(v) : torch::zeros_like(v);
    at::Tensor dq_kernel = use_d32_padding
        ? torch::zeros(q_kernel.sizes(), q.options().dtype(torch::kFloat16))
        : dq;
    at::Tensor dk_kernel = use_d32_padding
        ? torch::zeros(k_kernel.sizes(), k.options().dtype(torch::kFloat16))
        : dk;
    at::Tensor dv_kernel = use_d32_padding
        ? torch::zeros(v_kernel.sizes(), v.options().dtype(torch::kFloat16))
        : dv;

    at::Tensor dk_expanded = dk_kernel;
    at::Tensor dv_expanded = dv_kernel;
    if (num_heads != num_heads_k) {
        dk_expanded = torch::zeros({total_k, num_heads, kernel_head_size}, k.options().dtype(torch::kFloat16));
        dv_expanded = torch::zeros({total_k, num_heads, kernel_head_size}, v.options().dtype(torch::kFloat16));
    }
    at::Tensor do_o = torch::zeros(l.sizes(), l.options());

    Flash_bwd_params params;
    set_params_dgrad(
        params,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        kernel_head_size,
        q_kernel, k_kernel, v_kernel, out_kernel, l, dout_kernel,
        dq_kernel, dk_expanded, dv_expanded, do_o,
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        softmax_scale,
        is_causal
    );

    run_mha_bwd(params);

    if (num_heads != num_heads_k) {
        torch::sum_out(
            dk_kernel,
            torch::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, kernel_head_size}),
            {2}
        );
        torch::sum_out(
            dv_kernel,
            torch::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, kernel_head_size}),
            {2}
        );
    }

    if (use_d32_padding) {
        copy_head_dim_64_to_32(dq, dq_kernel);
        copy_head_dim_64_to_32(dk, dk_kernel);
        copy_head_dim_64_to_32(dv, dv_kernel);
    }

    return {dq, dk, dv};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("fwd_into", &mha_fwd_into, "Inference forward into caller-owned output/LSE buffers");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("varlen_fwd", &mha_varlen_fwd, "Varlen forward pass");
    m.def("varlen_bwd", &mha_varlen_bwd, "Varlen backward pass");
    // V2.3 forward tile control (no effect unless built with FLASH_TURING_SMALLM >= 1)
    m.def("set_fwd_tile_mode", &set_fwd_tile_mode, pybind11::arg("mode"),
          "0 = auto, 128/64/32 = force that kBlockM for subsequent forward launches");
    m.def("get_fwd_tile_mode", []() { return g_fwd_tile_mode.load(std::memory_order_relaxed); });
    m.def("set_fwd_tile_fixed_cost", &set_fwd_tile_fixed_cost, pybind11::arg("cost"),
          "Fixed per-row-block cost (in rows) used by the automatic tile model; default 48");
    m.def("fwd_tile_support", []() { return kFwdTileCompiled; },
          "0: only kBlockM=128; 1: also 64; 2: also 32");
    m.def("plan_fwd_tile", [](int batch, int seqlen_q, int heads) { return plan_fwd_tile(batch, seqlen_q, heads); },
          pybind11::arg("batch"), pybind11::arg("seqlen_q"), pybind11::arg("heads"),
          "kBlockM the forward launch would use for this shape under the current mode");
}
