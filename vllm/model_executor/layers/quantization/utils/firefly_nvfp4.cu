// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// firefly NVFP4: E2M1 (4-bit FP4) 权重 → int8 转换, 喂 cutlass_scaled_mm int8 GEMM。
//
// 与 firefly.cu (int4→int8) 同构, 但 NVFP4 是简单 uint8 行主序 (2 nibble/byte),
// 无 Marlin 布局要解析。每 block 负责一行 n (256 线程), 两遍: amax + 量化。
//
// NVFP4 张量:
//   weight:      uint8  [N, K/2]   每字节 = 2 个 E2M1 (低 4bit = k 偶, 高 4bit = k 奇)
//   group_scale: fp8_e4m3 [N, K/16]  每 16 个元素 1 个 group scale
//   global_scale: fp32  scalar      全局 scale
//
// E2M1 解码 (纯算术, 无 LUT):
//   sign = bit3, mag = bits[2:0]
//   mag<2: val = mag * 0.5 (subnormal: 0→0, 1→0.5)
//   mag>=2: val = (1.0 + 0.5*(mag&1)) * 2^((mag>>1)-1)
//   值域: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

namespace {

// E2M1 4-bit → float, 纯算术 (无 LUT, 无分支发散)。
__device__ __forceinline__ float e2m1_decode(uint8_t nibble) {
  const int sign = (nibble >> 3) & 1;
  const int mag = nibble & 7;
  float val;
  if (mag < 2) {
    // subnormal: mag=0→0.0, mag=1→0.5
    val = static_cast<float>(mag) * 0.5f;
  } else {
    // normal: (1 + 0.5*(mag&1)) * 2^((mag>>1)-1)
    // mag=2→1.0, 3→1.5, 4→2.0, 5→3.0, 6→4.0, 7→6.0
    val = (1.0f + 0.5f * static_cast<float>(mag & 1)) *
          exp2f(static_cast<float>((mag >> 1) - 1));
  }
  return sign ? -val : val;
}

// fp8_e4m3 字节 → float (经 __half 中转, CUDA 无直接 fp8→fp32 转换)。
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t byte) {
  return __half2float(__nv_fp8_e4m3(byte).__half());
}

// 两遍 kernel: pass1 求 per-row amax (得 c_n), pass2 量化到 int8。
// 每 block 一行 n, 256 线程。weight 是 uint8 [N, K/2], 每线程 stride 处理
// 2 个连续元素 (一个 byte 的 2 个 nibble)。global_scale 是设备指针 (读 [0]),
// 避免 host 标量带来的 D2H sync。
__global__ void nvfp4_to_int8_kernel(
    const uint8_t* __restrict__ weight,      // [N, K/2]
    const uint8_t* __restrict__ group_scale, // [N, K/16] (fp8_e4m3fn bytes)
    const float* __restrict__ global_scale,  // [1] 设备全局 scale
    int8_t* __restrict__ out_int8,          // [N, K]
    float* __restrict__ c_n,                // [N]
    int N, int K) {
  const int n = blockIdx.x;
  if (n >= N) return;
  const int tid = threadIdx.x;
  const int half_k = K / 2;
  const int scale_cols = K / 16;
  const float gscale = global_scale[0];

  __shared__ float sh_warp[32];
  __shared__ float s_max;

  // Pass 1: per-row amax |w_deq|
  // w_deq[n,k] = e2m1_decode(nibble) * group_scale[n, k/16] * global_scale
  float local_max = 0.f;
  for (int k = tid * 2; k < K; k += blockDim.x * 2) {
    const uint8_t byte = weight[n * half_k + k / 2];
    const float gs = fp8_e4m3_to_float(
        group_scale[n * scale_cols + k / 16]);
    const float v0 = e2m1_decode(byte & 0x0F) * gs * gscale;
    const float v1 = e2m1_decode((byte >> 4) & 0x0F) * gs * gscale;
    local_max = fmaxf(local_max, fmaxf(fabsf(v0), fabsf(v1)));
  }

  // warp reduce
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    local_max = fmaxf(local_max, __shfl_xor_sync(~0u, local_max, off));
  if ((tid & 31) == 0) sh_warp[tid >> 5] = local_max;
  __syncthreads();
  const int num_warps = (blockDim.x + 31) >> 5;
  if (tid == 0) {
    float v = sh_warp[0];
    for (int i = 1; i < num_warps; i++) v = fmaxf(v, sh_warp[i]);
    s_max = v;
  }
  __syncthreads();
  const float c = s_max / 127.0f;
  if (tid == 0) c_n[n] = c;

  // Pass 2: 量化到 int8
  for (int k = tid * 2; k < K; k += blockDim.x * 2) {
    const uint8_t byte = weight[n * half_k + k / 2];
    const float gs = fp8_e4m3_to_float(
        group_scale[n * scale_cols + k / 16]);
    const float v0 = e2m1_decode(byte & 0x0F) * gs * gscale;
    const float v1 = e2m1_decode((byte >> 4) & 0x0F) * gs * gscale;
    int q0 = (c > 0.f) ? __float2int_rn(v0 / c) : 0;
    int q1 = (c > 0.f) ? __float2int_rn(v1 / c) : 0;
    if (q0 > 127) q0 = 127;
    if (q0 < -127) q0 = -127;
    if (q1 > 127) q1 = 127;
    if (q1 < -127) q1 = -127;
    out_int8[n * K + k] = static_cast<int8_t>(q0);
    out_int8[n * K + k + 1] = static_cast<int8_t>(q1);
  }
}

// 单遍 kernel: c_n 已在 load 时预计算, 只跑量化 pass (免 amax, 乘倒数省除法)。
__global__ void nvfp4_to_int8_cached_kernel(
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ group_scale,
    const float* __restrict__ global_scale,  // [1] 设备全局 scale
    int8_t* __restrict__ out_int8,
    const float* __restrict__ c_n,
    int N, int K) {
  const int n = blockIdx.x;
  if (n >= N) return;
  const int tid = threadIdx.x;
  const int half_k = K / 2;
  const int scale_cols = K / 16;
  const float c = c_n[n];
  const float r = (c > 0.f) ? 1.0f / c : 0.f;
  const float gscale = global_scale[0];

  for (int k = tid * 2; k < K; k += blockDim.x * 2) {
    const uint8_t byte = weight[n * half_k + k / 2];
    const float gs = fp8_e4m3_to_float(
        group_scale[n * scale_cols + k / 16]);
    const float v0 = e2m1_decode(byte & 0x0F) * gs * gscale;
    const float v1 = e2m1_decode((byte >> 4) & 0x0F) * gs * gscale;
    int q0 = __float2int_rn(v0 * r);
    int q1 = __float2int_rn(v1 * r);
    if (q0 > 127) q0 = 127;
    if (q0 < -127) q0 = -127;
    if (q1 > 127) q1 = 127;
    if (q1 < -127) q1 = -127;
    out_int8[n * K + k] = static_cast<int8_t>(q0);
    out_int8[n * K + k + 1] = static_cast<int8_t>(q1);
  }
}

}  // namespace

// 两遍: weight + group_scale + global_scale → (out_int8 [N,K], c_n [N])
// global_scale 传 tensor (fp32, numel>=1), kernel 读其 [0] (设备指针, 免 D2H)。
void nvfp4_to_int8(at::Tensor weight, at::Tensor group_scale,
                   at::Tensor global_scale, at::Tensor out_int8,
                   at::Tensor c_n, int64_t N, int64_t K) {
  const dim3 grid(static_cast<unsigned>(N));
  constexpr int threads = 256;
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  nvfp4_to_int8_kernel<<<grid, threads, 0, stream>>>(
      static_cast<const uint8_t*>(weight.data_ptr()),
      static_cast<const uint8_t*>(group_scale.data_ptr()),
      static_cast<const float*>(global_scale.data_ptr()),
      static_cast<int8_t*>(out_int8.data_ptr()),
      static_cast<float*>(c_n.data_ptr()),
      static_cast<int>(N), static_cast<int>(K));
}

// 单遍 (cached): c_n 预计算, 只跑量化 (乘倒数, 省除法)。
void nvfp4_to_int8_cached(at::Tensor weight, at::Tensor group_scale,
                          at::Tensor global_scale, at::Tensor out_int8,
                          at::Tensor c_n, int64_t N, int64_t K) {
  const dim3 grid(static_cast<unsigned>(N));
  constexpr int threads = 256;
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  nvfp4_to_int8_cached_kernel<<<grid, threads, 0, stream>>>(
      static_cast<const uint8_t*>(weight.data_ptr()),
      static_cast<const uint8_t*>(group_scale.data_ptr()),
      static_cast<const float*>(global_scale.data_ptr()),
      static_cast<int8_t*>(out_int8.data_ptr()),
      static_cast<const float*>(c_n.data_ptr()),
      static_cast<int>(N), static_cast<int>(K));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  // NOLINT
  m.def("nvfp4_to_int8", &nvfp4_to_int8,
        "NVFP4 E2M1 → int8 (两遍: amax + 量化)");
  m.def("nvfp4_to_int8_cached", &nvfp4_to_int8_cached,
        "NVFP4 E2M1 → int8 (单遍: c_n 预计算, 乘倒数)");
}
