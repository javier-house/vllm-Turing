// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// firefly EXL3: 在线 trellis 解码(sm75 安全, 无 cp.async) + int8 转置, 喂 cutlass int8 GEMM。
//
// 设计(对齐 firefly "dequant kernel -> int8 GEMM" 两段式, 复用现有 int8_prefill_linear):
//   1) exl3_decode: EXL3 trellis 位流 -> 原基权重 W_hat = diag(suh)·H128·W̃·H128·diag(svh),
//      输出 fp16 [K, N](K-major, 即 Linear 的 [out=in? 见下])。解码数学移植自 exllamav3
//      mul1 码本(cb=2, __dp4a 值映射) + 128 点 Hadamard 蝶式(warp shuffle), 全 sm75 原生
//      (shf/bfe/dp4a/lop3/shfl/int4 加载), 不含 cp.async/ldmatrix/mma。
//   2) exl3_int8_transpose: fp16 [K, N] -> int8 [N, K] 行主序 + per-channel scale c_n[N]
//      (= 每列 amax/127), 正好是 cutlass_scaled_mm 的 B=w_int8 行主序约定。
//   3) GEMM 复用 firefly.py 的 int8_prefill_linear(x, w_int8, c_n) -> ops.cutlass_scaled_mm。
//
// EXL3 张量布局(dense [N, K], 两维 pad 到 ×128):
//   trellis: int16(=uint16) 张量, 形状 [K/16, N/16, 16*K], 第 3 轴是每 16×16 tile 的位流;
//            按 2D [K/16, N/16*16*K] 视图喂 kernel(K 行=块行, N 行=块列)。
//   suh: fp16 [K](输入通道 scale, 作用在 K 行); svh: fp16 [N](输出通道 scale, 作用在 N 列)。
//   K(=trellis.shape[0]*16) 是 in 维, N(=trellis.shape[1]*16) 是 out 维。
//
// mul1 值映射(码本 cb=2, codebook.cuh decode_3inst): 16-bit 字 w ->
//   x = w * 0x83DCD12D; sum = __dp4a(x, 0x01010101, 0x6400);
//   value = (sum & 0xFFFF 当 half) * 0.00677 - 10.39。纯程序化, 无码本表。
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

// ---- exllamav3 依赖的最小自包含副本(只取 mul1 + 解码/Hadamard 用到的) ----

// util.cuh: half 位型联合体(union), 用于把 dp4a 累加器低位当 half 解释。
union half_uint16 {
  uint16_t as_uint16;
  __half as_half;
  __device__ half_uint16(uint16_t v) : as_uint16(v) {}
  __device__ half_uint16(__half v) : as_half(v) {}
  __device__ half_uint16() : as_uint16(0) {}
};

// util.cuh: half4(CUDA 无内建 half4, exllamav3 自定义 4-half 向量)。
struct half4 {
  __half2 x, y;  // 2×half2 = 4 half(8B); .x/.y 是 half2, 直接喂 __hmul2 等
  __device__ half4()
      : x(__floats2half2_rn(0.f, 0.f)), y(__floats2half2_rn(0.f, 0.f)) {}
  __device__ half4(__half2 xv, __half2 yv) : x(xv), y(yv) {}
};

// ptx.cuh: Vec/FragB(treillis 解码产物, 每 lane 4 个 half)。
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};
using FragB = Vec<__half2, 2>;

// ptx.cuh: 位操作宏(mul1 4-bit 对齐路径用)。
#define FSHF_IMM(dst, lo, hi, imm) \
  asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define BFE16_IMM(dst, src, imm) \
  asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))

// fshift: 条件无关的 64-bit funnel shift(等价 __funnelshift_r, 但 K 任意时安全)。
__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a,
                                           int shift) {
  uint64_t merged = ((uint64_t)a << 32) | (uint64_t)b;
  return (uint32_t)(merged >> shift);
}

// codebook.cuh decode_mul1_product_2: 一次解两个 mul1 16-bit 字。
__device__ __forceinline__ __half2 decode_mul1_product_2(uint32_t x0,
                                                         uint32_t x1) {
  // mul1(cb=2) 值映射第一步: 16-bit word 经 0x83DCD12D 散成 32-bit hash
  // (对照 codebook.cuh decode_3inst_2<2>: x *= 0x83DCD12Du 再 dp4a 求和)。
  // 漏掉这步会让解出权重全错 -> GEMM 输出 garbage。
  x0 *= 0x83DCD12Du;
  x1 *= 0x83DCD12Du;
  const uint32_t acc = 0x6400u;  // 0x6400 -> 1024.0 .. 0x67FF -> 2047.0
  uint32_t sum0 = __dp4a(x0, 0x01010101u, acc);
  uint32_t sum1 = __dp4a(x1, 0x01010101u, acc);
  const __half2 k_inv_h2 =
      __half2half2(__ushort_as_half(0x1eee));  //  0.00677 = 1/147.7
  const __half2 k_bias_h2 =
      __half2half2(__ushort_as_half(0xc931));  // -10.39 = (-1024-510)*k_inv
  half_uint16 h0((uint16_t)sum0);
  half_uint16 h1((uint16_t)sum1);
  return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
}

// exl3_dq.cuh: 各 K 的 8 字位拆 + mul1 值映射。每 lane 出 8 个 half(2 个 FragB)。
// ptr 指向该 tile 的位流 uint32*(packed_size/2 个 uint32), idx = lane*8。

__device__ __forceinline__ void dq_mul1_4bit(const uint32_t* __restrict__ ptr,
                                             int idx, FragB& f0, FragB& f1) {
  uint32_t i0, i1, a, b, s, w0, w1, w2, w3, w4, w5, w6, w7;
  i1 = idx >> 3;
  i0 = (i1 + 31) & 31;
  a = ptr[i0];
  b = ptr[i1];
  FSHF_IMM(s, b, a, 20);
  w7 = b & 0xffff;
  BFE16_IMM(w6, b, 4);
  BFE16_IMM(w5, b, 8);
  BFE16_IMM(w4, b, 12);
  BFE16_IMM(w3, b, 16);
  w2 = s & 0xffff;
  BFE16_IMM(w1, s, 4);
  BFE16_IMM(w0, s, 8);
  f0[0] = decode_mul1_product_2(w0, w1);
  f0[1] = decode_mul1_product_2(w2, w3);
  f1[0] = decode_mul1_product_2(w4, w5);
  f1[1] = decode_mul1_product_2(w6, w7);
}

__device__ __forceinline__ void dq_mul1_2bit(const uint32_t* __restrict__ ptr,
                                             int idx, FragB& f0, FragB& f1) {
  uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
  i1 = idx >> 4;
  i0 = (i1 + 15) & 15;
  a = ptr[i0];
  b = ptr[i1];
  b = fshift(b, a, ((~idx) & 8) << 1);
  w7 = b & 0xffff;
  BFE16_IMM(w6, b, 2);
  BFE16_IMM(w5, b, 4);
  BFE16_IMM(w4, b, 6);
  BFE16_IMM(w3, b, 8);
  BFE16_IMM(w2, b, 10);
  BFE16_IMM(w1, b, 12);
  BFE16_IMM(w0, b, 14);
  f0[0] = decode_mul1_product_2(w0, w1);
  f0[1] = decode_mul1_product_2(w2, w3);
  f1[0] = decode_mul1_product_2(w4, w5);
  f1[1] = decode_mul1_product_2(w6, w7);
}

__device__ __forceinline__ void dq_mul1_1bit(const uint32_t* __restrict__ ptr,
                                             int idx, FragB& f0, FragB& f1) {
  uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
  i1 = idx >> 5;
  i0 = (i1 + 7) & 7;
  a = ptr[i0];
  b = ptr[i1];
  b = fshift(b, a, ((~idx) & 24));
  w7 = b & 0xffff;
  BFE16_IMM(w6, b, 1);
  BFE16_IMM(w5, b, 2);
  BFE16_IMM(w4, b, 3);
  BFE16_IMM(w3, b, 4);
  BFE16_IMM(w2, b, 5);
  BFE16_IMM(w1, b, 6);
  BFE16_IMM(w0, b, 7);
  f0[0] = decode_mul1_product_2(w0, w1);
  f0[1] = decode_mul1_product_2(w2, w3);
  f1[0] = decode_mul1_product_2(w4, w5);
  f1[1] = decode_mul1_product_2(w6, w7);
}

// K=3/5/6/7/8: 通用位拆(每字 16-bit, 用 fshift 跨字对齐)。
template <int K>
__device__ __forceinline__ void dq_mul1_4(const uint32_t* __restrict__ ptr,
                                          int t_offset, FragB& frag) {
  constexpr int words = K * 256 / 32;  // uint32 per 256-weight tile
  int b0 = (t_offset + 257) * K - 16;
  int b1 = b0 + 3 * K;
  int b2 = b1 + 16;
  int i0 = b0 / 32;
  int i2 = (b2 - 1) / 32;
  int s2 = (i2 + 1) * 32 - b2;
  uint32_t a = ptr[i0 % words];
  uint32_t b = ptr[i2 % words];
  uint32_t w3 = fshift(b, a, s2) & 0xffff;
  uint32_t w2 = fshift(b, a, s2 + K) & 0xffff;
  uint32_t w1 = fshift(b, a, s2 + K * 2) & 0xffff;
  uint32_t w0 = fshift(b, a, s2 + K * 3) & 0xffff;
  frag[0] = decode_mul1_product_2(w0, w1);
  frag[1] = decode_mul1_product_2(w2, w3);
}

template <int K>
__device__ __forceinline__ void dq_mul1_2x2(const uint32_t* __restrict__ ptr,
                                            int t_offset, FragB& frag) {
  constexpr int words = K * 256 / 32;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int b0 = (t_offset + 2 * i + 257) * K - 16;
    int b1 = b0 + 1 * K;
    int b2 = b1 + 16;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[i0 % words];
    uint32_t b = ptr[i2 % words];
    uint32_t w1 = fshift(b, a, s2) & 0xffff;
    uint32_t w0 = fshift(b, a, s2 + K) & 0xffff;
    frag[i] = decode_mul1_product_2(w0, w1);
  }
}

template <int K>
__device__ __forceinline__ void dq_dispatch_mul1(
    const uint32_t* __restrict__ ptr, int idx, FragB& f0, FragB& f1) {
  if constexpr (K == 1)
    dq_mul1_1bit(ptr, idx, f0, f1);
  else if constexpr (K == 2)
    dq_mul1_2bit(ptr, idx, f0, f1);
  else if constexpr (K == 3) {
    dq_mul1_4<3>(ptr, idx, f0);
    dq_mul1_4<3>(ptr, idx + 4, f1);
  } else if constexpr (K == 4)
    dq_mul1_4bit(ptr, idx, f0, f1);
  else if constexpr (K == 5) {
    dq_mul1_4<5>(ptr, idx, f0);
    dq_mul1_4<5>(ptr, idx + 4, f1);
  } else if constexpr (K == 6) {
    dq_mul1_4<6>(ptr, idx, f0);
    dq_mul1_4<6>(ptr, idx + 4, f1);
  } else if constexpr (K == 7) {
    dq_mul1_2x2<7>(ptr, idx, f0);
    dq_mul1_2x2<7>(ptr, idx + 4, f1);
  } else if constexpr (K == 8) {
    dq_mul1_4<8>(ptr, idx, f0);
    dq_mul1_4<8>(ptr, idx + 4, f1);
  }
}

// hadamard_inner.cuh: 128 点 Hadamard 的 warp 蝶式(last 一级, 跨 32 lane), half2 版。
__device__ __forceinline__ __half2 shuffle_had_h2x32(__half2 v, int lane_id) {
  for (int i = 1; i < 32; i <<= 1) {
    __half2 pv = __shfl_xor_sync(0xffffffff, v, i);
    uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
    int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
    *vi ^= (sfm & 0x80008000);
    v = __hadd2(v, pv);
  }
  return v;
}

// reconstruct.cu reconstruct_had_tile 移植(mul1 特化): 128×128 tile, 256 线程,
// 解 8×8 个 16×16 子 tile -> 列 Hadamard(4 行 warp 蝶式 + shuffle) -> 行 Hadamard
// (融进 store, 128 点/warp) + suh/svh scale。输出 W_hat fp16 [K, N](K-major)。
// r_scale = 1/sqrt(128), 列+行两次各一次, 合 1/128。
constexpr int RH_THREADS = 256;

template <int K>
__device__ __forceinline__ void exl3_had_tile(
    __half* __restrict__ g_unpacked, const uint16_t* __restrict__ g_packed,
    const __half* __restrict__ suh, const __half* __restrict__ svh,
    int packed_blocks_n) {
  constexpr int packed_size = 16 * K;  // uint16s per 16x16 tile
  constexpr float r_scale = 0.08838834764831845f;

  const int t = threadIdx.x;
  const int lane_id = t & 31;
  const int warp_id = t >> 5;
  const int kb = blockIdx.y;
  const int nb = blockIdx.x;
  const int n = nb * 8;
  const int row_len = gridDim.x * 128;

  __shared__ uint32_t s_packed[8][8][packed_size / 2];
  __shared__ __half2 stile[128 * 64];

  auto tix = [&](int R, int q, int p) {
    return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p;
  };

  const int j_int4 = packed_size / 8;
  for (int u = t; u < 8 * 8 * j_int4; u += RH_THREADS) {
    const int j = u / (8 * j_int4);
    const int r = u % (8 * j_int4);
    const uint16_t* gp =
        g_packed + ((size_t)((kb * 8 + j) * packed_blocks_n + n)) * packed_size;
    ((int4*)s_packed[j])[r] = ((const int4*)gp)[r];
  }
  __syncthreads();

  // 解 8×8 子 tile 到 stile(带 swizzle), 每 warp 负责 8 子 tile 中的一列(wn)。
  const int j_per_iter = 8 * 8 / (RH_THREADS / 32);
  for (int jj = 0; jj < j_per_iter; ++jj) {
    const int j = (warp_id / 8) * (8 / (RH_THREADS / 256)) + jj;
    const int wn = warp_id & 7;
    FragB frag[2];
    dq_dispatch_mul1<K>(s_packed[j][wn], lane_id * 8, frag[0], frag[1]);

    const __half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
    const __half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
    const __half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
    const __half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

    if (!(lane_id & 4)) {
      __half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
      __half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
      __half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
      __half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
      __half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
      __half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
      __half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
      __half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
      const int r0 = j * 16 + (lane_id & 3) * 2;
      const int r1 = r0 + 1;
      const int r2 = r0 + 8;
      const int r3 = r0 + 9;
      const int c0 = lane_id >> 3;
      const int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
      const int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
      stile[tix(r0, q0, p0)] = m0;
      stile[tix(r1, q0, p0)] = m1;
      stile[tix(r2, q0, p0)] = m2;
      stile[tix(r3, q0, p0)] = m3;
      stile[tix(r0, q1, p1)] = m4;
      stile[tix(r1, q1, p1)] = m5;
      stile[tix(r2, q1, p1)] = m6;
      stile[tix(r3, q1, p1)] = m7;
    }
  }
  __syncthreads();

  // 列 Hadamard: 每 warp 负责一列, 4 行/次, 4 元素块蝶式 + 32-lane shuffle。
  const __half2 rs2 = __float2half2_rn(r_scale);
  constexpr int CHUNKS_PW = 32 / (RH_THREADS / 32);
#pragma unroll
  for (int qq = 0; qq < CHUNKS_PW; ++qq) {
    const int q = warp_id * CHUNKS_PW + qq;
    const int qs = q ^ lane_id;
    __half2 a[4], b[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      half4 v =
          *((const half4*)(stile + (lane_id * 4 + i) * 64 + qs * 2));
      a[i] = v.x;
      b[i] = v.y;
    }
#pragma unroll
    for (int x = 0; x < 2; ++x) {
      __half2* v = (x == 0) ? a : b;
      __half2 s0 = __hadd2(v[0], v[1]), d0 = __hsub2(v[0], v[1]);
      __half2 s1 = __hadd2(v[2], v[3]), d1 = __hsub2(v[2], v[3]);
      v[0] = __hmul2(__hadd2(s0, s1), rs2);
      v[1] = __hmul2(__hadd2(d0, d1), rs2);
      v[2] = __hmul2(__hsub2(s0, s1), rs2);
      v[3] = __hmul2(__hsub2(d0, d1), rs2);
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = shuffle_had_h2x32(v[i], lane_id);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      half4 v;
      v.x = a[i];
      v.y = b[i];
      *((half4*)(stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
    }
  }
  __syncthreads();

  // 行 Hadamard 融进 store: 每 warp 4 行, lane l 持最终列 4l..4l+3; 128 点行蝶式
  // (4 元素 + 32-lane shuffle) + suh[k]/svh[n] scale, 直接写 [K,N] 行。
  constexpr int ROWS_PW = 128 / (RH_THREADS / 32);
  const half4 sv4 = ((const half4*)svh)[nb * 32 + lane_id];
#pragma unroll
  for (int rr = 0; rr < ROWS_PW; ++rr) {
    const int R = warp_id * ROWS_PW + rr;
    const int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
    const __half2 v01 = stile[base];
    const __half2 v23 = stile[base + 1];
    float v0 = __low2float(v01), v1 = __high2float(v01);
    float v2 = __low2float(v23), v3 = __high2float(v23);
    float s0 = v0 + v1, d0 = v0 - v1;
    float s1 = v2 + v3, d1 = v2 - v3;
    __half2 h01 =
        __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
    __half2 h23 =
        __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
    h01 = shuffle_had_h2x32(h01, lane_id);
    h23 = shuffle_had_h2x32(h23, lane_id);
    const __half2 su2 = __half2half2(suh[kb * 128 + R]);
    half4 v;
    v.x = __hmul2(__hmul2(h01, su2), sv4.x);
    v.y = __hmul2(__hmul2(h23, su2), sv4.y);
    *((half4*)(g_unpacked + (size_t)(kb * 128 + R) * row_len + nb * 128 +
                 lane_id * 4)) = v;
  }
}

template <int K>
__global__ __launch_bounds__(RH_THREADS) void exl3_decode_kernel(
    __half* __restrict__ g_unpacked, const uint16_t* __restrict__ g_packed,
    const __half* __restrict__ suh, const __half* __restrict__ svh,
    int packed_blocks_n) {
  exl3_had_tile<K>(g_unpacked, g_packed, suh, svh, packed_blocks_n);
}

// ---- int8 转置: fp16 [K, N](K-major) -> int8 [N, K] 行主序 + c_n[N]=列 amax/127 ----
// 每 block 负责 128 个 output channel(列), 沿完整 K 求 per-channel amax 再量化转置。
// 关键: c_n[n] 必须用整列(K 维全部)的 amax —— 若按 kb tile 分块各自 amax 会互相覆盖
// (后写赢), 导致 c_n 偏小 -> round(W/c) 大量 overflow clamp 到 ±127 -> 每层 GEMM 误差
// 逐层累积(28 层 cos 掉到 ~0.06) -> 输出乱码。故 grid 只按 N 切(N/128), 两遍走 K。
__global__ __launch_bounds__(RH_THREADS) void exl3_int8_transpose_kernel(
    const __half* __restrict__ w_hat,  // [K, N] fp16 K-major
    int8_t* __restrict__ out_int8,  // [N, K] 行主序
    float* __restrict__ c_n,  // [N]
    int K, int N) {
  const int nb = blockIdx.x;  // 每 block 128 列
  const int t = threadIdx.x;
  __shared__ __half s_tile[128 * 128];
  __shared__ float s_c[128];

  // Pass 1: per-channel(列) amax 沿完整 K, 分 128 k-chunk 经 smem 累积
  // (前 128 线程各占一列, 每 chunk 更新 max)。W_hat [K,N] K-major, 每行 128 列连续。
  float m = 0.f;
  for (int kch = 0; kch < K / 128; ++kch) {
    for (int u = t; u < 128 * 128; u += RH_THREADS) {
      const int k = u >> 7, n = u & 127;
      s_tile[u] = w_hat[(size_t)(kch * 128 + k) * N + nb * 128 + n];
    }
    __syncthreads();
    if (t < 128) {
#pragma unroll 4
      for (int r = 0; r < 128; ++r)
        m = fmaxf(m, fabsf(__half2float(s_tile[r * 128 + t])));
    }
    __syncthreads();
  }
  if (t < 128) {
    s_c[t] = m / 127.0f;        // per-channel scale, 喂量化
    c_n[nb * 128 + t] = s_c[t];  // 全局 per-channel scale (= GEMM scale_b)
  }
  __syncthreads();

  // Pass 2: 量化 + 转置写 int8 [N, K]: out[n*K + k] (分 128 k-chunk)
  for (int kch = 0; kch < K / 128; ++kch) {
    for (int u = t; u < 128 * 128; u += RH_THREADS) {
      const int k = u >> 7, n = u & 127;
      s_tile[u] = w_hat[(size_t)(kch * 128 + k) * N + nb * 128 + n];
    }
    __syncthreads();
    for (int u = t; u < 128 * 128; u += RH_THREADS) {
      const int k = u >> 7, n = u & 127;
      const int ng = nb * 128 + n;
      const float c = s_c[n];
      const float w = __half2float(s_tile[u]);
      int v = (c > 0.f) ? __float2int_rn(w / c) : 0;
      if (v > 127) v = 127;
      if (v < -127) v = -127;
      out_int8[(size_t)ng * K + kch * 128 + k] = (int8_t)v;
    }
    __syncthreads();
  }
}

}  // namespace

// ---- host launcher + binding(镜像 firefly.cu) ----

// 按 K 选择解码 kernel 实例(K=trellis.shape[0]*16 的每 16 行 bits, 即 bpw)。
// exl3_decode: trellis [K/16, N/16, 16*K] int16 -> W_hat fp16 [K, N](K-major)。
// bits(=bpw) 1..8 直接 switch 实例化对应 kernel(不经过泛型 dispatch, 避免
// f<1>() 被解析为 f<1 比较)。
void exl3_decode(at::Tensor packed, at::Tensor suh, at::Tensor svh,
                 at::Tensor out, int64_t bits) {
  TORCH_CHECK(packed.dim() == 3, "packed must be [K/16, N/16, 16*K]");
  const int Kblk = packed.size(0);
  const int Nblk16 = packed.size(1);
  const int K = Kblk * 16;
  const int N = Nblk16 * 16;
  TORCH_CHECK(K % 128 == 0, "EXL3 K(=in) must be multiple of 128, got ", K);
  TORCH_CHECK(N % 128 == 0, "EXL3 N(=out) must be multiple of 128, got ", N);
  TORCH_CHECK(packed.size(2) == 16 * bits, "packed last dim must be 16*bits");
  TORCH_CHECK(out.size(0) == K && out.size(1) == N, "out must be [K, N]");
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const uint16_t* packed_ptr =
      static_cast<const uint16_t*>(packed.data_ptr());
  const __half* suh_ptr = static_cast<const __half*>(suh.data_ptr());
  const __half* svh_ptr = static_cast<const __half*>(svh.data_ptr());
  __half* out_ptr = static_cast<__half*>(out.data_ptr());
  const dim3 grid(N / 128, K / 128);
#define EXL3_LAUNCH(bits_v)                                        \
  exl3_decode_kernel<bits_v><<<grid, RH_THREADS, 0, stream>>>(     \
      out_ptr, packed_ptr, suh_ptr, svh_ptr, Nblk16)
  switch ((int)bits) {
    case 1: EXL3_LAUNCH(1); break;
    case 2: EXL3_LAUNCH(2); break;
    case 3: EXL3_LAUNCH(3); break;
    case 4: EXL3_LAUNCH(4); break;
    case 5: EXL3_LAUNCH(5); break;
    case 6: EXL3_LAUNCH(6); break;
    case 7: EXL3_LAUNCH(7); break;
    case 8: EXL3_LAUNCH(8); break;
    default: TORCH_CHECK(false, "unsupported EXL3 bits: ", bits);
  }
#undef EXL3_LAUNCH
}

// exl3_int8_transpose: W_hat fp16 [K, N] -> int8 [N, K] + c_n[N]。
void exl3_int8_transpose(at::Tensor w_hat, at::Tensor out_int8,
                         at::Tensor c_n) {
  const int K = w_hat.size(0);
  const int N = w_hat.size(1);
  TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "K,N must be multiple of 128");
  TORCH_CHECK(out_int8.size(0) == N && out_int8.size(1) == K,
              "out_int8 must be [N, K]");
  TORCH_CHECK(c_n.size(0) == N, "c_n must be [N]");
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const __half* w_ptr = static_cast<const __half*>(w_hat.data_ptr());
  int8_t* out_ptr = static_cast<int8_t*>(out_int8.data_ptr());
  float* c_ptr = static_cast<float*>(c_n.data_ptr());
  // grid 只按 N 切: 每 block 128 列沿完整 K 求 amax (见 kernel 注释, 按 kb 切会覆盖 c_n)
  const dim3 grid(N / 128);
  exl3_int8_transpose_kernel<<<grid, RH_THREADS, 0, stream>>>(w_ptr, out_ptr,
                                                              c_ptr, K, N);
}

// 单调用便捷: trellis -> int8 [N, K] + c_n(内部分配 W_hat fp16 [K,N] 临时)。
void exl3_linear_dequant(at::Tensor packed, at::Tensor suh, at::Tensor svh,
                         at::Tensor out_int8, at::Tensor c_n, int64_t bits) {
  const int K = packed.size(0) * 16;
  const int N = packed.size(1) * 16;
  auto w_hat = torch::empty({K, N}, packed.options().dtype(torch::kHalf));
  exl3_decode(packed, suh, svh, w_hat, bits);
  exl3_int8_transpose(w_hat, out_int8, c_n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  // NOLINT
  m.def("exl3_decode", &exl3_decode,
        "EXL3 trellis -> fp16 [K,N] W_hat (mul1, sm75-safe)");
  m.def("exl3_int8_transpose", &exl3_int8_transpose,
        "EXL3 fp16 [K,N] -> int8 [N,K] + per-channel c_n[N]");
  m.def("exl3_linear_dequant", &exl3_linear_dequant,
        "EXL3 trellis -> int8 [N,K] + c_n (fused decode+transpose)");
}
