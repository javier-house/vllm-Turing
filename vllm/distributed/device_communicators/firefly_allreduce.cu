// SPDX-License-Identifier: Apache-2.0
// firefly_allreduce: fp8 2-GPU allreduce (SHM backend, PHB/无 P2P 如 T10)。
//
// 目标: TP2 每层 2 次 AllReduce([M,H] fp16 residual stream)在 PCIe 瓶颈上量减半
// (33.55MB fp16 -> 16.78MB fp8), 省通信时间 (PLAN-fp8-allreduce.md §2)。
//
// 传输: /dev/shm + cudaHostRegister (POC 验证 vfio 下可行, PLAN §7 风险4 已解除)。
//   cudaIpcGetMemHandle 对 host-mapped 内存 err=1 不通, 故不用 IPC, 改两进程 map 同一
//   tmpfs 区 + 各自 cudaHostRegister 成 pinned 设备可访问, 物理页经 tmpfs 共享。
//
// 流程 (每 rank, **全 GPU, 可被 cudagraph capture**):
//   round1 (GPU): 全局 amax (atomicMax) -> SHM 交换 amax (flag_scale spin) -> common scale
//   round2 (GPU):
//     quant (dev fp16 -> dev fp8 buffer, 用 scale_dev)
//     cudaMemcpyAsync D2H (dev fp8 -> shm.data[r] host)
//     set+spin data flag (release/acquire, 用 seq_dev)
//     cudaMemcpyAsync H2D (shm.data[1-r] host -> dev fp8 buffer)
//     dequant+sum ((self_q + peer_q) * scale_dev -> dev fp16, fp32 累加)
//     set+spin barrier flag (保证两 rank 都读完对端 data 才复用 slot)
//     bump seq_dev (本 allreduce 完成, flag 计数 +1)
//   barrier 保证两 rank 都读完对端 data 后才复用本端 slot (防 drift 覆盖)。
//   vllm forward 天然 lockstep(两 rank 同层同 allreduce), spin ~0 开销。
//
// **cudagraph 兼容**: seq/scale 放 device scratch (by-pointer, replay 时读当前值),
//   不用 by-value host 参数 (capture 会固定值, flag 不递增 → 无法区分本次/上次)。
//   seq_dev 每 allreduce +1 (ar_bump_seq), scale_dev 每步重算 (ar_scale_exchange)。
//   round1 不再走 CPU `.item()` D2H sync (那与 capture 的 stream 语义冲突)。
//
// fp8 转换: CUDA 原生 __nv_cvt_float_to_fp8 / __nv_cvt_fp8_to_halfraw (sm75 软转正确)。
//   c10 fp8e4m3fn_from_fp32_value 的 device 路径实测坏, 勿用 (proto 诊断确认)。
//
// SHM 布局 (字节, 与 python 侧约定一致; D = data_half = max M*H, fp8 1B/元素):
//   [0, D)            data_slot_0 (fp8)
//   [D, 2D)           data_slot_1 (fp8)
//   [2D, 2D+4)        amax_0 (f32)        round1 GPU 侧 (ar_scale_exchange)
//   [2D+4, 2D+8)      amax_1 (f32)
//   [2D+8, 2D+16)     flag_scale_0 (u64)  round1 GPU 侧
//   [2D+16, 2D+24)    flag_scale_1 (u64)
//   [2D+24, 2D+32)    flag_data_0 (u64)   round2 GPU 侧
//   [2D+32, 2D+40)    flag_data_1 (u64)
//   [2D+40, 2D+48)    barrier_0 (u64)     round2 GPU 侧 (读完对端后置位)
//   [2D+48, 2D+56)    barrier_1 (u64)
//   TOTAL = 2D + 56
//
// device scratch 布局 (16 字节, by-pointer 传, replay 读当前值):
//   [+0, +4)  amax_dev (f32)   全局 amax accumulator (每步 memset 0 后 atomicMax)
//   [+4, +8)  scale_dev (f32)  common scale (ar_scale_exchange 算, quant/dequant 读)
//   [+8, +16) seq_dev (u64)    flag 计数 (初始 1, 每 allreduce +1, 所有 flag 用它)
//
// 双 backend:
//   - SHM: 无 P2P (PHB, 如 T10)。data D2H/H2D 经 /dev/shm 共享 host 区,
//     launcher `firefly_ar_exchange` (host-mapped shm_base 单一指针)。
//   - P2P: 有 P2P (NVLink/PCIe 直连)。data/flag 全在 device 显存, 直接 P2P
//     读写 peer 显存 (无 host bounce), launcher `firefly_ar_exchange_p2p`
//     (own_base + peer_base 两显存指针)。buffer 布局与 SHM 完全一致, 只是指针
//     来自 CUDA IPC (每 rank 一 IPC buffer, 交换 handle 拿 peer 显存指针)。
//   quant/dequant/amax/set_spin/bump_seq 两 backend 共用; 仅 scale_exchange 有
//     指针版 (ar_scale_exchange_p2p), round2 数据搬运不同 (SHM 走 memcpy, P2P 直读)。
//
// N 卡 (world_size = 4/8, butterfly 递归折半): 两个 backend。
//   P2P/IPC: launcher `firefly_ar_butterfly` (bases_host host vector +
//     bases_dev_ptr device 数组首址, 后者仅作 kernel 实参 —— host 解引用
//     device tensor data_ptr 会 SIGSEGV, 4×T10 崩溃根因)。
//   SHM (无 P2P 如 T10): launcher `firefly_ar_butterfly_shm` (单一 shm_base
//     host-mapped 区, data 走 D2H/H2D memcpy)。
//   log2(N) 轮, 每轮 partner = rank ^ offset (offset=1,2,4,...); 每轮:
//     round1 (amax exchange 全体 N 卡, scale = global_amax * N / 448 防部分和
//       re-quant 到 e4m3 饱和 —— butterfly 中间和最大 N*amax) ->
//     round2 (quant -> 写 own data slot + flag -> partner 互读 -> dequant+sum
//       -> barrier 防下一轮覆盖)。out = sum_{r} x_r。精度同 2 卡 (fp8 近无损);
//   中间 re-quant 误差被 transformer 吸收 (PLAN §3.6), N 越大 re-quant 越多但
//   仍远小于 int8 定点 (PLAN §3.5)。quant/dequant/amax/set_spin/bump_seq 复用,
//   仅新增 ar_scale_exchange_n (N-way amax)。buffer 布局与 2 卡一致 (D=每 rank
//   slot 字节数), N 卡下 data 总量 = N*D (比 NCCL butterfly 2(N-1) 小)。
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>  // 轻量 stream API, 不拖 cusparse/cublas

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>  // __nv_cvt_float_to_fp8 / __nv_cvt_fp8_to_halfraw

#include <vector>  // std::vector<int64_t> (butterfly host 基址数组)

namespace {

// ---- 内存序原语 (.sys 域, 跨设备; 参考 custom_collective_common.cuh) ----
__device__ __forceinline__ void st_release_sys_u64(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v));
}
__device__ __forceinline__ uint64_t ld_acquire_sys_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p));
  return v;
}

// ---- GPU 自旋等待 + 有界 backoff (失锁步时不占满 SM) ----
// 正常锁步 (vllm forward 两 rank 同层同 allreduce): 命中前自旋次数远小于阈值,
//   不进 backoff, 开销 ~0。
// 失锁步 (某 rank 被 CPU 侧拖慢): 连续未命中达阈值后 __nanosleep, 释放 SM,
//   避免纯忙等把 nvidia-smi util 顶到 100%; peer 回来后仍微秒级重试快速响应。
// 语义与裸自旋完全等价 (release/acquire 由 st/ld 保证, backoff 只是"未命中时
//   睡一会儿"); 不引入 CPU-GPU 同步 / 回退, 保持 cudagraph capture 安全。
__device__ __forceinline__ void spin_wait_until_eq_sys(const uint64_t* p,
                                                       uint64_t expect) {
  int spins = 0;
  while (ld_acquire_sys_u64(p) != expect) {
    if (++spins >= 128) {
      __nanosleep(1000);
    }
  }
}

// fp8 转换 (sm75 无 fp8 硬件, CUDA 原生软转; c10 device 路径实测坏)
__device__ __forceinline__ uint8_t quant_to_fp8(float v) {
  return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float dequant_from_fp8(uint8_t q) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
  return __half2float(__half(h));
}

// ---- kernel: 全局 amax (atomicMax 到 amax_dev; |x| 非负) ----
// 非负 float 的 __float_as_uint 位模式随值单调递增, 故 atomicMax(uint)==float max
// (sm75 无 f32 atom 硬件, 用 uint atomicMax 绕开)。
__global__ void ar_amax_partial(const __half* __restrict__ x,
                                float* __restrict__ amax_dev, int64_t n) {
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  unsigned int* acc = reinterpret_cast<unsigned int*>(amax_dev);
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    atomicMax(acc, __float_as_uint(fabsf(__half2float(x[i]))));
  }
}

// ---- kernel: 1 block, SHM 交换 amax -> common scale (round1, GPU 侧) ----
// seq 从 seq_dev 读 (by-pointer, replay 安全); 写 scale_dev 供 quant/dequant 读。
__global__ void ar_scale_exchange(char* __restrict__ shm_base,
                                  int64_t data_half, int64_t rank,
                                  const float* __restrict__ amax_dev,
                                  float* __restrict__ scale_dev,
                                  const uint64_t* __restrict__ seq_dev) {
  if (threadIdx.x == 0) {
    uint64_t seq = *seq_dev;
    float local_amax =
        __uint_as_float(*reinterpret_cast<const unsigned int*>(amax_dev));
    char* base = shm_base;
    float* amax_shm = reinterpret_cast<float*>(base + 2 * data_half);
    uint64_t* flag_scale =
        reinterpret_cast<uint64_t*>(base + 2 * data_half + 8);
    amax_shm[rank] = local_amax;                       // 写本端 amax
    st_release_sys_u64(flag_scale + rank, seq);         // 发布 (release)
    spin_wait_until_eq_sys(flag_scale + (1 - rank), seq);  // 等对端 (acquire)
    float peer_amax = amax_shm[1 - rank];
    *scale_dev = fmaxf(local_amax, peer_amax) / 448.0f;  // e4m3fn max finite
  }
}

// ---- kernel: P2P 版 amax 交换 (round1, GPU 侧) ----
// own_base/peer_base = 本端/对端 IPC buffer 显存首址; 写 own[2D] amax 区 +
// own[2D+8] flag, 等 peer[2D+8] flag, 读 peer[2D] amax。内存序 .sys (跨设备)。
__global__ void ar_scale_exchange_p2p(char* __restrict__ own_base,
                                      char* __restrict__ peer_base,
                                      int64_t data_half,
                                      const float* __restrict__ amax_dev,
                                      float* __restrict__ scale_dev,
                                      const uint64_t* __restrict__ seq_dev) {
  if (threadIdx.x == 0) {
    uint64_t seq = *seq_dev;
    float local_amax =
        __uint_as_float(*reinterpret_cast<const unsigned int*>(amax_dev));
    // amax 区紧接 data 双区之后 (布局同 SHM: [2D, 2D+4) amax, [2D+8] flag_scale)
    float* own_amax =
        reinterpret_cast<float*>(own_base + 2 * data_half);
    float* peer_amax =
        reinterpret_cast<float*>(peer_base + 2 * data_half);
    uint64_t* own_flag =
        reinterpret_cast<uint64_t*>(own_base + 2 * data_half + 8);
    uint64_t* peer_flag =
        reinterpret_cast<uint64_t*>(peer_base + 2 * data_half + 8);
    *own_amax = local_amax;                        // 写本端 amax
    st_release_sys_u64(own_flag, seq);             // 发布 (release)
    spin_wait_until_eq_sys(peer_flag, seq);        // 等对端 (acquire)
    *scale_dev = fmaxf(local_amax, *peer_amax) / 448.0f;
  }
}

// ---- kernel: N-way amax 交换 (round1, butterfly 用, P2P/IPC) ----
// bases_dev[i] = 第 i 个 rank 的 IPC buffer 显存首址 (device array, 本 rank 持
// 有全体, 含自己; host 侧不解读它 —— 只作 kernel 实参)。写 own[N*D] amax +
// own[N*D+8] flag, 等其余 N-1 卡 [N*D+8] flag = seq, 读全体 [N*D] amax 取
// global max。
// **scale = global_amax * N / 448** (非 amax/448): butterfly 每轮 dequant+sum
// 后部分和最大到 k*amax (k 累加卡数), 下一轮 re-quant 到 e4m3 会饱和; 用 N 倍
// 裕量让中间和 (<=N*amax) 量化到 ~amax/e4m3_max 比例, 精度等价 2 卡 (最终
// /N 抵消)。内存序 .sys (跨设备显存)。
__global__ void ar_scale_exchange_n(char** __restrict__ bases_dev,
                                    int64_t data_half, int64_t rank,
                                    int64_t world,
                                    const float* __restrict__ amax_dev,
                                    float* __restrict__ scale_dev,
                                    const uint64_t* __restrict__ seq_dev) {
  if (threadIdx.x == 0) {
    uint64_t seq = *seq_dev;
    float local_amax =
        __uint_as_float(*reinterpret_cast<const unsigned int*>(amax_dev));
    char* own = bases_dev[rank];
    // metadata 在 N 个 data slot (N*D) 之后 (2-GPU 时 N*D==2D, 同 P2P 版布局)
    float* own_amax = reinterpret_cast<float*>(own + world * data_half);
    uint64_t* own_flag =
        reinterpret_cast<uint64_t*>(own + world * data_half + 8);
    *own_amax = local_amax;                    // 写本端 amax
    st_release_sys_u64(own_flag, seq);         // 发布 (release)
    float global_amax = local_amax;
    for (int64_t r = 0; r < world; ++r) {
      if (r == rank) continue;
      char* peer = bases_dev[r];
      uint64_t* peer_flag =
          reinterpret_cast<uint64_t*>(peer + world * data_half + 8);
      spin_wait_until_eq_sys(peer_flag, seq);        // 等其余卡 (acquire)
      float pa = *reinterpret_cast<float*>(peer + world * data_half);
      if (pa > global_amax) global_amax = pa;
    }
    *scale_dev = global_amax * (float)world / 448.0f;
  }
}

// ---- kernel: N-way amax 交换 (round1, butterfly 用, SHM) ----
// 同 ar_scale_exchange_n, 但基址是单一 shm_base (host-mapped pinned, 所有 rank
// map 同一 tmpfs 区), 无需 per-rank 基址数组。metadata 偏移与 P2P 版一致
// (相对 shm_base: [N*D] amax 数组, [N*D+8] flag_scale 数组, 各 N 个槽位)。
__global__ void ar_scale_exchange_n_shm(char* __restrict__ shm_base,
                                        int64_t data_half, int64_t rank,
                                        int64_t world,
                                        const float* __restrict__ amax_dev,
                                        float* __restrict__ scale_dev,
                                        const uint64_t* __restrict__ seq_dev) {
  if (threadIdx.x == 0) {
    uint64_t seq = *seq_dev;
    float local_amax =
        __uint_as_float(*reinterpret_cast<const unsigned int*>(amax_dev));
    float* amax_arr = reinterpret_cast<float*>(shm_base + world * data_half);
    uint64_t* flag_arr =
        reinterpret_cast<uint64_t*>(shm_base + world * data_half + 4 * world);
    amax_arr[rank] = local_amax;                // 写本端 amax
    st_release_sys_u64(flag_arr + rank, seq);   // 发布 (release)
    float global_amax = local_amax;
    for (int64_t r = 0; r < world; ++r) {
      if (r == rank) continue;
      spin_wait_until_eq_sys(flag_arr + r, seq);  // 等其余卡 (acquire)
      float pa = amax_arr[r];
      if (pa > global_amax) global_amax = pa;
    }
    *scale_dev = global_amax * (float)world / 448.0f;
  }
}

// ---- kernel: quantize (x dev fp16 -> xq dev fp8; scale 从 scale_dev 读) ----
__global__ void ar_quant(const __half* __restrict__ x,
                         uint8_t* __restrict__ dst,
                         const float* __restrict__ scale_dev, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = quant_to_fp8(__half2float(x[i]) / (*scale_dev));
}

// ---- kernel: dequant + sum (self_q + peer_q dev fp8 -> out dev fp16) ----
__global__ void ar_dequant_sum(const uint8_t* __restrict__ self_q,
                               const uint8_t* __restrict__ peer_q,
                               __half* __restrict__ out,
                               const float* __restrict__ scale_dev,
                               int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float a = dequant_from_fp8(self_q[i]);
    float b = dequant_from_fp8(peer_q[i]);
    out[i] = __float2half((a + b) * (*scale_dev));  // fp32 累加, 回 fp16
  }
}

// ---- kernel: set 本端 flag (release) + spin 对端 flag (acquire), 用 seq_dev ----
__global__ void ar_set_spin(uint64_t* __restrict__ my_flag,
                            const uint64_t* __restrict__ peer_flag,
                            const uint64_t* __restrict__ seq_dev) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint64_t seq = *seq_dev;
    st_release_sys_u64(my_flag, seq);
    spin_wait_until_eq_sys(peer_flag, seq);
  }
}

// ---- kernel: seq_dev += 1 (本 allreduce 完成, 供下次 flag 递增) ----
__global__ void ar_bump_seq(uint64_t* __restrict__ seq_dev) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *seq_dev += 1;
}

// ---- kernel: 只 set 本端 flag (release), 不 spin 对端 (流水线 per-chunk 用) ----
// 拆分自 ar_set_spin 的「set」半边: 流水线里 D2H 完一个 chunk 就发它的 flag,
// 不必等对端 (对端 flag 由 H2D 前的 ar_flag_wait 单独等)。
__global__ void ar_flag_set(uint64_t* __restrict__ my_flag,
                            const uint64_t* __restrict__ seq_dev) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    st_release_sys_u64(my_flag, *seq_dev);
}

// ---- kernel: 只 spin 对端 flag (acquire), 不 set (流水线 per-chunk 用) ----
// 拆分自 ar_set_spin 的「wait」半边: H2D 拉对端 chunk 前, 先等对端把该 chunk
// 的 D2H flag 置到 seq (acquire 保证拉到的是完整 chunk)。
__global__ void ar_flag_wait(const uint64_t* __restrict__ peer_flag,
                             const uint64_t* __restrict__ seq_dev) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    spin_wait_until_eq_sys(peer_flag, *seq_dev);
}

}  // namespace

// host launcher: 全 GPU round1+round2 (无 host sync, 可 cudagraph capture)。
// 单 stream: D2H/H2D 串行 (T10 PCIe 半双工, 双 stream 重叠实测无收益 2026-09-09)。
// x: fp16 [M,H] contiguous (输入); xq/xq_peer: fp8 [n] device scratch;
//   out: fp16 [M,H] (x0+x1); scratch: device [16B] (amax/scale/seq, by-pointer);
//   shm_base: 已 cudaHostRegister 的 shm 首址; data_half: 数据半区字节数(>=n);
//   rank: 0/1; n: 实际元素数(M*H)。
void firefly_ar_exchange(at::Tensor x, at::Tensor xq, at::Tensor xq_peer,
                         at::Tensor out, at::Tensor scratch,
                         int64_t shm_base, int64_t data_half,
                         int64_t rank, int64_t n) {
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const __half* xp = reinterpret_cast<const __half*>(x.data_ptr());
  uint8_t* xqp = reinterpret_cast<uint8_t*>(xq.data_ptr());
  uint8_t* xqpp = reinterpret_cast<uint8_t*>(xq_peer.data_ptr());
  __half* outp = reinterpret_cast<__half*>(out.data_ptr());
  // scratch: [+0] amax(f32) [+4] scale(f32) [+8] seq(u64)
  void* sp = scratch.data_ptr();
  float* amax_dev = reinterpret_cast<float*>(sp);
  float* scale_dev = reinterpret_cast<float*>(sp) + 1;
  uint64_t* seq_dev = reinterpret_cast<uint64_t*>(sp) + 1;
  char* base = reinterpret_cast<char*>(shm_base);
  uint8_t* my_data = reinterpret_cast<uint8_t*>(base) + rank * data_half;
  uint8_t* peer_data = reinterpret_cast<uint8_t*>(base) + (1 - rank) * data_half;
  uint64_t* flag_data =
      reinterpret_cast<uint64_t*>(base + 2 * data_half + 24);
  uint64_t* my_flag_data = flag_data + rank;
  uint64_t* peer_flag_data = flag_data + (1 - rank);
  uint64_t* flag_barrier =
      reinterpret_cast<uint64_t*>(base + 2 * data_half + 40);
  uint64_t* my_flag_barrier = flag_barrier + rank;
  uint64_t* peer_flag_barrier = flag_barrier + (1 - rank);
  constexpr int threads = 256;
  const dim3 grid((unsigned)((n + threads - 1) / threads));

  // round1 (GPU): 全局 amax -> SHM 交换 -> common scale
  cudaMemsetAsync(amax_dev, 0, sizeof(float), stream);
  ar_amax_partial<<<grid, threads, 0, stream>>>(xp, amax_dev, n);
  ar_scale_exchange<<<1, 1, 0, stream>>>(reinterpret_cast<char*>(shm_base),
                                         data_half, rank, amax_dev, scale_dev,
                                         seq_dev);
  // round2 (GPU): quant + D2H + data flag + H2D + dequant+sum + barrier + bump
  ar_quant<<<grid, threads, 0, stream>>>(xp, xqp, scale_dev, n);
  cudaMemcpyAsync(my_data, xqp, (size_t)n, cudaMemcpyDeviceToHost, stream);
  ar_set_spin<<<1, 1, 0, stream>>>(my_flag_data, peer_flag_data, seq_dev);
  cudaMemcpyAsync(xqpp, peer_data, (size_t)n, cudaMemcpyHostToDevice, stream);
  ar_dequant_sum<<<grid, threads, 0, stream>>>(xqp, xqpp, outp, scale_dev, n);
  ar_set_spin<<<1, 1, 0, stream>>>(my_flag_barrier, peer_flag_barrier, seq_dev);
  ar_bump_seq<<<1, 1, 0, stream>>>(seq_dev);
}

// ---- 流水线 (2 卡 SHM 分块全双工) 资源: 2 持久 stream + 3 event ----
// A=D2H (发我数据), B=H2D (收对端数据)。全双工 PCIe 链 (二号机 GPU0<->host) 上
// A/B 真并发 (两路不同 buffer / 相反方向, 无同进程依赖)。fork_a/fork_b = 主流
// quant 后 fork 到 A/B (A 读 xqp 须等 quant); join_b = B 拉回后 join 回主流
// (主流 dequant 读 xqpp 前等 B 拉完)。init 时建 (capture 外), 每次 allreduce
// 复用; destroy 释放。
// **为何不设 join_a (B 等 A 同块)**: 同进程 B 拉 peer_data / A 发 my_data 是不同
// host 区无冲突, 跨块正确性由 per-chunk SHM flag (peer_data_flag) 保证; 设
// join_a 反会把 A(c) 排在 B(c) 前、限制 overlap, 且单一 event 复用 K 次有
// 假满足风险 (wait 被更晚块的 record 满足)。去掉后 A/B 纯并发, 全双工 overlap
// 最大化。A 无需 join 回主流: 下次 AR 的 fork_a 在主流上 (本轮 barrier wait 之
// 后), 传递排序保证下次 AR 的 A-D2H 晚于本轮 peer 读完 my_data, 无覆盖 hazard。
namespace {
struct PipeState {
  cudaStream_t a = nullptr, b = nullptr;
  cudaEvent_t fork_a = nullptr, fork_b = nullptr;
  cudaEvent_t join_b = nullptr;
};
PipeState g_pipe;
}  // namespace

void firefly_ar_init_pipe() {
  if (g_pipe.a) return;  // 幂等
  cudaStreamCreateWithFlags(&g_pipe.a, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&g_pipe.b, cudaStreamNonBlocking);
  cudaEventCreateWithFlags(&g_pipe.fork_a, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&g_pipe.fork_b, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&g_pipe.join_b, cudaEventDisableTiming);
}

void firefly_ar_destroy_pipe() {
  if (g_pipe.a) {
    cudaStreamDestroy(g_pipe.a);
    cudaStreamDestroy(g_pipe.b);
    cudaEventDestroy(g_pipe.fork_a);
    cudaEventDestroy(g_pipe.fork_b);
    cudaEventDestroy(g_pipe.join_b);
  }
  g_pipe = PipeState{};
}

// host launcher (SHM 分块流水线, 2 卡): 大 n 全双工 overlap 版。
// **数值与串行 firefly_ar_exchange 逐 bit 一致**: round1 (整消息全局 amax ->
// common scale) + 每元素 quant/dequant 都不变, 仅把 D2H/H2D 传输按 chunk 字节
// 分块并跨 A/B stream overlap —— 全双工链 (二号机 GPU0<->host) 上 send 块 k+1
// (A D2H) 与 recv 块 k (B H2D) 并发, 通信项 ~2x; 半双工链 (T10/一号机) 上分块
// 串行化 ≈ 无回退。默认关, VLLM_FIREFLY_AR_PIPE 开才走这里 (见 .py)。
// round1 (主流): 全局 amax -> SHM 交换 -> common scale (整消息一次, 同串行)。
// round2 (主流 + A/B 双 stream 流水线, per-chunk flag, seq 每 allreduce +1):
//   主流: quant(整消息) -> record fork_a/fork_b (fork 到 A/B)
//   逐块 c (A/B 并发): A 上 D2H(my_data[c]) -> ar_flag_set(my_data_flag[c])
//           B 上 ar_flag_wait(peer_data_flag[c]) -> H2D(peer_data[c])
//             -> ar_flag_set(my_barrier_flag[c]) -> record join_b
//   主流: wait join_b (我 _xq_peer 拉回) -> ar_flag_wait(peer_barrier_flag[last])
//     (对端读完我全部 my_data, 防下次 AR D2H 覆盖未读 slot, 同串行 barrier 语义)
//     -> dequant(整消息, self _xq + peer _xq_peer) -> bump seq
// per-chunk flag 双数组 (python 分配, 紧跟 56B 元数据, 布局见上):
//   data_flag[rank][c]   = rank 的 D2H chunk c 完成 (peer H2D 前等)
//   barrier_flag[rank][c] = rank 读完 peer_data chunk c (H2D 完成, peer 等)
//   data_flag[c] 与 barrier_flag[c] 独立 (一个写 my_data 一个读 peer_data, 不同
//   host 区, 无冲突)。两卡对称锁步 (vllm forward 同层同 allreduce), 无死锁:
//   各 rank 的 A 流只发不等待 (D2H+set) 恒推进, B/主流只等对端已推进的 flag。
// **cudagraph 安全**: A/B + event 在 init 建 (capture 外); 每次 allreduce 用 event
// fork/join (主流 fork -> A/B overlap -> join 主流), 无 CPU sync, seq 由
// ar_bump_seq device 端 +1 (replay 读当前值), flag 全读 seq_dev (by-pointer)。
void firefly_ar_exchange_pipe(at::Tensor x, at::Tensor xq, at::Tensor xq_peer,
                              at::Tensor out, at::Tensor scratch,
                              int64_t shm_base, int64_t data_half,
                              int64_t rank, int64_t n, int64_t chunk,
                              int64_t max_chunks) {
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const cudaStream_t sa = g_pipe.a;
  const cudaStream_t sb = g_pipe.b;
  const __half* xp = reinterpret_cast<const __half*>(x.data_ptr());
  uint8_t* xqp = reinterpret_cast<uint8_t*>(xq.data_ptr());
  uint8_t* xqpp = reinterpret_cast<uint8_t*>(xq_peer.data_ptr());
  __half* outp = reinterpret_cast<__half*>(out.data_ptr());
  // scratch: [+0] amax(f32) [+4] scale(f32) [+8] seq(u64)
  void* sp = scratch.data_ptr();
  float* amax_dev = reinterpret_cast<float*>(sp);
  float* scale_dev = reinterpret_cast<float*>(sp) + 1;
  uint64_t* seq_dev = reinterpret_cast<uint64_t*>(sp) + 1;
  char* base = reinterpret_cast<char*>(shm_base);
  uint8_t* my_data = reinterpret_cast<uint8_t*>(base) + rank * data_half;
  uint8_t* peer_data = reinterpret_cast<uint8_t*>(base) + (1 - rank) * data_half;
  // per-chunk flag 区 (紧跟 56B 元数据): data_flag[rank][c] / barrier_flag[rank][c]
  const int64_t flag_off = 2 * data_half + 56;
  uint64_t* my_data_flag =
      reinterpret_cast<uint64_t*>(base + flag_off) + rank * max_chunks;
  uint64_t* peer_data_flag =
      reinterpret_cast<uint64_t*>(base + flag_off) + (1 - rank) * max_chunks;
  uint64_t* my_barrier_flag =
      reinterpret_cast<uint64_t*>(base + flag_off + 2 * max_chunks) +
      rank * max_chunks;
  uint64_t* peer_barrier_flag =
      reinterpret_cast<uint64_t*>(base + flag_off + 2 * max_chunks) +
      (1 - rank) * max_chunks;
  constexpr int threads = 256;
  const dim3 grid((unsigned)((n + threads - 1) / threads));
  const int64_t C = chunk;  // 每块 fp8 字节 (= fp16 元素数)

  // round1 (主流): 整消息全局 amax -> SHM 交换 -> common scale (数值同串行)
  cudaMemsetAsync(amax_dev, 0, sizeof(float), stream);
  ar_amax_partial<<<grid, threads, 0, stream>>>(xp, amax_dev, n);
  ar_scale_exchange<<<1, 1, 0, stream>>>(reinterpret_cast<char*>(shm_base),
                                         data_half, rank, amax_dev, scale_dev,
                                         seq_dev);
  // round2 (主流 + A/B 流水线): 整消息 quant -> fork 到 A/B -> 逐块 D2H/A +
  //   H2D/B overlap -> join 主流 -> 整消息 dequant+sum -> bump seq。
  ar_quant<<<grid, threads, 0, stream>>>(xp, xqp, scale_dev, n);
  cudaEventRecord(g_pipe.fork_a, stream);
  cudaEventRecord(g_pipe.fork_b, stream);
  cudaStreamWaitEvent(sa, g_pipe.fork_a, 0);
  cudaStreamWaitEvent(sb, g_pipe.fork_b, 0);
  for (int64_t off = 0, c = 0; off < n; off += C, ++c) {
    int64_t csz = (n - off < C) ? (n - off) : C;
    // A (发): D2H 我本块 -> set 我 data_flag[c]。A 恒推进, 不等待 (全双工)。
    cudaMemcpyAsync(my_data + off, xqp + off, (size_t)csz,
                    cudaMemcpyDeviceToHost, sa);
    ar_flag_set<<<1, 1, 0, sa>>>(my_data_flag + c, seq_dev);
    // B (收): 等对端本块 D2H 完 (ar_flag_wait peer_data_flag[c], 跨进程正确性)
    //   -> H2D 拉本块 -> set 我 barrier_flag[c] (我读完 peer 本块) -> join_b。
    //   B 不依赖 A (不同 host 区 / 相反方向), 与 A 并发。
    ar_flag_wait<<<1, 1, 0, sb>>>(peer_data_flag + c, seq_dev);
    cudaMemcpyAsync(xqpp + off, peer_data + off, (size_t)csz,
                    cudaMemcpyHostToDevice, sb);
    ar_flag_set<<<1, 1, 0, sb>>>(my_barrier_flag + c, seq_dev);
    cudaEventRecord(g_pipe.join_b, sb);
  }
  // 主流: 等 B 拉回全部 (join_b 传递依赖 A 全块已发) -> 等对端读完我全部
  // my_data (barrier, 防下次 AR D2H 覆盖未读 slot) -> 整消息 dequant+sum -> bump
  cudaStreamWaitEvent(stream, g_pipe.join_b, 0);
  const int64_t last = (n + C - 1) / C - 1;
  ar_flag_wait<<<1, 1, 0, stream>>>(peer_barrier_flag + last, seq_dev);
  ar_dequant_sum<<<grid, threads, 0, stream>>>(xqp, xqpp, outp, scale_dev, n);
  ar_bump_seq<<<1, 1, 0, stream>>>(seq_dev);
}

// host launcher (P2P backend): 全 GPU round1+round2, 同 SHM 版但 data 走 device
// 显存 P2P (无 D2H/H2D)。own_base/peer_base = 本端/对端 IPC buffer 显存首址;
// data slot / flag / barrier 偏移与 SHM 一致 (布局同, 只是指针来自 IPC)。
// x: fp16 [M,H] (输入); out: fp16 (x0+x1); scratch: device [16B] (by-pointer);
// data_half: 数据半区字节数(>=n); rank: 0/1; n: 实际元素数(M*H)。
void firefly_ar_exchange_p2p(at::Tensor x, at::Tensor out, at::Tensor scratch,
                             int64_t own_base, int64_t peer_base,
                             int64_t data_half, int64_t rank, int64_t n) {
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const __half* xp = reinterpret_cast<const __half*>(x.data_ptr());
  __half* outp = reinterpret_cast<__half*>(out.data_ptr());
  void* sp = scratch.data_ptr();
  float* amax_dev = reinterpret_cast<float*>(sp);
  float* scale_dev = reinterpret_cast<float*>(sp) + 1;
  uint64_t* seq_dev = reinterpret_cast<uint64_t*>(sp) + 1;
  char* own = reinterpret_cast<char*>(own_base);
  char* peer = reinterpret_cast<char*>(peer_base);
  uint8_t* own_data = reinterpret_cast<uint8_t*>(own) + rank * data_half;
  uint8_t* peer_data = reinterpret_cast<uint8_t*>(peer) + (1 - rank) * data_half;
  uint64_t* own_flag_data =
      reinterpret_cast<uint64_t*>(own + 2 * data_half + 24) + rank;
  uint64_t* peer_flag_data =
      reinterpret_cast<uint64_t*>(peer + 2 * data_half + 24) + (1 - rank);
  uint64_t* own_flag_barrier =
      reinterpret_cast<uint64_t*>(own + 2 * data_half + 40) + rank;
  uint64_t* peer_flag_barrier =
      reinterpret_cast<uint64_t*>(peer + 2 * data_half + 40) + (1 - rank);
  constexpr int threads = 256;
  const dim3 grid((unsigned)((n + threads - 1) / threads));

  // round1 (GPU): 全局 amax -> P2P 交换 -> common scale
  cudaMemsetAsync(amax_dev, 0, sizeof(float), stream);
  ar_amax_partial<<<grid, threads, 0, stream>>>(xp, amax_dev, n);
  ar_scale_exchange_p2p<<<1, 1, 0, stream>>>(
      own, peer, data_half, amax_dev, scale_dev, seq_dev);
  // round2 (GPU): quant -> 写 own_data (P2P 可被 peer 读) + data flag +
  //   dequant (P2P 读 own/peer data) + barrier + bump seq
  ar_quant<<<grid, threads, 0, stream>>>(xp, own_data, scale_dev, n);
  ar_set_spin<<<1, 1, 0, stream>>>(own_flag_data, peer_flag_data, seq_dev);
  ar_dequant_sum<<<grid, threads, 0, stream>>>(own_data, peer_data, outp,
                                               scale_dev, n);
  ar_set_spin<<<1, 1, 0, stream>>>(own_flag_barrier, peer_flag_barrier,
                                   seq_dev);
  ar_bump_seq<<<1, 1, 0, stream>>>(seq_dev);
}

// host launcher (N 卡 butterfly, world = 4/8, P2P/IPC backend): out = sum x_r。
// 算法: log2(N) 轮, 每轮 partner = rank ^ offset (offset=1,2,4,...)。每轮两卡
// 互读对方 data slot 后 dequant+sum 得新部分和写 work; 末轮 work = 全体和, 拷入
// out。data/flag 全 device 显存 (IPC buffer), 无 host bounce。
// 关键: dequant 的 out 参数用独立 work 缓冲, **不**写回 own data slot —— 若写回
// slot, dequant 的写会覆盖 quant 刚写入的 fp8 (同地址), peer 读到坏值。
// **host 侧严禁解引用 device 数组**: bases_dev (device int64[N]) 的 data_ptr
// 在 host 端解引用会 SIGSEGV (4×T10 崩溃根因, tmp/AR-crash-report)。故拆两份:
//   bases_host = host vector (python 传 self._ptrs list[int], pybind 转 vector),
//     host 端 pointer arithmetic 用 (own/peer/flag 指针全由此算);
//   bases_dev_ptr = device 数组首址 (int64), 仅作 ar_scale_exchange_n 的 kernel
//     实参 (kernel 内部解引用合法), host 不读。
// x: fp16 [M,H] (本 rank 输入, 只读); work: fp16 [n] device (运行部分和,
//   python 侧持久分配, 首轮前 python 已把 x 拷入); out: fp16 (sum, 独立缓冲);
// scratch: device [16B] (amax/scale/seq, by-pointer, replay 安全);
// data_half: 每 rank data slot 字节数(>=n); rank: 本 rank; n: 元素数(M*H)。
void firefly_ar_butterfly(at::Tensor x, at::Tensor work, at::Tensor out,
                          at::Tensor scratch,
                          std::vector<int64_t> bases_host,
                          int64_t bases_dev_ptr, int64_t data_half,
                          int64_t rank, int64_t n) {
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const __half* xp = reinterpret_cast<const __half*>(x.data_ptr());
  __half* wp = reinterpret_cast<__half*>(work.data_ptr());
  __half* outp = reinterpret_cast<__half*>(out.data_ptr());
  void* sp = scratch.data_ptr();
  float* amax_dev = reinterpret_cast<float*>(sp);
  float* scale_dev = reinterpret_cast<float*>(sp) + 1;
  uint64_t* seq_dev = reinterpret_cast<uint64_t*>(sp) + 1;
  int64_t world = (int64_t)bases_host.size();  // N (world_size)
  // ar_scale_exchange_n 的 kernel 实参: device 端 char** (kernel 内解引用合法)
  char** bases_dev = reinterpret_cast<char**>(bases_dev_ptr);
  constexpr int threads = 256;
  const dim3 grid((unsigned)((n + threads - 1) / threads));

  // 全局 amax (本 rank 原始输入 x, butterfly 前算一次; 各 rank x 同形状)
  cudaMemsetAsync(amax_dev, 0, sizeof(float), stream);
  ar_amax_partial<<<grid, threads, 0, stream>>>(xp, amax_dev, n);
  // N-way amax 交换 -> common scale, **只在 butterfly 前算一次**: scale 全程
  //   固定为 global_amax*N/448 (amax_dev 不变), 中间和 <= N*amax 不饱和; 循环内
  //   重复交换是同一个值, 纯属浪费 N-1 卡 spin (4 卡省 1 次, 8 卡省 2 次)。
  ar_scale_exchange_n<<<1, 1, 0, stream>>>(bases_dev, data_half, rank, world,
                                           amax_dev, scale_dev, seq_dev);

  for (int64_t offset = 1; offset < world; offset <<= 1) {
    int64_t partner = rank ^ offset;
    // host 指针 arithmetic 全走 bases_host (host int64, 合法)
    char* own = reinterpret_cast<char*>(bases_host[rank]);
    char* peer = reinterpret_cast<char*>(bases_host[partner]);
    uint8_t* own_data = reinterpret_cast<uint8_t*>(own) + rank * data_half;
    uint8_t* peer_data =
        reinterpret_cast<uint8_t*>(peer) + partner * data_half;
    // metadata 区在 N*D 之后 (布局, 相对 base=world*data_half):
    //   [0) amax f32 | [8) flag_scale u64 | [24, 24+8w) flag_data u64[w]
    //   [24+8w, 24+16w) barrier u64[w]  (w=world; world=2 时 barrier@40, 同 2 卡)
    uint64_t* own_flag_data =
        reinterpret_cast<uint64_t*>(own + world * data_half + 24) + rank;
    uint64_t* peer_flag_data =
        reinterpret_cast<uint64_t*>(peer + world * data_half + 24) + partner;
    uint64_t* own_flag_barrier =
        reinterpret_cast<uint64_t*>(own + world * data_half + 24 + 8 * world) +
        rank;
    uint64_t* peer_flag_barrier =
        reinterpret_cast<uint64_t*>(peer + world * data_half + 24 + 8 * world) +
        partner;

    // round2: quant(work, 当前部分和) -> 写 own slot + data flag +
    //   dequant(own+peer) 写独立 work (部分和) + barrier + bump seq。
    //   (scale 交换已提到循环外, 见上注释)
    ar_quant<<<grid, threads, 0, stream>>>(wp, own_data, scale_dev, n);
    ar_set_spin<<<1, 1, 0, stream>>>(own_flag_data, peer_flag_data, seq_dev);
    ar_dequant_sum<<<grid, threads, 0, stream>>>(own_data, peer_data, wp,
                                                 scale_dev, n);
    ar_set_spin<<<1, 1, 0, stream>>>(own_flag_barrier, peer_flag_barrier,
                                     seq_dev);
    ar_bump_seq<<<1, 1, 0, stream>>>(seq_dev);
  }
  // 末轮 work == 全体和 -> 拷入独立 out (不动输入 x)
  cudaMemcpyAsync(outp, wp, (size_t)n * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);
}

// host launcher (N 卡 butterfly, world = 4/8, SHM backend, 无 P2P 如 4×T10):
// out = sum x_r。算法同 P2P 版 (log2(N) 轮, partner = rank ^ offset), 但
// data/flag 全在单一 shm_base (host-mapped pinned, 所有 rank map 同一 tmpfs
// 区), data 搬运走 D2H/H2D memcpy (无 P2P 直读 peer 显存)。
// SHM 布局 (字节, D = data_half, N = world):
//   [0, N*D)              data slots (rank r 的 slot 在 r*D, fp8)
//   [N*D, N*D+4N)         amax 数组 (f32 × N)
//   [N*D+4N, +8N)         flag_scale 数组 (u64 × N)
//   [N*D+4N+8N, +8N)      flag_data 数组 (u64 × N)
//   [N*D+4N+16N, +8N)     barrier 数组 (u64 × N)
//   TOTAL = N*D + 28N
// 每轮: quant(work -> dev _xq) -> D2H 到 own shm slot + data flag ->
//   H2D partner slot 到 dev _xq_peer -> dequant_sum(_xq, _xq_peer) 写回 work
//   -> barrier + bump seq。dequant 两输入是 self quant (dev _xq) 与 peer quant
//   (H2D 回来的 _xq_peer), 输出写回 work (运行部分和)。
// x: fp16 [M,H] (只读); work: fp16 [n] (部分和, 首轮前 python 已拷入 x);
//   xq/xq_peer: fp8 [n] device scratch; out: fp16 (sum); scratch: device [16B];
//   shm_base: 已 cudaHostRegister 的 shm 首址; data_half: 每 rank slot 字节数;
//   rank/world: 本 rank / 总卡数; n: 元素数(M*H)。
void firefly_ar_butterfly_shm(at::Tensor x, at::Tensor work, at::Tensor xq,
                              at::Tensor xq_peer, at::Tensor out,
                              at::Tensor scratch, int64_t shm_base,
                              int64_t data_half, int64_t rank, int64_t world,
                              int64_t n) {
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const __half* xp = reinterpret_cast<const __half*>(x.data_ptr());
  __half* wp = reinterpret_cast<__half*>(work.data_ptr());
  uint8_t* xqp = reinterpret_cast<uint8_t*>(xq.data_ptr());
  uint8_t* xqpp = reinterpret_cast<uint8_t*>(xq_peer.data_ptr());
  __half* outp = reinterpret_cast<__half*>(out.data_ptr());
  void* sp = scratch.data_ptr();
  float* amax_dev = reinterpret_cast<float*>(sp);
  float* scale_dev = reinterpret_cast<float*>(sp) + 1;
  uint64_t* seq_dev = reinterpret_cast<uint64_t*>(sp) + 1;
  char* base = reinterpret_cast<char*>(shm_base);
  constexpr int threads = 256;
  const dim3 grid((unsigned)((n + threads - 1) / threads));

  // 全局 amax (本 rank 原始输入 x, butterfly 前算一次, 与 P2P 版一致)
  cudaMemsetAsync(amax_dev, 0, sizeof(float), stream);
  ar_amax_partial<<<grid, threads, 0, stream>>>(xp, amax_dev, n);
  // N-way amax 交换 -> common scale, **只在 butterfly 前算一次** (scale 全程
  //   固定, 循环内重复交换是同一个值, 纯属浪费; 4 卡省 1 次, 8 卡省 2 次)。
  ar_scale_exchange_n_shm<<<1, 1, 0, stream>>>(base, data_half, rank, world,
                                               amax_dev, scale_dev, seq_dev);

  // metadata 区指针 (相对 shm_base 单一基址, 布局见上)
  uint64_t* flag_data =
      reinterpret_cast<uint64_t*>(base + world * data_half + 4 * world +
                                  8 * world);
  uint64_t* flag_barrier =
      reinterpret_cast<uint64_t*>(base + world * data_half + 4 * world +
                                  16 * world);

  for (int64_t offset = 1; offset < world; offset <<= 1) {
    int64_t partner = rank ^ offset;
    uint8_t* own_data = reinterpret_cast<uint8_t*>(base) + rank * data_half;
    uint8_t* peer_data =
        reinterpret_cast<uint8_t*>(base) + partner * data_half;

    // round2: dev quant -> D2H 到 own shm slot + data flag -> H2D partner slot
    //   -> dequant+sum 写回 work -> barrier + bump seq。
    //   (scale 交换已提到循环外, 见上注释)
    ar_quant<<<grid, threads, 0, stream>>>(wp, xqp, scale_dev, n);
    cudaMemcpyAsync(own_data, xqp, (size_t)n, cudaMemcpyDeviceToHost, stream);
    ar_set_spin<<<1, 1, 0, stream>>>(flag_data + rank, flag_data + partner,
                                     seq_dev);
    cudaMemcpyAsync(xqpp, peer_data, (size_t)n, cudaMemcpyHostToDevice, stream);
    ar_dequant_sum<<<grid, threads, 0, stream>>>(xqp, xqpp, wp, scale_dev, n);
    ar_set_spin<<<1, 1, 0, stream>>>(flag_barrier + rank,
                                     flag_barrier + partner, seq_dev);
    ar_bump_seq<<<1, 1, 0, stream>>>(seq_dev);
  }
  // 末轮 work == 全体和 -> 拷入独立 out (不动输入 x)
  cudaMemcpyAsync(outp, wp, (size_t)n * sizeof(__half),
                  cudaMemcpyDeviceToDevice, stream);
}

// host: 把某 IPC buffer 的 metadata 区 (amax/flag_scale/flag_data/barrier)
// 清零。cudaMalloc'd 显存初始是垃圾, flag 初始非 0 → 首 spin 误判「已完成」假
// 命中、dequant 读未写 data 错值。base=buffer 首址; 零区 = [base+world*D,
// base+world*D + 24+16*world)。world=2 时零 56B (= 2 卡 metadata 区, 同 SHM)。
// python init 后对 own + 每个 peer buffer 各调一次, 再 synchronize。
void firefly_ar_zero_meta(int64_t base, int64_t data_half, int64_t world) {
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  char* p = reinterpret_cast<char*>(base) + world * data_half;
  cudaMemsetAsync(p, 0, (size_t)(24 + 16 * world), stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  // NOLINT
  m.def("firefly_ar_exchange", &firefly_ar_exchange,
        "firefly allreduce SHM full-GPU (round1 amax + round2 exchange): "
        "amax + scale-exchange + quant + D2H + flag + H2D + dequant+sum + "
        "barrier + bump-seq, capture-safe");
  m.def("firefly_ar_exchange_pipe", &firefly_ar_exchange_pipe,
        "firefly allreduce SHM full-GPU chunked full-duplex pipeline: same "
        "numerics as firefly_ar_exchange but D2H/H2D split into K chunks "
        "overlapped on 2 streams (per-chunk flags) for full-duplex PCIe "
        "links; needs firefly_ar_init_pipe + per-chunk flag region");
  m.def("firefly_ar_init_pipe", &firefly_ar_init_pipe,
        "create the 2 pipeline streams + 4 events (call once, outside "
        "cudagraph capture) before firefly_ar_exchange_pipe");
  m.def("firefly_ar_destroy_pipe", &firefly_ar_destroy_pipe,
        "destroy pipeline streams + events (call on teardown)");
  m.def("firefly_ar_exchange_p2p", &firefly_ar_exchange_p2p,
        "firefly allreduce P2P full-GPU: amax + scale-exchange + quant + "
        "P2P data flag + dequant(P2P read) + barrier + bump-seq, "
        "capture-safe");
  m.def("firefly_ar_butterfly", &firefly_ar_butterfly,
        "firefly allreduce N-GPU butterfly (world=4/8, P2P/IPC): logN rounds, "
        "each amax-exchange(N) + quant + data-flag + dequant(P2P read) + "
        "barrier + bump-seq; scale=amax*N/448; bases_host=host vector (no "
        "host deref of device array), bases_dev_ptr=kernel arg only; "
        "capture-safe");
  m.def("firefly_ar_butterfly_shm", &firefly_ar_butterfly_shm,
        "firefly allreduce N-GPU butterfly (world=4/8, SHM, no P2P): logN "
        "rounds, each amax-exchange(N, shm) + quant + D2H own slot + "
        "data-flag + H2D partner slot + dequant+sum -> work + barrier + "
        "bump-seq; single shm_base host-mapped region; capture-safe");
  m.def("firefly_ar_zero_meta", &firefly_ar_zero_meta,
        "firefly allreduce: zero an IPC buffer metadata region "
        "(amax/flags/barrier); call on own + each peer after init");
}
