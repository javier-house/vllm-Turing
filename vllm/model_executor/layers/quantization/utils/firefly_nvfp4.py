# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""firefly NVFP4: E2M1 (4-bit FP4) 权重 → int8 prefill 加速 (SM75)。

与 firefly.py (int4→int8) 同构的两段式:
  1) nvfp4 权重 (uint8 行主序 packed E2M1 + fp8 group scale + fp32 global)
     现反量化成 int8 [N, K] 行主序 + per-channel c_n[N] (firefly_nvfp4.cu);
  2) int8 走 cutlass_scaled_mm (SM75 即 IMMA tensor core, 2x fp16 TC 吞吐),
     复用 firefly.py 的 int8_prefill_linear。

decode 小 M 不触发 firefly, 走上游 NVFP4 Marlin (W4A16, E2M1→fp16→fp16 TC)。

NVFP4 权重比 int4 简单: 简单 uint8 行主序 (2 nibble/byte), 无 Marlin 布局
要解析, 反量化 kernel 比 firefly.cu (寄存器现算 marlin 位置) 轻。

数学:
    w_deq[n,k]  = E2M1(nibble[n,k]) * group_scale[n, k//16] * global_scale
    c_n[n]      = amax_k |w_deq[n,k]| / 127          # per-channel 正 scale
    w_int8[n,k] = clamp(round(w_deq[n,k] / c_n[n]), -127, 127)
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# NVFP4 CUDA kernel (独立 torch extension, 首次用时编译)。
_cuda_mod = None
_cuda_load_attempted = False


def _load_cuda_mod():
    """懒加载 firefly_nvfp4.cu, 失败返回 None。

    只在首次调用时编译 (缓存到 torch extensions 目录), 不拖慢 vllm import。
    只编 sm_75 (本 overlay 面向 SM75)。
    """
    global _cuda_mod, _cuda_load_attempted
    if _cuda_load_attempted:
        return _cuda_mod
    _cuda_load_attempted = True
    try:
        cu_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "firefly_nvfp4.cu"
        )
        if not os.path.exists(cu_path):
            logger.warning(
                "firefly_nvfp4 kernel source (firefly_nvfp4.cu) not found at %s; "
                "NVFP4 firefly prefill 不可用, 回退 Marlin",
                cu_path,
            )
            return None
        from torch.utils.cpp_extension import load as _load_ext

        # 只编 sm_75 (T10)。强制单 arch, 首次 prefill JIT 编译 ~2s
        # (而非基础镜像多 arch 默认的 ~50s)。
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
        _cuda_mod = _load_ext(
            name="firefly_nvfp4_cuda",
            sources=[cu_path],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        logger.info("firefly_nvfp4 CUDA kernel loaded (NVFP4→int8 fast path)")
    except Exception as e:  # noqa: BLE001 - 编译/无 CUDA 环境回退
        logger.warning(
            "firefly_nvfp4 CUDA ext load failed: %s", e
        )
        _cuda_mod = None
    return _cuda_mod


def nvfp4_to_int8(
    weight: torch.Tensor,  # uint8 [N, K/2] packed E2M1
    group_scale: torch.Tensor,  # fp8_e4m3 [N, K/16]
    global_scale: torch.Tensor,  # fp32 [1] (设备张量, 免 D2H sync)
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 权重 → (w_int8 [N, K] 行主序, c_n [N] per-channel scale)。

    优先 CUDA kernel (firefly_nvfp4.cu, global_scale 传设备指针); 编译失败/
    无 GPU 回退 PyTorch 版 (正确但慢, 仅作兜底, 此处才 .item() 取标量)。
    c_n 每次现算 (两遍 amax+量化)。
    """
    N, half_k = weight.shape
    K = half_k * 2
    mod = _load_cuda_mod()
    if mod is not None and weight.is_cuda:
        out_int8 = torch.empty((N, K), dtype=torch.int8, device=weight.device)
        c_n = torch.empty((N,), dtype=torch.float32, device=weight.device)
        mod.nvfp4_to_int8(
            weight, group_scale, global_scale, out_int8, c_n, N, K
        )
        return out_int8, c_n

    # PyTorch 回退 (CPU/无 CUDA): 解 E2M1 → 乘 scale → per-channel 量化。
    # 与 CUDA 版数学一致 (E2M1 LUT + fp8 group scale + fp32 global)。
    gs_val = float(global_scale)  # 仅回退路径取标量 (不常走, sync 可接受)
    w = weight.view(torch.uint8)  # [N, K/2]
    lo = w & 0x0F  # uint8 低 nibble (k 偶)
    hi = (w >> 4) & 0x0F  # uint8 高 nibble (k 奇)
    # E2M1 解码: 在 uint8 上做位拆 (sign=bit3, mag=bits[2:0]), LUT 取幅值,
    # sign 翻号。mag 0..7 直接对应 LUT 下标 (0→0, 1→0.5, 2→1, 3→1.5, 4→2, 5→3, 6→4, 7→6)。
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=weight.device, dtype=torch.float32,
    )
    mag_lo = (lo & 7).long()
    mag_hi = (hi & 7).long()
    sign_lo = ((lo >> 3) & 1).to(torch.float32)
    sign_hi = ((hi >> 3) & 1).to(torch.float32)
    val_lo = lut[mag_lo] * (1.0 - 2.0 * sign_lo)
    val_hi = lut[mag_hi] * (1.0 - 2.0 * sign_hi)
    # interleave: k 偶 = lo, k 奇 = hi → [N, K]
    deq_e2m1 = torch.stack((val_lo, val_hi), dim=-1).reshape(N, K)
    # group scale: 调用方传 uint8 (raw fp8 字节), 先 view 回 fp8 再转 float。
    gs_src = group_scale
    if gs_src.dtype == torch.uint8:
        gs_src = gs_src.view(torch.float8_e4m3fn)
    gs = gs_src.to(torch.float32).repeat_interleave(16, dim=1)  # [N, K]
    w_deq = deq_e2m1 * gs * gs_val  # [N, K]
    c_n = w_deq.abs().amax(dim=1) / 127.0  # [N]
    w_int8 = torch.clamp(
        torch.round(w_deq / c_n.unsqueeze(1)), -127, 127
    ).to(torch.int8)
    return w_int8, c_n


def nvfp4_to_int8_cached(
    weight: torch.Tensor,  # uint8 [N, K/2]
    group_scale: torch.Tensor,  # fp8_e4m3 [N, K/16]
    global_scale: torch.Tensor,  # fp32 [1] 设备张量
    c_n: torch.Tensor,  # [N] load 时预计算的 per-channel scale
) -> torch.Tensor:
    """单遍反量化: c_n 已预计算 (免 amax pass, 省 ~46% 反量化开销), 只跑量化。

    优先 CUDA kernel (firefly_nvfp4.cu 的 nvfp4_to_int8_cached, global_scale 传
    设备指针); 编译失败/无 GPU 回退 PyTorch 版 (乘倒数, off-by-one ≤0.06%)。
    返回 w_int8 [N, K] int8。
    """
    N, half_k = weight.shape
    K = half_k * 2
    mod = _load_cuda_mod()
    if mod is not None and weight.is_cuda:
        out_int8 = torch.empty((N, K), dtype=torch.int8, device=weight.device)
        mod.nvfp4_to_int8_cached(
            weight, group_scale, global_scale, out_int8, c_n, N, K
        )
        return out_int8

    # PyTorch 回退: 解 E2M1 → 乘 scale → 用预存 c_n 量化 (乘倒数)。
    gs_val = float(global_scale)  # 仅回退路径取标量
    w = weight.view(torch.uint8)
    lo = w & 0x0F  # uint8 (位拆须在整型上做, 不能在 float 上)
    hi = (w >> 4) & 0x0F
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=weight.device, dtype=torch.float32,
    )
    val_lo = lut[(lo & 7).long()] * (1.0 - 2.0 * (((lo >> 3) & 1).to(torch.float32)))
    val_hi = lut[(hi & 7).long()] * (1.0 - 2.0 * (((hi >> 3) & 1).to(torch.float32)))
    deq_e2m1 = torch.stack((val_lo, val_hi), dim=-1).reshape(N, K)
    gs_src = group_scale
    if gs_src.dtype == torch.uint8:
        gs_src = gs_src.view(torch.float8_e4m3fn)
    gs = gs_src.to(torch.float32).repeat_interleave(16, dim=1)
    w_deq = deq_e2m1 * gs * gs_val
    r = torch.where(c_n > 0, 1.0 / c_n, 0.0)
    return torch.clamp(torch.round(w_deq * r.unsqueeze(1)), -127, 127).to(torch.int8)


def nvfp4_prefill_linear(
    x: torch.Tensor,  # [*, K] fp16/bf16 激活
    weight: torch.Tensor,  # uint8 [N, K/2]
    group_scale: torch.Tensor,  # fp8_e4m3 [N, K/16]
    global_scale: torch.Tensor,  # fp32 [1] 设备张量 (免 D2H sync)
    c_n: torch.Tensor | None = None,  # [N] 预计算 (load 时); None 则现算 (两遍)
) -> torch.Tensor:
    """NVFP4 firefly prefill: 现反量化 int8 → cutlass_scaled_mm → [*, N]。

    复用 firefly.py 的 int8_prefill_linear (per-token 动态 int8 激活 + IMMA)。
    c_n 有则走单遍 (快), 无则两遍现算。
    """
    from vllm.model_executor.layers.quantization.utils.firefly import (
        int8_prefill_linear,
    )

    if c_n is not None:
        w_int8 = nvfp4_to_int8_cached(weight, group_scale, global_scale, c_n)
    else:
        w_int8, c_n = nvfp4_to_int8(weight, group_scale, global_scale)
    return int8_prefill_linear(x, w_int8, c_n)
