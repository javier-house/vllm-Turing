# SPDX-License-Identifier: Apache-2.0
"""vllm_firefly_hc —— qwen4_exp HyperConnection 投影的 SM75 decode 标量 GEMV。

背景: HC(GatedResidual) 每层跑三个 skinny 投影 (down+inject merged / down / up),
decode(M=1) 时都是「单行输入 × 瘦长权重」的 GEMV。走 cuBLAS 的通用 GEMM 路径对
这种 M=1 形状要么退化成多 kernel、要么被 CuBLAS heuristic 选到对 skinny 不友好的
tile, kernel launch / 同步固定成本主导 (Minachist decode-04 hc-fused-int8: 6 kernel
→ 2, 32us→13us)。本模块把这三个投影在 **单流 decode (M==1)** 时换成一个融合标量
GEMV triton kernel (逐行 w·x 归约, **不用 tl.dot** —— sm75 无 int8/fp8 tensor core,
标量归约即可, 与 firefly 其余标量 kernel 同路), 显著砍 HC 段 launch 开销。

设计要点 (比上游 0924 蓝本更保守):
  * 只在 M==1 单流 decode 命中自定义 kernel; prefill / batch>1 / 非 sm75 / 非 fp16
    一律回退**快照下来的原 _gemm_impl** (cuBLAS dispatch) —— prefill 数值逐 bit 不变。
  * 用 ``@torch.library.custom_op`` 把 kernel 包成 opaque op, prefill 的大 M 分支
    不会进入 torch.compile tracing (否则 inductor 会去 trace 我们的 triton 分支)。
  * 只装在 HC 三投影 (input_mix_weight_down_block_inject / _down / _up) 且
    quant_method 是 UnquantizedLinearMethod 的层上 —— 绝不全局装到 Linear。

与 vllm_hc_dequant 的关系: dequant 把 HC int8 权重 load 时反量化成 fp16 普通
``.weight`` (统一 unquant), 本模块在这份 fp16 权重上跑 GEMV —— **P2a 吃的是 fp16**
(kernel 融合是 decode 提速主力, HC 总权重仅 ~0.64GB, int8 省带宽是次要增量)。
P2b 再评估让 GEMV 直读 pack-quantized int8 的增量收益 (需拆 merged 层的
down/block_inject 段 + ppl 回归)。

注入: ``maybe_apply(Qwen4ExpModel)`` 包 ``__init__``, 实例化后遍历 named_modules,
给 HC 三投影的 quant_method 换 ``_gemm_impl``。由 install_sm75_overlay.py append 到
上游 qwen4_exp/nvidia/model.py 末尾触发 (与 hc_dequant 的 load_weights patch 链式共存)。
env ``VLLM_FIREFLY_HC`` 默认关 (见 envs_sm75.py): 关 = 完全不改 (走上游 cuBLAS)。
"""

from __future__ import annotations

import logging
import os

# torch 必须在模块作用域 import: 本文件用 `from __future__ import annotations`,
# custom_op ``hc_gemv`` 的注解 (torch.Tensor) 是惰性字符串, vllm 的 infer_schema 运行时
# 要在模块 globals 解析它; 若只在 _get_hc_gemv() 函数内 import torch, 解析时
# `name 'torch' is not defined` → 启动崩。triton 仍保持函数内懒加载。
import torch

logger = logging.getLogger("vllm.firefly_hc")

# HC 三投影模块名 (对齐底座 GatedResidual 属性名)。
_HC_PROJ_NAMES = (
    "input_mix_weight_down_block_inject",  # merged: [lora_rank, hc_count(+pad)] × D
    "input_mix_weight_down",               # 非 combine 层: lora_rank × D
    "input_mix_weight_up",                 # D × lora_rank
)


def _enabled() -> bool:
    return os.getenv("VLLM_FIREFLY_HC", "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


# --- triton 标量 GEMV (延迟 import torch/triton: 仅在装了且命中时构造) ---------- #
_kernel_mod = None


def _get_hc_gemv():
    """延迟构造 custom_op + triton kernel (避免 import 本模块就强依赖 triton/torch)。"""
    global _kernel_mod
    if _kernel_mod is not None:
        return _kernel_mod

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _row_gemv(
        X, W, Y,
        N: tl.constexpr, K: tl.constexpr,
        R: tl.constexpr, BK: tl.constexpr,
    ):
        # 一个 program 算 R 行: y[row] = sum_k W[row,k] * X[k] (M=1, x 全载入)。
        rows = tl.program_id(0) * R + tl.arange(0, R)
        cols = tl.arange(0, BK)
        x = tl.load(X + cols, cols < K, 0).to(tl.float32)
        w = tl.load(
            W + rows[:, None] * K + cols[None, :],
            (rows[:, None] < N) & (cols[None, :] < K),
            0,
        ).to(tl.float32)
        values = tl.sum(w * x[None, :], axis=1)
        tl.store(Y + rows, values, rows < N)

    @torch.library.custom_op("firefly_hc::gemv", mutates_args=())
    def hc_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # 调用方已保证: cuda/fp16/contig、x.numel()==K (M=1)、weight 为 HC 三形状、
        # sm75。这里只做 kernel 本身, 不再回退 (回退在 _dispatch 里用原 gemm)。
        n, k = weight.shape
        output = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
        rows, warps = (4, 4) if k == 320 else (1, 8)
        _row_gemv[(triton.cdiv(n, rows),)](
            x, weight, output, n, k, rows, triton.next_power_of_2(k), num_warps=warps
        )
        return output

    @hc_gemv.register_fake
    def _fake(x, weight):
        return x.new_empty((*x.shape[:-1], weight.shape[0]))

    _kernel_mod = hc_gemv
    return hc_gemv


def _make_dispatch(orig_gemm, is_sm75: bool):
    """为单个 HC 层造 dispatch: M==1 命中 GEMV, 否则回退原 cuBLAS gemm。"""
    import torch
    import torch.nn.functional as F

    def _dispatch(layer, x, weight, bias=None):
        if bias is not None or not is_sm75:
            return orig_gemm(layer, x, weight, bias)
        # 仅单流 decode (x 是一个 token, numel == K == weight.shape[1]) 走 GEMV。
        if x.dtype != torch.float16 or not x.is_contiguous() or not weight.is_contiguous():
            return orig_gemm(layer, x, weight, bias)
        k = weight.shape[1]
        if x.numel() != k:
            return orig_gemm(layer, x, weight, bias)
        # prefill (M>1) → x.numel() > k; 命中要求 numel==k, 即 M==1。
        flat = x.reshape(-1, k)[0] if x.dim() >= 2 else x
        return _hc_gemv_dispatch(flat, weight, x)

    def _hc_gemv_dispatch(flat_x, weight, orig_shape_x):
        hc_gemv = _get_hc_gemv()
        out = hc_gemv(flat_x.contiguous(), weight)
        # 还原非 GEMV 输入维度形状 (x 可能是 [.., K], 输出 [..., N])。
        return out.reshape(*orig_shape_x.shape[:-1], weight.shape[0])

    # torch 仅用于类型检查, F 备用 (避免未用告警)。
    _ = F
    return _dispatch


def _install_on_module(module) -> int:
    """给一个 GatedResidual 实例的 HC 三投影装 GEMV, 返回装成功的层数。"""
    import torch
    from vllm import envs
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    # batch-invariant 会在顶层 bypass _gemm_impl (走 linear_batch_invariant), 我们的
    # 装法失效且可能与用户显式选择冲突 → 直接拒绝, 让用户清楚二选一。
    if envs.VLLM_BATCH_INVARIANT:
        raise RuntimeError(
            "VLLM_FIREFLY_HC 与 VLLM_BATCH_INVARIANT 冲突 (后者绕过 _gemm_impl), "
            "请关闭其中之一"
        )

    is_sm75 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 5)
    installed = 0
    for attr in _HC_PROJ_NAMES:
        layer = getattr(module, attr, None)
        if layer is None:
            continue
        method = getattr(layer, "quant_method", None)
        if type(method) is not UnquantizedLinearMethod:
            # 不认识的量化方法 (理论上 HC 是 quant_config=None → unquant): 跳过保持原路。
            continue
        if getattr(method, "_firefly_hc_patched", False):
            continue
        method._firefly_hc_orig_gemm = method._gemm_impl
        method._gemm_impl = _make_dispatch(method._gemm_impl, is_sm75)
        method._firefly_hc_patched = True
        installed += 1
    return installed


def _install_after_init(self) -> None:
    """__init__ 后遍历 named_modules, 对所有 GatedResidual 装 HC GEMV。"""
    total = 0
    for _name, mod in self.named_modules():
        # 用 duck-type: 同时具备三个 HC 投影属性之一 + hc_count 的即 GatedResidual。
        if not hasattr(mod, "input_mix_weight_up") or not hasattr(mod, "hc_count"):
            continue
        total += _install_on_module(mod)
    if total:
        logger.info("firefly_hc: 已给 %d 个 HC 投影装 SM75 标量 GEMV", total)


def maybe_apply(model_cls: type) -> None:
    """给 ``Qwen4ExpModel.__init__`` 包一层 HC GEMV 安装 (幂等)。

    由 install_sm75_overlay.py append 到上游 qwen4_exp/nvidia/model.py 末尾触发,
    在类定义之后、实例化之前调用。env ``VLLM_FIREFLY_HC`` 关 → 不装 (上游原路)。
    """
    if not _enabled():
        logger.info("firefly_hc: VLLM_FIREFLY_HC 未开, 跳过 (HC 走上游 cuBLAS)")
        return
    if getattr(model_cls, "_firefly_hc_patched", False):
        return
    orig_init = model_cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            _install_after_init(self)
        except RuntimeError:
            raise
        except Exception:  # 装失败退回上游路径, 不让增强拖垮启动
            logger.exception("firefly_hc: HC GEMV 安装失败, 回退上游 cuBLAS")

    model_cls.__init__ = __init__
    model_cls._firefly_hc_patched = True
    logger.info("firefly_hc patch applied to %s.%s", model_cls.__module__, model_cls.__name__)


__all__ = ["maybe_apply"]
