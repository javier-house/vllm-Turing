# SPDX-License-Identifier: Apache-2.0
"""vllm_hc_dequant —— qwen4_exp 主模型 HyperConnection int8 权重 load 时 dequant。

背景: Minachist-...-Mixed-AutoRound checkpoint 把 HC 的 ``input_mix_weight_down/up``
量化成 int8 (W8A16 group-64, 张量 = weight_packed int32 + weight_scale fp16 +
weight_shape int64), 但 ``block_inject_weight`` 是 bf16 纯的。decoder 层
(attn/mlp_hyper_connection, ``use_combine=True``) 把 int8 的 ``down`` 和 bf16 的
``block_inject`` 熔进**同一个** ``MergedColumnParallelLinear``; stock compressed-tensors
要求融合层所有分量同套量化 → bf16 的 ``block_inject`` 破坏假设 → 装不了 (unquant 融合
层只建普通 ``.weight``, checkpoint 却是 ``weight_packed`` → "no parameter ...weight_packed")。

解法: 在 ``Qwen4ExpModel.load_weights`` 的权重流里 (``_HC_WEIGHTS_MAPPER`` 把
``down``/``block_inject`` 堆进融合 linear 的同一处), 把 ``down/up`` 的 int8 三元组
流式 dequant 成 model dtype 的普通 ``.weight`` (跟 ``block_inject`` 一样 fp16/bf16),
所有 HC linear 统一 unquant, 与 checkpoint 原生 (bf16) 加载路径一致; ``GatedResidual``
构造 (``quant_config=None``) 原样不动。``Qwen4ExpModel`` 是语言主干, 必被实例化, 是
唯一干净注入点 (避开顶层 CausalLM/ConditionalGeneration 二选一, 且 MTP 草稿头走独立的
``Qwen4ExpMTP.load_weights``, 其 HC 是纯 bf16, 不受本 patch 影响)。

为何不压 int8/fp8:
  * fp8: sm75 (2080Ti) 无 fp8 计算单元, 只能当存储 + 反量化, 纯亏。
  * int8: decoder 融合层 (int8 down + bf16 block_inject 混合) 绕不开 —— 要么 dequant
    down, 要么运行时量化 block_inject (更复杂), 省不了几个百分点。
  * 显存: 主模型 HC down/up 共 97 module × 2×hc_lowrank×hyper_hidden ≈ 0.64GB int8
    → 1.27GB fp16, 8 卡每卡 160MB (比 int8 多 80MB, 11G 卡可忽略)。
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch

logger = logging.getLogger("vllm.hc_dequant")

_SUFFIX_PACKED = ".weight_packed"
_SUFFIX_SCALE = ".weight_scale"
_SUFFIX_SHAPE = ".weight_shape"
# 只 dequant 主模型 HC 的 down/up (int8); block_inject/hc_norm 是 bf16 原样透传。
_HC_MARKERS = (".input_mix_weight_down.", ".input_mix_weight_up.")


def _dequant_w8a16_group(
    packed: torch.Tensor, scale: torch.Tensor, out: int, inp: int
) -> torch.Tensor:
    """packed int32 [out, inp//4] + scale [out, inp//64] → [out, inp] (scale dtype)。

    对称 int8 W8A16 group-64 (无 zero_point)。解包取低 shift=低列, 逐列对齐本仓库
    已验证可用的 embed_tokens int8 dequant (``_dequant_gather_kernel``):
      ``q = ((packed >> (col%PF)*bits) & mask) - (1<<7)`` (uint8 带 128 偏移,
      ``scalar_types.uint8b128``), 再乘 ``scale[row, col//group]``。**不是** two's
      complement (u-256)——那套约定会把 u=0 当 0 而非 -128, 静默错值。
    """
    p = packed.to(torch.int64)
    shifts = torch.arange(4, device=p.device, dtype=torch.int64) * 8
    u = ((p.unsqueeze(-1) >> shifts) & 0xFF).reshape(out, inp)  # [out, inp] uint8
    q = (u - 128).to(torch.float32)  # offset-128, 对齐 _dequant_gather_kernel
    g = inp // scale.shape[1]  # group size (64)
    q = q.view(out, inp // g, g) * scale.to(torch.float32).unsqueeze(-1)
    return q.reshape(out, inp).to(scale.dtype)


def _dequant_hc_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """流式: 把 HC down/up 的 int8 三元组 (packed+scale+shape) dequant 成单个 ``.weight``,
    其余权重透传。三元组按 name 前缀配对 (packed/scale/shape 同前缀不同后缀, 可乱序到达)。
    """
    pending: dict[str, dict] = {}
    for name, w in weights:
        if name.endswith(_SUFFIX_PACKED) and any(m in name for m in _HC_MARKERS):
            prefix = name[: -len(_SUFFIX_PACKED)]
            pending[prefix] = {"packed": w}
            continue
        if name.endswith(_SUFFIX_SCALE):
            prefix = name[: -len(_SUFFIX_SCALE)]
            if prefix in pending:
                pending[prefix]["scale"] = w
                continue
            yield name, w
            continue
        if name.endswith(_SUFFIX_SHAPE):
            prefix = name[: -len(_SUFFIX_SHAPE)]
            parts = pending.get(prefix)
            if parts and "packed" in parts and "scale" in parts:
                sh = w.tolist()
                deq = _dequant_w8a16_group(
                    parts["packed"], parts["scale"], int(sh[0]), int(sh[1])
                )
                del pending[prefix]
                yield prefix + ".weight", deq
                continue
            yield name, w
            continue
        yield name, w
    # 收尾: 正常情况三元组都齐; 残留说明 checkpoint 结构异常, 打日志让上层报错而非静默丢。
    for prefix, parts in pending.items():
        logger.warning(
            "hc_dequant: 未配对的 HC 张量 %s (已见 %s)", prefix, sorted(parts)
        )


def maybe_apply(model_cls: type) -> None:
    """给 ``Qwen4ExpModel.load_weights`` 包一层 HC dequant (幂等)。

    在类定义之后、实例化之前调用 (由 install_sm75_overlay.py append 到上游
    qwen4_exp/nvidia/model.py 末尾触发)。
    """
    if getattr(model_cls, "_hc_dequant_patched", False):
        return
    orig_load = model_cls.load_weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return orig_load(self, _dequant_hc_weights(weights))

    model_cls.load_weights = load_weights
    model_cls._hc_dequant_patched = True
    logger.info(
        "hc_dequant patch applied to %s.%s", model_cls.__module__, model_cls.__name__
    )


__all__ = ["maybe_apply"]
