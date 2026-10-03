# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 overlay: NVFP4 Marlin (W4A16) + firefly(int8 prefill) 混合。

以 v0.30.0 上游 nvfp4/marlin.py 为基线, 叠加 firefly 分支 (VLLM_FIREFLY_DIRECT
门控, off=上游行为):
  - process_weights_after_loading: 先快照原始 NVFP4 布局 (uint8 权重 + fp8 group
    scale + fp32 global scale) + 预计算 per-channel c_n, 再走上游 Marlin repack
    (prepare_fp4_layer_for_marlin 会覆盖 layer.weight 为 Marlin int32, 故须先存)。
  - apply_weights: 大 M (prefill) 把 NVFP4 权重现反量化 int8 走 cutlass_scaled_mm
    (SM75 即 IMMA tensor core, 2x fp16 TC 吞吐); 小 M (decode) 走上游 NVFP4 Marlin。

NVFP4 比 int4 简单: 权重是 uint8 行主序 (2 nibble/byte), 无 Marlin 布局要解析,
反量化 kernel 比 firefly.cu (寄存器现算 marlin 位置) 轻。
见 vllm/model_executor/layers/quantization/utils/firefly_nvfp4.py。
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.firefly_nvfp4 import (
    nvfp4_prefill_linear,
    nvfp4_to_int8,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)

import vllm.envs as envs

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


# ---- firefly 混合 op (dynamo/inductor 兼容) ----
# 与 mixed_precision/marlin.py 同款: "门控 + firefly/Marlin 选择 + 执行" 整体包成
# torch.library custom op。firefly 分支判据 (m > min_m, m 是动态 shape) 若留在编译
# 图里, inductor 会把 firefly 路径 (含 custom op) 编进 decode (M=1) 的图, 实测
# decode 性能掉; 图里只留一个不透明 leaf, decode 图干净 (= baseline 性能)。
# op 内部按真实 M 求值: prefill (m>min_m) 现反量化 int8 走 IMMA, decode 走 Marlin。
@torch.library.custom_op("firefly_nvfp4::hybrid_linear", mutates_args=())
def _ff_nvfp4_hybrid_linear(
    x: torch.Tensor,
    ff_weight: torch.Tensor,  # uint8 [N, K/2] 原始 NVFP4
    ff_group_scale: torch.Tensor,  # fp8_e4m3 [N, K/16]
    ff_global_scale: torch.Tensor,  # fp32 [1]
    ff_c_n: torch.Tensor,  # fp32 [N] 预计算 (numel==0 则现算)
    marlin_weight: torch.Tensor,  # Marlin repack int32
    marlin_weight_scale: torch.Tensor,
    marlin_weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor,
    min_m: int,
) -> torch.Tensor:
    m = x.numel() // x.shape[-1]
    if m > min_m:
        # prefill: NVFP4 权重现反量化 int8 + cutlass_scaled_mm (SM75 即 IMMA)
        # global_scale 传设备张量 (kernel 读 [0]), 不 .item() (避免 custom op 内 D2H sync)。
        return nvfp4_prefill_linear(
            x,
            ff_weight,
            ff_group_scale,
            ff_global_scale,
            c_n=ff_c_n if ff_c_n.numel() > 0 else None,
        )
    # decode: 上游 NVFP4 Marlin (W4A16, E2M1→fp16→fp16 TC)
    return apply_fp4_marlin_linear(
        input=x,
        weight=marlin_weight,
        weight_scale=marlin_weight_scale,
        weight_global_scale=marlin_weight_global_scale,
        workspace=workspace,
        size_n=size_n,
        size_k=size_k,
        bias=bias if bias.numel() > 0 else None,
        input_dtype=x.dtype,
    )


@_ff_nvfp4_hybrid_linear.register_fake
def _(
    x, _ff_w, _ff_gs, _ff_gscale, _ff_cn, _m_w, _m_ws, _m_wgs, _ws,
    size_n, _size_k, _bias, _min_m,
):
    # 输出 [*, size_n], dtype = 激活 dtype (与真实路径一致)。
    return torch.empty(x.shape[:-1] + (size_n,), dtype=x.dtype, device=x.device)


class MarlinNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 weight-only GEMM via Marlin (W4A16) + firefly int8 prefill (SM75)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # firefly: 先快照原始 NVFP4 布局 + 预计算 per-channel c_n。
        # prepare_fp4_layer_for_marlin 会把 layer.weight 覆盖成 Marlin int32 布局,
        # 故必须在 repack 之前存下原始 uint8 权重 + fp8 group scale + fp32 global。
        if envs.VLLM_FIREFLY_DIRECT:
            try:
                N = layer.output_size_per_partition
                K = layer.input_size_per_partition
                ff_w = layer.weight.detach().view(torch.uint8).contiguous()  # [N, K/2]
                # fp8_e4m3 → raw byte (bit 重解释, 非数值 cast; .to 会把浮点值 round 成 int 毁字节)
                ff_gs = layer.weight_scale.detach().view(torch.uint8).contiguous()  # [N, K/16]
                ff_gscale = layer.weight_global_scale.detach().to(torch.float32)  # [1]
                layer._ff_w = ff_w
                layer._ff_gs = ff_gs
                layer._ff_gscale = ff_gscale
                # 预计算 c_n (两遍 amax+量化, 仅 load 时一次); 运行期走单遍。
                # 无 GPU/编译失败时 nvfp4_to_int8 回退 PyTorch (慢, 但 load 时一次性)。
                try:
                    _, cn = nvfp4_to_int8(ff_w, ff_gs, ff_gscale)
                    layer._ff_cn = cn.detach().contiguous()
                except Exception:  # noqa: BLE001 - c_n 预计算失败, 运行期现算
                    layer._ff_cn = torch.empty(0, dtype=torch.float32, device=ff_w.device)
                logger.info_once(
                    "firefly NVFP4: 快照原始 FP4 布局 + 预计算 c_n (prefill int8 加速)"
                )
            except Exception as e:  # noqa: BLE001 - 快照失败回退纯 Marlin
                logger.warning(
                    "firefly NVFP4 快照失败, 该层回退纯 Marlin: %s", e
                )
                for attr in ("_ff_w", "_ff_gs", "_ff_gscale", "_ff_cn"):
                    if hasattr(layer, attr):
                        delattr(layer, attr)

        # 上游: Marlin repack (E2M1→int32 + scale 转 S0E5M3 + global scale 校正)。
        prepare_fp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # firefly 混合: 有快照 (VLLM_FIREFLY_DIRECT 开) 且激活是 fp16/bf16 时,
        # 走 custom op (op 内部按 M 分 prefill-int8 / decode-Marlin)。
        if (
            envs.VLLM_FIREFLY_DIRECT
            and hasattr(layer, "_ff_w")
            and x.dtype in (torch.float16, torch.bfloat16)
        ):
            bias_t = (
                bias if bias is not None
                else torch.empty(0, dtype=x.dtype, device=x.device)
            )
            return _ff_nvfp4_hybrid_linear(
                x,
                layer._ff_w,
                layer._ff_gs,
                layer._ff_gscale,
                layer._ff_cn,
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                layer.workspace,
                layer.output_size_per_partition,
                layer.input_size_per_partition,
                bias_t,
                envs.VLLM_FIREFLY_MIN_M,
            )
        # 非 firefly (off / 无快照 / 非 fp16 激活): 上游 NVFP4 Marlin。
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
