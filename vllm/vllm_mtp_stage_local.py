# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vllm_mtp_stage_local —— 让 Qwen4Exp MTP 草稿头在 pipeline 并行下保持 stage-local。

背景: Qwen3.8-Flash-Next (qwen4_exp) 以 MTP 投机解码 + PP>1 时引擎起不来。
末 rank 的 worker 在 AOT 全图编译期就炸:

    torch._dynamo.exc.Unsupported: Data-dependent assertion failed
      (cannot compile partial graph)
        assert intermediate_tensors is not None
        File "vllm/models/qwen4_exp/nvidia/mtp.py", ... in forward

根因: 草稿头 (``Qwen4ExpMultiTokenPredictor``) 是 **stage-local** 的 ——
``gpu_model_runner.execute_model`` 在每个非末 rank 就返回 IntermediateTensors,
投机解码只在末 rank 触发, 所以草稿头只会在末 rank 上跑, 且其权重是**复制**而非
切分 (embed_tokens / fc_embedding / fc_hidden / 每层 MTP 全 rank 都建)。

但它的 ``forward`` 却拿**目标模型**的 pipeline 位置做分支:
  * ``if get_pp_group().is_first_rank:`` 末 rank 上为 False → 走"收上游"分支 →
    ``assert intermediate_tensors is not None`` (没人给它送) → 全图 AOT 编译期
    成硬错误, 引擎起不来。
  * ``if not get_pp_group().is_last_rank:`` 末 rank 上为 False, 单 rank 上两者都
    True, 被删分支本就是 dead path。

修法 (对齐 1Cat 26a406ab): 草稿头 forward 删掉两个 PP 分支 —— 永远本地建 embedding
(去掉 is_first_rank 分支), 永远本地收尾 (去掉 not is_last_rank 分支, 不再向下传
IntermediateTensors)。单 rank 行为不变 (那里 is_first/is_last 都 True)。

做法: ``maybe_apply(cls)`` 给上游 ``Qwen4ExpMultiTokenPredictor`` 打 patch, 用
``_stage_local_forward`` 顶替其 ``forward`` (类属性)。

**对 @support_torch_compile 的编译路径生效的证据** (这是关键, 见汇报):
  * 装饰器 ``_support_torch_compile`` 只替换 ``cls.__init__`` (decorators.py:414)
    和 ``cls.__call__`` (decorators.py:730), **不动 ``cls.forward``**。
  * 被装饰的 ``__init__`` (decorators.py:355-412) 先跑模型原构造 (line 385), 再调
    ``TorchCompileWithNoGuardsWrapper.__init__(self, ...)`` (line 408-412)。
  * 后者在 ``wrapper.py:127`` 用 ``compiled_ptr: Any = self.forward`` 捕获要编译的
    forward, ``wrapper.py:148`` 用 ``torch.compile(compiled_ptr, fullgraph=True)``
    编译。``self.forward`` 是**实例构造时**按 MRO 解析到 ``type(self).forward``
    (类属性) 的 —— 本 hook 在 mtp.py 模块体末尾 (import 期, 任何实例化之前) 已把
    类属性 ``forward`` 换成 ``_stage_local_forward``, 故实例构造时捕获到的是本
    patch, 编译图 / AOT 产物用的都是它。
  * 同理 ``wrapper.py:208`` ``original_code_object`` 也取 ``self.__class__.forward.
    __code__`` (动态解析), 一并指向本 patch。

因此**无需** patch 实例、也无需动 ``Qwen4ExpMTP`` wrapper (其 ``is_last_rank`` 是
正确的模块放置逻辑)。默认开 (纯正确性修复, 非模式开关), 幂等。
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("vllm.mtp_stage_local")

_PATCH_FLAG = "_vllm_mtp_stage_local_patched"


def _stage_local_forward(
    self: Any,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    hidden_states: torch.Tensor | None = None,
    intermediate_tensors: Any | None = None,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """草稿头 forward (stage-local): 删掉两个 PP 分支, 永远本地建 embedding + 收尾。

    参数名/顺序与上游 ``Qwen4ExpMultiTokenPredictor.forward`` 完全一致 (``Qwen4ExpMTP.
    forward`` 按位置传参, ``_mark_dynamic_inputs`` 按名 bind dynamic_arg_dims), 切勿改。
    ``intermediate_tensors`` 保留在签名但**不使用** (对齐 1Cat 26a406ab: 草稿头无人
    送 IntermediateTensors, 删掉 is_first_rank 分支后它本就是 dead 输入)。
    """
    hc_count = self.hc_count
    hidden_size = self.hidden_size

    # 草稿头是完整模型: 永远本地建 embedding 分支 (pre-norm -> fc_embedding)。
    assert hidden_states is not None
    if inputs_embeds is None:
        assert input_ids is not None
        inputs_embeds = self.embed_input_ids(input_ids)
    inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
    inputs_embeds = self.fc_embedding(inputs_embeds)

    # 主干 hidden 是多流 [T, hc_count*H] (scheme A: 首步由主模型真正给出 pre-
    # final-mixer 多流, 后续步复用上一草稿步的多流)。
    num_tokens = hidden_states.shape[0]
    hidden_states = hidden_states.view(num_tokens, hc_count, hidden_size)
    hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(
        num_tokens, hc_count, hidden_size
    )
    hidden_states = self.fc_hidden(hidden_states)
    # 把 embedding 残差加到每个分支, 再折回 [T, hc_count*H] (HC 外层, HS 内层)。
    hidden_states = inputs_embeds.unsqueeze(-2) + hidden_states
    hidden_states = hidden_states.flatten(-2)

    current_step_idx = spec_step_idx % self.num_mtp_layers
    layer = self.layers[current_step_idx]
    hidden_states, block_output, injection = layer(
        hidden_states=hidden_states,
        prev_block_output=None,
        prev_injection=None,
        positions=positions,
        input_ids=None,
        query_start_loc=None,
        ngram_context=None,
    )

    # 草稿头 stage-local: 永远本地收尾 (不向下传 IntermediateTensors)。同时保留:
    #   (A) sample_hidden_states [T, H]  -> 单流, 喂 LM head
    #   (B) multi_hidden [T, hc_count*H] -> pre-final-mixer 多流, 喂下一草稿步
    multi_hidden, sample_hidden_states, _ = self.hyper_connection_mixer.combine_and_mix(
        hidden_states, block_output, injection
    )
    return sample_hidden_states, multi_hidden


def maybe_apply(cls: type) -> None:
    """给 ``Qwen4ExpMultiTokenPredictor`` 打 stage-local patch (幂等)。

    默认开 (纯正确性修复, 非 opt-in 模式: PP>1 + MTP 否则根本起不来; 单 rank
    行为不变)。在类定义之后、实例化之前调用 (由 install_sm75_overlay.py append 到
    上游 mtp.py 末尾触发) —— 此时 ``@support_torch_compile`` 已包装过该类的
    ``__init__``/``__call__`` 但 ``forward`` 仍是类属性, 换掉后实例构造期捕获的
    编译目标就是本 patch (见模块 docstring 的证据)。
    """
    # 防误 patch: 只认草稿头类, 且已 patch 过则跳过 (幂等)。
    if cls.__name__ != "Qwen4ExpMultiTokenPredictor":
        logger.warning(
            "mtp_stage_local: 期望 patch Qwen4ExpMultiTokenPredictor, 实得 %s, 跳过",
            cls.__name__,
        )
        return
    if getattr(cls, _PATCH_FLAG, False):
        return
    orig = cls.forward
    cls.forward = _stage_local_forward
    setattr(cls, _PATCH_FLAG, True)
    logger.info(
        "mtp_stage_local: 已 patch %s.%s (删 PP 分支, stage-local); 原 forward=%r",
        cls.__module__,
        cls.__name__,
        getattr(orig, "__qualname__", orig),
    )


__all__ = ["maybe_apply"]
