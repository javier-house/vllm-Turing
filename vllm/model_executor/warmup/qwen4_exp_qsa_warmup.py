# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 移植上游 #54873 (31a8a266): 把已加载的 QSA 模块接到 kernel 侧 warmup ——
# ① indexer decode kernel 的全部可达 specialization (按 decode_query_len ×
# num_requests 的 TILES_PER_PROG 分档, 见 qsa_indexer.warmup_qsa_mqa_paged_decode);
# ② sparse attention 的 split-K/merge 配置。
# 差异: ② 依赖我方 overlay ops/qsa.py 的 warmup_qsa_sparse_paged_attention,
# 但该函数上游 #54873 才进 ops/qsa.py, 我方 overlay 的 ops/qsa.py 是更早的
# 移植态 (无此函数, 且本次移植裁决不动它) → ② 用 try/except ImportError
# 兜底: 缺失时打 warning 并跳过, 不影响 ① (indexer decode warmup 是本移植
# 的重点 —— 新 kernel 首次真实 decode 前必须编好)。
"""Connect loaded Qwen4Exp QSA modules to their kernel-owned warmup."""

import sys
from typing import TYPE_CHECKING, cast

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def qwen4_exp_qsa_triton_warmup(worker: "Worker") -> None:
    """预热全部可达 QSA specialization: indexer decode-query-length 档位
    + sparse attention 的 split-K/merge 配置 (后者在 overlay ops/qsa.py
    缺 warmup 入口时跳过)。"""

    qsa_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.indexer_qsa")
    attn_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.qsa")
    if qsa_module is None or attn_module is None:
        return
    indexer = None
    owner = None
    for layer in worker.get_model().modules():
        if indexer is None and isinstance(layer, qsa_module.QSAIndexer):
            indexer = layer
        elif owner is None and isinstance(layer, attn_module.Qwen4ExpQSAAttention):
            owner = layer
    if indexer is None or owner is None:
        return

    runner = worker.model_runner

    def block_table_for(prefix: str) -> torch.Tensor:
        group_id = next(
            i
            for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
            if prefix in group.layer_names
        )
        if worker.use_v2_model_runner:
            runner_v2 = cast("GPUModelRunnerV2", runner)
            return runner_v2.block_tables.input_block_tables[group_id]
        return runner.input_batch.block_table[group_id].get_device_tensor(
            runner.max_num_reqs
        )

    if worker.use_v2_model_runner:
        max_decode_query_len = cast("GPUModelRunnerV2", runner).decode_query_len
    else:
        max_decode_query_len = runner.uniform_decode_query_len

    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
        warmup_qsa_mqa_paged_decode,
    )

    k_cache = indexer.compressed_key_cache.kv_cache
    assert k_cache.numel()
    profiles = warmup_qsa_mqa_paged_decode(
        k_cache,
        block_table_for(indexer.compressed_key_cache.prefix),
        num_heads=indexer.index_n_heads,
        head_dim=indexer.index_head_dim,
        max_decode_query_len=max_decode_query_len,
        max_num_reqs=runner.max_num_reqs,
        max_num_batched_tokens=runner.max_num_tokens,
    )
    logger.info("Warmed up Qwen4Exp QSA decode kernels: %s.", profiles)

    # ② sparse attention warmup: 我方 overlay ops/qsa.py 尚无
    # warmup_qsa_sparse_paged_attention (上游 #54873 才引入, 本次裁决不动
    # 该文件) → 缺失时跳过, 不阻塞 ①。
    try:
        from vllm.models.qwen4_exp.nvidia.ops.qsa import (
            warmup_qsa_sparse_paged_attention,
        )
    except ImportError:
        logger.warning(
            "Qwen4Exp QSA sparse attention warmup skipped: overlay "
            "ops/qsa.py lacks warmup_qsa_sparse_paged_attention."
        )
        return

    kv_cache = owner.kv_cache
    assert kv_cache.numel()
    attention_profiles = warmup_qsa_sparse_paged_attention(
        kv_cache,
        block_table_for(owner.layer_name),
        num_query_heads=owner.num_heads,
        selection_width=indexer.output_width,
    )
    logger.info(
        "Warmed up Qwen4Exp QSA sparse attention kernels: %s.",
        attention_profiles,
    )
