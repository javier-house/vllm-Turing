# SPDX-License-Identifier: Apache-2.0
"""Install the reviewed SM75 overlay into the base image's vLLM package."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# 锚点注入表: 对上游有同名、我方只改 1~2 行的文件, 不整文件覆盖, 改为按稳定
# 锚点注入(见 apply_injections)。每条 = (相对路径, [(锚点, 替换), ...])。
# 锚点取上游正文中唯一出现的稳定行(升级后仍易识别), 替换写回 SM75 改动后的行;
# 用 list-of-(anchor, replace) 而非单个锚点, 以支持同一文件多处注入(如 kv_cache
# 的两处 logger 降级、metrics 的豁免列表 + 尾 hook)。
# 每条 (anchor, replacement, probe): 用 anchor 定位、replace 成 replacement;
# probe 是"注入后才出现且唯一"的片段, 用于幂等判断(已注入则跳过)。不能用
# anchor 判断幂等——部分 replacement 保留了 anchor 本身(如 model.py 的函数插入
# 保留注释行), 第二次跑 anchor 仍在会重复注入。probe 必须只在注入后存在。
INJECTIONS: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "model_executor/layers/quantization/utils/marlin_utils_fp8.py",
        [(
            # 上游同一条 warning 文本出现 2 次(prepare_fp8_layer_for_marlin 与另一函数),
            # 只改 prepare_fp8_layer_for_marlin 那处: 用签名尾 ') -> None:' 定位使其唯一。
            '    input_dtype: torch.dtype | None = None,\n'
            ') -> None:\n'
            '    logger.warning_once(\n'
            '        "Your GPU does not have native support for FP8 computation but "',
            '    input_dtype: torch.dtype | None = None,\n'
            ') -> None:\n'
            '    logger.info_once(\n'
            '        "Your GPU does not have native support for FP8 computation but "',
            ') -> None:\n'
            '    logger.info_once(\n'
            '        "Your GPU does not have native support for FP8 computation but "',
        )],
    ),
    (
        "distributed/kv_transfer/kv_connector/v1/base.py",
        [(
            '        logger.warning(\n'
            '            "Initializing KVConnectorBase_V1. This API is experimental and "',
            '        logger.info(\n'
            '            "Initializing KVConnectorBase_V1. This API is experimental and "',
            '        logger.info(\n'
            '            "Initializing KVConnectorBase_V1. This API is experimental and "',
        )],
    ),
    (
        "model_executor/layers/quantization/kv_cache.py",
        [(
            '                logger.warning_once(\n'
            '                    "Checkpoint does not provide a q scaling factor. "',
            '                logger.info_once(\n'
            '                    "Checkpoint does not provide a q scaling factor. "',
            '                logger.info_once(\n'
            '                    "Checkpoint does not provide a q scaling factor. "',
        ), (
            '                logger.warning_once(\n'
            '                    "Using KV cache scaling factor 1.0 for fp8_e4m3. "',
            '                logger.info_once(\n'
            '                    "Using KV cache scaling factor 1.0 for fp8_e4m3. "',
            '                logger.info_once(\n'
            '                    "Using KV cache scaling factor 1.0 for fp8_e4m3. "',
        )],
    ),
    (
        "config/model.py",
        [(
            "# model_type -> reason\n",
            "def _normalize_config_dtype(dtype: Any) -> Any:\n"
            "    if not isinstance(dtype, str):\n"
            "        return dtype\n"
            "    normalized = str_dtype_to_torch_dtype(dtype.removeprefix(\"torch.\").lower())\n"
            "    return normalized if normalized is not None else dtype\n"
            "\n\n"
            "# model_type -> reason\n",
            "def _normalize_config_dtype(dtype: Any) -> Any:\n",
        ), (
            "        config, model_id, revision=revision, config_format=config_format\n"
            "    )\n",
            "        config, model_id, revision=revision, config_format=config_format\n"
            "    )\n"
            "    config_dtype = _normalize_config_dtype(config_dtype)\n",
            "    config_dtype = _normalize_config_dtype(config_dtype)\n",
        )],
    ),
    (
        "entrypoints/serve/instrumentator/metrics.py",
        [(
            '            "/server_info",\n'
            "        ],\n",
            '            "/server_info",\n'
            '            "/monitor",\n'
            '            "/test",\n'
            "        ],\n",
            '            "/monitor",\n',
        ), (
            "    app.routes.append(metrics_route)\n",
            "    app.routes.append(metrics_route)\n"
            "\n"
            "    # 单文件监控看板(自包含 HTML, 轮询同源 /metrics); 由 VLLM_MONITOR\n"
            "    # 控制开关, 关闭时不挂路由。\n"
            "    from .monitor import attach_router as attach_monitor_router\n"
            "\n"
            "    attach_monitor_router(app)\n",
            "    attach_monitor_router(app)\n",
        )],
    ),
    (
        # PLE(ngram) 大表 NVMe mmap 需要 PP>1 (8 卡 sm75 只能 PP4×TP2 用满)。
        # 上游 Qwen4ExpForConditionalGenerationConfig.verify_and_update_config 对
        # "有 ple_layer_ids 且 PP>1" 一律抛 NotImplementedError(理由: 非首 PP rank
        # 收不到 PLE 的 raw input_ids)。但该检查过严: 只有当某个 PLE 层落在
        # rank>0 才真炸; PLE 层全在 rank 0 (拥有 input_ids) 时 PP>1 完全安全。
        # get_pp_indices 把余数层只分给末/中段、从不给 rank 0, 故 rank 0 恒拥有
        # 0-indexed [0, num_hidden_layers//pp_size), 1-indexed id 在 rank 0 当且仅
        # 当 id <= num_hidden_layers//pp_size。把硬拒绝改成"仅当有 PLE 层离 rank 0
        # 才拒", PLE-mmap 的 PP>1 场景即可放行 (本模型 48 层/PP4, PLE id=2 在 rank 0)。
        "model_executor/models/config.py",
        [(
            "        if text_config.ple_layer_ids and parallel_config.pipeline_parallel_size > 1:\n"
            "            raise NotImplementedError(\n"
            "                \"Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1 \"\n"
            "                \"because non-first pipeline ranks do not receive the raw input_ids \"\n"
            "                \"it needs. Please run with PP=1.\"\n"
            "            )",
            "        if text_config.ple_layer_ids and parallel_config.pipeline_parallel_size > 1:\n"
            "            # 仅当某个 PLE 层落在 PP rank>0 (无 raw input_ids) 时才拒; PLE 层\n"
            "            # 全在 rank 0 (如 PLE-mmap 8 卡 PP4×TP2 场景) 则 PP>1 安全放行。\n"
            "            # rank 0 恒拥有 0-indexed [0, num_hidden_layers//pp_size) (见\n"
            "            # get_pp_indices), 故 1-indexed id 在 rank 0 当且仅当 id <= 该值。\n"
            "            _pp_size = parallel_config.pipeline_parallel_size\n"
            "            _n_rank0 = text_config.num_hidden_layers // _pp_size\n"
            "            if any(ple_id > _n_rank0 for ple_id in text_config.ple_layer_ids):\n"
            "                raise NotImplementedError(\n"
            "                    \"Qwen4Exp N-gram PLE embedding requires the PLE layers to sit \"\n"
            "                    \"on pipeline rank 0, the only rank receiving raw input_ids. \"\n"
            "                    \"PLE layer(s) not on rank 0: \"\n"
            "                    f\"{[i for i in text_config.ple_layer_ids if i > _n_rank0]}. \"\n"
            "                    \"Run with PP=1 or keep the PLE layers on rank 0.\"\n"
            "                )",
            "Qwen4Exp N-gram PLE embedding requires the PLE layers to sit ",
        )],
    ),
    (
        # QSA 索引侧 (indexer_qsa.py) 的 sm75 fp16 放行 —— 同目录 qsa.py 的
        # fp16/fp8 兼容是整文件覆盖 (见 files 列表), 但 QSAIndexer 在姊妹文件
        # indexer_qsa.py, 不在 files 列表里, 上游仍是 bf16-only:
        #   (1) model_config.dtype != bf16 → raise "Qwen4Exp QSA currently requires BF16"
        #   (2) raw_key_cache / compressed_key_cache 硬编码 dtype=bf16
        # sm75 无 bf16 计算, 模型回退 fp16 → 第 (1) 条炸,
        # 且 bf16 侧 cache 与 fp16 激活不一致。(sm75 实测):
        # 校验放宽 fp16/bf16, 两个 QSA side cache 的 dtype 跟 model_config.dtype
        # (激活 dtype) 一致 (K 是激活 dtype, 侧 cache 应同 dtype)。增量锚点注入,
        # 上游正文不整文件覆盖。
        "models/qwen4_exp/nvidia/indexer_qsa.py",
        [
            (
                "        if vllm_config.model_config.dtype != torch.bfloat16:\n"
                '            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")',
                "        if vllm_config.model_config.dtype not in (\n"
                "            torch.bfloat16, torch.float16\n"
                "        ):\n"
                '            raise NotImplementedError("Qwen4Exp QSA requires BF16 or FP16")',
                "Qwen4Exp QSA requires BF16 or FP16",
            ),
            (
                "        self.raw_key_cache = QSAKeyStateCache(\n"
                "            head_size=self.index_head_dim,\n"
                "            dtype=torch.bfloat16,",
                "        self.raw_key_cache = QSAKeyStateCache(\n"
                "            head_size=self.index_head_dim,\n"
                "            dtype=vllm_config.model_config.dtype,",
                "self.raw_key_cache = QSAKeyStateCache(\n"
                "            head_size=self.index_head_dim,\n"
                "            dtype=vllm_config.model_config.dtype,",
            ),
        ],
    ),
    (
        # HyperConnection 层 params_dtype 跟激活 dtype 一致 (sm75 fp16 放行)。
        # 上游 model.py 两处 (decoder layer 260 / Qwen4ExpModel 436) + mtp.py 229
        # 硬编码 params_dtype=bf16: sm75 模型回退 fp16 时, HC 层参数仍 bf16
        # 会与其余 fp16 层 dtype 不一致 (GEMM/累加 dtype mismatch)。改成
        # model_config.dtype。两处缩进不同 (8/12 空格) 可作独立 anchor。
        "models/qwen4_exp/nvidia/model.py",
        [
            (
                "        hc_config = HyperConnectionConfig(\n"
                "            hc_count=config.hc_count,\n"
                "            hidden_size=config.hidden_size,\n"
                "            params_dtype=torch.bfloat16,",
                "        hc_config = HyperConnectionConfig(\n"
                "            hc_count=config.hc_count,\n"
                "            hidden_size=config.hidden_size,\n"
                "            params_dtype=model_config.dtype,",
                "            params_dtype=model_config.dtype,",
            ),
            (
                "            hc_config = HyperConnectionConfig(\n"
                "                hc_count=config.hc_count,\n"
                "                hidden_size=config.hidden_size,\n"
                "                params_dtype=torch.bfloat16,",
                "            hc_config = HyperConnectionConfig(\n"
                "                hc_count=config.hc_count,\n"
                "                hidden_size=config.hidden_size,\n"
                "                params_dtype=vllm_config.model_config.dtype,",
                "                params_dtype=vllm_config.model_config.dtype,",
            ),
            (
                # lm_head 在 checkpoint 里是 int8 量化 (quant config group_3 的
                # re:.*lm_head, pack-quantized): 张量是 lm_head.weight_packed
                # (int32 打包) + lm_head.weight_scale (bf16)。但 Qwen4ExpForCausalLM
                # 构造 ParallelLMHead 没传 quant_config → VocabParallelEmbedding
                # 走 UnquantizedEmbeddingMethod 只建普通 .weight, 与 checkpoint 的
                # weight_packed 不匹配 → load 报 "no parameter lm_head.weight_packed"。
                # 对齐 hy_v4 (model.py:664) 传 quant_config: get_quant_method 对
                # ParallelLMHead 按 prefix="lm_head" 匹配 re:.*lm_head → 返回
                # CompressedTensorsLinearMethod, 建出 weight_packed/scale, 正确 dequant。
                # quant_config 为 None (非量化模型) 时 VocabParallelEmbedding:289 仍走
                # UnquantizedEmbeddingMethod, 行为不变。
                "        self.lm_head = ParallelLMHead(\n"
                "            config.vocab_size,\n"
                "            config.hidden_size,\n"
                "            prefix=maybe_prefix(prefix, \"lm_head\"),\n"
                "        )",
                "        self.lm_head = ParallelLMHead(\n"
                "            config.vocab_size,\n"
                "            config.hidden_size,\n"
                "            quant_config=self.quant_config,\n"
                "            prefix=maybe_prefix(prefix, \"lm_head\"),\n"
                "        )",
                "            quant_config=self.quant_config,\n"
                "            prefix=maybe_prefix(prefix, \"lm_head\"),",
            ),
            (
                # embed_tokens 在 checkpoint 里也是 int8 量化 (group_3 的
                # re:.*embed_tokens, pack-quantized): 张量 embed_tokens.weight_packed
                # (int32) + weight_scale (bf16 group) + weight_shape。Qwen4ExpModel
                # 构造 VocabParallelEmbedding 没传 quant_config → UnquantizedEmbeddingMethod
                # 只建普通 .weight, 与 checkpoint 的 weight_packed 不匹配 → load 报
                # "no parameter embed_tokens.weight_packed"。传 quant_config:
                # get_quant_method 对真 embedding (非 ParallelLMHead) 按 prefix 匹配
                # re:.*embed_tokens → _is_wNa16_group_channel (int8/group/static) 成立
                # → 返回 CompressedTensorsEmbeddingWNA16Int, 建 weight_packed/scale/shape,
                # forward 走 Triton dequant-gather (elementwise int32→scale, sm75 兼容)。
                "        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)",
                "        self.embed_tokens = VocabParallelEmbedding(\n"
                "            self.vocab_size,\n"
                "            config.hidden_size,\n"
                "            quant_config=vllm_config.quant_config,\n"
                "            prefix=maybe_prefix(prefix, \"embed_tokens\"),\n"
                "        )",
                "            quant_config=vllm_config.quant_config,\n"
                "            prefix=maybe_prefix(prefix, \"embed_tokens\"),",
            ),
        ],
    ),
    (
        # PLE(ngram) 大表在 VLLM_PLE_MMAP=1 时走磁盘 mmap, 加载阶段整块读一遍再丢弃
        # 纯属浪费 (105GB 顺序 I/O); 且 PP4×TP2 8 个 worker 并发 get_tensor 全部 ngram
        # 分片 → 同盘 mmap 并发压力 → 观察到 "could not determine the shape of
        # 'torch.storage.UntypedStorage'"。在 should_skip_weight (读盘**之前**, 见
        # weight_utils.py get_tensor 之前的 should_skip 判断) 跳过 ngram 分片, 与参考
        # 项目 (ep_weight_filter.py 同款) 一致。vllm_ple_mmap 的 load_weights 仍在
        # PLE 层其余张量 (conv1d/key_proj/value_proj/norm/... 不被 skip) 到来时调用 →
        # _setup_table 照常开 mmap 表。
        "model_executor/model_loader/ep_weight_filter.py",
        [
            (
                "import regex as re\n",
                "import os\n\nimport regex as re\n",
                "import os\n\nimport regex as re\n",
            ),
            (
                "def should_skip_weight(\n"
                "    weight_name: str,\n"
                "    local_expert_ids: set[int] | None,\n"
                ") -> bool:\n"
                "    \"\"\"Return ``True`` if *weight_name* is an expert weight that does not\n"
                "    belong to the local rank and should be skipped during loading.\"\"\"",
                "def should_skip_weight(\n"
                "    weight_name: str,\n"
                "    local_expert_ids: set[int] | None,\n"
                ") -> bool:\n"
                "    \"\"\"Return ``True`` if *weight_name* is an expert weight that does not\n"
                "    belong to the local rank and should be skipped during loading.\"\"\"\n"
                "    if os.environ.get(\"VLLM_PLE_MMAP\", \"0\") == \"1\" and _PLE_MMAP_SHARD_RE.search(\n"
                "        weight_name\n"
                "    ):\n"
                "        return True",
                "if os.environ.get(\"VLLM_PLE_MMAP\", \"0\") == \"1\" and _PLE_MMAP_SHARD_RE.search(",
            ),
            (
                "_EXPERT_ID_RE = re.compile(r\"\\.experts\\.(\\d+)\\.\")\n",
                "_EXPERT_ID_RE = re.compile(r\"\\.experts\\.(\\d+)\\.\")\n\n"
                "# PLE 磁盘 mmap: ngram 表分片由 mmap 从文件按需取, 读盘前跳过。\n"
                "_PLE_MMAP_SHARD_RE = re.compile(\n"
                "    r\"ple_embedding\\.ngram_embedding\\.shard_\\d+\\.weight$\"\n"
                ")\n",
                "_PLE_MMAP_SHARD_RE = re.compile(\n"
                "    r\"ple_embedding\\.ngram_embedding\\.shard_\\d+\\.weight$\"\n"
                ")\n",
            ),
        ],
    ),
    (
        # MTP 草稿头的 HC final mixer params_dtype 同理跟激活 dtype 一致。
        "models/qwen4_exp/nvidia/mtp.py",
        [
            (
                "        hc_config = HyperConnectionConfig(\n"
                "            hc_count=config.hc_count,\n"
                "            hidden_size=config.hidden_size,\n"
                "            params_dtype=torch.bfloat16,",
                "        hc_config = HyperConnectionConfig(\n"
                "            hc_count=config.hc_count,\n"
                "            hidden_size=config.hidden_size,\n"
                "            params_dtype=model_config.dtype,",
                "            params_dtype=model_config.dtype,",
            ),
            (
                # MTP 草稿模型的 embed_tokens: checkpoint 的顶层 embed_tokens.* (int8)
                # 经 _remap_mtp_weight_name (mtp.py:92/106) 映射到草稿模型自己的
                # model.embed_tokens.*, 故草稿 embed_tokens 也要 int8 量化装载, 否则
                # 同样 "no parameter embed_tokens.weight_packed"。Qwen4ExpMultiTokenPredictor
                # 收 target vllm_config (init 参数), 顶层 embed_tokens 是 target 的 int8,
                # 用 vllm_config.quant_config 匹配 re:.*embed_tokens → WNA16Int。
                "        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)",
                "        self.embed_tokens = VocabParallelEmbedding(\n"
                "            self.vocab_size,\n"
                "            self.hidden_size,\n"
                "            quant_config=vllm_config.quant_config,\n"
                "            prefix=maybe_prefix(prefix, \"embed_tokens\"),\n"
                "        )",
                "            quant_config=vllm_config.quant_config,\n"
                "            prefix=maybe_prefix(prefix, \"embed_tokens\"),",
            ),
            (
                # MTP 草稿头 lm_head: checkpoint 顶层 lm_head.* (int8) 经
                # _remap_mtp_weight_name (mtp.py:97-103) 映射到草稿头 self.lm_head,
                # 同理要 int8 量化 (CompressedTensorsLinearMethod)。tie_word_embeddings=False
                # 时走这个分支建真实 ParallelLMHead; 用 self.quant_config (target)。
                "                self.lm_head = ParallelLMHead(\n"
                "                    config.vocab_size,\n"
                "                    config.hidden_size,\n"
                "                    prefix=maybe_prefix(prefix, \"lm_head\"),\n"
                "                )",
                "                self.lm_head = ParallelLMHead(\n"
                "                    config.vocab_size,\n"
                "                    config.hidden_size,\n"
                "                    quant_config=self.quant_config,\n"
                "                    prefix=maybe_prefix(prefix, \"lm_head\"),\n"
                "                )",
                "                    quant_config=self.quant_config,\n"
                "                    prefix=maybe_prefix(prefix, \"lm_head\"),",
            ),
        ],
    ),
    (
        # model_state.py Qwen4ExpModelState.__init__ (每个 worker 都跑) 的第二道 PP>1
        # 硬检查: 上游对 uses_ngram_embedding 且 PP>1 一律 raise RuntimeError。与
        # config.py 的 gate 同理过严——只有某个 PLE 层落在 rank>0 (收不到 raw
        # input_ids) 才真炸; PLE 层全在 rank 0 时 PP>1 安全。改法与 config.py 一致:
        # 仅当有 ple_id > num_hidden_layers//pp_size 才拒。本模型 PLE id=2, 48层/PP4
        # → 边界 12, 2<=12 放行。注意这里 config = self.model_config.hf_text_config
        # (变量名是 config, 非 config.py 里的 text_config), parallel 在
        # vllm_config.parallel_config。
        "models/qwen4_exp/nvidia/model_state.py",
        [(
            "        if vllm_config.parallel_config.pipeline_parallel_size > 1:\n"
            "            raise RuntimeError(\n"
            "                \"N-gram PLE embedding currently requires \"\n"
            "                \"pipeline_parallel_size=1 because non-first pipeline ranks do \"\n"
            "                \"not receive the raw input_ids required by PLE. Please run \"\n"
            "                \"with PP=1.\"\n"
            "            )",
            "        if vllm_config.parallel_config.pipeline_parallel_size > 1:\n"
            "            # 仅当某个 PLE 层落在 PP rank>0 (无 raw input_ids) 时才拒; PLE 层\n"
            "            # 全在 rank 0 则 PP>1 安全放行。对齐 config.py 的 gate: rank 0 恒\n"
            "            # 拥有 0-indexed [0, num_hidden_layers//pp_size), 1-indexed id 在\n"
            "            # rank 0 当且仅当 id <= 该值。本模型 PLE id=2, 48层/PP4 → 边界 12。\n"
            "            _pp_size = vllm_config.parallel_config.pipeline_parallel_size\n"
            "            _n_rank0 = config.num_hidden_layers // _pp_size\n"
            "            if any(ple_id > _n_rank0 for ple_id in config.ple_layer_ids):\n"
            "                raise RuntimeError(\n"
            "                    \"N-gram PLE embedding requires the PLE layers to sit on \"\n"
            "                    \"pipeline rank 0, the only rank receiving raw input_ids. \"\n"
            "                    \"PLE layer(s) not on rank 0: \"\n"
            "                    f\"{[i for i in config.ple_layer_ids if i > _n_rank0]}. \"\n"
            "                    \"Run with PP=1 or keep the PLE layers on rank 0.\"\n"
            "                )",
            "N-gram PLE embedding requires the PLE layers to sit on ",
        )],
    ),
    (
        # PLE host-gather (P0, PLAN-w4a16-decode-ops): 把 PLE(ngram) 的 ids D2H +
        # 表 gather 从模型 forward 里挪到 model_state.prepare_inputs (forward 之前,
        # capture 之外), forward 只 return 预填 buffer 的 view → forward 内无 custom
        # op 分裂点 → decode 可走 FULL_DECODE_ONLY cudagraph (消灭 PIECEWISE 每 step
        # 在 eager 段间隙的 GPU 空转; 参考实现同改造 decode 45→80)。三处均为
        # append-only (anchor 原文保留 + 尾追), 不改上游正文:
        #   (1) __init__ 尾: 找 host-gather PLE 模块挂到 state (PP>1 时其他 rank
        #       找不到 → 静默 no-op);
        #   (2) prepare_inputs: 每步 (eager, capture 外) 预填 buffer;
        #   (3) prepare_dummy_inputs: profiling/capture 用零 id 预填 (capture 在
        #       cudagraph_utils 调本函数之后, 故 gather 不被 capture)。
        # helper 全在 vllm_ple_mmap; 这里只 append 调用行 (增量解耦)。env 关
        # (VLLM_PLE_HOST_GATHER=0) 时 attach_to_model_state 直接 return, 行为回退。
        "models/qwen4_exp/nvidia/model_state.py",
        [
            (
                "        self.ple_query_start_loc = torch.zeros(\n"
                "            self.max_num_reqs + 1,\n"
                "            dtype=torch.int32,\n"
                "            device=self.device,\n"
                "        )\n",
                "        self.ple_query_start_loc = torch.zeros(\n"
                "            self.max_num_reqs + 1,\n"
                "            dtype=torch.int32,\n"
                "            device=self.device,\n"
                "        )\n"
                "        # PLE host-gather: 挂预填钩子 (见 vllm_ple_mmap)。\n"
                "        import vllm.vllm_ple_mmap as _ple_mmap\n"
                "\n"
                "        _ple_mmap.attach_to_model_state(self, model)\n",
                "_ple_mmap.attach_to_model_state(self, model)",
            ),
            (
                "        model_inputs.update(\n"
                "            query_start_loc=query_start_loc,\n"
                "            ngram_context=self._prepare_ngram_context(input_batch, req_states),\n"
                "        )\n"
                "        return model_inputs\n",
                "        ngram_context = self._prepare_ngram_context(input_batch, req_states)\n"
                "        model_inputs.update(\n"
                "            query_start_loc=query_start_loc,\n"
                "            ngram_context=ngram_context,\n"
                "        )\n"
                "        # PLE host-gather: forward 之前预填 buffer (eager, capture 外)。\n"
                "        _hg = getattr(self, \"_ple_hg_layer\", None)\n"
                "        if _hg is not None:\n"
                "            _hg.host_gather(input_batch.input_ids, query_start_loc, ngram_context)\n"
                "        return model_inputs\n",
                "_ple_hg_layer\", None",
            ),
            (
                "        ngram_context = self.ngram_context\n"
                "        ngram_context.fill_(self.ngram_eos_token_id)\n"
                "        model_inputs.update(\n"
                "            query_start_loc=query_start_loc,\n"
                "            ngram_context=ngram_context,\n"
                "        )\n"
                "        return model_inputs\n",
                "        ngram_context = self.ngram_context\n"
                "        ngram_context.fill_(self.ngram_eos_token_id)\n"
                "        model_inputs.update(\n"
                "            query_start_loc=query_start_loc,\n"
                "            ngram_context=ngram_context,\n"
                "        )\n"
                "        # PLE host-gather: capture/profiling 用零 id 预填 buffer。\n"
                "        _hg = getattr(self, \"_ple_hg_layer\", None)\n"
                "        if _hg is not None:\n"
                "            _hg.host_gather(\n"
                "                self._ple_hg_dummy_ids[:num_tokens],\n"
                "                query_start_loc,\n"
                "                ngram_context,\n"
                "            )\n"
                "        return model_inputs\n",
                "_ple_hg_dummy_ids[:num_tokens]",
            ),
        ],
    ),
    (
        # v0.29.0 混合模型 (GDN + QSA + MTP 投机解码) PP>1 下, 某 KV cache group
        # 的层可能全落在单一 PP rank (MTP 草稿头 stage-local / 单 rank PLE 层)。
        # _project_kv_cache_groups_to_worker 对本 worker 空集的 UniformTypeKVCacheSpecs
        # 不裁剪 kv_cache_specs (保持全局完整 dict), get_kv_cache_config_from_groups
        # 又用 dict key 生成 KVCacheTensor → 含本 worker 没有的幽灵层名 →
        # allocate_kv_cache 的 next(...) 空 → StopIteration (8 worker 全崩, 262k e2e)。
        # 增量修 tensor 生成: uniform group 改以 group.layer_names (投影后恒正确,
        # 且为 kv_cache_specs key 的子集) 为准, 滤掉幽灵层名; 非空 group 行为不变。
        # 不改投影处裁 dict: 空 dict 会波及 max_memory_usage_pages 的 max() 崩溃。
        "v1/core/kv_cache_utils.py",
        [(
            "        if isinstance(group_spec, UniformTypeKVCacheSpecs):\n"
            "            for layer_name, spec in group_spec.kv_cache_specs.items():\n"
            "                layers_by_spec[spec].append(layer_name)",
            "        if isinstance(group_spec, UniformTypeKVCacheSpecs):\n"
            "            # sm75 overlay: PP 下某 group 的层可能全不在本 worker (MTP\n"
            "            # 草稿头 / 单 rank PLE 层)。投影不裁剪 kv_cache_specs (保全局\n"
            "            # dict), 直接用其 key 会生成含幽灵层名的 KVCacheTensor ->\n"
            "            # allocate_kv_cache 的 next(...) 空 -> StopIteration。改以\n"
            "            # group.layer_names (本 worker 实际拥有的层, 投影后恒正确且\n"
            "            # 为 dict key 子集) 为准, 幽灵层名被滤掉; 非空 group 行为不变。\n"
            "            for layer_name in group.layer_names:\n"
            "                spec = group_spec.kv_cache_specs[layer_name]\n"
            "                layers_by_spec[spec].append(layer_name)",
            "            # sm75 overlay: PP 下某 group 的层可能全不在本 worker (MTP",
        )],
    ),
    (
        # EXL3 (27B VLM) 视觉塔权重跳过: checkpoint 按 MHA 存 (q_proj/k_proj/v_proj),
        # 但 vllm Qwen3_VisionTransformer 用 fused qkv 模块 (k_proj 子模块不存在) ->
        # AutoWeightsLoader 递归进视觉塔后找不到 k_proj 模块, utils.py raise。文本
        # serving 视觉塔永不前向, 顶层 load_weights 直接过滤 visual.* 权重即可 (视觉
        # qkv 保持随机 init, 不影响文本)。anchor 取顶层 Qwen3_5ForConditionalGeneration.
        # load_weights 的 return 行 + 其后 @classmethod get_mamba_state_dtype_from_config
        # (仅顶层类后跟它), 须含 return 与 @classmethod 间的空行 (否则 count=0)。
        # 对应 EXL3 侧 get_quant_method 对 visual/vision 前缀返 UnquantizedLinearMethod。
        "model_executor/models/qwen3_5.py",
        [(
            "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
            "        loader = AutoWeightsLoader(self)\n"
            "        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n"
            "\n"
            "    @classmethod\n"
            "    def get_mamba_state_dtype_from_config(\n",
            "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
            "        # SM75 EXL3: skip visual tower weights (fused qkv name mismatch)\n"
            "        # 视觉塔 checkpoint 按分开 q/k/v_proj 存, vllm 视觉模块用 fused qkv,\n"
            "        # 递归 loader 找不到 k_proj 模块会 raise。文本推理不执行视觉塔, 顶层\n"
            "        # 直接跳过 visual.* 权重 (视觉 qkv 保持随机 init, 不影响文本)。\n"
            "        if getattr(type(self), \"__name__\", \"\") == \"Qwen3_5ForConditionalGeneration\":\n"
            "            weights = (\n"
            "                (n, w) for n, w in weights\n"
            "                if \"visual\" not in n.split(\".\")\n"
            "            )\n"
            "        loader = AutoWeightsLoader(self)\n"
            "        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n"
            "\n"
            "    @classmethod\n"
            "    def get_mamba_state_dtype_from_config(\n",
            "SM75 EXL3: skip visual tower weights (fused qkv name mismatch)",
        )],
    ),
    (
        # int8 权重 (compressed-tensors int-quantized int8, per-channel) 接
        # firefly int8 GEMM: 覆盖 W8A16 (无激活量化) 与 W8A8 (动态 per-token int8
        # 激活, SmoothQuant) —— 权重布局相同, 前向都是 per-token 动态 int8 GEMM。
        # stock vllm 0.29.0 对 W8A16 在 _get_scheme_from_parts 末尾 raise
        # NotImplementedError。增量注入 (不整文件覆盖): 在 compressed_tensors.py
        # **文件末尾** append import + maybe_apply() 调用 —— 模块加载时即
        # monkey-patch CompressedTensorsConfig.get_scheme, 早于任何 get_scheme
        # 调用 (每个 Linear 层配置 get_quant_method 时被调)。
        # VLLM_FIREFLY_DIRECT 开启时命中的 int8 per-channel 层切到 firefly
        # int8 GEMM, 否则 maybe_apply 直接 return (no-op, 不碰上游)。maybe_apply
        # 幂等 (防重复 import 二次 patch)。anchor 取文件末尾唯一且稳定的 del
        # 块 (EOF 处)。
        "model_executor/layers/quantization/compressed_tensors/compressed_tensors.py",
        [(
            "        # Discard all placeholders.\n"
            "        del layer.k_scale\n"
            "        del layer.v_scale\n"
            "        del layer.q_scale\n"
            "        del layer.k_zero_point\n"
            "        del layer.v_zero_point\n"
            "        del layer.q_zero_point",
            "        # Discard all placeholders.\n"
            "        del layer.k_scale\n"
            "        del layer.v_scale\n"
            "        del layer.q_scale\n"
            "        del layer.k_zero_point\n"
            "        del layer.v_zero_point\n"
            "        del layer.q_zero_point\n"
            "\n"
            "\n"
            "# SM75 firefly int8: 模块加载时打 monkey-patch (VLLM_FIREFLY_DIRECT\n"
            "# 开启时把 int8 权重 per-channel 层 (W8A16/W8A8) 切到 firefly int8 GEMM, 否则 no-op)。\n"
            "from vllm.model_executor.layers.quantization.utils import (\n"
            "    firefly_int8 as _ff_int8,\n"
            ")\n"
            "\n"
            "_ff_int8.maybe_apply()",
            "_ff_int8.maybe_apply()",
        )],
    ),
    (
        # firefly deferred-recv (VLLM_FIREFLY_DEFER): PP>1 非首 stage 把 irecv 从
        # execute_model 入口推迟到 forward 前, 让 host 准备与上一 stage GPU 计算
        # 重叠。三处注入: ①模块级插 DeferredRecvIntermediateTensors 子类 +
        # _defer_pp_recv helper(紧挨 AsyncIntermediateTensors 之后, class Worker
        # 之前); ②recv 分支加 deferred if / 原逻辑转 elif; ③execute_model 后补
        # wait_for_comm(没人读 .tensors 时也要与上一 rank send 配对)。设备操作
        # 顺序不变(recv→forward→send), 数值逐 bit 一致; PP1(TP8)恒 is_first_rank
        # 不进分支 = no-op。纯 host 调度顺序, 不改编译图(envs_sm75 里 pop)。
        "v1/worker/gpu_worker.py",
        [
            (
                # ① 子类 + helper 插到 class Worker 定义之前。
                "class Worker(WorkerBase):\n",
                'class DeferredRecvIntermediateTensors(AsyncIntermediateTensors):\n'
                '    """firefly deferred-recv: 把 PP 非首 stage 的 irecv 推迟到 forward 前。\n'
                '\n'
                '    irecv_tensor_dict 第一步是 CPU group 上阻塞收 pickled metadata, 而它\n'
                '    只有上一 PP stage launch 完 forward 后才发。在 execute_model 入口就调\n'
                '    它, 会让本 rank 的输入/attention-metadata host 准备全卡在"等上一 rank\n'
                '    launch"之后; 当 per-rank GPU 时间短(decode)时这段 host 活就成了 step\n'
                '    时长。model runner 只在 forward 前读 .tensors, 故把 irecv 推迟到那时\n'
                '    发 —— 本 rank 设备操作顺序不变(recv→forward→send), 数值逐 bit 一致。\n'
                '    见 VLLM_FIREFLY_DEFER / install_sm75_overlay.py。\n'
                '    """\n'
                '\n'
                '    def __init__(self, recv: Callable[[], tuple]) -> None:\n'
                '        super().__init__({})\n'
                '        self._recv = recv\n'
                '\n'
                '    def wait_for_comm(self) -> None:\n'
                '        if object.__getattribute__(self, "_comm_waited"):\n'
                '            return\n'
                '        tensor_dict, handles, postprocess = object.__getattribute__(\n'
                '            self, "_recv"\n'
                '        )()\n'
                '        assert tensor_dict is not None\n'
                '        object.__setattr__(self, "tensors", tensor_dict)\n'
                '        object.__setattr__(self, "_comm_handles", handles)\n'
                '        object.__setattr__(self, "_comm_postprocess", postprocess)\n'
                '        super().wait_for_comm()\n'
                '\n'
                '\n'
                'def _defer_pp_recv() -> bool:\n'
                '    """firefly deferred-recv 开关: VLLM_FIREFLY_DEFER (纯开关, 默认开)。"""\n'
                '    return envs.VLLM_FIREFLY_DEFER\n'
                '\n'
                '\n'
                'class Worker(WorkerBase):\n',
                'class DeferredRecvIntermediateTensors(AsyncIntermediateTensors):',
            ),
            (
                # ② recv: deferred 分支优先(v2 runner + 开关), 原逻辑转 elif 回退。
                '        if forward_pass and not get_pp_group().is_first_rank:\n'
                '            tensor_dict, comm_handles, comm_postprocess = (\n'
                '                get_pp_group().irecv_tensor_dict(\n'
                '                    all_gather_group=get_tp_group(),\n'
                '                    all_gather_tensors=all_gather_tensors,\n'
                '                )\n'
                '            )\n'
                '            assert tensor_dict is not None\n'
                '            intermediate_tensors = AsyncIntermediateTensors(\n'
                '                tensor_dict,\n'
                '                comm_handles=comm_handles,\n'
                '                comm_postprocess=comm_postprocess,\n'
                '            )\n',
                '        if (\n'
                '            forward_pass\n'
                '            and not get_pp_group().is_first_rank\n'
                '            and self.use_v2_model_runner\n'
                '            and _defer_pp_recv()\n'
                '        ):\n'
                '            # firefly deferred-recv: 此处只挂一个"到 forward 前才发\n'
                '            # irecv"的回调, 不在入口阻塞收 metadata。\n'
                '            pp_group, tp_group = get_pp_group(), get_tp_group()\n'
                '            intermediate_tensors = DeferredRecvIntermediateTensors(\n'
                '                lambda: pp_group.irecv_tensor_dict(\n'
                '                    all_gather_group=tp_group,\n'
                '                    all_gather_tensors=all_gather_tensors,\n'
                '                )\n'
                '            )\n'
                '        elif forward_pass and not get_pp_group().is_first_rank:\n'
                '            tensor_dict, comm_handles, comm_postprocess = (\n'
                '                get_pp_group().irecv_tensor_dict(\n'
                '                    all_gather_group=get_tp_group(),\n'
                '                    all_gather_tensors=all_gather_tensors,\n'
                '                )\n'
                '            )\n'
                '            assert tensor_dict is not None\n'
                '            intermediate_tensors = AsyncIntermediateTensors(\n'
                '                tensor_dict,\n'
                '                comm_handles=comm_handles,\n'
                '                comm_postprocess=comm_postprocess,\n'
                '            )\n',
                'intermediate_tensors = DeferredRecvIntermediateTensors(',
            ),
            (
                # ③ execute_model 后: 没人读 .tensors 也补一次 irecv, 与上一 rank send 配对。
                '            output = self.model_runner.execute_model(\n'
                '                scheduler_output, intermediate_tensors\n'
                '            )\n',
                '            output = self.model_runner.execute_model(\n'
                '                scheduler_output, intermediate_tensors\n'
                '            )\n'
                '            if isinstance(intermediate_tensors, DeferredRecvIntermediateTensors):\n'
                '                # firefly deferred-recv: 即便没人读 .tensors, 也要补发\n'
                '                # irecv 与上一 rank 的 send 配对(否则 P2P 挂起/下步错位)。\n'
                '                intermediate_tensors.wait_for_comm()\n',
                'isinstance(intermediate_tensors, DeferredRecvIntermediateTensors)',
            ),
        ],
    ),
    (
        # mamba grid 解耦: align 模式尊重显式 --mamba-block-size, 不再重置成
        # KV block_size。inert (2026-09-29 实测): v0.29.0 align 模式 mamba 常驻
        # 内存是固定滑动窗口 max_memory = page*(2+spec+prefill), 与 mbs 无关
        # (前面状态被 remove_skipped_blocks 回收), 放大 grid 不省 KV 显存
        # (TP8 262144 实测 375688 tokens 不变)。容量收益仅 all 模式 (关 prefix
        # cache, 逐 token 存检查点) 才存在, 当前未用。保留: all 模式预留 +
        # 语义正确。inert: 不传 flag 走原重置路径, 其他模型零影响。
        # 约束: grid 须是 block_size 整数倍, 否则 prefix 长度无法同时 block/
        # grid 对齐 → prefix cache 命中归零, 故 assert。
        "platforms/interface.py",
        [(
            '        if cache_config.mamba_cache_mode == "align":\n'
            '            cache_config.mamba_block_size = cache_config.block_size\n',
            '        if cache_config.mamba_cache_mode == "align":\n'
            '            if cache_config.user_specified_mamba_block_size:\n'
            '                # 显式 --mamba-block-size 是循环状态检查点 grid, 不是 KV\n'
            '                # page 倍数: 保留它而不是强制等于 KV block size。须为\n'
            '                # block size 整数倍, 否则 prefix 长度无法同时 block/\n'
            '                # grid 对齐 → prefix cache 命中归零。\n'
            '                grid = cache_config.mamba_block_size\n'
            '                assert grid is not None and grid % cache_config.block_size == 0, (\n'
            '                    "--mamba-block-size must be a multiple of --block-size in "\n'
            '                    f"align mode, got {grid} and {cache_config.block_size}"\n'
            '                )\n'
            '            else:\n'
            '                cache_config.mamba_block_size = cache_config.block_size\n',
            'grid = cache_config.mamba_block_size',
        )],
    ),
]


def apply_injections(package_root: Path) -> None:
    """按 INJECTIONS 把小改动注入上游同名文件(幂等, 锚点须唯一)。

    与整文件 copy 的 files 列表互斥: 一个文件要么整文件覆盖、要么锚点注入,
    不两者都做。probe 已存在(注入过)则跳过; 否则要求 anchor 恰好唯一且存在,
    否则抛错(防止注入打错位置/漏注入)。
    """
    for relative, pairs in INJECTIONS:
        target = package_root / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        text = target.read_text(encoding="utf-8")
        for anchor, replacement, probe in pairs:
            if probe in text:
                continue  # 已注入, 幂等跳过
            if text.count(anchor) != 1:
                raise RuntimeError(
                    f"anchor not unique in {relative} (count={text.count(anchor)}): {anchor!r}"
                )
            text = text.replace(anchor, replacement)
        target.write_text(text, encoding="utf-8")


def main() -> None:
    import vllm

    parser = argparse.ArgumentParser()
    parser.add_argument("overlay", type=Path)
    parser.add_argument("--source-copy", type=Path)
    args = parser.parse_args()

    source_root = args.overlay.resolve()
    package_root = Path(vllm.__file__).resolve().parent
    files = [
        "device_allocator/disk_snapshot.py",
        "device_allocator/disk_sleep.py",
        "engine/arg_utils.py",
        "entrypoints/serve/instrumentator/monitor.py",
        "entrypoints/serve/instrumentator/dashboard.html",
        "entrypoints/serve/instrumentator/test.html",
        "envs_sm75.py",
        "model_executor/kernels/linear/mixed_precision/marlin.py",
        "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "model_executor/layers/quantization/utils/firefly.py",
        "model_executor/layers/quantization/utils/firefly.cu",
        "model_executor/layers/quantization/utils/firefly_exl3.cu",
        "model_executor/layers/quantization/utils/firefly_int8.py",
        "model_executor/layers/quantization/exl3/__init__.py",
        "model_executor/layers/quantization/exl3/exl3.py",
        "distributed/device_communicators/firefly_allreduce.py",
        "distributed/device_communicators/firefly_allreduce.cu",
        "distributed/device_communicators/cuda_communicator.py",
        "v1/attention/backends/flashinfer.py",
        "v1/attention/backends/gdn_attn.py",
        "v1/core/sched/scheduler_sm75.py",
        "v1/engine/async_llm.py",
        "v1/engine/auto_sleep.py",
        "v1/engine/core.py",
        "v1/engine/core_client.py",
        "models/qwen4_exp/nvidia/qsa.py",
        "models/qwen4_exp/nvidia/ops/qsa.py",
        # QSA #54513/#54873 移植 (prefill/decode 路径分离):
        "models/qwen4_exp/nvidia/ops/qsa_indexer.py",
        "models/qwen4_exp/common/qsa_cache.py",
        "model_executor/warmup/qwen4_exp_qsa_warmup.py",
        "vllm_ple_mmap.py",
        "vllm_mtp_stage_local.py",
        "vllm_hc_dequant.py",
        "vllm_firefly_hc.py",
        "sm75_mtp_norm.py",
    ]
    for relative in files:
        source = source_root / relative
        destination = package_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    # 锚点注入: 对上游有同名、只改 1~2 行的文件不整文件覆盖(见 INJECTIONS)。
    apply_injections(package_root)

    backend_file = package_root / "device_allocator/sleep_mode_backend.py"
    registration = '\nSleepModeBackendFactory.register_backend(\n    "disk", "vllm.device_allocator.disk_sleep", "DiskSleepBackend",\n)\n'
    backend_text = backend_file.read_text()
    if '"vllm.device_allocator.disk_sleep"' not in backend_text:
        backend_file.write_text(backend_text + registration)

    # SM75 扩展 env: 不整文件覆盖上游 vllm/envs.py, 改为 import 注入。
    # 往底座 envs.py 尾部追加一行, 触发 envs_sm75.apply()(把 EXTENSIONS 灌进
    # environment_variables + wrap compile_factors)。append 在 envs 模块体最末,
    # 此刻 environment_variables / compile_factors 均已定义。
    # 上游 envs.py 升级随便改, 只需保证仍含 environment_variables dict +
    # compile_factors 返回 dict; 本行与上面 backend registration 手法一致。
    envs_file = package_root / "envs.py"
    envs_hook = "\n# vllm-turing overlay: inject SM75 extension envs (idempotent).\nimport vllm.envs_sm75\nvllm.envs_sm75.apply()\n"
    envs_text = envs_file.read_text()
    if "vllm.envs_sm75.apply()" not in envs_text:
        envs_file.write_text(envs_text + envs_hook)

    # EXL3 注册: 往 quantization/__init__.py 尾部追加 try-import (幂等)。
    # 不能在 get_quantization_config 函数内 lazy import —— line 109 的
    # `if quantization not in QUANTIZATION_METHODS: raise` 检查在函数体 import 之前,
    # 首次 get_quantization_config("exl3") 必失败。尾部 import 在模块加载时即触发
    # @register_quantization_config("exl3"), 先于任何 get_quantization_config 调用;
    # exl3.py 反向 import 本模块的 register_quantization_config (模块体顶部已定义,
    # 部分初始化下仍可用, 无循环)。try/except 兜底, 缺 .cu/无 CUDA 环境时静默跳过。
    quant_init = package_root / "model_executor/layers/quantization/__init__.py"
    exl3_hook = (
        "\n# vllm-turing overlay: 注册 EXL3 sm75 在线解码 (idempotent; 失败静默跳过).\n"
        "try:\n"
        "    from vllm.model_executor.layers.quantization.exl3 import Exl3Config  # noqa: F401\n"
        "except Exception:  # noqa: BLE001\n"
        "    pass\n"
    )
    quant_text = quant_init.read_text()
    if "quantization.exl3 import Exl3Config" not in quant_text:
        quant_init.write_text(quant_text + exl3_hook)

    # PLE(ngram) 两处改动, 都在 ngram_embedding.py: (1) fp8 表 pinned-lookup 改
    # raw-byte 直拷 (inline 锚点替换); (2) 大表 NVMe mmap offload (末尾 append hook,
    # 在类定义后触发 maybe_apply 打 patch)。v0.30 起 Qwen4ExpNGramEmbedding 类从
    # ple_layer.py 搬到 ngram_embedding.py (ple_layer.py 仅 import)。读一次文件,
    # 两个改动各自 marker 判幂等, 写一次。手法与上面 envs hook 一致。文件不存在
    # (底座没带 qwen4_exp 模型) 则跳过 —— PLE 只在该模型下有意义, 不影响其余行为。
    ngram_emb_file = package_root / "models/qwen4_exp/nvidia/ngram_embedding.py"
    if ngram_emb_file.is_file():
        ngram_emb_text = ngram_emb_file.read_text()
        # PLE fp8 表 pinned-lookup raw-byte 直拷: fp8 权重走 pinned host 拷贝路径时,
        # 原 PyTorch copy 会做 fp8->fp32->fp8 的 dtype 转换; 这里对 fp8 (e4m3/e5m2)
        # 的 weight 与 output 改 .view(torch.uint8), 按 raw byte 直拷, bit-exact 且省
        # 一次无意义往返。kernel 是纯 gather (无对 values 的算术) + fp8 恰好 1 字节/
        # 元素, 故 view 后 embedding_dim 索引语义不变。非 fp8 (int8/int4/mem) 走原路。
        # 幂等: 含 view(torch.uint8) 即已注入。断言防上游改锚点静默漏注入 (qwen4_exp
        # 确定要跑, 锚点必须命中)。
        if "view(torch.uint8)" not in ngram_emb_text:
            ngram_emb_text = ngram_emb_text.replace(
                "                self._uva_weight,\n"
                "                flat_ids,\n"
                "                output,\n",
                "                self._uva_weight.view(torch.uint8) "
                "if self.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) "
                "else self._uva_weight,\n"
                "                flat_ids,\n"
                "                output.view(torch.uint8) "
                "if self.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) "
                "else output,\n",
                1,
            )
            assert "self._uva_weight.view(torch.uint8)" in ngram_emb_text, (
                "PLE fp8 锚点未匹配 (上游 ngram_embedding.py 变更?)")
        # PLE 大表 NVMe mmap offload (idempotent)。
        ple_hook = (
            "\n# vllm-turing overlay: PLE 表 NVMe mmap offload (idempotent).\n"
            "import vllm.vllm_ple_mmap\n"
            "vllm.vllm_ple_mmap.maybe_apply(Qwen4ExpNGramEmbedding)\n"
        )
        if "vllm.vllm_ple_mmap.maybe_apply" not in ngram_emb_text:
            ngram_emb_text = ngram_emb_text + ple_hook
        ngram_emb_file.write_text(ngram_emb_text)

    # MTP 草稿头 stage-local: 往上游 qwen4_exp mtp.py 末尾 append 一行, 在
    # Qwen4ExpMultiTokenPredictor 类定义之后、实例化之前触发 maybe_apply (打 patch)。
    # 与 PLE hook 同手法 (尾部 append + marker 判幂等)。PP>1 + MTP 时草稿头只会在末
    # rank 跑, 但 forward 拿目标模型的 pp 位置分支 → 末 rank 炸 assert; patch 删掉
    # 两个 PP 分支。patch 类属性 forward 对 @support_torch_compile
    # 的编译路径生效 (wrapper.py:127 在实例构造期捕获 self.forward=类属性, 见
    # vllm_mtp_stage_local.py 模块 docstring 的证据)。文件不存在 (底座没带模型) 跳过。
    mtp_file = package_root / "models/qwen4_exp/nvidia/mtp.py"
    if mtp_file.is_file():
        mtp_text = mtp_file.read_text()
        # MTP fused GemmaRMSNorm (triton): 两个 pre_fc_norm 走单 kernel, FP32 累加,
        # sm75 无 bf16 纯 PyTorch 一串 elementwise 发射开销大。局部 import 在 forward
        # 内 (方法前段), 第二个 norm 调用复用。幂等: 含 sm75_mtp_gemma_norm 即已注入。
        # 断言防上游改锚点导致静默漏注入 (qwen4_exp 是我们确定要跑的模型, 锚点必须命中)。
        if "sm75_mtp_gemma_norm" not in mtp_text:
            mtp_text = mtp_text.replace(
                "            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)\n",
                "            from vllm.sm75_mtp_norm import sm75_mtp_gemma_norm\n"
                "            inputs_embeds = sm75_mtp_gemma_norm(\n"
                "                inputs_embeds, self.pre_fc_norm_embedding\n"
                "            )\n",
                1,
            )
            mtp_text = mtp_text.replace(
                "            hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(\n",
                "            hidden_states = sm75_mtp_gemma_norm(\n"
                "                hidden_states.flatten(-2), self.pre_fc_norm_hidden\n"
                "            ).view(\n",
                1,
            )
            assert "sm75_mtp_gemma_norm(\n                inputs_embeds" in mtp_text, (
                "MTP norm embedding 锚点未匹配 (上游 mtp.py 变更?)")
            assert "sm75_mtp_gemma_norm(\n                hidden_states" in mtp_text, (
                "MTP norm hidden 锚点未匹配 (上游 mtp.py 变更?)")
        # MTP 草稿头 stage-local: 类定义后 append hook, 删 PP>1 的两个 pp 分支
        # (末 rank 草稿头拿目标模型 pp 位置会炸 assert)。见 vllm_mtp_stage_local.py。
        mtp_hook = (
            "\n# vllm-turing overlay: MTP drafter stage-local (idempotent).\n"
            "import vllm.vllm_mtp_stage_local\n"
            "vllm.vllm_mtp_stage_local.maybe_apply(Qwen4ExpMultiTokenPredictor)\n"
        )
        if "vllm.vllm_mtp_stage_local.maybe_apply" not in mtp_text:
            mtp_text = mtp_text + mtp_hook
        mtp_file.write_text(mtp_text)

    # HC int8 权重 load 时 dequant: 往上游 qwen4_exp/nvidia/model.py 末尾 append 一行,
    # 给语言主干 Qwen4ExpModel (非顶层 CausalLM/ConditionalGeneration, 必被实例化, 是唯一
    # 干净注入点) 的 load_weights 打 patch, 把 int8 的 input_mix_weight_down/up 三元组
    # dequant 成 model dtype 普通 .weight (decoder 融合层 int8 down + bf16 block_inject
    # 混合, stock 量化匹配不了; MTP 草稿头 HC 是纯 bf16 不受影响)。见 vllm_hc_dequant.py。
    model_file = package_root / "models/qwen4_exp/nvidia/model.py"
    if model_file.is_file():
        hc_hook = (
            "\n# vllm-turing overlay: HC int8 权重 load 时 dequant (idempotent).\n"
            "import vllm.vllm_hc_dequant\n"
            "vllm.vllm_hc_dequant.maybe_apply(Qwen4ExpModel)\n"
        )
        hc_text = model_file.read_text()
        if "vllm.vllm_hc_dequant.maybe_apply" not in hc_text:
            model_file.write_text(hc_text + hc_hook)

    # firefly HC decode 标量 GEMV: 同 model.py 末尾再 append 一行, 给 Qwen4ExpModel
    # 的 __init__ 包一层, 实例化后遍历 named_modules 把 HC 三投影的 _gemm_impl 换成
    # SM75 标量 GEMV custom op (单流 decode 命中, prefill/batch>1/非 sm75 回退原
    # cuBLAS)。与上面 hc_dequant 链式共存 (那个包 load_weights、这个包 __init__, 不同
    # 阶段)。env VLLM_FIREFLY_HC 默认关 → maybe_apply 内直接 no-op, 上游原路。见
    # vllm_firefly_hc.py。
    model_file2 = package_root / "models/qwen4_exp/nvidia/model.py"
    if model_file2.is_file():
        fhc_hook = (
            "\n# vllm-turing overlay: firefly HC decode 标量 GEMV (idempotent).\n"
            "import vllm.vllm_firefly_hc\n"
            "vllm.vllm_firefly_hc.maybe_apply(Qwen4ExpModel)\n"
        )
        fhc_text = model_file2.read_text()
        if "vllm.vllm_firefly_hc.maybe_apply" not in fhc_text:
            model_file2.write_text(fhc_text + fhc_hook)

    third_party_source = source_root / "third_party/flash_qla_sm75"
    third_party_destination = package_root / "third_party/flash_qla_sm75"
    if not third_party_source.is_dir():
        raise FileNotFoundError(third_party_source)
    shutil.copytree(
        third_party_source,
        third_party_destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    if args.source_copy is not None:
        shutil.copytree(
            package_root,
            args.source_copy,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.so"),
        )
    print(package_root)


if __name__ == "__main__":
    main()
