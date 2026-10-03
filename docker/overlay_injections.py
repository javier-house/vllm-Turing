# SPDX-License-Identifier: Apache-2.0
"""SM75 overlay 锚点注入表 (从 install_sm75_overlay.py 拆出, 控制其体积).

每条 = (相对路径, [(锚点, 替换, probe), ...])。
- 锚点取上游正文中唯一出现的稳定行(升级后仍易识别), 替换写回 SM75 改动后的行;
- 用 list-of-(anchor, replace) 而非单个锚点, 以支持同一文件多处注入(如 kv_cache
  的两处 logger 降级、metrics 的豁免列表 + 尾 hook)。
- 每条 (anchor, replacement, probe): 用 anchor 定位、replace 成 replacement;
  probe 是"注入后才出现且唯一"的片段, 用于幂等判断(已注入则跳过)。不能用
  anchor 判断幂等——部分 replacement 保留了 anchor 本身(如 model.py 的函数插入
  保留注释行), 第二次跑 anchor 仍在会重复注入。probe 必须只在注入后存在。
"""

from __future__ import annotations

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
                "        ngram_context = self.ngram_context[:num_reqs]\n"
                "        ngram_context.fill_(self.ngram_eos_token_id)\n"
                "        model_inputs.update(\n"
                "            query_start_loc=query_start_loc,\n"
                "            ngram_context=ngram_context,\n"
                "        )\n"
                "        return model_inputs\n",
                "        ngram_context = self.ngram_context[:num_reqs]\n"
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
    (
        # gdn_attn: SM75 加 flashqla_sm75 到 Literal 类型注解。
        # 上游 Literal["triton", "flashinfer", "cutedsl"] → 加 "flashqla_sm75"。
        # 其余差异 (super().__init__ / num_decode_draft_tokens_cpu) 是 v0.29→v0.30
        # 版本漂移, 非 SM75 改动, 上游已自动处理。
        "v1/attention/backends/gdn_attn.py",
        [(
            '        self.gdn_prefill_backend: Literal["triton", "flashinfer", "cutedsl"]\n',
            '        self.gdn_prefill_backend: Literal[\n'
            '            "triton", "flashinfer", "cutedsl", "flashqla_sm75"\n'
            '        ]\n',
            '"flashqla_sm75"',
        )],
    ),
    (
        # qwen_gdn_linear_attn: SM75 FlashQLA 支持 + piecewise cudagraph 兼容 +
        # torch.compile fake。12 处注入, 上游正文不整文件覆盖。
        # 跳过的 2 处纯 docstring 措辞改动 (Hopper→high-CC) 不影响功能。
        "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        [
            (
                # ① import: 加 eager_break_during_capture (piecewise cudagraph 断点)
                'from vllm._aiter_ops import rocm_aiter_ops\n',
                'from vllm._aiter_ops import rocm_aiter_ops\n'
                'from vllm.compilation.breakable_cudagraph import eager_break_during_capture\n',
                'from vllm.compilation.breakable_cudagraph import eager_break_during_capture',
            ),
            (
                # ② _resolve 前插入 flashqla_sm75 支持检查 helper
                'def _resolve_gdn_prefill_backend(\n',
                'def _flashqla_sm75_prefill_unsupported_reason(\n'
                '    vllm_config: VllmConfig,\n'
                ') -> str | None:\n'
                '    """返回 None 表示 FlashQLA-SM75 prefill 可用, 否则返回原因。"""\n'
                '    if not current_platform.is_cuda():\n'
                '        return "the platform is not CUDA"\n'
                '    if not current_platform.is_device_capability(75):\n'
                '        return "the device compute capability is not 7.5"\n'
                '\n'
                '    text_config = vllm_config.model_config.hf_text_config\n'
                '    head_k_dim = getattr(text_config, "linear_key_head_dim", None)\n'
                '    head_v_dim = getattr(text_config, "linear_value_head_dim", None)\n'
                '    if head_k_dim != 128 or head_v_dim != 128:\n'
                '        return f"the GDN head dimensions are K={head_k_dim}, V={head_v_dim}"\n'
                '    if vllm_config.model_config.dtype != torch.float16:\n'
                '        return f"the model activation dtype is {vllm_config.model_config.dtype}"\n'
                '\n'
                '    try:\n'
                '        from vllm.third_party.flash_qla_sm75 import (  # noqa: F401\n'
                '            chunk_gated_delta_rule_fwd_sm70_vlk_varlen,\n'
                '        )\n'
                '    except ImportError as exc:\n'
                '        return f"the vendored FlashQLA package cannot be imported: {exc}"\n'
                '    return None\n'
                '\n'
                '\n'
                'def _resolve_gdn_prefill_backend(\n',
                'def _flashqla_sm75_prefill_unsupported_reason(',
            ),
            (
                # ③ _resolve 签名: 加 "flashqla_sm75" 到 Literal
                ') -> tuple[str, Literal["triton", "flashinfer", "cutedsl"]]:\n',
                ') -> tuple[\n'
                '    str,\n'
                '    Literal["triton", "flashinfer", "cutedsl", "flashqla_sm75"],\n'
                ']:\n',
                'Literal["triton", "flashinfer", "cutedsl", "flashqla_sm75"]',
            ),
            (
                # ④ _resolve 函数体: flashqla_sm75 优先检查 (在 flashinfer 分支前)
                '    if backend in ["flashinfer", "auto"] and supports_flashinfer:\n',
                '    if (\n'
                '        backend == "flashqla_sm75"\n'
                '        and _flashqla_sm75_prefill_unsupported_reason(vllm_config) is None\n'
                '    ):\n'
                '        return backend, "flashqla_sm75"\n'
                '\n'
                '    if backend in ["flashinfer", "auto"] and supports_flashinfer:\n',
                'return backend, "flashqla_sm75"',
            ),
            (
                # ⑤ chosen dict: 加 FlashQLA-SM75 显示名
                '        "flashinfer": "FlashInfer",\n',
                '        "flashinfer": "FlashInfer",\n'
                '        "flashqla_sm75": "FlashQLA-SM75",\n',
                '"FlashQLA-SM75"',
            ),
            (
                # ⑥ @CustomOp.register 前插入 flashqla_sm75 kernel wrapper 函数
                '@CustomOp.register("chunk_gated_delta_rule")\n',
                'def flashqla_sm75_chunk_gated_delta_rule(\n'
                '    q: torch.Tensor,\n'
                '    k: torch.Tensor,\n'
                '    v: torch.Tensor,\n'
                '    g: torch.Tensor,\n'
                '    beta: torch.Tensor,\n'
                '    initial_state: torch.Tensor,\n'
                '    output_final_state: bool,\n'
                '    cu_seqlens: torch.Tensor | None = None,\n'
                '    use_qk_l2norm_in_kernel: bool = True,\n'
                '    core_attn_out: torch.Tensor | None = None,\n'
                '):\n'
                '    """FlashQLA-SM75 prefill kernel wrapper (vendored, sm75-only)."""\n'
                '    from vllm.third_party.flash_qla_sm75 import (\n'
                '        chunk_gated_delta_rule_fwd_sm70_vlk_varlen,\n'
                '    )\n'
                '\n'
                '    if use_qk_l2norm_in_kernel:\n'
                '        q = l2norm_fwd(q)\n'
                '        k = l2norm_fwd(k)\n'
                '    if cu_seqlens is None:\n'
                '        cu_seqlens = torch.tensor(\n'
                '            [0, q.shape[1]], device=q.device, dtype=torch.int32\n'
                '        )\n'
                '    elif cu_seqlens.dtype != torch.int32:\n'
                '        cu_seqlens = cu_seqlens.to(torch.int32)\n'
                '\n'
                '    output = None\n'
                '    if core_attn_out is not None:\n'
                '        candidate = core_attn_out[: q.shape[1]].unsqueeze(0)\n'
                '        if candidate.shape == v.shape and candidate.is_contiguous():\n'
                '            output = candidate\n'
                '\n'
                '    return chunk_gated_delta_rule_fwd_sm70_vlk_varlen(\n'
                '        q=q.contiguous(),\n'
                '        k=k.contiguous(),\n'
                '        v=v.contiguous(),\n'
                '        g=g.contiguous(),\n'
                '        beta=beta.contiguous(),\n'
                '        cu_seqlens=cu_seqlens.contiguous(),\n'
                '        initial_state=initial_state.contiguous(),\n'
                '        scale=q.shape[-1] ** -0.5,\n'
                '        output_final_state=output_final_state,\n'
                '        validate_cu_seqlens=False,\n'
                '        output=output,\n'
                '    )\n'
                '\n'
                '\n'
                '@CustomOp.register("chunk_gated_delta_rule")\n',
                'def flashqla_sm75_chunk_gated_delta_rule(',
            ),
            (
                # ⑦ __init__ if 条件: 加 flashqla_sm75 + 动态 reason
                '        if backend in ("flashinfer", "cutedsl") and active_backend != backend:\n',
                '        if (\n'
                '            backend in ("flashinfer", "cutedsl", "flashqla_sm75")\n'
                '            and active_backend != backend\n'
                '        ):\n',
                '("flashinfer", "cutedsl", "flashqla_sm75")',
            ),
            (
                # ⑧ __init__ flashinfer 分支后插入 flashqla_sm75 分支
                '        if active_backend == "flashinfer":\n'
                '            self._forward_method = self.forward_cuda\n',
                '        if active_backend == "flashinfer":\n'
                '            self._forward_method = self.forward_cuda\n'
                '        elif active_backend == "flashqla_sm75":\n'
                '            self._forward_method = self.forward_flashqla_sm75\n',
                'self.forward_flashqla_sm75',
            ),
            (
                # ⑨ forward_cuda return 后插入 forward_flashqla_sm75 方法
                '        return o, final_state\n'
                '\n'
                '    def forward_native(\n',
                '        return o, final_state\n'
                '\n'
                '    def forward_flashqla_sm75(\n'
                '        self,\n'
                '        q: torch.Tensor,\n'
                '        k: torch.Tensor,\n'
                '        v: torch.Tensor,\n'
                '        g: torch.Tensor,\n'
                '        beta: torch.Tensor,\n'
                '        initial_state: torch.Tensor,\n'
                '        output_final_state: bool,\n'
                '        cu_seqlens: torch.Tensor | None = None,\n'
                '        chunk_indices: torch.Tensor | None = None,\n'
                '        chunk_offsets: torch.Tensor | None = None,\n'
                '        use_qk_l2norm_in_kernel: bool = True,\n'
                '        core_attn_out: torch.Tensor | None = None,\n'
                '    ):\n'
                '        capability = (\n'
                '            torch.cuda.get_device_capability(q.device) if q.is_cuda else None\n'
                '        )\n'
                '        unsupported = (\n'
                '            capability != (7, 5)\n'
                '            or q.dtype != torch.float16\n'
                '            or k.dtype != torch.float16\n'
                '            or v.dtype != torch.float16\n'
                '            or q.shape[-1] != 128\n'
                '            or v.shape[-1] != 128\n'
                '        )\n'
                '        if unsupported:\n'
                '            logger.warning_once(\n'
                '                "FlashQLA-SM75 received unsupported runtime tensors; "\n'
                '                "falling back to Triton/FLA for this prefill call."\n'
                '            )\n'
                '            return self.forward_native(\n'
                '                q=q,\n'
                '                k=k,\n'
                '                v=v,\n'
                '                g=g,\n'
                '                beta=beta,\n'
                '                initial_state=initial_state,\n'
                '                output_final_state=output_final_state,\n'
                '                cu_seqlens=cu_seqlens,\n'
                '                chunk_indices=chunk_indices,\n'
                '                chunk_offsets=chunk_offsets,\n'
                '                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,\n'
                '                core_attn_out=core_attn_out,\n'
                '            )\n'
                '        return flashqla_sm75_chunk_gated_delta_rule(\n'
                '            q=q,\n'
                '            k=k,\n'
                '            v=v,\n'
                '            g=g,\n'
                '            beta=beta,\n'
                '            initial_state=initial_state,\n'
                '            output_final_state=output_final_state,\n'
                '            cu_seqlens=cu_seqlens,\n'
                '            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,\n'
                '            core_attn_out=core_attn_out,\n'
                '        )\n'
                '\n'
                '    def forward_native(\n',
                'def forward_flashqla_sm75(',
            ),
            (
                # ⑩ warmup else 块: flashqla_sm75 跳过 warmup (vendor kernel 无需 autotune)
                '        else:\n'
                '            logger.debug(\n'
                '                "GDN prefill kernel warmup (T=%d) completed for layer %s",\n'
                '                T,\n'
                '                self.prefix,\n'
                '            )\n'
                '        finally:\n',
                '        else:\n'
                '            if self.gdn_prefill_backend == "flashqla_sm75":\n'
                '                raise\n'
                '            logger.debug(\n'
                '                "GDN prefill kernel warmup (T=%d) completed for layer %s",\n'
                '                T,\n'
                '                self.prefix,\n'
                '            )\n'
                '        finally:\n',
                'if self.gdn_prefill_backend == "flashqla_sm75":',
            ),
            (
                # ⑪ qwen_gdn_attention_core: 加 @eager_break_during_capture
                # (GDN core 内含 host 侧状态操作, 不可被 capture 进 breakable graph)
                '\ndef qwen_gdn_attention_core(\n',
                '\n'
                '# piecewise cudagraph 兼容: GDN core 内含 host 侧状态操作 (按请求变的 SSM state、\n'
                '# flashqla metadata 构建), 不可被 capture 进 breakable graph —— 用\n'
                '# eager_break_during_capture 在 capture 期自动切成 eager 断点。\n'
                '@eager_break_during_capture\n'
                'def qwen_gdn_attention_core(\n',
                '@eager_break_during_capture\ndef qwen_gdn_attention_core',
            ),
            (
                # ⑫ qwen_gdn_attention_core_fused_norm_packed: 加 @eager_break_during_capture
                '\ndef qwen_gdn_attention_core_fused_norm_packed(\n',
                '\n'
                '@eager_break_during_capture\n'
                'def qwen_gdn_attention_core_fused_norm_packed(\n',
                '@eager_break_during_capture\ndef qwen_gdn_attention_core_fused_norm_packed',
            ),
        ],
    ),
]
