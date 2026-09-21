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
        # MTP draft model 漏声明 SupportsPP: PP>1 时 draft config 校验
        # verify_with_parallel_config -> is_pp_supported_model 抛
        # NotImplementedError("Pipeline parallelism is not supported for this model")。
        # 代码本身已为 PP 适配(forward 收 intermediate_tensors / __init__ 用
        # is_last_rank 处理 lm_head), 只缺声明, 对齐已支持 PP 的 Qwen4ExpMTP。
        # 三处: 1) import 引入 SupportsPP; 2) 类声明加 SupportsPP(Qwen3_5MoeMTP
        # 经 MRO 一并继承); 3) __init__ 暴露 make_empty_intermediate_tensors
        # (PP rank>0 profiling 需要, 且是 supports_pp 检出的 PP 必需属性)。
        "model_executor/models/qwen3_5_mtp.py",
        [(
            "    MultiModalEmbeddings,\n"
            "    SupportsMultiModal,\n"
            "    _require_is_multimodal,\n"
            ")",
            "    MultiModalEmbeddings,\n"
            "    SupportsMultiModal,\n"
            "    SupportsPP,\n"
            "    _require_is_multimodal,\n"
            ")",
            "    SupportsMultiModal,\n"
            "    SupportsPP,\n",
        ), (
            "class Qwen3_5MTP(LocalArgmaxMixin, nn.Module, SupportsMultiModal):",
            "class Qwen3_5MTP(LocalArgmaxMixin, nn.Module, SupportsMultiModal, SupportsPP):",
            "nn.Module, SupportsMultiModal, SupportsPP",
        ), (
            "        self.logits_processor = LogitsProcessor(config.vocab_size)",
            "        self.logits_processor = LogitsProcessor(config.vocab_size)\n"
            "        # PP: 暴露 make_empty_intermediate_tensors(PP rank>0 profiling\n"
            "        # 需要, 且是 supports_pp 检出的 PP 必需属性); 对齐 Qwen4ExpMTP。\n"
            "        self.make_empty_intermediate_tensors = (\n"
            "            self.model.make_empty_intermediate_tensors\n"
            "        )",
            "self.model.make_empty_intermediate_tensors",
        ), (
            # PP+MTP 运行时: drafter 整体放在最后一个 PP rank
            # (gpu_model_runner "put the entire draft model on the last PP rank"),
            # MTP 层并不按 PP 切分(range(num_mtp_layers) 全量)。但 forward 用全局
            # get_pp_group().is_first_rank 判分支, 在 PP>1 时 last rank 走
            # intermediate_tensors 分支, 而 drafter 的 dummy_run/propose 从不传该参
            # -> assert 崩 (KV profiling 阶段 Worker died)。MTP 恒整体驻留单个
            # rank, 恒走 embedding 路径即可; 尾部 is_last_rank 分支在该 rank 为
            # True, 正常返回 hidden_states。
            "        if get_pp_group().is_first_rank:\n",
            "        # SM75 PP+MTP: drafter 整体驻留单个 PP rank, 恒走 embedding 路径\n"
            "        if True:\n",
            "        # SM75 PP+MTP: drafter 整体驻留单个 PP rank, 恒走 embedding 路径\n",
        )],
    ),
    (
        # PP+MTP 第三层: sample_tokens 里 drafter MM embedding 守卫。
        # encoder_cache 只在 is_first_pp_rank 建 (model_runner 251-252), drafter 在
        # last pp rank 故 encoder_cache=None; 但守卫只看 speculator.supports_mm_inputs
        # (按 model_config 全局算, 恒 True) -> gather_mm_embeddings 需不存在的
        # encoder_runner -> AttributeError 崩 (sample_tokens 阶段)。TP4 (PP=1) first==last,
        # encoder_cache 正常, 不撞。加 encoder_cache is not None: 无 encoder 的 rank
        # 本就没有 MM embedding 可 gather, draft 走纯文本即可 (mm_inputs 默认 None)。
        "v1/worker/gpu/model_runner.py",
        [(
            "        if self.speculator is not None and self.speculator.supports_mm_inputs:\n",
            "        if (self.speculator is not None and self.speculator.supports_mm_inputs\n"
            "                and self.encoder_cache is not None):\n",
            "                and self.encoder_cache is not None):\n",
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
