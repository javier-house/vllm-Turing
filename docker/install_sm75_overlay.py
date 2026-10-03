# SPDX-License-Identifier: Apache-2.0
"""Install the reviewed SM75 overlay into the base image's vLLM package."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# 锚点注入表拆到 overlay_injections.py (控制本文件体积, 注入条目持续增长时
# 不拖累 install 逻辑的可读性)。apply_injections 从那里 import INJECTIONS。
from overlay_injections import INJECTIONS



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
        "engine/arg_utils.py",
        "entrypoints/serve/instrumentator/monitor.py",
        "entrypoints/serve/instrumentator/dashboard.html",
        "entrypoints/serve/instrumentator/test.html",
        "envs_sm75.py",
        "model_executor/kernels/linear/mixed_precision/marlin.py",
        "model_executor/layers/quantization/utils/firefly.py",
        "model_executor/layers/quantization/utils/firefly.cu",
        "model_executor/layers/quantization/utils/firefly_exl3.cu",
        "model_executor/layers/quantization/utils/firefly_int8.py",
        # NVFP4 权重 firefly prefill (E2M1→int8, 复用 cutlass int8 GEMM):
        # 调度层 + CUDA dequant kernel + NVFP4 Marlin kernel 覆盖 (加 firefly 混合
        # dispatch)。VLLM_FIREFLY_DIRECT 门控, off=上游 NVFP4 Marlin。
        # 见 vllm/model_executor/layers/quantization/utils/firefly_nvfp4.py。
        "model_executor/layers/quantization/utils/firefly_nvfp4.py",
        "model_executor/layers/quantization/utils/firefly_nvfp4.cu",
        "model_executor/kernels/linear/nvfp4/marlin.py",
        "model_executor/layers/quantization/exl3/__init__.py",
        "model_executor/layers/quantization/exl3/exl3.py",
        "distributed/device_communicators/firefly_allreduce.py",
        "distributed/device_communicators/firefly_allreduce.cu",
        "distributed/device_communicators/cuda_communicator.py",
        "v1/attention/backends/flashinfer.py",
        # tq KV prefill 走 FlashInfer ragged wrapper (O(N), sm75 替代 FA2 崩 /
        # SDPA OOM): 10 处改动含大段新代码, 走整文件覆盖而非锚点注入。
        # VLLM_FIREFLY_TQ 默认开, 0 回退上游 FA2/SDPA。见 PLAN-firefly-tq-prefill。
        "v1/attention/backends/turboquant_attn.py",
        "v1/core/sched/scheduler_sm75.py",
        "v1/engine/async_llm.py",
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
