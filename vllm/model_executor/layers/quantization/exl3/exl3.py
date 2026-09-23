# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 (exllamav3 trellis) 在线解码 serving —— sm75 原生实现 (firefly 内核)。

与 vllm-exl3 插件不同: 不用 exllamav3 的 LinearEXL3/cp.async GEMM (sm80-only),
而是自研 firefly_exl3.cu 的在线解码 kernel (sm75-safe, dp4a/shf/bfe), 输出
int8 [N, K] + per-channel scale, 直接喂 firefly 的 int8 GEMM (cutlass_scaled_mm,
sm75 原生 IMMA)。

权重保持 EXL3 trellis 压缩态常驻 (不反量化副本, 因反量化 fp16/int8 都装不下),
forward 时在线解码。数学:
    W_hat = diag(suh) · H128 · decode(trellis) · H128 · diag(svh)   # fp16 [K, N]
    w_int8[n, k] = clamp(round(W_hat[k, n] / c_n[n]), -127, 127)     # int8 [N, K]
    c_n[n] = amax_k |W_hat[k, n]| / 127                              # per-channel
    y = cutlass_scaled_mm(scaled_int8_quant(x), w_int8.t(), c_n)
只支持 mul1 codebook (1.5B/27B dense 试金石), mcg 留里程碑 B。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = logging.getLogger(__name__)


def _linear_base_cls():
    """惰性 import LinearBase。

    不能在 exl3.py 顶层 import: serve 启动时本模块由 quantization/__init__.py 尾部
    hook 触发, 此刻 vllm.config 尚未加载完, 顶层 `from ...linear import LinearBase`
    会经 linear->vllm.config 撞循环 (ImportError: partially initialized)。运行时
    (get_quant_method/apply) vllm 已全加载, 惰性 import 安全。
    """
    from vllm.model_executor.layers.linear import LinearBase

    return LinearBase


def _int8_prefill_linear():
    """惰性 import firefly int8 GEMM (同 LinearBase, 避开顶层循环)。"""
    from vllm.model_executor.layers.quantization.utils.firefly import (
        int8_prefill_linear,
    )

    return int8_prefill_linear


def _tp_rank() -> int:
    """惰性 import TP rank (同 LinearBase 避开顶层循环; 运行时 vllm 已全加载)。

    TP 切分时每 rank 只装 1/tp 的 trellis/suh/svh 段 (128 倍数, Hadamard 按
    128x128 tile 自包含 -> 每 rank 独立解码免通信)。TP1 返回 0。
    """
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
    )

    return get_tensor_model_parallel_rank()

# ---------------------------------------------------------------------------
# firefly_exl3.cu 懒加载 (镜像 firefly.py 的 _load_cuda_mod)
# ---------------------------------------------------------------------------
_exl3_cuda_mod = None
_exl3_load_attempted = False


def _load_exl3_cuda_mod():
    """懒加载 firefly_exl3.cu (torch.utils.cpp_extension.load, sm75)。失败返 None。"""
    global _exl3_cuda_mod, _exl3_load_attempted
    if _exl3_load_attempted:
        return _exl3_cuda_mod
    _exl3_load_attempted = True
    try:
        # __file__ = .../quantization/exl3/exl3.py -> 两层 dirname 到 quantization
        cu_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "utils",
            "firefly_exl3.cu",
        )
        if not os.path.exists(cu_path):
            logger.warning("firefly_exl3.cu not found at %s", cu_path)
            return None
        from torch.utils.cpp_extension import load as _load_ext

        # 只编 sm_75 (Turing), 首次 forward JIT ~30-60s
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
        _exl3_cuda_mod = _load_ext(
            name="firefly_exl3",
            sources=[cu_path],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        logger.info("firefly EXL3 kernel loaded (sm75 online-decode + int8)")
    except Exception as e:  # noqa: BLE001
        logger.warning("firefly_exl3 CUDA ext load failed: %s", e)
        _exl3_cuda_mod = None
    return _exl3_cuda_mod


def _pad128(n: int) -> int:
    """EXL3 把矩阵两维 pad 到 128 倍数 (kernel 按 128 tile 解码/转置)。"""
    return (n + 127) // 128 * 128


class Exl3LinearMethod(QuantizeMethodBase):
    """EXL3 dense linear: 在线解码 -> int8 -> firefly cutlass_scaled_mm。

    覆盖 ColumnParallelLinear (qkv/gate_up) / RowParallelLinear (down) /
    MergedColumnParallelLinear (down/gate_up/qkv)。trellis/suh/svh 压缩态常驻。
    """

    def __init__(self, quant_config: "Exl3Config", bits: int | None = None):
        self.quant_config = quant_config
        self.bits = int(bits) if bits is not None else quant_config.bits

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        n_shards = len(output_partition_sizes)
        # EXL3 两维 pad 到 128 倍数 (kernel 按 128 tile)。首测 TP1, in/out 已是 128 倍数。
        K = _pad128(int(input_size_per_partition))
        out_sizes = [_pad128(int(s)) for s in output_partition_sizes]
        N = sum(out_sizes)
        k_words = self.bits * 16  # trellis 最后一维 (每 16 元素 bits*16 个 int16)

        if K % 128 or N % 128:
            raise ValueError(f"EXL3 K,N must be multiple of 128: K={K} N={N}")

        def _mk(data: torch.Tensor, suffix: str) -> torch.nn.Parameter:
            p = torch.nn.Parameter(data, requires_grad=False)
            p.weight_loader = self._make_weight_loader(suffix, n_shards, out_sizes, layer)
            return p

        # fused trellis 覆盖所有 shard: [K/16, ΣN/16, 16*bits] int16
        trellis = _mk(torch.empty(K // 16, N // 16, k_words, dtype=torch.int16), "trellis")
        # per-shard suh (输入通道 scale): [n_shards, K] fp16
        suh = _mk(torch.empty(n_shards, K, dtype=torch.float16), "suh")
        # per-shard svh (输出通道 scale, 拼接): [ΣN] fp16
        svh = _mk(torch.empty(N, dtype=torch.float16), "svh")
        # mul1 codebook 标记 (每 shard 一个 int32, 1.5B 全 mul1)
        mul1 = _mk(torch.zeros(n_shards, dtype=torch.int32), "mul1")

        layer.register_parameter("trellis", trellis)
        layer.register_parameter("suh", suh)
        layer.register_parameter("svh", svh)
        layer.register_parameter("mul1", mul1)

        # 元数据 (apply 用)
        layer._exl3_n_shards = n_shards
        layer._exl3_output_partition_sizes = out_sizes
        layer._exl3_K = K
        layer._exl3_true_out = [int(s) for s in output_partition_sizes]
        layer._exl3_bits = self.bits

    def _shard_idx(self, shard_id) -> int:
        """shard_id -> shard 索引。

        vLLM 0.29 的 Linear 层 weight_loader 是 3 参调用
        `param.weight_loader(param, loaded_weight, shard_id)` (linear.py:978),
        shard_id 由 WeightsMapper (orig_to_new_stacked) 解析后作为第 3 参直接传入:
        qkv -> "q"/"k"/"v"; gate_up -> int 0/1; 单 shard (row/lm_head) -> None。
        """
        if shard_id is None:
            return 0
        if isinstance(shard_id, str):
            return {"q": 0, "k": 1, "v": 2}[shard_id]
        return int(shard_id)

    def _make_weight_loader(self, suffix, n_shards, out_sizes, layer):
        """按 suffix 生成 weight_loader: 切 loaded_weight 写到 fused param 的 shard 段。"""

        def weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            shard_id=None,  # vLLM 0.29 第 3 参 (q/k/v 或 int); 兼容 generic 2 参调用
        ):
            # checkpoint 张量保持原 dtype (trellis int16 / suh,svh fp16 / mul1 int32)
            loaded = loaded_weight.detach()
            shard_idx = self._shard_idx(shard_id)
            if shard_idx >= n_shards:
                raise ValueError(
                    f"EXL3 {suffix}: shard_idx={shard_idx} out of range n_shards={n_shards}"
                )
            if suffix == "trellis":
                out_tiles_start = sum(s // 16 for s in out_sizes[:shard_idx])
                out_tiles_end = out_tiles_start + out_sizes[shard_idx] // 16
                dest = param.data[:, out_tiles_start:out_tiles_end, :]
            elif suffix == "suh":
                dest = param.data[shard_idx]
            elif suffix == "svh":
                out_start = sum(out_sizes[:shard_idx])
                out_end = out_start + out_sizes[shard_idx]
                dest = param.data[out_start:out_end]
            elif suffix == "mul1":
                # 标量标记, 每 shard 一位
                param.data[shard_idx] = int(loaded.reshape(-1)[0].item())
                return
            else:
                raise ValueError(f"unknown EXL3 suffix={suffix}")
            # TP 切分: checkpoint 存全量张量, 每 rank 只装 1/tp 段 (128 倍数, Hadamard
            # 按 128x128 tile 自包含 -> 每 rank 独立解码免通信)。逐维判断: loaded>dest
            # 则该维被 TP 切 -> 取本 rank 段 [tp_rank*len : (tp_rank+1)*len]; 相等 ->
            # replicated 取全量。覆盖 trellis(dim0=K,dim1=N)/suh(K)/svh(N) 及 qkv 输出。
            src = loaded
            for d in range(src.dim()):
                if src.shape[d] > dest.shape[d]:
                    start = _tp_rank() * dest.shape[d]
                    src = src.narrow(d, start, dest.shape[d])
                elif src.shape[d] < dest.shape[d]:
                    raise RuntimeError(
                        f"EXL3 {suffix} load: loaded dim{d}={src.shape[d]} < "
                        f"dest={dest.shape[d]} (unexpected) shard={shard_idx}"
                    )
            if tuple(src.shape) != tuple(dest.shape):
                raise RuntimeError(
                    f"EXL3 {suffix} load shape mismatch shard={shard_idx}: "
                    f"dest {tuple(dest.shape)} != src {tuple(src.shape)}"
                )
            dest.copy_(src.to(dest.dtype))

        return weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # 加载时算一次 per-channel c_n (每列 amax/127): c_n 只依赖权重(常量), 不在每步
        # 在线解码时重算 —— 否则 decode step 每步都白跑一次 amax 扫描。存成每层 [N]
        # 小数组(float32, 1.5B ~几十 KB/层), 不是常驻 int8 权重副本, 72GB MoE 也装得下。
        mod = _load_exl3_cuda_mod()
        if mod is None:
            return  # 无 CUDA ext (非 GPU 环境) 时跳过, apply 时会再报
        K = layer._exl3_K
        out_sizes = layer._exl3_output_partition_sizes
        n_shards = layer._exl3_n_shards
        bits = layer._exl3_bits
        N = sum(out_sizes)
        dev = layer.trellis.device
        c_n = torch.empty(N, dtype=torch.float32, device=dev)
        full_n16 = layer.trellis.shape[1]  # 全 N/16 (fused trellis 的 dim1)
        for i in range(n_shards):
            out_start = sum(out_sizes[:i])
            n_shard = out_sizes[i]
            # 解 shard 的 W_hat fp16 [K, n_shard] 求每列 amax (QUANT=false 全量解码)
            w_hat = torch.empty(K, n_shard, dtype=torch.float16, device=dev)
            mod.exl3_decode(
                layer.trellis.data,
                layer.suh.data[i],
                layer.svh.data[out_start : out_start + n_shard],
                w_hat,
                out_start // 16,  # nb_offset (16-tile N 起点)
                full_n16,  # packed_blocks_n (全 N/16)
                bits,
            )
            # per-channel (列) amax / 127 = GEMM scale_b
            c_n[out_start : out_start + n_shard] = (
                w_hat.abs().amax(dim=0) / 127.0
            )
        layer._exl3_c_n = c_n  # apply 用 (常驻小数组, 非 int8 权重)

    @torch.compiler.disable
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # @torch.compiler.disable: apply 内含 os.path (cu 懒加载) + 自定义 CUDA kernel
        # (mod.exl3_decode_to_int8) + per-shard 循环, torch.compile (dynamo) 无法 trace,
        # graph break 转 eager 执行。vLLM 默认开 inductor 编译, 不加会撞 dynamo 报错。
        mod = _load_exl3_cuda_mod()
        if mod is None:
            raise RuntimeError("firefly_exl3 CUDA kernel 未加载, 无法在线解码")
        int8_prefill_linear = _int8_prefill_linear()

        n_shards = layer._exl3_n_shards
        out_sizes = layer._exl3_output_partition_sizes
        K = layer._exl3_K
        bits = layer._exl3_bits
        dev = x.device
        out_dtype = x.dtype

        c_n_full = layer._exl3_c_n  # 加载时算好的 per-channel scale [N]
        full_n16 = layer.trellis.shape[1]  # 全 N/16 (fused trellis 的 dim1)
        # 激活 -> fp16 2D (GEMM 输入)
        x_fp16 = x.to(torch.float16)
        orig_shape = x_fp16.shape
        x2d = x_fp16.reshape(-1, x_fp16.shape[-1])
        if x2d.shape[1] != K:
            # 输入 pad 到 K (128 倍数), 多余位是 0 (suh 缩放后不影响真实维)
            x2d = F.pad(x2d, (0, K - x2d.shape[1]))

        outputs = []
        for i in range(n_shards):
            out_start = sum(out_sizes[:i])
            n_shard = out_sizes[i]
            # fused decode+quantize: 全量 trellis/suh/svh + shard 偏移(nb_offset),
            # 单 kernel 解完直接量化写 int8 (免 W_hat fp16 中间流量, 免每步 .contiguous()
            # 切片拷贝)。c_n 用加载时算好的本 shard 段。
            w_int8 = torch.empty(n_shard, K, dtype=torch.int8, device=dev)
            mod.exl3_decode_to_int8(
                layer.trellis.data,
                layer.suh.data[i],
                layer.svh.data[out_start : out_start + n_shard],
                c_n_full[out_start : out_start + n_shard],
                w_int8,
                out_start // 16,  # nb_offset (16-tile N 起点)
                full_n16,  # packed_blocks_n (全 N/16)
                bits,
            )
            # firefly int8 GEMM (cutlass_scaled_mm, sm75 原生 IMMA)
            y = int8_prefill_linear(x2d, w_int8, c_n_full[out_start : out_start + n_shard])
            outputs.append(y)

        y = torch.cat(outputs, dim=1) if n_shards > 1 else outputs[0]
        # trim 128 padding 列到真实 out
        true_out = sum(layer._exl3_true_out)
        if y.shape[1] != true_out:
            y = y[:, :true_out]
        y = y.to(out_dtype)
        if bias is not None:
            y = y + bias.to(y.dtype)
        return y.reshape(*orig_shape[:-1], -1)


# ---------------------------------------------------------------------------
# Exl3Config (注册 "exl3")
# ---------------------------------------------------------------------------
@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """EXL3 dense quant config。1.5B/27B 试金石: 全 dense Linear, mul1, 整数 bits。

    MoE/ngram PLE/QSA/MTP 留里程碑 B (此处对 RoutedExperts/embedding 返 None)。
    """

    # lm_head 在 EXL3 checkpoint 里是独立量化 (lm_head.trellis/suh/svh/mul1), 但
    # 1.5B 是 tie_word_embeddings (lm_head.weight 复用 bf16 embed_tokens)。vLLM tie
    # 后只建 weight param, lm_head 的 EXL3 辅助张量无对应 param → 加载报 "no
    # parameter lm_head.mul1"。忽略这些后缀, lm_head 走 tied bf16 embed (1.5B 单卡
    # 放得下 466MB fp16)。EXL3 原生 lm_head 解码 (在线 decode->logits) 留后续。
    # dense linear 的 .trellis 有对应 param, 走加载分支不受此 ignore 影响。
    _ignore_unexpected_suffixes = (
        ".q_scale",
        ".k_scale",
        ".v_scale",
        ".q_zero_point",
        ".k_zero_point",
        ".v_zero_point",
        ".trellis",
        ".suh",
        ".svh",
        ".mul1",
        ".mcg",
    )

    def __init__(self, bits: int = 4, codebook: str = "mul1", **kwargs):
        super().__init__()
        self.bits = int(bits)
        self.codebook = str(codebook)
        if self.codebook != "mul1":
            raise ValueError(f"EXL3 暂只支持 mul1, got {self.codebook!r} (mcg 留里程碑 B)")
        if self.bits not in (1, 2, 3, 4, 5, 6, 7, 8):
            raise ValueError(f"EXL3 不支持 bits={self.bits}")

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # sm75 原生 (在线解码 dp4a/shf/bfe + cutlass IMMA, 无 cp.async)
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        # EXL3 的 quantization_config.json: {"quant_method":"exl3","bits":4.0,
        # "codebook":"mul1","tensor_storage":{...}}。bits 可能是 float (4.0)
        bits = config.get("bits", 4)
        # 整数 bpw (4.0 -> 4); 非整数 (如 3.40) 暂不支持 (首测 1.5B=4.0)
        if isinstance(bits, float) and not bits.is_integer():
            raise ValueError(f"EXL3 首测只支持整数 bits, got {bits} (非整数 bpw 后续支持)")
        return cls(bits=int(bits), codebook=str(config.get("codebook", "mul1")))

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        method = str((hf_quant_cfg or {}).get("quant_method", "")).lower()
        return "exl3" if method == "exl3" else None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        # 惰性 import LinearBase (避开顶层循环; 此时 vllm 已全加载)
        if isinstance(layer, _linear_base_cls()):
            return Exl3LinearMethod(self)
        # lm_head/embedding: 首测 1.5B 是 tie_word_embeddings, lm_head 复用 bf16
        # embed_tokens (非量化), 返 None 走 UnquantizedEmbeddingMethod + tie_weights。
        # 若 checkpoint 的 lm_head 是独立 EXL3 量化 (非 tie), 加载阶段会报
        # "no parameter weight for lm_head.trellis", 届时再加 Exl3EmbeddingMethod
        # (在线解码 GEMV)。embedding 本体恒 bf16 非量化, 返 None。
        return None
