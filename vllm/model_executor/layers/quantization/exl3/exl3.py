# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm project
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

per-shard bits (27B): 每个 shard (q/k/v/z/gate/up/down) 独立 bits (3/4/2/5/6 混用),
故 trellis 按 shard 独立存储 (各自 dim3=16*bits), 不用单一 flat 张量。TP 按 128 倍数
切 (Hadamard 按 128x128 tile 自包含 -> 每 rank 独立解码免通信)。
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
    """惰性 import TP rank (同 LinearBase 避开顶层循环; 运行时 vllm 已全加载)。"""
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
    )

    return get_tensor_model_parallel_rank()


def _tp_size() -> int:
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
    )

    return get_tensor_model_parallel_world_size()


def _hf_tie_word_embeddings() -> bool:
    """惰性读 hf_config.tie_word_embeddings (运行时 vllm 已全加载, 安全)。

    区分 tied lm_head (1.5B, 复用 bf16 embed, 无独立 EXL3 张量, 走 Unquantized)
    与 non-tied (27B, 独立 EXL3 量化, 走 Exl3EmbeddingMethod)。qwen3_5 的 flag 在
    text_config, qwen2 在顶层, 两层都查。
    """
    try:
        from vllm.config import get_current_vllm_config

        hc = get_current_vllm_config().model_config.hf_config
    except Exception:  # noqa: BLE001
        return True  # 拿不到时保守按 tied (走 Unquantized, 1.5B 安全)
    for c in (hc, getattr(hc, "text_config", None)):
        if c is not None and hasattr(c, "tie_word_embeddings"):
            return bool(c.tie_word_embeddings)
    return True


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
        cu_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "utils",
            "firefly_exl3.cu",
        )
        if not os.path.exists(cu_path):
            logger.warning("firefly_exl3.cu not found at %s", cu_path)
            return None
        from torch.utils.cpp_extension import load as _load_ext

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


# ---------------------------------------------------------------------------
# 在线解码 + firefly int8 GEMM 的通用 step (Linear / Embedding 共用)
# ---------------------------------------------------------------------------
def _exl3_forward_shards(
    mod, x: torch.Tensor, layer: torch.nn.Module, shards: list[dict],
) -> torch.Tensor:
    """逐 shard fused decode+quantize -> int8 GEMM, 拼回 [*, N]。

    shards: [{trellis, suh, svh, c_n, n_shard, bits}, ...] (每 shard 独立 bits)。
    x: [*, K] (K = 本 rank 输入维)。返回 [*, Σn_shard] (未 trim padding 由调用方裁)。
    """
    int8_prefill_linear = _int8_prefill_linear()
    K = layer._exl3_K
    dev = x.device
    x_fp16 = x.to(torch.float16)
    x2d = x_fp16.reshape(-1, x_fp16.shape[-1])
    if x2d.shape[1] != K:
        x2d = F.pad(x2d, (0, K - x2d.shape[1]))
    outputs = []
    for s in shards:
        w_int8 = torch.empty(s["n_shard"], K, dtype=torch.int8, device=dev)
        mod.exl3_decode_to_int8(
            s["trellis"], s["suh"], s["svh"], s["c_n"], w_int8,
            0, s["n_shard"] // 16, s["bits"],
        )
        outputs.append(int8_prefill_linear(x2d, w_int8, s["c_n"]))
    y = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
    return y


class Exl3LinearMethod(QuantizeMethodBase):
    """EXL3 dense linear: per-shard 在线解码 -> int8 -> firefly cutlass_scaled_mm。

    覆盖 ColumnParallelLinear / RowParallelLinear / MergedColumnParallelLinear /
    QKVParallelLinear。trellis/suh/svh 压缩态常驻 (per-shard bits)。
    """

    def __init__(self, quant_config: "Exl3Config", bits: int | None = None):
        self.quant_config = quant_config
        # per-shard bits 从加载的 trellis dim3 推断, 这里只存 (27B 各 shard 不同)。
        self.bits = int(bits) if bits is not None else None

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
        K = int(input_size_per_partition)      # 本 rank 输入维 (Row 切 / Column 全)
        K_full = int(input_size)               # 全输入维
        out_per_rank = [int(s) for s in output_partition_sizes]  # 本 rank 每 shard 输出
        tp = _tp_size()
        out_full = [s * tp for s in out_per_rank]  # 全每 shard 输出 (fused-on-disk 切分用)
        N = sum(out_per_rank)

        if K % 128 or N % 128:
            raise ValueError(f"EXL3 K,N must be multiple of 128: K={K} N={N}")

        def _mk(data: torch.Tensor, suffix: str) -> torch.nn.Parameter:
            p = torch.nn.Parameter(data, requires_grad=False)
            p.weight_loader = self._make_weight_loader(suffix, n_shards, layer)
            return p

        # trellis: 占位 param (真 per-shard 张量按 bits 不同存 layer._exl3_shards,
        # 加载时 stash)。占位只为让 vllm 路由 checkpoint 的 *.trellis 到本 weight_loader。
        trellis = _mk(torch.empty(0, dtype=torch.int16), "trellis")
        # suh: per-shard 输入通道 scale [n_shards, K] (每 shard 独立; 同输入故 q/k/v 同值)
        suh = _mk(torch.empty(n_shards, K, dtype=torch.float16), "suh")
        # svh: 输出通道 scale 拼接 [ΣN per rank]
        svh = _mk(torch.empty(N, dtype=torch.float16), "svh")
        # mul1: mul1 标记 (每 shard 一位, 1.5B/27B 全 mul1)。值不参与路径分叉, 但
        # checkpoint 有 *.mul1 张量, 必须注册 param 让 vllm 路由到本 weight_loader
        # (否则 getattr 回退到 module -> module.weight_loader -> param.data 崩)。
        mul1 = _mk(torch.zeros(n_shards, dtype=torch.int32), "mul1")

        layer.register_parameter("trellis", trellis)
        layer.register_parameter("suh", suh)
        layer.register_parameter("svh", svh)
        layer.register_parameter("mul1", mul1)

        # 元数据 (weight_loader / apply 用)
        layer._exl3_n_shards = n_shards
        layer._exl3_K = K
        layer._exl3_K_full = K_full
        layer._exl3_out_per_rank = out_per_rank
        layer._exl3_out_full = out_full
        layer._exl3_tp = tp
        layer._exl3_true_out = [int(s) for s in output_partition_sizes]
        layer._exl3_shards: list[dict] = [None] * n_shards  # 加载后填
        layer._exl3_c_n: list[torch.Tensor] = [None] * n_shards

    # ---- weight_loader (int / tuple / None shard_id, per-dim TP narrow) ----

    def _store_shard_trellis(
        self, layer, shard_idx: int, seg: torch.Tensor
    ) -> None:
        """存一个 shard 的 trellis (已 TP narrow 到本 rank) + 推断 bits。

        必须 .clone() 强制真实拷贝: trellis kernel 用 int4 (16B) 向量加载, 基址须
        16B 对齐。RowParallel 切 K (dim0, narrow 仍是连续) 时 .contiguous() 是 no-op,
        会沿用源张量 (safetensors mmap 文件偏移) 的未对齐基址 -> misaligned address;
        .clone() 总分配新对齐 buffer, 两种切法都安全。
        """
        bits = int(seg.shape[-1] // 16)
        layer._exl3_shards[shard_idx] = {
            "trellis": seg.detach().to(torch.int16).clone(),
            "bits": bits,
            "n_shard": layer._exl3_out_per_rank[shard_idx],
        }

    def _narrow_tp(self, t: torch.Tensor, dim: int, full: int, per_rank: int):
        """若该维是 TP 切的 (full>per_rank), 取本 rank 段; 否则原样。"""
        if t.shape[dim] > per_rank:
            start = _tp_rank() * per_rank
            return t.narrow(dim, start, per_rank)
        return t

    def _make_weight_loader(self, suffix, n_shards, layer):
        def weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            shard_id=None,
        ):
            loaded = loaded_weight.detach()
            K_per_rank = layer._exl3_K
            K_full = layer._exl3_K_full
            out_per_rank = layer._exl3_out_per_rank
            out_full = layer._exl3_out_full

            # 归一化 shard_id -> list[int]: None->[0](row/lm_head), int->[i](gate_up),
            # tuple->list (fused-on-disk, 如 GDN in_proj_qkv -> (0,1,2)), str->idx
            # (QKVParallel 的 "q"/"k"/"v").
            _STR2IDX = {"q": 0, "k": 1, "v": 2}

            def _norm(sid):
                if isinstance(sid, str):
                    if sid in _STR2IDX:
                        return _STR2IDX[sid]
                    return int(sid)
                return int(sid)

            if shard_id is None:
                sids = [0]
            elif isinstance(shard_id, (tuple, list)):
                sids = [_norm(i) for i in shard_id]
            else:
                sids = [_norm(shard_id)]
            for s in sids:
                if s >= n_shards:
                    raise ValueError(
                        f"EXL3 {suffix}: shard {s} out of range n_shards={n_shards}"
                    )

            if suffix == "mul1":
                # mul1 标记 (1.5B/27B 全 mul1), 值不参与路径分叉。仅吸收 checkpoint
                # 张量 (必须注册 param 避免 getattr 回退到 module), 写占位即可。
                param.data.copy_(torch.zeros_like(param.data))
                return

            if suffix == "trellis":
                # trellis [K/16, N/16, 16*bits]。N 维 (dim1) 按 shard 切 + 可能 TP 切;
                # K 维 (dim0) 若 Row 则 TP 切。fused-on-disk (tuple) 先按 full shard 切。
                if len(sids) > 1:
                    # 切 dim1 成 full shard 段
                    starts = [0]
                    for i, s in enumerate(sids):
                        starts.append(starts[-1] + out_full[s] // 16)
                    for j, s in enumerate(sids):
                        seg = loaded[:, starts[j]:starts[j + 1], :]
                        seg = self._narrow_tp(seg, 0, K_full // 16, K_per_rank // 16)
                        self._store_shard_trellis(layer, s, seg)
                else:
                    s = sids[0]
                    # 单 shard: loaded 即本 shard 全量 (或 fused 的整个 = 单 shard)
                    seg = self._narrow_tp(loaded, 0, K_full // 16, K_per_rank // 16)
                    seg = self._narrow_tp(
                        seg, 1, out_full[s] // 16, out_per_rank[s] // 16
                    )
                    self._store_shard_trellis(layer, s, seg)
                return

            if suffix == "suh":
                # suh [K] (输入通道 scale, 每 shard 同值; K 维若 Row 则 TP 切)
                seg = self._narrow_tp(loaded, 0, K_full, K_per_rank)
                for s in sids:
                    layer.suh.data[s].copy_(seg.to(torch.float16))
                return

            if suffix == "svh":
                # svh [N] (输出通道 scale, 按 shard + 可能 TP 切 N)
                if len(sids) > 1:
                    starts = [0]
                    for i, s in enumerate(sids):
                        starts.append(starts[-1] + out_full[s])
                    for j, s in enumerate(sids):
                        seg = loaded[starts[j]:starts[j + 1]]
                        seg = self._narrow_tp(
                            seg.unsqueeze(0), 0, out_full[s], out_per_rank[s]
                        ).squeeze(0)
                        o_start = sum(out_per_rank[:s])
                        layer.svh.data[o_start:o_start + out_per_rank[s]].copy_(
                            seg.to(torch.float16)
                        )
                else:
                    s = sids[0]
                    seg = self._narrow_tp(loaded, 0, out_full[s], out_per_rank[s])
                    o_start = sum(out_per_rank[:s])
                    layer.svh.data[o_start:o_start + out_per_rank[s]].copy_(
                        seg.to(torch.float16)
                    )
                return

            raise ValueError(f"unknown EXL3 suffix={suffix}")

        return weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # 加载时算每 shard 的 per-channel c_n (整列 amax/127): c_n 只依赖权重(常量),
        # 不在每步在线解码时重算。解该 shard 的 W_hat fp16 求列 amax。
        mod = _load_exl3_cuda_mod()
        if mod is None:
            return
        dev = layer.suh.device
        for i in range(layer._exl3_n_shards):
            s = layer._exl3_shards[i]
            if s is None:
                continue
            K = layer._exl3_K
            n_shard = s["n_shard"]
            w_hat = torch.empty(K, n_shard, dtype=torch.float16, device=dev)
            mod.exl3_decode(
                s["trellis"], layer.suh.data[i],
                layer.svh.data[sum(layer._exl3_out_per_rank[:i]):sum(
                    layer._exl3_out_per_rank[:i + 1])],
                w_hat, 0, n_shard // 16, s["bits"],
            )
            s["c_n"] = (w_hat.abs().amax(dim=0) / 127.0).float().contiguous()
            s["suh"] = layer.suh.data[i]
            s["svh"] = layer.svh.data[sum(layer._exl3_out_per_rank[:i]):sum(
                layer._exl3_out_per_rank[:i + 1])]

    @torch.compiler.disable
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # @torch.compiler.disable: apply 内含 os.path (cu 懒加载) + 自定义 CUDA kernel
        # (mod.exl3_decode_to_int8) + per-shard 循环, torch.compile (dynamo) 无法 trace。
        mod = _load_exl3_cuda_mod()
        if mod is None:
            raise RuntimeError("firefly_exl3 CUDA kernel 未加载, 无法在线解码")
        out_dtype = x.dtype
        orig_shape = x.shape
        shards = [s for s in layer._exl3_shards if s is not None]
        y = _exl3_forward_shards(mod, x, layer, shards)
        # trim 到真实输出 (若 shard 有 128 padding; 1.5B/27B 各维已 128 倍数则不变)
        true_out = sum(layer._exl3_true_out)
        if y.shape[-1] != true_out:
            y = y[..., :true_out]
        y = y.to(out_dtype)
        if bias is not None:
            y = y + bias.to(y.dtype)
        return y.reshape(*orig_shape[:-1], -1)


# ---------------------------------------------------------------------------
# lm_head (ParallelLMHead / VocabParallelEmbedding) 的 EXL3 量化
# ---------------------------------------------------------------------------
class Exl3EmbeddingMethod(QuantizeMethodBase):
    """lm_head EXL3: trellis 压缩态常驻, forward 在线解码 -> int8 -> GEMM -> logits。

    27B tie_word_embeddings=False, lm_head 独立 EXL3 量化 (bits 6, vocab 248320)。
    按 vocab (输出维) TP 切, hidden (输入维) 不切。逻辑等同单 shard Linear。
    """

    def __init__(self, quant_config: "Exl3Config", bits: int | None = None):
        self.quant_config = quant_config
        self.bits = int(bits) if bits is not None else None

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
        # lm_head [vocab, hidden]: vocab=output (按 TP 切), hidden=input (不切)
        K = int(input_size_per_partition)   # = hidden (不切)
        K_full = int(input_size)             # = hidden
        vocab_rank = int(output_partition_sizes[0])
        vocab_full = int(output_size)
        if K % 128 or vocab_rank % 128:
            raise ValueError(
                f"EXL3 lm_head K,vocab must be multiple of 128: K={K} vocab={vocab_rank}"
            )
        p = torch.nn.Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False)
        p.weight_loader = self._make_weight_loader(layer)
        layer.register_parameter("trellis", p)
        suh = torch.nn.Parameter(
            torch.empty(K, dtype=torch.float16), requires_grad=False
        )
        suh.weight_loader = self._make_suh_loader(layer)
        svh = torch.nn.Parameter(
            torch.empty(vocab_rank, dtype=torch.float16), requires_grad=False
        )
        svh.weight_loader = self._make_svh_loader(layer, vocab_full)
        layer.register_parameter("suh", suh)
        layer.register_parameter("svh", svh)
        layer._exl3_K = K
        layer._exl3_K_full = K_full
        layer._exl3_vocab_full = vocab_full
        layer._exl3_vocab_rank = vocab_rank
        layer._exl3_shards = [None]  # 单 shard

    def _make_weight_loader(self, layer):
        def weight_loader(param, loaded_weight, shard_id=None):
            loaded = loaded_weight.detach()
            # trellis [K/16, vocab/16, 16*bits]: K 不切, vocab (dim1) 按 TP 切
            seg = self._narrow(loaded, 1, layer._exl3_vocab_full // 16,
                               layer._exl3_vocab_rank // 16)
            bits = int(seg.shape[-1] // 16)
            layer._exl3_shards[0] = {
                "trellis": seg.to(torch.int16).contiguous(),
                "bits": bits,
                "n_shard": layer.svh.data.shape[0],
            }
        return weight_loader

    def _narrow(self, t, dim, full, per_rank):
        if t.shape[dim] > per_rank:
            start = _tp_rank() * per_rank
            return t.narrow(dim, start, per_rank)
        return t

    def _make_suh_loader(self, layer):
        def loader(param, loaded_weight, shard_id=None):
            # suh [hidden] 不切
            param.data.copy_(loaded_weight.detach().to(torch.float16))
        return loader

    def _make_svh_loader(self, layer, vocab_full):
        def loader(param, loaded_weight, shard_id=None):
            seg = self._narrow(loaded_weight.detach(), 0, vocab_full,
                               param.data.shape[0])
            param.data.copy_(seg.to(torch.float16))
        return loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        mod = _load_exl3_cuda_mod()
        if mod is None:
            return
        s = layer._exl3_shards[0]
        if s is None:
            return
        dev = layer.suh.device
        K = layer._exl3_K
        n_shard = s["n_shard"]
        w_hat = torch.empty(K, n_shard, dtype=torch.float16, device=dev)
        mod.exl3_decode(s["trellis"], layer.suh.data, layer.svh.data, w_hat,
                        0, n_shard // 16, s["bits"])
        s["c_n"] = (w_hat.abs().amax(dim=0) / 127.0).float().contiguous()
        s["suh"] = layer.suh.data
        s["svh"] = layer.svh.data

    @torch.compiler.disable
    def apply(self, layer, x, bias=None):
        mod = _load_exl3_cuda_mod()
        if mod is None:
            raise RuntimeError("firefly_exl3 CUDA kernel 未加载")
        s = layer._exl3_shards[0]
        if s is None:
            raise RuntimeError("EXL3 lm_head 未加载")
        x2d = x.to(torch.float16).reshape(-1, x.shape[-1])
        K = layer._exl3_K
        if x2d.shape[1] != K:
            x2d = F.pad(x2d, (0, K - x2d.shape[1]))
        w_int8 = torch.empty(s["n_shard"], K, dtype=torch.int8, device=x.device)
        mod.exl3_decode_to_int8(s["trellis"], s["suh"], s["svh"], s["c_n"],
                                w_int8, 0, s["n_shard"] // 16, s["bits"])
        y = _int8_prefill_linear()(x2d, w_int8, s["c_n"])
        if bias is not None:
            y = y + bias.to(y.dtype)
        return y


# ---------------------------------------------------------------------------
# Exl3Config (注册 "exl3")
# ---------------------------------------------------------------------------
@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """EXL3 dense quant config。1.5B/27B 试金石: 全 dense Linear, mul1, per-shard bits。"""

    # dense linear / lm_head 的 EXL3 辅助张量有对应 param, 走加载分支; 这里 ignore
    # 只在 "无对应 param" 时兜底 (如某些层没有 EXL3 量化时)。
    _ignore_unexpected_suffixes = (
        ".q_scale", ".k_scale", ".v_scale",
        ".q_zero_point", ".k_zero_point", ".v_zero_point",
        ".trellis", ".suh", ".svh", ".mul1", ".mcg",
    )

    def __init__(self, bits: int = 4, codebook: str = "mul1", **kwargs):
        super().__init__()
        # bits 可能是 per-tensor 均值 (27B=3.4, 非整数); 实际每 tensor 的 bits 从
        # 加载的 trellis dim3 推断, 这里只校验 codebook。
        self.bits = int(bits) if isinstance(bits, (int, float)) and float(bits).is_integer() else bits
        self.codebook = str(codebook)
        if self.codebook != "mul1":
            raise ValueError(f"EXL3 暂只支持 mul1, got {self.codebook!r} (mcg 留里程碑 B)")

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        # bits 可能是非整数均值 (27B=3.40); per-tensor 实际 bits 从 trellis dim3 推断。
        return cls(bits=config.get("bits", 4),
                   codebook=str(config.get("codebook", "mul1")))

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config: Any = None
    ) -> str | None:
        method = str((hf_quant_cfg or {}).get("quant_method", "")).lower()
        return "exl3" if method == "exl3" else None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        # 惰性 import (避开顶层循环; 此时 vllm 已全加载)
        if isinstance(layer, _linear_base_cls()):
            return Exl3LinearMethod(self)
        try:
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                ParallelLMHead,
            )
        except ImportError:
            ParallelLMHead = ()
        if ParallelLMHead and isinstance(layer, ParallelLMHead):
            # non-tied lm_head (27B, 独立 EXL3 量化) 走在线解码; tied (1.5B, 复用
            # bf16 embed, 无 EXL3 张量) 返 None 走 UnquantizedEmbeddingMethod。
            if not _hf_tie_word_embeddings():
                return Exl3EmbeddingMethod(self)
        return None
