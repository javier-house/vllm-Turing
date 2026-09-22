# SPDX-License-Identifier: Apache-2.0
"""vllm_ple_mmap —— 用 NVMe mmap 服务 qwen4_exp 的 PLE(ngram) 嵌入大表。

背景: Qwen3.8-Flash-Next (qwen4_exp) 有一个 105GB 级的 ngram PLE 表
(``ngram_embedding`` 128 片, 每片 [2500012, 160] BF16)。vLLM 默认把它整块
常驻 GPU (VocabParallelEmbedding), 11G 的 2080Ti 根本装不下。但每个 token 只
lookup 16 行 x 160 列 (ngram_heads=16, head_dim=160), 所以这张表可以整盘留在
NVMe、靠内核 page cache 换页, 按行 gather —— 正是 llama.cpp 对 GGUF 的 mmap
做法。

做法: ``VLLM_PLE_MMAP=1`` 时 ``maybe_apply(cls)`` 给上游
``Qwen4ExpNGramEmbedding`` 打三处 patch:
  * ``__init__``: 跑原构造, 但把其中的 ``PLEVocabParallelEmbedding`` 临时换成
    ``_MmapNgramEmbedding`` 占位 (不分配那张大 GPU 张量);
  * ``load_weights``: 跳过所有 ``ngram_embedding.shard_N.weight`` (不 copy、不
    进显存, 直接由磁盘 mmap 服务), 其余张量照原逻辑加载, 随后开 mmap 表;
  * ``forward``: 把 "算 ngram_ids + mmap gather + pageable H2D" 包进 custom op
    ``vllm::qwen4_exp_ple_mmap_lookup``, 经 ``no_compile_layers[layer_name]`` 拿
    到 layer 在图外执行 —— gather 是 CPU 活 + pageable H2D, 进不了 CUDA graph。
    写法对齐上游 ``ple_layer.py`` 末尾已有的 ``qwen4_exp_compute_ple_ngram_ids`` /
    ``qwen4_exp_ple_short_conv`` 两个 op。

只支持 BF16/F16 透传 (Minachist 模型是 BF16 无 scale, 必须能跑) + FP8 (保留)。
默认关 (未设 ``VLLM_PLE_MMAP`` 直接 return), 对现有行为零影响。

Knobs (env):
  VLLM_PLE_MMAP=1             总开关
  VLLM_PLE_MMAP_WORKERS=32    gather 线程数 (page fault 跨线程 overlap)
  VLLM_PLE_MMAP_CHUNK=2048    每 gather 任务的行数
  VLLM_PLE_MMAP_RANDOM=1      mmap 标 MADV_RANDOM 关内核预读 (默认 1)
  VLLM_PLE_MMAP_PREWARM=0     1=加载时把整表流读一遍填 page cache
"""

from __future__ import annotations

import fcntl
import glob
import json
import logging
import math
import mmap as _mmap
import os
import re
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("vllm.ple_mmap")

ENV_ENABLE = "VLLM_PLE_MMAP"
ENV_MODE = "VLLM_PLE_MMAP_MODE"
ENV_RANDOM = "VLLM_PLE_MMAP_RANDOM"
_OP_NAME = "qwen4_exp_ple_mmap_lookup"

# safetensors 字符串 dtype -> torch dtype。BF16/F16 原样透传 (不反量化);
# FP8 保留 (参考实现逻辑, 非重点)。
_FP8_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
}
_NON_FP8_DTYPES = {"BF16", "F16"}

# 每 dtype 的字节数 (算 row_bytes 用: fp8=1, bf16/f16=2)。
_ITEMSIZE = {
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U8": 1,
    "I8": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
}


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "1" if default else "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def mode() -> str:
    """ngram(PLE) 表的 offload 模式, 三选一可独立设置 (对齐 1Cat 三模式):

    - ``disk`` (默认 / 或 ``VLLM_PLE_MMAP=1``): 磁盘 ``np.memmap``, page cache 换页。
      表 ~95G 走 NVMe, 不进显存/常驻 RAM, 适合主机内存有限时。
    - ``mem``: 整表读进 RAM (numpy 连续数组), gather 纯内存无磁盘 I/O; 需 ~95G 主机
      内存 (二号机 238G 空闲内存装得下), 比 disk 稳 (无换页抖动)。
    - ``vram``: 不 offload, 整表 load 进 GPU (占 ~95G 显存, 多卡 PP 下每卡都持有),
      仅小表或验证时用。

    优先级: 显式 ``VLLM_PLE_MMAP_MODE`` > ``VLLM_PLE_MMAP`` (1→disk, 向后兼容) >
    默认 disk。未设任何开关时退回 ``vram`` (原行为: 表 load 进 GPU, 不打 patch)。
    """
    m = os.environ.get(ENV_MODE, "").strip().lower()
    if m:
        if m in ("vram", "gpu", "none", "0", "off"):
            return "vram"
        if m in ("mem", "memory", "ram", "cpu"):
            return "mem"
        if m in ("disk", "mmap", "nvme"):
            return "disk"
        logger.warning("PLE mmap: 未知 MODE %r, 退回 disk", m)
        return "disk"
    if os.environ.get(ENV_ENABLE, "0").strip().lower() in ("1", "true", "yes", "on"):
        return "disk"
    return "vram"


def enabled() -> bool:
    """是否启用 PLE 表 offload (disk/mem 都 offload); vram = 不 offload, 不打 patch。"""
    return mode() in ("disk", "mem")


def _apply_madv_random(handler: object) -> None:
    """把 PLE 表的 mmap 标成随机访问, 关内核预读。

    PLE 表 (100GB 级) 的访问是稀疏随机行 gather —— 每次只碰几个 4K 页, 顺序预读
    读进来的邻居页几乎用不上, 纯浪费 NVMe 带宽和 page cache。内核的
    read_ahead_kb 是整盘队列级参数, 无法按文件区分, 故用 per-VMA 的 MADV_RANDOM
    精确只关这张表。设 ``VLLM_PLE_MMAP_RANDOM=0`` 可关。
    """
    if not _env_flag(ENV_RANDOM, True):
        return
    if handler is None:
        return
    try:
        handler.madvise(_mmap.MADV_RANDOM)
    except Exception:
        logger.warning("PLE mmap: madvise(MADV_RANDOM) 失败, 退回内核预读", exc_info=True)


# --------------------------------------------------------------------------- #
# safetensors 头部解析 (不依赖 safetensors 包: 我们要裸文件字节 offset)
# --------------------------------------------------------------------------- #
def parse_safetensors_header(path: str) -> tuple[dict, int]:
    """返回 (header_dict, data_start_offset)。"""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


class MmapPleTable:
    """在切成 ``split_ngram_parts`` 片文件上的行 gather。

    ``shards``: {shard_index: (path, 绝对字节 offset, rows)}。第 i 片存全局行
    ``[i*shard_size, i*shard_size + rows)`` (vLLM ``copy_ple_embedding_shard_``
    的布局)。

    mem 模式走 /dev/shm 跨 rank 共享 (见下方 __init__ mem 分支): 同表的所有
    worker 进程只占一份物理页。
    """

    def __init__(
        self,
        shards: dict[int, tuple[str, int, int]],
        shard_size: int,
        row_bytes: int,
        torch_dtype: torch.dtype,
        workers: int = 32,
        chunk: int = 2048,
        mem: bool = False,
        key: str = "",
    ) -> None:
        if not shards:
            raise ValueError("no PLE shards")
        self.shard_size = int(shard_size)
        self.row_bytes = int(row_bytes)
        self.torch_dtype = torch_dtype
        self.chunk = max(1, int(chunk))
        max_idx = max(shards)
        self.paths: list[str | None] = [None] * (max_idx + 1)
        self.mm: list[np.memmap | None] = [None] * (max_idx + 1)
        self.rows_total = 0
        for idx, (path, offset, rows) in shards.items():
            self.paths[idx] = path
            self.mm[idx] = np.memmap(
                path, dtype=np.uint8, mode="r", offset=offset, shape=(rows, row_bytes)
            )
            _apply_madv_random(getattr(self.mm[idx], "_mmap", None))
            self.rows_total += rows
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        # mem 模式: 整表放进 /dev/shm 共享表文件, 跨 TP rank (独立 worker 进程)
        # 共享同一份物理页 —— 单份 ~95G (不是每 rank 一份), 磁盘只读一遍。
        # owner 判定: 对表文件 flock 非阻塞排他; 拿到者 = owner (填表并终身持锁),
        # 拿不到者 = waiter (等 owner 写完 ready 标志后 attach 同一文件)。
        # owner 崩了内核自动放锁 + ready 标志缺失 → waiter 升 owner 重填 (自愈)。
        self.mem: np.ndarray | None = None
        if mem:
            import hashlib

            # key 含模型路径+形状: 换模型/换表自动换表文件, 旧表成孤儿可被清。
            h = hashlib.sha1(
                f"{key}:{self.rows_total}:{self.row_bytes}".encode()
            ).hexdigest()[:16]
            tbl = f"/dev/shm/ple_mmap_{h}"
            ready = f"{tbl}.ready"
            # owner 选举: 对表文件 flock 非阻塞排他。拿到 = owner, 且必须终身
            # 持锁 (存到 self, 进程活多久锁多久; 内核在进程死亡时自动释放) ——
            # waiter 用同法试锁, 锁被占即证明 "owner 活着且已填完" (ready 与锁
            # 同时成立), owner 崩 → 锁自动放 → 下次启动无 ready → 新 owner 重填。
            lock_fd = os.open(tbl, os.O_RDWR | os.O_CREAT, 0o666)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._shm_lock_fd = lock_fd  # owner: 终身持锁 (勿 close)
            except OSError:
                os.close(lock_fd)
                self._shm_lock_fd = None
            me_owner = self._shm_lock_fd is not None
            # 热启动复用: 上次 owner 填好的表还在 /dev/shm (ready 存在 + 大小
            # 匹配), 直接 attach, 跳过 ~95G 重填 (重启秒级)。表内容只由
            # key(模型路径)+形状决定, 同一 key 内容必一致。
            reuse = (
                os.path.exists(ready)
                and os.path.getsize(tbl) == self.rows_total * self.row_bytes
            )
            if me_owner and not reuse:
                # owner: 填表。ftruncate 到准确大小后按片拷入, 最后 fsync +
                # 写 ready 标志 (waiter 看到 ready 才 attach)。
                logger.info(
                    "PLE mmap(mem): 本进程为 owner, 整表写入共享表 %s (%.1f GiB)...",
                    tbl,
                    self.rows_total * self.row_bytes / 2**30,
                )
                try:
                    if os.path.exists(ready):
                        os.unlink(ready)
                except OSError:
                    pass
                fd = os.open(tbl, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o666)
                os.ftruncate(fd, self.rows_total * self.row_bytes)
                buf = np.memmap(tbl, dtype=np.uint8, mode="r+",
                                shape=(self.rows_total, self.row_bytes))
                for idx in range(max_idx + 1):
                    mm = self.mm[idx]
                    if mm is None:
                        continue
                    n = mm.shape[0]
                    start = idx * self.shard_size
                    buf[start : start + n] = np.asarray(mm)
                buf.flush()
                with open(ready, "w") as f:
                    f.write(str(self.rows_total))
                    f.flush()
                    os.fsync(f.fileno())
                del buf
                self.mem = np.memmap(tbl, dtype=np.uint8, mode="r",
                                     shape=(self.rows_total, self.row_bytes))
            elif me_owner:
                # owner 但表已就绪 (热启动): 直接 attach, 不重填。
                self.mem = np.memmap(tbl, dtype=np.uint8, mode="r",
                                     shape=(self.rows_total, self.row_bytes))
                logger.info(
                    "PLE mmap(mem): 本进程为 owner, 复用已有共享表 %s (%.1f GiB)",
                    tbl,
                    self.rows_total * self.row_bytes / 2**30,
                )
            else:
                # waiter: 等 ready (若热启动表已在, 立刻通过; 否则等 owner 填)。
                deadline = time.monotonic() + 1800  # 最多等 30 分钟
                while not os.path.exists(ready):
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            f"PLE mmap(mem): 等共享表 {tbl} ready 超时 "
                            "(owner 进程可能未运行或已卡住)"
                        )
                    time.sleep(2.0)
                self.mem = np.memmap(tbl, dtype=np.uint8, mode="r",
                                     shape=(self.rows_total, self.row_bytes))
                logger.info(
                    "PLE mmap(mem): 本进程为 waiter, attach 共享表 %s (%.1f GiB)",
                    tbl,
                    self.rows_total * self.row_bytes / 2**30,
                )
            # 已整表进 RAM (共享), 不再需要磁盘 memmap 视图。
            self.mm = [None] * (max_idx + 1)

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """ids: int64 [N] 全局行号 -> uint8 [N, row_bytes] (新数组)。"""
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        # mem 模式: 整表一个连续数组, 去重后单次花式索引 (内存随机访问, 无 I/O)。
        if self.mem is not None:
            uniq, inverse = np.unique(ids, return_inverse=True)
            if uniq[0] < 0 or uniq[-1] >= self.rows_total:
                raise IndexError(
                    f"PLE row id out of range: [{uniq[0]}, {uniq[-1]}] "
                    f"for {self.rows_total} rows"
                )
            return self.mem[uniq][inverse]
        # 去重 + 排序: 重复 ngram 常见, 且排序行在片内局部性更好。
        uniq, inverse = np.unique(ids, return_inverse=True)
        if uniq[0] < 0 or uniq[-1] >= self.shard_size * len(self.mm):
            raise IndexError(
                f"PLE row id out of range: [{uniq[0]}, {uniq[-1]}] "
                f"for {self.rows_total} rows"
            )
        shard = uniq // self.shard_size
        local = uniq - shard * self.shard_size
        out = np.empty((uniq.size, self.row_bytes), dtype=np.uint8)

        bounds = np.flatnonzero(np.diff(shard)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [uniq.size]))
        tasks: list[tuple[int, int, int]] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            si = int(shard[s])
            for c in range(s, e, self.chunk):
                tasks.append((si, c, min(c + self.chunk, e)))

        def run(task: tuple[int, int, int]) -> None:
            si, a, b = task
            mm = self.mm[si]
            if mm is None:
                raise IndexError(f"PLE shard {si} missing")
            # 对 memmap 做花式索引: page fault 触发 I/O; NumPy 拷贝时放 GIL, 跨线程 overlap。
            out[a:b] = mm[local[a:b]]

        if len(tasks) == 1:
            run(tasks[0])
        else:
            for _ in self.pool.map(run, tasks):
                pass
        return out[inverse]

    def prewarm(self) -> None:
        """把每片流读一遍, 让 page cache 能装多少装多少。"""
        block = 64 << 20
        for path, mm in zip(self.paths, self.mm):
            if path is None or mm is None:
                continue
            start = mm.offset
            end = start + mm.shape[0] * mm.shape[1]
            with open(path, "rb", buffering=0) as f:
                pos = start
                while pos < end:
                    n = f.readinto(bytearray(min(block, end - pos)))  # noqa: F841
                    if not n:
                        break
                    pos += n


# --------------------------------------------------------------------------- #
# 顶替 PLEVocabParallelEmbedding 的占位 (不分配大张量)
# --------------------------------------------------------------------------- #
class _MmapNgramEmbedding(nn.Module):
    """Duck-type 上游 PLE 代码会读到的 ``VocabParallelEmbedding`` 字段。

    故意不设 ``weight``: 我们的 patched ``load_weights`` 跳过全部分片, 上游
    ``_dequantize_embeddings`` 对非 FP8 也不碰 ``weight_scale`` (走 None 分支)。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.org_vocab_size = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.table: MmapPleTable | None = None
        self._zeros_dtype = torch.bfloat16

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        table = self.table
        if table is None:
            # 权重从未加载 (如 --load-format dummy): 用零占位保管道通。
            return torch.zeros(
                (*ids.shape, self.embedding_dim),
                dtype=self._zeros_dtype,
                device=ids.device,
            )
        ids_np = ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)
        rows = table.gather(ids_np)  # uint8 [N, row_bytes], 新且可写
        out = torch.from_numpy(rows).view(table.torch_dtype)
        out = out.to(ids.device, non_blocking=True)
        return out.reshape(*ids.shape, self.embedding_dim)


# --------------------------------------------------------------------------- #
# 分片定位
# --------------------------------------------------------------------------- #
def _find_shards(
    model_path: str, layer_idx: int
) -> tuple[dict[int, tuple[str, int, int]], str | None, tuple | None]:
    """定位 ``layers.<idx>.ple.ple_embedding.ngram_embedding.shard_N.weight``。

    返回 (shards, dtype_str, scale_entry)。scale_entry 为
    (path, abs_offset, nbytes, dtype_str) 或 None。
    """
    shard_re = re.compile(
        rf"layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
    )
    scale_re = re.compile(
        rf"layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\.weight_scale$"
    )
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted(
            {
                os.path.join(model_path, fn)
                for name, fn in weight_map.items()
                if shard_re.search(name) or scale_re.search(name)
            }
        )
    else:
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    shards: dict[int, tuple[str, int, int]] = {}
    dtype_str: str | None = None
    scale_entry: tuple | None = None
    cols = 0
    for path in files:
        header, data_start = parse_safetensors_header(path)
        for name, meta in header.items():
            m = shard_re.search(name)
            if m:
                start, end = meta["data_offsets"]
                rows, cols = meta["shape"]
                if dtype_str is None:
                    dtype_str = meta["dtype"]
                elif meta["dtype"] != dtype_str:
                    raise ValueError("PLE shards have mixed dtypes")
                if dtype_str not in _ITEMSIZE:
                    raise ValueError(f"PLE shard {name}: unsupported dtype {dtype_str}")
                if end - start != rows * cols * _ITEMSIZE[dtype_str]:
                    raise ValueError(f"PLE shard {name}: size/shape mismatch")
                shards[int(m.group(1))] = (path, data_start + start, rows)
            elif scale_re.search(name):
                start, end = meta["data_offsets"]
                scale_entry = (path, data_start + start, end - start, meta["dtype"])
    return shards, dtype_str, (cols, scale_entry)


def _read_scale(entry: tuple) -> torch.Tensor:
    path, offset, nbytes, dtype_str = entry
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(nbytes)
    if dtype_str == "F32":
        return torch.tensor(struct.unpack("<f", raw[:4])[0], dtype=torch.float32)
    if dtype_str == "BF16":
        u16 = struct.unpack("<H", raw[:2])[0]
        return torch.tensor(u16 << 16, dtype=torch.int32).view(torch.float32).squeeze()
    if dtype_str == "F16":
        return torch.frombuffer(bytearray(raw[:2]), dtype=torch.float16).clone().squeeze()
    raise ValueError(f"unsupported weight_scale dtype {dtype_str}")


def _model_path() -> str | None:
    try:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config().model_config.model
    except Exception as exc:  # pragma: no cover - 防御
        logger.warning("PLE mmap: 读不到 model path: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# custom op: 把 compute_ngram_ids + mmap gather + H2D 包成图外 op
# --------------------------------------------------------------------------- #
def _qwen4_exp_ple_mmap_lookup(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    from vllm.forward_context import get_forward_context

    layer = get_forward_context().no_compile_layers[layer_name]
    ngram_emb = layer.ple_embedding  # Qwen4ExpNGramEmbedding (已 patch)
    ngram_ids = ngram_emb.compute_ngram_ids(
        input_ids, query_start_loc, ngram_context
    )
    gathered = ngram_emb.ngram_embedding(ngram_ids)  # [num_tokens, heads, head_dim]
    flat = gathered.reshape(ngram_ids.shape[0], -1)  # [num_tokens, embedding_dim]
    output[: flat.shape[0]].copy_(flat.to(output.dtype))


def _qwen4_exp_ple_mmap_lookup_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


def _register_op() -> None:
    if hasattr(torch.ops.vllm, _OP_NAME):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=_OP_NAME,
        op_func=_qwen4_exp_ple_mmap_lookup,
        mutates_args=["output"],
        fake_impl=_qwen4_exp_ple_mmap_lookup_fake,
    )


def _ensure_splitting_op() -> None:
    """把本模块的 PLE lookup custom op 加进当前 VllmConfig 的 splitting_ops (幂等)。

    背景: 本 op 内做 CPU gather + pageable H2D (``ids.to("cpu")`` / rows H2D),
    这些 D2H/H2D 拷贝在 CUDA graph capture 期间非法 (除非 pinned)。PIECEWISE
    cudagraph 会把图里每个非 split 点算子 capture 进 graph, 故必须把本 op 变成
    Dynamo FX 的 split 点, 让它在 capture 之外 eager 执行。对齐 1Cat: 其 PLE op
    列在 ``CompilationConfig._attention_ops`` (PIECEWISE split 点列表)。

    timing: ``set_splitting_ops_for_v1`` 在 VllmConfig 初始化 (model 加载前) 就已把
    ``_attention_ops`` 拷进**实例** ``splitting_ops``; 而 ``backends.py`` 在 compile
    (cudagraph capture 前, model 加载后) 才读实例 ``splitting_ops`` 做 FX split。
    故在 model 加载期 (``load_weights``, 早于 capture) 往实例追加本 op 即生效,
    且无需改底座 ``compilation.py``。多个 PLE 层重复调用由 ``in`` 判断去重。
    """
    try:
        from vllm.config import get_current_vllm_config

        cc = get_current_vllm_config().compilation_config
    except Exception as exc:  # pragma: no cover - 防御
        logger.warning(
            "PLE mmap: 读不到 compilation_config, 跳过 splitting op 注入: %s", exc
        )
        return
    qualified = f"vllm::{_OP_NAME}"
    if cc.splitting_ops is None:
        cc.splitting_ops = []
    if qualified not in cc.splitting_ops:
        cc.splitting_ops.append(qualified)
        logger.info(
            "PLE mmap: 注入 splitting op %s (把 D2H gather 移出 cudagraph capture)",
            qualified,
        )


# --------------------------------------------------------------------------- #
# 补丁
# --------------------------------------------------------------------------- #
def _setup_table(self) -> None:
    """开 mmap 表 (幂等)。"""
    if self.ngram_embedding.table is not None:
        return
    model_path = getattr(self, "_ple_mmap_model_path", None) or _model_path()
    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError(
            f"PLE mmap: model path {model_path!r} 不是本地目录; "
            "把 --model 指向下载好的快照"
        )
    m = re.search(r"layers\.(\d+)\.", self._ple_mmap_prefix)
    if not m:
        raise RuntimeError(
            f"PLE mmap: 无法从 {self._ple_mmap_prefix!r} 解析层号"
        )
    layer_idx = int(m.group(1))
    shards, dtype_str, (cols, scale_entry) = _find_shards(model_path, layer_idx)
    if not shards:
        raise RuntimeError(
            f"PLE mmap: {model_path} 下找不到 layer {layer_idx} 的分片张量"
        )
    if dtype_str not in _FP8_DTYPES:
        raise RuntimeError(f"PLE mmap: 不支持的分片 dtype {dtype_str}")
    if cols != self.head_dim:
        raise RuntimeError(f"PLE mmap: 分片列宽 {cols} != head_dim {self.head_dim}")

    # FP8 表必须有全局 scale (上游 _dequantize_embeddings 读 ngram_embedding.weight_scale)。
    # BF16/F16 无 scale, 原样透传; 仅 FP8 缺 scale 才报错。
    if dtype_str not in _NON_FP8_DTYPES:
        if scale_entry is None:
            raise RuntimeError("PLE mmap: FP8 分片缺少 ngram_embedding.weight_scale")
        self.ngram_embedding.weight_scale = _read_scale(scale_entry).to(
            "cuda", non_blocking=False
        )

    parts = int(self.split_ngram_parts)
    vocab = int(self.ngram_embedding.org_vocab_size)
    shard_size = math.ceil(vocab / parts)
    for idx, (_p, _o, rows) in shards.items():
        expected = max(0, min(shard_size, vocab - idx * shard_size))
        if rows != expected:
            raise RuntimeError(
                f"PLE mmap: shard {idx} 有 {rows} 行, 期望 {expected} "
                f"(org_vocab_size={vocab}, split_ngram_parts={parts}); "
                "config 与 checkpoint 分片切分不一致"
            )
    row_bytes = cols * _ITEMSIZE[dtype_str]
    ple_mode = mode()
    use_mem = ple_mode == "mem"
    total_rows = sum(rows for (_p, _o, rows) in shards.values())
    total_gib = total_rows * row_bytes / 2**30
    if use_mem:
        logger.info(
            "PLE mmap(mem): 整表读进 RAM (%.1f GiB)...", total_gib
        )
    # key: 模型路径+层号, 保证换模型/换表自动换共享表文件 (不串表)。
    shm_key = f"{model_path}:layer{layer_idx}"
    table = MmapPleTable(
        shards,
        shard_size,
        row_bytes,
        _FP8_DTYPES[dtype_str],
        workers=_env_int("VLLM_PLE_MMAP_WORKERS", 32),
        chunk=_env_int("VLLM_PLE_MMAP_CHUNK", 2048),
        mem=use_mem,
        key=shm_key,
    )
    # mem 模式整表已在 RAM, 无需再预热线 page cache; disk 模式才走 page cache。
    if not use_mem and _env_flag("VLLM_PLE_MMAP_PREWARM", False):
        logger.info(
            "PLE mmap(disk): 预热线 page cache (%.1f GiB)...",
            table.rows_total * row_bytes / 2**30,
        )
        table.prewarm()
    self.ngram_embedding.table = table
    if use_mem:
        logger.info(
            "PLE mmap(mem): layer %d, %d 片, %d 行 x %d B (RAM %.1f GiB), dtype %s",
            layer_idx,
            len(shards),
            table.rows_total,
            row_bytes,
            table.rows_total * row_bytes / 2**30,
            dtype_str,
        )
    else:
        logger.info(
            "PLE mmap(disk): layer %d, %d 片, %d 行 x %d B (磁盘 %.1f GiB), dtype %s, %d 线程",
            layer_idx,
            len(shards),
            table.rows_total,
            row_bytes,
            table.rows_total * row_bytes / 2**30,
            dtype_str,
            table.pool._max_workers,
        )


def maybe_apply(cls: type) -> None:
    """``VLLM_PLE_MMAP`` 开时给 ``Qwen4ExpNGramEmbedding`` 打 patch (幂等)。

    关 (未设 / 0) 直接 return, 对现有行为零影响。在类定义之后、实例化之前调用
    (由 install_sm75_overlay.py append 到上游 ple_layer.py 末尾触发)。
    """
    if not enabled():
        return
    if getattr(cls, "_ple_mmap_patched", False):
        return
    mod = sys.modules[cls.__module__]
    orig_init = cls.__init__
    orig_load_weights = cls.load_weights

    def __init__(
        self,
        config,
        embedding_dim,
        ple_dense_layer_id,
        max_total_tokens,
        max_num_reqs,
        prefix,
        layer_name,
        quant_config=None,
        params_dtype=None,
    ) -> None:
        # 用占位顶替 PLEVocabParallelEmbedding, 跑原构造 (hash buffer/workspace
        # 照常建), 但不分配那张大 GPU 张量。quant_config=None 让上游不走 FP8
        # 量化方法 (反正占位不建 FP8 权重参数)。
        real_cls = mod.PLEVocabParallelEmbedding
        mod.PLEVocabParallelEmbedding = (
            lambda n, d, **_kw: _MmapNgramEmbedding(n, d)
        )
        try:
            orig_init(
                self,
                config,
                embedding_dim,
                ple_dense_layer_id,
                max_total_tokens,
                max_num_reqs,
                prefix,
                layer_name,
                quant_config=None,
                params_dtype=params_dtype,
            )
        finally:
            mod.PLEVocabParallelEmbedding = real_cls
        self._ple_mmap_prefix = prefix
        self._ple_mmap_model_path = _model_path()
        if params_dtype is not None:
            self.ngram_embedding._zeros_dtype = params_dtype
            self._ple_mmap_out_dtype = params_dtype
        else:
            self._ple_mmap_out_dtype = torch.bfloat16
        logger.info(
            "PLE mmap: %s -> 占位 embedding (%d 行 x %d), 表将走 mmap",
            prefix,
            self.ngram_embedding.org_vocab_size,
            self.head_dim,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        rest: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            if name.startswith("ngram_embedding.shard_") and name.endswith(".weight"):
                loaded.add(name)  # 磁盘服务, 不落显存
                continue
            if name == "ngram_embedding.weight_scale":
                # 仅 FP8 有; 存到占位上, 上游 _dequantize_embeddings 经
                # ngram_embedding.weight_scale 读到。
                self.ngram_embedding.weight_scale = w.detach().to(
                    device=torch.device("cuda"), non_blocking=True
                )
                loaded.add(name)
                continue
            rest.append((name, w))
        loaded.update(orig_load_weights(self, rest))
        # 把本 op 注入当前 config 的 splitting_ops, 让 Dynamo 在本 op 处 split,
        # 使 CPU gather + H2D 在 cudagraph capture 之外 eager 执行 (避免 capture
        # 期间非法 D2H 拷贝)。幂等, 早于 profile_run/capture 的 compile。
        _ensure_splitting_op()
        _setup_table(self)
        return loaded

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        num_tokens = input_ids.shape[0]
        # 输出 dtype: FP8 表保留 fp8 (留给上游 _dequantize_embeddings 乘 scale);
        # bf16/f16 表改为 model dtype —— sm75 无 bf16 计算, 模型回退 fp16, 若输出
        # 保留表原生 bf16 会污染后续 fp16 激活 (model.py 里 hidden + ple -> 类型提升
        # 成 bf16 -> GDN in_proj 的 marlin 收 bf16 激活, Turing 拒: "only support FP16
        # or INT8 activation")。sm80+ model=bf16 且表=bf16, _ple_mmap_out_dtype=bf16,
        # 行为不变。
        table = self.ngram_embedding.table
        if table is not None and table.torch_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            out_dtype = table.torch_dtype
        else:
            out_dtype = self._ple_mmap_out_dtype
        # gather 是 CPU 活 + pageable H2D, 进不了 CUDA graph; 用图外 custom op
        # 拿 layer 执行 (对齐上游 qwen4_exp_compute_ple_ngram_ids 写法)。
        output = torch.empty(
            (num_tokens, self.embedding_dim),
            dtype=out_dtype,
            device=input_ids.device,
        )
        getattr(torch.ops.vllm, _OP_NAME)(
            input_ids,
            query_start_loc,
            ngram_context,
            output,
            self.layer_name,
        )
        return output

    _register_op()
    cls.__init__ = __init__
    cls.load_weights = load_weights
    cls.forward = forward
    cls._ple_mmap_patched = True
    logger.info("PLE mmap patch applied to %s.%s", cls.__module__, cls.__name__)


__all__ = ["maybe_apply", "enabled"]
