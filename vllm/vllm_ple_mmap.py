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
  VLLM_PLE_MMAP_MODE=mem      offload 模式 disk/mem/vram (默认 disk; mem 需 ~95G RAM)
  VLLM_PLE_MEM_LAZY=1         mem 模式走 lazy 后台填 (秒级 attach, 默认开; 0=旧阻塞)
  VLLM_PLE_MEM_FILL_WORKERS=16 lazy 后台填表并行片数 (NVMe 多队列, 默认 16)
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
import threading
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
ENV_HOST_GATHER = "VLLM_PLE_HOST_GATHER"
_OP_NAME = "qwen4_exp_ple_mmap_lookup"


def host_gather_enabled() -> bool:
    """host-gather 子模式开关 (VLLM_PLE_HOST_GATHER, 默认开, envs_sm75 同源语义)。

    开 = PLE 的 ids D2H + 表 gather 从模型 forward 里挪进
    ``Qwen4ExpModelState.prepare_inputs`` (forward 之前预填常驻 buffer), forward
    只 return buffer view → forward 内无 custom op 分裂点 → decode 可走
    FULL_DECODE_ONLY cudagraph。仅在 PLE offload (disk/mem) 生效时有意义。
    """
    return _env_flag(ENV_HOST_GATHER, True)

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

# ---- lazy-mem 共享表 header (lazy 子模式专用, 见 PLAN-ple-mmap-lazy-mem) ----
# 表文件前 64KB 当 header, 数据区从 offset _HDR 起。header 里唯一"会跨进程变"的
# 字段是 shards_done (水位)。owner 用 pwrite 写水位 (脏页落共享 page cache + fsync
# 回写 tmpfs), 各 rank 用 open()+read() 从同一 page cache 页读必见 (同机共享, 无需
# 屏障)。水位写成单调 CAS (只增), 对发布线程调度无关。注: 本机 tmpfs 上 ftruncate
# 后对已有页 pwrite, np.memmap 视图可能读到 stale 页, 故水位走文件读 (实测恒一致)。
_HDR = 64 << 10  # header 字节数 (64KB); 数据区起点偏移
_HDR_MAGIC = b"PLELZM01"  # 8 字节魔数, 用于判断表文件是否为 lazy 结构
_HDR_OFF_ROWS = 8  # int64 rows_total (小端)
_HDR_OFF_SHARDS = 16  # int64 shards_total (小端)
_HDR_OFF_DONE = 24  # int64 shards_done (小端) = 水位 (单调递增的连续前缀片数)


def _i64(v: int) -> bytes:
    """int -> 8 字节小端 (写 header 字段用)。"""
    return struct.pack("<q", int(v))


def _i64_from(raw: bytes, off: int) -> int:
    """读 8 字节小端 int64。"""
    return struct.unpack("<q", raw[off : off + 8])[0]


def _write_header(tbl: str, rows_total: int, shards_total: int, shards_done: int) -> None:
    """写 lazy-mem 表 header (magic + 三个 int64 字段)。

    用 pwrite 直写文件头 (不 mmap 额外区域), 写完 flush+fsync 让其他进程可见。
    字段固定偏移 (见 _HDR_OFF_*), 未用字节补 0 (64KB 区前 _HDR 之外无所谓)。
    """
    buf = bytearray(_HDR)
    buf[0 : len(_HDR_MAGIC)] = _HDR_MAGIC
    buf[_HDR_OFF_ROWS : _HDR_OFF_ROWS + 8] = _i64(rows_total)
    buf[_HDR_OFF_SHARDS : _HDR_OFF_SHARDS + 8] = _i64(shards_total)
    buf[_HDR_OFF_DONE : _HDR_OFF_DONE + 8] = _i64(shards_done)
    fd = os.open(tbl, os.O_RDWR)
    try:
        os.pwrite(fd, bytes(buf), 0)
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_header(tbl: str) -> dict | None:
    """读 lazy-mem 表 header。

    返回 {"rows_total","shards_total","shards_done"}; 文件不存在/太小/magic 不对
    → None (调用方据此判"半死表"重填)。读 64KB 即可 (header 全在前 32 字节内)。
    """
    try:
        with open(tbl, "rb") as f:
            raw = f.read(_HDR)
    except OSError:
        return None
    if len(raw) < 32 or raw[0 : len(_HDR_MAGIC)] != _HDR_MAGIC:
        return None
    return {
        "rows_total": _i64_from(raw, _HDR_OFF_ROWS),
        "shards_total": _i64_from(raw, _HDR_OFF_SHARDS),
        "shards_done": _i64_from(raw, _HDR_OFF_DONE),
    }


def _publish_watermark(tbl: str, k: int) -> None:
    """把水位 shards_done 提升为 k (单调, 只增不回退, 跨进程可见)。

    调用方不止一个线程 (owner 的 fill 发布线程 + fill 收尾), 故做成**单调 CAS**:
    先 pread 当前水位 cur, 仅当 k > cur 才 pwrite —— 保证水位永不回退, 与发布
    线程调度无关 (例: fill 很快完成时, 发布线程的收尾 publish 可能带旧值, 不会
    把已发布的更高水位改回去)。pwrite 写 header 页 → 脏页落共享 page cache; 读方
    (各 rank / gather) open()+read() 从同一页读必见 (同机共享页缓存, 无需屏障)。
    """
    fd = os.open(tbl, os.O_RDWR)
    try:
        try:
            raw = os.pread(fd, 8, _HDR_OFF_DONE)
            cur = _i64_from(raw, 0) if len(raw) == 8 else -1
        except OSError:
            cur = -1
        if k > cur:
            os.pwrite(fd, _i64(k), _HDR_OFF_DONE)
            os.fsync(fd)
    finally:
        os.close(fd)


def _read_watermark_file(tbl: str) -> int:
    """文件直读水位 shards_done (open()+read 64KB, tmpfs 页缓存 ~µs 级)。

    这是水位热路径的可靠通路 (见 MmapPleTable._read_watermark): owner pwrite 把
    水位写进共享 page cache 页, 各 rank 文件读必见。文件不存在/太小/magic 不对 → 0。
    """
    hdr = _read_header(tbl)
    if hdr is None:
        return 0
    return max(0, hdr["shards_done"])


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
      内存 (二号机 238G 空闲内存装得下), 比 disk 稳 (无换页抖动)。默认再走 lazy
      子模式 (``VLLM_PLE_MEM_LAZY=1``): 启动只建 header+空数据区秒级 attach,
      owner 后台逐片填, gather 按片水位路由、未填片回退磁盘 (砍掉 ~11 分钟阻塞)。
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

    lazy-mem 子模式 (mem 且 lazy=True): 启动时只建 header + 空数据区即 attach
    (秒级), 整表由 owner 的后台线程逐片填; gather 按片水位路由, 未填的片回退
    磁盘读 (miss 路径数据永远正确)。水位 = header 里跨进程共享的单调前缀片数。
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
        lazy: bool = True,
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
        # lazy-mem 状态: self.lazy=True 时, self._done_flags(每片是否已填) 与
        # self._fill_stop(fill 完成信号) 供 owner 后台填线程 + gather 水位路由使用;
        # self._tbl 是 /dev/shm 表文件路径, gather 据此文件读水位 (owner pwrite
        # 写进共享页缓存, 各 rank 文件读必见, 仅 lazy 用)。
        self.lazy: bool = False
        self._tbl: str | None = None
        self._done_flags: list[bool] | None = None
        self._fill_stop: threading.Event | None = None
        if mem:
            import hashlib

            # key 含模型路径+形状: 换模型/换表自动换表文件, 旧表成孤儿可被清。
            h = hashlib.sha1(
                f"{key}:{self.rows_total}:{self.row_bytes}".encode()
            ).hexdigest()[:16]
            tbl = f"/dev/shm/ple_mmap_{h}"
            ready = f"{tbl}.ready"
            ready_full = f"{tbl}.ready.full"
            self._tbl = tbl
            # owner 选举: 对表文件 flock 非阻塞排他。拿到 = owner, 且必须终身
            # 持锁 (存到 self, 进程活多久锁多久; 内核在进程死亡时自动释放) ——
            # waiter 用同法试锁, 锁被占即证明 owner 活着。owner 崩 → 锁自动放 →
            # 下次启动无 ready → 新 owner 重填 (自愈)。
            lock_fd = os.open(tbl, os.O_RDWR | os.O_CREAT, 0o666)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._shm_lock_fd = lock_fd  # owner: 终身持锁 (勿 close)
            except OSError:
                os.close(lock_fd)
                self._shm_lock_fd = None
            me_owner = self._shm_lock_fd is not None

            if not lazy:
                # ---- 旧 mem 路径 (VLLM_PLE_MEM_LAZY=0 保险丝): 阻塞填完再 attach。
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
            else:
                # ---- lazy-mem 路径: 秒级 attach + 后台按片填, gather 按水位路由。
                total_size = _HDR + self.rows_total * self.row_bytes
                n_shards = max_idx + 1
                self.lazy = True
                self._fill_stop = threading.Event()
                self._shards_total = n_shards
                # 水位读取走 _read_watermark() 的文件读通路 (open+read header 页),
                # 不在此建任何视图 —— 文件读跨进程恒一致 (owner pwrite → 共享页缓存)。
                # 热启动复用: 表文件在 + 大小匹配 + header magic 对 + 水位全满 →
                # 直接 attach, 不重填、不起 fill 线程 (重启秒级)。
                hdr = _read_header(tbl)
                reuse = (
                    hdr is not None
                    and hdr["rows_total"] == self.rows_total
                    and hdr["shards_total"] == n_shards
                    and hdr["shards_done"] == n_shards
                    and os.path.getsize(tbl) == total_size
                )
                if me_owner and not reuse:
                    # owner-lazy: 建结构 (ftruncate + header + 空数据区), 立即写
                    # .ready (语义改: "结构就绪可 attach", 不等填表), 起后台填线程,
                    # __init__ 秒级返回不阻塞。
                    logger.info(
                        "PLE mmap(lazy-mem): 本进程为 owner, 建共享表 %s (%.1f GiB), "
                        "后台填表 (启动不阻塞)...",
                        tbl,
                        total_size / 2**30,
                    )
                    try:
                        if os.path.exists(ready):
                            os.unlink(ready)
                        if os.path.exists(ready_full):
                            os.unlink(ready_full)
                    except OSError:
                        pass
                    fd = os.open(tbl, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o666)
                    os.ftruncate(fd, total_size)
                    # header: magic/rows_total/shards_total/shards_done=0 (数据区从
                    # offset _HDR 起, 此处数据区先留空, 由 fill 线程逐片填)。
                    _write_header(tbl, self.rows_total, n_shards, 0)
                    os.close(fd)
                    # 结构就绪 → 写 .ready, waiter 据此 attach (不必等填表)。水位
                    # 读取由 _read_watermark() 走文件读, 无需在此建任何视图。
                    with open(ready, "w") as f:
                        f.write(str(self.rows_total))
                        f.flush()
                        os.fsync(f.fileno())
                    # 数据区视图 (offset=_HDR 起, 行布局与全局行号一致, 可写)。
                    self.mem = np.memmap(
                        tbl, dtype=np.uint8, mode="r+",
                        offset=_HDR, shape=(self.rows_total, self.row_bytes),
                    )
                    # 每片完成标志 (线程间共享, bool 赋值原子); 启动后台填表线程。
                    self._done_flags = [False] * n_shards
                    self._start_fill(tbl)
                elif me_owner:
                    # owner 但表已全满 (热启动): 直接 attach, 不重填、不起填线程。
                    self.mem = np.memmap(
                        tbl, dtype=np.uint8, mode="r",
                        offset=_HDR, shape=(self.rows_total, self.row_bytes),
                    )
                    self._done_flags = [True] * n_shards
                    logger.info(
                        "PLE mmap(lazy-mem): 本进程为 owner, 复用已有全满共享表 %s "
                        "(%.1f GiB)",
                        tbl,
                        total_size / 2**30,
                    )
                else:
                    # waiter-lazy: 等 .ready (owner 秒写, 故秒过), attach 数据区。
                    # 不读水位、不起 fill 线程; 保留 self.mm 磁盘视图供 gather
                    # miss 回退 (水位未到的片走磁盘读, 数据永远正确)。
                    deadline = time.monotonic() + 1800  # 最多等 30 分钟
                    while not os.path.exists(ready):
                        if time.monotonic() > deadline:
                            raise RuntimeError(
                                f"PLE mmap(lazy-mem): 等共享表 {tbl} ready 超时 "
                                "(owner 进程可能未运行或已卡住)"
                            )
                        time.sleep(2.0)
                    self.mem = np.memmap(
                        tbl, dtype=np.uint8, mode="r",
                        offset=_HDR, shape=(self.rows_total, self.row_bytes),
                    )
                    logger.info(
                        "PLE mmap(lazy-mem): 本进程为 waiter, attach 共享表 %s "
                        "(%.1f GiB, 未填片走磁盘回退)",
                        tbl,
                        total_size / 2**30,
                    )
                # lazy 下全程保留磁盘 memmap 视图: fill 线程读源 + gather miss 回退。
                # (128 个 np.memmap 对象只是元数据, 无 RAM 成本, page cache 归内核管。)

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """ids: int64 [N] 全局行号 -> uint8 [N, row_bytes] (新数组)。

        三种路由:
          - lazy-mem (self.lazy): 按片水位路由 —— 已填片 (si<k) 走内存 self.mem,
            未填片 (si>=k) 走磁盘 self.mm miss 回退。两来源写同一 out 的不同区段,
            数据永远正确 (磁盘分片只读不可变, 水位只影响"从哪读"不影响内容)。
          - 非 lazy mem (self.mem 非 None 且非 lazy): 整表连续数组单次花式索引。
          - disk: 按片分组对磁盘 memmap 花式索引 (page cache 换页)。
        """
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        # 去重 + 排序: 重复 ngram 常见, 且排序行在片内局部性更好。
        uniq, inverse = np.unique(ids, return_inverse=True)
        if uniq[0] < 0 or uniq[-1] >= self.rows_total:
            raise IndexError(
                f"PLE row id out of range: [{uniq[0]}, {uniq[-1]}] "
                f"for {self.rows_total} rows"
            )
        # 非 lazy mem: 整表一个连续数组, 单次花式索引 (内存随机访问, 无 I/O)。
        if self.mem is not None and not self.lazy:
            return self.mem[uniq][inverse]

        shard = uniq // self.shard_size
        local = uniq - shard * self.shard_size
        out = np.empty((uniq.size, self.row_bytes), dtype=np.uint8)

        # lazy-mem: 读当前水位 k (一次即可; 调用中途水位上涨只影响性能不影响
        # 正确性 —— 已填片的数据在发布水位前早已落盘可见)。k==0 时全走磁盘
        # (等价 disk 模式), 刚启动一片没填也正确。
        k = self._read_watermark() if self.lazy else -1

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
            if self.lazy and si < k:
                # 已填片: 内存读。self.mem 数据区从全局行 0 连续排 (行布局不变,
                # 仅文件内字节偏移 +_HDR, 而 self.mem 已是 offset=_HDR 起的视图),
                # 故 self.mem[全局行号] 直接对应。
                out[a:b] = self.mem[uniq[a:b]]
            else:
                mm = self.mm[si]
                if mm is None:
                    raise IndexError(f"PLE shard {si} missing")
                # 未填片/disk: 对磁盘 memmap 花式索引; page fault 触发 I/O,
                # NumPy 拷贝时放 GIL, 跨线程 overlap。
                out[a:b] = mm[local[a:b]]

        if len(tasks) == 1:
            run(tasks[0])
        else:
            for _ in self.pool.map(run, tasks):
                pass
        return out[inverse]

    def _read_watermark(self) -> int:
        """读当前水位 shards_done (gather 每次调用读一次)。

        从表文件 open()+read() header 页 (64KB, tmpfs 页缓存, ~µs 级, 相对 ms 级
        gather 可忽略) —— 这是跨进程可见的可靠通路: owner pwrite 把水位写进共享
        page cache 页, 各 rank (含本进程) 都从同一页读必见。
        注: 不用 np.memmap 视图直读 —— 本机 tmpfs 上 ftruncate 后对已有页做
        pwrite, mmap 视图可能读到 stale 旧页 (实测 _hdr_view 与文件读不一致),
        而 open()+read() 恒一致; 故水位热路径走文件读, 保证所有 rank 看到同一值。
        读到 magic 不对 / 文件未就绪 / 负值 → 0 (保守: 全走磁盘回退, 数据仍正确)。
        """
        if self._tbl is None:
            return 0
        return _read_watermark_file(self._tbl)

    def _start_fill(self, tbl: str) -> None:
        """起后台填表 (owner-lazy 专用): fill 工作线程池 + 水位发布线程 (均 daemon)。

        主线程 (本 __init__) 立即返回不阻塞。
          - fill 工作线程各领不同片: 读磁盘 ``self.mm[idx]`` -> 写
            ``self.mem[start:start+cnt]``, 每片写完置 ``done[idx]=True``
            (bool 赋值原子, 线程安全)。用单独 fill 池 (非 self.pool), 不与在线
            gather 抢 worker。
          - 水位发布线程每 100ms 扫 done 找最大连续前缀 k, ``_publish_watermark``
            单调 CAS 写入 ``header.shards_done`` (pwrite+fsync, 跨进程经共享页缓存
            可见; 只增不回退, 与线程调度无关)。
          - 全填完 -> 再发布最终水位 n (权威值, 热启动复用据此判全满), 写
            ``.ready.full`` 标志, 线程退出, 打一条日志。
        fill 异常不崩 worker 进程: 打日志 + 水位停住, 未填片后续走磁盘回退 (正确)。
        """
        n = self._shards_total
        fill_workers = max(1, _env_int("VLLM_PLE_MEM_FILL_WORKERS", 16))
        indices = [i for i in range(n) if self.mm[i] is not None]
        full_ready = f"{tbl}.ready.full"
        done = self._done_flags  # [False]*n, fill/发布线程共享

        def _fill_shard(idx: int) -> None:
            mm = self.mm[idx]
            if mm is None:
                done[idx] = True  # 无源片视为完成
                return
            cnt = mm.shape[0]
            start = idx * self.shard_size
            # 读磁盘片 -> 写共享表对应区段; NumPy 拷贝放 GIL, 多线程 overlap。
            self.mem[start : start + cnt] = np.asarray(mm)
            done[idx] = True

        def _publisher(last_pub: list[int]) -> None:
            # 保守水位: 只发布"0..k-1 全 done"的最大连续前缀 k, 单调递增。
            while not self._fill_stop.is_set():
                k = 0
                while k < n and done[k]:
                    k += 1
                if k > last_pub[0]:
                    last_pub[0] = k
                    try:
                        _publish_watermark(tbl, k)
                    except OSError:
                        logger.warning("PLE mmap(lazy-mem): 发布水位失败", exc_info=True)
                time.sleep(0.1)
            # 收尾: 确保最终水位落盘 (正常应 == n)。
            try:
                _publish_watermark(tbl, last_pub[0])
            except OSError:
                pass

        def _fill_main() -> None:
            last_pub = [0]
            threading.Thread(
                target=_publisher, args=(last_pub,), daemon=True,
                name="ple-fill-publisher",
            ).start()
            try:
                with ThreadPoolExecutor(max_workers=fill_workers) as fill_pool:
                    futs = [fill_pool.submit(_fill_shard, idx) for idx in indices]
                    for fu in futs:
                        fu.result()
                # 全部片已落盘: 发布最终水位 n (权威值, 热启动复用据此判全满)。
                # 此前发布线程只发 <=n 的保守前缀, 此处 n 单调收尾。
                try:
                    _publish_watermark(tbl, n)
                except OSError:
                    pass
                self._fill_stop.set()
                try:
                    with open(full_ready, "w") as f:
                        f.write(str(n))
                        f.flush()
                        os.fsync(f.fileno())
                except OSError:
                    pass
                logger.info("PLE mmap(lazy-mem): 填表完成, 水位 %d/%d", n, n)
            except Exception:
                logger.warning(
                    "PLE mmap(lazy-mem): 后台填表异常, 未填片将走磁盘回退",
                    exc_info=True,
                )
                self._fill_stop.set()

        threading.Thread(target=_fill_main, daemon=True, name="ple-fill-main").start()

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
def _ple_out_dtype(self) -> torch.dtype:
    """PLE 输出 dtype。

    FP8 表保留原生 fp8 (留给上游 ``_dequantize_embeddings`` 乘 weight_scale);
    bf16/f16 表改为 model dtype —— sm75 无 bf16 计算, 模型回退 fp16, 若输出保留表
    原生 bf16 会污染后续 fp16 激活 (model.py 里 hidden + ple -> 类型提升成 bf16 ->
    GDN in_proj 的 marlin 收 bf16 激活, Turing 拒: "only support FP16 or INT8
    activation")。sm80+ model=bf16 且表=bf16 → 返回 bf16, 行为不变。
    """
    table = self.ngram_embedding.table
    if table is not None and table.torch_dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return table.torch_dtype
    return self._ple_mmap_out_dtype


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
    # lazy-mem: mem 模式默认走 lazy 后台填 (VLLM_PLE_MEM_LAZY=0 回退旧阻塞填)。
    use_lazy = use_mem and _env_flag("VLLM_PLE_MEM_LAZY", True)
    total_rows = sum(rows for (_p, _o, rows) in shards.values())
    total_gib = total_rows * row_bytes / 2**30
    if use_mem:
        if use_lazy:
            logger.info(
                "PLE mmap(lazy-mem): 建共享表 + 后台填 RAM (%.1f GiB, 启动不阻塞)...",
                total_gib,
            )
        else:
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
        lazy=use_lazy,
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
        if use_lazy:
            logger.info(
                "PLE mmap(lazy-mem): layer %d, %d 片, %d 行 x %d B (RAM %.1f GiB), "
                "dtype %s, 后台填表中, 启动不阻塞 (未填片 gather 走磁盘回退)",
                layer_idx,
                len(shards),
                table.rows_total,
                row_bytes,
                table.rows_total * row_bytes / 2**30,
                dtype_str,
            )
        else:
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
        # host-gather 子模式 (VLLM_PLE_HOST_GATHER, 默认开): gather 挪进
        # model_state.prepare_inputs (capture 外), forward 只 return 预填 buffer
        # 的 view。buffer 常驻 (capture 前分配, 地址固定), 尺寸 max_total_tokens。
        # 仅 PLE offload (disk/mem) + 本层真在本 worker 实例化时有意义。
        self.prefetch_from_model_state = host_gather_enabled()
        self._ple_hg_max_tokens = int(max_total_tokens)
        self._ple_hg_buffer: torch.Tensor | None = None  # load_weights 后分配
        logger.info(
            "PLE mmap: %s -> 占位 embedding (%d 行 x %d), 表将走 mmap (host_gather=%s)",
            prefix,
            self.ngram_embedding.org_vocab_size,
            self.head_dim,
            self.prefetch_from_model_state,
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
        _setup_table(self)
        if self.prefetch_from_model_state:
            # host-gather: 分配常驻预填 buffer (max_total_tokens 行), capture 前
            # 就存在 → 地址固定, cudagraph replay 读同一地址。forward 直接 return
            # 它的 view, 不再调 custom op → 无需 splitting op (forward 是纯 GPU
            # copy, 可整段进 FULL cudagraph)。
            self._ple_hg_dtype = _ple_out_dtype(self)
            self._ple_hg_buffer = torch.zeros(
                (self._ple_hg_max_tokens, self.embedding_dim),
                dtype=self._ple_hg_dtype,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
            logger.info(
                "PLE mmap(host-gather): 预填 buffer 常驻 (%d x %d, %s), "
                "forward 不调 custom op → 无 splitting op, decode 可上 FULL graph",
                self._ple_hg_max_tokens,
                self.embedding_dim,
                self._ple_hg_dtype,
            )
        else:
            # 保险丝路径 (VLLM_PLE_HOST_GATHER=0): gather 仍在 forward 内 custom
            # op, 注入 splitting op 让 Dynamo 在此 split (CPU gather + H2D 移出
            # cudagraph capture), 行为等同改造前。幂等, 早于 compile/capture。
            _ensure_splitting_op()
        return loaded

    def host_gather(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        """在 forward 之前填预填 buffer (由 model_state.prepare_inputs 调用)。

        对齐 Minachist host_gather: 算 ngram_ids → mmap gather → 写常驻 buffer
        前 n 行。在 forward 之外 (eager, capture 之外) 执行 → 其中的 ids D2H +
        host np.take + H2D 都不进 cudagraph。H2D 走当前 stream (forward/replay
        紧接着用同一 stream, 无需 event)。
        """
        buf = self._ple_hg_buffer
        if buf is None:
            return
        input_ids = input_ids.reshape(-1)
        num_tokens = min(input_ids.shape[0], buf.shape[0])
        if num_tokens == 0:
            return
        ngram_ids = self.compute_ngram_ids(
            input_ids[:num_tokens], query_start_loc, ngram_context
        )
        gathered = self.ngram_embedding(ngram_ids)  # [num_tokens, heads, head_dim]
        flat = gathered.reshape(num_tokens, -1).to(buf.dtype)
        # 只覆盖前 num_tokens 行 (copy_ 保持 buffer 地址不变, replay 仍读同址)。
        buf[:num_tokens].copy_(flat, non_blocking=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        # host-gather 模式: gather 已在 model_state.prepare_inputs -> host_gather()
        # 里填好常驻 buffer, forward 只 return 前 num_tokens 行的 view (纯 GPU
        # 操作, 无 D2H/H2D/自定义 op -> 可被 FULL cudagraph 整段 capture)。
        if self.prefetch_from_model_state:
            buf = self._ple_hg_buffer
            if buf is None:
                # buffer 未建 (load 尚未完成 / dummy 早期): 零占位保管道通。
                return torch.zeros(
                    (input_ids.reshape(-1).shape[0], self.embedding_dim),
                    dtype=self._ple_mmap_out_dtype,
                    device=input_ids.device,
                )
            return buf[: input_ids.reshape(-1).shape[0]]
        input_ids = input_ids.reshape(-1)
        num_tokens = input_ids.shape[0]
        out_dtype = _ple_out_dtype(self)
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
    cls.host_gather = host_gather
    cls._ple_mmap_patched = True
    logger.info("PLE mmap patch applied to %s.%s", cls.__module__, cls.__name__)


def attach_to_model_state(state, model) -> None:
    """给 ``Qwen4ExpModelState`` 找 host-gather PLE 模块 (overlay 注入用)。

    由注入到 model_state.py 的一行在 ``Qwen4ExpModelState.__init__`` 尾调用。
    扫模型找 prefetch_from_model_state 的 PLE 模块 (本模型仅 1 个 PLE 层,
    ple_layer_ids=[2], 且 PP>1 时只存在于 PP rank 0 —— 其他 rank 找不到, 静默
    no-op, 与 Minachist decode-02 行为一致), 找到则:
      * ``state._ple_hg_layer`` = 该模块 (prepare_inputs / prepare_dummy_inputs
        每步调它的 host_gather 在 capture 外预填 buffer);
      * ``state._ple_hg_dummy_ids`` = max_num_tokens 个 0 id (profiling/capture
        时 prepare_dummy_inputs 只有 token 数没有 token, 填 0 行号即可)。
    """
    if not host_gather_enabled():
        return
    for _, module in model.named_modules():
        if (
            getattr(module, "prefetch_from_model_state", False)
            and hasattr(module, "host_gather")
        ):
            state._ple_hg_layer = module
            state._ple_hg_dummy_ids = torch.zeros(
                state.max_num_tokens, dtype=torch.int32, device=state.device
            )
            logger.info(
                "PLE mmap(host-gather): model_state 钩子已挂载 layer=%s",
                getattr(module, "layer_name", "?"),
            )
            return


__all__ = ["maybe_apply", "enabled"]
