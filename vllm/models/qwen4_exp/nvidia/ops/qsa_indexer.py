# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 移植上游 #54873 (31a8a266) 的 QSA indexer prefill/decode 路径分离:
# decode 走 request-major uniform kernel (program_id=request, 读预算好的
# visible_blocks), prefill 保留原逐 query 打分逻辑。#54873 态的 expand kernel
# 自带 packed 尾列 (TRAILING COUNT COLUMN, output_width+1), 与我方 ops/qsa.py
# 的 packed 消费者 (selection_width=shape[1]-1) 直接兼容。
# sm75 降档: 上游 tile 档是大显存旗舰卡调的 (prefill TILE_R=64/BLOCK_N=64/
# K_TILES=16, decode BLOCK_N=64), sm75 (64KB shared 上限, 无 bf16
# tensor-core) 会 OOM 或串行化, 故在 launch 处按 device capability 降档
# (见 _decode_block_n / _prefill_logits 的中文注释)。
"""Triton kernels for Qwen4Exp QSA index selection."""

import torch

import vllm.envs as envs
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_TOPK_WORKSPACE_BYTES = 1024 * 1024
# 上游档 decode 块宽; sm75 档见 _decode_block_n()。
_DECODE_BLOCK_N = 64


@triton.jit
def _qsa_mqa_paged_uniform_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
    DECODE_QUERY_LEN_PADDED: tl.constexpr = triton.next_power_of_2(DECODE_QUERY_LEN)
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    query_offsets = tl.arange(0, DECODE_QUERY_LEN_PADDED)
    valid_query_offsets = query_offsets < DECODE_QUERY_LEN
    rows = request * DECODE_QUERY_LEN + query_offsets
    visible = tl.load(
        visible_blocks_ptr + rows,
        mask=valid_query_offsets,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    if tile_start * BLOCK_N >= max_visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(max_visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(NUM_COLUMNS, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    n = tl.arange(0, DECODE_QUERY_LEN_PADDED * NUM_HEADS_PADDED)
    query_offset = n // NUM_HEADS_PADDED
    head = n % NUM_HEADS_PADDED
    valid_query = (query_offset < DECODE_QUERY_LEN) & (head < NUM_HEADS)
    query = tl.load(
        q_ptr
        + (request * DECODE_QUERY_LEN + query_offset)[None, :] * stride_q_row
        + head[None, :] * stride_q_head
        + dims[:, None],
        mask=valid_query[None, :] & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        keys = tl.load(
            k_cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(valid_query[None, :], tl.maximum(scores, 0.0), 0.0)
        scores = tl.reshape(
            scores,
            (BLOCK_N, DECODE_QUERY_LEN_PADDED, NUM_HEADS_PADDED),
        )
        score = tl.sum(scores, axis=2)
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=valid_query_offsets[None, :]
            & (columns[:, None] < NUM_COLUMNS)
            & (columns[:, None] < visible[None, :]),
        )


@triton.jit(do_not_specialize=["num_rows", "query_offset"])
def _qsa_mqa_paged_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_start_loc_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_rows,
    query_offset,
    page_table_width,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    num_columns = page_table_width * PAGE_SIZE
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    query_base = tl.load(query_start_loc_ptr)
    query_end = query_offset + num_rows
    request_start = tl.maximum(
        tl.load(query_start_loc_ptr + request) - query_base, query_offset
    )
    request_end = tl.minimum(
        tl.load(query_start_loc_ptr + request + 1) - query_base, query_end
    )
    absolute_row_start = request_start + tl.program_id(1) * TILE_R
    if absolute_row_start >= request_end:
        return

    lanes = tl.arange(0, TILE_R)
    absolute_rows = absolute_row_start + lanes
    rows = absolute_rows - query_offset
    valid_rows = absolute_rows < request_end
    visible = tl.load(
        visible_blocks_ptr + absolute_rows,
        mask=valid_rows,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    k_tile_start = tl.program_id(2) * K_TILES
    if k_tile_start * BLOCK_N >= max_visible:
        return
    k_tile_end = tl.minimum(k_tile_start + K_TILES, tl.cdiv(max_visible, BLOCK_N))
    k_tile_end = tl.minimum(k_tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    m = tl.arange(0, TILE_R * NUM_HEADS_PADDED)
    q_row_offsets = m // NUM_HEADS_PADDED
    q_rows = absolute_row_start + q_row_offsets
    heads = m % NUM_HEADS_PADDED
    query = tl.load(
        q_ptr
        + q_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None],
        mask=(heads[None, :] < NUM_HEADS)
        & (absolute_row_start + q_row_offsets[None, :] < request_end)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(k_tile_start, k_tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, page_table_width - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        keys = tl.load(
            k_cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_N, TILE_R, NUM_HEADS_PADDED))
        score = tl.sum(tl.maximum(scores, 0.0), axis=2)
        store_mask = (
            valid_rows[None, :]
            & (columns[:, None] < visible[None, :])
            & (columns[:, None] < num_columns)
        )
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=store_mask,
        )


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    visible_blocks_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    OUTPUT_WIDTH: tl.constexpr = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    visible_blocks = tl.load(visible_blocks_ptr + row)
    complete_blocks = tl.minimum(visible_blocks, BLOCK_TOPK)
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (columns < OUTPUT_WIDTH) & (is_expanded | is_tail) & (token >= 0)
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=columns < OUTPUT_WIDTH,
    )
    # 尾列 (packed 约定): 输出 buffer 宽 OUTPUT_WIDTH+1, 最后一列存本行有效
    # 条目数 (完整块展开 + 因果尾), 不是 token 下标 —— sparse attention
    # kernel 读它作 tile 循环上界 (与我方 ops/qsa.py 的 packed 消费者一致)。
    if tl.program_id(1) == 0:
        tl.store(
            output_ptr + row * stride_output_row + OUTPUT_WIDTH * stride_output_column,
            expanded_count + tail_count,
        )


def _decode_block_n() -> int:
    """decode kernel 的列块宽: 上游档 64, sm75 档 32。

    sm75 降档算式 —— 本模型 config: indexer_n_heads=4
    (NUM_HEADS_PADDED=4), indexer_head_dim=128 (BLOCK_D=128),
    DECODE_QUERY_LEN_PADDED=dql_padded (1+投机 token 数, 上界按 8→16 估):
      - keys tile [BLOCK_N, 128] fp16 = BLOCK_N*256 B, STAGES=2 流水双缓冲
        → BLOCK_N=64 时 32 KB, =32 时 16 KB;
      - query [128, N] fp16 作 dot B 操作数 (N=dql_padded*4, 最坏 64): 最坏
        16 KB;
      - scores 累加器 [BLOCK_N, N] fp32 占寄存器: =64 时最坏 16 KB / 64 线程
        (num_warps=2) = 每线程 256 B 起步, 加 query/指针/索引逼近 255 寄存器
        上限 → 溢出串行化; =32 时减半 (~8 KB) 安全。
    合计 shared: BLOCK_N=64 最坏 32+16+流水开销 ≈ 64 KB 顶格 (sm75 单块
    上限 64 KB, 对齐 ops/qsa.py 64 列降 16 列的先例); =32 时 ≈ 40 KB, 留足
    余量。TILES_PER_PROG 档位阈值按 program 数分档, BLOCK_N 减半 → program
    数翻倍 → 自动落低一档, 每 program 列跨度不变, 无需另调。
    """
    if not current_platform.has_device_capability(80):
        return 32
    return _DECODE_BLOCK_N


def _decode_tiles_per_program(num_requests: int, columns: int, block_n: int) -> int:
    programs = num_requests * triton.cdiv(columns, block_n)
    if programs < 16384:
        return 1
    if programs < 32768:
        return 2
    if programs < 131072:
        return 4
    return 8


def _qsa_decode_warmup_profiles(
    max_dql: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
    columns: int,
    block_n: int,
) -> tuple[tuple[int, int], ...]:
    profiles: list[tuple[int, int]] = []
    for dql in range(1, max_dql + 1):
        max_reqs = min(max_num_reqs, max_num_batched_tokens // dql)
        requests_by_grouping: dict[int, int] = {}
        for num_requests in range(1, max_reqs + 1):
            tiles_per_program = _decode_tiles_per_program(num_requests, columns, block_n)
            requests_by_grouping.setdefault(tiles_per_program, num_requests)
        profiles.extend(
            (dql, num_requests) for num_requests in requests_by_grouping.values()
        )
    return tuple(profiles)


def warmup_qsa_mqa_paged_decode(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    max_decode_query_len: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> tuple[tuple[int, int], ...]:
    """Compile every reachable decode specialization without launching it."""

    page_size = k_cache.shape[1]
    page_table_width = page_table.shape[1]
    columns = page_table_width * page_size
    # 档必须与真实 launch (qsa_select_paged_decode) 一致, 否则 warmup 白编。
    block_n = _decode_block_n()
    profiles = _qsa_decode_warmup_profiles(
        max_decode_query_len,
        max_num_reqs,
        max_num_batched_tokens,
        columns,
        block_n,
    )
    if not profiles:
        return ()

    k_cache_ptr = TritonWarmupTensor(k_cache.dtype, shape=tuple(k_cache.shape))
    page_table_ptr = TritonWarmupTensor(
        page_table.dtype,
        shape=(max_num_reqs, page_table_width),
    )
    visible_blocks_ptr = TritonWarmupTensor(torch.int32)

    for decode_query_len, num_requests in profiles:
        num_rows = decode_query_len * num_requests
        # q mock 必须与 k_cache mock 同 dtype: 真实 dispatch 里 q 与
        # compressed_key_cache 都是模型 dtype (sm75 fp16), 无 cast 直接进
        # tl.dot —— 两操作数 dtype 必须一致。上游 #54873 写死 bf16 (上游档
        # 巧合正确), 这里改用 k_cache.dtype 跟随实跑 dtype (fp16 模型→fp16)。
        q_ptr = TritonWarmupTensor(
            k_cache.dtype,
            shape=(num_rows, num_heads, head_dim),
        )
        logits_ptr = TritonWarmupTensor(
            torch.float32,
            shape=(num_rows, columns),
        )
        tiles_per_program = _decode_tiles_per_program(num_requests, columns, block_n)
        _qsa_mqa_paged_uniform_kernel.warmup(
            q_ptr,
            k_cache_ptr,
            page_table_ptr,
            visible_blocks_ptr,
            logits_ptr,
            num_heads * head_dim,
            head_dim,
            k_cache.stride(0),
            k_cache.stride(1),
            page_table.stride(0),
            columns,
            PAGE_SIZE=page_size,
            PAGE_TABLE_WIDTH=page_table_width,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            DECODE_QUERY_LEN=decode_query_len,
            BLOCK_N=block_n,
            TILES_PER_PROG=tiles_per_program,
            STAGES=2,
            num_warps=2,
            grid=(
                num_requests,
                triton.cdiv(columns, block_n * tiles_per_program),
            ),
        )
    return profiles


def _prefill_logits(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    max_query_len: int,
    logits_width: int,
    query_offset: int,
    num_queries: int,
) -> torch.Tensor:
    assert query_start_loc.shape == (page_table.shape[0] + 1,)
    assert visible_blocks.shape == (q.shape[0],)
    assert 0 <= query_offset <= query_offset + num_queries <= q.shape[0]
    assert 0 < logits_width <= page_table.shape[1] * k_cache.shape[1]

    logits = torch.empty(
        (num_queries, logits_width), dtype=torch.float32, device=q.device
    )
    # 上游档: TILE_R=64/BLOCK_N=64/K_TILES=16。sm75 档降为
    # TILE_R=16/BLOCK_N=32/K_TILES=8, 算式 (num_warps=4 → 128 线程,
    # 本模型 NUM_HEADS_PADDED=4, BLOCK_D=128):
    #   - scores 累加器 [BLOCK_N, TILE_R*4] fp32: 上游档 [64,256]=64 KB
    #     → 每线程 512 B (128 寄存器) 起步, 叠加 query 寄存器必溢出 255
    #     上限; sm75 档 [32,64]=8 KB → 每线程 64 B (16 寄存器) 安全。
    #   - keys tile [BLOCK_N, 128] fp16 STAGES=2 双缓冲: 上游 32 KB /
    #     sm75 16 KB; query [128, TILE_R*4] fp16 作 dot B 操作数: 上游
    #     32 KB / sm75 16 KB → sm75 合计 shared ≈ 32 KB + 流水开销 < 64 KB
    #     上限 (上游档合计 ≈ 64 KB 顶格, 且其 shared 上限 228 KB 本就不紧)。
    #   - grid 维度: 行 tile 数 64→16 升 4 倍, k tile 数 1024→256 升 4 倍,
    #     总 program 数 16 倍, 单 program 工作量 1/8 → 总工作量不变, 换取
    #     更细并行 (sm75 弱单 CTA 吞吐, 靠多 CTA 填 SM)。
    if current_platform.has_device_capability(80):
        TILE_R, BLOCK_N, K_TILES = 64, 64, 16
    else:
        TILE_R, BLOCK_N, K_TILES = 16, 32, 8
    grid = (
        page_table.shape[0],
        triton.cdiv(min(num_queries, max_query_len), TILE_R),
        triton.cdiv(logits_width, BLOCK_N * K_TILES),
    )
    _qsa_mqa_paged_prefill_kernel[grid](
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        logits,
        *q.stride()[:-1],
        *k_cache.stride()[:2],
        *page_table.stride()[:-1],
        *logits.stride()[:-1],
        num_queries,
        query_offset,
        page_table.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        TILE_R=TILE_R,
        BLOCK_N=BLOCK_N,
        K_TILES=K_TILES,
        STAGES=2,
        num_warps=4,
    )
    return logits


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    visible_blocks: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor,
) -> None:
    """展开压缩块, 并补齐当前未闭合组 (open group) 的因果尾。"""

    assert token_topk % compress_ratio == 0
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    assert block_indices.shape == (query_positions.numel(), block_topk)
    assert visible_blocks.shape == query_positions.shape
    # +1: packed buffer 的尾列存本行有效条目数 (不是 token 下标); 下方索引
    # 区只写 [0, output_width) 列, 尾列由 kernel 的 program_id(1)==0 写。
    assert out.shape == (block_indices.shape[0], output_width + 1)
    column_block = 256
    grid = (block_indices.shape[0], triton.cdiv(output_width, column_block))
    _expand_qsa_indices_kernel[grid](
        block_indices,
        query_positions,
        visible_blocks,
        out,
        *block_indices.stride(),
        *out.stride(),
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )


def _topk(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    block_indices: torch.Tensor,
    topk_workspace: torch.Tensor,
) -> None:
    # similar dispatch logic as DeepSeek indexer
    block_topk = token_topk // compress_ratio
    # cooperative_topk 仅高 CC (9.0+) 命中; sm75 走 persistent_topk (上游自带分派)。
    use_cooperative_topk = (
        logits.shape[0] <= 64
        and logits.stride(0) % 4 == 0
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
    )
    topk_op = (
        torch.ops._C.cooperative_topk
        if use_cooperative_topk
        else torch.ops._C.persistent_topk
    )
    topk_op(
        logits,
        visible_blocks,
        block_indices,
        topk_workspace,
        block_topk,
        logits.shape[1],
    )


def qsa_select_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    decode_query_len: int,
    block_indices: torch.Tensor,
) -> None:
    """对 request-major 的 decode batch 打分并选块。

    Args:
        q: Query tensor shaped ``[num_requests * decode_query_len, heads,
            head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Request block table shaped ``[num_requests, max_pages]``.
        visible_blocks: Number of visible compressed blocks per query.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        decode_query_len: Number of query tokens per request.
        block_indices: Compressed-index output buffer.
    """

    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    assert decode_query_len > 0 and q.shape[0] % decode_query_len == 0
    num_requests = q.shape[0] // decode_query_len
    assert page_table.shape[0] == num_requests
    assert visible_blocks.shape == (q.shape[0],)

    columns = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    block_n = _decode_block_n()
    tiles_per_program = _decode_tiles_per_program(num_requests, columns, block_n)
    grid = (
        num_requests,
        triton.cdiv(columns, block_n * tiles_per_program),
    )
    _qsa_mqa_paged_uniform_kernel[grid](
        q,
        k_cache,
        page_table,
        visible_blocks,
        logits,
        *q.stride()[:-1],
        *k_cache.stride()[:2],
        *page_table.stride()[:-1],
        *logits.stride()[:-1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        DECODE_QUERY_LEN=decode_query_len,
        BLOCK_N=block_n,
        TILES_PER_PROG=tiles_per_program,
        STAGES=2,
        num_warps=2,
    )
    _topk(
        logits,
        visible_blocks,
        token_topk,
        compress_ratio,
        block_indices,
        torch.empty((_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device),
    )


def qsa_select_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    max_query_len: int,
    block_indices: torch.Tensor,
    max_seq_len: int,
) -> None:
    """分块 (bounded chunks) 对 prefill 打分并选压缩块。

    Args:
        q: Packed prefill query tensor shaped ``[num_tokens, heads, head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Block table shaped ``[num_requests, max_pages]``.
        query_start_loc: Packed prefill query offsets with a terminal offset.
            Offsets may share the base of a larger backing tensor.
        visible_blocks: Number of visible compressed blocks per query.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        max_query_len: Maximum number of query tokens in one request.
        block_indices: Compressed-index output buffer.
        max_seq_len: Longest context length in the batch this step.
    """

    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    rows = q.shape[0]
    # 没有任何一行需要给 cdiv(max_seq_len, compress_ratio) 列之后的压缩块
    # 打分。向上取 64 的倍数保持 logits 行宽与 cooperative_topk 兼容。
    logits_width = triton.cdiv(triton.cdiv(max_seq_len, compress_ratio), 64) * 64
    logits_width = min(max(64, logits_width), page_table.shape[1] * k_cache.shape[1])

    # 按 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB 限制临时 logits 显存, 分块打分。
    max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    rows_per_chunk = max(1, max_logits_bytes // (logits_width * 4))
    topk_workspace = torch.empty(
        (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device
    )

    for query_start in range(0, rows, rows_per_chunk):
        query_end = min(query_start + rows_per_chunk, rows)
        query_slice = slice(query_start, query_end)
        logits = _prefill_logits(
            q,
            k_cache,
            page_table,
            query_start_loc,
            visible_blocks,
            max_query_len,
            logits_width,
            query_offset=query_start,
            num_queries=query_end - query_start,
        )
        _topk(
            logits,
            visible_blocks[query_slice],
            token_topk,
            compress_ratio,
            block_indices[query_slice],
            topk_workspace,
        )


__all__ = [
    "expand_qsa_block_indices",
    "qsa_select_paged_decode",
    "qsa_select_paged_prefill",
    "warmup_qsa_mqa_paged_decode",
]
