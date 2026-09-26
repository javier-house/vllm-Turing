# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM75 扩展 env —— 对上游 vllm/envs.py 的纯增量, 不再整文件覆盖。

底座保留上游原样 vllm/envs.py; 本模块只装 SM75 自定义的 env getter,
构建时由 install_sm75_overlay.py 把一行 `import vllm.envs_sm75; apply()`
append 到上游 envs.py 尾部触发注入。注入做两件事:

1) 把 EXTENSIONS getter 灌进 vllm.envs.environment_variables —— 上游
   __getattr__ / __dir__ / is_set / validate_environ / enable_envs_cache
   全围绕该 dict, 灌入后 envs.VLLM_FIREFLY 等属性访问自动生效(无需改调用方)。
2) 包一层 vllm.envs.compile_factors, 把 INSTALL_IGNORED(idle auto-sleep 计时器)
   从 hash factors pop 掉 —— 它们只影响调度/checkpoint, 不改变编译图。

好处: 上游 envs.py 后续升级随便改(内容级), 本文件只跟「dict + __getattr__
+ compile_factors 返回 dict」这个稳定机制耦合, 冲突面从整文件缩到一处注入。
"""

import os

# 需从 compile_factors hash 中排除的 key: idle auto-sleep 计时器/路径只影响
# 调度与 checkpoint bookkeeping, 不改变编译图。切 test/prod sleep 计时不应
# 使 torch.compile 缓存失配。
INSTALL_IGNORED = {
    "VLLM_AUTO_SLEEP_IDLE_TIMEOUT",
    "VLLM_AUTO_SLEEP_OFFLOAD_TARGET",
    "VLLM_AUTO_SLEEP_RELOAD_PATH",
    "VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL",
}


def _firefly_mode() -> str:
    """VLLM_FIREFLY 归一化: '1'=开(=auto) / '0'=关。

    未设默认 '0'(不激活); '1' 与 'auto' 等价(都算开); 其余值一律 '0'。
    开 = int4(AWQ/GPTQ) 走 int8 加速, fp8 走上游 marlin(fp8 加速走
    VLLM_FIREFLY_AR fp8 allreduce, 另见 PLAN-fp8-allreduce)。
    """
    v = os.getenv("VLLM_FIREFLY", "").strip().lower()
    return "1" if v in ("1", "auto", "on", "true", "yes") else "0"


def _firefly_ar_mode() -> str:
    """VLLM_FIREFLY_AR 归一化: 'auto'(默认, 跟随 VLLM_FIREFLY) / '0'=强制关 /
    'fp8'=强制开。

    auto = firefly 模式开 (VLLM_FIREFLY=1) 时 fp8 allreduce 自动启用 (fp8 近乎
    无损, PLAN-fp8-allreduce §3.5); firefly 关 → AR 关。'fp8' 单独开 (firefly
    不用也开 AR); '0' 单独关 (firefly 开但 AR 不用)。FireflyAllReduce 只做 fp8,
    双 backend (P2P / SHM, 见 VLLM_FIREFLY_AR_BACKEND)。
    """
    v = os.getenv("VLLM_FIREFLY_AR", "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return "0"
    if v in ("fp8", "1", "on", "true", "yes"):
        return "fp8"
    return "auto"


def _firefly_ar_backend() -> str:
    """VLLM_FIREFLY_AR_BACKEND 归一化: 'auto'(默认, 运行时按 _can_p2p 选 P2P
    优先) / 'p2p'(强制 P2P) / 'shm'(强制 SHM)。

    auto = 有 P2P (NVLink/PCIe 直连) 选 P2P (data/flag 全 device 显存, 无 host
    bounce); 无 P2P (PHB, 如 T10) 回 SHM (/dev/shm + cudaHostRegister)。
    'p2p'/'shm' 强制指定 (P2P 初始化失败仍自动回 SHM 兜底)。
    """
    v = os.getenv("VLLM_FIREFLY_AR_BACKEND", "").strip().lower()
    if v in ("p2p",):
        return "p2p"
    if v in ("shm", "host", "shared"):
        return "shm"
    return "auto"


def _firefly_ar_pipe() -> bool:
    """VLLM_FIREFLY_AR_PIPE 归一化: 默认关; '1'/'on'/'true'/'yes' 开。

    开 = 2-GPU SHM 大消息按 4MB 分块跨 2 stream 全双工 overlap (D2H 发 与 H2D
    收 并发), 全双工 PCIe 链 (二号机 GPU0<->host) 通信项 ~2x, 数值与串行
    firefly_ar_exchange 逐 bit 一致; 半双工链 (T10/一号机) 上分块串行化 ≈ 无
    回退。仅 2-GPU SHM 生效, 小消息 (<=4MB) 仍走串行。默认关保证不改变现有
    行为 (T10 零影响)。见 firefly_allreduce.py / firefly_ar_exchange_pipe。
    """
    return os.getenv("VLLM_FIREFLY_AR_PIPE", "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _monitor() -> bool:
    """VLLM_MONITOR 归一化: 默认开; '0'/'off'/'false'/'no' 关。

    开 = serve 在 /monitor 挂单文件 HTML 监控页(自拉同源 /metrics 渲染, 无 CDN,
    见 entrypoints/serve/instrumentator/monitor.py); 关 = 不挂该路由。
    """
    return os.getenv("VLLM_MONITOR", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _test_index() -> bool:
    """VLLM_TEST_INDEX 归一化: 默认开; '0'/'off'/'false'/'no' 关。

    开 = serve 在 /test 挂开源测速工具 llm_speedtest 的单文件 HTML 页(前端直连
    模型 API 测 Prefill/Decode 吞吐, 无 CDN, 见
    entrypoints/serve/instrumentator/test.html); 关 = 不挂该路由。
    """
    return os.getenv("VLLM_TEST_INDEX", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _ple_mmap() -> bool:
    """VLLM_PLE_MMAP 归一化: 默认关; '1'/'on'/'true'/'yes' 开。

    开 = qwen4_exp 的 PLE(ngram) 大表 (如 Qwen3.8-Flash-Next 的 105GB 表) 不进
    显存, 走 NVMe mmap 行 gather 经内核 page cache 换页 (见 vllm/vllm_ple_mmap.py)。
    gather 是 CPU 活 + pageable H2D, 改走图外 custom op —— 会改变编译图行为,
    故**不**加进 INSTALL_IGNORED、也不在 _compile_factors_sm75 pop: 开了就应是
    新编译产物。
    """
    return os.getenv("VLLM_PLE_MMAP", "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _ple_mmap_random() -> bool:
    """VLLM_PLE_MMAP_RANDOM 归一化: 默认开; '0'/'off' 关。

    开 = PLE mmap 标 MADV_RANDOM 关内核预读 (稀疏随机行 gather 用不上顺序预读,
    省 NVMe 带宽 + page cache)。见 vllm/vllm_ple_mmap.py::_apply_madv_random。
    """
    return os.getenv("VLLM_PLE_MMAP_RANDOM", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _ple_mmap_prewarm() -> bool:
    """VLLM_PLE_MMAP_PREWARM 归一化: 默认关; '1'/'on' 开。

    开 = 加载时把整表流读一遍填 page cache (无害可驱逐, 取决于空闲内存)。
    见 vllm/vllm_ple_mmap.py::MmapPleTable.prewarm。
    """
    return os.getenv("VLLM_PLE_MMAP_PREWARM", "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _ple_mmap_mode() -> str:
    """VLLM_PLE_MMAP_MODE 归一化: PLE(ngram) 表 offload 模式, 三选一可独立设置。

    - disk (默认 / 或仅 VLLM_PLE_MMAP=1): 磁盘 mmap + page cache 换页, 不进显存/
      常驻 RAM。
    - mem: 整表读进 RAM (numpy 连续数组), gather 纯内存无换页 (需 ~95G 主机内存)。
    - vram: 不 offload, 整表进显存 (占 ~95G, 仅小表/验证用)。

    优先级: 显式 MODE > VLLM_PLE_MMAP(1→disk) > 默认 disk。见
    vllm/vllm_ple_mmap.py::mode。纯运行时选择, 不改编译图 → 不进 _compile_factors。
    """
    m = os.getenv("VLLM_PLE_MMAP_MODE", "").strip().lower()
    if m:
        if m in ("mem", "memory", "ram", "cpu"):
            return "mem"
        if m in ("vram", "gpu", "none", "0", "off"):
            return "vram"
        return "disk"
    if os.getenv("VLLM_PLE_MMAP", "").strip().lower() in ("1", "on", "true", "yes"):
        return "disk"
    return "vram"


def _ple_mem_lazy() -> bool:
    """VLLM_PLE_MEM_LAZY 归一化: 默认开; '0'/'off' 关。

    开 = mem 模式走 lazy 后台填表: 启动时只建 header + 空数据区即 attach (秒级,
    不阻塞), 整表由 owner 的后台线程逐片填, gather 按片水位路由、未填片回退磁盘
    读。关 = 旧行为 (owner 阻塞填完整表再 attach, ~11 分钟在启动关键路径), 作回退
    保险丝。纯运行时选择, 不改编译图 → pop 出 compile_factors 复用历史编译产物。
    见 vllm/vllm_ple_mmap.py::MmapPleTable (lazy 分支)。
    """
    return os.getenv("VLLM_PLE_MEM_LAZY", "1").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _ple_mem_fill_workers() -> int:
    """VLLM_PLE_MEM_FILL_WORKERS 归一化: 默认 16。

    lazy-mem 后台填表的并行片数 (NVMe 多队列, 16 足够跑满; 与权重加载并发 IO 想
    少抢可降到 4 —— 反正不阻塞启动)。纯运行时, 不改编译图 → pop 出 compile_factors。
    见 vllm/vllm_ple_mmap.py::MmapPleTable._start_fill。
    """
    try:
        return int(os.environ.get("VLLM_PLE_MEM_FILL_WORKERS", "16"))
    except ValueError:
        return 16


# SM75 自定义 env → getter。install 时灌进 vllm.envs.environment_variables,
# 之后 envs.<NAME> 属性访问 / is_set / validate_environ / __dir__ 自动生效。
EXTENSIONS: dict[str, object] = {
    # firefly(SM75) prefill 加速总开关, 默认关(_firefly_mode 归一化):
    #   未设 / 0 = 关(全走上游 marlin, 默认不激活)。
    #   1 / auto = 开(两者等价): int4(AWQ/GPTQ, W4A16) 走 int8 加速; fp8 走上游
    #     marlin(sm75 实测 firefly-fp8 不比 marlin 快, 已移除; fp8 加速改走
    #     VLLM_FIREFLY_AR fp8 allreduce, 见 PLAN-fp8-allreduce)。
    # 大 M 现反量化成 int8 走 CUTLASS(IMMA), 小 M/decode 保持 marlin。见
    # model_executor/layers/quantization/utils/firefly.py。
    "VLLM_FIREFLY": _firefly_mode,
    # 默认 1024: T10 上 p3_perf_sweep(6144x5120 层)测得 int8 反量化 ~1ms/层
    # (M 无关地板), crossover M≈854; M>1024 int8 才稳定快于 marlin(1.13-1.33x)。
    "VLLM_FIREFLY_MIN_M": lambda: int(
        os.environ.get("VLLM_FIREFLY_MIN_M", "1024")
    ),
    # int8 prefill 反量化量化步: "def"(默认) 除法(与两遍 pass2 逐 bit 一致);
    # "fast" 乘倒数(per-row r=1/c_n, 再省 ~30% 反量化, off-by-one ≤0.06%)。
    "VLLM_FIREFLY_DEQUANT_MODEL": lambda: (
        os.environ.get("VLLM_FIREFLY_DEQUANT_MODEL", "def")
    ),
    # fp8 allreduce (FireflyAllReduce, SHM backend, 无 P2P 如 T10)。auto(默认)
    # 跟随 VLLM_FIREFLY (firefly 开→AR 自动开); fp8 单独开; 0 单独关。TP2 每层 2 次
    # AllReduce 量减半 (fp16->fp8), 省 ~480-530ms/27B prefill。见
    # distributed/device_communicators/firefly_allreduce.py / PLAN-fp8-allreduce。
    "VLLM_FIREFLY_AR": _firefly_ar_mode,
    # 只对大消息走 FireflyAllReduce, 小消息回退 NCCL。decode 小消息 (几百 KB)
    # 时 firefly 的 amax 扫描 + 多轮 flag spin 固定开销可能超过砍半省下的传输
    # 时间 → 反而更慢。默认 1MB (fp16 字节): 27B decode M=1 单 AR 8KB << 1MB,
    # 走 NCCL; prefill M>=256 (2MB) 起走 firefly。见 firefly_allreduce.py。
    "VLLM_FIREFLY_AR_MIN_SIZE": lambda: int(
        os.environ.get("VLLM_FIREFLY_AR_MIN_SIZE", "1048576")
    ),
    # FireflyAllReduce 传输 backend: auto(默认, 运行时 _can_p2p 选 P2P 优先) /
    # p2p(强制) / shm(强制)。P2P 有 NVLink/PCIe 直连时 data/flag 全 device 显存
    # 无 host bounce; 无 P2P (PHB 如 T10) 回 SHM。见 firefly_allreduce.py。
    "VLLM_FIREFLY_AR_BACKEND": _firefly_ar_backend,
    # SHM 分块全双工流水线开关, 默认关(_firefly_ar_pipe 归一化)。
    # 开 = 2-GPU SHM 大消息按 4MB 分块跨 2 stream overlap (全双工链通信项 ~2x,
    # 数值与串行一致); 半双工链 (T10/一号机) ≈ 无回退。仅 2-GPU SHM 生效。
    "VLLM_FIREFLY_AR_PIPE": _firefly_ar_pipe,
    # 单文件 HTML 监控页开关, 默认开(_monitor 归一化)。
    # 开 = serve 在 /monitor 挂自包含 HTML 看板(纯前端 canvas 图表, 无 CDN,
    # 轮询同源 /metrics); 0/off/false/no = 关(不挂路由)。见
    # entrypoints/serve/instrumentator/monitor.py。
    "VLLM_MONITOR": _monitor,
    # 单文件 HTML 测速页开关, 默认开(_test_index 归一化)。
    # 开 = serve 在 /test 挂 llm_speedtest 测速工具页(开源前端静态页, 直连模型
    # API 测 Prefill/Decode 吞吐, 无 CDN); 0/off/false/no = 关(不挂路由)。见
    # entrypoints/serve/instrumentator/test.html。
    "VLLM_TEST_INDEX": _test_index,
    # PLE(ngram) 大表 NVMe mmap 总开关, 默认关(_ple_mmap 归一化)。
    # 开 = qwen4_exp PLE 表 (如 105GB 的 Qwen3.8-Flash-Next) 不进显存, 走内核
    # page cache 换页行 gather。改变编译图 (gather 走图外 op) → 正常参与编译
    # hash, 不加 INSTALL_IGNORED、不 pop。见 vllm/vllm_ple_mmap.py。
    "VLLM_PLE_MMAP": _ple_mmap,
    # PLE(ngram) 表 offload 模式: disk/mem/vram 三选一 (_ple_mmap_mode 归一化)。
    # disk=磁盘 mmap 换页(默认), mem=整表进 RAM, vram=不进显存 (不 offload, 不打
    # patch → 图不同) → 影响编译图, 正常参与编译 hash, 不 pop。
    "VLLM_PLE_MMAP_MODE": _ple_mmap_mode,
    # PLE mmap 标 MADV_RANDOM 关内核预读, 默认开(_ple_mmap_random 归一化)。
    "VLLM_PLE_MMAP_RANDOM": _ple_mmap_random,
    # PLE mmap 加载时预热 page cache, 默认关(_ple_mmap_prewarm 归一化)。
    "VLLM_PLE_MMAP_PREWARM": _ple_mmap_prewarm,
    # PLE mmap gather 线程数 (运行时直读, 注册仅 validate_environ + hash)。
    "VLLM_PLE_MMAP_WORKERS": lambda: int(
        os.environ.get("VLLM_PLE_MMAP_WORKERS", "32")
    ),
    # PLE mmap 每 gather 任务行数 (运行时直读, 注册仅 validate_environ + hash)。
    "VLLM_PLE_MMAP_CHUNK": lambda: int(
        os.environ.get("VLLM_PLE_MMAP_CHUNK", "2048")
    ),
    # PLE mem 模式 lazy 后台填表开关, 默认开(_ple_mem_lazy 归一化)。
    # 开 = 启动秒级 attach + owner 后台逐片填 + gather 按片水位路由 (未填片回退
    # 磁盘); 0 = 旧行为 (owner 阻塞填完再 attach)。纯运行时, 不改编译图。
    # 见 vllm/vllm_ple_mmap.py::MmapPleTable。
    "VLLM_PLE_MEM_LAZY": _ple_mem_lazy,
    # PLE lazy-mem 后台填表并行片数 (运行时直读, 默认 16; 同权重加载 IO 抢带宽
    # 可降到 4, 反正不阻塞启动)。纯运行时, 不改编译图。
    # 见 vllm/vllm_ple_mmap.py::MmapPleTable._start_fill。
    "VLLM_PLE_MEM_FILL_WORKERS": _ple_mem_fill_workers,
    # A3(sm75 参考): custom allreduce 在 cuda graph capture 时的图输入策略。
    # auto=full decode 走 registered 快路径, piecewise/prefill 回退 staging
    # buffer(sm75 图私有大 buffer 无法经 CUDA IPC 导出); registered/staging
    # 可强制覆盖。
    "VLLM_CUSTOM_ALLREDUCE_GRAPH_INPUT_MODE": lambda: os.getenv(
        "VLLM_CUSTOM_ALLREDUCE_GRAPH_INPUT_MODE", "auto"
    ),
    # vllm-turing overlay: idle auto-sleep. Populated by EngineArgs from the
    # --auto-sleep-* CLI flags; consumed inside the engine-core process by
    # vllm.v1.engine.auto_sleep(那里用 os.environ.get 直读, 此处注册仅为让
    # validate_environ 不告警 + 被 compile_factors 看到后由 INSTALL_IGNORED 排除)。
    # Idle minutes (float) before the engine auto-sleeps; 0 disables.
    "VLLM_AUTO_SLEEP_IDLE_TIMEOUT": lambda: float(
        os.getenv("VLLM_AUTO_SLEEP_IDLE_TIMEOUT", "0")
    ),
    # 'cpu' (sleep level 1, pinned CPU backup) or 'reload' (sleep level 2,
    # weights discarded and reloaded from the checkpoint on wake).
    "VLLM_AUTO_SLEEP_OFFLOAD_TARGET": lambda: os.getenv(
        "VLLM_AUTO_SLEEP_OFFLOAD_TARGET", "cpu"
    ),
    # Checkpoint path used to reload weights on wake for the 'reload' target.
    "VLLM_AUTO_SLEEP_RELOAD_PATH": lambda: os.getenv("VLLM_AUTO_SLEEP_RELOAD_PATH", ""),
    # Seconds between page-cache re-warm ticks while sleeping (reload mode);
    # keeps the checkpoint in the OS page cache so the wake-time reload read
    # is fast. 0 disables the background keeper (one-shot warm on sleep/wake
    # still happens).
    "VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL": lambda: float(
        os.getenv("VLLM_AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL", "600")
    ),
}


def apply() -> None:
    """把 SM75 扩展 env 注入上游 vllm.envs(幂等)。

    由 install_sm75_overlay.py append 到上游 envs.py 尾部的
    `import vllm.envs_sm75; vllm.envs_sm75.apply()` 触发。此刻 envs 模块的
    environment_variables / compile_factors 均已定义(append 在模块体最末)。
    用 sys.modules 取正在加载的 vllm.envs(即 append 所在模块本身), 不触发
    `import vllm` 的副作用链; 测试里 patch sys.modules["vllm.envs"] 即可复用。
    """
    import sys

    envs = sys.modules["vllm.envs"]

    # 1) 注入 getter —— dict.update 天然幂等(同 key 覆盖同值)。
    envs.environment_variables.update(EXTENSIONS)

    # 2) 包 compile_factors, 从 hash factors pop 掉 INSTALL_IGNORED(幂等: 已包跳过)。
    if not getattr(envs.compile_factors, "_sm75_wrapped", False):
        _orig = envs.compile_factors

        def _compile_factors_sm75():
            factors = _orig()
            for key in INSTALL_IGNORED:
                factors.pop(key, None)
            # 监控看板/测速页都只挂 HTTP 路由, 不影响编译图; 沿用"关"的历史缓存
            # 签名, 使开/关 UI 都能复用既有生产编译产物。不改 getter: 开关实际仍
            # 按 VLLM_MONITOR / VLLM_TEST_INDEX 原值生效, 这里只归一化 hash 因子:
            # VLLM_MONITOR 沿用历史签名(=False); VLLM_TEST_INDEX 是新增 env, 历史
            # 签名里没有该 key, pop 掉才能与之对齐。
            if "VLLM_MONITOR" in factors:
                factors["VLLM_MONITOR"] = False
            factors.pop("VLLM_TEST_INDEX", None)
            # PLE lazy-mem 两个 env 纯运行时 (后台填表行为 + 填表线程数), 不改编译
            # 图; 都是新增 env, 历史编译签名里没有这俩 key, pop 掉才能与之对齐,
            # 复用既有生产编译产物 (getter 实际仍按 env 原值生效)。
            factors.pop("VLLM_PLE_MEM_LAZY", None)
            factors.pop("VLLM_PLE_MEM_FILL_WORKERS", None)
            return factors

        _compile_factors_sm75._sm75_wrapped = True  # type: ignore[attr-defined]
        _compile_factors_sm75.__wrapped__ = _orig
        envs.compile_factors = _compile_factors_sm75
