# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""自包含监控看板: serve 在 /monitor 返回本目录 dashboard.html 单文件页。

页面纯前端(无 CDN/无外部依赖, 图表用手写 canvas 实现), 轮询同源 /metrics
(Prometheus 文本)渲染并发/KV 趋势、prefix 缓存命中、token 统计、延迟分位
(P50/P90/P99)、prompt/generation token 分布、preemption 与 sleep 状态。
HTML 独立成 dashboard.html, 可直接用浏览器打开看样式; 每次请求读盘, 改样式
无需重启。VLLM_MONITOR 默认开, '0'/'off'/'false'/'no' 关(不挂路由)。

同目录 test.html 是引入的开源 llm_speedtest 测速页(前端直连模型 API 测
Prefill/Decode 吞吐), 由 VLLM_TEST_INDEX 控制 /test 路由, 同默认开/同关值。
"""

import hashlib
import os
import secrets
import subprocess
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from vllm import envs

# 看板 HTML(独立文件, 便于直接打开/编辑), 与 monitor.py 同目录。
_DASHBOARD_HTML_PATH = Path(__file__).resolve().parent / "dashboard.html"

# 测速页 HTML(引入开源 llm_speedtest 静态页, 直连模型 API 测吞吐), 同目录。
_TEST_HTML_PATH = Path(__file__).resolve().parent / "test.html"

# 关值集合(移植到其他 vllm 时, 若其 envs 无对应字段, 回退直读该 env)。
_OFF_VALUES = ("0", "off", "false", "no")


def _monitor_enabled() -> bool:
    """VLLM_MONITOR 开关: 走 envs 归一化; 移植到无此字段的 vllm 时
    (envs.VLLM_MONITOR 抛 AttributeError), 回退直读环境变量, 未设默认开。"""
    try:
        return bool(envs.VLLM_MONITOR)
    except AttributeError:
        return os.environ.get("VLLM_MONITOR", "1").strip().lower() not in _OFF_VALUES


def _test_index_enabled() -> bool:
    """VLLM_TEST_INDEX 开关: 同 _monitor_enabled 的双路取法, 未设默认开。"""
    try:
        return bool(envs.VLLM_TEST_INDEX)
    except AttributeError:
        return (
            os.environ.get("VLLM_TEST_INDEX", "1").strip().lower() not in _OFF_VALUES
        )


def _query_gpus() -> dict:
    """nvidia-smi 一次性采集所有 GPU 的瞬时状态(无状态, 每次调用现采)。

    返回 {ok:bool, gpus:[{...}], error:str}。前端 /monitor/gpu 轮询本端点
    拿 GPU 监控数据(温度/显存/功率/频率/PCIe/带宽/ECC/利用率)。

    失败(无 nvidia-smi / 非 NVIDIA / 超时 / 解析异常)时返回 {ok:False,
    gpus:[], error:msg}, 前端据此降级为「GPU 数据不可用」, 绝不抛到路由层
    拖垮 /monitor。nvidia-smi 是 NVIDIA 容器运行时的标准入口, vllm 容器内
    只要 host 装了驱动即可用, 无需额外权限。
    """
    fields = [
        "index", "name",
        "temperature.gpu",
        "memory.used", "memory.total",
        "power.draw", "power.limit",
        "clocks.current.graphics", "clocks.current.memory",
        "pcie.link.gen.current", "pcie.link.gen.max",
        "pcie.link.width.current", "pcie.link.width.max",
        "utilization.gpu", "utilization.memory",
        "ecc.errors.corrected.volatile.total",
        "ecc.errors.uncorrected.volatile.total",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5
        )
    except FileNotFoundError:
        return {"ok": False, "gpus": [], "error": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "gpus": [], "error": "nvidia-smi timeout"}
    except Exception as exc:  # noqa: BLE001 - 任何异常都降级, 不让路由崩
        return {"ok": False, "gpus": [], "error": f"nvidia-smi error: {exc}"}
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        # 截断避免长报错灌进看板
        return {"ok": False, "gpus": [], "error": err[:200] or f"exit {proc.returncode}"}
    gpus: list[dict] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            continue
        row = dict(zip(fields, parts))

        def _num(key: str):
            try:
                return float(row[key])
            except (ValueError, KeyError):
                return None

        gpus.append({
            "index": int(_num("index") or 0),
            "name": row.get("name") or "GPU",
            "temp": _num("temperature.gpu"),
            "mem_used": _num("memory.used"),
            "mem_total": _num("memory.total"),
            "power": _num("power.draw"),
            "power_limit": _num("power.limit"),
            "clk_g": _num("clocks.current.graphics"),
            "clk_m": _num("clocks.current.memory"),
            "pcie_gen": _num("pcie.link.gen.current"),
            "pcie_gen_max": _num("pcie.link.gen.max"),
            "pcie_width": _num("pcie.link.width.current"),
            "pcie_width_max": _num("pcie.link.width.max"),
            "util": _num("utilization.gpu"),
            "mem_bw": _num("utilization.memory"),
            "ecc_s": _num("ecc.errors.corrected.volatile.total"),
            "ecc_d": _num("ecc.errors.uncorrected.volatile.total"),
        })
    if not gpus:
        return {"ok": False, "gpus": [], "error": "no GPU parsed"}
    return {"ok": True, "gpus": gpus, "error": ""}


def _configured_api_keys(request: Request) -> list[str]:
    """取配置的 API key 列表(CLI --api-key 优先, VLLM_API_KEY 兜底)。

    与 api_server.py 取 token 的逻辑一致: args.api_key 为空时回退 env。
    """
    args = getattr(request.app.state, "args", None)
    cli_keys = getattr(args, "api_key", None) if args is not None else None
    if cli_keys:
        return [k for k in cli_keys if k]
    try:
        env_key = envs.VLLM_API_KEY
    except AttributeError:
        env_key = os.environ.get("VLLM_API_KEY", "")
    return [env_key] if env_key else []


def _verify_api_key(request: Request) -> bool:
    """校验请求头 x-api-key-hash 是否匹配任一配置 key 的 SHA-256。

    未配置任何 --api-key 时直接放行(与 vLLM 无 key 时全开放一致)。
    前端把 API key 哈希成 hex 再传(原 key 不裸传), 这里对每个配置 key 算
    SHA-256 后 compare_digest 比对, 与 authenticate.py 同一套手法。
    """
    keys = _configured_api_keys(request)
    if not keys:
        return True
    digest = request.headers.get("x-api-key-hash", "").strip().lower()
    if not digest:
        return False
    for key in keys:
        if secrets.compare_digest(digest, hashlib.sha256(key.encode("utf-8")).hexdigest()):
            return True
    return False


def attach_router(app: FastAPI) -> None:
    """按 VLLM_MONITOR / VLLM_TEST_INDEX 开关把 /monitor、/test 挂到 app。

    各自关(0/off/false/no)时不挂对应路由, 都关则不挂任何路由。/monitor
    与 /test 均不在 GUARDED_PREFIX 内, 浏览器无需 API key 即可访问; 页面内
    fetch /metrics 亦同源无鉴权。每次请求读盘 HTML, 改样式直接刷新即可
    (无需重启)。

    /monitor 额外挂投机解码开关:
    - GET  /monitor/spec_decode  只读, 无鉴权, 返回 {spec_configured, enabled}
    - POST /monitor/spec_decode  写, 需 x-api-key-hash(SHA-256), 切换开关
    开关只跳草稿计算, 不卸显存; 未配 --speculative-config 时 POST 返回 409。
    """
    if not _monitor_enabled() and not _test_index_enabled():
        return

    if _monitor_enabled():

        @app.get("/monitor", response_class=HTMLResponse, include_in_schema=False)
        def monitor() -> HTMLResponse:  # noqa: N802
            return HTMLResponse(_DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))

        @app.get("/monitor/gpu", include_in_schema=False)
        async def get_gpu() -> JSONResponse:  # noqa: N802
            """只读, 无鉴权(与 /metrics 一致)。前端 GPU 监控部件轮询本端点。

            每次现调 nvidia-smi 采一次(无状态, 不缓存), 前端负责累加温度/功率
            history 画曲线。失败时 ok=False, 前端降级不报错。
            """
            return JSONResponse(_query_gpus())

        @app.get("/monitor/spec_decode", include_in_schema=False)
        async def get_spec_decode(request: Request) -> JSONResponse:  # noqa: N802
            engine = request.app.state.engine_client
            configured = await engine.is_speculative_decoding_configured()
            enabled = (
                await engine.is_speculative_decoding_enabled() if configured else False
            )
            return JSONResponse(
                {
                    "spec_configured": configured,
                    "enabled": enabled,
                    # 无 --api-key 时为 False, 前端据此免弹 key 输入框。
                    "auth_required": bool(_configured_api_keys(request)),
                }
            )

        @app.post("/monitor/spec_decode", include_in_schema=False)
        async def set_spec_decode(request: Request) -> JSONResponse:  # noqa: N802
            if not _verify_api_key(request):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            try:
                payload = await request.json()
            except Exception:  # noqa: BLE001 - 空/坏 body 一律当 {enabled:true}
                payload = {}
            enabled = bool(payload.get("enabled", True))
            engine = request.app.state.engine_client
            if not await engine.is_speculative_decoding_configured():
                return JSONResponse(
                    {"error": "speculative decoding not configured"},
                    status_code=409,
                )
            await engine.set_speculative_decoding(enabled)
            return JSONResponse({"ok": True, "enabled": enabled})

    if _test_index_enabled():

        @app.get("/test", response_class=HTMLResponse, include_in_schema=False)
        def test_index() -> HTMLResponse:  # noqa: N802
            """llm_speedtest 测速页: 前端直连模型 API 测 Prefill/Decode 吞吐,
            同源无鉴权(与 /monitor 一致)。"""
            return HTMLResponse(_TEST_HTML_PATH.read_text(encoding="utf-8"))
