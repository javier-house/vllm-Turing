# SPDX-License-Identifier: Apache-2.0
"""firefly HC decode 标量 GEMV (P2a) 单测: env 归一化 + AST 结构 + 注入 dry-run。

本机无 torch/triton → 涉及 torch 的 vllm_firefly_hc 用 AST 静态验证 (custom_op 注册、
三投影名、M==1 gate、回退原 gemm、batch-invariant 拒绝), 真实张量行为在镜像 e2e 验。
model.py 注入 dry-run: 拿底座真实 model.py 模拟 append fhc_hook 两次 → 幂等 + 语法可编译。

底座路径: env VLLM_BASE_ROOT, 默认 /home/admin/code/vllm/vllm。找不到底座则跳过 dry-run。
"""

import ast
import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASE_ROOT = Path(os.environ.get("VLLM_BASE_ROOT", "/home/admin/code/vllm/vllm"))


def _load_standalone(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 1) envs_sm75: VLLM_FIREFLY_HC 归一化
# --------------------------------------------------------------------------- #
def test_firefly_hc_env_default_off(monkeypatch):
    envs_sm75 = _load_standalone("vllm/envs_sm75.py", "envs_sm75_hc1")
    monkeypatch.delenv("VLLM_FIREFLY_HC", raising=False)
    assert envs_sm75._firefly_hc() is False
    for on in ("1", "on", "true", "yes", "ON", "True"):
        monkeypatch.setenv("VLLM_FIREFLY_HC", on)
        assert envs_sm75._firefly_hc() is True
    for off in ("0", "off", "false", "no", ""):
        monkeypatch.setenv("VLLM_FIREFLY_HC", off)
        assert envs_sm75._firefly_hc() is False


def test_firefly_hc_registered_and_hashed(monkeypatch):
    """VLLM_FIREFLY_HC 改 decode 图内 HC 算子 → 在 EXTENSIONS 且不被 pop (参与编译 hash)。"""
    envs_sm75 = _load_standalone("vllm/envs_sm75.py", "envs_sm75_hc2")
    assert "VLLM_FIREFLY_HC" in envs_sm75.EXTENSIONS
    # 不在 INSTALL_IGNORED (改图, 须各用各编译产物)。
    assert "VLLM_FIREFLY_HC" not in envs_sm75.INSTALL_IGNORED
    fake_envs = types.ModuleType("vllm.envs")
    fake_envs.environment_variables = {"VLLM_FIREFLY_HC": lambda: False}
    fake_envs.compile_factors = lambda: {"VLLM_FIREFLY_HC": True}
    monkeypatch.setitem(sys.modules, "vllm.envs", fake_envs)
    envs_sm75.apply()
    assert "VLLM_FIREFLY_HC" in fake_envs.compile_factors()


# --------------------------------------------------------------------------- #
# 2) vllm_firefly_hc AST 结构验证 (无 torch, 只查结构)
# --------------------------------------------------------------------------- #
def _hc_tree() -> ast.Module:
    src = (REPO / "vllm/vllm_firefly_hc.py").read_text(encoding="utf-8")
    return ast.parse(src)


def _top_func(tree, name):
    return next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )


def test_hc_top_symbols():
    tree = _hc_tree()
    for fn in ("_enabled", "maybe_apply", "_install_on_module", "_install_after_init"):
        assert _top_func(tree, fn) is not None, fn
    # ENV 名常量 + 三投影名常量
    names = {
        t.id
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    assert "_HC_PROJ_NAMES" in names
    src = (REPO / "vllm/vllm_firefly_hc.py").read_text(encoding="utf-8")
    for proj in (
        "input_mix_weight_down_block_inject",
        "input_mix_weight_down",
        "input_mix_weight_up",
    ):
        assert proj in src


def test_hc_custom_op_registered_no_tl_dot():
    """custom_op 注册名 + register_fake + 标量 GEMV 不用 tl.dot (sm75 安全)。"""
    src = (REPO / "vllm/vllm_firefly_hc.py").read_text(encoding="utf-8")
    assert 'custom_op("firefly_hc::gemv"' in src
    assert "@hc_gemv.register_fake" in src
    # 标量归约用 tl.sum, 绝不用 tl.dot 调用 (sm75 无 int8/fp8 tensor core)。
    # 注: 检查调用形式 "tl.dot(" —— 源码注释里"不用 tl.dot"字样不算调用。
    assert "tl.sum(" in src
    assert "tl.dot(" not in src
    # 只在 w·x 上做 fp32 累加归约。
    assert ".to(tl.float32)" in src


def test_hc_dispatch_m1_gate_and_fallback():
    """_make_dispatch: M==1 (x.numel()==k) 命中 GEMV; bias/非 sm75/非 fp16/非 contig/
    numel!=k (prefill M>1) 全部回退原 gemm。"""
    tree = _hc_tree()
    mk = _top_func(tree, "_make_dispatch")
    assert mk is not None
    disp = [
        n for n in ast.walk(mk)
        if isinstance(n, ast.FunctionDef) and n.name == "_dispatch"
    ][0]
    ds = ast.unparse(disp)
    # 回退原 gemm 的四道闸。
    assert "bias is not None or not is_sm75" in ds
    assert "x.dtype != torch.float16" in ds
    assert "x.numel() != k" in ds
    # 命中才调 GEMV dispatch。
    assert "_hc_gemv_dispatch(" in ds


def test_hc_install_requires_unquant_and_rejects_batch_invariant():
    """只装 UnquantizedLinearMethod 层; batch-invariant 直接报错 (绕过 _gemm_impl)。"""
    tree = _hc_tree()
    inst = _top_func(tree, "_install_on_module")
    isrc = ast.unparse(inst)
    assert "UnquantizedLinearMethod" in isrc
    assert "type(method) is not UnquantizedLinearMethod" in isrc
    assert "VLLM_BATCH_INVARIANT" in isrc
    # 装法: 快照原 _gemm_impl + 换 _gemm_impl + 打 patched 标志 (幂等)。
    assert "_firefly_hc_orig_gemm" in isrc
    assert "method._gemm_impl = _make_dispatch" in isrc
    assert "_firefly_hc_patched" in isrc


def test_hc_maybe_apply_gated_idempotent():
    """env 关 → 直接 return 不改; 已 patch → 跳过; 包 __init__。"""
    tree = _hc_tree()
    ma = _top_func(tree, "maybe_apply")
    src = ast.unparse(ma)
    assert "if not _enabled():" in src
    assert "_firefly_hc_patched" in src
    assert "model_cls.__init__ = __init__" in src
    # 安装异常退回上游路径 (RuntimeError 除外, 那是配置冲突要冒泡)。
    assert "回退上游 cuBLAS" in src


# --------------------------------------------------------------------------- #
# 3) model.py 注入 dry-run: 模拟 overlay 的 fhc_hook append, 验证幂等 + 可编译
# --------------------------------------------------------------------------- #
_FHC_HOOK = (
    "\n# vllm-turing overlay: firefly HC decode 标量 GEMV (idempotent).\n"
    "import vllm.vllm_firefly_hc\n"
    "vllm.vllm_firefly_hc.maybe_apply(Qwen4ExpModel)\n"
)


@pytest.mark.skipif(
    not (BASE_ROOT / "models/qwen4_exp/nvidia/model.py").is_file(),
    reason=f"底座 checkout 不存在: {BASE_ROOT}",
)
def test_model_injection_dry_run(tmp_path):
    base = BASE_ROOT / "models/qwen4_exp/nvidia/model.py"
    target = tmp_path / "model.py"
    shutil.copy(base, target)
    text = target.read_text(encoding="utf-8")

    # overlay 的 marker 判幂等逻辑: 命中 marker 不再 append。模拟两次。
    for _ in range(2):
        if "vllm.vllm_firefly_hc.maybe_apply" not in text:
            text += _FHC_HOOK
    target.write_text(text, encoding="utf-8")

    # 恰好注入一次 (幂等)。
    assert text.count("vllm.vllm_firefly_hc.maybe_apply(Qwen4ExpModel)") == 1
    # 语法可编译 (hook 接在真实 model.py 之后)。
    compile(text, str(target), "exec")
    # Qwen4ExpModel 类在底座存在 → hook 引用的符号有效。
    assert "class Qwen4ExpModel" in text


@pytest.mark.skipif(
    not (BASE_ROOT / "models/qwen4_exp/nvidia/model.py").is_file(),
    reason=f"底座 checkout 不存在: {BASE_ROOT}",
)
def test_model_injection_coexists_with_hc_dequant(tmp_path):
    """hc_dequant(load_weights) + firefly_hc(__init__) 两条 hook 可同存于 model.py 末尾。"""
    base = BASE_ROOT / "models/qwen4_exp/nvidia/model.py"
    target = tmp_path / "model.py"
    shutil.copy(base, target)
    text = target.read_text(encoding="utf-8")
    deq_hook = (
        "\nimport vllm.vllm_hc_dequant\n"
        "vllm.vllm_hc_dequant.maybe_apply(Qwen4ExpModel)\n"
    )
    text += deq_hook + _FHC_HOOK
    target.write_text(text, encoding="utf-8")
    compile(text, str(target), "exec")
    assert "vllm.vllm_hc_dequant.maybe_apply" in text
    assert "vllm.vllm_firefly_hc.maybe_apply" in text
