# SPDX-License-Identifier: Apache-2.0
"""PLE host-gather (P0) 单测: env 归一化 + overlay 注入 dry-run + AST 结构验证。

本机无 torch → 涉及 torch 的 vllm_ple_mmap 用 AST 静态验证 (方法/分支存在),
真实张量行为在镜像 e2e 验。overlay 注入用底座 checkout 的真实 model_state.py
做 dry-run: anchor 唯一命中 + 结果语法可编译 + probe 幂等 (apply 两次不变)。

底座路径: 环境变量 VLLM_BASE_ROOT, 默认 /home/admin/code/vllm (上游 v0.29.0
checkout)。找不到底座 (如 CI 镜像内构建后跑) 则跳过 dry-run。
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
BASE_ROOT = Path(
    os.environ.get("VLLM_BASE_ROOT", "/home/admin/code/vllm/vllm")
)


def _load_standalone(rel_path: str, name: str):
    """按文件路径加载无重依赖的模块 (envs_sm75 / install_sm75_overlay)。"""
    spec = importlib.util.spec_from_file_location(name, REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 1) envs_sm75: VLLM_PLE_HOST_GATHER 归一化
# --------------------------------------------------------------------------- #
def test_ple_host_gather_env_default_on(monkeypatch):
    envs_sm75 = _load_standalone("vllm/envs_sm75.py", "envs_sm75_t1")
    monkeypatch.delenv("VLLM_PLE_HOST_GATHER", raising=False)
    assert envs_sm75._ple_host_gather() is True
    monkeypatch.setenv("VLLM_PLE_HOST_GATHER", "1")
    assert envs_sm75._ple_host_gather() is True
    for off in ("0", "off", "false", "no", "OFF"):
        monkeypatch.setenv("VLLM_PLE_HOST_GATHER", off)
        assert envs_sm75._ple_host_gather() is False


def test_ple_host_gather_registered_not_popped(monkeypatch):
    """VLLM_PLE_HOST_GATHER 改变 forward 编译图 → 必须在 EXTENSIONS 且不被 pop。"""
    envs_sm75 = _load_standalone("vllm/envs_sm75.py", "envs_sm75_t2")
    assert "VLLM_PLE_HOST_GATHER" in envs_sm75.EXTENSIONS
    # 模拟 apply() 后 compile_factors: 该 key 必须存活 (参与编译 hash)。
    fake_envs = types.ModuleType("vllm.envs")
    fake_envs.environment_variables = {"VLLM_FIREFLY": lambda: "0"}
    fake_envs.compile_factors = lambda: {
        "VLLM_FIREFLY": "0",
        "VLLM_PLE_HOST_GATHER": True,
        "VLLM_PLE_MEM_LAZY": True,  # lazy 是运行时, 应被 pop
        "VLLM_PLE_MEM_FILL_WORKERS": 16,  # 同上
    }
    monkeypatch.setitem(sys.modules, "vllm.envs", fake_envs)
    envs_sm75.apply()
    factors = fake_envs.compile_factors()
    assert "VLLM_PLE_HOST_GATHER" in factors  # 不 pop: 开关各用各编译产物
    assert "VLLM_PLE_MEM_LAZY" not in factors
    assert "VLLM_PLE_MEM_FILL_WORKERS" not in factors


# --------------------------------------------------------------------------- #
# 2) vllm_ple_mmap AST 结构验证 (无 torch, 只查结构)
# --------------------------------------------------------------------------- #
def _ple_mmap_tree() -> ast.Module:
    src = (REPO / "vllm/vllm_ple_mmap.py").read_text(encoding="utf-8")
    return ast.parse(src)


def _top_func(tree, name):
    return next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )


def test_mmap_ast_has_host_gather_symbols():
    tree = _ple_mmap_tree()
    assert _top_func(tree, "host_gather_enabled") is not None
    assert _top_func(tree, "attach_to_model_state") is not None
    assert _top_func(tree, "_ple_out_dtype") is not None
    # ENV_HOST_GATHER 常量存在
    names = {
        t.id
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    assert "ENV_HOST_GATHER" in names


def test_mmap_ast_maybe_apply_wires_host_gather():
    """maybe_apply 内: host_gather 定义 + 挂到 cls; 无 prefetch 标志时行为回退。"""
    tree = _ple_mmap_tree()
    maybe = _top_func(tree, "maybe_apply")
    assert maybe is not None
    src = ast.unparse(maybe)
    assert "cls.host_gather = host_gather" in src
    assert "self.prefetch_from_model_state" in src  # __init__ 置标志
    # load_weights: host-gather 模式分配 buffer 且不调 _ensure_splitting_op;
    # 保险丝 (标志为假) 才注入 splitting op。
    assert "_ple_hg_buffer = torch.zeros" in src
    lw_src = [
        ast.unparse(n)
        for n in ast.walk(maybe)
        if isinstance(n, ast.FunctionDef) and n.name == "load_weights"
    ][0]
    # splitting op 注入必须在 else (保险丝) 分支, 不在 host-gather 分支。
    assert "if self.prefetch_from_model_state:" in lw_src
    assert "else:" in lw_src
    # forward: host-gather 分支直接 return buffer view, 不调 custom op。
    fwd_src = [
        ast.unparse(n)
        for n in ast.walk(maybe)
        if isinstance(n, ast.FunctionDef) and n.name == "forward"
    ][0]
    i = fwd_src.index("if self.prefetch_from_model_state:")
    j = fwd_src.index("getattr(torch.ops.vllm, _OP_NAME)")
    assert i < j  # 早退分支在 custom op 调用之前
    assert "return buf[" in fwd_src


def test_mmap_ast_host_gather_semantics():
    """host_gather: buffer 缺失 no-op; 越界裁剪; copy_ 原地写 (地址不变)。"""
    tree = _ple_mmap_tree()
    maybe = _top_func(tree, "maybe_apply")
    hg = [
        ast.unparse(n)
        for n in ast.walk(maybe)
        if isinstance(n, ast.FunctionDef) and n.name == "host_gather"
    ][0]
    assert "if buf is None:" in hg
    assert "min(input_ids.shape[0], buf.shape[0])" in hg
    assert "buf[:num_tokens].copy_(flat" in hg
    # 只走 compute_ngram_ids + ngram_embedding (复用现有 gather 通路, 含 lazy 水位)
    assert "self.compute_ngram_ids" in hg
    assert "self.ngram_embedding(ngram_ids)" in hg


def test_mmap_ast_attach_to_model_state_semantics():
    tree = _ple_mmap_tree()
    fn = _top_func(tree, "attach_to_model_state")
    src = ast.unparse(fn)
    assert "if not host_gather_enabled():\n        return" in src  # env 关 → no-op
    assert "named_modules()" in src
    assert "prefetch_from_model_state" in src
    assert "state._ple_hg_layer" in src
    assert "state._ple_hg_dummy_ids" in src


# --------------------------------------------------------------------------- #
# 3) overlay 注入 dry-run: 拿底座真实 model_state.py 跑 apply 逻辑
# --------------------------------------------------------------------------- #
def _injections_for(relative: str) -> list:
    inst = _load_standalone("docker/install_sm75_overlay.py", "overlay_t")
    pairs = []
    for rel, ps in inst.INJECTIONS:
        if rel == relative:
            pairs.extend(ps)
    return pairs


@pytest.mark.skipif(
    not (BASE_ROOT / "models/qwen4_exp/nvidia/model_state.py").is_file(),
    reason=f"底座 checkout 不存在: {BASE_ROOT}",
)
def test_model_state_injection_dry_run(tmp_path):
    base = BASE_ROOT / "models/qwen4_exp/nvidia/model_state.py"
    target = tmp_path / "models/qwen4_exp/nvidia/model_state.py"
    target.parent.mkdir(parents=True)
    shutil.copy(base, target)

    inst = _load_standalone("docker/install_sm75_overlay.py", "overlay_dry")
    # 临时把 INJECTIONS 裁成只剩 model_state 条目 (其余目标文件不在 tmp_path)。
    saved = inst.INJECTIONS
    inst.INJECTIONS = [
        (rel, ps) for rel, ps in saved if "model_state" in rel
    ]
    try:
        inst.apply_injections(tmp_path)
        # 第二次幂等 (probe 生效不重复注入)。
        inst.apply_injections(tmp_path)
    finally:
        inst.INJECTIONS = saved

    text = target.read_text(encoding="utf-8")
    # 语法可编译。
    compile(text, str(target), "exec")
    # 三处注入全部到位且各恰好一次 (幂等)。
    assert text.count("_ple_mmap.attach_to_model_state(self, model)") == 1
    assert text.count("host_gather(input_batch.input_ids") == 1
    assert text.count("self._ple_hg_dummy_ids[:num_tokens]") == 1
    # 上游原有逻辑保留 (append-only): PP gate 注入仍在 (来自既有 overlay 条目?
    # 不 —— 本 dry-run 只跑新条目, 故查上游原句未被破坏)。
    assert "def prepare_inputs" in text
    assert "def prepare_dummy_inputs" in text
    assert "query_start_loc.copy_(input_batch.query_start_loc" in text
    # AST 级结构完好: 两个方法体里 host_gather 在 return 之前。
    tree = ast.parse(text)
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpModelState"
    )
    for fname, needle in (
        ("prepare_inputs", "host_gather(input_batch.input_ids"),
        ("prepare_dummy_inputs", "_ple_hg_dummy_ids[:num_tokens]"),
    ):
        m = next(
            n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == fname
        )
        ms = ast.unparse(m)
        assert needle in ms
        # 末尾 return 之前 (方法开头有 "if not uses_ngram: return model_inputs"
        # 早退守卫, 那次 return 在 host_gather 之前是正常的非 ngram 路径) → 用 rindex。
        assert ms.index(needle) < ms.rindex("return model_inputs")


@pytest.mark.skipif(
    not (BASE_ROOT / "models/qwen4_exp/nvidia/model_state.py").is_file(),
    reason=f"底座 checkout 不存在: {BASE_ROOT}",
)
def test_existing_pp_injection_still_compatible(tmp_path):
    """既有 PP>1 gate 注入 + 新 host-gather 注入共存: 顺序应用两块 INJECTIONS。"""
    base = BASE_ROOT / "models/qwen4_exp/nvidia/model_state.py"
    target = tmp_path / "models/qwen4_exp/nvidia/model_state.py"
    target.parent.mkdir(parents=True)
    shutil.copy(base, target)
    inst = _load_standalone("docker/install_sm75_overlay.py", "overlay_co")
    saved = inst.INJECTIONS
    inst.INJECTIONS = [
        (rel, ps)
        for rel, ps in saved
        if rel == "models/qwen4_exp/nvidia/model_state.py"
    ]
    try:
        inst.apply_injections(tmp_path)
    finally:
        inst.INJECTIONS = saved
    text = target.read_text(encoding="utf-8")
    compile(text, str(target), "exec")
    # 两块都在: PP gate (pipeline rank 0 措辞) + host-gather。
    assert "N-gram PLE embedding requires the PLE layers to sit on" in text
    assert "_ple_mmap.attach_to_model_state(self, model)" in text
