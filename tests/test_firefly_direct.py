# SPDX-License-Identifier: Apache-2.0
"""VLLM_FIREFLY_DIRECT (firefly 族总开关) 单测: env 归一化判定表 +
compile_factors pop 行为 + firefly_int8.maybe_apply 幂等。

本机无 torch —— maybe_apply 的依赖 (torch / vllm.model_executor...) 用
fake sys.modules mock 掉, 只验 patch 接线与幂等, 真实张量行为在镜像 e2e 验。
envs_sm75 无重依赖, 按文件路径独立加载 (同 test_ple_host_gather 手法)。
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ON_VALUES = ("1", "on", "true", "yes", "auto", "AUTO", " 1 ")
OFF_VALUES = ("0", "off", "false", "no", "OFF")


def _load_envs_sm75(name: str):
    """按文件路径独立加载 envs_sm75 (无重依赖)。"""
    spec = importlib.util.spec_from_file_location(
        name, REPO / "vllm" / "envs_sm75.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DirectEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs_sm75 = _load_envs_sm75("envs_sm75_direct_test")

    def _clear(self, name: str):
        import os

        os.environ.pop(name, None)

    def test_direct_default_auto_on(self):
        """未设 = auto = 开 (合适权重自动启用)。"""
        self._clear("VLLM_FIREFLY_DIRECT")
        self.assertTrue(self.envs_sm75._firefly_direct())

    def test_direct_on_table(self):
        import os

        for v in ON_VALUES:
            os.environ["VLLM_FIREFLY_DIRECT"] = v
            self.assertTrue(
                self.envs_sm75._firefly_direct(), f"DIRECT={v!r} 应判开"
            )
        self._clear("VLLM_FIREFLY_DIRECT")

    def test_direct_off_table(self):
        import os

        for v in OFF_VALUES:
            os.environ["VLLM_FIREFLY_DIRECT"] = v
            self.assertFalse(
                self.envs_sm75._firefly_direct(), f"DIRECT={v!r} 应判关"
            )
        self._clear("VLLM_FIREFLY_DIRECT")

    def test_ar_default_off(self):
        self._clear("VLLM_FIREFLY_AR")
        self.assertFalse(self.envs_sm75._firefly_ar_mode())

    def test_ar_on_table(self):
        import os

        for v in ("1", "on", "true", "yes"):
            os.environ["VLLM_FIREFLY_AR"] = v
            self.assertTrue(self.envs_sm75._firefly_ar_mode(), f"AR={v!r} 应判开")
        self._clear("VLLM_FIREFLY_AR")

    def test_ar_legacy_values_off(self):
        # 旧 'auto'(跟随总开关)/'fp8' 取值: 总开关已删、getter 改纯开关,
        # 归一化器不再认, 缺省/旧值一律判关。
        import os

        for v in ("fp8", "auto", "AUTO"):
            os.environ["VLLM_FIREFLY_AR"] = v
            self.assertFalse(self.envs_sm75._firefly_ar_mode(), f"AR={v!r} 应判关")
        self._clear("VLLM_FIREFLY_AR")

    def test_ar_off_table(self):
        import os

        for v in OFF_VALUES:
            os.environ["VLLM_FIREFLY_AR"] = v
            self.assertFalse(self.envs_sm75._firefly_ar_mode(), f"AR={v!r} 应判关")
        self._clear("VLLM_FIREFLY_AR")

    def test_defer_default_on(self):
        self._clear("VLLM_FIREFLY_DEFER")
        self.assertTrue(self.envs_sm75._firefly_defer_recv())

    def test_defer_on_off_tables(self):
        import os

        for v in ("1", "on", "true", "yes", "auto"):
            os.environ["VLLM_FIREFLY_DEFER"] = v
            self.assertTrue(
                self.envs_sm75._firefly_defer_recv(), f"DEFER={v!r} 应判开"
            )
        for v in OFF_VALUES:
            os.environ["VLLM_FIREFLY_DEFER"] = v
            self.assertFalse(
                self.envs_sm75._firefly_defer_recv(), f"DEFER={v!r} 应判关"
            )
        self._clear("VLLM_FIREFLY_DEFER")

    def test_extensions_registration(self):
        ext = self.envs_sm75.EXTENSIONS
        self.assertIn("VLLM_FIREFLY_DIRECT", ext)
        self.assertNotIn("VLLM_FIREFLY", ext, "总开关已删, 不应再注册")

    def test_apply_pops_firefly_keys(self):
        """firefly 族 env 纯运行时 → apply 后 compile_factors 必须 pop;
        旧 VLLM_FIREFLY key 也要 pop (迁移期防 hash 漂移);
        VLLM_PLE_HOST_GATHER 不 pop (改变编译图, 行为回归)。"""
        import os

        envs_mod = self.envs_sm75
        fake_envs = types.ModuleType("vllm.envs")
        fake_envs.environment_variables = {"VLLM_FIREFLY_DIRECT": lambda: True}
        fake_envs.compile_factors = lambda: {
            "VLLM_FIREFLY": "1",  # 旧 key (若历史 env 仍设), 应 pop
            "VLLM_FIREFLY_DIRECT": True,
            "VLLM_FIREFLY_AR": True,
            "VLLM_FIREFLY_DEFER": True,
            "VLLM_PLE_HOST_GATHER": True,  # 改变编译图, 必须存活
            "VLLM_PLE_MEM_LAZY": True,  # 运行时, 应 pop
        }
        saved = sys.modules.get("vllm.envs")
        sys.modules["vllm.envs"] = fake_envs
        try:
            envs_mod.apply()
            factors = fake_envs.compile_factors()
        finally:
            if saved is not None:
                sys.modules["vllm.envs"] = saved
            else:
                sys.modules.pop("vllm.envs", None)
        for key in (
            "VLLM_FIREFLY",
            "VLLM_FIREFLY_DIRECT",
            "VLLM_FIREFLY_AR",
            "VLLM_FIREFLY_DEFER",
            "VLLM_PLE_MEM_LAZY",
        ):
            self.assertNotIn(key, factors, f"{key} 应被 pop 出 compile_factors")
        self.assertIn("VLLM_PLE_HOST_GATHER", factors)
        # getter 实际仍按 env 原值生效 (pop 只动 hash 因子, 不动行为)。
        os.environ["VLLM_FIREFLY_DIRECT"] = "0"
        self.assertFalse(envs_mod._firefly_direct())
        os.environ.pop("VLLM_FIREFLY_DIRECT", None)


def _install_fake_deps():
    """mock torch / vllm.model_executor 依赖链, 供 firefly_int8 无 torch 加载。

    返回 (fake_compressed_tensors_module, fake_config_cls)。
    """
    fake_torch = types.ModuleType("torch")

    class _FakeModule:
        def __init__(self, *a, **k):
            pass

    fake_torch.nn = types.SimpleNamespace(Module=_FakeModule)
    fake_torch.Tensor = _FakeModule
    fake_torch.dtype = "dtype"  # 类体注解 params_dtype: torch.dtype 需要
    fake_torch.int8 = "int8"
    fake_torch.float32 = "float32"

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_me = types.ModuleType("vllm.model_executor")
    fake_me.__path__ = []
    fake_layers = types.ModuleType("vllm.model_executor.layers")
    fake_layers.__path__ = []
    fake_quant = types.ModuleType("vllm.model_executor.layers.quantization")
    fake_quant.__path__ = []
    fake_ct = types.ModuleType(
        "vllm.model_executor.layers.quantization.compressed_tensors"
    )

    class CompressedTensorsScheme:
        pass

    fake_schemes = types.ModuleType(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes"
    )
    fake_schemes.CompressedTensorsScheme = CompressedTensorsScheme
    fake_ct.compressed_tensors = fake_ct  # 模块自引用, 供 from ... import 链
    fake_parameter = types.ModuleType("vllm.model_executor.parameter")
    fake_parameter.ModelWeightParameter = type("ModelWeightParameter", (), {})
    fake_parameter.ChannelQuantScaleParameter = type(
        "ChannelQuantScaleParameter", (), {}
    )

    return {
        "torch": fake_torch,
        "vllm": fake_vllm,
        "vllm.model_executor": fake_me,
        "vllm.model_executor.layers": fake_layers,
        "vllm.model_executor.layers.quantization": fake_quant,
        "vllm.model_executor.layers.quantization.compressed_tensors": fake_ct,
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes": fake_schemes,
        "vllm.model_executor.parameter": fake_parameter,
    }


def _load_firefly_int8(name: str):
    """fake 依赖就位后按路径加载 firefly_int8 (不触发真 vllm/torch)。"""
    spec = importlib.util.spec_from_file_location(
        name,
        REPO
        / "vllm"
        / "model_executor"
        / "layers"
        / "quantization"
        / "utils"
        / "firefly_int8.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MaybeApplyTest(unittest.TestCase):
    """firefly_int8.maybe_apply: 判据切 VLLM_FIREFLY_DIRECT + 幂等。

    fake 依赖在 setUp 安装、tearDown 恢复 (maybe_apply 内部是**延迟 import**,
    调用时依赖必须仍在 sys.modules)。每用例全新 CompressedTensorsConfig,
    避免 patch 状态串扰。
    """

    def setUp(self):
        self._direct_value = None
        self._saved: dict[str, object | None] = {}
        self.cfg_mod = None
        self.cfg_cls = None
        self.ff = None

    def tearDown(self):
        for k, v in self._saved.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)

    def _install(self, direct_value: bool):
        fakes = _install_fake_deps()
        cfg_mod = types.ModuleType("_fake_compressed_tensors_cfg")

        class CompressedTensorsConfig:
            def get_scheme(self, layer, layer_name=None):
                return "upstream"

            def get_scheme_dict(self, layer, layer_name=None):
                return None

            def _check_scheme_supported(self, cap):
                pass

        cfg_mod.CompressedTensorsConfig = CompressedTensorsConfig
        # from ...compressed_tensors import compressed_tensors as _ct_mod:
        # 包模块上挂同名属性 (import 先查属性再查子模块), 自引用即通。
        cfg_mod.compressed_tensors = cfg_mod
        self.cfg_mod = cfg_mod
        self.cfg_cls = CompressedTensorsConfig
        fakes["vllm.model_executor.layers.quantization.compressed_tensors"] = cfg_mod
        fake_envs_sm75 = types.ModuleType("vllm.envs_sm75")
        fake_envs_sm75._firefly_direct = (lambda: direct_value)
        fakes["vllm.envs_sm75"] = fake_envs_sm75

        for k, v in fakes.items():
            self._saved[k] = sys.modules.get(k)
            sys.modules[k] = v
        self.ff = _load_firefly_int8("firefly_int8_direct_test")

    def test_apply_patches_when_direct_on(self):
        self._install(True)
        self.ff.maybe_apply()
        self.assertTrue(getattr(self.cfg_cls.get_scheme, "_sm75_int8_patched", False))
        # 幂等: 二次 import/调用不重复 patch (flag 在 wrapper 上)。
        first_wrapper = self.cfg_cls.get_scheme
        self.ff.maybe_apply()
        self.assertIs(self.cfg_cls.get_scheme, first_wrapper)

    def test_apply_noop_when_direct_off(self):
        self._install(False)
        self.ff.maybe_apply()
        self.assertFalse(
            getattr(self.cfg_cls.get_scheme, "_sm75_int8_patched", False)
        )


if __name__ == "__main__":
    unittest.main()
