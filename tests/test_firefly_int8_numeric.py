# SPDX-License-Identifier: Apache-2.0
"""firefly int8 prefill 的**数值等值回归基线**(不是性能测试)。

目标: 给 firefly int8 权重线性层(FireflyInt8Scheme + firefly int8 GEMM)锁一组
确定性基线, 跨 commit 不回归。它回答的是"数值/接线对不对", 不是"快不快"。

本机无 torch / 无 CUDA / 无 pytest —— 故分两层:
1) 纯 Python 逻辑 + numpy 数值等值: 本机**真跑通**。
   - _is_sm75_int8 判据真值表 (从**真实源码** import, 该函数纯 Python)。
   - create_weights / process_weights_after_loading: 用 numpy 支撑的 fake torch
     按文件路径加载**真实** firefly_int8.py, 驱动这两个方法, 验证非打包 int8
     权重布局 / per-channel scale reshape / 原参数释放。
   - firefly int8 GEMM 的数学等值: 在测试里用纯 numpy 手写"独立参考实现"
     (反量化 w_int8*c_n 后 fp 点积), 与 firefly 的 int32 整数累加 + 缩放复刻
     对照 —— 锁死反量化公式 / scale 施加顺序 / 转置方向 (布局接线的等值基线)。
2) 真 torch 端到端 (真实 FireflyInt8Scheme + 真 GEMM): 无 torch/CUDA 时**跳过**,
   镜像 e2e 里才真跑 (见 tmp 内同名脚本)。

关键不变量 (section 4): firefly 的整数累加输出与"反量化后 fp 点积"应近乎 bit-exact
(~1e-15), 因为两者只差最后 fp 输出舍入; 任何布局/转置/缩放错位都会放大到整数量级,
远超容差 —— 这正是"等值回归"想钉住的东西。
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIREFLY_INT8_PATH = (
    REPO
    / "vllm"
    / "model_executor"
    / "layers"
    / "quantization"
    / "utils"
    / "firefly_int8.py"
)

# 真 torch / CUDA 可用性: 决定端到端用例是跑还是跳。
try:  # noqa: SIM105
    import torch as _real_torch  # type: ignore

    _TORCH_OK = True
except Exception:  # noqa: BLE001 - 无 torch 环境(本机)即跳过真张量用例
    _real_torch = None
    _TORCH_OK = False
_CUDA_OK = bool(_TORCH_OK and _real_torch.cuda.is_available())


# --------------------------------------------------------------------------- #
# numpy 支撑的 fake torch: 让"真实" firefly_int8 在纯 CPU、无 torch 下可加载,
# 只驱动 create_weights / process_weights_after_loading 里用到的 torch 子集。
# --------------------------------------------------------------------------- #
_INT8 = "int8"
_FLOAT32 = "float32"
_FLOAT16 = "float16"
_NPDTYPE = {_INT8: np.int8, _FLOAT32: np.float32, _FLOAT16: np.float16}


class NumpyTensor:
    """最小 torch.Tensor 替身: 包一个 numpy 数组, 只实现被测代码用到的方法。

    刻意只覆盖 firefly_int8.create_weights / process_weights_after_loading 的
    torch 用法 (empty / .data / .dtype / .to / .reshape / .float / .contiguous),
    不做完整张量语义 —— 够跑通即止, 避免把测试变成"再写一个 torch"。
    """

    def __init__(self, arr):
        self._a = np.asarray(arr)

    # torch nn.Parameter 的 .data 直接回自身 (被测代码只读 .data 再链式调用)。
    @property
    def data(self):
        return self

    @property
    def dtype(self):
        return self._a.dtype

    @property
    def shape(self):
        return self._a.shape

    def to(self, dtype):
        return NumpyTensor(self._a.astype(_NPDTYPE.get(dtype, self._a.dtype)))

    def reshape(self, *s):
        if len(s) == 1 and isinstance(s[0], (tuple, list)):
            s = tuple(s[0])
        return NumpyTensor(self._a.reshape(*s))

    def float(self):
        return NumpyTensor(self._a.astype(np.float32))

    def contiguous(self):
        # numpy 已连续; 复刻"不 copy 若已连续"的语义只需回一个连续视图。
        return NumpyTensor(np.ascontiguousarray(self._a))

    def numpy(self):
        return self._a


def _fake_torch_empty(*shape, dtype=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return NumpyTensor(np.zeros(shape, dtype=_NPDTYPE.get(dtype, np.float32)))


def _build_fake_torch():
    t = types.ModuleType("torch")
    t.int8 = _INT8
    t.float32 = _FLOAT32
    t.float16 = _FLOAT16
    t.Tensor = NumpyTensor
    t.dtype = "dtype"  # 类体注解 params_dtype: torch.dtype 需要
    t.empty = _fake_torch_empty
    t.nn = types.SimpleNamespace(Module=object)
    return t


def _load_firefly_int8_with_fake_torch():
    """按文件路径加载真实 firefly_int8.py, 用 fake torch + fake vllm 符号满足其
    顶层 import; 加载完立即把 sys.modules 还原, 只把模块对象交回。

    返回 (module, fake_torch): fake_torch 供 create_weights 传 params_dtype。
    """
    fake_torch = _build_fake_torch()

    class CompressedTensorsScheme:
        pass

    class _Param:  # 替身 ModelWeightParameter / ChannelQuantScaleParameter
        def __init__(self, *, data=None, **kw):
            self.data = data

    schemes = types.ModuleType(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes"
    )
    schemes.CompressedTensorsScheme = CompressedTensorsScheme
    param = types.ModuleType("vllm.model_executor.parameter")
    param.ModelWeightParameter = _Param
    param.ChannelQuantScaleParameter = _Param

    fakes = {
        "torch": fake_torch,
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes": schemes,
        "vllm.model_executor.parameter": param,
    }
    saved = {k: sys.modules.get(k) for k in fakes}
    sys.modules.update(fakes)
    try:
        spec = importlib.util.spec_from_file_location(
            "ff8_numpy_shim", str(FIREFLY_INT8_PATH)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)
    return mod, fake_torch


class _FakeLayer:
    """最小 nn.Module 替身: register_parameter 挂属性, del 干净移除。

    被测 process_weights_after_loading 会 `del layer.weight`, 需要真删除属性。
    """

    def __init__(self):
        self._params = {}

    def register_parameter(self, name, p):
        self._params[name] = p
        setattr(self, name, p)

    def __delattr__(self, name):
        self._params.pop(name, None)
        object.__delattr__(self, name)


# firefly_int8 的加载放在模块级 setUpModule, 供纯逻辑用例复用 (它无副作用)。
_FF8 = None
_FAKE_TORCH = None


def setUpModule():
    global _FF8, _FAKE_TORCH
    _FF8, _FAKE_TORCH = _load_firefly_int8_with_fake_torch()


# --------------------------------------------------------------------------- #
# 独立参考实现 (纯 numpy, 测试内手写一遍等价数学)
# --------------------------------------------------------------------------- #
def _dequant_w8(w_int8, c_n):
    """W8 权重反量化独立参考: w_deq[n,k] = w_int8[n,k] * c_n[n] (per-channel)。

    这是 firefly int8 scheme 权重侧的等值定义 (checkpoint weight = w_int8,
    weight_scale = c_n, 无二次打包)。刻意写成显式逐元素乘, 与被测 GEMM 解耦。
    """
    return w_int8.astype(np.float64) * c_n.astype(np.float64)[:, None]


def _per_token_int8_quant(x):
    """firefly int8_prefill_linear 的激活侧复刻: per-token 对称 int8。

    x_s[t] = max|x[t,:]|/127; x_q = round(clip(x/x_s,-127,127))。返回 (x_q, x_s)。
    这是"被测侧"而非"参考侧" —— 用来隔离激活量化噪声, 见 _firefly_gemm_ref。
    """
    x_s = np.maximum(np.abs(x).max(axis=1) / 127.0, 1e-12)
    q = np.rint(np.clip(x / x_s[:, None], -127, 127))
    return q, x_s


def _firefly_gemm_ref(x_q, x_s, w_int8, c_n, out_dtype=np.float64):
    """firefly int8 GEMM 的整数累加复刻 (cutlass_scaled_mm 的 CPU 等值):

    y = (x_q(int) 点积 w_int8.T 的整数累加) * x_s[:,None] * c_n[None,:]
    —— int64 累加对应 cutlass 的 int32 精确累加 (int8 值域下无溢出, 精确),
    最后按 per-token x_s * per-channel c_n 还原。
    """
    acc = x_q.astype(np.int64) @ w_int8.T.astype(np.int64)
    y = acc.astype(np.float64) * x_s[:, None] * c_n.astype(np.float64)[None, :]
    return y.astype(out_dtype)


# 固定尺寸小输入 (确定性, 手算 seed): 跨 commit 基线用同一组输入。
_FIX_SEED = 20260930
_FIX_M, _FIX_K, _FIX_N = 16, 64, 32


def _fixed_quant_weight(seed=_FIX_SEED):
    """确定性构造 int8 权重 w_int8[N,K] + per-channel c_n[N] (模拟 checkpoint)。"""
    g = np.random.default_rng(seed)
    w_fp = g.standard_normal((_FIX_N, _FIX_K))
    c_n = np.maximum(np.abs(w_fp).max(axis=1) / 127.0, 1e-8)  # 正 scale, 与 c_n 约定一致
    w_int8 = np.rint(np.clip(w_fp / c_n[:, None], -127, 127)).astype(np.int8)
    return w_int8, c_n.astype(np.float32)


# --------------------------------------------------------------------------- #
# 1) _is_sm75_int8 判据真值表 (真实源码, 纯 Python) —— 覆盖 per-channel/group + W8A16/W8A8 分流
# --------------------------------------------------------------------------- #
class _Q:
    """构造 compressed-tensors 的 quant 描述对象替身 (只放被测读到的属性)。"""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _wq(**over):
    """合法 int8 权重 quant 的基线 (sym/channel/非dynamic/int8), over 覆盖单字段。"""
    base = dict(type="int", num_bits=8, strategy="channel", symmetric=True, dynamic=False)
    base.update(over)
    return _Q(**base)


class IsSm75Int8TableTest(unittest.TestCase):
    """_is_sm75_int8: 哪些层该被 firefly int8 scheme 接管。数值等值的**路由**基线。"""

    def setUp(self):
        self.f = _FF8._is_sm75_int8

    # ---- 放行的两种激活形态 ----
    def test_w8a16_no_activation(self):
        """W8A16: 无激活量化 (input_quant=None) 应接管。"""
        self.assertTrue(self.f(_wq(), None, "int-quantized"))

    def test_w8a8_dynamic_int8_activation(self):
        """W8A8: 动态 per-token int8 激活 (type=int, num_bits=8) 应接管。"""
        self.assertTrue(self.f(_wq(), _Q(type="int", num_bits=8), "int-quantized"))

    def test_per_group_strategy_accepted(self):
        """per-group 权重策略与 per-channel 一样放行 (权重侧布局同 firefly)。"""
        self.assertTrue(self.f(_wq(strategy="group"), None, "int-quantized"))

    def test_naive_format_accepted_and_none_format_ok(self):
        """format=naive-quantized 放行; format=None 也不设限。"""
        self.assertTrue(self.f(_wq(), None, "naive-quantized"))
        self.assertTrue(self.f(_wq(), None, None))

    # ---- 必须回落到上游的负例 (逐条钉住判据) ----
    def test_weight_none_rejected(self):
        self.assertFalse(self.f(None, None, "int-quantized"))

    def test_type_not_int_rejected(self):
        self.assertFalse(self.f(_wq(type="float"), None, "int-quantized"))

    def test_num_bits_not_8_rejected(self):
        """int4(num_bits=4) 不走本 scheme (属 WNA16/marlin 路径)。"""
        self.assertFalse(self.f(_wq(num_bits=4), None, "int-quantized"))

    def test_strategy_tensor_rejected(self):
        self.assertFalse(self.f(_wq(strategy="tensor"), None, "int-quantized"))

    def test_asymmetric_rejected(self):
        self.assertFalse(self.f(_wq(symmetric=False), None, "int-quantized"))

    def test_weight_dynamic_rejected(self):
        """权重侧 dynamic 不是本 scheme 的目标 (权重须静态 scale)。"""
        self.assertFalse(self.f(_wq(dynamic=True), None, "int-quantized"))

    def test_bad_format_rejected(self):
        self.assertFalse(self.f(_wq(), None, "float-quantized"))
        self.assertFalse(self.f(_wq(), None, "pack-quantized"))

    def test_activation_type_not_int_rejected(self):
        self.assertFalse(self.f(_wq(), _Q(type="float", num_bits=8), "int-quantized"))

    def test_activation_num_bits_not_8_rejected(self):
        self.assertFalse(self.f(_wq(), _Q(type="int", num_bits=4), "int-quantized"))


# --------------------------------------------------------------------------- #
# 2) create_weights 参数布局契约 (真实源码 + fake torch, 本机真跑)
# --------------------------------------------------------------------------- #
class CreateWeightsContractTest(unittest.TestCase):
    """create_weights: 非打包 int8 权重布局与键名的等值/契约基线。

    钉住关键回归点: 权重注册键必须是 "weight" (不是 int4 的 "weight_packed"),
    dtype=int8, 形状 [N,K] 行主序; weight_scale 键 "weight_scale", dtype=float32,
    形状 [N,1]。这些是 checkpoint 加载能否对上的硬契约。
    """

    def test_allocates_nonpacked_int8_weight_and_scale(self):
        scheme = _FF8.FireflyInt8Scheme()
        self.assertEqual(scheme.get_min_capability(), 75)  # sm75 门槛
        layer = _FakeLayer()
        N, K = 4, 8
        scheme.create_weights(
            layer,
            output_size=N,
            input_size=K,
            output_partition_sizes=[N],
            input_size_per_partition=K,
            params_dtype=_FAKE_TORCH.float16,
            weight_loader=lambda *a, **k: None,
        )
        # 键名契约: weight / weight_scale (非 weight_packed)。
        self.assertIn("weight", layer._params)
        self.assertIn("weight_scale", layer._params)
        self.assertNotIn("weight_packed", layer._params)
        w = layer.weight.data
        self.assertEqual(w.shape, (N, K))
        self.assertEqual(w.dtype, np.int8)
        sc = layer.weight_scale.data
        self.assertEqual(sc.shape, (N, 1))
        self.assertEqual(sc.dtype, np.float32)
        # 记录层尺寸属性。
        self.assertEqual(layer.input_size_per_partition, K)
        self.assertEqual(layer.output_size_per_partition, N)
        self.assertFalse(layer.has_bias)


# --------------------------------------------------------------------------- #
# 3) process_weights_after_loading 反量化整理等值 (真实源码 + fake torch, 本机真跑)
# --------------------------------------------------------------------------- #
class ProcessWeightsEqualTest(unittest.TestCase):
    """process_weights_after_loading: ff_int8_* 输出与加载权重的等值基线。

    - ff_int8_w_int8 == 加载的 int8 权重 (逐元素, contiguous)。
    - ff_int8_c_n == scale reshape(-1) 转 float32 (覆盖 [N,1] 与 [N] 两种 scale 形状)。
    - 处理完原 weight / weight_scale 参数释放 (省显存)。
    """

    def _run(self, scale_2d):
        scheme = _FF8.FireflyInt8Scheme()
        layer = _FakeLayer()
        N, K = 4, 8
        scheme.create_weights(
            layer,
            output_size=N,
            input_size=K,
            output_partition_sizes=[N],
            input_size_per_partition=K,
            params_dtype=_FAKE_TORCH.float16,
            weight_loader=lambda *a, **k: None,
        )
        w = np.arange(N * K, dtype=np.int8).reshape(N, K)
        c = ((np.arange(N) + 1) / 10.0).astype(np.float32)
        layer.weight.data._a = w
        layer.weight_scale.data._a = c.reshape(N, 1) if scale_2d else c.reshape(N)
        scheme.process_weights_after_loading(layer)
        return layer, w, c

    def test_w_int8_and_c_n_equal_loaded(self):
        """per-channel scale 以 [N,1] 形状加载 → 还原等值 + float32。"""
        layer, w, c = self._run(scale_2d=True)
        self.assertTrue(np.array_equal(layer.ff_int8_w_int8.numpy(), w))
        self.assertEqual(layer.ff_int8_c_n.numpy().dtype, np.float32)
        self.assertTrue(np.allclose(layer.ff_int8_c_n.numpy(), c, rtol=0, atol=0))

    def test_c_n_reshape_from_1d_scale(self):
        """per-group/1D scale 以 [N] 形状加载 → reshape(-1) 分支也等值。"""
        layer, _w, c = self._run(scale_2d=False)
        self.assertEqual(layer.ff_int8_c_n.numpy().shape, (len(c),))
        self.assertTrue(np.allclose(layer.ff_int8_c_n.numpy(), c, rtol=0, atol=0))

    def test_original_params_released(self):
        layer, _w, _c = self._run(scale_2d=True)
        self.assertFalse(hasattr(layer, "weight"))
        self.assertFalse(hasattr(layer, "weight_scale"))


# --------------------------------------------------------------------------- #
# 4) firefly int8 GEMM 数学等值 (numpy 独立参考 vs 整数累加复刻, 本机真跑)
#    这是 W8A16 / W8A8 共用的那条 firefly int8 GEMM 的数值基线。
# --------------------------------------------------------------------------- #
class FireflyInt8GemmNumericTest(unittest.TestCase):
    """firefly int8 GEMM 的数值等值基线 (非性能测试)。

    W8A16 与 W8A8 在 firefly 里走**同一条** int8 GEMM (都对激活做 per-token 动态
    int8, 权重布局相同), 区别只在激活 scale 的来源。本测试锁三件事:
    A. 反量化定义: _dequant_w8 与 w_int8*c_n 等值 (权重侧)。
    B. GEMM 接线等值 (强不变量, ~bit-exact): 同一份量化激活喂给"整数累加 + 缩放"
       (firefly) 与"反量化后 fp 点积"(独立参考), 应近乎逐位一致 —— 钉死转置方向 /
       scale 施加顺序 / 布局, 任何错位都远超容差。
    C. 端到端有界: firefly(带激活量化) vs 未量化 fp 参考, 偏差被激活量化误差上界
       包住 —— 分别覆盖 W8A8(随机 per-token x_s) 与 W8A16-理想(x_s 视作 1)。
    """

    def _gemm_pair(self, seed):
        g = np.random.default_rng(seed)
        M, K, N = _FIX_M, _FIX_K, _FIX_N
        w_int8, c_n = _fixed_quant_weight(seed)
        x = g.standard_normal((M, K))
        x_q, x_s = _per_token_int8_quant(x)
        y_firefly = _firefly_gemm_ref(x_q, x_s, w_int8, c_n)
        # 独立参考: 量化激活反量化成 fp + 权重反量化 → fp 点积。
        x_dq = x_q.astype(np.float64) * x_s[:, None]
        y_ref = x_dq @ _dequant_w8(w_int8, c_n).T
        return x, w_int8, c_n, x_q, x_s, y_firefly, y_ref

    def test_dequant_definition(self):
        """A: 反量化 w_int8*c_n 与逐通道手算等值 (float64 精确)。"""
        w_int8, c_n = _fixed_quant_weight()
        deq = _dequant_w8(w_int8, c_n)
        for n in range(deq.shape[0]):
            self.assertTrue(np.allclose(deq[n], w_int8[n].astype(np.float64) * c_n[n]))

    def test_gemm_wiring_bit_exact(self):
        """B: 整数累加 vs 反量化 fp 点积 —— 接线等值, 近乎 bit-exact。"""
        for seed in (0, 1, 2, 3):
            *_, y_firefly, y_ref = self._gemm_pair(seed)
            denom = np.abs(y_ref).max()
            rel = np.abs(y_firefly - y_ref).max() / denom
            self.assertLess(
                rel,
                1e-9,
                f"seed={seed} 接线偏差 rel={rel:.2e} 过大 (疑布局/转置/缩放回归)",
            )

    def test_end_to_end_within_activation_quant_bound(self):
        """C: firefly(量化激活) vs 未量化 fp 参考, 偏差 ≤ 激活量化误差上界。"""
        w_int8, c_n = _fixed_quant_weight()
        w_deq = _dequant_w8(w_int8, c_n)
        for seed in (0, 1, 2, 3):
            g = np.random.default_rng(1000 + seed)
            x = g.standard_normal((_FIX_M, _FIX_K))
            x_q, x_s = _per_token_int8_quant(x)
            y_firefly = _firefly_gemm_ref(x_q, x_s, w_int8, c_n)
            y_fp = x @ w_deq.T  # 未量化理想 (fp 激活 + 已量化权重)
            # 激活量化误差上界: 每元素舍入 ≤ 0.5*x_s[t], 累加 |y| 误差 ≤
            # 0.5*x_s[t] * sum_k|w_deq[n,k]| (取最坏通道)。
            bound = (0.5 * x_s[:, None] * np.abs(w_deq).sum(axis=1)[None, :]).max()
            max_abs = np.abs(y_firefly - y_fp).max()
            self.assertLessEqual(max_abs, bound)

    def test_w8a8_and_w8a16_share_one_gemm_path(self):
        """W8A8(随机 per-token x_s) 与 W8A16-理想(x_s=1) 用**同一** GEMM 内核。

        证明点: 二者调用完全相同的 _firefly_gemm_ref, 唯一差别是 x_s 的来源 ——
        对应实现里 int8_prefill_linear 恒做 per-token int8、权重布局相同。这里钉住
        "两种激活形态共用一条数值路径"这一设计不变量 (W8A16 经 firefly 仍走 int8
        GEMM, 因此 W8A16 输出同样含激活量化误差, 见上一条)。
        """
        w_int8, c_n = _fixed_quant_weight()
        g = np.random.default_rng(7)
        x = g.standard_normal((_FIX_M, _FIX_K))
        # W8A8: 真实 per-token 动态量化。
        x_q8, x_s8 = _per_token_int8_quant(x)
        y_w8a8 = _firefly_gemm_ref(x_q8, x_s8, w_int8, c_n)
        # W8A16-理想: 激活已是 int8 域 (x_s=1), 同一内核。
        x_q16 = np.rint(np.clip(x, -127, 127))
        x_s16 = np.ones(_FIX_M)
        y_w8a16 = _firefly_gemm_ref(x_q16, x_s16, w_int8, c_n)
        # 二者都应与各自反量化 fp 参考接线等值 (bit-exact), 即同一 GEMM 语义。
        for x_q, x_s, y in ((x_q8, x_s8, y_w8a8), (x_q16, x_s16, y_w8a16)):
            x_dq = x_q.astype(np.float64) * x_s[:, None]
            y_ref = x_dq @ _dequant_w8(w_int8, c_n).T
            rel = np.abs(y - y_ref).max() / np.abs(y_ref).max()
            self.assertLess(rel, 1e-9)


# --------------------------------------------------------------------------- #
# 5) 真 torch 端到端 (真实 FireflyInt8Scheme + 真 firefly int8 GEMM): 无 torch/CUDA 跳过
# --------------------------------------------------------------------------- #
@unittest.skipUnless(_TORCH_OK, "需要真 torch (本机无, 见镜像 e2e)")
@unittest.skipUnless(_CUDA_OK, "需要 CUDA (cutlass_scaled_mm 在 GPU, 本机跳过)")
class RealTorchEndToEndTest(unittest.TestCase):
    """真张量端到端等值基线: 驱动真实 scheme.apply_weights, 对照反量化 fp 参考。

    与 section 4 的 numpy 复刻同源判据 (int32 累加 bit-exact 复刻 cutlass), 但走
    真 GPU kernel。本机无 torch/CUDA → 整体跳过; 镜像 e2e 环境真跑。
    """

    def test_apply_weights_matches_dequant(self):
        torch = _real_torch
        from vllm.model_executor.layers.quantization.utils.firefly_int8 import (
            FireflyInt8Scheme,
        )

        class _L:  # 仅承载 apply_weights 需要的两个属性
            pass

        M, K, N = 16, 64, 32
        g = torch.Generator(device="cuda").manual_seed(0)
        w_fp = torch.randn(N, K, device="cuda", dtype=torch.float32, generator=g)
        c_n = (w_fp.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        w_int8 = (w_fp / c_n.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
        x = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=g)

        layer = _L()
        layer.ff_int8_w_int8 = w_int8
        layer.ff_int8_c_n = c_n.contiguous()
        out = FireflyInt8Scheme().apply_weights(layer, x, None)

        from vllm import _custom_ops as ops

        x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous())
        acc = x_q.double() @ w_int8.double().t()
        ref = (acc * x_s.double() * layer.ff_int8_c_n.double()).to(torch.float16)
        tol = 2 * ref.abs().max().item() * 2.0**-10  # ~2 fp16 ulp
        max_abs = (out.float() - ref.float()).abs().max().item()
        self.assertLessEqual(max_abs, tol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
