# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 (exllamav3 trellis) 在线解码 serving (sm75 原生实现)。

import 本包即触发 Exl3Config 的 @register_quantization_config("exl3"),
使其进 _CUSTOMIZED_METHOD_TO_QUANT_CONFIG, 由 get_quantization_config 的
method_to_config.update(...) pick up。
"""

from vllm.model_executor.layers.quantization.exl3.exl3 import Exl3Config

__all__ = ["Exl3Config"]
