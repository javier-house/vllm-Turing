#!/bin/sh
# W4A16 TP8 启动示例（8×T10 16GB sm75）。API key 与模型名按需替换。

set -a
export FLASH_QLA_SM75_PREBUILT_EXTENSION_PATH=/opt/vllm-turing/extensions/flash_qla_sm75_gdn.so
export FLASH_QLA_SM75_ALLOW_JIT=0
export VLLM_FIREFLY_DIRECT=0
export PYTHONUNBUFFERED=1
export NCCL_P2P_LEVEL=SYS
export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1
export VLLM_PLE_CPU_OFFLOAD=0

set +a
exec vllm serve /model_w4a16 \
  --served-model-name qwen \
  --api-key 'replace-with-your-api-key' \
  --port 55530 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --kv-cache-dtype fp8_e4m3 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --long-prefill-token-threshold 2048 \
  --gpu-memory-utilization 0.90 \
  --mamba-ssm-cache-dtype float32 \
  --gdn-prefill-backend flashqla_sm75 \
  --moe-backend auto \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --enable-sleep-mode
