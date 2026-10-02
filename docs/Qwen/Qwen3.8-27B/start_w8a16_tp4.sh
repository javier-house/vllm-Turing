#!/bin/sh
# API key 与模型名按需替换。
# 模型: Qwen3.8-27B-heretic-ara-INT8-W8A16-MTP (dense, GDN hybrid, 不开 MTP/dflash2)。
# dense 无 MoE, 故无 --enable-expert-parallel / --moe-backend。
# 容器编排: --gpus all + CUDA_VISIBLE_DEVICES=4,5,6,7, --shm-size 需 ≥16g (docker 默认 64MB 会崩)。

set -a
export FLASH_QLA_SM75_PREBUILT_EXTENSION_PATH=/opt/vllm-turing/extensions/flash_qla_sm75_gdn.so
export FLASH_QLA_SM75_ALLOW_JIT=0
export VLLM_FIREFLY_DIRECT=1
export PYTHONUNBUFFERED=1
export NCCL_P2P_LEVEL=SYS
export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1
export VLLM_PLE_CPU_OFFLOAD=0

set +a
exec vllm serve /model \
  --served-model-name qwen \
  --api-key 'replace-with-your-api-key' \
  --port 55530 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8_e4m3 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --long-prefill-token-threshold 2048 \
  --gpu-memory-utilization 0.90 \
  --mamba-ssm-cache-dtype float32 \
  --gdn-prefill-backend flashqla_sm75 \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}' \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --enable-sleep-mode
