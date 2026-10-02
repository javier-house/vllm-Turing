# vllm-Turing

持续同步官方 [vLLM](https://github.com/vllm-project/vllm)，完善 SM75 兼容支持与内核优化。

本项目基于 vLLM 0.29.0，集成 MTP、DFlash2 和自动休眠适配。

## 关于本项目

一个出于兴趣的项目：为图灵架构（SM75）显卡提升 vLLM 的推理速度。项目在原生 vLLM 之上叠加上游相关优化，并进一步适配图灵显卡，尽量贴合原版、不引入额外功能。

- 不做正式版本发布，灵活跟随 vLLM 上游可用版本。
- 目前 AWQ INT4 与 W4A16 路径的 prefill 阶段有显著提升。

## 增强功能简要

- FlashQLA-SM75 GDN prefill、Triton decode、FlashInfer 0.6.18、Marlin FP8 和 FP8 KV。
- SM75 CUDA Graph、GDN 状态准备融合及原生 MTP5 验证路径。
- DFlash2 的 SM75 数值兼容、AWQ 数据类型和 TP4 处理。
- ModelScope、模型缓存与 vLLM/FlashInfer 编译缓存持久化。
- 统一镜像支持普通推理、MTP5 和 DFlash2，通过启动参数选择模式。
- 宿主机 nvidia-pstated 空闲限频（P-State）：空闲卡降 P8 省功耗、来请求秒回满速，不释放显存。

## 快速复现

以下覆盖基础推理验收与自动休眠测试。

### 编译缓存持久化

启动脚本将 vLLM、FlashInfer、Triton 和 PyTorch 扩展编译缓存持久化到宿主机，包含主模型、DFlash2 草稿和候选选择器的匹配产物。模型目录与编译目录分开，变量示例见下面启动命令；更新镜像或重建容器时保留挂载。首次运行及代码、依赖或计算配置变化仍可能需要编译，缓存加载时间不等于完整启动时间。

### 1. 克隆

```bash
git clone https://github.com/javier-house/vllm-Turing.git
cd vllm-Turing
```

### 2. 构建

Linux x86_64，需安装 Docker、Git 和 Bash。启动模型另需 NVIDIA 驱动与 NVIDIA Container Toolkit。

```bash
bash docker/build.sh
```

基于固定 digest 的官方 `vllm/vllm-openai:v0.29.0-cu129` 镜像，安装全部适配并编译 SM75 扩展，生成 `vllm-turing` 镜像（不做正式发布，不带版本 tag）。

构建默认在线安装 flashinfer 0.6.18 与 transformers 5.15.1。若在无法直连 github 的环境构建，可先在能联网的机器上预下载 wheel，构建时检测到 `docker/tmp/wheels/` 里的对应 wheel 就离线安装（缺失的包仍在线兜底）：

```bash
bash docker/prefetch_deps.sh   # 把 flashinfer/transformers wheel 下到 docker/tmp/wheels(已 gitignore)
```

wheel 按镜像的 Python 3.12 下载，与构建机自身 Python 版本无关；flashinfer 的 wheel 来自 flashinfer.ai/github，建议在该环境能访问 github 时预下载。

### 3. 启动

用 `docker run` 启动：镜像名 `vllm-turing` 后接模型与启动参数。容器入口（entrypoint）默认**不更新**——直接用镜像内置代码，普通运行不受拉取影响；需要拉取最新 `vllm-Turing` 并重新 overlay 时加 `-e VLLM_TURING_UPDATE=1`（示例 A/B）。更多启动命令参考 `docs/`（按模型分子目录，如 `docs/Qwen/Qwen3.8-Flash-Next/start_w4a16_tp8.sh`）。先设置：

```bash
export VLLM_API_KEY='replace-with-your-api-key'
# 编译缓存与模型目录分开（实际绝对路径）。
export VLLM_SM75_CACHE_ROOT=/path/to/vllm-turing/cache
export VLLM_SM75_MODEL_CACHE_ROOT=/path/to/model-cache
```

**示例 A：挂载本地项目**——把宿主机 `vllm-Turing` 项目挂到 `/vllm-Turing`，entrypoint 在该目录 `git pull` 同步最新：

```bash
docker run -d --name vllm-turing-fp8 --gpus all --shm-size 16g \
  -v /path/to/vllm-Turing:/vllm-Turing \
  -v "$VLLM_SM75_MODEL_CACHE_ROOT":/root/.cache/modelscope \
  -v "$VLLM_SM75_MODEL_CACHE_ROOT":/root/.cache/huggingface \
  -v "$VLLM_SM75_CACHE_ROOT/fp8/vllm":/root/.cache/vllm \
  -v "$VLLM_SM75_CACHE_ROOT/shared/flashinfer":/root/.cache/flashinfer \
  -v "$VLLM_SM75_CACHE_ROOT/fp8/triton":/root/.triton/cache \
  -v "$VLLM_SM75_CACHE_ROOT/shared/torch_extensions":/root/.cache/torch_extensions \
  -e VLLM_FIREFLY_DIRECT=1 -e VLLM_FIREFLY_AR=1 -e VLLM_TURING_UPDATE=1 \
  -e TRITON_CACHE_DIR=/root/.triton/cache -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  vllm-turing Qwen/Qwen3.8-27B-FP8 \
  --served-model-name VLLM-Qwen3.8-27B --host 0.0.0.0 --port 8000 --api-key "$VLLM_API_KEY" \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 4096 \
  --gpu-memory-utilization 0.87 --max-model-len auto \
  --attention-config '{"backend":"FLASHINFER"}' --gdn-prefill-backend flashqla_sm75 \
  --kv-cache-dtype fp8_e4m3 --block-size 32 --dtype float16 \
  --hf-overrides '{"dtype":"float16"}' --generation-config vllm \
  --enable-prefix-caching --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}'
```

**示例 B：不挂载本地项目**——与示例 A 唯一区别是去掉 `-v /path/to/vllm-Turing:/vllm-Turing`；entrypoint 改用镜像内自带的 `/vllm-Turing` 并自行 `clone`/`pull` 最新：

```bash
docker run -d --name vllm-turing-fp8 --gpus all --shm-size 16g \
  -v "$VLLM_SM75_MODEL_CACHE_ROOT":/root/.cache/modelscope \
  -v "$VLLM_SM75_MODEL_CACHE_ROOT":/root/.cache/huggingface \
  -v "$VLLM_SM75_CACHE_ROOT/fp8/vllm":/root/.cache/vllm \
  -v "$VLLM_SM75_CACHE_ROOT/shared/flashinfer":/root/.cache/flashinfer \
  -v "$VLLM_SM75_CACHE_ROOT/fp8/triton":/root/.triton/cache \
  -v "$VLLM_SM75_CACHE_ROOT/shared/torch_extensions":/root/.cache/torch_extensions \
  -e VLLM_FIREFLY_DIRECT=1 -e VLLM_FIREFLY_AR=1 -e VLLM_TURING_UPDATE=1 \
  -e TRITON_CACHE_DIR=/root/.triton/cache -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  vllm-turing Qwen/Qwen3.8-27B-FP8 \
  --served-model-name VLLM-Qwen3.8-27B --host 0.0.0.0 --port 8000 --api-key "$VLLM_API_KEY" \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 4096 \
  --gpu-memory-utilization 0.87 --max-model-len auto \
  --attention-config '{"backend":"FLASHINFER"}' --gdn-prefill-backend flashqla_sm75 \
  --kv-cache-dtype fp8_e4m3 --block-size 32 --dtype float16 \
  --hf-overrides '{"dtype":"float16"}' --generation-config vllm \
  --enable-prefix-caching --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}' \
```

**示例 C：启动但不更新代码（默认行为，生产环境建议）**——与示例 A 相同（仍挂载本地项目），不设 `VLLM_TURING_UPDATE` 即默认不拉取、用镜像内置代码。生产环境建议使用此方式：代码版本固定、启动时不联网更新，行为可预期。

```bash
docker run -d --name vllm-turing-fp8 --gpus all --shm-size 16g \
  -v /path/to/vllm-Turing:/vllm-Turing \
  -v "$VLLM_SM75_MODEL_CACHE_ROOT":/root/.cache/modelscope \
  -v "$VLLM_SM75_MODEL_CACHE_ROOT":/root/.cache/huggingface \
  -v "$VLLM_SM75_CACHE_ROOT/fp8/vllm":/root/.cache/vllm \
  -v "$VLLM_SM75_CACHE_ROOT/shared/flashinfer":/root/.cache/flashinfer \
  -v "$VLLM_SM75_CACHE_ROOT/fp8/triton":/root/.triton/cache \
  -v "$VLLM_SM75_CACHE_ROOT/shared/torch_extensions":/root/.cache/torch_extensions \
  -e VLLM_FIREFLY_DIRECT=1 -e VLLM_FIREFLY_AR=1 \
  -e TRITON_CACHE_DIR=/root/.triton/cache -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  vllm-turing Qwen/Qwen3.8-27B-FP8 \
  --served-model-name VLLM-Qwen3.8-27B --host 0.0.0.0 --port 8000 --api-key "$VLLM_API_KEY" \
  --tensor-parallel-size 4 --disable-custom-all-reduce \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 4096 \
  --gpu-memory-utilization 0.87 --max-model-len auto \
  --attention-config '{"backend":"FLASHINFER"}' --gdn-prefill-backend flashqla_sm75 \
  --kv-cache-dtype fp8_e4m3 --block-size 32 --dtype float16 \
  --hf-overrides '{"dtype":"float16"}' --generation-config vllm \
  --enable-prefix-caching --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}' \
```

`--long-prefill-token-threshold N`：限制单个请求一个 step 内最多消化的 prefill token 数（chunked prefill 下生效），用来压住"超长 prompt 一步吃满整个 batch、把同批 decode 请求的输出间隔拖长"。取值须明显小于 `--max-num-batched-tokens` 才有效。副作用是长 prompt 的首字延迟（TTFT）变高、prefill 总时间变长；对本项目的 27B 模型无 firefly 顾虑——firefly int8 prefill 在 27B 上的交叉点 M\*<561，切到 2048~4096 的 chunk 仍远在收益区内。

FP8 默认通过 ModelScope 加载 `Qwen/Qwen3.8-27B-FP8`，示例已配置监听地址、端口、API key 和持久化挂载。使用相同 GPU 的模式按需互斥启动，容器需手动重建（不会自动替换同名容器）。

MTP5 / DFlash2 复用上面同一个 `docker run` 命令，仅替换模型并把对应的 `--speculative-config` 加到参数末尾：

```bash
# MTP5：追加 --speculative-config 并把 --compilation-config 的 cudagraph_capture_sizes 设为 [6]
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --scheduler-cls vllm.v1.core.sched.scheduler_sm75.SM75Scheduler
# DFlash2：先把匹配 draft 下载到自选目录并挂到 /models，再追加
  --speculative-config '{"method":"dflash","model":"/models/Qwen3.8-27B-DFlash2","num_speculative_tokens":7,"draft_tensor_parallel_size":4,"max_model_len":262144,"kv_cache_dtype":"auto","attention_backend":"FLASHINFER","draft_sample_method":"probabilistic"}' \
  --scheduler-cls vllm.v1.core.sched.scheduler_sm75.SM75Scheduler
```

DFlash draft 使用 `incoai/Qwen3.8-27B-DFlash2`，完整模型文件放入挂载到 `/models` 的目录。
AWQ 使用 `philbert440/Qwen3.8-27B-W4A16-AWQ`，下载后把模型参数换成 `MODEL=/models/Qwen3.8-27B-W4A16-AWQ`，并加 `--max-num-seqs 8 --max-num-batched-tokens 16384`。

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models \
  --header "Authorization: Bearer $VLLM_API_KEY"
```

## Firefly 内核

INT4 路径用于大批量 prefill；本版本 FP8 线性计算仍使用 Marlin，FP8 测试启用的是 Firefly all-reduce。不能把未启用的 FP8 线性内核或其他模型数据作为当前27B的加速结果。

示例默认 `-e VLLM_FIREFLY_DIRECT=1 -e VLLM_FIREFLY_AR=1`，通信后端自动选择，阈值为1MiB；小消息回退 NCCL。为匹配已测路径，独立的官方 FlashInfer all-reduce 默认设为0。Firefly 用上面 `docker run` 里的 `-e` 调整即可：

```bash
# 关闭 Firefly 计算和通信优化：把示例里两处 -e 改成
  -e VLLM_FIREFLY_DIRECT=0 -e VLLM_FIREFLY_AR=0
# 只关闭 Firefly all-reduce
  -e VLLM_FIREFLY_AR=0
```

`--disable-custom-all-reduce` 不会关闭 Firefly。AWQ 路径已完成1–128K吞吐测试；量化通信对模型质量的影响未完成本轮验收。

## `/monitor` 监控看板

启动后访问 `http://主机IP:端口/monitor`。单文件HTML页面自动读取同源 `/metrics`，显示吞吐、并发、KV缓存、延迟分位、抢占与休眠状态，无CDN或额外监控服务依赖。

默认开启；关闭时在 `docker run` 参数里加 `-e VLLM_MONITOR=0`，重新创建容器后生效。看板和 `/metrics` 当前无需模型API key即可访问。

## `/test` 测速页

启动后访问 `http://主机IP:端口/test`。页面引入开源工具 [llm_speedtest](https://github.com/gengchaogit/llm_speedtest) 的单文件HTML，前端直连模型 API（填本服务地址与 key）实测 Prefill/Decode 吞吐与首字延迟，无CDN或额外服务依赖。

默认开启；关闭时在 `docker run` 参数里加 `-e VLLM_TEST_INDEX=0`，重新创建容器后生效。页面与 `/metrics` 当前无需模型API key即可访问。

## 空闲自动休眠

请求完成进入空闲后，宿主机上的 `nvidia-pstated`（NVIDIA 专有驱动的 NVAPI P-State 守护）把空闲 GPU 降到 **P8**（核心进低功耗态，约 15–20W/卡）；推理请求到达时自动切回 **boost**（P0/1590MHz）。只限频、不释放显存，适合模型常驻的推理服务——空闲省功耗、来请求秒回满速，API 服务保持在线。

相比容器内自动休眠（释放显存、下一请求冷启动重建），nvidia-pstated 保显存，常驻场景下无冷启动开销，作为本项目推荐的空闲省电方案。

### 部署

宿主机需 NVIDIA 专有驱动（自带 `libnvidia-api.so.1`）与 root 权限。`nvidia-pstated` 二进制从 [sasha0552/nvidia-pstated](https://github.com/sasha0552/nvidia-pstated) 下载 v1.0.9；`nvidia-pstated.service` 见本仓库 [`docs/nvidia-pstated/`](docs/nvidia-pstated/nvidia-pstated.service)。

```bash
sudo install -m 755 nvidia-pstated /usr/local/sbin/t10-gpu-thermal/nvidia-pstated
sudo install -m 644 docs/nvidia-pstated/nvidia-pstated.service /etc/systemd/system/nvidia-pstated.service
sudo systemctl daemon-reload && sudo systemctl enable --now nvidia-pstated
```

### 配置

全部内联在 `nvidia-pstated.service` 的 `ExecStart` 里（无独立配置文件）：

| 项 | 位置 | 默认 / 说明 |
| --- | --- | --- |
| 作用卡 | `ExecStart` 串里 `GPUS=""` | 空=所有卡；改 `"4,5,6,7"`=只指定卡 |
| 空闲阈值 | `-ut 5` | 利用率 ≤5% 视为空闲 |
| 防抖 | `-ibs 30` | 空闲持续 30×100ms=3s 才降 P8（避免偶发低负载误判） |
| 空闲 / 负载目标 | `-psl 8 -psh 16` | P8（空闲）/ P16=auto（boost） |
| 温度强制降频 | `-tt 80` | >80°C 强制降频 |

改任意项后 `sudo systemctl daemon-reload && sudo systemctl restart nvidia-pstated`。

### 实测（真机 8×Tesla T10）

空闲 8 卡全 **P8 / 645MHz / 15–20W**；请求到达**第 1 秒**即冲 1590MHz（NVAPI 轮询 100ms，无 TTFT 损失）；请求结束约 5s 回 P8。有 CUDA context（模型常驻）的卡同样能稳定切 P8。

## License

vLLM 修改遵循上游 Apache-2.0 License。FlashQLA-SM75 保留原始 MIT License 与[来源说明](vllm/third_party/flash_qla_sm75/SOURCE.md)。
