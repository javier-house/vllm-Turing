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
- 空闲自动睡眠支持 CPU、reload 和 exit；exit 模式释放引擎进程、CUDA context、worker 和显存，下一请求透明冷启动。

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

基于固定 digest 的官方 `vllm/vllm-openai:v0.29.0-cu129` 镜像，安装全部适配并编译 SM75 扩展，生成 `vllm-Turing` 镜像。

### 3. 启动

使用 `docker run` 启动容器，在镜像名后填写模型路径和启动参数。



```bash
export VLLM_API_KEY='replace-with-your-api-key'
# 替换为实际绝对路径：编译缓存与模型下载目录分开。
export VLLM_SM75_CACHE_ROOT=/path/to/vllm-sm75/cache
export VLLM_SM75_MODEL_CACHE_ROOT=/path/to/model-cache

VARIANT=base FORMAT=fp8 bash docker/run.sh
```

FP8 默认通过 ModelScope 加载 `Qwen/Qwen3.8-27B-FP8`。脚本配置监听地址、端口、API key 和持久化挂载。使用相同 GPU 的模式按需互斥启动，脚本不会停止现有服务。

Firefly 控制变量可直接传给 `docker/run.sh`，配置与关闭方式见下文。

```bash
# MTP5：使用同一个镜像
VARIANT=mtp FORMAT=fp8 bash docker/run.sh

# DFlash2：先将匹配 draft 下载到自选目录
export MODEL_ROOT=/path/to/downloaded-models
DRAFT_MODEL=/models/Qwen3.8-27B-DFlash2 VARIANT=dflash2 FORMAT=fp8 \
  bash docker/run.sh
```

DFlash draft 使用 `incoai/Qwen3.8-27B-DFlash2`，完整模型文件放入上述挂载目录。
AWQ 使用 `philbert440/Qwen3.8-27B-W4A16-AWQ`，下载后设置 `MODEL=/models/Qwen3.8-27B-W4A16-AWQ FORMAT=awq`。

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models \
  --header "Authorization: Bearer $VLLM_API_KEY"
```

## Firefly 内核

INT4 路径用于大批量 prefill；本版本 FP8 线性计算仍使用 Marlin，FP8 测试启用的是 Firefly all-reduce。不能把未启用的 FP8 线性内核或其他模型数据作为当前27B的加速结果。

脚本默认 `VLLM_FIREFLY=1`、`VLLM_FIREFLY_AR=auto`，通信后端自动选择，阈值为1MiB；小消息回退 NCCL。为匹配已测路径，独立的官方 FlashInfer all-reduce 默认设为0。

```bash
# 关闭 Firefly 计算和通信优化，其他推理参数不变
VLLM_FIREFLY=0 VLLM_FIREFLY_AR=0 bash docker/run.sh
# 只关闭 Firefly all-reduce
VLLM_FIREFLY_AR=0 bash docker/run.sh
```

`--disable-custom-all-reduce` 不会关闭 Firefly。AWQ 路径已完成1–128K吞吐测试；量化通信对模型质量的影响未完成本轮验收。

## `/monitor` 监控看板

启动后访问 `http://主机IP:端口/monitor`。单文件HTML页面自动读取同源 `/metrics`，显示吞吐、并发、KV缓存、延迟分位、抢占与休眠状态，无CDN或额外监控服务依赖。

默认开启；关闭时在 Docker/Unraid 容器环境变量中设置 `VLLM_MONITOR=0`，重新创建容器后生效。`docker run` 对应参数为 `-e VLLM_MONITOR=0`；当前启动脚本不单独透传宿主机的该变量。看板和 `/metrics` 当前无需模型API key即可访问。

## 空闲自动休眠

请求完成并进入空闲后开始计时，达到设定时间自动休眠；新推理请求到达时自动恢复，API 服务保持在线。主要用途是降低长时间闲置时的显存占用和 GPU 功耗。

**日常省电、内存有限或使用 DFlash2，推荐 `exit`。** 启动脚本默认空闲 30 分钟后退出引擎及 GPU worker，下一请求自动重建；直接调用 `vllm serve` 则默认关闭自动休眠。

### 模式怎么选

| 模式 | 休眠和唤醒方式 | 必要条件 | 效果与限制 |
| --- | --- | --- | --- |
| **`exit`（推荐）** | 退出引擎和 worker；从已有磁盘模型文件重建，并复用匹配编译缓存 | 主模型、草稿模型及配置文件持续可读；保留缓存挂载；不需要 `--enable-sleep-mode` | 释放本引擎的 CUDA 上下文，本地已测四卡 P8；首个请求需等待完整重建 |
| `cpu` | 权重备份到 pinned CPU 内存，唤醒复制回 GPU | `--enable-sleep-mode`；额外 RAM 足够容纳实际权重备份（含草稿），另留服务内存 | 减少从磁盘重载权重的工作；保留进程和 CUDA 上下文，不保证 P8 |
| `reload` | 丢弃 GPU 权重，唤醒从 checkpoint 重载主模型 | `--enable-sleep-mode`；可读且支持重载的 checkpoint；**当前不要用于 DFlash2**，草稿不随主模型一起重载 | 不保留整模型权重备份，但进程、缓冲区等仍占内存；不保证 P8 |

exit/reload **不把运行时内存快照写入磁盘**，恢复来源是已有模型文件。编译缓存保存编译产物，不保存对话 KV；重启或 exit 唤醒后，长对话可能仍需重新处理输入。

内存与磁盘预算：cpu 需额外预留权重备份空间，不能简单按压缩模型文件大小计算；本地约 31 GiB 内存主机不采用整模型 cpu 备份。脚本原有 **8 GiB CPU KV offload** 是另一项内存开销，选择 exit/reload 不会取消它。磁盘保留完整主模型、草稿和编译缓存即可，无需单独准备休眠快照文件；文件页缓存也会使用可回收主机内存。

### 怎么配置

保留原推理参数，在 `vllm serve` 命令末尾添加：

```bash
--auto-sleep-idle-timeout 30 --auto-sleep-offload-target exit
```

超时单位是**分钟**：`1` = 60 秒测试，`30` = 日常 30 分钟，`0` = 关闭。使用仓库脚本时，在前文模型和路径设置的基础上任选对应配置：

```bash
# 60 秒测试
AUTO_SLEEP_IDLE_TIMEOUT=1 AUTO_SLEEP_OFFLOAD_TARGET=exit bash docker/run.sh
# 日常 30 分钟
AUTO_SLEEP_IDLE_TIMEOUT=30 AUTO_SLEEP_OFFLOAD_TARGET=exit bash docker/run.sh
# 关闭自动休眠
AUTO_SLEEP_IDLE_TIMEOUT=0 bash docker/run.sh
```

原有 `VARIANT`、`FORMAT`、模型等设置继续保留。已存在的同名容器需要按新参数重建，脚本不会自动替换它；重建时保留模型与编译缓存挂载。

| 参数 | 作用 / 默认值 |
| --- | --- |
| `--auto-sleep-idle-timeout` | 空闲分钟数；直接 CLI 默认 `0`，脚本变量 `AUTO_SLEEP_IDLE_TIMEOUT` 默认 `30` |
| `--auto-sleep-offload-target` | `exit` / `cpu` / `reload`；直接 CLI 默认 `cpu`，脚本变量 `AUTO_SLEEP_OFFLOAD_TARGET` 默认 `exit` |
| `--enable-sleep-mode` | cpu/reload 必需；脚本选择这两种模式时自动添加，exit 不需要 |
| `--auto-sleep-reload-path` | reload 的容器内 checkpoint 路径，默认启动模型路径；脚本变量 `AUTO_SLEEP_RELOAD_PATH` |
| `--auto-sleep-page-cache-keep-interval` | reload 文件页预热间隔，默认 `600` 秒；脚本变量 `AUTO_SLEEP_PAGE_CACHE_KEEP_INTERVAL`。`0` 关闭睡眠时及后台预热，唤醒前仍提示预热一次 |

exit 只在退出前提示预热主模型文件页，没有后台预热进程。预热是 OS 提示，不能保证唤醒必定命中内存中的文件页。客户端及反向代理超时应覆盖完整唤醒时间。

## License

vLLM 修改遵循上游 Apache-2.0 License。FlashQLA-SM75 保留原始 MIT License 与[来源说明](vllm/third_party/flash_qla_sm75/SOURCE.md)。
