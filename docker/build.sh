#!/usr/bin/env bash
set -euo pipefail
# 构建 vllm-turing 统一镜像(Linux Docker host; 不需要 GPU, 不停现有容器)。
# 不做正式版本发布(见 README): 镜像不版本化, 运行时入口在容器内同步到上游最新。
# 单条构建路径: 官方底座 + 依赖/patch/预编译 flashqla + 整个项目, 构建期应用 overlay,
# 镜像开箱即可 serve(无网也能跑); 运行时入口再按需在容器内更新+重新 overlay。
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v docker >/dev/null

# 发布元数据单源: BASE_IMAGE / UPSTREAM commit / jobs cap 等以 PROJECT_RELEASE.env 为准,
# 消灭与 Dockerfile 两处手抄 digest 的漂移。
# shellcheck source=/dev/null
[ -f "$ROOT/PROJECT_RELEASE.env" ] && . "$ROOT/PROJECT_RELEASE.env"
BASE_IMAGE="${VLLM_TURING_BASE_IMAGE:?PROJECT_RELEASE.env 缺 VLLM_TURING_BASE_IMAGE}"
UPSTREAM_COMMIT="${VLLM_TURING_UPSTREAM_COMMIT:-}"
FLASHQLA_COMMIT="${VLLM_TURING_FLASHQLA_COMMIT:-}"

# MAX_JOBS 自动钳制: nvcc 并发编译吃内存, 低内存构建机易 OOM(我方有 OOM 史)。
# 显式设了 MAX_JOBS 就用它; 否则 auto = min(nproc, 可用内存GB/2, cap), 下限 1。
# cap 来自 PROJECT_RELEASE.env(默认 8)。
_jobs_cap="${VLLM_TURING_BUILD_MAX_JOBS_CAP:-8}"
if [ -z "${MAX_JOBS:-}" ]; then
    _nproc="$(nproc 2>/dev/null || printf '1')"
    _mem_gb="$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || printf '0')"
    _jobs_mem=$((_mem_gb / 2))
    MAX_JOBS="$_nproc"
    [ "$_jobs_mem" -lt "$MAX_JOBS" ] && MAX_JOBS="$_jobs_mem"
    [ "$_jobs_cap" -lt "$MAX_JOBS" ] && MAX_JOBS="$_jobs_cap"
    [ "$MAX_JOBS" -lt 1 ] && MAX_JOBS=1
    echo "[build] MAX_JOBS 自动钳制: nproc=$_nproc mem_gb=$_mem_gb cap=$_jobs_cap -> MAX_JOBS=$MAX_JOBS"
fi

# 构建源 commit 烤进镜像(供容器内 entrypoint 判断 overlay 是否最新); 无 git 时退化。
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf 'source-archive')}"
docker build --file "$ROOT/docker/Dockerfile" \
    --target final --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg MAX_JOBS="$MAX_JOBS" \
    --build-arg BUILD_DATE="$(TZ='Asia/Shanghai' date +%Y-%m-%dT%H:%M:%S%:z)" \
    --build-arg SOURCE_REVISION="$SOURCE_REVISION" \
    --build-arg BASE_IMAGE_ID="$BASE_IMAGE" \
    ${UPSTREAM_COMMIT:+--build-arg UPSTREAM_VLLM_COMMIT="$UPSTREAM_COMMIT"} \
    ${FLASHQLA_COMMIT:+--build-arg ONECAT_FLASHQLA_COMMIT="$FLASHQLA_COMMIT"} \
    --tag vllm-turing "$ROOT"
