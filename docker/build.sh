#!/usr/bin/env bash
set -euo pipefail
# 构建 vllm-turing 统一镜像(Linux Docker host; 不需要 GPU, 不停现有容器)。
# 不做正式版本发布(见 README): 镜像不版本化, 运行时入口在容器内同步到上游最新。
# 单条构建路径: 官方底座 + 依赖/patch/预编译 flashqla + 整个项目, 构建期应用 overlay,
# 镜像开箱即可 serve(无网也能跑); 运行时入口再按需在容器内更新+重新 overlay。
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v docker >/dev/null
# 构建源 commit 烤进镜像(供容器内 entrypoint 判断 overlay 是否最新); 无 git 时退化。
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf 'source-archive')}"
BASE_IMAGE='vllm/vllm-openai:v0.29.0-cu129@sha256:7ef5a35d1ef8ce2cf9d671dd91eec6e367c5849262e0362b4d3d4a26be0d87d2'
docker build --file "$ROOT/docker/Dockerfile" \
    --target final --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg MAX_JOBS="${MAX_JOBS:-1}" \
    --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --build-arg SOURCE_REVISION="$SOURCE_REVISION" \
    --build-arg BASE_IMAGE_ID="$BASE_IMAGE" \
    --tag vllm-turing "$ROOT"
