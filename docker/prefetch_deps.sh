#!/usr/bin/env bash
# 预下载镜像依赖 wheel 到 docker/tmp/wheels/(已 gitignore), 供 Dockerfile 离线安装。
# 构建时若检测到对应 wheel 就离线装, 缺则在线兜底(见 Dockerfile runtime 阶段)。
#
# 关键点:
#   - wheel 必须匹配镜像的 Python(3.12)/平台(x86_64 manylinux), 故强制
#     --python-version 312 --abi cp312, 不用构建机自身 Python 版本。
#   - 不指定 --platform, 沿用构建机平台(与镜像同为 x86_64), 避免 manylinux tag 不匹配。
#   - flashinfer 的 wheel 在 flashinfer.ai/whl + github release, 下载较慢/可能需直连,
#     建议在能访问 github 的环境(如 VM)上跑本脚本; 下不全也没关系, 缺的构建时在线装。
#
# 用法:  bash docker/prefetch_deps.sh   (幂等, 已存在的 wheel 跳过, 可重复/断点跑)
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELS_DIR="$ROOT/docker/tmp/wheels"
PY_VER="${PREFETCH_PYVER:-312}"   # 镜像 Python 版本(vllm-openai base = 3.12)
mkdir -p "$WHEELS_DIR"

# PyPI 主索引候选(按顺序试); flashinfer 额外索引单独处理。
PYPY_MIRRORS=(
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://mirrors.aliyun.com/pypi/simple"
  "https://pypi.mirrors.ustc.edu.cn/simple"
  "https://pypi.org/simple"
)
FI_EXTRA="https://flashinfer.ai/whl/"
# flashinfer 版本固定(与 Dockerfile 一致)
FI_VERSION="0.6.18"
TF_VERSION="5.15.1"

log() { echo "[prefetch] $*" >&2; }

# 下载一个包(给定 index-url 与 extra-index 列表)。成功落盘到 WHEELS_DIR 返回 0。
fetch() {
  local label="$1"; shift
  local requirement="$1"; shift
  # 余下参数原样传给 pip(如 --extra-index-url ...)
  local extra=("$@")

  local index
  for index in "${PYPY_MIRRORS[@]}"; do
    log "尝试 $label via $index ${extra[*]:-}"
    if python3 -m pip download --only-binary=:all: --no-deps --no-build-isolation \
        --python-version "$PY_VER" --implementation cp --abi "cp${PY_VER}" \
        --index-url "$index" "${extra[@]}" \
        --timeout 60 --retries 3 \
        -d "$WHEELS_DIR" "$requirement" 2>&1 | grep -v 'Using cached'; then
      return 0
    fi
    log "  $label via $index 失败, 换下一个镜像…"
  done
  return 1
}

status=0

# transformers 5.15.1(纯 py3-none-any wheel, PyPI)
if ls "$WHEELS_DIR"/transformers-${TF_VERSION}-*.whl >/dev/null 2>&1; then
  log "transformers-${TF_VERSION} 已存在, 跳过"
else
  fetch "transformers" "transformers==${TF_VERSION}" || { log "  transformers 下载失败(构建时将在线兜底)"; status=1; }
fi

# flashinfer-cubin(纯 py3-none-any, 主索引是 github release, 经 flashinfer.ai/whl 索引)
if ls "$WHEELS_DIR"/flashinfer_cubin-${FI_VERSION}-*.whl >/dev/null 2>&1; then
  log "flashinfer-cubin-${FI_VERSION} 已存在, 跳过"
else
  fetch "flashinfer-cubin" "flashinfer-cubin==${FI_VERSION}" --extra-index-url "$FI_EXTRA" \
    || { log "  flashinfer-cubin 下载失败(构建时将在线兜底)"; status=1; }
fi

# flashinfer-python(manylinux 平台 wheel, 体积大, 经 flashinfer.ai/whl 索引)
if ls "$WHEELS_DIR"/flashinfer_python-${FI_VERSION}-*.whl >/dev/null 2>&1; then
  log "flashinfer-python-${FI_VERSION} 已存在, 跳过"
else
  fetch "flashinfer-python" "flashinfer-python==${FI_VERSION}" --extra-index-url "$FI_EXTRA" \
    || { log "  flashinfer-python 下载失败(构建时将在线兜底)"; status=1; }
fi

log "wheel 目录: $WHEELS_DIR"
ls -la "$WHEELS_DIR" >&2
if [ "$status" -ne 0 ]; then
  log "部分 wheel 未下载成功——不影响构建(Dockerfile 对缺失的包会在线安装)。"
fi
exit "$status"
