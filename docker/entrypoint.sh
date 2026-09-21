#!/usr/bin/env bash
# vllm-turing 容器入口: 拉取最新 -> 判断是否需重新 overlay -> exec vllm serve "$@"
#
# 两种模式(自动检测):
#   挂载模式: 宿主机把 vllm-Turing 项目挂到 /vllm-Turing(含 .git), entrypoint 在该
#             目录 git pull, 以它作为 overlay 源。
#   不挂载:   /vllm-Turing 是镜像构建时 COPY 进去的源码(无 .git), entrypoint clone
#             一份到 /opt/vllm-turing/repo 作为 overlay 源。
#
# 拉取源: github(javier-house/vllm-Turing) 失败试 gitee(javier_house/vllm-Turing),
# 都失败(离线)则跳过更新, 用镜像构建时已应用的 overlay(镜像开箱可离线运行)。
#
# "是否需重新 overlay" 判定: 拉取后的 HEAD 与镜像构建时的源 commit
# (/opt/vllm-turing/image-source-revision) 一致 -> 镜像已最新, 跳过 overlay(秒起);
# 不一致 -> 跑 fast_compile.sh 重新 overlay(.py 秒级, .so 仅 .cu 变化才重编)。
#
# 安全: 上游 vLLM 大版本变化(仓库 UPSTREAM_VERSION 与已装 vllm 主版本不符)时只打印
#       提醒、不强行覆盖 site-packages(锚点注入会因上游改动失败/错位)。
set -uo pipefail

REPO_GITHUB="https://github.com/javier-house/vllm-Turing"
REPO_GITEE="https://gitee.com/javier_house/vllm-Turing"
IMAGE_DIR=/vllm-Turing
CLONE_DIR=/opt/vllm-turing/repo
IMAGE_REV_FILE=/opt/vllm-turing/image-source-revision

log() { echo "[entrypoint] $*" >&2; }

# 启动时是否拉取(git pull)最新代码: 默认不更新(生产安全); 显式 1/on/true/yes
# 才开启。读完即 unset, 不传给 vllm 进程, 避免 vllm env 校验把它当未知 VLLM_*
# 变量告警。
_update="${VLLM_TURING_UPDATE:-0}"
unset VLLM_TURING_UPDATE
case "$_update" in
  1 | on | true | yes)
    _update=1
    ;;
  *)
    _update=0
    ;;
esac

# 默认不更新: 不 clone/pull, 直接用镜像构建时已应用的 overlay(离线可跑)。
# 只有显式 VLLM_TURING_UPDATE=1 才走下面的"拉最新 + 按需重新 overlay"路径。
if [[ "$_update" != 1 ]]; then
  log "未开启 VLLM_TURING_UPDATE: 不拉取最新, 用镜像构建时已应用的 overlay。"
  exec vllm serve "$@"
fi

# --- 1) 确定 overlay 源仓库 ---
WORK=""
if [[ -d "$IMAGE_DIR/.git" ]]; then
  WORK="$IMAGE_DIR"  # 挂载模式: 宿主机项目挂载进来
  log "挂载模式: 用 $WORK 作为 overlay 源"
else
  log "不挂载模式: 取最新仓库 -> $CLONE_DIR"
  if ! [[ -d "$CLONE_DIR/.git" ]]; then
    for url in "$REPO_GITHUB" "$REPO_GITEE"; do
      if git clone --depth 1 --branch main "$url" "$CLONE_DIR" 2>/dev/null; then
        break
      fi
    done
  fi
  if [[ -d "$CLONE_DIR/.git" ]]; then
    WORK="$CLONE_DIR"
  else
    log "警告: 无法 clone(github/gitee 均失败, 可能离线)。跳过更新, 用镜像构建时已应用的 overlay。"
    exec vllm serve "$@"
  fi
fi

# --- 2) 更新到最新分支(github -> gitee -> 跳过) ---
# 到此处 _update 必为 1(默认不更新已在前面返回); git 缺失时不阻断: 用当前仓库
# 状态判断是否重新 overlay(离线镜像开箱可跑)。
if ! command -v git >/dev/null 2>&1; then
  log "警告: 容器内无 git, 跳过拉取, 用镜像构建时已应用的 overlay。"
  exec vllm serve "$@"
fi

# 仓库 overlay 脚本自动同步: 运行时链(entrypoint 调 fast_compile -> overlay 脚本)
# 若沿用镜像内烘焙副本, 仓库改了这些脚本就必须逐个 --volume 挂载覆盖。这里在确定
# overlay 源($WORK)后, 把仓库 docker/ 下同名 helper 脚本同步到 /opt/vllm-turing,
# 使后续 fast_compile 一律用仓库最新版; 镜像烘焙副本退为离线/不挂载时的兜底。
# 不复制 entrypoint.sh 自身(正在执行中), 其逻辑变化靠重建镜像生效(极少改)。
if [[ -d "$WORK/docker" ]]; then
  for _f in fast_compile.sh install_sm75_overlay.py install_speculative.py \
            patch_transformers_startup.py verify_sm75_image.py; do
    if [[ -f "$WORK/docker/$_f" ]]; then
      cp -f "$WORK/docker/$_f" "/opt/vllm-turing/$_f"
    fi
  done
  log "已把仓库 docker/ 脚本同步到 /opt/vllm-turing(免 --volume 逐文件挂载)。"
fi
updated=0
if git -C "$WORK" pull --ff-only 2>/dev/null; then
  updated=1
else
  for url in "$REPO_GITHUB" "$REPO_GITEE"; do
    git -C "$WORK" remote remove sm75mirror 2>/dev/null || true
    if git -C "$WORK" remote add sm75mirror "$url" 2>/dev/null \
       && git -C "$WORK" fetch sm75mirror main 2>/dev/null \
       && git -C "$WORK" merge --ff-only sm75mirror/main 2>/dev/null; then
      updated=1
      break
    fi
  done
fi
[[ "$updated" == 1 ]] || log "警告: 拉取最新失败(可能离线), 按当前仓库状态判断是否重新 overlay。"

# --- 3) 上游 vLLM 版本检查(主.次版本) ---
# vLLM 形如 0.x.y, 主版本恒为 0, 比较"主.次"(0.29)才有意义: 镜像按某上游编译,
# 仓库 UPSTREAM_VERSION 跳到新主.次版本时 overlay 锚点可能不匹配, 只提醒不覆盖。
installed_mm=$(python3 -c 'import vllm; p=vllm.__version__.split("+")[0].split("-")[0].split("."); print(p[0]+"."+p[1])' 2>/dev/null || echo "")
upstream_line=$(cat "$WORK/UPSTREAM_VERSION" 2>/dev/null | tr -d '[:space:]')
upstream_mm=$(printf '%s\n' "$upstream_line" | awk -F. '{if(NF>=2)print $1"."$2}')
if [[ -n "$installed_mm" && -n "$upstream_mm" && "$installed_mm" != "$upstream_mm" ]]; then
  log "警告: 上游版本不一致——镜像已装 vLLM ${installed_mm}.* 但仓库 UPSTREAM_VERSION=${upstream_line}(${upstream_mm})。"
  log "      overlay 锚点可能不匹配, 跳过重新 overlay, 沿用镜像构建时已应用的版本。"
  exec vllm serve "$@"
fi

# --- 4) 是否需要重新 overlay ---
# 一致且无本地改动 -> 镜像 overlay 已最新, 跳过(秒起)。
# 否则(新 commit / 本地未提交改动 / 离线取不到 rev) -> 重新 overlay。
# 含"本地改动"判定: 挂载模式下本地编辑了 overlay 文件(未提交)也能在启动时被应用。
cur_rev=$(git -C "$WORK" rev-parse HEAD 2>/dev/null || echo "")
img_rev=$(cat "$IMAGE_REV_FILE" 2>/dev/null || echo "")
dirty=$(git -C "$WORK" status --porcelain 2>/dev/null || true)
if [[ -n "$cur_rev" && "$cur_rev" == "$img_rev" && -z "$dirty" ]]; then
  log "仓库 HEAD($cur_rev) 与镜像构建源一致且无本地改动, overlay 已最新, 跳过重新 overlay。"
else
  log "仓库 HEAD(${cur_rev:-未知}) 与镜像构建源(${img_rev:-未知}) 不一致或存在本地改动, 重新 overlay(编译)..."
  if WORK="$WORK" bash /opt/vllm-turing/fast_compile.sh; then
    log "overlay 已同步到 ${cur_rev:-当前状态}。"
  else
    log "警告: fast_compile 失败, 沿用镜像构建时的 overlay 继续启动。"
  fi
fi

# --- 5) 启动 ---
log "启动 vllm serve ..."
exec vllm serve "$@"
