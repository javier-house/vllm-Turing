# syntax=docker/dockerfile:1.7
# vllm-turing 统一镜像: 官方底座 + 稳定基建(依赖/patch/预编译 flashqla)+ 整个项目源码,
# 构建期把 overlay 应用到已装 vllm(镜像开箱即可运行、可离线); 运行时入口
# (/opt/vllm-turing/entrypoint.sh) 负责拉最新并按需重新 overlay(见 entrypoint.sh)。
#
# 与旧两版差异:
#   - 不再 build 期逐文件 COPY overlay 到 /tmp/sm75-overlay 再装(改为 COPY 整个项目);
#   - 合并了原 Dockerfile.fast(基础镜像 + 容器内软编译), 只保留一条构建路径 build.sh;
#   - overlay 在构建期也应用一次, 因此无 git/无网时镜像仍能直接 serve;
#   - flashinfer/transformers 支持离线安装: 先跑 docker/prefetch_deps.sh 把 wheel 下到
#     docker/tmp/wheels(已 gitignore), 构建时若检测到对应 wheel 就离线装, 缺则在线下载。

ARG BASE_IMAGE=vllm/vllm-openai:v0.29.0-cu129@sha256:7ef5a35d1ef8ce2cf9d671dd91eec6e367c5849262e0362b4d3d4a26be0d87d2

# 1) 预编译 flash_qla GDN prefill .so(独立 build stage, 稳定内核, nvcc 慢, 编一次进镜像)
#    只依赖 base 镜像里的 torch, 不需要 flashinfer, 故直接 FROM base。
FROM ${BASE_IMAGE} AS flashqla-builder
ENV TORCH_CUDA_ARCH_LIST=7.5
ARG MAX_JOBS=1
ENV MAX_JOBS=${MAX_JOBS}
WORKDIR /opt/flashqla-build
COPY vllm/third_party/flash_qla_sm75 /opt/flashqla-build/source
RUN python3 /opt/flashqla-build/source/build_extension.py \
    --build-directory /opt/flashqla-build/work \
    --output /opt/flashqla-build/flash_qla_sm75_gdn.so \
    --verbose && \
    cuobjdump --list-elf /opt/flashqla-build/flash_qla_sm75_gdn.so \
        > /opt/flashqla-build/cuobjdump.txt && \
    test "$(grep -Eo 'sm_[0-9]+' /opt/flashqla-build/cuobjdump.txt | sort -u)" = "sm_75"

# 2) runtime: 整个项目(含 docker/tmp 预下载 wheel) + 依赖(离线优先) + 工具 + 预编译 .so + 构建期 overlay + 校验
FROM ${BASE_IMAGE} AS runtime
# git: entrypoint 挂载模式/离线 fallback 判 overlay 源状态需要它(base 镜像原本没有,
# 缺失时 entrypoint 直接跳过重新 overlay, 分支修复无法在运行时生效)。
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git; \
    rm -rf /var/lib/apt/lists/*
# 整个项目先拷进来(含 docker/tmp/wheels 预下载 wheel; 该目录不存在也不影响 COPY .)。
# 注意 .dockerignore 不再排除 /.git: 镜像自带完整仓库, 未挂载时 entrypoint 可直接
# 以镜像内 /vllm-Turing 为 overlay 源判断(见 entrypoint.sh)。
COPY . /vllm-Turing
# flashinfer 0.6.18 + transformers 5.15.1: docker/tmp/wheels 里有对应 wheel 就离线安装,
# 缺则在线下载。去 flashinfer-jit-cache: 无 SM75 AOT cubin, 会挂 CUDA graph warmup。
# 注意: wheel 必须匹配镜像的 Python(3.12)/平台(manylinux x86_64), 由 prefetch_deps.sh 按此下载。
RUN set -eux; \
    W=/vllm-Turing/docker/tmp/wheels; \
    fi_py="$(ls "$W"/flashinfer_python-0.6.18-*.whl 2>/dev/null || true)"; \
    fi_cu="$(ls "$W"/flashinfer_cubin-0.6.18-*.whl 2>/dev/null || true)"; \
    if [ -n "$fi_py" ] && [ -n "$fi_cu" ]; then \
      echo "[deps] 离线安装 flashinfer (来自 $W)"; \
      uv pip install --system --no-cache --no-deps "$fi_py" "$fi_cu"; \
    else \
      echo "[deps] docker/tmp 缺 flashinfer wheel, 在线安装"; \
      uv pip install --system --no-cache --no-deps flashinfer-python==0.6.18 flashinfer-cubin==0.6.18 \
          --extra-index-url https://flashinfer.ai/whl/; \
    fi; \
    uv pip uninstall --system flashinfer-jit-cache || true; \
    tf="$(ls "$W"/transformers-5.15.1-*.whl 2>/dev/null || true)"; \
    if [ -n "$tf" ]; then \
      echo "[deps] 离线安装 transformers (来自 $W)"; \
      uv pip install --system --no-cache --no-deps "$tf"; \
    else \
      echo "[deps] docker/tmp 缺 transformers wheel, 在线安装"; \
      uv pip install --system transformers==5.15.1; \
    fi; \
    rm -rf "$W"

ARG BUILD_DATE
ARG SOURCE_REVISION
ARG BASE_IMAGE_ID
ARG VLLM_SM75_RELEASE=vllm-turing
ARG UPSTREAM_VLLM_COMMIT=98dff2a81d747d1dba01a47f939f48c3526d4206
ARG ONECAT_FLASHQLA_COMMIT=187b932dbd11940f0bcf52fb3675dd47fd69f313
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE_ID}" \
      org.opencontainers.image.version="${VLLM_SM75_RELEASE}" \
      org.opencontainers.image.title="vLLM-Turing" \
      org.opencontainers.image.description="vLLM-Turing: base image + whole project; entrypoint pulls and re-applies overlay" \
      ai.vllm.upstream.commit="${UPSTREAM_VLLM_COMMIT}" \
      ai.vllm.upstream.version="0.29.0" \
      ai.vllm.sm75.source-revision="${SOURCE_REVISION}" \
      ai.vllm.flashqla.commit="${ONECAT_FLASHQLA_COMMIT}" \
      ai.vllm.sm75.release="${VLLM_SM75_RELEASE}"

# 工具(构建期校验 + 运行时 entrypoint / fast_compile 调用)
COPY docker/install_sm75_overlay.py /opt/vllm-turing/install_sm75_overlay.py
COPY docker/install_speculative.py /opt/vllm-turing/install_speculative.py
COPY docker/verify_sm75_image.py /opt/vllm-turing/verify_sm75_image.py
COPY docker/fast_compile.sh /opt/vllm-turing/fast_compile.sh
COPY docker/patch_transformers_startup.py /opt/vllm-turing/patch_transformers_startup.py
COPY docker/entrypoint.sh /opt/vllm-turing/entrypoint.sh

# 预编译 flashqla .so 进镜像; 记录构建源 commit 与 flashqla 源 hash
#   - image-source-revision: 供 entrypoint 判断"镜像 overlay 是否已是最新"
#   - flashqla.hash: 与构建源一致, 免运行时因 hash 不同触发重编
COPY --from=flashqla-builder /opt/flashqla-build/flash_qla_sm75_gdn.so /opt/vllm-turing/extensions/flash_qla_sm75_gdn.so
COPY --from=flashqla-builder /opt/flashqla-build/cuobjdump.txt /opt/vllm-turing/evidence/flashqla_cuobjdump.txt
RUN echo "${SOURCE_REVISION}" > /opt/vllm-turing/image-source-revision && \
    sha256sum /vllm-Turing/vllm/third_party/flash_qla_sm75/csrc/gdn_forward.cu \
        | awk '{print $1}' > /opt/vllm-turing/evidence/flashqla.hash && \
    mkdir -p /opt/vllm-turing/work /opt/vllm-turing/evidence && \
    chmod +x /opt/vllm-turing/entrypoint.sh /opt/vllm-turing/fast_compile.sh

# 构建期应用 overlay 到已装 vllm(镜像开箱可运行); --source-copy 供 verify 做 AST 契约校验
RUN python3 /opt/vllm-turing/patch_transformers_startup.py && \
    python3 /opt/vllm-turing/install_sm75_overlay.py /vllm-Turing/vllm \
        --source-copy /opt/vllm-turing/source/vllm && \
    python3 /opt/vllm-turing/verify_sm75_image.py && \
    test -s /opt/vllm-turing/extensions/flash_qla_sm75_gdn.so

# firefly 默认开(int4 走 int8 prefill, fp8 走 marlin); flashqla 用镜像预编译(禁 JIT)
ENV FLASH_QLA_SM75_PREBUILT_EXTENSION_PATH=/opt/vllm-turing/extensions/flash_qla_sm75_gdn.so \
    FLASH_QLA_SM75_ALLOW_JIT=0 \
    PYTHONWARNINGS=ignore::FutureWarning \
    TRANSFORMERS_VERBOSITY=error \
    VLLM_FIREFLY=auto \
    VLLM_USE_MODELSCOPE=true \
    MODELSCOPE_CACHE=/root/.cache/modelscope/hub \
    SM75_FA2_SMALLQ_GRAPH=1 \
    SM75_GDN_B1_METADATA=1

# 4) final: 装 speculative overlay + 设入口
FROM runtime AS final
COPY docker/speculative /opt/vllm-turing/speculative
RUN python3 /opt/vllm-turing/install_speculative.py
# 容器时间统一东八区(容器内 date/日志时间戳); base 缺 tzdata 时装, 再落地 /etc/localtime
RUN (command -v zdump >/dev/null 2>&1 || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tzdata && rm -rf /var/lib/apt/lists/*)) && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone
ENV TZ=Asia/Shanghai
ENTRYPOINT ["/opt/vllm-turing/entrypoint.sh"]
LABEL ai.vllm.sm75.modes="base,mtp,dflash2"
