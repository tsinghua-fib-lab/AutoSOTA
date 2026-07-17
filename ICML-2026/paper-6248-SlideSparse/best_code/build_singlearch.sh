#!/bin/bash
# =============================================================================
# vllmbench GPU 单架构构建脚本 (Single-Arch Builder)
# =============================================================================
#
# [功能]
#   仅构建当前宿主机架构的镜像，不再使用 QEMU 模拟。
#   自动添加架构后缀 (e.g., :0.13.0_cu129_amd64 或 :0.13.0_cu129_arm64)。
#
# [优势]
#   速度快（原生编译），避免了跨架构模拟的性能损耗。
# cd vllmbench
# chmod +x build_singlearch.sh && ./build_singlearch.sh
# =============================================================================

set -e

# =================配置区域=================
IMAGE_BASE_NAME="bcacdwk/vllmbench"
TAG_VERSION="0.13.0_cu129"
# =========================================

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP] $1${NC}"; }

# 1. 检测当前架构
HOST_ARCH=$(uname -m)
case $HOST_ARCH in
    x86_64)
        ARCH_SUFFIX="amd64"
        DOCKER_PLATFORM="linux/amd64"
        ;;
    aarch64)
        ARCH_SUFFIX="arm64"
        DOCKER_PLATFORM="linux/arm64"
        ;;
    *)
        echo "不支持的架构: $HOST_ARCH"
        exit 1
        ;;
esac

FULL_IMAGE_NAME="${IMAGE_BASE_NAME}:${TAG_VERSION}_${ARCH_SUFFIX}"

log_step "1/3 检测环境..."
log_info "当前架构: ${HOST_ARCH}"
log_info "目标镜像: ${FULL_IMAGE_NAME}"

# 2. 检查登录
if [ ! -f "$HOME/.docker/config.json" ] || ! grep -q "auths" "$HOME/.docker/config.json"; then
    log_info "请登录 Docker Hub..."
    docker login
fi

# 3. 原生构建并推送
log_step "2/3 开始原生构建..."
# 使用 DOCKER_BUILDKIT 加速
export DOCKER_BUILDKIT=1

# 注意：这里不再需要 buildx 的复杂配置，直接用标准 build 即可
# 或者继续用 buildx 但只指定单一 platform (推荐用 buildx 因为快)
docker buildx build \
    --platform "${DOCKER_PLATFORM}" \
    --tag "${FULL_IMAGE_NAME}" \
    --push \
    .

log_step "3/3 完成！"
echo ""
log_info "✅ 镜像已推送: ${FULL_IMAGE_NAME}"
echo "--------------------------------------------------------"
echo "在新机器上拉取时，init_env.sh 会自动识别后缀。"
echo "--------------------------------------------------------"