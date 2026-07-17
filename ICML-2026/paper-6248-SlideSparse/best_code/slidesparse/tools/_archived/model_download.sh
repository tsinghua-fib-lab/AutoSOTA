#!/bin/bash
# ============================================================================
# SlideSparse Model Download Script
# ============================================================================
# 用于批量下载 vLLM 基线测试所需的 W8A8 量化模型
# 支持 INT8 (quantized.w8a8) 和 FP8 (-FP8-dynamic) 两种格式
# 
# 使用方法:
#   ./model_download.sh [选项]
#   
# 选项:
#   -a, --all           下载所有模型
#   -i, --int8          仅下载 INT8 模型
#   -f, --fp8           仅下载 FP8 模型
#   -q, --qwen          仅下载 Qwen2.5 系列
#   -l, --llama         仅下载 Llama3.2 系列
#   -m, --model NAME    下载指定模型 (如: qwen2.5-7b-int8)
#   -c, --check         检查已下载模型状态
#   -h, --help          显示帮助信息
#
# 示例:
#   ./model_download.sh --all                    # 下载全部模型
#   ./model_download.sh --int8 --qwen            # 下载 Qwen INT8 模型
#   ./model_download.sh --model qwen2.5-7b-fp8   # 下载指定模型
#   ./model_download.sh --check                  # 检查下载状态
# ============================================================================


# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"

# HuggingFace 组织名
HF_ORG="RedHatAI"

# ============================================================================
# 模型定义
# ============================================================================
# 格式: [简短key]="HF模型名|本地文件夹名"
# 本地文件夹命名规则: {模型系列}-{规模}-{量化类型}
# 例如: Qwen2.5-7B-FP8, Llama3.2-1B-INT8

# INT8 W8A8 模型 (quantized.w8a8)
declare -A INT8_MODELS=(
    ["qwen2.5-0.5b-int8"]="Qwen2.5-0.5B-Instruct-quantized.w8a8|Qwen2.5-0.5B-INT8"
    ["qwen2.5-1.5b-int8"]="Qwen2.5-1.5B-Instruct-quantized.w8a8|Qwen2.5-1.5B-INT8"
    ["qwen2.5-3b-int8"]="Qwen2.5-3B-Instruct-quantized.w8a8|Qwen2.5-3B-INT8"
    ["qwen2.5-7b-int8"]="Qwen2.5-7B-Instruct-quantized.w8a8|Qwen2.5-7B-INT8"
    ["qwen2.5-14b-int8"]="Qwen2.5-14B-Instruct-quantized.w8a8|Qwen2.5-14B-INT8"
    ["llama3.2-1b-int8"]="Llama-3.2-1B-Instruct-quantized.w8a8|Llama3.2-1B-INT8"
    ["llama3.2-3b-int8"]="Llama-3.2-3B-Instruct-quantized.w8a8|Llama3.2-3B-INT8"
)

# FP8 W8A8 模型 (FP8-dynamic)
declare -A FP8_MODELS=(
    ["qwen2.5-0.5b-fp8"]="Qwen2.5-0.5B-Instruct-FP8-dynamic|Qwen2.5-0.5B-FP8"
    ["qwen2.5-1.5b-fp8"]="Qwen2.5-1.5B-Instruct-FP8-dynamic|Qwen2.5-1.5B-FP8"
    ["qwen2.5-3b-fp8"]="Qwen2.5-3B-Instruct-FP8-dynamic|Qwen2.5-3B-FP8"
    ["qwen2.5-7b-fp8"]="Qwen2.5-7B-Instruct-FP8-dynamic|Qwen2.5-7B-FP8"
    ["qwen2.5-14b-fp8"]="Qwen2.5-14B-Instruct-FP8-dynamic|Qwen2.5-14B-FP8"
    ["llama3.2-1b-fp8"]="Llama-3.2-1B-Instruct-FP8-dynamic|Llama3.2-1B-FP8"
    ["llama3.2-3b-fp8"]="Llama-3.2-3B-Instruct-FP8-dynamic|Llama3.2-3B-FP8"
)

# 解析模型信息的辅助函数
get_hf_name() {
    echo "$1" | cut -d'|' -f1
}

get_local_dir_name() {
    echo "$1" | cut -d'|' -f2
}

# ============================================================================
# 辅助函数
# ============================================================================

print_header() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "SlideSparse Model Download Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -a, --all           Download all models (INT8 + FP8)"
    echo "  -i, --int8          Download INT8 (quantized.w8a8) models only"
    echo "  -f, --fp8           Download FP8 (FP8-dynamic) models only"
    echo "  -q, --qwen          Download Qwen2.5 series only"
    echo "  -l, --llama         Download Llama3.2 series only"
    echo "  -m, --model NAME    Download specific model"
    echo "  -c, --check         Check downloaded model status"
    echo "  -s, --size          Show estimated model sizes"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Available Models:"
    echo ""
    echo "  INT8 Models (quantized.w8a8):"
    for key in "${!INT8_MODELS[@]}"; do
        echo "    - $key"
    done | sort
    echo ""
    echo "  FP8 Models (FP8-dynamic):"
    for key in "${!FP8_MODELS[@]}"; do
        echo "    - $key"
    done | sort
    echo ""
    echo "Examples:"
    echo "  $0 --all                         # Download all models"
    echo "  $0 --int8                        # Download all INT8 models"
    echo "  $0 --fp8 --qwen                  # Download Qwen FP8 models"
    echo "  $0 --model qwen2.5-7b-int8       # Download specific model"
    echo "  $0 --check                       # Check download status"
}

# 创建目录结构
setup_directories() {
    print_info "Checking directory structure..."
    
    # 创建 checkpoints 根目录
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        mkdir -p "$CHECKPOINT_DIR"
        print_success "Created checkpoints directory: $CHECKPOINT_DIR"
    fi
}

# 下载单个模型
download_model() {
    local model_key=$1
    local model_info=$2
    local quant_type=$3  # int8 或 fp8 (仅用于显示)
    
    local hf_name=$(get_hf_name "$model_info")
    local local_dir_name=$(get_local_dir_name "$model_info")
    local hf_path="${HF_ORG}/${hf_name}"
    local local_dir="${CHECKPOINT_DIR}/${local_dir_name}"
    
    print_header "Downloading: ${local_dir_name}"
    print_info "HuggingFace path: ${hf_path}"
    print_info "Local path: ${local_dir}"
    echo ""
    
    # 检查是否已下载
    if [ -d "$local_dir" ] && [ -f "${local_dir}/config.json" ]; then
        print_warning "Model exists, skipping: ${local_dir_name}"
        echo ""
        return 0
    fi
    
    # 创建目标目录
    mkdir -p "$local_dir"
    
    # 使用 hf download 下载
    print_info "Starting download..."
    if hf download \
        "${hf_path}" \
        --local-dir "${local_dir}"; then
        print_success "Download completed: ${local_dir_name}"
    else
        print_error "Download failed: ${local_dir_name}"
        return 1
    fi
    
    echo ""
}

# 检查模型状态
check_model_status() {
    local model_info=$1
    local local_dir_name=$(get_local_dir_name "$model_info")
    local local_dir="${CHECKPOINT_DIR}/${local_dir_name}"
    
    if [ -d "$local_dir" ] && [ -f "${local_dir}/config.json" ]; then
        # 计算目录大小
        local size=$(du -sh "$local_dir" 2>/dev/null | cut -f1)
        echo -e "  ${GREEN}✓${NC} ${local_dir_name} - ${size}"
        return 0
    else
        echo -e "  ${RED}✗${NC} ${local_dir_name} - not downloaded"
        return 1
    fi
}

# 检查所有模型状态
check_all_models() {
    print_header "Model Download Status"
    
    local downloaded=0
    local missing=0
    
    echo "INT8 Models (quantized.w8a8):"
    echo "-----------------------------------------"
    for key in $(echo "${!INT8_MODELS[@]}" | tr ' ' '\n' | sort); do
        if check_model_status "${INT8_MODELS[$key]}"; then
            downloaded=$((downloaded + 1))
        else
            missing=$((missing + 1))
        fi
    done
    
    echo ""
    echo "FP8 Models (FP8-dynamic):"
    echo "-----------------------------------------"
    for key in $(echo "${!FP8_MODELS[@]}" | tr ' ' '\n' | sort); do
        if check_model_status "${FP8_MODELS[$key]}"; then
            downloaded=$((downloaded + 1))
        else
            missing=$((missing + 1))
        fi
    done
    
    echo ""
    echo "-----------------------------------------"
    echo -e "Total: ${GREEN}${downloaded} downloaded${NC}, ${RED}${missing} missing${NC}"
    echo ""
    
    # 显示 checkpoints 总大小
    if [ -d "$CHECKPOINT_DIR" ]; then
        local total_size=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
        print_info "Checkpoints directory size: ${total_size}"
    fi
}

# 显示模型预估大小
show_model_sizes() {
    print_header "Estimated Model Sizes"
    echo "Note: These are rough estimates, actual sizes may vary"
    echo ""
    echo "Model size reference:"
    echo "  - 0.5B model: ~1-2 GB"
    echo "  - 1B model:   ~2-3 GB"
    echo "  - 1.5B model: ~3-4 GB"
    echo "  - 3B model:   ~6-8 GB"
    echo "  - 7B model:   ~14-16 GB"
    echo "  - 14B model:  ~28-32 GB"
    echo ""
    echo "Estimated total size (all models):"
    echo "  - INT8 all:  ~65-80 GB"
    echo "  - FP8 all:   ~65-80 GB"
    echo "  - Total:     ~130-160 GB"
}

# 下载 INT8 模型
download_int8_models() {
    local filter=$1  # qwen, llama, or empty for all
    
    for key in $(echo "${!INT8_MODELS[@]}" | tr ' ' '\n' | sort); do
        local model_info="${INT8_MODELS[$key]}"
        
        # 过滤
        if [ -n "$filter" ]; then
            case $filter in
                qwen)
                    [[ ! $key == qwen* ]] && continue
                    ;;
                llama)
                    [[ ! $key == llama* ]] && continue
                    ;;
            esac
        fi
        
        download_model "$key" "$model_info" "int8"
    done
}

# 下载 FP8 模型
download_fp8_models() {
    local filter=$1  # qwen, llama, or empty for all
    
    for key in $(echo "${!FP8_MODELS[@]}" | tr ' ' '\n' | sort); do
        local model_info="${FP8_MODELS[$key]}"
        
        # 过滤
        if [ -n "$filter" ]; then
            case $filter in
                qwen)
                    [[ ! $key == qwen* ]] && continue
                    ;;
                llama)
                    [[ ! $key == llama* ]] && continue
                    ;;
            esac
        fi
        
        download_model "$key" "$model_info" "fp8"
    done
}

# 下载指定模型
download_specific_model() {
    local model_key=$1
    
    # 检查是否在 INT8 模型列表中
    if [ -n "${INT8_MODELS[$model_key]}" ]; then
        download_model "$model_key" "${INT8_MODELS[$model_key]}" "int8"
        return 0
    fi
    
    # 检查是否在 FP8 模型列表中
    if [ -n "${FP8_MODELS[$model_key]}" ]; then
        download_model "$model_key" "${FP8_MODELS[$model_key]}" "fp8"
        return 0
    fi
    
    print_error "Model not found: $model_key"
    echo "Use --help to see available models"
    return 1
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    # 检查 hf CLI 是否安装（新版命令）
    if ! command -v hf &> /dev/null; then
        # 回退检查旧版命令
        if ! command -v huggingface-cli &> /dev/null; then
            print_error "HuggingFace CLI not installed"
            print_info "Please run: pip install -U huggingface_hub"
            exit 1
        fi
    fi
    
    # 解析参数
    local download_int8=false
    local download_fp8=false
    local filter_qwen=false
    local filter_llama=false
    local specific_model=""
    local check_only=false
    local show_sizes=false
    
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--all)
                download_int8=true
                download_fp8=true
                shift
                ;;
            -i|--int8)
                download_int8=true
                shift
                ;;
            -f|--fp8)
                download_fp8=true
                shift
                ;;
            -q|--qwen)
                filter_qwen=true
                shift
                ;;
            -l|--llama)
                filter_llama=true
                shift
                ;;
            -m|--model)
                specific_model="$2"
                shift 2
                ;;
            -c|--check)
                check_only=true
                shift
                ;;
            -s|--size)
                show_sizes=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Run '$0 --help' for help"
                exit 1
                ;;
        esac
    done
    
    # 显示大小信息
    if [ "$show_sizes" = true ]; then
        show_model_sizes
        exit 0
    fi
    
    # 检查模式
    if [ "$check_only" = true ]; then
        check_all_models
        exit 0
    fi
    
    # 设置目录
    setup_directories
    
    # 下载指定模型
    if [ -n "$specific_model" ]; then
        download_specific_model "$specific_model"
        exit $?
    fi
    
    # 确定过滤器
    local filter=""
    if [ "$filter_qwen" = true ] && [ "$filter_llama" = false ]; then
        filter="qwen"
    elif [ "$filter_llama" = true ] && [ "$filter_qwen" = false ]; then
        filter="llama"
    fi
    
    # 执行下载
    if [ "$download_int8" = true ]; then
        print_header "Starting download of INT8 (quantized.w8a8) models"
        download_int8_models "$filter"
    fi
    
    if [ "$download_fp8" = true ]; then
        print_header "Starting download of FP8 (FP8-dynamic) models"
        download_fp8_models "$filter"
    fi
    
    # 显示最终状态
    check_all_models
    
    print_success "Model download task completed!"
}

# 运行主程序
main "$@"
