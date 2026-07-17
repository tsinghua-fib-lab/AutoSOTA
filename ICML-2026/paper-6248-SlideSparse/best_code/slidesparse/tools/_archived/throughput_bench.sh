#!/bin/bash
# ============================================================================
# SlideSparse vLLM Throughput Benchmark 脚本
# ============================================================================
# 用于精确测试 W8A8 量化模型在不同 M (GEMM batch size) 下的 Prefill/Decode 性能
# 
# 核心设计思想：
#   - Prefill 测试: 控制 M_prefill = max_num_seqs × prompt_length，最小化 Decode 开销
#   - Decode 测试:  控制 M_decode = max_num_seqs，最小化 Prefill 开销
#   - 动态计算 max-model-len 以最大化 KV Cache 利用率 (Tight Fit 策略)
#   - 禁用 Chunked Prefill 以获得纯净的性能数据
#
# 使用方法:
#   ./throughput_bench.sh [选项]
#
# 示例:
#   ./throughput_bench.sh --model qwen2.5-0.5b-int8 --prefill --M 16,32,64,128,256
#   ./throughput_bench.sh --model llama3.2-1b-int8 --decode --M 1,2,4,8,16
#   ./throughput_bench.sh --model qwen2.5-0.5b-fp8 --prefill --M 16,32,64,128,256
#   ./throughput_bench.sh --model llama3.2-1b-fp8 --decode --M 1,2,4,8,16
#   ./throughput_bench.sh --all --prefill --M 16,256
#   ./throughput_bench.sh --all --decode --M 1,16
# ============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"

# 硬件信息 工具脚本路径
HW_INFO_UTILS_PY="${SCRIPT_DIR}/HW_info_utils.py"

# ============================================================================
# 全局配置参数
# ============================================================================

# 测试模式: "prefill" 或 "decode"
TEST_MODE="prefill"

# Prefill 测试配置
# M_prefill 列表: GEMM 的 M 维度 (= max_num_seqs × prompt_length)
M_LIST_PREFILL=(16 32 64 128 256 512 1024 2048 4096 8192 16384 32768)
N_PREFILL=128              # Prefill 重复次数，用于稳定统计

# Decode 测试配置
# M_decode 列表: GEMM 的 M 维度 (= max_num_seqs = batch_size)
M_LIST_DECODE=(1 2 4 8 16 32 48 64 80 96 112 128)
N_DECODE=512               # Decode 生成的 token 数 (= max_tokens)

# Prompt length 配置
PROMPT_LENGTH_CAP_PREFILL=1024   # Prefill 模式下 prompt_length 的上限 (考虑 RoPE 限制)
PROMPT_LENGTH_FIXED_DECODE=16    # Decode 模式下固定的 prompt_length

# max-model-len 计算的 Buffer
MODEL_LEN_BUFFER=128

# 日志级别控制 (减少 vLLM 输出)
# 可选值: DEBUG, INFO, WARNING, ERROR
VLLM_LOG_LEVEL="WARNING"

# GPU 设备编号 (逗号分隔，支持多 GPU)
# 单卡: GPU_ID="0"
# 多卡: GPU_ID="0,1" 或 GPU_ID="2,3"
# 与 TENSOR_PARALLEL_SIZE 配合使用
GPU_ID="0,1"

# GPU 内存利用率 (0.0-1.0)
# 默认 0.9，如果遇到内存不足可以降低到 0.85 或 0.8
GPU_MEMORY_UTILIZATION=0.8

# Tensor Parallelism 配置
# 默认 1 (单卡)，可设置为 2, 4, 8 等使用多卡并行
# 14B 模型在 16GB 显存卡上需要至少 TP=2
TENSOR_PARALLEL_SIZE=1

# torch.compile 控制
# 在不支持的架构上 (如 GB10 sm_121a) 需要禁用 torch.compile
# 0 = 禁用 (eager mode), 1-3 = 不同级别的编译优化
# 默认 "auto" 表示自动检测，在不支持的架构上自动禁用
TORCH_COMPILE_LEVEL="auto"

# ============================================================================
# 模型定义 (与 model_download.sh 保持一致)
# ============================================================================
# 格式: [简短key]="HF模型名|本地文件夹名"

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

print_subheader() {
    echo ""
    echo -e "${MAGENTA}------------------------------------------------------------${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}------------------------------------------------------------${NC}"
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

# 输出函数：同时输出到终端和日志文件
log_and_echo() {
    echo "$1" | tee -a "${LOG_FILE}"
}

# 按模型大小排序 (提取 XXb 中的数字进行数值排序)
# 输入: 模型 key 列表 (通过管道)
# 输出: 按模型大小排序后的列表
sort_by_model_size() {
    # 提取模型 key 中的数字 (如 qwen2.5-0.5b-int8 -> 0.5)
    # 使用 awk 进行数值排序
    awk -F'-' '{
        line = $0
        for (i=1; i<=NF; i++) {
            if (tolower($i) ~ /^[0-9.]+b$/) {
                size = $i
                gsub(/[bB]/, "", size)
                print size, line
                break
            }
        }
    }' | sort -t' ' -k1 -n | cut -d' ' -f2-
}

# 根据 TENSOR_PARALLEL_SIZE 和 GPU_ID 计算实际使用的 GPU 列表
# 输出: CUDA_VISIBLE_DEVICES 的值 (e.g., "0,1")
# 副作用: 设置 ACTUAL_TP_SIZE 为实际使用的 GPU 数量
get_gpu_devices_for_tp() {
    local tp_size=${TENSOR_PARALLEL_SIZE}
    local gpu_ids=${GPU_ID}
    
    # 单卡模式: 直接返回第一个 GPU
    if [[ ${tp_size} -le 1 ]]; then
        local first_gpu=$(echo "${gpu_ids}" | cut -d',' -f1)
        ACTUAL_TP_SIZE=1
        echo "${first_gpu}"
        return
    fi
    
    # 多卡模式: 解析 GPU_ID 列表
    IFS=',' read -ra gpu_array <<< "${gpu_ids}"
    local available_count=${#gpu_array[@]}
    
    if [[ ${available_count} -lt ${tp_size} ]]; then
        # GPU 数量不足，警告并使用所有可用的
        print_warning "GPU_ID specifies ${available_count} GPUs but TP=${tp_size} requested. Using all ${available_count} available GPUs." >&2
        ACTUAL_TP_SIZE=${available_count}
        echo "${gpu_ids}"
    elif [[ ${available_count} -gt ${tp_size} ]]; then
        # GPU 数量过多，取前 tp_size 个
        local selected_gpus=$(echo "${gpu_ids}" | cut -d',' -f1-${tp_size})
        print_warning "GPU_ID specifies ${available_count} GPUs but TP=${tp_size}. Using first ${tp_size}: ${selected_gpus}" >&2
        ACTUAL_TP_SIZE=${tp_size}
        echo "${selected_gpus}"
    else
        # 数量匹配
        ACTUAL_TP_SIZE=${tp_size}
        echo "${gpu_ids}"
    fi
}

# ============================================================================
# 硬件信息获取函数 (通过 Python 工具脚本)
# ============================================================================

# 获取硬件信息的特定字段
get_hw_field() {
    local field=$1
    python3 "${HW_INFO_UTILS_PY}" --field "${field}" 2>/dev/null || echo "unknown"
}

# 获取硬件信息 JSON
get_hw_info_json() {
    python3 "${HW_INFO_UTILS_PY}" --json 2>/dev/null
}

# 打印硬件信息表格
print_hw_info_table() {
    python3 "${HW_INFO_UTILS_PY}" --table 2>/dev/null
}

# 获取输出目录的 GPU 文件夹名称 (如 A100_cc80)
get_gpu_folder_name() {
    get_hw_field "folder_name"
}

# 显示帮助信息
show_help() {
    echo "SlideSparse vLLM Throughput Benchmark Script"
    echo ""
    echo "Usage: $0 [model options] [test mode] [param overrides]"
    echo ""
    echo "Model Options:"
    echo "  -a, --all              Test all models (INT8 + FP8)"
    echo "  -i, --int8             Test INT8 (quantized.w8a8) models only"
    echo "  -f, --fp8              Test FP8 (FP8-dynamic) models only"
    echo "  -q, --qwen             Test Qwen2.5 series only"
    echo "  -l, --llama            Test Llama3.2 series only"
    echo "  -m, --model NAME       Test specific model"
    echo "  -c, --check            Check downloaded model status"
    echo ""
    echo "Test Mode:"
    echo "  --prefill              Prefill test mode [default]"
    echo "                         M_prefill controls GEMM size, output_len=1 to minimize Decode"
    echo "  --decode               Decode test mode"
    echo "                         M_decode controls batch size, prompt_len=16 to minimize Prefill"
    echo ""
    echo "Param Overrides (optional):"
    echo "  --M LIST               Override M value list (comma-separated, e.g.: 16,32,64,128)"
    echo "  --N NUM                Override repeat count (Prefill: N_prefill, Decode: N_decode)"
    echo ""
    echo "Compilation Options:"
    echo "  --eager                Force eager mode (disable torch.compile)"
    echo "                         Required for unsupported GPU architectures like GB10 (sm_121a)"
    echo "  --compile              Force enable torch.compile (override auto-detection)"
    echo "                         [default: auto-detect based on GPU architecture]"
    echo ""
    echo "Hardware Options:"
    echo "  --tp NUM               Tensor parallelism size (default: 1)"
    echo "                         Use --tp 2 for 14B models on 16GB GPUs"
    echo ""
    echo "Other Options:"
    echo "  -h, --help             Show this help message"
    echo "  --dry-run              Show commands without running"
    echo ""
    echo "Available Models:"
    echo ""
    echo "  INT8 Models:"
    for key in "${!INT8_MODELS[@]}"; do
        echo "    - $key"
    done | sort
    echo ""
    echo "  FP8 Models:"
    for key in "${!FP8_MODELS[@]}"; do
        echo "    - $key"
    done | sort
    echo ""
    echo "============================================================"
    echo "Test Mode Details:"
    echo "============================================================"
    echo ""
    echo "[Prefill Test Mode]"
    echo "  Goal: Test Prefill performance at different M_prefill"
    echo "  Strategy: Test Prefill only, no Decode (output_len=1)"
    echo "  M_list (default): ${M_LIST_PREFILL[*]}"
    echo "  N_prefill (default): ${N_PREFILL}"
    echo ""
    echo "  Param calculation:"
    echo "    If M_prefill <= 1024: max_num_seqs=1, prompt_length=M_prefill"
    echo "    If M_prefill > 1024:  prompt_length=1024, max_num_seqs=M_prefill/1024"
    echo "    num_prompts = N_prefill x max_num_seqs"
    echo ""
    echo "[Decode Test Mode]"
    echo "  Goal: Test Decode performance at different M_decode (batch size)"
    echo "  Strategy: Run Prefill once, then N_decode Decode iterations"
    echo "  M_list (default): ${M_LIST_DECODE[*]}"
    echo "  N_decode (default): ${N_DECODE}"
    echo ""
    echo "  Param calculation:"
    echo "    M_decode = max_num_seqs (batch size)"
    echo "    prompt_length = ${PROMPT_LENGTH_FIXED_DECODE} (fixed)"
    echo "    num_prompts = max_num_seqs -> N_prefill = 1"
    echo "    max_tokens = N_decode"
    echo ""
    echo "============================================================"
    echo "Examples:"
    echo "============================================================"
    echo "  $0 --model qwen2.5-7b-fp8 --prefill"
    echo "  $0 --model qwen2.5-7b-fp8 --decode"
    echo "  $0 --all --prefill --M 16,32,64,128 --N 64"
    echo "  $0 --int8 --qwen --decode --dry-run"
}

# 检查模型状态
check_model_status() {
    local model_info=$1
    local local_dir_name=$(get_local_dir_name "$model_info")
    local local_dir="${CHECKPOINT_DIR}/${local_dir_name}"
    
    if [ -d "$local_dir" ] && [ -f "${local_dir}/config.json" ]; then
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
    
    if [ -d "$CHECKPOINT_DIR" ]; then
        local total_size=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
        print_info "Checkpoints directory size: ${total_size}"
    fi
}

# ============================================================================
# 参数计算函数
# ============================================================================

# 根据测试模式和 M 值计算所有测试参数
# 返回: prompt_length max_num_seqs num_prompts output_len max_model_len
calculate_test_params() {
    local m_value=$1
    
    local prompt_length
    local max_num_seqs
    local num_prompts
    local output_len
    local max_model_len
    local n_prefill_actual
    local n_decode_actual
    
    if [[ "$TEST_MODE" == "prefill" ]]; then
        # ==================== Prefill 测试参数计算 ====================
        # M_prefill = max_num_seqs × prompt_length
        # 目标: 最小化 Decode 阶段 (output_len=1)
        
        local m_prefill=${m_value}
        
        if [[ ${m_prefill} -le ${PROMPT_LENGTH_CAP_PREFILL} ]]; then
            # M_prefill <= 1024: 单个长 prompt
            prompt_length=${m_prefill}
            max_num_seqs=1
        else
            # M_prefill > 1024: 多个 1024 长度的 prompt
            prompt_length=${PROMPT_LENGTH_CAP_PREFILL}
            max_num_seqs=$((m_prefill / prompt_length))
        fi
        
        # Prefill 次数 = N_prefill, 总 prompt 数 = N_prefill × max_num_seqs
        num_prompts=$((N_PREFILL * max_num_seqs))
        output_len=1  # 最小化 Decode
        
        # max-model-len: Tight Fit 策略
        max_model_len=$((prompt_length + output_len + MODEL_LEN_BUFFER))
        
        n_prefill_actual=${N_PREFILL}
        n_decode_actual=0
        
    else
        # ==================== Decode 测试参数计算 ====================
        # M_decode = max_num_seqs (batch size)
        # 目标: 最小化 Prefill 阶段 (N_prefill=1, 短 prompt)
        
        local m_decode=${m_value}
        
        prompt_length=${PROMPT_LENGTH_FIXED_DECODE}  # 固定为 16
        max_num_seqs=${m_decode}
        
        # N_prefill = 1: num_prompts = max_num_seqs
        num_prompts=${max_num_seqs}
        output_len=${N_DECODE}  # Decode 次数 = max_tokens
        
        # max-model-len: Tight Fit 策略
        max_model_len=$((prompt_length + output_len + MODEL_LEN_BUFFER))
        
        n_prefill_actual=1
        n_decode_actual=${N_DECODE}
    fi
    
    # 输出所有计算结果 (空格分隔)
    echo "${prompt_length} ${max_num_seqs} ${num_prompts} ${output_len} ${max_model_len} ${n_prefill_actual} ${n_decode_actual}"
}

# ============================================================================
# 核心测试函数
# ============================================================================

# 运行单个 M 值的吞吐测试
run_single_m_test() {
    local model_key=$1
    local model_info=$2
    local quant_type=$3
    local m_value=$4
    
    local local_dir_name=$(get_local_dir_name "$model_info")
    local model_path="${CHECKPOINT_DIR}/${local_dir_name}"
    
    # 检查模型是否存在
    if [ ! -d "$model_path" ] || [ ! -f "${model_path}/config.json" ]; then
        print_warning "Model not downloaded, skipping: ${local_dir_name}"
        return 1
    fi
    
    # 计算测试参数
    read prompt_length max_num_seqs num_prompts output_len max_model_len n_prefill n_decode <<< $(calculate_test_params ${m_value})
    
    # 计算 M_prefill 和 M_decode 用于显示
    local m_prefill=$((max_num_seqs * prompt_length))
    local m_decode=${max_num_seqs}
    
    # 计算 RoPE 位置范围
    local max_position=$((prompt_length + output_len - 1))
    
    echo ""
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                    Test Parameters                          │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Model: ${local_dir_name}"
    echo "│ Mode:  ${TEST_MODE} test"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ GEMM M Dimension:"
    echo "│   M_prefill     = ${m_prefill} (= ${max_num_seqs} x ${prompt_length})"
    echo "│   M_decode      = ${m_decode}"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ vLLM Args:"
    echo "│   --input-len       = ${prompt_length}"
    echo "│   --output-len      = ${output_len}"
    echo "│   --num-prompts     = ${num_prompts}"
    echo "│   --max-num-seqs    = ${max_num_seqs}"
    echo "│   --max-model-len   = ${max_model_len}"
    if [[ ${TENSOR_PARALLEL_SIZE} -gt 1 ]]; then
    echo "│   --tensor-parallel = ${TENSOR_PARALLEL_SIZE}"
    fi
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Iterations:"
    echo "│   N_prefill     = ${n_prefill}"
    echo "│   N_decode      = ${n_decode}"
    if [[ "${ENFORCE_EAGER}" == "true" ]]; then
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Compile Mode:"
    echo "│   --enforce-eager   = true (torch.compile disabled)"
    fi
    echo "└─────────────────────────────────────────────────────────────┘"
    echo ""
    
    # 结果文件
    local result_file="${RESULT_JSON_DIR}/${local_dir_name}_M${m_value}.json"
    
    # 动态计算 max-num-batched-tokens (= max_num_seqs × max_model_len，确保禁用 chunking)
    local max_num_batched_tokens=$((max_num_seqs * max_model_len))
    
    # 计算实际使用的 GPU 列表 (先调用函数设置 ACTUAL_TP_SIZE)
    ACTUAL_TP_SIZE=1  # 初始化
    local gpu_devices=$(get_gpu_devices_for_tp)
    # 重新计算 ACTUAL_TP_SIZE (因为子 shell 中的设置不会传回)
    IFS=',' read -ra _gpu_arr <<< "${gpu_devices}"
    ACTUAL_TP_SIZE=${#_gpu_arr[@]}
    
    # 构建环境变量前缀
    local env_prefix="CUDA_VISIBLE_DEVICES=${gpu_devices} VLLM_LOGGING_LEVEL=${VLLM_LOG_LEVEL}"
    
    # 构建 eager mode 参数 (用于禁用 torch.compile)
    local eager_flag=""
    if [[ "${ENFORCE_EAGER:-false}" == "true" ]]; then
        eager_flag="--enforce-eager"
    fi
    
    # 构建 TP 参数
    local tp_flag=""
    if [[ ${ACTUAL_TP_SIZE} -gt 1 ]]; then
        tp_flag="--tensor-parallel-size ${ACTUAL_TP_SIZE}"
    fi
    
    # 构建命令 (设置环境变量减少日志输出)
    local cmd="${env_prefix} vllm bench throughput \
        --model ${model_path} \
        --dataset-name random \
        --input-len ${prompt_length} \
        --output-len ${output_len} \
        --num-prompts ${num_prompts} \
        --max-num-seqs ${max_num_seqs} \
        --max-model-len ${max_model_len} \
        --max-num-batched-tokens ${max_num_batched_tokens} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        ${tp_flag} \
        ${eager_flag} \
        --disable-log-stats \
        --output-json ${result_file}"
    
    # 记录到日志
    echo "" >> "${LOG_FILE}"
    echo "========== M=${m_value} ==========" >> "${LOG_FILE}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}"
    echo "Params: prompt_len=${prompt_length}, output_len=${output_len}, num_prompts=${num_prompts}, max_num_seqs=${max_num_seqs}" >> "${LOG_FILE}"
    echo "Command: ${cmd}" >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
    
    # Dry-run 模式
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Command to execute:"
        echo "$cmd"
        echo ""
        # 生成模拟结果用于 dry-run 测试
        echo '{"requests_per_second": 0, "tokens_per_second": 0, "elapsed_time": 0, "num_requests": 0}' > "${result_file}"
        return 0
    fi
    
    # 执行测试
    print_info "Starting test..."
    local start_time=$(date +%s)
    local exit_code=0
    local output_tmp=$(mktemp)
    
    # 运行并捕获输出到临时文件和日志
    # 使用 sed 过滤 ANSI 转义码，避免日志文件出现乱码
    eval $cmd 2>&1 | tee "$output_tmp" | sed 's/\x1b\[[0-9;]*m//g' >> "${LOG_FILE}" || exit_code=$?
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $exit_code -eq 0 ]] && [[ -f "$result_file" ]]; then
        rm -f "$output_tmp"
        print_success "Test completed! Duration: ${duration}s"
        
        # 解析并显示结果
        echo ""
        echo -e "${GREEN}Test Results:${NC}"
        python3 -c "
import json
with open('${result_file}', 'r') as f:
    data = json.load(f)
    req_per_s = data.get('requests_per_second', 0)
    tok_per_s = data.get('tokens_per_second', 0)
    elapsed = data.get('elapsed_time', 0)
    num_req = data.get('num_requests', 0)
    
    print(f'  Requests/s:   {req_per_s:.2f}')
    print(f'  Tokens/s:     {tok_per_s:.2f}')
    print(f'  Total Reqs:   {num_req}')
    print(f'  Elapsed:      {elapsed:.2f}s')
    
    # 计算单次操作的性能
    n_prefill = ${n_prefill}
    n_decode = ${n_decode}
    
    if '${TEST_MODE}' == 'prefill' and n_prefill > 0:
        # Prefill 测试: 计算每次 Prefill 的 tokens/s
        prefill_tokens = ${m_prefill}
        total_prefill_tokens = prefill_tokens * n_prefill
        if elapsed > 0:
            prefill_tps = total_prefill_tokens / elapsed
            print(f'')
            print(f'  [Prefill Analysis]')
            print(f'  Total Prefill Tokens: {total_prefill_tokens}')
            print(f'  Prefill Tokens/s:     {prefill_tps:.2f}')
    elif '${TEST_MODE}' == 'decode' and n_decode > 0:
        # Decode 测试: 计算每次 Decode 的 tokens/s  
        decode_tokens = ${m_decode} * n_decode
        if elapsed > 0:
            decode_tps = decode_tokens / elapsed
            print(f'')
            print(f'  [Decode Analysis]')
            print(f'  Total Decode Tokens:  {decode_tokens}')
            print(f'  Decode Tokens/s:      {decode_tps:.2f}')
"
        return 0
    else
        print_error "Test failed: M=${m_value} (exit code: ${exit_code})"
        echo "ERROR: Test failed for M=${m_value}" >> "${LOG_FILE}"
        
        # 提取错误信息并记录
        local error_snippet=""
        if [[ -f "$output_tmp" ]]; then
            # 提取最后 20 行作为错误摘要
            error_snippet=$(tail -20 "$output_tmp" 2>/dev/null || echo "Unable to read error output")
            echo "ERROR OUTPUT:" >> "${LOG_FILE}"
            echo "${error_snippet}" >> "${LOG_FILE}"
            
            # 打印错误摘要到终端
            echo ""
            echo -e "${RED}─── Error Output (last 10 lines) ───${NC}"
            tail -10 "$output_tmp" 2>/dev/null || echo "Unable to read error output"
            echo -e "${RED}─────────────────────────────────${NC}"
        fi
        
        rm -f "$output_tmp"
        return 1
    fi
}

# 测试单个模型的所有 M 值
# 返回值: 0=成功, 1=普通错误, 2=精度不支持(应跳过该精度的其他模型)
run_model_benchmark() {
    local model_key=$1
    local model_info=$2
    local quant_type=$3
    
    local local_dir_name=$(get_local_dir_name "$model_info")
    local model_path="${CHECKPOINT_DIR}/${local_dir_name}"
    
    # 检查模型是否存在
    if [ ! -d "$model_path" ] || [ ! -f "${model_path}/config.json" ]; then
        print_warning "Model not downloaded, skipping: ${local_dir_name}"
        return 1
    fi
    
    print_header "Testing Model: ${local_dir_name}"
    
    local total_tests=${#M_LIST[@]}
    local current_test=0
    local failed_tests=0
    local first_error_encountered=false
    
    for m_value in "${M_LIST[@]}"; do
        current_test=$((current_test + 1))
        echo ""
        echo "=============================================="
        echo "[${current_test}/${total_tests}] Testing M=${m_value}"
        echo "=============================================="
        
        run_single_m_test "$model_key" "$model_info" "$quant_type" "$m_value"
        local test_result=$?
        
        if [[ $test_result -ne 0 ]]; then
            failed_tests=$((failed_tests + 1))
            
            # 统一试错策略: 首次错误时跳过该模型的剩余测试
            # 并返回特殊退出码 2 表示应跳过该精度的所有后续模型
            if [[ "$first_error_encountered" == false ]]; then
                first_error_encountered=true
                echo ""
                print_warning "=================================================="
                print_warning "⚠️  Error encountered for ${quant_type^^} model: ${local_dir_name}"
                print_warning "    Skipping remaining tests for this model."
                print_warning "    Will also skip all other ${quant_type^^} models."
                print_warning "=================================================="
                echo "SKIP: ${quant_type^^} test failed, skipping remaining ${quant_type^^} tests" >> "${LOG_FILE}"
                
                # 跳过该模型的剩余测试，并返回 2 表示应跳过该精度
                return 2
            fi
        fi
    done
    
    # 生成该模型的 CSV 结果
    generate_model_csv "$local_dir_name"
    
    echo ""
    print_info "Model ${local_dir_name} completed: ${total_tests} tests, ${failed_tests} failed"
    return 0
}

# 生成单个模型的 CSV 结果
generate_model_csv() {
    local model_name=$1
    local csv_file="${OUTPUT_DIR}/${model_name}_${TEST_MODE}.csv"
    
    echo ""
    print_subheader "Generating CSV: ${model_name}"
    
    # CSV 表头
    if [[ "$TEST_MODE" == "prefill" ]]; then
        echo "M_prefill,prompt_len,max_num_seqs,num_prompts,N_prefill,requests_per_s,tokens_per_s,elapsed_time_s" > "${csv_file}"
    else
        echo "M_decode,prompt_len,max_num_seqs,num_prompts,N_decode,output_len,requests_per_s,tokens_per_s,elapsed_time_s" > "${csv_file}"
    fi
    
    # 遍历所有 M 值的结果文件
    for m_value in "${M_LIST[@]}"; do
        local result_file="${RESULT_JSON_DIR}/${model_name}_M${m_value}.json"
        
        if [[ -f "$result_file" ]]; then
            # 重新计算参数用于 CSV
            read prompt_length max_num_seqs num_prompts output_len max_model_len n_prefill n_decode <<< $(calculate_test_params ${m_value})
            
            # 解析 JSON 结果
            local result=$(python3 -c "
import json
try:
    with open('${result_file}', 'r') as f:
        data = json.load(f)
        req_s = data.get('requests_per_second', 0)
        tok_s = data.get('tokens_per_second', 0)
        elapsed = data.get('elapsed_time', 0)
        print(f'{req_s:.4f},{tok_s:.4f},{elapsed:.4f}')
except:
    print('0,0,0')
")
            
            if [[ "$TEST_MODE" == "prefill" ]]; then
                echo "${m_value},${prompt_length},${max_num_seqs},${num_prompts},${n_prefill},${result}" >> "${csv_file}"
            else
                echo "${m_value},${prompt_length},${max_num_seqs},${num_prompts},${n_decode},${output_len},${result}" >> "${csv_file}"
            fi
        fi
    done
    
    print_success "CSV saved to: ${csv_file}"
    
    # 显示 CSV 预览
    echo ""
    echo "Preview:"
    echo "----------------------------------------------"
    cat "${csv_file}"
    echo "----------------------------------------------"
}

# ============================================================================
# 量化格式支持检测函数 (原生支持预拦截)
# ============================================================================
# 设计思路:
# - INT8 原生支持: CC >= 8.0 (Ampere+), 拦截 V100 等老卡
# - FP8 原生支持:  CC >= 8.9 (Ada/Hopper+), 拦截 A100 等不支持原生 FP8 的卡
# 这两个函数是对称的，都是为了避免 vLLM 的 fallback 机制污染 Benchmark 数据
# ============================================================================

# INT8 预拦截: 检测 GPU 是否支持原生 INT8 GEMM (CC >= 8.0)
check_int8_support() {
    python3 "${HW_INFO_UTILS_PY}" --check-int8 2>/dev/null
    return $?
}

# FP8 预拦截: 检测 GPU 是否支持原生 FP8 GEMM (CC >= 8.9)
check_fp8_support() {
    python3 "${HW_INFO_UTILS_PY}" --check-fp8 2>/dev/null
    return $?
}

# ============================================================================
# 批量测试函数
# ============================================================================
# 统一试错机制:
# 1. 预拦截: 检测硬件原生支持 (INT8: CC>=8.0, FP8: CC>=8.9)
# 2. 运行时试错: 如果任何错误发生，跳过该精度的所有后续模型
# ============================================================================

# 测试 INT8 模型
test_int8_models() {
    local filter=$1
    
    # INT8 预检测拦截: 如果 GPU 不支持原生 INT8 (V100 等 CC < 8.0)
    if ! check_int8_support; then
        print_warning "Skipping all INT8 model tests (GPU does not support native INT8 Tensor Core GEMM)"
        return 0
    fi
    
    local skip_remaining=false
    
    for key in $(echo "${!INT8_MODELS[@]}" | tr ' ' '\n' | sort_by_model_size); do
        # 如果之前的测试失败，跳过剩余的 INT8 模型
        if [[ "$skip_remaining" == true ]]; then
            print_warning "Skipping INT8 model ${key} due to previous error"
            continue
        fi
        
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
        
        run_model_benchmark "$key" "$model_info" "int8"
        local result=$?
        
        # 试错机制: 如果返回 2 表示应跳过该精度的所有后续模型
        if [[ $result -eq 2 ]]; then
            skip_remaining=true
            echo ""
            print_warning "INT8 test failed, skipping remaining INT8 models"
        fi
    done
}

# 测试 FP8 模型
test_fp8_models() {
    local filter=$1
    
    # FP8 预检测拦截: 如果 GPU 不支持原生 FP8 (A100 等 CC < 8.9)
    if ! check_fp8_support; then
        print_warning "Skipping all FP8 model tests (GPU does not support native FP8 GEMM)"
        return 0
    fi
    
    local skip_remaining=false
    
    for key in $(echo "${!FP8_MODELS[@]}" | tr ' ' '\n' | sort_by_model_size); do
        # 如果之前的测试失败，跳过剩余的 FP8 模型
        if [[ "$skip_remaining" == true ]]; then
            print_warning "Skipping FP8 model ${key} due to previous error"
            continue
        fi
        
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
        
        run_model_benchmark "$key" "$model_info" "fp8"
        local result=$?
        
        # 试错机制: 如果返回 2 表示应跳过该精度的所有后续模型
        if [[ $result -eq 2 ]]; then
            skip_remaining=true
            echo ""
            print_warning "FP8 test failed, skipping remaining FP8 models"
        fi
    done
}

# 测试指定模型
test_specific_model() {
    local model_key=$1
    
    # 检查是否在 INT8 模型列表中
    if [ -n "${INT8_MODELS[$model_key]}" ]; then
        # INT8 预检测拦截
        if ! check_int8_support; then
            print_error "Skipping INT8 model ${model_key} (GPU does not support native INT8 Tensor Core GEMM)"
            return 1
        fi
        run_model_benchmark "$model_key" "${INT8_MODELS[$model_key]}" "int8"
        return $?
    fi
    
    # 检查是否在 FP8 模型列表中
    if [ -n "${FP8_MODELS[$model_key]}" ]; then
        # FP8 预检测拦截
        if ! check_fp8_support; then
            print_error "Skipping FP8 model ${model_key} (GPU does not support native FP8 GEMM)"
            return 1
        fi
        run_model_benchmark "$model_key" "${FP8_MODELS[$model_key]}" "fp8"
        return $?
    fi
    
    print_error "Model not found: $model_key"
    echo "Use --help to see available models"
    return 1
}

# ============================================================================
# 信号处理和清理
# ============================================================================

cleanup() {
    echo ""
    echo "=============================================="
    echo "Test interrupted!"
    if [[ -n "${OUTPUT_DIR:-}" ]]; then
        echo "Results directory: ${OUTPUT_DIR}"
    fi
    echo "=============================================="
    exit 130
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# 主程序
# ============================================================================

main() {
    # 检查 vllm 是否安装
    if ! command -v vllm &> /dev/null; then
        print_error "vllm not installed or not in PATH"
        print_info "Please ensure vLLM is installed correctly"
        exit 1
    fi
    
    # 解析参数
    local test_int8=false
    local test_fp8=false
    local filter_qwen=false
    local filter_llama=false
    local specific_model=""
    local check_only=false
    local custom_m_list=""
    local custom_n=""
    DRY_RUN=false
    
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--all)
                test_int8=true
                test_fp8=true
                shift
                ;;
            -i|--int8)
                test_int8=true
                shift
                ;;
            -f|--fp8)
                test_fp8=true
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
            --prefill)
                TEST_MODE="prefill"
                shift
                ;;
            --decode)
                TEST_MODE="decode"
                shift
                ;;
            --M)
                custom_m_list="$2"
                shift 2
                ;;
            --N)
                custom_n="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --eager)
                # 强制使用 eager mode (禁用 torch.compile)
                TORCH_COMPILE_LEVEL="0"
                shift
                ;;
            --compile)
                # 强制启用 torch.compile (覆盖自动检测)
                TORCH_COMPILE_LEVEL="force"
                shift
                ;;
            --tp)
                # Tensor parallelism size
                TENSOR_PARALLEL_SIZE="$2"
                shift 2
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
    
    # 检查模式
    if [ "$check_only" = true ]; then
        check_all_models
        exit 0
    fi
    
    # ========================================================================
    # 自动检测是否需要 eager mode (针对不支持的 GPU 架构如 GB10)
    # ========================================================================
    ENFORCE_EAGER=false
    if [[ "${TORCH_COMPILE_LEVEL}" == "auto" ]]; then
        # 调用 Python 工具检测 (只取 stdout，忽略 PyTorch 警告)
        local needs_eager=$(python3 "${HW_INFO_UTILS_PY}" --check-triton-support 2>/dev/null | head -1)
        if [[ "${needs_eager}" == "needs_eager" ]]; then
            print_warning "Detected unsupported GPU architecture for torch.compile (e.g., GB10 sm_121a)"
            print_warning "Automatically enabling eager mode (--enforce-eager)"
            ENFORCE_EAGER=true
        fi
    elif [[ "${TORCH_COMPILE_LEVEL}" == "0" ]]; then
        print_info "Using eager mode (torch.compile disabled via --enforce-eager)"
        ENFORCE_EAGER=true
    elif [[ "${TORCH_COMPILE_LEVEL}" == "force" ]]; then
        print_info "Force enabling torch.compile (ignoring architecture compatibility)"
        ENFORCE_EAGER=false
    fi
    
    # 根据测试模式选择默认 M_LIST
    if [[ "$TEST_MODE" == "prefill" ]]; then
        M_LIST=("${M_LIST_PREFILL[@]}")
        PHASE_LABEL="Prefill"
    else
        M_LIST=("${M_LIST_DECODE[@]}")
        PHASE_LABEL="Decode"
    fi
    
    # 应用自定义 M 列表
    if [ -n "$custom_m_list" ]; then
        IFS=',' read -ra M_LIST <<< "$custom_m_list"
    fi
    
    # 应用自定义 N
    if [ -n "$custom_n" ]; then
        if [[ "$TEST_MODE" == "prefill" ]]; then
            N_PREFILL=$custom_n
        else
            N_DECODE=$custom_n
        fi
    fi
    
    # 获取硬件信息用于目录命名
    GPU_FOLDER_NAME=$(get_gpu_folder_name)
    if [[ "${GPU_FOLDER_NAME}" == "unknown" ]] || [[ -z "${GPU_FOLDER_NAME}" ]]; then
        print_warning "Unable to detect GPU info, using 'unknown_gpu' as folder name"
        GPU_FOLDER_NAME="unknown_gpu"
    fi
    
    # 设置输出目录
    # 目录结构: throughput_bench_results/{prefill|decode}/{GPU_ccXX}/{timestamp}/
    #           ├── result_json/  (JSON结果文件)
    #           ├── benchmark.log
    #           └── {Model}_{mode}.csv (多个CSV文件)
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="${SCRIPT_DIR}/throughput_bench_results/${TEST_MODE}/${GPU_FOLDER_NAME}/${TIMESTAMP}"
    mkdir -p "${OUTPUT_DIR}"
    LOG_FILE="${OUTPUT_DIR}/benchmark.log"
    RESULT_JSON_DIR="${OUTPUT_DIR}/result_json"
    mkdir -p "${RESULT_JSON_DIR}"
    
    # 显示配置信息
    print_header "SlideSparse vLLM Throughput Benchmark"
    
    # 显示硬件信息
    echo ""
    print_hw_info_table
    echo ""
    
    echo "Test Mode: ${TEST_MODE} (testing ${PHASE_LABEL} phase)"
    echo ""
    echo "User Config:"
    if [[ "$TEST_MODE" == "prefill" ]]; then
        echo "  M_list (M_prefill): ${M_LIST[*]}"
        echo "  N_prefill: ${N_PREFILL}"
    else
        echo "  M_list (M_decode): ${M_LIST[*]}"
        echo "  N_decode: ${N_DECODE}"
    fi
    echo ""
    echo "Auto-calculated Params:"
    if [[ "$TEST_MODE" == "prefill" ]]; then
        echo "  prompt_length:     dynamic (<=1024: = M_prefill, else = 1024)"
        echo "  max_num_seqs:      dynamic (<=1024: =1, else = M_prefill / 1024)"
        echo "  M_decode:          dynamic (= max_num_seqs)"
        echo "  num_prompts:       dynamic (= N_prefill x max_num_seqs)"
        echo "  output_len:        1 (no Decode phase)"
        echo ""
        echo "  Verify: For M_prefill=${M_LIST[0]}:"
        if [[ ${M_LIST[0]} -le $PROMPT_LENGTH_CAP_PREFILL ]]; then
            echo "    prompt_length = M_prefill = ${M_LIST[0]}"
            echo "    max_num_seqs  = 1"
        else
            echo "    prompt_length = 1024"
            echo "    max_num_seqs  = M_prefill / 1024 = $((M_LIST[0] / PROMPT_LENGTH_CAP_PREFILL))"
        fi
    else
        echo "  prompt_length (fixed): ${PROMPT_LENGTH_FIXED_DECODE}"
        echo "  M_prefill:         dynamic (= ${PROMPT_LENGTH_FIXED_DECODE} x M_decode)"
        echo "  max_num_seqs:      dynamic (= M_decode)"
        echo "  num_prompts:       dynamic (= M_decode)"
        echo "  output_len:        ${N_DECODE} (= N_decode)"
        echo ""
        echo "  Verify: For M_decode=${M_LIST[0]}:"
        echo "    M_prefill = ${PROMPT_LENGTH_FIXED_DECODE} x ${M_LIST[0]} = $((PROMPT_LENGTH_FIXED_DECODE * M_LIST[0]))"
        echo "    num_prompts = M_decode = ${M_LIST[0]}"
        echo "    N_prefill = 1"
        echo "    N_decode = output_len = ${N_DECODE}"
    fi
    echo ""
    echo "Advanced Params:"
    echo "  max-num-batched-tokens: dynamic (= max_num_seqs x max_model_len, disables Chunked Prefill)"
    echo "  max-model-len:          dynamic (Tight Fit: input + output + ${MODEL_LEN_BUFFER})"
    echo "  vLLM log level:         ${VLLM_LOG_LEVEL}"
    echo ""
    echo "Output Dir: ${OUTPUT_DIR}"
    echo "Log File:   ${LOG_FILE}"
    echo "=============================================="
    
    # 将配置信息写入日志文件头部
    {
        echo "=============================================="
        echo "SlideSparse vLLM Throughput Benchmark Log"
        echo "=============================================="
        echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Test Mode: ${TEST_MODE} (testing ${PHASE_LABEL} phase)"
        echo ""
        echo "Hardware Information:"
        print_hw_info_table
        echo ""
        echo "User Config:"
        if [[ "$TEST_MODE" == "prefill" ]]; then
            echo "  M_list (M_prefill): ${M_LIST[*]}"
            echo "  N_prefill: ${N_PREFILL}"
        else
            echo "  M_list (M_decode): ${M_LIST[*]}"
            echo "  N_decode: ${N_DECODE}"
        fi
        echo "  max-num-batched-tokens: dynamic (= max_num_seqs x max_model_len)"
        echo "  vLLM log level: ${VLLM_LOG_LEVEL}"
        echo "=============================================="
        echo ""
    } > "${LOG_FILE}"
    
    # 执行测试
    if [ -n "$specific_model" ]; then
        # 测试指定模型
        test_specific_model "$specific_model"
    else
        # 确定过滤器
        local filter=""
        if [ "$filter_qwen" = true ] && [ "$filter_llama" = false ]; then
            filter="qwen"
        elif [ "$filter_llama" = true ] && [ "$filter_qwen" = false ]; then
            filter="llama"
        fi
        
        # 批量测试
        if [ "$test_int8" = true ]; then
            print_header "Testing INT8 (quantized.w8a8) models"
            test_int8_models "$filter"
        fi
        
        if [ "$test_fp8" = true ]; then
            print_header "Testing FP8 (FP8-dynamic) models"
            test_fp8_models "$filter"
        fi
    fi
    
    echo ""
    print_header "Benchmark Completed!"
    echo "Results directory: ${OUTPUT_DIR}"
    echo "  ├── benchmark.log"
    echo "  ├── result_json/  (JSON result files)"
    echo "  └── *.csv         (CSV summary files)"
    echo "=============================================="
}

# 运行主程序
main "$@"
