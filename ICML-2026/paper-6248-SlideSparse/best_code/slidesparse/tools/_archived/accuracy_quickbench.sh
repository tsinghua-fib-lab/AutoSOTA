#!/bin/bash
# ============================================================================
# SlideSparse vLLM Accuracy Quick Benchmark 脚本
# ============================================================================
# 用于快速测试 W8A8 量化模型的精度，使用 vllm run-batch 命令进行推理
# 
# 核心设计思想：
#   - 使用预定义的 prompt 文件进行批量推理
#   - 固定输出长度 (max_tokens) 以确保测试效率
#   - 保留 FP8/INT8 硬件支持检测和回退机制
#   - 输出结果以 JSON 格式保存，便于人工检查
#
# 使用方法:
#   ./accuracy_quickbench.sh [选项]
#
# 示例:
#   ./accuracy_quickbench.sh --model qwen2.5-0.5b-int8
#   ./accuracy_quickbench.sh --model llama3.2-1b-fp8
#   ./accuracy_quickbench.sh --all
#   ./accuracy_quickbench.sh --int8 --qwen
#   ./accuracy_quickbench.sh --fp8 --llama
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

# Prompt 输入文件路径
PROMPT_INPUT_FILE="${SCRIPT_DIR}/accuracy_quickbench_prompts.jsonl"

# ============================================================================
# 全局配置参数
# ============================================================================

# 输出长度配置

MAX_OUTPUT_TOKENS=64

# max-model-len 配置
# 需要足够容纳 prompt + output
MAX_MODEL_LEN=512

# 日志级别控制 (减少 vLLM 输出)
# 可选值: DEBUG, INFO, WARNING, ERROR
VLLM_LOG_LEVEL="WARNING"

# GPU 设备编号 (逗号分隔，支持多 GPU)
# 单卡: GPU_ID="0"
# 多卡: GPU_ID="0,1" 或 GPU_ID="2,3"
# 与 TENSOR_PARALLEL_SIZE 配合使用
GPU_ID="0,1"

# GPU 内存利用率 (0.0-1.0)
# 默认 0.8，如果遇到内存不足可以降低
GPU_MEMORY_UTILIZATION=0.8

# Tensor Parallelism 配置
# 默认 1 (单卡)，可设置为 2, 4, 8 等使用多卡并行
# 14B 模型在 16GB 显存卡上需要至少 TP=2
TENSOR_PARALLEL_SIZE=2

# torch.compile 控制
# 在不支持的架构上 (如 GB10 sm_121a) 需要禁用 torch.compile
# 0 = 禁用 (eager mode), 1-3 = 不同级别的编译优化
# 默认 "auto" 表示自动检测，在不支持的架构上自动禁用
TORCH_COMPILE_LEVEL="auto"

# Temperature 配置 (用于生成采样)
TEMPERATURE=0.0

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
    echo "SlideSparse vLLM Accuracy Quick Benchmark Script"
    echo ""
    echo "Usage: $0 [model options] [other options]"
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
    echo "Output Options:"
    echo "  --max-tokens NUM       Maximum output tokens (default: ${MAX_OUTPUT_TOKENS})"
    echo "  --temperature NUM      Sampling temperature (default: ${TEMPERATURE})"
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
    echo "Test Details:"
    echo "============================================================"
    echo ""
    echo "This script performs accuracy quick benchmarks using vllm run-batch."
    echo "It reads prompts from: ${PROMPT_INPUT_FILE}"
    echo "And outputs results to: accuracy_quickbench_results/{GPU_NAME}/{timestamp}/"
    echo ""
    echo "Output Format:"
    echo "  For each test run, a timestamped folder is created containing:"
    echo "  - {Model}.json           : Raw JSON output from vllm run-batch"
    echo "  - {Model}_responses.txt  : Extracted Q&A pairs for easy reading"
    echo "  - benchmark.log          : Execution log"
    echo ""
    echo "============================================================"
    echo "Examples:"
    echo "============================================================"
    echo "  $0 --model qwen2.5-0.5b-int8"
    echo "  $0 --model llama3.2-1b-fp8"
    echo "  $0 --all"
    echo "  $0 --int8 --qwen"
    echo "  $0 --fp8 --llama --dry-run"
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
# 核心测试函数
# ============================================================================

# 从 JSON 输出文件提取回答内容到 txt 文件
extract_responses_to_txt() {
    local json_file=$1
    local txt_file=$2
    
    if [[ ! -f "$json_file" ]]; then
        print_warning "JSON file not found: ${json_file}"
        return 1
    fi
    
    print_info "Extracting responses to: ${txt_file}"
    
    python3 << PYEOF
import json
import sys

json_file = "${json_file}"
txt_file = "${txt_file}"
prompts_file = "${PROMPT_INPUT_FILE}"

# 先读取原始 prompts 文件构建 custom_id -> prompt 的映射
prompts_map = {}
try:
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                cid = entry.get('custom_id', '')
                body = entry.get('body', {})
                msgs = body.get('messages', [])
                if msgs:
                    prompts_map[cid] = msgs[0].get('content', 'N/A')
            except:
                pass
except:
    pass

try:
    with open(json_file, 'r', encoding='utf-8') as f_in, open(txt_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, 1):
            try:
                entry = json.loads(line.strip())
                custom_id = entry.get('custom_id', f'req-{line_num:03d}')
                
                # 从 prompts_map 获取原始 prompt
                prompt = prompts_map.get(custom_id, 'N/A')
                
                # 获取回答内容
                response = entry.get('response', {})
                body = response.get('body', {})
                choices = body.get('choices', [])
                
                if choices and len(choices) > 0:
                    content = choices[0].get('message', {}).get('content', '[No content]')
                else:
                    error = entry.get('error', {})
                    if error:
                        content = f"[Error: {error}]"
                    else:
                        content = "[No response]"
                
                f_out.write(f"=== {custom_id} ===\\n")
                f_out.write(f"Q: {prompt}\\n")
                f_out.write(f"A: {content}\\n")
                f_out.write("\\n")
                
            except json.JSONDecodeError as e:
                f_out.write(f"=== Line {line_num} ===\\n")
                f_out.write(f"[JSON Parse Error: {e}]\\n\\n")
    
    print(f"Successfully extracted {line_num} responses")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

    if [[ $? -eq 0 ]]; then
        print_success "Responses extracted to: ${txt_file}"
    else
        print_warning "Failed to extract responses"
    fi
}

# 运行单个模型的精度测试
run_accuracy_test() {
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
    
    # 检查 prompt 输入文件是否存在
    if [ ! -f "${PROMPT_INPUT_FILE}" ]; then
        print_error "Prompt input file not found: ${PROMPT_INPUT_FILE}"
        return 1
    fi
    
    print_header "Testing Model: ${local_dir_name}"
    
    # 输出文件（目录已包含时间戳，文件名只用模型名）
    local output_file="${OUTPUT_DIR}/${local_dir_name}.json"
    local response_txt="${OUTPUT_DIR}/${local_dir_name}_responses.txt"
    
    echo ""
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                    Test Parameters                          │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Model:           ${local_dir_name}"
    echo "│ Quant Type:      ${quant_type^^}"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Input File:      ${PROMPT_INPUT_FILE}"
    echo "│ Output File:     ${output_file}"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ vLLM Args:"
    echo "│   --max-model-len   = ${MAX_MODEL_LEN}"
    echo "│   --temperature     = ${TEMPERATURE}"
    echo "│   --max-tokens      = ${MAX_OUTPUT_TOKENS} (via generation-config)"
    if [[ ${TENSOR_PARALLEL_SIZE} -gt 1 ]]; then
    echo "│   --tensor-parallel = ${TENSOR_PARALLEL_SIZE}"
    fi
    if [[ "${ENFORCE_EAGER}" == "true" ]]; then
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Compile Mode:"
    echo "│   --enforce-eager   = true (torch.compile disabled)"
    fi
    echo "└─────────────────────────────────────────────────────────────┘"
    echo ""
    
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
    
    # 构建命令 (使用 vllm run-batch)
    # 使用 --override-generation-config 来设置 max_new_tokens 和 temperature
    # 使用 --served-model-name model 来匹配 prompts 文件中的 "model": "model"
    local tp_flag=""
    if [[ ${ACTUAL_TP_SIZE} -gt 1 ]]; then
        tp_flag="--tensor-parallel-size ${ACTUAL_TP_SIZE}"
    fi
    
    local cmd="${env_prefix} vllm run-batch \
        --model ${model_path} \
        --served-model-name model \
        --input-file ${PROMPT_INPUT_FILE} \
        --output-file ${output_file} \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --override-generation-config '{\"max_new_tokens\": ${MAX_OUTPUT_TOKENS}, \"temperature\": ${TEMPERATURE}}' \
        ${tp_flag} \
        ${eager_flag} \
        --disable-log-stats"
    
    # 记录到日志
    echo "" >> "${LOG_FILE}"
    echo "========== ${local_dir_name} ==========" >> "${LOG_FILE}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}"
    echo "Model: ${model_path}" >> "${LOG_FILE}"
    echo "Output: ${output_file}" >> "${LOG_FILE}"
    echo "Command: ${cmd}" >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
    
    # Dry-run 模式
    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Command to execute:"
        echo "$cmd"
        echo ""
        # 生成模拟结果用于 dry-run 测试
        echo '{"status": "dry-run", "model": "'${local_dir_name}'"}' > "${output_file}"
        return 0
    fi
    
    # 执行测试
    print_info "Starting accuracy test..."
    local start_time=$(date +%s)
    local exit_code=0
    local output_tmp=$(mktemp)
    
    # 运行并捕获输出到临时文件和日志
    # 使用 sed 过滤 ANSI 转义码，避免日志文件出现乱码
    eval $cmd 2>&1 | tee "$output_tmp" | sed 's/\x1b\[[0-9;]*m//g' >> "${LOG_FILE}" || exit_code=$?
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $exit_code -eq 0 ]] && [[ -f "$output_file" ]]; then
        rm -f "$output_tmp"
        print_success "Test completed! Duration: ${duration}s"
        print_success "Output saved to: ${output_file}"
        
        # 显示输出文件的简要预览
        echo ""
        echo -e "${GREEN}Output Preview (first 5 entries):${NC}"
        echo "----------------------------------------------"
        head -5 "${output_file}" 2>/dev/null || echo "Unable to read output file"
        echo "----------------------------------------------"
        
        # 统计输出条目数
        local output_count=$(wc -l < "${output_file}" 2>/dev/null || echo "0")
        print_info "Total output entries: ${output_count}"
        
        # 提取回答内容到 txt 文件
        extract_responses_to_txt "${output_file}" "${response_txt}"
        
        return 0
    else
        print_error "Test failed for ${local_dir_name} (exit code: ${exit_code})"
        echo "ERROR: Test failed for ${local_dir_name}" >> "${LOG_FILE}"
        
        # 提取错误信息并记录
        if [[ -f "$output_tmp" ]]; then
            # 提取最后 20 行作为错误摘要
            echo "ERROR OUTPUT:" >> "${LOG_FILE}"
            tail -20 "$output_tmp" >> "${LOG_FILE}" 2>/dev/null
            
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

# 测试单个模型
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
    
    run_accuracy_test "$model_key" "$model_info" "$quant_type"
    local test_result=$?
    
    if [[ $test_result -ne 0 ]]; then
        # 统一试错策略: 错误时返回特殊退出码 2 表示应跳过该精度的所有后续模型
        echo ""
        print_warning "=================================================="
        print_warning "⚠️  Error encountered for ${quant_type^^} model: ${local_dir_name}"
        print_warning "    Will skip all other ${quant_type^^} models."
        print_warning "=================================================="
        echo "SKIP: ${quant_type^^} test failed, skipping remaining ${quant_type^^} tests" >> "${LOG_FILE}"
        
        return 2
    fi
    
    echo ""
    print_info "Model ${local_dir_name} completed successfully"
    return 0
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
            --max-tokens)
                MAX_OUTPUT_TOKENS="$2"
                shift 2
                ;;
            --temperature)
                TEMPERATURE="$2"
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
    
    # 获取硬件信息用于目录命名
    GPU_FOLDER_NAME=$(get_gpu_folder_name)
    if [[ "${GPU_FOLDER_NAME}" == "unknown" ]] || [[ -z "${GPU_FOLDER_NAME}" ]]; then
        print_warning "Unable to detect GPU info, using 'unknown_gpu' as folder name"
        GPU_FOLDER_NAME="unknown_gpu"
    fi
    
    # 设置输出目录
    # 目录结构: accuracy_quickbench_results/{GPU_NAME}/{timestamp}/
    #           ├── {Model}.json
    #           ├── {Model}_responses.txt
    #           └── benchmark.log
    RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="${SCRIPT_DIR}/accuracy_quickbench_results/${GPU_FOLDER_NAME}/${RUN_TIMESTAMP}"
    mkdir -p "${OUTPUT_DIR}"
    LOG_FILE="${OUTPUT_DIR}/benchmark.log"
    
    # 显示配置信息
    print_header "SlideSparse vLLM Accuracy Quick Benchmark"
    
    # 显示硬件信息
    echo ""
    print_hw_info_table
    echo ""
    
    echo "Test Configuration:"
    echo "  Prompt Input File:  ${PROMPT_INPUT_FILE}"
    echo "  Max Output Tokens:  ${MAX_OUTPUT_TOKENS}"
    echo "  Max Model Length:   ${MAX_MODEL_LEN}"
    echo "  Temperature:        ${TEMPERATURE}"
    echo "  GPU Memory Util:    ${GPU_MEMORY_UTILIZATION}"
    echo "  vLLM Log Level:     ${VLLM_LOG_LEVEL}"
    echo ""
    echo "Output Directory: ${OUTPUT_DIR}"
    echo "Log File:         ${LOG_FILE}"
    echo "=============================================="
    
    # 将配置信息写入日志文件头部 (追加模式)
    {
        echo ""
        echo "=============================================="
        echo "SlideSparse vLLM Accuracy Quick Benchmark Log"
        echo "=============================================="
        echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        echo "Hardware Information:"
        print_hw_info_table
        echo ""
        echo "Test Configuration:"
        echo "  Prompt Input File:  ${PROMPT_INPUT_FILE}"
        echo "  Max Output Tokens:  ${MAX_OUTPUT_TOKENS}"
        echo "  Max Model Length:   ${MAX_MODEL_LEN}"
        echo "  Temperature:        ${TEMPERATURE}"
        echo "  GPU Memory Util:    ${GPU_MEMORY_UTILIZATION}"
        echo "  vLLM Log Level:     ${VLLM_LOG_LEVEL}"
        echo "=============================================="
        echo ""
    } >> "${LOG_FILE}"
    
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
    echo "  ├── {Model}.json"
    echo "  └── {Model}_responses.txt"
    echo ""
    echo "To inspect results:"
    echo "  - View *_responses.txt for easy-to-read Q&A pairs"
    echo "  - View *.json for full API response details"
    echo "=============================================="
}

# 运行主程序
main "$@"
