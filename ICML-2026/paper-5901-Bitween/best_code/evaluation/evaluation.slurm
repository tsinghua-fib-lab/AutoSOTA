#!/bin/bash

#SBATCH --output=logs/%x-%j_%A_%a.out
#SBATCH --error=logs/%x-%j_%A_%a.err
#SBATCH --mail-type=ALL

#SBATCH --job-name=evaluation_rsr_bench_agentic
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --mem=100G

# source ~/.bashrc

# module purge
# module load miniconda
# conda activate bitween

export TOKENIZERS_PARALLELISM=true
export PYTORCH_ALLOC_CONF=expandable_segments:True

available_port() {
    first_port=8000
    last_port=8100

    for port in $(seq "${first_port}" "${last_port}"); do
        if ! (echo >"/dev/tcp/localhost/${port}") >/dev/null 2>&1; then
            echo "${port}"
            return 0
        fi
    done

    echo "No available port found"
    exit 1
}

start_vllm() {
    port="${1}"
    model_id="${2}"
    out="${3}"

    vllm_log="${out}/vllm_server.log"
    host="localhost"
    gpu_mem_util="0.8"
    max_model_len="32000"
    tool_call_parser="hermes"

    echo "starting vllm"
    conda activate vllm

    # --enable-reasoning --reasoning-parser deepseek_r1 \

    # start vllm server in the background
    vllm serve --host "${host}" --port "${port}" \
        --gpu_memory_utilization "${gpu_mem_util}" \
        --max_model_len "${max_model_len}" \
        --enable-auto-tool-choice --tool-call-parser "${tool_call_parser}" \
        "${model_id}" \
        >>"${vllm_log}" 2>&1 &

    VLLM_PID=$!

    # wait for the vllm server to connect
    until nc -z "${host}" "${port}" &>/dev/null; do
        if ! kill -0 "${VLLM_PID}" &>/dev/null; then
            echo "vllm terminated abruptly"
            exit 1
        else
            echo "waiting for vllm to connect to ${host}:${port}"
            sleep 10
        fi
    done

    echo "vllm is connected"
    conda deactivate

    # return the process id of vllm as exit status
    return "${VLLM_PID}"
}

stop_vllm() {
    VLLM_PID="${1}"

    echo "shutting down vllm"

    kill -SIGTERM "${VLLM_PID}"
    wait "${VLLM_PID}"

    echo "vllm has shut down"
}

get_pyfile() {
    run_type="${1}"
    bench_type="${2}"

    pyfile="../src/bitween/evaluation/evaluation_rsr_bench"

    if [ "${run_type}" == "agentic" ]; then
        pyfile+="_agentic"
    elif [ ! "${run_type}" == "normal" ]; then
        echo "Unsupported run mode: ${run_type}"
        return 1
    fi

    pyfile+="_paper"

    if [ "${bench_type}" == "algebraic" ]; then
        pyfile+="_algebraic"
    elif [ "${bench_type}" == "algebraic_extended" ]; then
        pyfile+="_algebraic_extended"
    elif [ "${bench_type}" == "rebuttal" ]; then
        pyfile+="_rebuttal"
    elif [ "${bench_type}" == "extended" ]; then
        pyfile+="_extended"
    elif [ ! "${bench_type}" == "original" ]; then
        echo "Unsupported bench type: ${bench_type}"
        return 1
    fi

    pyfile+=".py"

    echo "${pyfile}"
}

run_with_normal() {
    run_type="normal"
    conf_type="${1}"
    postfix="${2}"
    bench_type="${3}"
    rest_args=("${@:4}")

    res_dir="results/${run_type}/${conf_type}_${bench_type}${postfix}"
    mkdir -p "${res_dir}"

    out_log="${res_dir}/log.out"
    err_log="${res_dir}/log.err"

    pyfile=$(get_pyfile "${run_type}" "${bench_type}")
    if [ $? -ne 0 ]; then
        echo "Error occurred: ${pyfile}"
        exit 1
    fi

    python3 "${pyfile}" \
        --res_dir "${res_dir}" \
        "${rest_args[@]}" \
        >>"${out_log}" 2>"${err_log}"
}

run_with_agentic() {
    run_type="agentic"
    conf_type="${1}"
    postfix="${2}"
    bench_type="${3}"
    agent_host="${4}"
    model_id="${5}"

    model_norm="${model_id//\//--}"
    model_norm="${model_norm//:/.}"

    res_dir="results/${run_type}/${agent_host}/${model_norm}_${conf_type}_${bench_type}${postfix}"
    mkdir -p "${res_dir}"

    out_log="${res_dir}/log.out"
    err_log="${res_dir}/log.err"

    pyfile=$(get_pyfile "${run_type}" "${bench_type}")
    if [ $? -ne 0 ]; then
        echo "Error occurred: ${pyfile}"
        exit 1
    fi

    case "${agent_host}" in
    local_openai)
        port=$(available_port)
        start_vllm "${port}" "${model_id}" "${res_dir}"
        VLLM_PID=$!

        agent_type="openai"
        base_url="http://localhost:${port}/v1"
        api_key="None"
        rest_args=("${@:6}")

        python3 "${pyfile}" \
            --agent_type "${agent_type}" \
            --model_id "${model_id}" \
            --base_url "${base_url}" \
            --api_key "${api_key}" \
            --res_dir "${res_dir}" \
            "${rest_args[@]}" \
            >>"${out_log}" 2>"${err_log}"

        stop_vllm "${VLLM_PID}"
        ;;

    remote_openai)
        agent_type="openai"
        base_url="${6}"
        api_key="${OPENAI_API_KEY}"
        rest_args=("${@:7}")

        python3 "${pyfile}" \
            --agent_type "${agent_type}" \
            --model_id "${model_id}" \
            --base_url "${base_url}" \
            --api_key "${api_key}" \
            --res_dir "${res_dir}" \
            "${rest_args[@]}" \
            >>"${out_log}" 2>"${err_log}"
        ;;

    remote_bedrock)
        agent_type="bedrock"
        region_name="${6}"
        rest_args=("${@:7}")

        # If AWS_PROFILE is set in the environment, boto3 will pick it up;
        # otherwise it falls back to the default profile.
        echo "Bedrock run: AWS_PROFILE=${AWS_PROFILE:-default} model=${model_id}"

        python3 "${pyfile}" \
            --agent_type "${agent_type}" \
            --model_id "${model_id}" \
            --region_name "${region_name}" \
            --res_dir "${res_dir}" \
            "${rest_args[@]}" \
            >>"${out_log}" 2>"${err_log}"
        ;;

    *)
        echo "Unsupported agent host: ${agent_host}"
        ;;
    esac
}

run_bitween_pysr() {
    run_with_normal "pysr" "$@" --method pysr
}

run_bitween_gplearn() {
    run_with_normal "gplearn" "$@" --method gplearn
}

run_bitween_milp() {
    run_with_normal "milp" "$@" --method eager_milp --milp gurobi
}

run_bitween_mreg() {
    run_with_normal "mreg" "$@" --method multiple_regression
}

run_bitween_rag_sr() {
    # also requires installed faiss package
    module load OpenBLAS
    run_with_normal "rag-sr" "$@" --method rag_sr
}

run_bitween_parfam() {
    run_with_normal "parfam" "$@" --method parfam
}

run_bitween_metasymnet() {
    run_with_normal "metasymnet" "$@" --method metasymnet
}

run_deep_research_agent() {
    run_with_agentic "deep-research" "$@" --custom_tools
}

run_bitween_agent() {
    run_with_agentic "agentic" "$@"
}

# Available bench_type: original, extended, algebraic, algebraic_extended

## ============================================================
## Symbolic backends
## ============================================================

run_bitween_pysr     "" "original"
run_bitween_pysr     "" "extended"
run_bitween_pysr     "" "algebraic"
run_bitween_pysr     "" "algebraic_extended"

run_bitween_gplearn  "" "original"
run_bitween_gplearn  "" "extended"
run_bitween_gplearn  "" "algebraic"
run_bitween_gplearn  "" "algebraic_extended"

run_bitween_milp     "" "original"
run_bitween_milp     "" "extended"
run_bitween_milp     "" "algebraic"
run_bitween_milp     "" "algebraic_extended"

run_bitween_mreg     "" "original"
run_bitween_mreg     "" "extended"
run_bitween_mreg     "" "algebraic"
run_bitween_mreg     "" "algebraic_extended"

# run_bitween_rag_sr "" "original"
# run_bitween_rag_sr "" "extended"

# run_bitween_parfam "" "original"
# run_bitween_parfam "" "extended"

# run_bitween_metasymnet "" "original"
# run_bitween_metasymnet "" "extended"

## ============================================================
## Agentic backends
## ============================================================
## For different agent_host:
##   "local_openai"  "model_id"
##   "remote_openai" "model_id" "base_url"
##   "remote_bedrock" "model_id" "region_name"
##   "remote_bedrock" "model_id" "region_name" --enable_thinking

## -- gpt-oss-120b (run locally via vLLM, matching the paper text)
##
## Requires a GPU host with enough memory to serve openai/gpt-oss-120b
## (the H200 partition declared at the top of this script). The
## Bedrock-hosted version of this model does not reliably engage tool
## calls via Strands' converse-API path; running it locally through
## start_vllm sidesteps that issue and matches the configuration
## reported in the paper.
##
model_id="openai/gpt-oss-120b"

run_deep_research_agent "" "original"           "local_openai" "${model_id}"
run_deep_research_agent "" "extended"           "local_openai" "${model_id}"
run_deep_research_agent "" "algebraic"          "local_openai" "${model_id}"
run_deep_research_agent "" "algebraic_extended" "local_openai" "${model_id}"
run_bitween_agent       "" "original"           "local_openai" "${model_id}"
run_bitween_agent       "" "extended"           "local_openai" "${model_id}"
run_bitween_agent       "" "algebraic"          "local_openai" "${model_id}"
run_bitween_agent       "" "algebraic_extended" "local_openai" "${model_id}"

## -- claude-sonnet-4
model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"
region_name="us-west-2"

run_deep_research_agent "" "original"           "remote_bedrock" "${model_id}" "${region_name}"
run_deep_research_agent "" "extended"           "remote_bedrock" "${model_id}" "${region_name}"
run_deep_research_agent "" "algebraic"          "remote_bedrock" "${model_id}" "${region_name}"
run_deep_research_agent "" "algebraic_extended" "remote_bedrock" "${model_id}" "${region_name}"
run_bitween_agent       "" "original"           "remote_bedrock" "${model_id}" "${region_name}"
run_bitween_agent       "" "extended"           "remote_bedrock" "${model_id}" "${region_name}"
run_bitween_agent       "" "algebraic"          "remote_bedrock" "${model_id}" "${region_name}"
run_bitween_agent       "" "algebraic_extended" "remote_bedrock" "${model_id}" "${region_name}"

## -- claude-opus-4-1
model_id="us.anthropic.claude-opus-4-1-20250805-v1:0"
region_name="us-west-2"

run_deep_research_agent "" "original"           "remote_bedrock" "${model_id}" "${region_name}"
run_deep_research_agent "" "extended"           "remote_bedrock" "${model_id}" "${region_name}"
run_deep_research_agent "" "algebraic"          "remote_bedrock" "${model_id}" "${region_name}"
run_deep_research_agent "" "algebraic_extended" "remote_bedrock" "${model_id}" "${region_name}"
run_bitween_agent       "" "original"           "remote_bedrock" "${model_id}" "${region_name}"
run_bitween_agent       "" "extended"           "remote_bedrock" "${model_id}" "${region_name}"
run_bitween_agent       "" "algebraic"          "remote_bedrock" "${model_id}" "${region_name}"
run_bitween_agent       "" "algebraic_extended" "remote_bedrock" "${model_id}" "${region_name}"