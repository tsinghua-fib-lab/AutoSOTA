# """
# Base AV-LLMs eval script for SAVVY-Bench evaluation - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
# Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
# """
#!/bin/bash

set -e


if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    gpu_count=${#devices[@]}
fi

###
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export HF_HOME=""
export HF_DATASETS_CACHE=""
export HF_TOKENIZERS_CACHE=""
export GOOGLE_API_KEY=""

output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
launcher=accelerate

available_models="gemini_flash,gemini_pro,longvale_7b,ola_7b,egogpt_7b,salmonn_vid_13b,minicpm_o_8b,video_llama2_7b"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --max_frame_num)
        max_frame_num="$2"
        shift 2
        ;;
    --benchmark)
        benchmark="$2"
        shift 2
        ;;
    --num_processes)
        num_processes="$2"
        shift 2
        ;;
    --model)
        IFS=',' read -r -a models <<<"$2"
        shift 2
        ;;
    --output_path)
        output_path="$2"
        shift 2
        ;;
    --limit)
        limit="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done


# Override default num_frames if max_frame_num is provided
if [ -n "$max_frame_num" ]; then
    num_frames="$max_frame_num"
fi


if [ "$models" = "all" ]; then
    IFS=',' read -r -a models <<<"$available_models"
fi

for model in "${models[@]}"; do
    echo "Start evaluating $model..."

    case "$model" in
    "gemini_flash")
        model_family="gemini_api"
        model_args="model_version=gemini-2.5-flash,modality=video"
        ;;
    "gemini_pro")
        model_family="gemini_api"
        model_args="model_version=gemini-2.5-pro,modality=video"
        ;;
    "video_llama2_7b")
        model_family="llama_vid2_av"
        model="video_llama2_7b_${num_frames}f"
        model_args="pretrained=DAMO-NLP-SG/VideoLLaMA2.1-7B-AV,modality=video,conv_template=qwen_1_5,max_frames_num=$num_frames"
        num_processes=4
        ;;
    "longvale_7b")
        model_family="longvale"
        model="longvale_7b_${num_frames}f"
        model_args="modality=video,conv_template=qwen_1_5,max_frames_num=$num_frames"
        num_processes=4
        ;;
    "minicpm_o_8b")
        model_family="minicpm_o"
        model="minicpm_o_8b_${num_frames}f"
        model_args="pretrained=openbmb/MiniCPM-o-2_6,max_frames_num=$num_frames"
        num_processes=1
        ;;
    "egogpt_7b")
        model_family="egogpt"
        model="egogpt_7b_${num_frames}f"
        model_args="pretrained=lmms-lab/EgoGPT-7b-EgoIT-EgoLife,max_frames_num=$num_frames"
        num_processes=1
        ;;
    "ola_7b")
        model_family="ola"
        model="ola_7b_${num_frames}f"
        model_args="pretrained=THUdyh/Ola-7b,max_frames_num=$num_frames"
        num_processes=1
        ;;
    "salmonn_vid_13b")
        model_family="salmonn_vid"
        model="salmonn_7b_${num_frames}f"
        model_args="arg_file_path=third_party/lmms_eval/lmms_eval/models/salmonn_configs/test.yaml,max_frames_num=$num_frames"
        num_processes=1
        ;;
    *)
        echo "Unknown model: $model"
        exit -1
        ;;
    esac

    if [ "$launcher" = "python" ]; then
        export LMMS_EVAL_LAUNCHER="python"
        evaluate_script="python \
            "
    elif [ "$launcher" = "accelerate" ]; then
        export LMMS_EVAL_LAUNCHER="accelerate"
        evaluate_script="accelerate launch \
            --num_processes=$num_processes \
            "
    fi

    evaluate_script="$evaluate_script -m lmms_eval \
        --model $model_family \
        --model_args $model_args \
        --tasks $benchmark \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $model \
        --output_path $output_path/$benchmark \
        "

    if [ -n "$limit" ]; then
        evaluate_script="$evaluate_script \
            --limit $limit \
        "
    fi
    echo $evaluate_script
    eval $evaluate_script
done
