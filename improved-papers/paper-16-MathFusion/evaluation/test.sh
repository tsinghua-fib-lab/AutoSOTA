source activate mathfusion

[ -z "$MODEL_NAME" ] && MODEL_NAME=llama3_8b_mathfusion
RES_PATH=$(echo "${MODEL_NAME##*/}" | tr '[:upper:]' '[:lower:]')

PROMPT_TYPE="cot-qa"
n_shots=0

DATASETS_LIST=(
  "gsm8k/test"
  "math/test"
  "mwpbench/college-math/test"
  "deepmind-mathematics"
  "olympiadbench/OE_TO_maths_en_COMP"
  "theoremqa"
)

for DATASETS in ${DATASETS_LIST[@]}; do
  save_pre=outputs/${DATASETS////_}
  mkdir -p $save_pre
  save_path="${save_pre}/${RES_PATH}_${PROMPT_TYPE}_n${n_shots}_seed0.jsonl"
  echo $save_path

  if [ -f "$save_path" ]; then
    echo "File already exists: $save_path. Exiting..."
    exit 0
  fi

  set -x

  python evaluation/run.py \
      --gen_save_path $save_path \
      --model_name_or_path "${MODEL_NAME}" \
      --datasets $DATASETS \
      --max_new_toks 2048 --temperature 0 \
      --prompt_template $PROMPT_TYPE \
      --n_shots $n_shots \
      --inf_seed 0 \
      --max_n_trials 1

  python evaluation/print_metric.py --gen_save_path $save_path
  
done

if [[ "$MODEL_NAME" == *"llama3"* ]]; then
  DATASETS=deepmind-mathematics
  PROMPT_TYPE=alpaca
  save_pre=outputs/${DATASETS////_}
  mkdir -p $save_pre
  save_path="${save_pre}/${MODEL_NAME}_${PROMPT_TYPE}_n${n_shots}.jsonl"
  echo $save_path

  if [ -f "$save_path" ]; then
    echo "File already exists: $save_path. Exiting..."
    exit 0
  fi
  
  python evaluation/run.py \
      --gen_save_path $save_path \
      --model_name_or_path "${MODEL_NAME}" \
      --datasets $DATASETS \
      --max_new_toks 2048 --temperature 0 \
      --prompt_template $PROMPT_TYPE \
      --n_shots $n_shots \
      --inf_seed 0 \
      --max_n_trials 1

  python evaluation/print_metric.py --gen_save_path $save_path
fi