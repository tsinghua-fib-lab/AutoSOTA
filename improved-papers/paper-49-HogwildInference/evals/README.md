#  Hogwild! Inference: Parallel LLM Generation with a Concurrent Attention Cache 

## Sections 4.1 and 4.2

Common environment variables for all scripts in this section:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# ^-- set it to a list of gpu ids to be used for evaluation, even if you use all GPUs
export GPUS_PER_PROCESS=1
# ^-- keep at 1 unless your model does not fit on a GPU; if the model doesn't fit, increase until it does.

if (( GPUS_PER_PROCESS > 1 )); then
  export HOGWILD_USE_TRITON=0
  export HOGWILD_NO_COMPILE=1
fi
```

### LIMO

Running on LIMO (after the common part above):
```bash
export TOTAL_TASKS=817
export PYTHONPATH=`pwd`/evals/limo:`pwd`:$PYTHONPATH
# run Hogwild! Inference
python3 utils/gpu_parallel.py --start 0 --end $TOTAL_TASKS --use_queue --gpus_per_process $GPUS_PER_PROCESS \
  --script evals/limo/limo_generate_hogwild.py \
  --extra_args "--eval_folder ./results --seed 42  --model_name Qwen/QwQ-32B"
# to run run baseline + early stopping, replace the line above with:
# python3 utils/gpu_parallel.py --start 0 --end $TOTAL_TASKS --use_queue --gpus_per_process $GPUS_PER_PROCESS \
#    --script evals/limo/limo_generate_baselines.py \
#    --extra_args "--eval_folder ./results --seed 42 --model_name Qwen/QwQ-32B"
```


### LiveCodeBench 
Running on LiveCodeBench v5 2024.08-2025.02 (one seed, QwQ subset)
```bash
export TOTAL_TASKS=279
export PYTHONPATH=`pwd`/evals/livecodebench:`pwd`:$PYTHONPATH
python3 evals/livecodebench/process_data.py --num_samples 1 --dataset_path $HOME/.cache/lcb --dataset_size None \
  --output_livecodebench_v5_tests_dir ./lcb_tests --output_livecodebench_v5_data_path ./livecodebench_v5.jsonl

# force use the test subset from QwQ-32B
cp evals/livecodebench/data/livecodebench_v5_qwq-32b_subset.jsonl ./livecodebench_v5.jsonl
export ACTUAL_TOTAL_TASKS=`python -c "print(end=str(len(list(open('livecodebench_v5.jsonl')))))"`
if [ "$TOTAL_TASKS" -ne "$ACTUAL_TOTAL_TASKS" ]; then echo "Actual TOTAL_TASKS should be $ACTUAL_TOTAL_TASKS"; exit 1; fi

# run Hogwild! Inference
python3 utils/gpu_parallel.py --start 0 --end $TOTAL_TASKS --use_queue --script evals/livecodebench/lcb_generate_hogwild.py \
  --extra_args "--lcb_input_file ./livecodebench_v5.jsonl --eval_folder ./results --seed 42 --finisher_max_new_tokens 1024 --model_name Qwen/QwQ-32B --seed 42"
# to run baseline, replace the line above with this:
# python3 utils/gpu_parallel.py --start 0 --end $TOTAL_TASKS --use_queue --script evals/livecodebench/lcb_generate_baselines.py \
#   --extra_args "--lcb_input_file ./livecodebench_v5.jsonl --eval_folder ./results --seed 42 --finisher_max_new_tokens 1024 --model_name Qwen/QwQ-32B --seed 42"

# run tests to evaluate Pass@1 (over this one seed)
LIVECODEBENCH_TESTS=./lcb_tests python3 evals/livecodebench/eval.py \
  --input_folder_path ./results/evals_data/livecodebench --cache_path ./results/eval_cache/ \
  --results_path ./results/results.csv
```
### OlympiadBench

To run on OlympiadBench (Math or Physics subsets),
you need to download the benchmark tasks from https://github.com/OpenBMB/OlympiadBench
(see download instructions in the README).

The code below assumes that the dataset is downloaded and extracted to `OLYMPIADBENCH_PATH` - set your path accordingly.

Additionally, you need to install a package that the benchmark uses for automatic scoring:

`pip install antlr4-python3-runtime==4.11`

Running on OlympiadBench with Math and Physics subsets (see comments inside on choosing the subset)
```bash
export TOTAL_TASKS=675
export OLYMPIADBENCH_PATH=<...>
# ^-- path to the downloaded and extracted OlympiadBench dataset (e.g. ./OlympiadBench_Dataset )

# to evaluate on Math subset:
export DATASET_PATH=$OLYMPIADBENCH_PATH/data/OE_TO_maths_en_COMP.json
# to evaluate on Physics subset, use:
# export DATASET_PATH=$OLYMPIADBENCH_PATH/data/OE_TO_physics_en_COMP.json

export PYTHONPATH=`pwd`/evals/olympiadbench:`pwd`:`pwd`/evals/olympiadbench/inference/code:$PYTHONPATH
python3 utils/gpu_parallel.py --start 0 --end $TOTAL_TASKS --use_queue --script evals/olympiadbench/generate_hogwild.py\
    --extra_args "--eval_folder ./results/evals_data/olympiadbench --dataset_path $DATASET_PATH --model_name Qwen/QwQ-32B --seed 42"
# to run baseline, replace the line above with this:
# python3 utils/gpu_parallel.py --start 0 --end $TOTAL_TASKS --use_queue --script evals/olympiadbench/generate_baselines.py \
#   --extra_args "--eval_folder ./results/evals_data/olympiadbench --dataset_path $OLYMPIADBENCH_PATH/data/OE_TO_maths_en_COMP.json --model_name Qwen/Qwen3-32B --seed 42"

# reformat saved files into an evaluation-appropriate format
python3 evals/olympiadbench/inference/judge.py --input_dir ./results/evals_data/olympiadbench --output_dir ./results/evals_data/merged

# evaluate and print accuracy for each budget
python3 evals/olympiadbench/inference/calculate_accuracy.py --model_output ./results/evals_data/merged --ref_dir $OLYMPIADBENCH_PATH/data/
```