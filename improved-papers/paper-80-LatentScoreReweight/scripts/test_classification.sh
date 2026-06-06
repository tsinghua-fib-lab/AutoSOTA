
attrs=("race" "sex" "marital.status")

datasets=( 
    "adult"
)

paths=(
    "Replace the checkpoint of trained classification model"
)

temperatures=( 3 )

for attr in "${attrs[@]}"
do
    for ((j=0; j<${#paths[@]}; j++))
    do
        path=${paths[j]}
        dataset=${datasets[j]}
        T=${temperatures[j]}
        CUDA_VISIBLE_DEVICES=2 python ./classification_process/main.py --dataname $dataset \
            --log_path ./classification_process/discrete_log \
            --method several_sigma_error_diff --mode test --comment several_sigma_error_diff \
            --use_weight --weight_criterion several_timestep_error_diff \
            --timestep_weight_criterion EDM --temperature $T --error_reflection softmax \
            --eval_attribute $attr \
            --evaluated_model_path $path \
            --seed 42
    done
done