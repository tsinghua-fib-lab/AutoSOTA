# First, train a vae to extract features (the vae trained in score-based modeling could also be directly used here once you copy the checkpoint into proper path)
CUDA_VISIBLE_DEVICES=0 python ./classification_process/train_vae.py --dataname adult


# Second, take adult dataset as an example, train a robust classification model with our score-based reweighting
datasets=("adult") 

timeweights=("EDM") 

temperatures=(3) 

for ((j=0; j<${#timeweights[@]}; j++))
do
        dataset=${datasets[j]}
        timeweight=${timeweights[j]}
        T=${temperatures[j]}
        CUDA_VISIBLE_DEVICES=3 python ./classification_process/main.py --dataname $dataset \
            --log_path ./classification_process/discrete_log \
            --method several_sigma_error_diff --mode train --comment several_sigma_error_diff \
            --use_weight --weight_criterion several_timestep_error_diff \
            --timestep_weight_criterion $timeweight --temperature $T --error_reflection softmax \
            --seed 42
done