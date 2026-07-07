export WANDB_MODE="offline"

MASTER_PORT=$((RANDOM % 50001 + 10000))
forget_losses=(
    IDK+GD
)

split_list=(forget05)
#split_list=(forget10 forget01)
#split_list=(forget10 forget05 forget01)
# You can specify any forget task from 1 to 10
# the standard TOFU benchmark is task 1
#task_list=(1)
task_list=(1 2 3 4 5)
# pass to python script, for continual learning setting
# export TASK_LIST=$(IFS=,; echo "${task_list[*]}")

learning_rates=(1e-5)

num_epochs=5

mask=true
use_LoRA=false
save_root_base=./llm_unlearn_results

forget_coeff=(1.0)
#forget_coeff=(0.8 0.6 0.4 0.2 0.1)
regularization_coeff=1.0

save_checkpoint=false

### evaluate only at the last epoch
save_steps=last
eval_steps=(last)

### evaluate at each unlearning epoch
#save_steps=12
#eval_steps=(12 24 36 48 60 72 84 96 108 120 132 144 156 168 180 192 204 216 228 240 252 264 276 288 last)

max_steps=300
alternate=true
#optim_cfg=dual_adam_mix
#optim_cfg=dual_adam_mix_8bit
#optim_cfg=dual_adam_plus_8bit_quantize_base
#optim_cfg=dual_adam_plus_8bit_quantize_delta
#optim_cfg=dual_adam_plus_8bit
optim_cfg=dual_adam_plus
#optim_cfg=dual_adam
#optim_cfg=dual_adam_8bit
#optim_cfg=adam
#optim_cfg_lst=(adam)
retain_freq=5

forget_lr=(1e-5)
#forget_lr=(1e-5 6e-6 8e-6 1.2e-5 1.4e-5)
alpha_lst=(1.0)
beta1=0.9
beta2=0.95
base_beta1=0.9
base_beta2=0.95

#for optim_cfg in ${optim_cfg_lst[@]}; do
for flr in ${forget_lr[@]}; do
  for fc in ${forget_coeff[@]}; do
    for split in ${split_list[@]}; do
      for forget_loss in ${forget_losses[@]}; do
        for alpha in ${alpha_lst[@]}; do
          save_root="${save_root_base}_test/optim_${optim_cfg}_v5_beta_${beta1}_${beta2}_${base_beta1}_${base_beta2}"
#          save_root="${save_root_base}/optim_${optim_cfg}_joint"
          for lr in ${learning_rates[@]}; do
            for task_id in ${task_list[@]}; do
              export TASK_LIST=$(IFS=,; echo "${task_id}") # not continual learning setting
              COMMON="use_LoRA=$use_LoRA forget_coeff=$fc regularization_coeff=$regularization_coeff lr=$lr forget_lr=$flr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                  mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint alternate=$alternate optim_cfg=$optim_cfg retain_freq=$retain_freq  \
                  alpha=$alpha beta1=$beta1 beta2=$beta2 base_beta1=$base_beta1 base_beta2=$base_beta2 max_steps=$max_steps"
              CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
                      forget.py \
                      --config-name=phi1-5_tofu.yaml \
                      task_id=$task_id \
                      save_steps=$save_steps \
                      $COMMON
              for step in ${eval_steps[@]}; do
                  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                          eval.py \
                          --config-name=phi1-5_tofu.yaml \
                          task_id=$task_id \
                          eval_unlearn_step=$step \
                          $COMMON
              done
            done
          done
        done
      done
    done
  done
done
#done

