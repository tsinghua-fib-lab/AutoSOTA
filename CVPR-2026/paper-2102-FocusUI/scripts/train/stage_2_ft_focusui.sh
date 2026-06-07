export WANDB_MODE="online"
export WANDB_API_KEY=""
export WANDB_PROJECT="FocusUI"


model_type="focusui_3b"
llm_model="huggingface/Qwen2.5-VL-3B-Instruct"
output_dir="./localhome/focus_runs/checkpoints/${model_type}_ft_final"

export WANDB_NAME="FocusUI_Qwen25VL_3B_FT"

# === GPU Assignment ===
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# === Training Command ===
torchrun --nproc_per_node=8 --master_port=29666 \
  train_focusui.py \
  --deepspeed ./scripts/zero2.json \
  --data_path data/data_config.yaml \
  --image_folder "" \
  --model_type ${model_type} \
  --model_name_or_path ${llm_model} \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ${output_dir} \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 100 \
  --learning_rate 5e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --ddp_timeout 3600 \
  --tf32 True \
  --model_max_length 24576 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --max_pixels 5720064 \
  --unfreeze_all_parameters True \
  --unfreeze_pointer_head False \
  --unfreeze_lm_head False \
  --unfreeze_base_model False \
  --unfreeze_last_n_layers -1 \
  --unfreeze_visual False \
  --unfreeze_patch_scorer False \
  --unfreeze_new_pointer_tokens False \
  --unfreeze_new_image_drop_tokens False \
  --unfreeze_new_all_tokens False \
  --lm_loss_weight 1.0 \
  --ps_loss_weight 0.1 \
  --pointer_loss_weight 1.0 \
  --train_patch_scorer_only False \
  --focus_ui_train_visual_reduct_ratio_min 0.0 \
  --focus_ui_train_visual_reduct_ratio_max 0.95 \
  --focus_ui_visual_reduct_ratio 0.5 \
  --train_patch_scorer_from_ckpt checkpoints/patch_scorer_qwen25vl_3b.pt
