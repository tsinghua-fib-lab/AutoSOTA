_base_ = './deeplabv3plus_r50-d8_4xb2-40k_cityscapes-769x769.py'
model = dict(pretrained='checkpoints/DINOv2_base2small_training_a2mimresnet101_IMNET_FM_bias_head_2_epoch100_GTA_finetune_epoch300.pth', backbone=dict(depth=101))
