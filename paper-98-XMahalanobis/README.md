# X-Mahalanobis: Transformer Feature Mixing for Reliable OOD Detection
PyTorch Code for the following paper published at NeurIPS 2025

<b>Title</b>: <i>X-Mahalanobis: Transformer Feature Mixing for Reliable OOD Detection</i> 

<b>Abstract</b>
Recognizing out-of-distribution (OOD) samples is essential for deploying robust machine learning systems in the open-world environments. Conventional OOD detection approaches rely on feature representations from the final layer of neuron networks, often neglecting the rich information encapsulated in shallow layers. Leveraging the strengths of transformer-based architectures, we introduce an adaptive fusion module, which dynamically assigns importance weights to representations learned by each Transformer layer and detects OOD samples using the Mahalanobis distance. Compared to existing approaches, our method enables a lightweight fine-tuning of pre-trained models, and retains all feature representations that are beneficial to the OOD detection. We also thoroughly study various parameter-efficient fine-tuning strategies. Our experiments show the benefit of using shallow features, and demonstrate the influence of different Transformer layers. We fine-tune pre-trained models in both class-balanced and long-tailed in-distribution classification tasks, and show that our method achieves state-of-the-art OOD detection performance averaged across nine OOD datasets. The source code is provided in the supplementary material.


## Train


CIFAR100:

```
CUDA_VISIBLE_DEVICES=0 python main_train.py -d cifar100 -m in21k_vit_b16_peft \
    batch_size 64   output_dir  cifar100_adpf   test_ensemble False   \
    micro_batch_size 64  lr 0.01 \
    adaptformer True  adapter False  lora False  vpt_deep  False  vpt_shallow False  bias_tuning  False  full_tuning False 
```


ImageNet-LT:

```
CUDA_VISIBLE_DEVICES=0 python main_train.py -d imagenet_lt -m clip_vit_b16_peft \
    adaptformer True  batch_size 64   output_dir imagenetlt_adpf  test_ensemble False   \
    lr 0.1   num_epochs 20   micro_batch_size 64 \
    adaptformer True  adapter False  lora False  vpt_deep  False  vpt_shallow False  bias_tuning  False  full_tuning False 
```

## Test for other baselines

CIFAR100:

```
CUDA_VISIBLE_DEVICES=0 python main_test.py -d cifar100 -m in21k_vit_b16_peft \
    batch_size 64   output_dir  cifar100_adpf    test_ensemble False   \
    micro_batch_size 64  lr 0.01 \
    adaptformer True  adapter False  lora False  vpt_deep  False  vpt_shallow False  bias_tuning  False  full_tuning False \
    test_only True model_dir  <where_to_save_the_ckpt>
```


ImageNet-LT:

```
CUDA_VISIBLE_DEVICES=0 python main_test.py -d imagenet_lt    -m clip_vit_b16_peft \
    batch_size 64   output_dir  imagenetlt_adpf  test_ensemble False \
    lr 0.1   num_epochs 20  micro_batch_size 64   \
    adaptformer True  adapter False  lora False  vpt_deep  False  vpt_shallow False  bias_tuning  False  full_tuning False \
    test_only True model_dir  <where_to_save_the_ckpt>
```

Note: While reproducing the PEFT experiment, the corresponding PEFT method needs to be adjusted to True. 
For different data sets, you can choose among "cifar100_ir100", "cifar100", "imagenet_lt", and "imagenet".
For different pre-trained models, you can choose between "in21k_vit_b16_peft" and "clip_vit_b16_peft".


