# ScatterBrain Implementation

## Setup

Building from source: https://github.com/HazyResearch/fly/blob/master/Dockerfile
```
git clone https://github.com/idiap/fast-transformers \
    && sed -i 's/\["-arch=compute_60"\]/\["-arch=compute_80"\]/' fast-transformers/setup.py \
    && pip install fast-transformers/ \
    && rm -rf fast-transformers
```
NOTE: check your NVIDIA GPU architecture [here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

NOTE: important that GPU driver, CUDA compiler, and pytorch CUDA versions all match

## KDEformer Table 2 configuration

From the authors of KDEformer (Zandieh et al., 2023) on 4/11/2024:

> I also used the code for sblocal_attention from the same repo: https://github.com/HazyResearch/fly/blob/master/src/models/attention/sblocal_attention.py with this setting:
accuracy: 0.81946000, corrects: 40973, seed: 1, config.model: {'_target_': 'src.models.vit.t2t_vit.t2t_vit_t_24', 't2tattn1_cfg': {'_target_': 'src.models.attention.sblocal_attention.SBLocalAttention', 'dim_heads': 64, 'local_context': 49, 'nb_features': 48}, 't2tattn2_cfg': {'_target_': 'src.models.attention.sblocal_attention.SBLocalAttention', 'dim_heads': 64, 'local_context': 12, 'nb_features': 6}, 'drop_rate': 0.0, 'drop_path_rate': 0.1, 'img_size': 224}
