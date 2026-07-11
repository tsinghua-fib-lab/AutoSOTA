# Performer Implementation

## KDEformer Table 2 configuration

From the authors of KDEformer (Zandieh et al., 2023) on 4/11/2024:

> Here is our setting/result of Performer (in Table 2):
accuracy: 0.80498000, corrects: 40249, seed: 1, config.model: {'_target_': 'src.models.vit.t2t_vit.t2t_vit_t_24', 't2tattn1_cfg': {'_target_': 'src.models.attention.performer_attention.PerformerAttention', 'dim_heads': 64, 'nb_features': 49, 'softmax_eps': 0.0, 'normalization_eps': 0.0}, 't2tattn2_cfg': {'_target_': 'src.models.attention.performer_attention.PerformerAttention', 'dim_heads': 64, 'nb_features': 12, 'softmax_eps': 0.0, 'normalization_eps': 0.0}, 'drop_rate': 0.0, 'drop_path_rate': 0.1, 'img_size': 224}
I used the Performer code from this repo: https://github.com/HazyResearch/fly/blob/master/src/models/attention/performer_attention.py. Its parameters are "dim_heads" and "nb_features", and they are set to the above values.
