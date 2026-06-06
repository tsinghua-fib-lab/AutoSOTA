_base_ = "./convnext-L_mask2former_512x512_bs1x4.py"
model = dict(
    backbone=dict(
        type="ReinsConvNeXt",
        reins_config=dict(
            type="LoRAReins",
            token_length=100,
            patch_size=16,
            link_token_to_query=True,
            lora_dim=16,
        ),
        distinct_cfgs=(
            dict(
                num_layers=3,
                embed_dims=192,
            ),
            dict(
                num_layers=3,
                embed_dims=384,
            ),
            dict(
                num_layers=27,
                embed_dims=768,
            ),
            dict(
                num_layers=3,
                embed_dims=1536,
            ),
        ),
    ),
    decode_head=dict(
        type="ReinMask2FormerHead",
        replace_query_feat=True,
    ),
)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
)

