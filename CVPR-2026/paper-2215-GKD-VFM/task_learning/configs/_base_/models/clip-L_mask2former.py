crop_size = (512, 512)
num_classes = 19
norm_cfg = dict(type="SyncBN", requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=crop_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)
checkpoint_file = "checkpoints/ViT-L-14.pt"
# load from https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint_file,
    backbone=dict(
        type="CLIPVisionTransformer",
        patch_size=16,
        width=1024,
        output_dim=512,
        get_embeddings=False,
        drop_path_rate=0.1,
        layers=24,
        input_resolution=512,
        style="pytorch",
        out_indices=[7, 11, 15, 23],
        heads=16,
    ),
    neck=dict(
        type="MultiLevelNeck",
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=1024,
        scales=[4, 2, 1, 0.5],
    ),
    decode_head=dict(
        type="Mask2FormerHead",
        in_channels=[1024, 1024, 1024, 1024],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type="mmdet.MSDeformAttnPixelDecoder",
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None,
                    ),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                ),
                init_cfg=None,
            ),
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=128, normalize=True
            ),
            init_cfg=None,
        ),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=128, normalize=True
        ),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type="ReLU", inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True,
                ),
            ),
            init_cfg=None,
        ),
        loss_cls=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            reduction="mean",
            class_weight=[1.0] * num_classes + [0.1],
        ),
        loss_mask=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=True,
            reduction="mean",
            loss_weight=5.0,
        ),
        loss_dice=dict(
            type="mmdet.DiceLoss",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0,
        ),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type="mmdet.HungarianAssigner",
                match_costs=[
                    dict(type="mmdet.ClassificationCost", weight=2.0),
                    dict(
                        type="mmdet.CrossEntropyLossCost", weight=5.0, use_sigmoid=True
                    ),
                    dict(type="mmdet.DiceCost", weight=5.0, pred_act=True, eps=1.0),
                ],
            ),
            sampler=dict(type="mmdet.MaskPseudoSampler"),
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        crop_size=(512, 512),
        stride=(341, 341),
    ),
)
