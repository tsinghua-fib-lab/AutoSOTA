# pet_dino_UseGlobalBox_CoordinateToEncoding_acrossBatchV2_aggregate_bs4_swin-t_pretrain_obj365.py
_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pretrained = './pretrained/swin/swin_tiny_patch4_window7_224.pth'  # noqa
lang_model_name = './pretrained/bert-base-uncased'

model = dict(
    type='PetDINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    language_model=dict(
        type='BertModel',
        name=lang_model_name,
        max_tokens=256,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    visual_prompt_model=dict(
        type='VisualPromptBertModel',
        max_tokens=256,
        pad_to_max=False,
        use_global_box=True,
        hidden_dim_D=128,
        use_coordinate_to_encoding=True,
        prompt_across_batch_v2=True,
        aggregate_NonSelf_prompts=True
    ),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder_first_then_visual_prompt_encoder=True,
    visual_prompt_encoder=dict(
        type='VisualPromptEncoder',
        num_layers=3,
        num_cp=3,
        # visual prompt layer config
        visual_layer_cfg=dict(      # PetDINOVisualPromptEncoderLayer
            cross_attn_cfg=dict(    # MultiScaleDeformableAttention
                num_heads=4,
                embed_dims=256,
                dropout=0.0,
                ),
            self_attn_cfg=dict(     # MultiheadAttention
                num_heads=4,
                embed_dims=256,
                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
    ),
    memory_bank=dict(
        # NOTE. default memory_bank is for dataset that do not have label_map_tag or label_map_tag=None.
        # If all datasets have label_map_tag, you do not need to set default memory_bank.
        # NOTE. Each dataset must have a corresponding memory_bank for now.
        objv1=dict(
            # init args
            max_length=16,
            dim=256,
            label_map_file='data/objects365v1/o365v1_label_map.json',
            # use args
            min_FeatureNum = 1,
            max_num_sample_visual_prompt = 40,
            only_sample_neg = False,
            use_mean_feature = True,
        ),
    ),
    encoder=dict(
        num_layers=6,
        num_cp=6,
        # visual layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        # text layer config
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        # fusion layer config
        fusion_layer_cfg=dict(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to visual or not
            with_cross_attn_visual=False,
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='PetDINOHead',
        num_classes=256,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True, independent_visual_bias=False),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        # loss_iou=dict(type='GIoULoss', loss_weight=2.0),  # loss_iou default is this
        loss_bbox=dict(type='L1Loss', loss_weight=5.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300, prompt_type='Visual'))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=False),     # NOTE. original is True
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        # This step performs label remap, so GetVisualPromptfromGT needs to come after
        type='RandomSamplingNegPos',
        tokenizer_name=lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(type='GetVisualPromptfromGT',
         filter_empty_results=True,
         prompt_bboxes_already_transformed=True,
         mode='random'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode',
                   # visual prompt
                   'prompt_type', 'prompt_img_path', 'prompt_ori_shape', 'prompt_img_shape',
                   'prompt_bboxes_labels', 'prompt_bboxes', 
                   'homography_matrix', 'prompt_bboxes_already_transformed',
                   # only training in visual prompt for prompt across batch
                   'reverse_label_remap_dict', 'label_map_tag'
                   ))
]

# NOTE. 1. for validation during training; 2. for standalone validation; 3. for test.
test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    # NOTE. Since evaluation visualizes on the original image, LoadAnnotations should be placed after FixScaleResize.
    # GT boxes are not resized, so prompt_bboxes_already_transformed below needs to be set to False.
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='GetVisualPromptfromGT',
         filter_empty_results=False,
         prompt_bboxes_already_transformed=False,
         mode='one-shot'),     # random / one-shot / full-shot
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'text', 'custom_entities', 'tokens_positive',
                   # visual prompt
                   'prompt_type', 'prompt_img_path', 'prompt_ori_shape', 'prompt_img_shape',
                   'prompt_bboxes_labels', 'prompt_bboxes', 
                   'homography_matrix', 'prompt_bboxes_already_transformed',
                   ))
]

prompt_image_test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    # NOTE. LoadAnnotations is placed after FixScaleResize, so GT boxes are not resized simultaneously;
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='GetVisualPromptfromGT',
         filter_empty_results=False,
         prompt_bboxes_already_transformed=False),
         # no need to set 'mode', no use here.
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'text', 'custom_entities', 'tokens_positive',
                   # visual prompt
                   'prompt_type', 'prompt_img_path', 'prompt_ori_shape', 'prompt_img_shape',
                   'prompt_bboxes_labels', 'prompt_bboxes', 
                   'homography_matrix', 'prompt_bboxes_already_transformed',
                   # inference with memory_visual
                   'memory_visual', 'memory_label', 'memory_class_name'
                   ))
]

dataset_type = 'ODVGDataset'
data_root = 'data/objects365v1/'

coco_od_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    # ann_file='o365v1_train_odvg.json',
    ann_file='objects365_train_od.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    need_text=True,
    label_map_tag='objv1',
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None)

train_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    # batch_size=1,
    # num_workers=0,
    # persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[coco_od_dataset]))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    # batch_size=1,
    # num_workers=0,
    # persistent_workers=False,
    dataset=dict(pipeline=test_pipeline, return_classes=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0004,
                   weight_decay=0.0001),  # bs=16 0.0001
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            # NOTE.
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.01),
            'language_model': dict(lr_mult=0.01),
            # NOTE. 
            'text_feat_map': dict(lr_mult=0.01),
            'visual_feat_map': dict(lr_mult=1),
            'backbone': dict(lr_mult=0.01),
            'language_model': dict(lr_mult=0.01),
            'visual_prompt_model': dict(lr_mult=1),
            'neck': dict(lr_mult=0.01),
            'visual_prompt_encoder': dict(lr_mult=1),
            'encoder': dict(lr_mult=0.01),
            # 'encoder.fusion_layers': dict(lr_mult=1),
            'encoder.layers': dict(lr_mult=1),
            'decoder': dict(lr_mult=1),
            'bbox_head': dict(lr_mult=1),
        }))

# learning policy
max_epochs = 12
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

default_hooks = dict(visualization=dict(type='GroundingVisualizationHook'),
                     checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
                     logger=dict(type='LoggerHook', interval=50))

custom_hooks = [
    dict(type='ChangePetDINOPromptTypeHook', text_prompt_iter=1, visual_prompt_iter=8, print_log=False),
]

load_from = './pretrained/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth'

# NOTE tensorboard
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# NOTE test_cfg
test_cfg = dict(type='TestLoop')
# test_cfg = dict(type='CustomTestLoop', prompt_visual_embedding_path='/path/to/visual_embedding/')