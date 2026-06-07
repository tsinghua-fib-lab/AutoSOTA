_base_ = '../pet_dino_swin-t_8xb4_12e_obj365.py'

model = dict(test_cfg=dict(
    max_per_img=300,
    chunked_size=-1,    # for visual-I visual prompt
))

dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis/'

val_dataloader = dict(
    batch_size=2,       # for visual-I visual prompt
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        ann_file='annotations/lvis_v1_minival_inserted_image_name.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader

# numpy < 1.24.0
val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root +
    'annotations/lvis_v1_minival_inserted_image_name.json')
test_evaluator = val_evaluator
