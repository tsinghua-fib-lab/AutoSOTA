import os
import tqdm
import torch
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler
from diffusion_feature import FeatureExtractor

from components.dataset import get_dataset
from components.downstream_dispatch import downstream
from components.postprocess_dispatch import postprocess
from components.space_attn_dispatch import space_attn, space_attn_prepare
from components.cross_attn_dispatch import cross_attn, cross_attn_prepare, store_scale_record

from configs.current_model import ModelConfig
from configs.current_dataset import DatasetConfig


# settings
model_config = ModelConfig()
dataset_config = DatasetConfig()

version = model_config.version
img_size = model_config.img_size
feat_size = model_config.feat_size
t = model_config.t
device = 'cuda'

test_set = get_dataset(
    dataset_config.which_dataset,
    dataset_config.sample,
    dataset_config.label,
    dataset_config.prompt.replace(dataset_config.name, f'{dataset_config.name}-{model_config.name}'),
    dataset_config.limit,
    dataset_config.additional_annotation if hasattr(dataset_config, 'additional_annotation') else None,
)

cross_attn_setting = model_config.cross_attn_setting
if 'keep_all_objects' not in cross_attn_setting:
    cross_attn_setting['keep_all_objects'] = None
space_attn_setting= model_config.space_attn_setting
postprocess_setting = model_config.postprocess_setting
downstream_setting = model_config.downstream_setting
downstream_setting['save_path_root'] += f'_{dataset_config.name}'
downstream_setting.update(dataset_config.background_setting)


# 0. prepare feature extractor
layer = set()
attention = set()

cross_attn_prepare(cross_attn_setting['task'], attention, layer, cross_attn_setting)
space_attn_prepare(space_attn_setting['task'], attention, layer, space_attn_setting)
for task in postprocess_setting:
    if 'dense_feat_id' in task:
        layer.add(task['dense_feat_id'])

feature_extractor = FeatureExtractor(
    layer={l: True for l in layer},
    version=version,
    device=device,
    img_size=img_size,
    # attention=list(attention),
)
# feature_extractor.pipe.scheduler = DDIMScheduler.from_config(feature_extractor.pipe.scheduler.config)

class_count = len(test_set.class_name)
for i, content in enumerate(tqdm.tqdm(test_set)):
    img, label, prompt, objects, rescaling_token_id, background_objects = content
    img_path = img
    img = Image.open(img)

    with torch.no_grad():
        # 1. get cross attn
        cross_attn_setting.update({
            'feature_extractor': feature_extractor, 'img': img, 'size': feat_size, 't': t,
            'prompt': prompt, 'rescaling_token_id': rescaling_token_id, 'objects': objects,
            'background_objects': background_objects,
        })
        cross_feat, features = cross_attn(cross_attn_setting['task'], cross_attn_setting)

        # 2. get space attn
        space_attn_setting.update({'features': features, 'size': feat_size})
        space_feat = space_attn(space_attn_setting['task'], space_attn_setting)

        # 3. conduct postprocess
        if cross_feat is not None:
            if isinstance(postprocess_setting, dict):
                postprocess_setting.update({
                    'cross_feat': cross_feat, 'space_feat': space_feat, 'image_path': img_path,
                })
                cross_feat = postprocess(postprocess_setting['task'], postprocess_setting)
            else:
                for postprocess_step in postprocess_setting:
                    postprocess_step.update({
                        'cross_feat': cross_feat, 'space_feat': space_feat, 'image_path': img_path,
                        'features': features, 'img_size': 512, 'label': label, 'class_count': class_count,
                        'image': img,
                    })
                    cross_feat = postprocess(postprocess_step['task'], postprocess_step)

        # 4. run downstream task
        downstream_setting.update({
            'pred': cross_feat,
            'label': label,
            'save_path': os.path.join(downstream_setting['save_path_root'], str(i)),
            'present_objects': list(objects.keys()),
            'class_count': class_count,
        })
        downstream(downstream_setting['task'], downstream_setting)

downstream_setting.update({'save_path': downstream_setting['save_path_root']})
downstream(downstream_setting['task'], downstream_setting, final_call=True)

# store_scale_record()
