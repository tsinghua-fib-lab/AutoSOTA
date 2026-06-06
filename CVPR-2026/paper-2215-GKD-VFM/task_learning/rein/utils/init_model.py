# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Union
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from mmseg.utils import SampleList, dataset_aliases, get_classes, get_palette
from rein.hooks.load_backbone_hook import load_backbone
import torch
from collections import OrderedDict

def init_model(config: Union[str, Path, Config],
               checkpoint = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None):
    """Initialize a segmentor from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint: 
            if type==str: load checkpoint path directly
            if type==dict: load checkpoint ['backbone'] and checkpoint['rein_head'].
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
        cfg_options (dict, optional): Options to override some settings in
            the used config.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.pretrained = None
    config.model.train_cfg = None
    init_default_scope(config.get('default_scope', 'mmseg'))

    model = MODELS.build(config.model)
    if checkpoint is not None:
        if isinstance(checkpoint,str):
            checkpoint=torch.load(checkpoint,map_location='cpu')
        elif isinstance(checkpoint,dict):
            backbone=checkpoint['backbone']
            rein_head=checkpoint['rein_head']
            checkpoint=torch.load(rein_head,map_location='cpu')
            load_backbone(checkpoint,backbone)
        else:
            raise NotImplementedError()
        if 'meta' not in checkpoint:
            checkpoint['meta']={}
        dataset_meta = checkpoint['meta'].get('dataset_meta', None)
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmseg 1.x
            model.dataset_meta = dataset_meta
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # < mmseg 1.x
            classes = checkpoint['meta']['CLASSES']
            palette = checkpoint['meta']['PALETTE']
            model.dataset_meta = {'classes': classes, 'palette': palette}
        else:
            warnings.simplefilter('once')
            warnings.warn(
                'dataset_meta or class names are not saved in the '
                'checkpoint\'s meta data, classes and palette will be'
                'set according to num_classes ')
            num_classes = model.decode_head.num_classes
            dataset_name = None
            for name in dataset_aliases.keys():
                if len(get_classes(name)) == num_classes:
                    dataset_name = name
                    break
            if dataset_name is None:
                warnings.warn(
                    'No suitable dataset found, use Cityscapes by default')
                dataset_name = 'cityscapes'
            model.dataset_meta = {
                'classes': get_classes(dataset_name),
                'palette': get_palette(dataset_name)
            }
    model.cfg = config  # save the config in the model for convenience

    # checkpoints = OrderedDict()
    # for k,v in checkpoint['state_dict'].items():
    #     if 'model' in k:
    #         checkpoints[k.replace("model.", "")] = v
    # model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model