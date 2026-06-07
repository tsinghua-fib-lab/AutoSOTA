from collections import OrderedDict

from mmengine.dist import get_dist_info
from mmengine.hooks import Hook
from torch import nn

from mmdet.registry import HOOKS
from mmdet.utils import all_reduce_dict
from mmengine.model import is_model_wrapper
import os
import torch


@HOOKS.register_module()
class ChangePetDINOPromptTypeHook(Hook):
    """Change Visual Text Mode Hook, currently used in PET-DINO."""

    def __init__(self,
                 text_prompt_iter:int = 1,
                 visual_prompt_iter: int = 8,
                 print_log: bool = False,
                 ):
        self.text_prompt_iter = text_prompt_iter
        self.visual_prompt_iter = visual_prompt_iter
        self.print_log = print_log
    
    def before_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch) -> None:
        # change after every n iterations.
        # or (self.end_of_epoch(runner.train_dataloader, batch_idx)):
        # start from 1, when occur iter%self.interval==0, change prompt_type

        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        assert model.prompt_type in ['Text', 'Visual']
        if (runner.iter + 1) % (self.visual_prompt_iter + self.text_prompt_iter) < self.visual_prompt_iter:
            model.prompt_type = 'Visual'
        else:
            model.prompt_type = 'Text'
        if self.print_log:
            runner.logger.info(f"TRAIN : Current prompt type is {model.prompt_type}.")

        