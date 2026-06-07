# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmengine.runner import ValLoop as MMENGINE_ValLoop
from mmengine.dist import broadcast_object_list, is_main_process, get_world_size, get_rank, barrier, collect_results
import math
import torch
from mmengine.model import is_model_wrapper
from types import MethodType
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from transformers import GenerationConfig

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

class TestLoop(MMENGINE_ValLoop):
    def __init__(self, runner, dataloader, evaluator=None, torch_dtype='fp16', select_metric='first') -> None:
        # must be concatset
        super(MMENGINE_ValLoop, self).__init__(runner, dataloader)
        self._runner = runner
        self.torch_dtype = torch_dtype
        if torch_dtype is not None:
            self.torch_dtype = TORCH_DTYPE_MAP[torch_dtype]
        self.select_metric = select_metric

    def run(self) -> dict:
        """Launch Test."""
        self.runner.logger.info('==================== Start test loop ===================')
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')

        if is_model_wrapper(self.runner.model):
            model = self.runner.model.module
        else:
            model = self.runner.model

        model.gradient_checkpointing_disable()
        model.eval()
        model.cuda()

        rank = get_rank()
        metrics = []
        # Ensure that eta and log are displayed correctly.
        current_run_total_ids = 0
        for _, dataset in enumerate(self.dataloader.dataset.datasets):
            if not hasattr(model, 'preparing_for_generation'):
                model.preparing_for_generation = MethodType(default_preparing_for_generation, model)
                print("Warning, the model do not have the preparing_for_generation() function, using the default!!!")
            model.preparing_for_generation(dataset.metainfo)

            # split per rank
            results = []
            n_samples = len(dataset)
            per_rank_samples = math.ceil(n_samples / get_world_size())
            per_rank_ids = range(per_rank_samples * rank,
                                 min(n_samples, per_rank_samples * (rank + 1)))
            for idx in per_rank_ids:
                data_batch = dataset[idx]
                self.run_iter(current_run_total_ids, data_batch, results, model)
                current_run_total_ids += 1

            barrier()
            self.runner.logger.info('==================== Start collect results ===================')
            results = collect_results(results, len(dataset))
            self.runner.logger.info('========= Starting the evaluation of a data ===========')
            if is_main_process():
                metric = dataset.evaluate(results, self.runner.work_dir)
                objects = [metric]
            else:
                objects = [None]
            broadcast_object_list(objects)
            metric = objects[0]
            metrics.append(metric)

        # select metrics
        if self.select_metric == 'first':
            metrics = metrics[0]
        else:
            raise NotImplementedError

        self.runner.logger.info('================ Ending test loop ================')
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch, results, model):
        assert 'text_prompts' in data_batch and 'pixel_values' in data_batch and 'img_id' in data_batch
        prediction = {'img_id': data_batch['img_id']}

        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)

        outputs = model.predict_forward(**data_batch)
        prediction.update(outputs)
        results.append(prediction)

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)

def default_preparing_for_generation(self, metainfo):
    # set stop criteria and generation configs for model

    assert hasattr(self, 'tokenizer'), "The Model does not have the tokenizer!!!"

    self.bot_name = 'BOT'
    template = PROMPT_TEMPLATE['internlm2_chat']
    self.template = template
    stop_words = []
    stop_words += template.get('STOP_WORDS', [])
    stop_criteria = get_stop_criteria(
        tokenizer=self.tokenizer, stop_words=stop_words)
    self.stop_criteria = stop_criteria

    default_generation_kwargs = dict(
        max_new_tokens=2048,
        do_sample=False,
        eos_token_id=self.tokenizer.eos_token_id,
        pad_token_id=(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        ),
    )
    default_generation_kwargs.update(metainfo.get('generation_kwargs', {}))
    self.gen_config = GenerationConfig(**default_generation_kwargs)
    return


class AnnoLoop(MMENGINE_ValLoop):
    def __init__(self, runner, dataloader, evaluator=None, torch_dtype='fp16', select_metric='first') -> None:
        # must be concatset
        super(MMENGINE_ValLoop, self).__init__(runner, dataloader)
        self._runner = runner
        self.torch_dtype = torch_dtype
        if torch_dtype is not None:
            self.torch_dtype = TORCH_DTYPE_MAP[torch_dtype]
        self.select_metric = select_metric

    def run(self) -> dict:
        """Launch Test."""
        self.runner.logger.info('==================== Start test loop ===================')
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')

        if is_model_wrapper(self.runner.model):
            model = self.runner.model.module
        else:
            model = self.runner.model

        model.eval()

        rank = get_rank()
        metrics = []
        # Ensure that eta and log are displayed correctly.
        current_run_total_ids = 0
        for _, dataset in enumerate(self.dataloader.dataset.datasets):

            # split per rank
            results = []
            n_samples = len(dataset)
            per_rank_samples = math.ceil(n_samples / get_world_size())
            per_rank_ids = range(per_rank_samples * rank,
                                 min(n_samples, per_rank_samples * (rank + 1)))
            for idx in per_rank_ids:
                data_batch = dataset[idx]
                self.run_iter(current_run_total_ids, data_batch, results, model)
                current_run_total_ids += 1
            if hasattr(model, 'save_step'):
                model.save_step(last=True)

            barrier()
            self.runner.logger.info('==================== Start collect results ===================')
            results = collect_results(results, len(dataset))
            self.runner.logger.info('========= Starting the evaluation of a data ===========')
            if is_main_process():
                metric = dataset.evaluate(results, self.runner.work_dir)
                objects = [metric]
            else:
                objects = [None]
            broadcast_object_list(objects)
            metric = objects[0]
            metrics.append(metric)

        # select metrics
        if self.select_metric == 'first':
            metrics = metrics[0]
        else:
            raise NotImplementedError

        self.runner.logger.info('================ Ending test loop ================')
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch, results, model):
        prediction = {}

        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)

        outputs = model.predict_forward(**data_batch)
        prediction.update(outputs)
        results.append(prediction)

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)