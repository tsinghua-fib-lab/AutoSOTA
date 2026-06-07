# Copyright (c) OpenMMLab. All rights reserved.
import os.path

import cv2
import mmengine
from mmengine.runner import ValLoop as MMENGINE_ValLoop
from mmengine.dist import broadcast_object_list, is_main_process, get_world_size, get_rank, barrier, collect_results
import math
import torch
from mmengine.model import is_model_wrapper
from types import MethodType
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.tools.utils import get_stop_criteria
from transformers import GenerationConfig
from pycocotools import mask as _mask
from mmengine.visualization.visualizer import Visualizer

from vlm.utils import VideoReader

TORCH_DTYPE_MAP = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

VID_INTERVAL = 4

def visualize(data_batch, prediction, visualize_path='work_dirs/visualize'):
    if 'video_path' in data_batch:
        vid_frames = VideoReader(data_batch['video_path'])[::VID_INTERVAL]
        vid_id = os.path.basename(data_batch['video_path']).split('.')[0]
        text_prompts = data_batch['text_prompts']
        mmengine.mkdir_or_exist(os.path.join(visualize_path, vid_id))
        visualizer = Visualizer()

        mmengine.mkdir_or_exist(os.path.join(visualize_path, vid_id, "vid"))
        for id_frame, img in enumerate(vid_frames):
            out_path = os.path.join(visualize_path, vid_id, "vid", "{:06d}.jpg".format(id_frame))
            cv2.imwrite(out_path, img)

        for id_text, text in enumerate(text_prompts):
            mmengine.mkdir_or_exist(os.path.join(visualize_path, vid_id, "sample_{:06d}".format(id_text)))
            mmengine.put_text(text, os.path.join(visualize_path, vid_id, "sample_{:06d}".format(id_text), 'text.txt'))
            for id_frame, img in enumerate(vid_frames):
                visualizer.set_image(img)
                mask = prediction['prediction_masks'][id_text][id_frame]
                mask = _mask.decode(mask).astype(bool)
                visualizer.draw_binary_masks(mask, colors='g')
                visual_result = visualizer.get_image()
                out_path = os.path.join(visualize_path, vid_id, "sample_{:06d}".format(id_text),
                                        "{:06d}.jpg".format(id_frame))
                cv2.imwrite(out_path, visual_result)
    else:
        images_files = data_batch['images']
        vid_id = data_batch['video_id']
        text_prompts = data_batch['text_prompts']
        image_folder = data_batch['image_folder']
        mmengine.mkdir_or_exist(os.path.join(visualize_path, "{:06d}".format(vid_id)))
        visualizer = Visualizer()

        mmengine.mkdir_or_exist(os.path.join(visualize_path, "{:06d}".format(vid_id), "vid"))
        for id_frame, img_file in enumerate(images_files):
            img = cv2.imread(os.path.join(image_folder, img_file))
            out_path = os.path.join(visualize_path, "{:06d}".format(vid_id), "vid", os.path.basename(img_file))
            cv2.imwrite(out_path, img)

        for id_text, text in enumerate(text_prompts):
            mmengine.mkdir_or_exist(os.path.join(visualize_path, "{:06d}".format(vid_id), "sample_{:06d}".format(id_text)))
            mmengine.put_text(text, os.path.join(visualize_path, "{:06d}".format(vid_id), "sample_{:06d}".format(id_text),
                                                 'text.txt'))
            for id_frame, img_file in enumerate(images_files):
                img = cv2.imread(os.path.join(image_folder, img_file))
                visualizer.set_image(img)
                mask = prediction['prediction_masks'][id_text][id_frame]
                mask = _mask.decode(mask).astype(bool)
                visualizer.draw_binary_masks(mask, colors='g')
                visual_result = visualizer.get_image()

                out_path = os.path.join(visualize_path, "{:06d}".format(vid_id), "sample_{:06d}".format(id_text),
                                        os.path.basename(img_file))
                cv2.imwrite(out_path, visual_result)



class VideoTestLoop(MMENGINE_ValLoop):
    def __init__(self, runner, dataloader, torch_dtype='fp16', select_metric='first', visualize=None, evaluator=None) -> None:
        # must be concatset
        super(MMENGINE_ValLoop, self).__init__(runner, dataloader)
        self._runner = runner
        self.torch_dtype = torch_dtype
        if torch_dtype is not None:
            self.torch_dtype = TORCH_DTYPE_MAP[torch_dtype]
        self.select_metric = select_metric

        self.visualize = visualize
        self.evaluator = evaluator

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
            running_tot = per_rank_samples * get_world_size()
            assert running_tot >= n_samples
            per_rank_ids = range(per_rank_samples * rank, per_rank_samples * (rank + 1))
            for idx in per_rank_ids:
                if n_samples <= idx:
                    data_batch = dataset[n_samples - 1]
                else:
                    data_batch = dataset[idx]
                self.run_iter(current_run_total_ids, data_batch, results, model)
                current_run_total_ids += 1

            barrier()
            self.runner.logger.info('==================== Start collect results ===================')
            results = collect_results(results, n_samples)
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
        prediction = {'video_id': data_batch['video_id']}

        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)

        outputs = model.predict_forward(**data_batch)
        prediction.update(outputs)
        results.append(prediction)

        if self.visualize:
            # if not prediction['is_exists'][0].all():
            #     print(prediction['is_exists'])
            visualize(data_batch=data_batch, prediction=prediction, visualize_path=self.visualize)

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
