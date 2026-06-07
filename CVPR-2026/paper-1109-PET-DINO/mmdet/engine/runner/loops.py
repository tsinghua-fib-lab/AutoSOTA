# Copyright (c) OpenMMLab. All rights reserved.
from unicodedata import category
from mmengine.model import is_model_wrapper
from mmengine.runner import ValLoop, TestLoop

from mmdet.registry import LOOPS
import torch
from mmengine.fileio import (isdir, join_path, list_dir_or_file)


@LOOPS.register_module()
class TeacherStudentValLoop(ValLoop):
    """Loop for validation of model teacher and student."""

    def run(self):
        """Launch validation for model teacher and student."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        model = self.runner.model
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'teacher')
        assert hasattr(model, 'student')

        predict_on = model.semi_test_cfg.get('predict_on', None)
        multi_metrics = dict()
        for _predict_on in ['teacher', 'student']:
            model.semi_test_cfg['predict_on'] = _predict_on
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
            # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            multi_metrics.update(
                {'/'.join((_predict_on, k)): v
                 for k, v in metrics.items()})
        model.semi_test_cfg['predict_on'] = predict_on

        self.runner.call_hook('after_val_epoch', metrics=multi_metrics)
        self.runner.call_hook('after_val')

@LOOPS.register_module()
class CustomTestLoop(TestLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def __init__(self,
                 prompt_visual_embedding_path: str,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.prompt_visual_embedding_path = prompt_visual_embedding_path

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        # Load visual embeddings into data_samples for inference
        memory_visual, memory_label, memory_class_name = self.read_visual_embedding(self.prompt_visual_embedding_path)
        for idx, data_batch in enumerate(self.dataloader):
            for i in range(len(data_batch['data_samples'])):
                data_batch['data_samples'][i].set_metainfo({'memory_visual': memory_visual,
                                                            'memory_label': memory_label,
                                                            'memory_class_name': memory_class_name})
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics
    
    def read_visual_embedding(self, prompt_visual_embedding_path):
        if isdir(prompt_visual_embedding_path):
            filename_list = list_dir_or_file(
                prompt_visual_embedding_path, list_dir=False, suffix=('.pt',))
            prompt_visual_embedding_file_list = [
                join_path(prompt_visual_embedding_path, filename) for filename in filename_list
            ]
        else:
            prompt_visual_embedding_file_list = [prompt_visual_embedding_path]
        visual_embedding_zipped = []
        for prompt_visual_embedding_file in prompt_visual_embedding_file_list:
            print(f"CustomTestLoop : Loads visual_embedding from path: {prompt_visual_embedding_file}.")
            visual_embedding_dict = torch.load(prompt_visual_embedding_file)
            visual_embedding_zipped.append((visual_embedding_dict['visual_embedding'],
                                            visual_embedding_dict['label'],                  
                                            visual_embedding_dict['class_name']))
        # sort by label
        visual_embedding_zipped_sorted = sorted(visual_embedding_zipped, key=lambda x: x[1])
        visual_embedding_tuple, category_id_tuple, class_name_tuple = zip(*visual_embedding_zipped_sorted)

        # Convert category_id_tuple to zero-based label_tuple for COCO metric evaluation
        label_tuple = list(range(len(category_id_tuple)))

        memory_visual, memory_label, memory_class_name = torch.stack(visual_embedding_tuple), torch.tensor(label_tuple), list(class_name_tuple)

        return memory_visual, memory_label, memory_class_name