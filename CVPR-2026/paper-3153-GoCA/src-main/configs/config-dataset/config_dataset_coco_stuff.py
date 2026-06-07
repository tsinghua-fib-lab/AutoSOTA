import os

class DatasetConfig:
    def __init__(self):
        self.name = 'coco-stuff'

        # ----- basic setting ----- #

        dataset_path = 'path/to/datasets/coco-stuff'
        self.which_dataset = 'coco-stuff'
        self.sample = os.path.join(dataset_path, 'val2017/*.jpg')
        self.label = os.path.join(dataset_path, 'val2017/*.png')
        self.prompt = './prompt_annotation/coco-stuff.json'
        self.limit=10000

        # ----- gpt labeling setting ----- #

        self.gpt_output_dir = './gpt_output/coco-stuff'
        self.caption_len = 20

        # ----- captioning setting ----- #

        self.add_background_token = False
        self.add_missing_class = True
        self.caption_input_file = './gpt_output/coco-stuff.json'
        self.caption_output_file = './prompt_annotation/coco-stuff.json'
        # note: 1-5 and xl tokenizers are identical

        # ----- running setting ----- #
        self.background_setting = {
            # 'method': 'vanilla',
            # 'background_threshold': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            # 'method': 'avg',
            # 'background_threshold': [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
            'method': 'offset',
            'background_threshold': [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4],
        }
