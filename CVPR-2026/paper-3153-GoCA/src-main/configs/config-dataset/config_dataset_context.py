import os

class DatasetConfig:
    def __init__(self):
        self.name = 'context'

        # ----- basic setting ----- #

        image_set_path = 'path/to/datasets/pascal-context'
        dataset_path = 'path/to/datasets/VOCdevkit/VOC2012'
        filename = os.path.join(image_set_path, 'ImageSets/Segmentation/train.txt')
        with open(filename) as f:
            traindata = [ff.rstrip() for ff in f.readlines()]
            testdata = list(reversed(traindata))
        self.which_dataset = 'pascal-context'
        self.sample = [os.path.join(dataset_path, 'JPEGImages', f'{d}.jpg') for d in testdata]
        self.label = [os.path.join(image_set_path, 'trainval', f'{d}.mat') for d in testdata]
        self.prompt = './prompt_annotation/context.json'
        self.limit=3000

        # ----- gpt labeling setting ----- #

        self.gpt_output_dir = './gpt_output/context'
        self.caption_len = 35

        # ----- captioning setting ----- #

        self.add_background_token = False
        self.add_missing_class = True
        self.caption_input_file = './gpt_output/context.json'
        self.caption_output_file = './prompt_annotation/context.json'
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
