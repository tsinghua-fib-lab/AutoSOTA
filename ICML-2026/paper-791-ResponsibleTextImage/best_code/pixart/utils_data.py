import glob
import os
import json
from PIL import Image
import numpy as np
import torch


def int_to_onehot(x, n):
    if not isinstance(x, list):
        x = [x]
    assert isinstance(x[0], int)
    x = torch.tensor(x).long()
    v = torch.zeros(n)
    v[x] = 1.
    return v


random_select = lambda l: l[np.random.choice(len(l))]
top_select = lambda l: l[0]


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform, tokenizer, max_concept_length, select):
        # Look for images named 0.jpg to 1999.jpg
        image_paths = []

        for ext in ['.jpg', '.jpeg', '.png']:
            pattern = os.path.join(image_folder, f"*{ext}")
            found_files = glob.glob(pattern)

            # Filter to only numeric names
            for file_path in found_files:
                basename = os.path.basename(file_path)
                name_without_ext = os.path.splitext(basename)[0]
                try:
                    int(name_without_ext)
                    image_paths.append(file_path)
                except ValueError:
                    continue

            if image_paths:
                break

        # Sort images by their numerical ID
        def get_image_id(path):
            basename = os.path.basename(path)
            name_without_ext = os.path.splitext(basename)[0]
            try:
                return int(name_without_ext)
            except ValueError:
                return float('inf')

        image_paths = sorted(image_paths, key=get_image_id)

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {image_folder}")

        self.image_paths = image_paths
        self.transform = transform
        self.tokenizer = tokenizer

        # Load concept_dict.json - REQUIRED for training
        concept_dict_path = os.path.join(image_folder, 'concept_dict.json')
        if not os.path.exists(concept_dict_path):
            raise FileNotFoundError(
                f"concept_dict.json not found in {image_folder}. "
                f"Please ensure your dataset has concept_dict.json"
            )
        
        self.concept_dict = json.load(open(concept_dict_path, 'r'))
        print(f"✓ Loaded concept dictionary: {self.concept_dict}")

        self.max_concept_length = max_concept_length

        if select == "top":
            self.select_method = top_select
        elif select == "random":
            self.select_method = random_select
        else:
            raise NotImplementedError(f"Unknown select method: {select}")

        # Load labels.json - REQUIRED for training
        labels_path = os.path.join(image_folder, 'labels.json')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"labels.json not found in {image_folder}. "
                f"Please ensure your dataset has labels.json"
            )
        
        self.labels = json.load(open(labels_path, 'r'))
        print(f"✓ Loaded {len(self.labels)} labels")
        print(f"✓ Found {len(self.image_paths)} images")

    def __getitem__(self, index):
        # Get input prompt and target concept
        if index < len(self.labels):
            input_prompt, target_concept = self.select_method(self.labels[index])
        else:
            # Fallback for missing labels
            input_prompt = "a photo of a person"
            target_concept = ["female"]  # Default to first concept

        # Tokenize prompt
        input_prompt_tokens = self.tokenizer([input_prompt])[0]

        # Convert target concept to onehot
        try:
            target_concept = [self.concept_dict[c] for c in target_concept if c in self.concept_dict]
            if not target_concept:
                target_concept = [0]  # Default to first concept if none match
        except (KeyError, TypeError):
            target_concept = [0]

        target_concept = int_to_onehot(target_concept, self.max_concept_length)

        # Load image
        image_path = self.image_paths[index]
        x = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)

        return x, input_prompt_tokens, target_concept

    def __len__(self):
        return len(self.image_paths)


def get_dataloader(image_folder, batch_size, transform, tokenizer, collate_fn=None, num_workers=4, shuffle=False,
                   max_concept_length=100, select="random"):
    dataset = TrainingDataset(image_folder, transform=transform, tokenizer=tokenizer, select=select,
                              max_concept_length=max_concept_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=collate_fn)
    return dataloader


def parse_concept(input_concept):
    """
    parse the input concept into a list of concepts for evaluation, supported formats:
    concept: str:
        'man' -> ['man'] (generate an image of a man)
    concept: str:
        'man,young' -> ['man', 'young'] (generate an image of a young man)
    concept: list[str]:
        ['man', 'woman'] -> ['man'], ['woman'] (generate two images: a man, a woman)
    concept: list[str]:
        ['man,young', 'woman,young'] -> [['man','young'],['woman','young']] (generate two images: a young man, an old woman)

    The output of this function is directly fed to int_to_onehot and return a multi-hot vector which can be directly used by the model
    """

    def parse_concept_string(concept):
        assert isinstance(concept, str)
        concept = concept.split(',')
        concept = [x.strip() for x in concept]
        return concept

    if isinstance(input_concept, str):
        input_concept = parse_concept_string(input_concept)
        input_concept = [input_concept]

    elif isinstance(input_concept, list):
        input_concept = [parse_concept_string(x) for x in input_concept]

    else:
        raise ValueError(input_concept)

    return input_concept


def get_test_data(data_dir, given_prompt=None, given_concept=None, with_baseline=True, device='cuda',
                  max_concept_length=100):
    """
    data_dir: path to data file
    prompt: str
    concept: str or list[str]
    """
    concept_dict = json.load(open(os.path.join(data_dir, 'concept_dict.json'), 'r'))
    if not given_prompt or not given_concept:
        prompt, concept = json.load(open(os.path.join(data_dir, 'test.json'), 'r'))
    if given_prompt:
        prompt = given_prompt
    if given_concept:
        concept = given_concept

    concept = parse_concept(concept)
    print(f'eval with concept: {concept}')

    concept = [int_to_onehot([concept_dict[c_i] for c_i in c], max_concept_length).to(device).unsqueeze(0) for c in
               concept]
    if with_baseline:
        concept.insert(0, None)
    prompt = [prompt] * len(concept)
    return prompt, concept


def get_i2p_data(data_dir=None, given_prompt=None, given_concept=None, with_baseline=True, device='cuda',
                 max_concept_length=100):
    import pandas as pd
    i2p = pd.read_csv("./i2p_benchmark.csv")
    if given_prompt:
        prompts = i2p[i2p.categories.apply(lambda x: given_prompt in x)].prompt.values.tolist()
    else:
        prompts = i2p.prompt.values.tolist()

    concept_label = [given_concept] if isinstance(given_concept, str) else given_concept
    concept_dict = json.load(open(os.path.join(data_dir, 'concept_dict.json'), 'r'))
    concept = [int_to_onehot(concept_dict[x], max_concept_length).to(device).unsqueeze(0) for x in concept_label]
    if with_baseline:
        concept.insert(0, None)
        concept_label.insert(0, 'none')

    inputs = []
    for prompt in prompts:
        for c_i, c in zip(concept, concept_label):
            inputs.append([prompt, c_i, c])
    return inputs
