import torch

import random
import numpy as np

import matplotlib as m

m.use("Agg")

import torch
import torch.nn as nn

import torchvision.transforms as transforms

import datasets
from datasets import load_dataset, Dataset

from utils.io_utils import *


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


### credit to https://github.com/somepago/DCR
def insert_rand_word(sentence, word):
    sent_list = sentence.split(" ")
    sent_list.insert(random.randint(0, len(sent_list)), word)
    new_sent = " ".join(sent_list)
    return new_sent


def prompt_augmentation(prompt, aug_style, tokenizer=None, repeat_num=4):
    if aug_style == "rand_numb_add":
        for i in range(repeat_num):
            randnum = np.random.choice(100000)
            prompt = insert_rand_word(prompt, str(randnum))
    elif aug_style == "rand_word_add":
        for i in range(repeat_num):
            randword = tokenizer.decode(list(np.random.randint(49400, size=1)))
            prompt = insert_rand_word(prompt, randword)
    elif aug_style == "rand_word_repeat":
        wordlist = prompt.split(" ")
        for i in range(repeat_num):
            randword = np.random.choice(wordlist)
            prompt = insert_rand_word(prompt, randword)
    else:
        raise Exception("This style of prompt augmnentation is not written")
    return prompt


def get_dataset(dataset_name, pipe=None):
    if "jsonl" in dataset_name:
        dataset = load_jsonlines(dataset_name)
        prompt_key = "caption"
    elif dataset_name.endswith(".txt"):
        # txt file support: one prompt per line
        with open(dataset_name, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        dataset = [{"caption": prompt} for prompt in prompts]
        prompt_key = "caption"
    elif dataset_name == "random":
        dataset = []
        for _ in range(2000):
            k = random.randrange(pipe.tokenizer.model_max_length)
            rand_tokens = random.sample(range(pipe.tokenizer.vocab_size), k)
            dataset.append({"Prompt": pipe.tokenizer.decode(rand_tokens)})
        prompt_key = "Prompt"
    elif dataset_name == "ChristophSchuhmann/MS_COCO_2017_URL_TEXT":
        dataset = load_dataset(dataset_name)["train"]
        prompt_key = "TEXT"
    elif dataset_name == "Gustavosta/Stable-Diffusion-Prompts":
        dataset = load_dataset(dataset_name)["test"]
        prompt_key = "Prompt"
    else:
        raise NotImplementedError

    return dataset, prompt_key


def get_dataset_finetune(
    dataset_name, non_mem_dataset=None, end=None, repeats=1, non_mem_ratio=0
):
    if "groundtruth" in dataset_name:
        dataset = load_jsonlines(f"{dataset_name}/{dataset_name}.jsonl")
        prompt_key = "caption"
    else:
        all_files = glob.glob(f"{dataset_name}/*.jpg")
        all_files.sort()

        if end is not None:
            all_files = all_files[:end]

        all_data = {"image": [], "text": []}
        for file in all_files:
            f = open(file.replace("jpg", "txt"), "r")
            captions = f.read()

            all_data["image"].append(file)
            all_data["text"].append(captions)

        all_data["image"] = all_data["image"] * repeats
        all_data["text"] = all_data["text"] * repeats
        mem_len = len(all_data["image"])

        if non_mem_dataset is not None:
            ### add non-mem data points
            all_files = glob.glob(f"{non_mem_dataset}/*.jpg")
            all_files.sort()

            for file in all_files:
                if len(all_data["image"]) >= mem_len * (1 + non_mem_ratio):
                    break

                f = open(file.replace("jpg", "txt"), "r")
                captions = f.read()

                all_data["image"].append(file)
                all_data["text"].append(captions)

            all_data["image"] = all_data["image"][: int(mem_len * (1 + non_mem_ratio))]
            all_data["text"] = all_data["text"][: int(mem_len * (1 + non_mem_ratio))]
            ### add non-mem data points

        dataset = Dataset.from_dict(all_data).cast_column("image", datasets.Image())
        prompt_key = "text"

    return dataset, prompt_key


def measure_CLIP_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)


### credit: https://github.com/somepago/DCR
def measure_SSCD_similarity(gt_images, images, model, device):
    ret_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    gt_images = torch.stack([ret_transform(x.convert("RGB")) for x in gt_images]).to(
        device
    )
    images = torch.stack([ret_transform(x.convert("RGB")) for x in images]).to(device)

    with torch.no_grad():
        feat_1 = model(gt_images).clone()
        feat_1 = nn.functional.normalize(feat_1, dim=1, p=2)

        feat_2 = model(images).clone()
        feat_2 = nn.functional.normalize(feat_2, dim=1, p=2)

        return torch.mm(feat_1, feat_2.T)
