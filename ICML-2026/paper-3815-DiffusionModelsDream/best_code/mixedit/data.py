# Adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/data.py
# and https://github.com/kuleshov-group/mdlm/blob/master/dataloader.py


import re
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import numpy as np

import os
import fsspec
from .utils import utils
import shutil
import urllib
import zipfile
import transformers
import tokenizers
import typing
import requests
import json
import itertools


"""Data utilities, including tokenizing and dataloading."""

def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x

def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()


# Character-level tokenizer for text8 dataset
class Text8Tokenizer(transformers.PreTrainedTokenizer):
    def __init__(
        self,
        **kwargs):
        self.characters = list('abcdefghijklmnopqrstuvwxyz ')
        self._vocab_str_to_int = {** {ch: i for i, ch in enumerate(self.characters)}}
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


# Tokenizer loader
# Modify this to use different tokenizers for different datasets
def get_tokenizer(dataset, eval_data=False):
    if dataset == "text8":
        return Text8Tokenizer()
    elif dataset == "lm1b" and not eval_data:
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        # Default set as gpt2 tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id)
        )
    
    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(
                'Tokenizer must have a bos_token or '
                f'cls_token: {tokenizer}')
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(
                'Tokenizer must have a eos_token '
                f'or sep_token: {tokenizer}')
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset


def get_text8_dataset(cache_dir, max_seq_length=256, drop_last=True, crop_train=False):
    """Adapted from:
	https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

	Args:
	cache_dir: str, path to cache directory.
	max_seq_length: int, maximum length of sequences. (default: 256, as in D3PM codebase.)
	drop_last: bool, whether to drop the last incomplete batch. (default: True, as in D3PM codebase.)
	crop_train: bool, whether to subsample contiguous subsequences from training example. serves to
        make sure transformer models with absolute position
        embeddings do not have incorrect position-wise
        marginals. (default: False, but necessary to match D3PM AR)

	Returns:
	dataset: dataset.DatasetDict, with keys 'train','valid', 'test'.
    """
    url = 'http://mattmahoney.net/dc/text8.zip'
    if not crop_train:
        cache_dir = f'{cache_dir}/text8'
    else:
        cache_dir = f'{cache_dir}/text8-crop-train'
    split_names = ['train', 'validation', 'test']
    if not all([
        utils.fsspec_exists(os.path.join(cache_dir, split))
        for split in split_names
    ]):
        # Check if raw data exists
        raw_cache_dir = os.path.join(cache_dir, 'raw_data')
        if not all([
            utils.fsspec_exists(
                os.path.join(raw_cache_dir, f'text8.{split}.txt'))
            for split in split_names
        ]):
            if not utils.fsspec_exists(
                os.path.join(raw_cache_dir, 'text8.zip')):
                utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
                # LOGGER.info('Downloading text8 from URL {}.'.format(url))
                with (urllib.request.urlopen(url) as in_stream,
                            open(os.path.join(raw_cache_dir, 'text8.zip'),
                                     'wb') as out_file):
                    shutil.copyfileobj(in_stream, out_file)

            with fsspec.open(os.path.join(raw_cache_dir, 'text8.zip'), 'rb') as f:
                rawdata = zipfile.ZipFile(f).read(
                    'text8').decode('utf-8')

            # Splits taken from D3PM codebase
            splits = {
                'train': rawdata[:90000000],
                'validation': rawdata[90000000: 95000000],
                'test': rawdata[95000000:],
            }

            for split, data in splits.items():
                _path = os.path.join(raw_cache_dir, f'text8.{split}.txt')
                with fsspec.open(_path, 'w') as f:
                    f.write(data)
        else:
            splits = {}
            for split in split_names:
                _path = os.path.join(raw_cache_dir, f'text8.{split}.txt')
                with fsspec.open(_path, 'r') as f:
                    splits[split] = f.read()

        # Chunk and save as datasets.DatasetDict
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        dataset_dict = {}
        for k, v in splits.items():
            if k == 'train' and crop_train == True:
                chunk_size = 2 * max_seq_length
            else:
                chunk_size = max_seq_length
            text = list(chunks(v, chunk_size))
            if drop_last and len(text[-1]) < chunk_size:
                text = text[:-1]
            dataset_dict[k] = Dataset.from_dict({'text': text})
        dataset = DatasetDict(dataset_dict)
        dataset.save_to_disk(cache_dir)
    else:
        dataset = load_from_disk(cache_dir)

    return dataset


# Modify this to use different datasets
def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8, **kwargs):

    if name == "wikitext103":
        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir, trust_remote_code=True)
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir, trust_remote_code=True)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir, trust_remote_code=True)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()        
    elif name == "openwebtext-train":
        dataset = load_dataset("openwebtext", split='train[:-100000]', cache_dir=cache_dir, trust_remote_code=True)
    elif name == "openwebtext-valid":
        dataset = load_dataset("openwebtext", split='train[-100000:]', cache_dir=cache_dir, trust_remote_code=True)
    elif name == 'text8':
        dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
    else:
        dataset = load_dataset(name, cache_dir=cache_dir, trust_remote_code=True)

    if name in ["lambada", "openwebtext-train", "openwebtext-valid"]:
        data = dataset
    else:
        data = dataset[mode]

    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text
        return detok

    tokenizer = get_tokenizer(name, eval_data=kwargs.get('eval_data', False))
    if name != "text8":
        EOS = tokenizer.encode(tokenizer.eos_token)[0]
        BOS = tokenizer.encode(tokenizer.bos_token)[0]

    def preprocess_and_tokenize(example):
        if name == "ptb":
            text = example['sentence']
        else:
            text = example["text"]
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        if name == "text8":
            tokens = tokenizer(text, return_attention_mask=False)

        else:
            tokenizer.padding_side = 'right'
            tokenizer.truncation_side = 'right'

            tokens = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False
            )
            tokens = {'input_ids': [t + [EOS] for t in tokens['input_ids']]}
        return tokens
    
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')
    

    def group_texts(examples):
        if name == "text8":
            # Concatenate all texts.
            concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        else:
            concatenated_examples = list(itertools.chain(* examples['input_ids']))
            total_length = len(concatenated_examples)
            new_block_size = block_size - 2  # [BOS] and [EOS] to be added
            total_length = (total_length // new_block_size) * new_block_size
            # Split by chunks of max_len.
            result = {}
            _values = []
            _attn_masks = []
            for i in range(0, total_length, new_block_size):
                _values.append(
                [BOS]
                + concatenated_examples[i : i + new_block_size]
                + [EOS])
                _attn_masks.append(torch.ones(block_size))
            result['input_ids'] = _values
            result['attention_mask'] = _attn_masks
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    chunked_dataset = chunked_dataset.with_format('torch')

    return chunked_dataset


# Dataloader for training and validation
def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")

    train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length)
    valid_mode = "validation" if config.data.valid not in ["lm1b"] else "test"
    valid_set = get_dataset(config.data.valid, valid_mode, cache_dir=config.data.cache_dir, block_size=config.model.length)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        val_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=(val_sampler is None),
    ))
    return train_loader, valid_loader


# Dataloader for test
def get_test_dataloaders(config, distributed=True):
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")

    test_set = get_dataset(config.data.valid, "test", cache_dir=config.data.cache_dir, block_size=config.model.length)

    if distributed:
        test_sampler = DistributedSampler(test_set)
    else:
        test_sampler = None
    
    valid_loader = cycle_loader(DataLoader(
        test_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=False,
        shuffle=(test_sampler is None),
    ))
    return valid_loader