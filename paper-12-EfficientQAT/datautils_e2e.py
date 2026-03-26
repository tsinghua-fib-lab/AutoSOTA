## code from qlora
import torch
from typing import Dict, Sequence
from datasets import load_dataset
import os
from itertools import chain
from pathlib import Path
from transformers import default_data_collator
import transformers
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
import copy
import numpy as np

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"



@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict


ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning or continue pre-train.
    """
    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'deita-6k':
            dataset = load_dataset("hkust-nlp/deita-6k-v0", split = "train")
            dataset = [row for row in dataset]
            return dataset
        elif dataset_name == 'deita-10k':
            dataset = load_dataset("hkust-nlp/deita-10k-v0", split = "train")
            dataset = [row for row in dataset]
            return dataset
        elif dataset_name == 'c4':
            try:
                # load from local file, a fast manner
                dataset = load_dataset("arrow",
                data_files={
                    "train": "/cpfs01/user/chenmengzhao/huggingface/datasets/allenai___json/allenai--c4-6fbe877195f42de5/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51/json-train-00000-of-00002.arrow",
                    "validation": "/cpfs01/user/chenmengzhao/huggingface/datasets/allenai___json/allenai--c4-efc3d4f4606f44bd/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/json-validation.arrow",
                },
                )
            except:
                dataset = load_dataset("allenai/c4","allenai--c4",
                data_files={
                    "train": "en/c4-train.00000-of-01024.json.gz",
                    "validation": "en/c4-validation.00000-of-00008.json.gz",
                },
                )
            return dataset
        elif dataset_name == 'redpajama':
            try:
                loacal_dataset = "/cpfs01/user/chenmengzhao/huggingface/datasets/togethercomputer___red_pajama-data-1_t-sample"
                dataset = load_dataset(loacal_dataset)
            except:
                dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")   
            if "validation" not in dataset.keys():
                validation_split = args.eval_dataset_size
                dataset["validation"] = load_dataset(
                    loacal_dataset,
                    split=f"train[:{validation_split}]",
                )
                dataset["train"] = load_dataset(
                    loacal_dataset,
                    split=f"train[{validation_split}:]",
                )
            return dataset  
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")
        

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'pt' or (dataset_format is None and args.dataset in ['c4', 'redpajama']):
            block_size = args.pt_context_len
            column_names = list(dataset["train"].features)
            text_column_name = "text" if "text" in column_names else column_names[0]

            def tokenize_function(examples):
                output = tokenizer(examples[text_column_name])
                return output
            tokenized_datasets = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                if total_length >= block_size:
                    total_length = (total_length // block_size) * block_size
                # Split by chunks of max_len.
                result = {
                    k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result
            dataset = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        # Remove unused columns for instruction-tuning
        if not dataset_format == 'pt':
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
            )
        return dataset

    # Load dataset.
    print(f"loading {args.dataset}")
    if args.dataset in ['c4', 'redpajama']:
        cache_dir = './cache'
        cache_dataloader = f'{cache_dir}/e2e_dataloader_{args.model_family}_{args.dataset}_{args.pt_context_len}.cache'
        if os.path.exists(cache_dataloader):
            dataset = torch.load(cache_dataloader)
            print(f"load dataset from {cache_dataloader}")
        else: 
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            dataset = load_data(args.dataset)
            dataset = format_dataset(dataset, args.dataset_format)
            torch.save(dataset, cache_dataloader)    
    elif args.dataset in ['deita-6k', 'deita-10k']:
        # Split train/eval for deita datasets
        raw_data = load_data(args.dataset)
        np.random.seed(0)
        train_raw_data = raw_data
        perm = np.random.permutation(len(raw_data))
        split = int(len(perm) * 0.98)
        train_indices = perm[:split]
        eval_indices = perm[split:]
        train_raw_data = [raw_data[i] for i in train_indices]
        eval_raw_data = [raw_data[i] for i in eval_indices]
        print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")
        from deita_dataset.train import SupervisedDataset, LazySupervisedDataset
        dataset_cls = LazySupervisedDataset
        train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer, conv_template = args.conv_temp, mask_user = args.mask_use)
        eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer, conv_template = args.conv_temp, mask_user = args.mask_use)  
    elif args.dataset == 'mix_deita_redpajama':
        cache_dir = './cache'
        cache_dataloader = f'{cache_dir}/dataloader_{args.model_family}_{args.dataset}_{args.pt_context_len}.cache'
        if os.path.exists(cache_dataloader):
            dataset = torch.load(cache_dataloader)
            print(f"load dataset from {cache_dataloader}")
        else:
            deita_dataset = load_data('deita-10k')
            np.random.seed(0)
            from datasets import concatenate_datasets, Dataset
            from deita_dataset.train import SupervisedDataset
            print('tokenizr deita, need a long time.')
            deita_dataset = SupervisedDataset(deita_dataset, tokenizer=tokenizer, conv_template = args.conv_temp, mask_user = args.mask_use)
            deita_dataset = Dataset.from_dict(
                {
                    "input_ids":deita_dataset.input_ids,
                    "labels":deita_dataset.labels,
                    "attention_mask":deita_dataset.attention_mask,
                }
            )
            dataset = load_data('redpajama')
            redpajama_dataset = format_dataset(dataset, 'pt')

            train_dataset = concatenate_datasets([deita_dataset,redpajama_dataset['train'].select(range(len(deita_dataset)))])
            dataset = {
                "train":train_dataset,
                "validation":redpajama_dataset['validation']
            }
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            torch.save(dataset, cache_dataloader)   
    else:
        dataset = load_data(args.dataset)
        dataset = format_dataset(dataset, args.dataset_format)
    print(f"loading {args.dataset} successfully")
    
    # Split train/eval, reduce size for other datasets
    if not args.dataset in ['deita-6k', 'deita-10k']:
        if args.do_eval or args.do_predict:
            if 'eval' in dataset:
                eval_dataset = dataset['eval']
            elif 'validation' in dataset:
                eval_dataset = dataset['validation']
            else:
                print('Splitting train dataset in train and validation according to `eval_dataset_size`')
                dataset = dataset["train"].train_test_split(
                    test_size=args.eval_dataset_size, shuffle=True, seed=42
                )
                eval_dataset = dataset['test']
            if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
                eval_dataset = eval_dataset.select(range(args.max_eval_samples))
            if args.group_by_length:
                eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
        if args.do_train:
            train_dataset = dataset['train']
            train_dataset = train_dataset.shuffle(seed=0)
            if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
                train_dataset = train_dataset.select(range(args.max_train_samples))
            if args.group_by_length:
                train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.dataset in ['c4', 'redpajama', 'deita-6k', 'deita-10k','mix_deita_redpajama']:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            train_on_source=args.train_on_source,
            predict_with_generate=args.predict_with_generate,
        )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )
    