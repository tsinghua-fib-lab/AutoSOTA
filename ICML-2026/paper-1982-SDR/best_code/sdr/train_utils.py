import argparse
import random
import os
from datasets import Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model import LLMWithScorer

def parse_args():
    parser = argparse.ArgumentParser(description="train calibration head")

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--embed_path', type=str, default=None)

    parser.add_argument('--slm_name', type=str, default='slm')
    parser.add_argument('--llm_names', type=str, nargs='+', default='llm')

    # data arguments
    parser.add_argument('--dataset_name', type=str, default='mmlu',
                        help='name of the dataset')
    parser.add_argument('--local_answer_paths', type=str, nargs='+',
                        help='path of the local answers')
    parser.add_argument('--remote_answer_paths', type=str, nargs='+',
                        help='path of the remote answers')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='ratio of train data (among all data)')
    parser.add_argument('--eval_ratio', type=float, default=0.1,
                        help='ratio of eval data (among train data)')
    
    # training arguments
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_train_epochs', type=float, default=2.0)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--loss_type', type=str, default='pointwise')

    # model (decision module) arguments
    parser.add_argument('--input_layer', type=float, default=-2)
    parser.add_argument('--head_model', type=str, default='transformer')
    parser.add_argument('--num_remote', type=int, default=1)
    parser.add_argument('--multi_remote_strategy', type=str, default='head')


    args = parser.parse_args()
    args_dict = vars(args)

    # split arguments
    model_keys = ['model_path', 'input_layer', 'head_model', 'embed_path', 'num_remote', 'slm_name', 'llm_names', 'multi_remote_strategy']
    data_keys = ['dataset_name', 'dataset_splits', 'local_answer_paths', 'remote_answer_paths', 'use_all_token', 'seed', 'train_ratio', 'eval_ratio']
    train_keys = ['output_dir', 'per_device_train_batch_size', 'per_device_eval_batch_size', 'gradient_accumulation_steps', 'lr_scheduler_type', 'logging_steps', 'save_steps', 'eval_steps', 'warmup_steps',
                  'learning_rate', 'num_train_epochs', 'bf16', 'overwrite_output_dir', 'seed', 'save_total_limit', 'loss_type']

    model_args = {k: args_dict[k] for k in model_keys if k in args_dict}
    data_args = {k: args_dict[k] for k in data_keys if k in args_dict}
    train_args = {k: args_dict[k] for k in train_keys if k in args_dict}

    print(f"model arguments:\n{model_args}\n")
    print(f"data arguments:\n{data_args}\n")
    print(f"training arguments:\n{train_args}\n")

    return model_args, data_args, train_args

def get_dataset(dataset_name, local_answer_paths, remote_answer_paths, seed, train_ratio, tokenizer, **kwargs):
    # append local and remote metrics to the dataset
    local_answers = [pd.read_parquet(local_answer_path) for local_answer_path in local_answer_paths]
    remote_answers = [pd.read_parquet(remote_answer_path) for remote_answer_path in remote_answer_paths]

    if len(remote_answers) > len(local_answers): # multi-remote
        assert len(remote_answers) % len(local_answers) == 0, "incorrect number of remote answer files"
        concated_remote_answers = []
        num_remotes = len(remote_answers) // len(local_answers)
        for i in range(0, len(remote_answers), num_remotes):
            metrics = pd.concat([df.loc[:,'metric'] for df in remote_answers[i:i+num_remotes]], axis=1).values.tolist()
            concated_remote_answers.append(pd.DataFrame({'metric': metrics}))
        remote_answers = concated_remote_answers

    datasets = []
    num_samples = 0
    for local_answer, remote_answer in zip(local_answers, remote_answers):
        datasets.append(
            Dataset.from_dict({
                "prompt": local_answer["prompt"].tolist(),
                "predict": local_answer["predict"].tolist(),
                "local_metric": local_answer["metric"].tolist(),
                "remote_metric": remote_answer["metric"].tolist(),
                "index": list(range(num_samples, num_samples + len(local_answer))) # assign unique index for every sample to extract embeddings
            })
        )
        num_samples += len(local_answer)

    if len(local_answers) == 1: # split train and test sets within it
        random.seed(seed)
        indices = list(range(len(datasets[0])))
        random.shuffle(indices)
        train_dataset = datasets[0].select(indices[:int(train_ratio*len(indices))])
        test_dataset = datasets[0].select(indices[int(train_ratio*len(indices)):])   
    else: # use first two
        train_dataset, test_dataset = datasets[0], datasets[1]           

    def preprocess_function_mmlu_and_gsm8k(ex):
        model_inputs = tokenizer(ex['prompt'] + ex['predict'])
        model_inputs["advantages"] = np.array(ex['remote_metric']) - np.array(ex['local_metric'])
        model_inputs["base_metric"] = np.array(ex['local_metric'])
        return model_inputs
    
    def preprocess_function_squad(ex):
        model_inputs = tokenizer(ex['prompt'] + ex['predict'])
        model_inputs["advantages"] = (np.array(ex['remote_metric']) - np.array(ex['local_metric'])) / 100
        model_inputs["base_metric"] = np.array(ex['local_metric']) / 100
        return model_inputs

    removes = ["prompt", "predict", "local_metric", "remote_metric"]
    if dataset_name.endswith("mmlu") or dataset_name.endswith("gsm8k"):
        train_dataset = train_dataset.map(preprocess_function_mmlu_and_gsm8k, batched=False, remove_columns=removes)
        test_dataset = test_dataset.map(preprocess_function_mmlu_and_gsm8k, batched=False, remove_columns=removes)
    elif dataset_name.endswith("squad"):
        train_dataset = train_dataset.map(preprocess_function_squad, batched=False, remove_columns=removes)
        test_dataset = test_dataset.map(preprocess_function_squad, batched=False, remove_columns=removes)
    else:
        raise ValueError(f"unsupport dataset {dataset_name}")

    return train_dataset, test_dataset

def load_model_and_tokenizer(model_path, input_layer, num_remote, head_model, embed_path, multi_remote_strategy, **kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LLMWithScorer(
        base_model_name=model_path, 
        input_layer=input_layer,
        num_remote=num_remote,
        head_model=head_model,
        multi_remote_strategy=multi_remote_strategy,
        embed_path=embed_path,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    return model, tokenizer

def compute_metrics(eval_pred):
    _, labels = eval_pred
    all_advantages, all_scores = labels

    aucs = []

    if all_advantages.ndim == 1:
        all_advantages = all_advantages[:, None]   # (N,) -> (N, 1)
    if all_scores.ndim == 1:
        all_scores = all_scores[:, None]   # (N,) -> (N, 1)
    for i in range(all_advantages.shape[1]):
        advantages = all_advantages[:, i]
        scores = all_scores[:, i]

        if advantages.sum() < 0:
            advantages = -advantages
            scores = -scores

        advantages_sorted = advantages[np.argsort(-scores)]
        auc = advantages_sorted.cumsum().sum()
        auc_norm = auc / (len(advantages) * advantages.sum())
        aucs.append(auc_norm)

    return {
        "auc_norm": sum(aucs) / len(aucs)
    }

def find_best_temperature(model, eval_dataset, data_collator, device="cuda"):
    """Sweep temperature values on validation set to find optimal tau for AUC."""
    import numpy as np
    from torch.utils.data import DataLoader
    
    data_loader = DataLoader(
        eval_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=data_collator
    )
    
    advantages = []
    scores = []
    model.eval()
    
    for batch in data_loader:
        advantages.append(batch['advantages'].numpy())
        with torch.no_grad():
            _, output = model(
                input_ids=batch['input_ids'].to(device), 
                attention_mask=batch['attention_mask'].to(device), 
                index=batch['index']
            )
        score = output.cpu().float().numpy()
        scores.append(score)
    
    advantages = np.concatenate(advantages)
    scores = np.concatenate(scores)
    
    if advantages.ndim == 1:
        advantages = advantages[:, None]
    if scores.ndim == 1:
        scores = scores[:, None]
    
    adv = advantages[:, 0]
    raw = scores[:, 0]
    
    best_tau = 1.0
    best_auc = -1
    
    for tau in np.arange(0.5, 2.05, 0.1):
        calibrated = np.tanh(raw / tau)
        if adv.sum() < 0:
            adv_use = -adv
            cal_use = -calibrated
        else:
            adv_use = adv
            cal_use = calibrated
        sorted_adv = adv_use[np.argsort(-cal_use)]
        auc = sorted_adv.cumsum().sum()
        auc_norm = auc / (len(adv_use) * adv_use.sum())
        
        if auc_norm > best_auc:
            best_auc = auc_norm
            best_tau = tau
    
    print(f"Best temperature on validation: tau={best_tau:.1f}, val_auc={best_auc:.6f}")
    return best_tau


def do_test(test_dataset, data_collator, model, output_dir, slm_name, llm_names, temperature=1.0):
    data_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=data_collator
    )
    advantages = []
    scores = []
    base_metrics = []
    device = next(model.parameters()).device
    model.eval()

    for batch in tqdm(data_loader):
        advantages.append(batch['advantages'].numpy())
        base_metrics.append(batch['base_metric'].numpy())

        with torch.no_grad():
            _, output = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), index=batch['index'])

        score = output.cpu().float().numpy()
        scores.append(score)

    advantages = np.concatenate(advantages)
    scores = np.concatenate(scores)
    base_metrics = np.concatenate(base_metrics)

    auc = compute_metrics((None, (advantages, scores)))['auc_norm']

    print(f'auc: {auc:.6f}')
    
    # prepare result file for multi-remote evaluate
    result = pd.DataFrame({
        f"{slm_name}-pred": [0.] * scores.shape[0], # data num
        f"{slm_name}-perf": base_metrics.squeeze()
    })

    if len(llm_names) == 1:
        llm_names *= scores.shape[1] # llm num
    assert len(llm_names) == scores.shape[1], "number of args.llm_names must be 1 or equals to number of remote LLMs"

    columns = [f"{llm}-perf" for llm in llm_names]
    if advantages.ndim == 1:
        advantages = advantages[:, None]
    result[columns] = base_metrics + advantages

    columns = [f"{llm}-pred" for llm in llm_names]
    result[columns] = np.tanh(scores / temperature)
    
    result.to_parquet(os.path.join(output_dir, "test_result.parquet"))
