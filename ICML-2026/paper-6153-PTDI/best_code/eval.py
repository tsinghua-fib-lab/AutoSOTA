
import zlib
import os
import argparse
from utils.PTDI import PowerEnhancedStableEstimator, SelectivePrediction, out_calibrated_sampling
from utils.utils import seed_everything
from utils.tools import *
from tqdm import tqdm
import torch
import numpy as np
from peft import PeftConfig, PeftModel
from datasets import load_dataset
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle

def flatten_dict(result_metrics,alphas,keys =None):
    # Initialize an empty dictionary to hold the flattened structure
    flattened = {}

    # Loop through each metric in the result_metrics dictionary
    for metric, values in result_metrics.items():
        # Loop through each key in the inner dictionary
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if keys:
                if key not in keys:
                    continue

            if np.isscalar(value):
                new_key = f"{metric}_{key}"
                flattened[new_key] = float(value)
                print(f"{new_key}: {value}")
                continue
            # Create a new key by combining the metric and the original key
            for idx, alpha in enumerate(alphas):
                
                new_key = f"{metric}_{key}_alpha_{alpha}"
                # Add this to the flattened dictionary

                flattened[new_key] = float(value[idx])
                print(f"{new_key}: {value[idx]}")
    return flattened


def to_numpy(q_hat):
    if isinstance(q_hat, torch.Tensor):
        return q_hat.detach().cpu().numpy()  
    elif isinstance(q_hat, np.ndarray):
        return q_hat  
    else:
        return np.array(q_hat) 

def convert_huggingface_data_to_lists_by_label(dataset):
    label_0_data = []
    label_1_data = []
    
    for i in range(len(dataset)):
        ex = dataset[i]
        if ex['label'] == 0:
            label_0_data.append(ex)
        else:
            label_1_data.append(ex)
    
    return label_1_data, label_0_data 

def load_model(args, fine_tuning):

    if not fine_tuning:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, return_dict=True, device_map="auto"
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model_path = args.model_path
    elif fine_tuning:
        config = PeftConfig.from_pretrained(args.fine_tuned_para)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, device_map="auto"
        )
        lora_model = PeftModel.from_pretrained(model, args.fine_tuned_para)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = lora_model
        model.eval()
        model_path = args.fine_tuned_para
    print("model path: ", model_path)
    return model, tokenizer

def compute_p_values(eval_scores, cal_scores, high_score_is_positive=True):
    """
    Convert scores to conformal p-values (row-wise).
    """
    n_exp, n_samples = eval_scores.shape
    n_cal = cal_scores.shape[1]

    cal_sorted = np.sort(cal_scores, axis=1)
    p_values = np.zeros((n_exp, n_samples), dtype=np.float64)

    for i in range(n_exp):
        row_cal = cal_sorted[i]
        row_eval = eval_scores[i]

        if high_score_is_positive:
            idx = np.searchsorted(row_cal, row_eval, side='left')
            p_values[i] = (n_cal - idx + 1.0) / (n_cal + 1.0)
        else:
            idx = np.searchsorted(row_cal, row_eval, side='right')
            p_values[i] = (idx + 1.0) / (n_cal + 1.0)

    return np.clip(p_values, 1e-10, 1.0 - 1e-10)



def inference(model, tokenizer, sentence, example):
    pred = {}
    p1, all_prob, p1_likelihood = calculatePerplexity(
        sentence, model, tokenizer, gpu=model.device
    )

    p_lower, _, p_lower_likelihood = calculatePerplexity(
        sentence.lower(), model, tokenizer, gpu=model.device
    )

    pred["ppl"] = p1  # ppl

    # Ratio of log ppl of lower-case and normal-case
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()

    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(sentence, "utf-8")))
    pred["ppl/zlib"] = np.log(p1) / zlib_entropy

    # min-k
    ratio = 0.2
    k_length = int(len(all_prob) * ratio)
    topk_prob = np.sort(all_prob)[:k_length]
    pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()
    return pred

def get_dataset(args):
    if args.dataset_name == "WikiMIA":
        text_lens = [32, 64, 128, 256]
        all_data = []
        for tl in text_lens:
            split_name = f"WikiMIA_length{tl}"
            dataset = load_dataset("swj0419/WikiMIA", split=split_name)
            all_data.extend(dataset)
        member, non_member = convert_huggingface_data_to_lists_by_label(all_data)
    else:
        raise ValueError("Please check if the dataset name is valid.")

    return member, non_member


def perform_ptdi( member_score_value, non_member_score_value, metrics, target_fdrs):
    result_metrics ={}
    results_total ={}
    target_fdrs = to_numpy(target_fdrs)
    for key in metrics:

        certain_member_score_value = -to_numpy(member_score_value[key] )
        certrain_non_member_score_value = - to_numpy(non_member_score_value[key] )
        
        selector = SelectivePrediction()
        
        member_matrix, non_member_matrix, cal_matrix = out_calibrated_sampling(member_score=certain_member_score_value.squeeze(), non_member_score=certrain_non_member_score_value.squeeze() )
        

        total_score = np.concatenate([member_matrix, non_member_matrix], axis=1)
        p_values = compute_p_values(total_score, cal_matrix)

        estimator = PowerEnhancedStableEstimator()
        estimate_pi = estimator.estimate_pi0_and_gamma(p_values)
        
        result_metrics[key], results_total[key] = selector.calculate_average_results_from_matrix(
        mem_p_score_matrix = -member_matrix, 
        non_mem_p_score_matrix = -non_member_matrix, 
        cal_p_score_matrix = -cal_matrix, 
        target_fdrs = target_fdrs,
        scale_ratio= estimate_pi
        )
        
            
    return result_metrics



def prepare_score_dict(data, args=None, cache_dir="score_cache", overwrite=False):
    """
    Prepare score dictionaries with disk cache to avoid repeated inference.

    Args:
        data: (member, non_member)
        args: argparse args (used for cache key)
        cache_dir: directory to save cached scores
        overwrite: if True, ignore cache and recompute

    Returns:
        member_score_dict, non_member_score_dict
    """

    os.makedirs(cache_dir, exist_ok=True)

    # ---------- build cache name ----------
    model_name = args.model_path.replace("/", "_")
    dataset_name = args.dataset_name
    cache_file = os.path.join(
        cache_dir,
        f"scores_{dataset_name}_{model_name}.pkl"
    )

    # ---------- load cache ----------
    if os.path.exists(cache_file) and not overwrite:
        print(f"[Cache] Loading cached scores from {cache_file}")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        return cache["member"], cache["non_member"]

    print("[Cache] No valid cache found. Running inference...")

    # ---------- inference ----------
    model, tokenizer = load_model(args, fine_tuning=False)

    member_score_dict = defaultdict(list)
    non_member_score_dict = defaultdict(list)

    member, non_member = data

    print("Processing member data...")
    for example in tqdm(member):
        text = example["input"]
        scores = inference(model, tokenizer, text, example)
        for metric, score in scores.items():
            member_score_dict[metric].append(score)

    print("\nProcessing non-member data...")
    for example in tqdm(non_member):
        text = example["input"]
        scores = inference(model, tokenizer, text, example)
        for metric, score in scores.items():
            non_member_score_dict[metric].append(score)

    # ---------- save cache ----------
    cache = {
        "member": dict(member_score_dict),
        "non_member": dict(non_member_score_dict),
    }
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

    print(f"[Cache] Scores saved to {cache_file}")

    return member_score_dict, non_member_score_dict


def calculatePerplexity(sentence, model, tokenizer, gpu):

    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]  # loss, scale
    
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]

    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="WikiMIA",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="huggyllama/llama-7b",
        help="the model to infer",
    )

    parser.add_argument("--fine_tuned_para", default="")

    parser.add_argument("--dataset", type=str, default="WikiMIA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--len", type=int, default=32)
    parser.add_argument("--train_size", type=float, default=0.3)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    seed_everything(args.seed)
    data = get_dataset(args)
    member_score_dict, non_member_score_dict = prepare_score_dict(data, args)
    
    target_fdrs = [0.1,0.2,0.3,0.4,0.5]
    result = perform_ptdi(member_score_dict, non_member_score_dict, metrics= ['Min_20.0% Prob'], target_fdrs=target_fdrs)
    save_dict =flatten_dict(result, target_fdrs, keys= ["FDP","Power"])
