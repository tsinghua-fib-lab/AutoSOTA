import os
import pickle
import argparse
from config import lazy_interval
from src.probe.train import load_probe_model, build_prompt_samples, collect_activations, MODEL_LAYER_DEFAULTS
from data import get_dataset_generator

import numpy as np
from src.utils import load_data
from typing import Dict, List

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--data_name', type=str, default='mmlu')
    parser.add_argument('--layer_idx', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--classifier_type', type=str, default='probe')
    return parser


def load_classifier(model_name, clf_name):
    layer_idx = MODEL_LAYER_DEFAULTS.get(model_name, None)
    model_path = f"probe/{model_name}/{clf_name}_model_layer{layer_idx}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classifier model file not found at {model_path}")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    return clf


def fixed_sample_indices(abs_span, lazy_interval = lazy_interval) -> List[int]:
    # lazy_interval is the interval at which to sample tokens
    return [abs_span[i] for i in range(0, len(abs_span), lazy_interval)]


def calibrate_thresholds(prompt_samples, tokenizer, model, clf_model, alpha: float, layer_idx: int, batch_size: int, lazy_interval: int):
    dataset = collect_activations(
        prompt_samples,
        tokenizer,
        model,
        layer_idx,
        batch_size,
        lambda abs_span: fixed_sample_indices(abs_span, lazy_interval)
    )
    X = np.stack([item["x"] for item in dataset])

    try:
        preds = clf_model.predict_proba(X)[:, 1]  # probability of the positive class
    except Exception as e:
        preds = clf_model.predict(X)

    # group the predictions by whether it comes from the same prompt
    prompt_dict: Dict[int, List[float]] = {}
    for i, item in enumerate(dataset):
        prompt_index = item["meta"]["prompt_index"]
        if prompt_index not in prompt_dict:
            prompt_dict[prompt_index] = []
        prompt_dict[prompt_index].append(preds[i])

    # need to select the threshold that only rejects at most alpha fraction of negative positives
    # prompt_dict maps prompt_index -> list of prediction scores, so iterate over values()
    max_by_prompt = [max(vals) for vals in prompt_dict.values() if vals]
    max_by_prompt = np.array(max_by_prompt)
    n = len(max_by_prompt)
    T = float(np.quantile(max_by_prompt, (1.0 - alpha) * (n+1) / n))
    frac_rejected = float(np.mean(max_by_prompt > T))
    return {
        "threshold": T,
        "fraction_rejected": frac_rejected,
    }

class ProbeBasedStoppingRule():
    """
    A stopping rule based on probe model predictions.
    """
    def __init__(self, model_name, data_name, batch_size: int = 4, lazy_interval: int = 250, alpha: float = 0.05, clf_name: str = "probe"):
        self.data_name = data_name
        self.probe_tokenizer, self.probe_model = load_probe_model(model_name)
        self.probe_classifier = load_classifier(model_name, clf_name)
        self.threshold = None
        calibration_data = load_data(model_name, "gsm8k", split = 'test')
        data_generator = get_dataset_generator(data_name)
        test_data = load_data(model_name, data_name, split = 'full')
        self.calibration_data = build_prompt_samples(calibration_data, data_generator, self.probe_tokenizer)
        # calibration data is only one sided
        self.calibration_data = [item for item in self.calibration_data if item["label"] == 0]
        self.test_data = build_prompt_samples(test_data, data_generator, self.probe_tokenizer)
        self.alpha = alpha
        self.lazy_interval = lazy_interval
        self.layer_idx = MODEL_LAYER_DEFAULTS.get(model_name, None)
        self.batch_size = batch_size

    def calibrate(self):
        self.threshold = calibrate_thresholds(
            self.calibration_data,
            self.probe_tokenizer,
            self.probe_model,
            self.probe_classifier,
            self.alpha,
            self.layer_idx,
            self.batch_size,
            self.lazy_interval
        )
        # save the threshold
        print(f"Calibrated threshold: {self.threshold['threshold']}, fraction rejected: {self.threshold['fraction_rejected']}")

    def test(self):
        test_dataset = collect_activations(
            self.test_data,
            self.probe_tokenizer,
            self.probe_model,
            self.layer_idx,
            self.batch_size,
            lambda abs_span: fixed_sample_indices(abs_span, self.lazy_interval)
        )
        X_test = np.stack([item["x"] for item in test_dataset])
        y = np.array([item["label"] for item in test_dataset])
        try:
            preds = self.probe_classifier.predict_proba(X_test)[:, 1]  # probability of the positive class
        except Exception as e:
            preds = self.probe_classifier.predict(X_test)
        # Apply the threshold to determine stopping points
        # for testing, there are two settings: normal vs alternative -- currently this is not distinguished
        # for normal, we only calculate early stopping rate
        # for alternative, we calculate tokens saved as well
        test_data_well_posed = [item for item, label in zip(test_dataset, y) if label == 0]
        test_data_ill_posed = [item for item, label in zip(test_dataset, y) if label == 1]
        preds = np.array(preds)
        preds_well_posed = preds[y == 0]
        early_stopping_count = 0
        total_count = 0
        i = 0
        while i < len(preds_well_posed):
            # find out the last index with the same row_index
            j = i + 1
            while j < len(preds_well_posed):
                if test_data_well_posed[j]["meta"]["row_index"] != test_data_well_posed[i]["meta"]["row_index"]:
                    break
                j += 1
            chunk_preds = preds_well_posed[i:j]
            if np.any(chunk_preds >= self.threshold["threshold"]):
                early_stopping_count += 1
            i = j
            total_count += 1
        early_stopping_rate_well_posed = early_stopping_count / total_count * 100 if total_count > 0 else 0
            
        preds_ill_posed = preds[y == 1]
        # For the alternative setting, calculate tokens saved
        tokens_saved_list = []
        percentage_saved_list = []
        i = 0
        while i < len(preds_ill_posed):
            # find out the last index with the same row_index
            j = i + 1
            while j < len(preds_ill_posed):
                if test_data_ill_posed[j]["meta"]["row_index"] != test_data_ill_posed[i]["meta"]["row_index"]:
                    break
                j += 1
            chunk_preds = preds_ill_posed[i:j]
            stop_indices = np.where(chunk_preds >= self.threshold["threshold"])[0]
            if stop_indices.size > 0:
                stop_index = i + stop_indices[0]
                token_index = test_dataset[stop_index]["meta"]['token_index']  # stop index at the token level
                total_tokens = test_dataset[stop_index]["meta"]['total_tokens']
                tokens_saved = total_tokens - token_index
                percentage_saved = tokens_saved / total_tokens * 100
            else:
                tokens_saved = 0
                percentage_saved = 0.0
            tokens_saved_list.append(tokens_saved)
            percentage_saved_list.append(percentage_saved)
            i = j
        return early_stopping_rate_well_posed, tokens_saved_list, percentage_saved_list


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.classifier_type == "probe":
        clf_name = "probe"
    elif args.classifier_type == "pls":
        clf_name = "pls"
    else:
        raise ValueError(f"Unknown classifier type: {args.classifier_type}")

    stopping_rule = ProbeBasedStoppingRule(
        model_name=args.model_name,
        data_name=args.data_name,
        batch_size=args.batch_size,
        lazy_interval=args.batch_size,
        alpha=0.05,
        clf_name=clf_name
    )

    threshold_path = f"probe/{args.model_name}/{clf_name}_stopping_threshold.pkl"
    if os.path.exists(threshold_path):
        with open(threshold_path, "rb") as f:
            stopping_rule.threshold = pickle.load(f)
        print(f"Loaded calibrated threshold from {threshold_path}")
    else:
        stopping_rule.calibrate()
        # save the calibrated threshold
        os.makedirs(os.path.dirname(threshold_path), exist_ok=True)
        with open(threshold_path, "wb") as f:
            pickle.dump(stopping_rule.threshold, f)
    print(f"Calibrated threshold saved to {threshold_path}")
    early_stopping_rate_well_posed, tokens_saved_list, percentage_saved_list = stopping_rule.test()
    avg_tokens_saved = sum(tokens_saved_list) / len(tokens_saved_list) if tokens_saved_list else 0
    avg_percentage_saved = sum(percentage_saved_list) / len(percentage_saved_list) if percentage_saved_list else 0
    early_stopping_rate = sum(ts > 0 for ts in tokens_saved_list) / len(tokens_saved_list) * 100 if tokens_saved_list else 0

    # save the results
    results_path = f"probe/{args.model_name}/{args.data_name}_{clf_name}_stopping_results.jsonl"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results = {
        "stopping_rule": "ProbeBasedStoppingRule",
        "early_stopping_rate_well_posed": early_stopping_rate_well_posed,
        "avg_tokens_saved_ill_posed": avg_tokens_saved,
        "avg_percentage_saved_ill_posed": avg_percentage_saved,
        "early_stopping_rate_ill_posed": early_stopping_rate,
        "tokens_saved_list_ill_posed": tokens_saved_list,
    }
    with open(results_path, "w") as f:
        f.write(str(results) + "\n")
    print(f"Results saved to {results_path}")