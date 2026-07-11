import os
import pickle
import argparse
import numpy as np
import json

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--data_name', type=str, default='mmlu')
    return parser


def calibrate_thresholds(data, alpha: float):
    max_by_row = np.max(data, axis=1)
    n = len(max_by_row)
    T = float(np.quantile(max_by_row, (1.0 - alpha) * (n+1) / n))
    return T

def get_power(data, T):
    max_by_row = np.max(data, axis=1)
    power = np.mean(max_by_row >= T)
    return power

class Matrix():
    def __init__(self, metric_name=None):
        self.pad_value = 0 if metric_name == "entropy" else 1
        self.metric_name = metric_name
    
    def fit_transform(self, data):
        max_length = max(len(lst) for lst in data)
        X = np.array([lst + [self.pad_value] * (max_length - len(lst)) for lst in data])
        if self.metric_name == "deer":
            X = 1 - X
        return X
        

class LogitsBasedStoppingRule():
    def __init__(self, data, metric_name, alpha: float = 0.05, cold_start_index: int = 0, threshold: float = 0.0):
        self.metric_name = metric_name
        self.alpha = alpha
        self.data = data
        self.matrix = Matrix(metric_name=self.metric_name)
        self.cold_start_index = cold_start_index
        self.threshold = threshold

    def train(self):
        """
        There are two paramters to calibrate:
        1. Cold start index: the number of tokens to wait before making a decision.
        2. Threshold: the maximum allowed value of the stopping rule metric.
        To do this, we impliment a search over the possible values of the cold start index.
        For each cold start index, we compute the threshold using the calibration data and the provided alpha value.
        Then, we choose the stopping rule with the highest power of the stopping rule.
        """
        X, y = self.data['train']['X'], self.data['train']['y']
        train_X = self.matrix.fit_transform(X)
        train_y = np.array(y)
        self.cold_start_index = 0
        self.train_power = 0
        
        for cold_start_index in range(min(len(train_X[0]), 10)):
            threshold = calibrate_thresholds(train_X[:, cold_start_index:][train_y == 0], self.alpha)
            power = get_power(train_X[:, cold_start_index:][train_y == 1], threshold)
            if power > self.train_power:
                self.train_power = power
                self.cold_start_index = cold_start_index
        
        return self.cold_start_index

    def calibrate(self):
        X, y = self.data['cal']['X'], self.data['cal']['y']
        cal_y = np.array(y)
        cal_X = self.matrix.fit_transform(X)[cal_y == 0]
        self.threshold = calibrate_thresholds(cal_X[:, self.cold_start_index:], self.alpha)
        return self.threshold

    def test(self):
        X, y = self.data['test']['X'], self.data['test']['y']
        test_X = self.matrix.fit_transform(X)
        test_y = np.array(y)
        if self.cold_start_index > len(test_X[0]):
            print("Cold start index is larger than the length of the test data")
            return 0, 0
        max_by_row = np.max(test_X[:, self.cold_start_index:], axis=1)
        early_stopping_rate_well_posed = np.mean(max_by_row[test_y == 0] >= self.threshold) * 100
        early_stopping_rate_ill_posed = np.mean(max_by_row[test_y == 1] >= self.threshold) * 100
        return early_stopping_rate_well_posed, early_stopping_rate_ill_posed

def load_data(method_dict, split, path):
    with open(path, "r") as f:
        for line in f:
            data = eval(line.strip())
            method_dict['data'][split]['X'].append(data[method_dict["metric_name"]])
            method_dict['data'][split]['y'].append(data["label"])

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    model_name = args.model_name
    data_name = args.data_name

    results_path = f"logits/{model_name}/{data_name}_stopping_results.jsonl"
    # if it exists, remove it
    if os.path.exists(results_path):
        os.remove(results_path)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    name_dict = {
        "deer": {
            "metric_name": "pred_prob_list",
            "stopping_rule": "DEERBasedStoppingRule",
            "data": {"train": {"X": [], "y": []}, "cal": {"X": [], "y": []}, "test": {"X": [], "y": []}}
        },
        "entropy": {
            "metric_name": "H_bits_list",
            "stopping_rule": "EntropyBasedStoppingRule",
            "data": {"train": {"X": [], "y": []}, "cal": {"X": [], "y": []}, "test": {"X": [], "y": []}}
        }
    }

    for key, value in name_dict.items():
        load_data(value, 'test', f"logits/{model_name}/{data_name}_pred_prob_entropy.jsonl")
        threshold_path = f"logits/{model_name}/{key}_stopping_threshold.pkl"
        if os.path.exists(threshold_path):
            with open(threshold_path, "rb") as f:
                cold_start_index, threshold = pickle.load(f)
            stopping_rule = LogitsBasedStoppingRule(
                value['data'], key, cold_start_index=cold_start_index, threshold=threshold
            )
        else:
            train_path = f"logits/{model_name}/gsm8k_train_pred_prob_entropy.jsonl"
            cal_path = f"logits/{model_name}/gsm8k_cal_pred_prob_entropy.jsonl"
            for split, path in zip(["train", "cal"], [train_path, cal_path]):
                load_data(value, split, path)
            
            stopping_rule = LogitsBasedStoppingRule(
                value['data'], key
            )
            cold_start_index = stopping_rule.train()
            threshold = stopping_rule.calibrate()
            print(f"Calibrated {key} stopping rule with cold start index {cold_start_index} and threshold {threshold}")
            # save the calibrated threshold
            os.makedirs(os.path.dirname(threshold_path), exist_ok=True)
            with open(threshold_path, "wb") as f:
                pickle.dump((cold_start_index, threshold), f)
        
        early_stopping_rate_well_posed, early_stopping_rate_ill_posed = stopping_rule.test()
        
        # save the results
        results = {
            "stopping_rule": value['stopping_rule'],
            "early_stopping_rate_well_posed": early_stopping_rate_well_posed,
            "early_stopping_rate_ill_posed": early_stopping_rate_ill_posed,
        }
        with open(results_path, "a") as f:
            f.write(json.dumps(results) + '\n')