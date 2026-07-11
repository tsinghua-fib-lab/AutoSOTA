import os
from dotenv import load_dotenv

load_dotenv()

from transformers import AutoTokenizer
from src.detect.uncertainty_keywords import get_uncertainty_keywords

import argparse
from src.utils import load_data, save_jsonl_for_simulation

from src.stopping_rules import *
from src.detect.classifier import SoftUpperbound

def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--data_name', type=str, default='icraft',
                        help='Choose among: gpqa, gsm8k, mmlu, umwp, mc, mip, hle')
    parser.add_argument('--results_path', type=str, default='results',
                        help='Path to save the results')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for uncertainty stopping rule')
    parser.add_argument('--ablation', action='store_true', help='''Set to true to run ablation studies''')
    return parser

class DecodingTestEnvironment:
    """
    A class to set up the decoding test environment.
    It initializes the stopping rule and runs the simulation for each reasoning trace in the dataset.
    """
    def __init__(self, model_name: str, data, stopping_rule: StoppingRule, mode: str = 'normal'):
        assert mode in ['normal', 'alternative'], "Mode must be either 'normal' or 'alternative'."
        self.model_name = model_name
        self.stopping_rule = stopping_rule
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mode = mode
        if data is None:
            raise FileNotFoundError(f"Data for {model_name} not found.")
        if self.mode == 'normal':
            self.reasoning_traces = [row.get("reasoning_well_posed", "") for row in data]
        else:
            self.reasoning_traces = [row.get("reasoning_ill_posed", "") for row in data]

    def run_simulation(self):
        tokens_saved_list = []
        percentage_saved_list = []
        tokens_saved_list, percentage_saved_list = self.stopping_rule.test(self.reasoning_traces)

        avg_tokens_saved = sum(tokens_saved_list) / len(tokens_saved_list) if tokens_saved_list else 0
        avg_percentage_saved = sum(percentage_saved_list) / len(percentage_saved_list) if percentage_saved_list else 0
        early_stopping_rate = sum(ts > 0 for ts in tokens_saved_list) / len(tokens_saved_list) * 100 if tokens_saved_list else 0
        
        return avg_tokens_saved, avg_percentage_saved, early_stopping_rate, tokens_saved_list

def run_simulation(model_name, test_data, train_reasoning_traces, soft_upperbound, rule, lazy_interval, path, ablation, kwset, alpha=0.05):
        
    if rule == 'Length':
        stopping_rule = LengthStoppingRule(model_name, alpha=alpha)
    elif rule == 'Renewal':
        stopping_rule = UncertaintyArrivalStoppingRule(model_name, lazy_interval=lazy_interval, alpha=alpha, uncertainty_keywords=kwset)
    elif rule == 'Maxwise':
        stopping_rule = MaxUncertaintyStoppingRule(model_name, lazy_interval=lazy_interval, alpha=alpha, uncertainty_keywords=kwset)
    else:
        raise ValueError(f"Unknown stopping rule: {rule}")
    
    stopping_rule.calibrate(train_reasoning_traces)

    decoding_env_ill_posed = DecodingTestEnvironment(model_name=model_name, data=test_data, stopping_rule=stopping_rule, mode='alternative')
    avg_tokens_saved_ill_posed, avg_percentage_saved_ill_posed, early_stopping_rate_ill_posed, tokens_saved_list_ill_posed = decoding_env_ill_posed.run_simulation()

    decoding_env = DecodingTestEnvironment(model_name=model_name, data=test_data, stopping_rule=stopping_rule, mode='normal')
    _, _, early_stopping_rate, tokens_saved_list_well_posed = decoding_env.run_simulation()
    results = {
        "stopping_rule": rule,
        "alpha": stopping_rule.alpha,
        "lazy_interval": lazy_interval,
        "ablation": ablation,
        "early_stopping_rate_well_posed": early_stopping_rate,
        "tokens_saved_list_well_posed": tokens_saved_list_well_posed,
        "avg_tokens_saved_ill_posed": avg_tokens_saved_ill_posed,
        "avg_percentage_saved_ill_posed": avg_percentage_saved_ill_posed,
        "early_stopping_rate_ill_posed": early_stopping_rate_ill_posed,
        "tokens_saved_list_ill_posed": tokens_saved_list_ill_posed,
        "soft_upperbound": soft_upperbound.get_soft_upperbound_at_alpha(max(early_stopping_rate / 100, 0.05)) * 100 if rule != 'Length' else None,
        "cal_length_quartiles": stopping_rule.cal_length_quantiles if rule == 'Length' else None,
        "test_length_quartiles": stopping_rule.test_length_quantiles if rule == 'Length' else None,
    }
    save_jsonl_for_simulation(path, results)

def main(model_name, data_name, results_path, path, alpha=0.05, ablation=False):

    if data_name in ['mmlu', 'mc', 'umwp', 'mip', 'gpqa', 'hle']:
        cal_data = load_data(model_name, "gsm8k", split = 'test', results_path="results")
        test_data = load_data(model_name, data_name, split = 'full', results_path=results_path)
    else:
        cal_data, test_data = load_data(model_name, data_name, split = 'both', results_path=results_path)
    
    soft_upperbound = SoftUpperbound(test_data)

    train_reasoning_traces = [row.get("reasoning_well_posed", "") for row in cal_data]
    
    kwset = get_uncertainty_keywords(ablation="none")
    if not ablation:
        run_simulation(model_name, test_data, train_reasoning_traces, soft_upperbound, 'Length', "none", path, "none", []) 
    for rule in ['Renewal', 'Maxwise']:
        if ablation:
            for lazy_interval in [100, 500]:
                run_simulation(model_name, test_data, train_reasoning_traces, soft_upperbound, rule, lazy_interval, path, f"none", kwset)

            from config import lazy_interval
            run_simulation(model_name, test_data, train_reasoning_traces, soft_upperbound, rule, lazy_interval, path, "add_maybe", kwset + ['maybe', 'perhaps'])

            run_simulation(model_name, test_data, train_reasoning_traces, soft_upperbound, rule, lazy_interval, path, "add_wait", kwset + ['wait', 'alternatively'])

            for item in get_uncertainty_keywords(ablation="loo"):
                kwset_, category = item
                run_simulation(model_name, test_data, train_reasoning_traces, soft_upperbound, rule, lazy_interval, path, f"loo_{category}", kwset_)
        else: 
            from config import lazy_interval
            run_simulation(model_name, test_data, train_reasoning_traces, soft_upperbound, rule, lazy_interval, path, "none", kwset, alpha=alpha)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    model_name = args.model_name
    data_name = args.data_name
    results_path = args.results_path
    ablation = args.ablation
    alpha = args.alpha

    path = f"{results_path}/{model_name}/{data_name}_stopping_rule_results.jsonl"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    main(model_name, data_name, results_path, path, alpha=alpha, ablation=ablation)

    # Save main results
