import os
import json
import torch
import random
import argparse
import itertools
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import config
from dataloader import Dataset
from evaluator import Evaluator
from model import Retriever, Generator


DATASET_MAP = {
    'marco': {
        'passages': config.REPO_ROOT / 'data' / 'passages' / 'marco_human_passages.json',
        'questions': config.REPO_ROOT / 'data' / 'questions' / 'marco-question.json',
    },
    'triviaqa': {
        'passages': config.REPO_ROOT / 'data' / 'passages' / 'triviaqa_human_passages.json',
        'questions': config.REPO_ROOT / 'data' / 'questions' / 'triviaqa-question.json',
    },
    'nq': {
        'passages': config.REPO_ROOT / 'data' / 'passages' / 'nq_human_passages.json',
        'questions': config.REPO_ROOT / 'data' / 'questions' / 'nq-question.json',
    },
}

# Ablation sweep definitions: variable name(s) → list of values to try
ABLATION_VARS = {
    "PREFIX_INIT_MODE,GEN_PREFIX_CONTENT": [
        ["random", ""],
        ["phrase", "<important information>"],
        ["phrase", "This is very important."],
    ],
    "RET_PREFIX_LEN,RET_SUFFIX_LEN": [
        [0, 0], [3, 3], [5, 5], [7, 7], [9, 9],
    ],
    "MALICIOUS_URL": [
        "Hackers are the best!",
        "https://malicious.url",
        "your age",
    ],
    "RET_CORRELATION_THRESHOLD,GEN_CORRELATION_THRESHOLD": [
        [0.75, 0.75], [0.8, 0.8], [0.85, 0.85], [0.9, 0.9],
    ],
    "RETRIEVER_TOP_K": [10, 15, 20],
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline():
    set_seed(config.SEED)

    dataset = Dataset(
        question_path=config.QUESTIONS_PATH,
        passage_path=config.PASSAGES_PATH,
        trigger_ratio=(config.TRIGGER_RATIO_MIN, config.TRIGGER_RATIO_MAX),
        num_test_triggers=config.NUM_TEST_TRIGGERS,
        filtered_word=config.FILTERED_WORD,
    )

    trigger = dataset.filtered_words[0]
    malicious_url = config.MALICIOUS_URL

    retriever = Retriever(
        dataset=dataset,
        save_dir=config.SAVE_RESULTS_DIR,
        model_path=config.RETRIEVER_MODEL_PATH,
        model_type=config.RETRIEVER_TYPE,
        trigger_phrase=trigger,
        k=config.RET_CORRELATION_THRESHOLD,
        malicious_template=config.RET_MALICIOUS_TRIGGER_DOC_TEMPLATE,
    )
    retrieval_results = retriever.optimize()

    generator = Generator(
        dataset=dataset,
        save_dir=config.SAVE_RESULTS_DIR,
        retrieval_results=retrieval_results,
        model_path=config.GENERATOR_MODEL_PATH,
        model_type=config.GENERATOR_TYPE,
        trigger_phrase=malicious_url,
        k=config.GEN_CORRELATION_THRESHOLD,
    )
    generation_results = generator.optimize()

    evaluator = Evaluator(
        generation_results,
        malicious_str=malicious_url[1:-2],
        trigger_phrase=config.FILTERED_WORD,
    )
    ret_success, gen_success, logs = evaluator.evaluate(dataset)
    print(f"Retrieval Success: {ret_success / len(logs):.4f}")
    print(f"Generation Success: {gen_success / len(logs):.4f}")


def run_experiment(exp_config, save_dir):
    for key, value in exp_config.items():
        setattr(config, key, value)
    os.makedirs(save_dir, exist_ok=True)
    with open(Path(save_dir) / "config.json", "w") as f:
        json.dump({k: str(v) for k, v in exp_config.items()}, f, indent=2)
    run_pipeline()


def process_var(keys, values):
    if ',' not in keys:
        return {keys: values}
    return dict(zip(keys.split(','), values))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Eyes-on-Me: Scalable RAG Poisoning via Transferable Attention Attractors'
    )
    parser.add_argument('--mode', choices=['main', 'ablations'], default='main',
                        help='Run mode: full experiment grid or hyperparameter sweep')
    parser.add_argument('--retrievers', nargs='+',
                        choices=['phantom', 'attention', 'llm', 'GCG', 'AutoDAN'],
                        default=['attention'],
                        help='Retriever type(s) to run')
    parser.add_argument('--generators', nargs='+',
                        choices=['MCG', 'attention', 'llm', 'GCG', 'AutoDAN'],
                        default=['attention'],
                        help='Generator type(s) to run')
    parser.add_argument('--retriever-models', nargs='+', dest='retriever_models',
                        default=[str(config.MODEL_DIR / 'bce-embedding-base_v1')],
                        help='Path(s) to retriever model checkpoints')
    parser.add_argument('--generator-models', nargs='+', dest='generator_models',
                        default=[str(config.MODEL_DIR / 'Qwen2.5-0.5B-Instruct')],
                        help='Path(s) to generator model checkpoints')
    parser.add_argument('--datasets', nargs='+',
                        choices=['marco', 'triviaqa', 'nq'],
                        default=['marco'],
                        help='Dataset(s) to use')
    parser.add_argument('--triggers', nargs='+',
                        default=['president'],
                        help='Trigger word(s) for the attack')
    parser.add_argument('--results-dir', dest='results_dir',
                        default=str(config.RESULTS_DIR),
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--ablation-vars', nargs='+', dest='ablation_vars',
                        choices=list(ABLATION_VARS.keys()),
                        help='Variable(s) to sweep (only for --mode ablations)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results_dir = Path(args.results_dir)
    config.SEED = args.seed

    if args.mode == 'main':
        combos = list(itertools.product(
            args.retriever_models,
            args.generator_models,
            args.retrievers,
            args.generators,
            args.datasets,
            args.triggers,
        ))
        for retr_model, gen_model, retr_type, gen_type, dataset_name, trigger in tqdm(combos):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = results_dir / retr_type / f"exp_{timestamp}"
            dataset_paths = DATASET_MAP[dataset_name]
            overrides = {
                'RETRIEVER_MODEL_PATH': retr_model,
                'GENERATOR_MODEL_PATH': gen_model,
                'RETRIEVER_TYPE': retr_type,
                'GENERATOR_TYPE': gen_type,
                'PASSAGES_PATH': str(dataset_paths['passages']),
                'QUESTIONS_PATH': str(dataset_paths['questions']),
                'FILTERED_WORD': trigger,
                'SAVE_RESULTS_DIR': str(save_dir),
                'SAVE_RESULTS_PATH': str(save_dir / 'experiment_results.json'),
                'MCG_C': 1,
            }
            run_experiment(overrides, save_dir)

    elif args.mode == 'ablations':
        vars_to_run = args.ablation_vars or list(ABLATION_VARS.keys())
        for key in tqdm(vars_to_run, desc='ablation vars'):
            for i, item in enumerate(ABLATION_VARS[key]):
                save_dir = results_dir / 'ablations' / key / str(i)
                overrides = process_var(key, item) | {
                    'RETRIEVER_TYPE': args.retrievers[0],
                    'GENERATOR_TYPE': args.generators[0],
                    'RETRIEVER_MODEL_PATH': args.retriever_models[0],
                    'GENERATOR_MODEL_PATH': args.generator_models[0],
                    'SAVE_RESULTS_DIR': str(save_dir),
                    'SAVE_RESULTS_PATH': str(save_dir / 'experiment_results.json'),
                }
                try:
                    run_experiment(overrides, save_dir)
                except Exception as e:
                    print(f"[ERROR] {key}={item}: {e}")
