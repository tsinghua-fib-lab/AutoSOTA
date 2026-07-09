from __future__ import annotations
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from .trainers import RunConfig
from .utils import adapter_is_complete, llm_adapters_dir
MATH_TASKS = ['gsm8k', 'AQuA', 'mawps', 'SVAMP']

def _task_dir_name(task: str) -> str:
    low = task.lower()
    if low == 'gsm8k':
        return 'gsm8k'
    if low == 'aqua':
        return 'AQuA'
    if low == 'svamp':
        return 'SVAMP'
    if low == 'mawps':
        return 'mawps'
    return task

def _safe_load(path: Path):
    try:
        data = json.load(open(path, 'r', encoding='utf-8'))
        if isinstance(data, list) and data:
            acc = sum((1 for x in data if x.get('flag'))) / max(1, len(data))
            return (acc, data)
    except Exception:
        pass
    return (None, None)

def evaluate_math10k(cfg: RunConfig, batch_size: int=64, num_beams: int=4, max_new_tokens: int=256, max_input_length: int=1024) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg.finalize()
    adapter_dir = Path(cfg.output_dir)
    if not adapter_is_complete(adapter_dir):
        raise RuntimeError(f'Adapter is not complete: {adapter_dir}')
    adapters = llm_adapters_dir(cfg.root)
    eval_py = adapters / 'evaluate.py'
    spec = importlib.util.spec_from_file_location('llm_adapters_math_evaluate', eval_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load {eval_py}')
    llm_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llm_eval)
    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    shared_exp_dir = adapters / 'experiment'
    shared_exp_dir.mkdir(parents=True, exist_ok=True)
    try:
        lora_weights = str(adapter_dir.relative_to(adapters))
    except ValueError:
        lora_weights = str(adapter_dir)
    results: Dict[str, float] = {}
    rows = []
    for task in MATH_TASKS:
        ds_name = _task_dir_name(task)
        unique_json = result_dir / f'{ds_name}.json'
        acc, data = _safe_load(unique_json)
        if acc is not None and (not cfg.force_eval):
            print(f'[eval cache] {task}: {unique_json}')
            results[ds_name] = float(acc)
            rows.append({'dataset': ds_name, 'accuracy': float(acc), 'n': len(data), 'json': str(unique_json)})
            continue
        shared_json = shared_exp_dir / f'other-LoRA-{ds_name}.json'
        if shared_json.exists():
            shared_json.unlink()
        sys.argv = ['evaluate.py', '--dataset', task, '--model', 'other', '--adapter', 'LoRA', '--base_model', cfg.base_model, '--lora_weights', lora_weights, '--batch_size', str(int(batch_size)), '--num_beams', str(int(num_beams)), '--max_new_tokens', str(int(max_new_tokens)), '--max_input_length', str(int(max_input_length)), '--log_every', '0']
        print('[eval]', ' '.join(sys.argv))
        llm_eval.main()
        if not shared_json.exists():
            raise FileNotFoundError(f'Expected eval output not found: {shared_json}')
        shutil.copy2(shared_json, unique_json)
        acc, data = _safe_load(unique_json)
        if acc is None:
            raise RuntimeError(f'Could not parse eval output: {unique_json}')
        results[ds_name] = float(acc)
        rows.append({'dataset': ds_name, 'accuracy': float(acc), 'n': len(data), 'json': str(unique_json)})
    df = pd.DataFrame([results])
    df['Average'] = df.mean(axis=1)
    detail_df = pd.DataFrame(rows)
    df.to_csv(result_dir / 'summary.csv', index=False)
    detail_df.to_csv(result_dir / 'details.csv', index=False)
    print(df)
    print('Saved:', result_dir / 'summary.csv')
    return (df, detail_df)
