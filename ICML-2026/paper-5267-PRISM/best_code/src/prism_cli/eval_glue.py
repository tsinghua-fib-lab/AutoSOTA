from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
try:
    import torch
except Exception:
    torch = None
from .trainers import RunConfig
from .utils import adapter_is_complete, build_prompt, cleanup_cuda
GLUE_TASKS = ['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte']

def glue_to_instruction_input(task: str, ex: dict):
    task = task.lower()
    if task == 'cola':
        return ('Task: CoLA (linguistic acceptability). Determine whether the sentence is grammatically acceptable. Answer with exactly one label: acceptable, unacceptable.', f"Sentence: {ex['sentence']}")
    if task == 'sst2':
        return ('Task: SST-2 (sentiment). Determine the sentiment of the sentence. Answer with exactly one label: positive, negative.', f"Sentence: {ex['sentence']}")
    if task == 'mrpc':
        return ('Task: MRPC (paraphrase). Determine whether the two sentences are semantically equivalent. Answer with exactly one label: equivalent, not_equivalent.', f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}")
    if task == 'stsb':
        return ('Task: STS-B (semantic textual similarity). Rate the semantic similarity of the two sentences on a scale from 0 to 5 (0 = completely different, 5 = equivalent in meaning). Answer with a single number (you may use one decimal place).', f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}")
    if task == 'qqp':
        return ('Task: QQP (duplicate questions). Determine whether the two questions are duplicates. Answer with exactly one label: duplicate, not_duplicate.', f"Question 1: {ex.get('question1') or ''}\nQuestion 2: {ex.get('question2') or ''}")
    if task == 'mnli':
        return ('Task: MNLI (natural language inference). Given a premise and a hypothesis, determine the relationship. Answer with exactly one label: entailment, neutral, contradiction.', f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}")
    if task == 'qnli':
        return ('Task: QNLI (question-answering NLI). Given a question and a sentence, determine whether the sentence entails the answer to the question. Answer with exactly one label: entailment, not_entailment.', f"Question: {ex['question']}\nSentence: {ex['sentence']}")
    if task == 'rte':
        return ('Task: RTE (recognizing textual entailment). Given a premise and a hypothesis, determine whether the premise entails the hypothesis. Answer with exactly one label: entailment, not_entailment.', f"Premise: {ex['sentence1']}\nHypothesis: {ex['sentence2']}")
    raise ValueError(task)

def _norm(s: str) -> str:
    s = (s or '').strip().lower().replace('\n', ' ')
    return re.sub('\\s+', ' ', s)

def parse_pred(task: str, text: str):
    task = task.lower()
    t = _norm(text)
    for prefix in ['answer:', 'label:', 'prediction:', 'pred:']:
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    if task == 'stsb':
        nums = re.findall('-?\\d+\\.?\\d*', t.replace(',', ''))
        if not nums:
            return (0.0, False)
        try:
            return (max(0.0, min(5.0, float(nums[0]))), True)
        except Exception:
            return (0.0, False)
    if task == 'mnli':
        if 'contradiction' in t:
            return (2, True)
        if 'neutral' in t:
            return (1, True)
        if 'entailment' in t:
            return (0, True)
        if t.startswith('yes'):
            return (0, False)
        if t.startswith('no'):
            return (2, False)
        return (0, False)
    if task == 'cola':
        if 'unacceptable' in t or 'not acceptable' in t:
            return (0, True)
        if 'acceptable' in t:
            return (1, True)
        if t.startswith('yes'):
            return (1, False)
        if t.startswith('no'):
            return (0, False)
        return (0, False)
    if task == 'sst2':
        if 'positive' in t:
            return (1, True)
        if 'negative' in t:
            return (0, True)
        return (0, False)
    if task == 'mrpc':
        if 'not_equivalent' in t or 'not equivalent' in t:
            return (0, True)
        if 'equivalent' in t:
            return (1, True)
        if t.startswith('yes'):
            return (1, False)
        if t.startswith('no'):
            return (0, False)
        return (0, False)
    if task == 'qqp':
        if 'not_duplicate' in t or 'not duplicate' in t:
            return (0, True)
        if 'duplicate' in t:
            return (1, True)
        if t.startswith('yes'):
            return (1, False)
        if t.startswith('no'):
            return (0, False)
        return (0, False)
    if task in ('qnli', 'rte'):
        if 'not_entailment' in t or 'not entailment' in t:
            return (1, True)
        if 'entailment' in t:
            return (0, True)
        if t.startswith('yes'):
            return (0, False)
        if t.startswith('no'):
            return (1, False)
        return (0, False)
    raise ValueError(task)

def _dtype():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def load_eval_model(base_model_id: str, adapter_dir: Path, num_beams: int):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    dtype = _dtype()
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype, device_map='auto', trust_remote_code=True)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(adapter_dir), torch_dtype=dtype, device_map='auto')
    gen_cfg = GenerationConfig(do_sample=False, num_beams=max(1, int(num_beams)), pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if hasattr(gen_cfg, 'remove_invalid_values'):
        gen_cfg.remove_invalid_values = True
    if hasattr(gen_cfg, 'renormalize_logits'):
        gen_cfg.renormalize_logits = True
    model.eval()
    return (model, tokenizer, gen_cfg)

def generate_labels(model, tokenizer, gen_cfg, prompts, max_input_length: int, max_new_tokens: int):
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=int(max_input_length))
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg, max_new_tokens=int(max_new_tokens), return_dict_in_generate=True, output_scores=False)
    decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
    preds = []
    for text in decoded:
        preds.append(text.split('### Response:', 1)[1].strip() if '### Response:' in text else text.strip())
    return preds

def _maybe_limit(ds, n):
    if n is None or int(n) <= 0:
        return ds
    return ds.select(range(min(int(n), len(ds))))

def eval_task(model, tokenizer, gen_cfg, task: str, batch_size: int, max_input_length: int, max_new_tokens: int, fast_dev_run: int=0, split_name: str='validation'):
    import evaluate as hf_evaluate
    from datasets import load_dataset
    from tqdm.auto import tqdm
    ds = load_dataset('glue', task)
    metric = hf_evaluate.load('glue', task)
    split = ds[split_name].filter(lambda x: x['label'] != -1)
    split = _maybe_limit(split, fast_dev_run)
    refs, prompts = ([], [])
    for ex in split:
        inst, inp = glue_to_instruction_input(task, ex)
        prompts.append(build_prompt(inst, inp))
        refs.append(ex['label'])
    pred_ids, parse_ok = ([], 0)
    for i in tqdm(range(0, len(prompts), int(batch_size)), desc=f'predict {task}:{split_name}'):
        outs = generate_labels(model, tokenizer, gen_cfg, prompts[i:i + int(batch_size)], max_input_length, max_new_tokens)
        for o in outs:
            pred, ok = parse_pred(task, o)
            pred_ids.append(pred)
            parse_ok += int(ok)
    if task == 'stsb':
        res = metric.compute(predictions=[float(p) for p in pred_ids], references=[float(r) for r in refs])
        score = (float(res['pearson']) + float(res['spearmanr'])) / 2.0
    else:
        res = metric.compute(predictions=[int(p) for p in pred_ids], references=[int(r) for r in refs])
        if task in ('mrpc', 'qqp'):
            score = (float(res['accuracy']) + float(res['f1'])) / 2.0
        elif task == 'cola':
            score = float(res.get('matthews_correlation', res.get('matthews', 0.0)))
        else:
            score = float(res['accuracy'])
    return {'task': task, 'score': float(score), 'parse_rate': float(parse_ok / max(1, len(pred_ids))), 'n': int(len(pred_ids)), 'metrics': {k: float(v) for k, v in res.items()}}

def eval_mnli(model, tokenizer, gen_cfg, batch_size: int, max_input_length: int, max_new_tokens: int, fast_dev_run: int=0):
    import evaluate as hf_evaluate
    from datasets import load_dataset
    from tqdm.auto import tqdm
    ds = load_dataset('glue', 'mnli')
    metric = hf_evaluate.load('glue', 'mnli')
    details = {}
    for split_name in ['validation_matched', 'validation_mismatched']:
        split = ds[split_name].filter(lambda x: x['label'] != -1)
        split = _maybe_limit(split, fast_dev_run)
        refs, prompts = ([], [])
        for ex in split:
            inst, inp = glue_to_instruction_input('mnli', ex)
            prompts.append(build_prompt(inst, inp))
            refs.append(ex['label'])
        pred_ids, parse_ok = ([], 0)
        for i in tqdm(range(0, len(prompts), int(batch_size)), desc=f'predict mnli:{split_name}'):
            outs = generate_labels(model, tokenizer, gen_cfg, prompts[i:i + int(batch_size)], max_input_length, max_new_tokens)
            for o in outs:
                pred, ok = parse_pred('mnli', o)
                pred_ids.append(pred)
                parse_ok += int(ok)
        res = metric.compute(predictions=[int(p) for p in pred_ids], references=[int(r) for r in refs])
        details[split_name] = {'metrics': {k: float(v) for k, v in res.items()}, 'parse_rate': float(parse_ok / max(1, len(pred_ids))), 'n': int(len(pred_ids))}
    score = (details['validation_matched']['metrics']['accuracy'] + details['validation_mismatched']['metrics']['accuracy']) / 2.0
    return {'task': 'mnli', 'score': float(score), 'matched_acc': float(details['validation_matched']['metrics']['accuracy']), 'mismatched_acc': float(details['validation_mismatched']['metrics']['accuracy']), 'details': details}

def evaluate_glue8(cfg: RunConfig, batch_size: int=128, num_beams: int=1, max_new_tokens: int=8, max_input_length: int=384, fast_dev_run: int=0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg.finalize()
    adapter_dir = Path(cfg.output_dir)
    if not adapter_is_complete(adapter_dir):
        raise RuntimeError(f'Adapter is not complete: {adapter_dir}')
    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    cached, tasks_to_run = ({}, [])
    for task in GLUE_TASKS:
        p = result_dir / f'{task}.json'
        if p.exists() and (not cfg.force_eval):
            cached[task] = json.load(open(p, 'r', encoding='utf-8'))
        else:
            tasks_to_run.append(task)
    model = tokenizer = gen_cfg = None
    if tasks_to_run:
        model, tokenizer, gen_cfg = load_eval_model(cfg.base_model, adapter_dir, num_beams)
    task_rows = []
    for task in GLUE_TASKS:
        p = result_dir / f'{task}.json'
        if task in cached:
            rec = cached[task]
            print(f'[eval cache] {task}: {p}')
        else:
            rec = eval_mnli(model, tokenizer, gen_cfg, batch_size, max_input_length, max_new_tokens, fast_dev_run) if task == 'mnli' else eval_task(model, tokenizer, gen_cfg, task, batch_size, max_input_length, max_new_tokens, fast_dev_run)
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)
        task_rows.append(rec)
    if model is not None:
        del model, tokenizer, gen_cfg
        cleanup_cuda()
    score_row = {'method': cfg.method, 'privacy': cfg.privacy, 'base_model': cfg.base_model, 'lora_r': int(cfg.lora_r)}
    for rec in task_rows:
        score_row[rec['task']] = float(rec['score'])
    score_row['GLUE8_Avg'] = sum((score_row[t] for t in GLUE_TASKS)) / len(GLUE_TASKS)
    scores_df = pd.DataFrame([score_row])
    task_df = pd.DataFrame(task_rows)
    scores_df.to_csv(result_dir / 'summary.csv', index=False)
    task_df.to_csv(result_dir / 'details.csv', index=False)
    print(scores_df[['method', 'privacy', 'cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'GLUE8_Avg']])
    print('Saved:', result_dir / 'summary.csv')
    return (scores_df, task_df)
