from __future__ import annotations
import gc
import json
import os
import random
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import torch

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def llm_adapters_dir(root: Optional[Path]=None) -> Path:
    root = project_root() if root is None else Path(root)
    return root / 'LLM-Adapters'

def tag_text(x: str) -> str:
    return str(x).replace('/', '_').replace(':', '_').replace(' ', '_')

def set_seed(seed: int=42) -> None:
    random.seed(seed)
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def build_prompt(instruction: str, input_text: str, output: str='') -> str:
    data_point = {'instruction': instruction, 'input': input_text, 'output': output}
    return generate_prompt(data_point)

def generate_prompt(data_point: Dict[str, Any]) -> str:
    if data_point.get('input'):
        return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n                ### Instruction:\n                {data_point['instruction']}\n                \n                ### Input:\n                {data_point['input']}\n                \n                ### Response:\n                {data_point['output']}"
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.  \n\n                ### Instruction:\n                {data_point['instruction']}\n                \n                ### Response:\n                {data_point['output']}"

def build_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token_id = 0
    return tok

def tokenize_prompt(tokenizer, example: Dict[str, Any], cutoff_len: int, train_on_inputs: bool, base_model: str):
    full_prompt = generate_prompt(example)
    result = tokenizer(full_prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
    if result['input_ids'][-1] != tokenizer.eos_token_id and len(result['input_ids']) < cutoff_len:
        result['input_ids'].append(tokenizer.eos_token_id)
        if 'chatglm' not in base_model:
            result['attention_mask'].append(1)
    result['labels'] = result['input_ids'].copy()
    if not train_on_inputs:
        user_prompt = generate_prompt({**example, 'output': ''})
        user_tok = tokenizer(user_prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        user_prompt_len = len(user_tok['input_ids'])
        result['labels'] = [-100] * user_prompt_len + result['labels'][user_prompt_len:]
    return result

def freeze_vision_tower_params(model) -> None:
    frozen = 0
    for name, p in model.named_parameters():
        if 'vision_tower' in name or '.vision_model.' in name:
            if p.requires_grad:
                p.requires_grad = False
                frozen += 1
    if frozen:
        print(f'[train] Froze {frozen} trainable vision-tower parameters for text-only training.')

def iter_microbatches(batch_dict: Dict[str, torch.Tensor], micro_bs: int):
    bs = batch_dict['input_ids'].shape[0]
    for start in range(0, bs, micro_bs):
        end = min(bs, start + micro_bs)
        yield {k: v[start:end] for k, v in batch_dict.items()}

def unwrap_for_save(model):
    return getattr(model, '_module', model)

def adapter_is_complete(path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    return (path / 'adapter_config.json').exists() and any(((path / name).exists() for name in ['adapter_model.safetensors', 'adapter_model.bin']))

def status_path(output_dir: Path) -> Path:
    return Path(output_dir) / 'run_status.json'

def save_status(output_dir: Path, **payload: Any) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(status_path(out), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)

def checkpoint_file(output_dir: Path) -> Path:
    return Path(output_dir) / '_resume_checkpoint.pt'

def get_rng_state() -> Dict[str, Any]:
    state = {'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state

def set_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])

def trainable_state_dict_cpu(model) -> Dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}

def load_trainable_state_dict(model, state: Dict[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            if n in named:
                named[n].copy_(v.to(device=named[n].device, dtype=named[n].dtype))

def save_resume_checkpoint(output_dir: Path, model, optimizer, update_steps: int, extra: Optional[Dict[str, Any]]=None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {'update_steps': int(update_steps), 'trainable_state': trainable_state_dict_cpu(model), 'optimizer_state': optimizer.state_dict() if optimizer is not None else None, 'rng_state': get_rng_state(), 'extra': extra or {}}
    torch.save(payload, checkpoint_file(output_dir))

def load_resume_checkpoint_if_available(output_dir: Path) -> Optional[Dict[str, Any]]:
    path = checkpoint_file(output_dir)
    if not path.exists():
        return None
    print(f'[resume] loading {path}')
    return torch.load(path, map_location='cpu')

class JsonlLogger:

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, 'a', encoding='utf-8')

    def log(self, rec: Dict[str, Any]) -> None:
        self.f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        self.f.flush()

    def close(self) -> None:
        self.f.close()

def clean_output_dir(path: Path, force: bool) -> None:
    path = Path(path)
    if force and path.exists():
        print(f'[train] removing {path}')
        shutil.rmtree(path)
