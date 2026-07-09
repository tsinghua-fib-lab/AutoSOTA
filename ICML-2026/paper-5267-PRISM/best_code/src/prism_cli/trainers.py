from __future__ import annotations
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from tqdm.auto import tqdm
from .utils import JsonlLogger, adapter_is_complete, build_tokenizer, checkpoint_file, clean_output_dir, cleanup_cuda, freeze_vision_tower_params, generate_prompt, iter_microbatches, llm_adapters_dir, load_resume_checkpoint_if_available, load_trainable_state_dict, save_resume_checkpoint, save_status, set_rng_state, set_seed, tag_text, tokenize_prompt, unwrap_for_save

@dataclass
class RunConfig:
    dataset: str
    method: str
    privacy: str
    root: Path
    base_model: str = 'google/gemma-3-4b-pt'
    seed: int = 42
    data_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    result_dir: Optional[Path] = None
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj'])
    total_update_steps: Optional[int] = None
    batch_size: int = 64
    micro_batch_size: int = 4
    learning_rate: Optional[float] = None
    cutoff_len: Optional[int] = None
    train_on_inputs: Optional[bool] = None
    val_set_size: int = 0
    eval_step: int = 100
    save_step: int = 200
    dp_epsilon: float = 6.0
    dp_delta: float = 1e-05
    dp_max_grad_norm: float = 1.0
    dp_grad_sample_mode: str = 'functorch'
    force_train: bool = False
    force_eval: bool = False
    resume: bool = True
    checkpoint_every: int = 25
    run_train: bool = True
    run_eval: bool = True
    spectral_svd_device: str = 'cpu'
    spectral_oversample: int = 8
    spectral_n_iter: int = 2
    prism_floor_factor: float = 0.5
    prism_floor_mode: str = 'scalar'
    prism_cond_max: float = 10000.0
    prism_cond_strategy: str = 'raise_small'
    prism_lift_fix: str = 'both'
    prism_debias_second_moment: bool = True  # IDEA-09
    max_update_norm: float = 0.0
    noise_decay_enabled: bool = False
    noise_decay_start_step: int = 200
    noise_decay_factor: float = 0.8

    def finalize(self) -> 'RunConfig':
        self.dataset = self.dataset.lower().replace('-', '')
        if self.dataset in {'math', 'math10k'}:
            self.dataset = 'math10k'
        elif self.dataset in {'glue', 'glue8'}:
            self.dataset = 'glue8'
        else:
            raise ValueError('dataset must be one of: math10k, glue8')
        self.method = 'prism'
        self.privacy = self.privacy.lower().replace('_', '-')
        if self.privacy in {'non-dp', 'nondp', 'none'}:
            self.privacy = 'nondp'
        if self.privacy not in {'dp', 'nondp'}:
            raise ValueError('privacy must be dp or nondp')
        adapters = llm_adapters_dir(self.root)
        if self.dataset == 'math10k':
            self.total_update_steps = self.total_update_steps or 300
            self.learning_rate = self.learning_rate or 0.0003
            self.cutoff_len = self.cutoff_len or 256
            self.train_on_inputs = True if self.train_on_inputs is None else self.train_on_inputs
            self.val_set_size = int(self.val_set_size or 120)
            self.eval_step = int(self.eval_step or 10)
            self.save_step = int(self.save_step or 20)
            self.data_path = self.data_path or adapters / 'ft-training_set' / 'math_10k.json'
        else:
            self.total_update_steps = self.total_update_steps or 500
            self.learning_rate = self.learning_rate or 0.0002
            self.cutoff_len = self.cutoff_len or 384
            self.train_on_inputs = False if self.train_on_inputs is None else self.train_on_inputs
            self.val_set_size = int(self.val_set_size or 1000)
            self.eval_step = int(self.eval_step or 100)
            self.save_step = int(self.save_step or 200)
            self.data_path = self.data_path or adapters / 'ft-training_set' / 'glue8_1250.json'
        base_tag = tag_text(self.base_model)
        if self.privacy == 'dp':
            run_tag = f'{self.dataset}_{self.method}_dp_eps{self.dp_epsilon}_seed{self.seed}_r{self.lora_r}_{base_tag}'
        else:
            run_tag = f'{self.dataset}_{self.method}_nondp_seed{self.seed}_r{self.lora_r}_{base_tag}'
        self.output_dir = self.output_dir or adapters / 'trained_models' / run_tag
        self.result_dir = self.result_dir or adapters / 'experiment' / run_tag
        return self

    @property
    def log_jsonl(self) -> Path:
        return Path(self.output_dir) / 'train_log.jsonl'

def _preimport_prism_training_stack() -> None:
    import datasets
    import opacus
    import peft
    import transformers
    from torch.utils.data import DataLoader
    from .optim import prism as _prism_module

def _ensure_paths(cfg: RunConfig) -> None:
    if not cfg.data_path or not Path(cfg.data_path).exists():
        raise FileNotFoundError(f'Training data not found: {cfg.data_path}')
    Path(cfg.output_dir).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.result_dir).mkdir(parents=True, exist_ok=True)

def _device_and_dtype() -> Tuple[torch.device, torch.dtype, dict]:
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return (torch.device(f'cuda:{local_rank}'), dtype, {'': local_rank})
    return (torch.device('cpu'), torch.float32, {'': 'cpu'})

def _make_private(privacy_engine, model, optimizer, train_loader, cfg: RunConfig, epochs_est: int):
    try:
        out = privacy_engine.make_private_with_epsilon(module=model, optimizer=optimizer, data_loader=train_loader, target_epsilon=cfg.dp_epsilon, target_delta=cfg.dp_delta, epochs=epochs_est, max_grad_norm=cfg.dp_max_grad_norm, grad_sample_mode=cfg.dp_grad_sample_mode)
        return (out, cfg.dp_grad_sample_mode)
    except TypeError:
        out = privacy_engine.make_private_with_epsilon(module=model, optimizer=optimizer, data_loader=train_loader, target_epsilon=cfg.dp_epsilon, target_delta=cfg.dp_delta, epochs=epochs_est, max_grad_norm=cfg.dp_max_grad_norm)
        return (out, 'default')
    except Exception as e:
        print(f'[warn] make_private_with_epsilon failed for grad_sample_mode={cfg.dp_grad_sample_mode}: {e!r}')
        out = privacy_engine.make_private_with_epsilon(module=model, optimizer=optimizer, data_loader=train_loader, target_epsilon=cfg.dp_epsilon, target_delta=cfg.dp_delta, epochs=epochs_est, max_grad_norm=cfg.dp_max_grad_norm, grad_sample_mode='hooks')
        return (out, 'hooks')

def _step_accountant(privacy_engine, noise_multiplier: float, sample_rate: float) -> None:
    try:
        privacy_engine.accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    except TypeError:
        privacy_engine.accountant.step(noise_multiplier, sample_rate)

def _make_loader(cfg: RunConfig, tokenizer):
    import transformers
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    data = load_dataset('json', data_files=str(cfg.data_path)) if str(cfg.data_path).endswith('.json') else load_dataset(str(cfg.data_path))
    train_ds = data['train']

    def map_fn(ex):
        return tokenize_prompt(tokenizer, ex, cutoff_len=int(cfg.cutoff_len), train_on_inputs=bool(cfg.train_on_inputs), base_model=cfg.base_model)
    train_ds = train_ds.map(map_fn, remove_columns=train_ds.column_names, desc='Tokenizing')
    collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True)
    loader_kwargs = dict(dataset=train_ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False, collate_fn=collator)
    # IDEA-08: Always seed DataLoader for reproducibility
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader_kwargs['generator'] = generator
    loader_kwargs['num_workers'] = 0
    loader = DataLoader(**loader_kwargs)
    return (loader, train_ds)

def _build_lora_model(cfg: RunConfig):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    device, dtype, device_map = _device_and_dtype()
    tokenizer = build_tokenizer(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
    model.config.use_cache = False
    lora_cfg = LoraConfig(r=int(cfg.lora_r), lora_alpha=int(cfg.lora_alpha), target_modules=list(cfg.target_modules), lora_dropout=float(cfg.lora_dropout), bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_cfg)
    freeze_vision_tower_params(model)
    return (model, tokenizer, device)

def _initialize_prism_factors(cfg: RunConfig, model) -> None:
    from .optim.prism import spectral_init_peft_model
    print('[init] spectral residual initialization')
    spectral_init_peft_model(model, svd_device=cfg.spectral_svd_device, svd_oversample=int(cfg.spectral_oversample), svd_n_iter=int(cfg.spectral_n_iter), verbose=True)

def _build_prism_optimizer(cfg: RunConfig, model):
    from .optim.prism import PRISM, get_paired_lora_parameters
    params = get_paired_lora_parameters(model)
    print(f'[optimizer] paired LoRA tensors = {len(params)} / modules = {len(params) // 2}')
    return PRISM(params, lr=float(cfg.learning_rate), betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, use_adaptive=True, dp_precond_floor_factor=float(cfg.prism_floor_factor), dp_floor_mode=cfg.prism_floor_mode, precond_cond_max=float(cfg.prism_cond_max), precond_cond_strategy=cfg.prism_cond_strategy, precond_update_mode='current', lift_gauge_fix=cfg.prism_lift_fix, gauge_fix_eps=1e-12, max_update_norm=float(cfg.max_update_norm), dp_debias_second_moment=bool(cfg.prism_debias_second_moment))

def _rebase_prism_for_save(model) -> None:
    from .optim.prism import spectral_rebase_adapter_inplace
    try:
        spectral_rebase_adapter_inplace(model, adapter_name='default', verbose=True)
    except Exception as e:
        print('[warn] spectral rebase for saving failed:', repr(e))

def _restore_if_possible(cfg: RunConfig, model, optimizer) -> int:
    if not cfg.resume or cfg.force_train:
        return 0
    ckpt = load_resume_checkpoint_if_available(Path(cfg.output_dir))
    if ckpt is None:
        return 0
    load_trainable_state_dict(model, ckpt.get('trainable_state', {}))
    if optimizer is not None and ckpt.get('optimizer_state') is not None:
        try:
            if hasattr(optimizer, '_ensure_state'):
                optimizer._ensure_state()
            optimizer.load_state_dict(ckpt['optimizer_state'])
        except Exception as e:
            print('[resume] optimizer state not restored:', repr(e))
    set_rng_state(ckpt.get('rng_state'))
    step = int(ckpt.get('update_steps', 0))
    print(f'[resume] starting at update step {step}')
    return step

def train_prism_manual(cfg: RunConfig) -> Path:
    clean_output_dir(Path(cfg.output_dir), cfg.force_train)
    if adapter_is_complete(Path(cfg.output_dir)) and (not cfg.force_train):
        print('[train] completed adapter found; skipping:', cfg.output_dir)
        return Path(cfg.output_dir)
    if cfg.dataset == 'math10k':
        _preimport_prism_training_stack()
        set_seed(cfg.seed)
        model, tokenizer, device = _build_lora_model(cfg)
        _initialize_prism_factors(cfg, model)
        train_loader, train_ds = _make_loader(cfg, tokenizer)
        optimizer = _build_prism_optimizer(cfg, model)
    else:
        set_seed(cfg.seed)
        model, tokenizer, device = _build_lora_model(cfg)
        _initialize_prism_factors(cfg, model)
        optimizer = _build_prism_optimizer(cfg, model)
        train_loader, train_ds = _make_loader(cfg, tokenizer)
    start_step = _restore_if_possible(cfg, model, optimizer)
    privacy_engine = None
    noise_multiplier = None
    sample_rate = None
    if cfg.privacy == 'dp':
        from opacus import PrivacyEngine
        epochs_est = max(1, math.ceil(int(cfg.total_update_steps) / max(1, len(train_loader))))
        privacy_engine = PrivacyEngine()
        (model, dp_opt, train_loader), used_mode = _make_private(privacy_engine, model, optimizer, train_loader, cfg, epochs_est)
        noise_multiplier = float(getattr(dp_opt, 'noise_multiplier', None))
        sample_rate = getattr(dp_opt, 'sample_rate', None) or float(cfg.batch_size) / float(len(train_ds))
        base_opt = getattr(dp_opt, 'original_optimizer', optimizer)
        optimizer = base_opt
        for _ in range(start_step):
            _step_accountant(privacy_engine, noise_multiplier, float(sample_rate))
        print(f'[DP] grad_sample_mode={used_mode} noise_multiplier={noise_multiplier} sample_rate≈{float(sample_rate):.6g}')
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(cfg.log_jsonl)
    model.train()
    update_steps = int(start_step)
    pbar = tqdm(total=int(cfg.total_update_steps), initial=update_steps, desc=f'{cfg.method}/{cfg.privacy} updates')
    while update_steps < int(cfg.total_update_steps):
        for batch in train_loader:
            if update_steps >= int(cfg.total_update_steps):
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            loss_sum = 0.0
            token_sum = 0
            seen = 0
            if cfg.privacy == 'dp':
                # IDEA-02: Noise multiplier decay for late-stage fine convergence
                current_noise = float(noise_multiplier)
                if cfg.noise_decay_enabled and update_steps >= cfg.noise_decay_start_step:
                    current_noise = current_noise * cfg.noise_decay_factor
                optimizer.dp_begin(max_grad_norm=float(cfg.dp_max_grad_norm))
                for micro in iter_microbatches(batch, int(cfg.micro_batch_size)):
                    optimizer.zero_grad(set_to_none=True)
                    out = model(**micro)
                    loss = out.loss if hasattr(out, 'loss') else out[0]
                    micro_bs = int(micro['input_ids'].shape[0])
                    loss_sum += float(loss.detach().cpu().item()) * micro_bs
                    seen += micro_bs
                    token_sum += int(micro.get('attention_mask', torch.ones_like(micro['input_ids'])).detach().sum().item())
                    (loss * micro_bs).backward()
                    optimizer.dp_accumulate()
                total_seen = optimizer.dp_finalize(noise_multiplier=float(current_noise))
                _step_accountant(privacy_engine, noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
                seen = max(seen, int(total_seen))
            else:
                optimizer.zero_grad(set_to_none=True)
                micros = list(iter_microbatches(batch, int(cfg.micro_batch_size)))
                grad_accum = max(1, len(micros))
                for micro in micros:
                    out = model(**micro)
                    loss = out.loss if hasattr(out, 'loss') else out[0]
                    micro_bs = int(micro['input_ids'].shape[0])
                    loss_sum += float(loss.detach().cpu().item()) * micro_bs
                    seen += micro_bs
                    token_sum += int(micro.get('attention_mask', torch.ones_like(micro['input_ids'])).detach().sum().item())
                    (loss / float(grad_accum)).backward()
                optimizer.step()
            update_steps += 1
            pbar.update(1)
            rec = {'method': cfg.method, 'privacy': cfg.privacy, 'step': update_steps, 'loss_mean': loss_sum / max(1, seen), 'tokens': token_sum, 'batch_n': seen, 'base_model': cfg.base_model, 'dataset': cfg.dataset, 'lora_r': int(cfg.lora_r), 'lr': float(cfg.learning_rate), 'spectral_oversample': int(cfg.spectral_oversample), 'spectral_n_iter': int(cfg.spectral_n_iter), 'lift_gauge_fix': cfg.prism_lift_fix, 'prism_floor_factor': float(cfg.prism_floor_factor), 'prism_floor_mode': cfg.prism_floor_mode, 'prism_debias_second_moment': bool(cfg.prism_debias_second_moment)}
            rec.update(getattr(optimizer, 'last_log', {}) or {})
            if privacy_engine is not None:
                try:
                    rec['eps_spent'] = float(privacy_engine.get_epsilon(float(cfg.dp_delta)))
                except Exception:
                    pass
            logger.log(rec)
            if update_steps % 10 == 0 or update_steps == int(cfg.total_update_steps):
                msg = f"[{cfg.method}/{cfg.privacy}] step={update_steps} loss={rec['loss_mean']:.4f}"
                if 'eps_spent' in rec:
                    msg += f" eps≈{rec['eps_spent']:.3f}"
                print(msg)
            if cfg.resume and update_steps % int(cfg.checkpoint_every) == 0 and (update_steps < int(cfg.total_update_steps)):
                save_resume_checkpoint(Path(cfg.output_dir), model, optimizer, update_steps)
    pbar.close()
    logger.close()
    to_save = unwrap_for_save(model)
    _rebase_prism_for_save(to_save)
    to_save.save_pretrained(str(cfg.output_dir))
    if checkpoint_file(Path(cfg.output_dir)).exists():
        checkpoint_file(Path(cfg.output_dir)).unlink()
    save_status(Path(cfg.output_dir), state='completed', method=cfg.method, privacy=cfg.privacy, update_steps=update_steps, dataset=cfg.dataset, base_model=cfg.base_model, lora_r=int(cfg.lora_r))
    cleanup_cuda()
    print('Saved adapter to:', cfg.output_dir)
    return Path(cfg.output_dir)

def train(cfg: RunConfig) -> Path:
    cfg.finalize()
    _ensure_paths(cfg)
    print('Run config:')
    for k, v in sorted(cfg.__dict__.items()):
        print(f'  {k}: {v}')
    return train_prism_manual(cfg)
