from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))
from prism_cli.eval_glue import evaluate_glue8
from prism_cli.eval_math import evaluate_math10k
from prism_cli.trainers import RunConfig, train

def parse_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).lower()
    if s in {'1', 'true', 'yes', 'y'}:
        return True
    if s in {'0', 'false', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Expected boolean, got {x!r}')

def main():
    p = argparse.ArgumentParser(description='Train and evaluate PRISM LoRA adapters on Math-10K or GLUE8.')
    p.add_argument('--dataset', choices=['math10k', 'math', 'glue8', 'glue'], required=True)
    p.add_argument('--privacy', choices=['dp', 'nondp', 'non-dp'], default='dp')
    p.add_argument('--epsilon', type=float, default=6.0)
    p.add_argument('--delta', type=float, default=1e-05)
    p.add_argument('--base_model', default='google/gemma-3-4b-pt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_path', type=Path, default=None)
    p.add_argument('--output_dir', type=Path, default=None)
    p.add_argument('--result_dir', type=Path, default=None)
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=16)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--target_modules', default='q_proj,k_proj,v_proj,up_proj,down_proj')
    p.add_argument('--steps', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--micro_batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--cutoff_len', type=int, default=None)
    p.add_argument('--train_on_inputs', type=parse_bool, default=None)
    p.add_argument('--dp_max_grad_norm', type=float, default=1.0)
    p.add_argument('--dp_grad_sample_mode', default='functorch')
    p.add_argument('--run_train', type=parse_bool, default=True)
    p.add_argument('--run_eval', type=parse_bool, default=True)
    p.add_argument('--force_train', action='store_true')
    p.add_argument('--force_eval', action='store_true')
    p.add_argument('--no_resume', action='store_true')
    p.add_argument('--checkpoint_every', type=int, default=25)
    p.add_argument('--repeat_id', type=int, default=None)
    p.add_argument('--eval_batch_size', type=int, default=None)
    p.add_argument('--num_beams', type=int, default=None)
    p.add_argument('--max_new_tokens', type=int, default=None)
    p.add_argument('--max_input_length', type=int, default=None)
    p.add_argument('--fast_dev_run', type=int, default=0)
    p.add_argument('--prism_floor_factor', type=float, default=None)
    p.add_argument('--prism_floor_mode', default=None)
    p.add_argument('--prism_lift_fix', default=None)
    p.add_argument('--prism_debias_second_moment', type=parse_bool, default=None)
    p.add_argument('--noise_decay_enabled', type=parse_bool, default=None)
    p.add_argument('--noise_decay_start_step', type=int, default=None)
    p.add_argument('--noise_decay_factor', type=float, default=None)

    args = p.parse_args()

    cfg = RunConfig(dataset=args.dataset, method='prism', privacy=args.privacy, root=ROOT, base_model=args.base_model, seed=args.seed, data_path=args.data_path, output_dir=args.output_dir, result_dir=args.result_dir, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=[x.strip() for x in args.target_modules.split(',') if x.strip()], total_update_steps=args.steps, batch_size=args.batch_size, micro_batch_size=args.micro_batch_size, learning_rate=args.lr, cutoff_len=args.cutoff_len, train_on_inputs=args.train_on_inputs, dp_epsilon=args.epsilon, dp_delta=args.delta, dp_max_grad_norm=args.dp_max_grad_norm, dp_grad_sample_mode=args.dp_grad_sample_mode, force_train=args.force_train, force_eval=args.force_eval, resume=not args.no_resume, checkpoint_every=args.checkpoint_every, run_train=args.run_train, run_eval=args.run_eval)
    if args.prism_floor_factor is not None:
        cfg.prism_floor_factor = args.prism_floor_factor
    if args.prism_floor_mode is not None:
        cfg.prism_floor_mode = args.prism_floor_mode
    if args.prism_lift_fix is not None:
        cfg.prism_lift_fix = args.prism_lift_fix
    if args.prism_debias_second_moment is not None:
        cfg.prism_debias_second_moment = args.prism_debias_second_moment
    if args.noise_decay_enabled is not None:
        cfg.noise_decay_enabled = args.noise_decay_enabled
    if args.noise_decay_start_step is not None:
        cfg.noise_decay_start_step = args.noise_decay_start_step
    if args.noise_decay_factor is not None:
        cfg.noise_decay_factor = args.noise_decay_factor
    cfg = cfg.finalize()
    if args.repeat_id is not None and args.output_dir is None:
        cfg.output_dir = Path(str(cfg.output_dir) + f'_repeat{args.repeat_id}')
    if args.repeat_id is not None and args.result_dir is None:
        cfg.result_dir = Path(str(cfg.result_dir) + f'_repeat{args.repeat_id}')
    if cfg.run_train:
        train(cfg)
    else:
        print('[skip] run_train=False')
    if cfg.run_eval:
        if cfg.dataset == 'math10k':
            evaluate_math10k(cfg, batch_size=args.eval_batch_size or 64, num_beams=args.num_beams or 4, max_new_tokens=args.max_new_tokens or 256, max_input_length=args.max_input_length or 1024)
        else:
            evaluate_glue8(cfg, batch_size=args.eval_batch_size or 128, num_beams=args.num_beams or 1, max_new_tokens=args.max_new_tokens or 8, max_input_length=args.max_input_length or 384, fast_dev_run=args.fast_dev_run)
    else:
        print('[skip] run_eval=False')
if __name__ == '__main__':
    main()
