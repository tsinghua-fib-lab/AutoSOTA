"""Run the original third-place trainer with packed data and an audited spectrum.

The default is a scratch pilot of V2; it does not reproduce the 23-model final
ensemble. Supplied synthetic data, per-fold pseudo labels and checkpoints enable
the corresponding original stages without guessing missing artifacts.
"""
import argparse
import functools
import json
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal.windows import tukey
from upstream_models import ROOT, TEAM3


def estimate_spectrum(waves, indices, output, device):
    if len(indices) == 0:
        raise ValueError("No training negatives for whitening")
    window = torch.tensor(tukey(8192, 0.5), dtype=torch.float32, device=device)
    total = torch.zeros(3, 8192, dtype=torch.float64, device=device)
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            batch = torch.from_numpy(np.array(waves[indices[start:start + 512]], copy=True)).to(device)
            expanded = torch.cat([-batch.flip(-1)[:, :, 2047:-1] + 2 * batch[:, :, :1], batch,
                                  -batch.flip(-1)[:, :, 1:2049] + 2 * batch[:, :, -1:]], dim=-1)
            total += torch.fft.fft(expanded * window).abs().double().sum(0)
    spectrum = (total / len(indices)).float().cpu()
    if not torch.isfinite(spectrum).all() or not (spectrum > 0).all():
        raise ValueError("Invalid average spectrum")
    torch.save(spectrum, output)
    return spectrum


class DatasetNumpyProxy:
    def __init__(self, arrays):
        self.arrays = arrays

    def __getattr__(self, name):
        return getattr(np, name)

    def load(self, path, *args, **kwargs):
        if isinstance(path, str) and path.startswith("packed:"):
            _, split, index = path.split(":")
            return self.arrays[split][int(index)].astype(np.float64) / 1e20
        return np.load(path, *args, **kwargs)


class DatasetPandasProxy:
    def __getattr__(self, name):
        return getattr(pd, name)

    def read_csv(self, path, **kwargs):
        kwargs.setdefault('dtype', {'id': str})
        return pd.read_csv(path, **kwargs)


class CheckpointTorchProxy:
    """Deserialize upstream checkpoints on CPU before copying into the model.

    Upstream checkpoints record their original CUDA device. Loading one without
    map_location opens that GPU as well as the explicitly assigned training GPU,
    which can block a queued worker waiting for the original GPU to become idle.
    Do not patch the global torch module or modify the contestant's source.
    """
    def __getattr__(self, name):
        return getattr(torch, name)

    def __init__(self, folder=None, selection='best'):
        self.folder, self.selection = folder, selection

    def load(self, path, *args, **kwargs):
        if self.selection == 'last' and self.folder is not None and Path(path).parent.resolve() == self.folder.resolve():
            path = str(path).replace('_best_model.pth', '_last_model.pth')
        if not args:
            kwargs.setdefault('map_location', 'cpu')
        return torch.load(path, *args, **kwargs)

    def save(self, value, path, *args, **kwargs):
        if isinstance(path, (str, Path)) and str(path).endswith('.pth'):
            destination = Path(path)
            temporary = destination.with_suffix('.partial.pth')
            torch.save(value, temporary, *args, **kwargs)
            temporary.replace(destination)
            return
        return torch.save(value, path, *args, **kwargs)


class PackedSyntheticBank:
    def __init__(self, directory):
        self.metadata = json.loads((directory / 'metadata.json').read_text())
        if self.metadata['amplitude_scale'] != 1e20:
            raise ValueError('Synthetic bank scale must match the original dataset')
        self.waves = np.load(directory / 'signals.npy', mmap_mode='r')

    def keys(self):
        return range(len(self.waves))

    def __getitem__(self, index):
        return np.array(self.waves[index], copy=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="V2", choices=["V2", "V-V2SD", "M-1D", "M-1DC16", "M-1DS32", "M-1DS32_pretrain", "M-1DS32_adjust", "M-1D_pretrain", "M-1D_adjust"])
    parser.add_argument("--packed", type=Path, default=ROOT / "data/packed")
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--pseudo-label-dir", type=Path)
    parser.add_argument("--synthetic", type=Path)
    parser.add_argument("--synthetic-bank", type=Path)
    parser.add_argument('--noise-bank', type=Path)
    parser.add_argument("--experiment-name")
    parser.add_argument("--fold-seed", type=int)
    parser.add_argument('--training-seed', type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--amp", action='store_true')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--keep-last', action='store_true')
    parser.add_argument('--save-epoch-checkpoints', action='store_true')
    parser.add_argument('--checkpoint-selection', choices=['best', 'last'], default='best')
    parser.add_argument('--criterion', choices=['bce', 'rank'])
    parser.add_argument('--rank-weight', type=float, default=0.)
    parser.add_argument('--spectral-dropout', type=float)
    parser.add_argument('--synthetic-snr-range', type=float, nargs=2)
    parser.add_argument('--synthetic-snr-probability', type=float, default=1.)
    parser.add_argument('--noise-real-probability', type=float, default=0.)
    parser.add_argument('--pseudo-weight', type=float, default=1.)
    parser.add_argument('--stop-at', type=float)
    parser.add_argument('--max-local-epochs', type=int)
    parser.add_argument('--optimizer', choices=['upstream', 'SGD'], default='upstream')
    parser.add_argument('--no-augmentation', action='store_true')
    parser.add_argument('--stable-gem', action='store_true')
    parser.add_argument('--audited-training', action='store_true')
    parser.add_argument('--sam-rho', type=float, default=0.)
    parser.add_argument('--gradient-clip', type=float, default=0.)
    parser.add_argument('--freeze-bn', action='store_true')
    parser.add_argument('--gain-jitter', type=float, default=0.)
    parser.add_argument('--resume-optimizer-moments', action='store_true')
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    if args.audited_training and (args.amp or not (args.bf16 or args.fp32)):
        raise ValueError('Audited training requires explicit BF16 or FP32')
    if any([args.sam_rho, args.gradient_clip, args.freeze_bn, args.gain_jitter, args.resume_optimizer_moments]) and not args.audited_training:
        raise ValueError('Final campaign optimizer options require audited training')
    if min(args.sam_rho, args.gradient_clip, args.gain_jitter) < 0:
        raise ValueError('Invalid optimizer or augmentation strength')
    if sum([args.amp, args.fp32, args.bf16]) > 1:
        raise ValueError('Choose one precision mode')
    if not 0 <= args.noise_real_probability <= 1:
        raise ValueError('Invalid real-noise probability')
    if not 0 <= args.synthetic_snr_probability <= 1:
        raise ValueError('Invalid controlled-SNR sampling probability')
    if args.max_local_epochs is not None and args.max_local_epochs < 1:
        raise ValueError('Local epoch cap must be positive')
    if not 0 < args.pseudo_weight <= 1:
        raise ValueError('Pseudo weight must be positive and at most one')
    metadata = json.loads((args.packed / "metadata.json").read_text())
    if bool(metadata.get("fixture", False)) != args.fixture:
        raise ValueError("Fixture data must be explicitly marked")
    torch.set_num_threads(8)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    sys.path.insert(0, str(TEAM3))
    # Protocol 5 is built into Python 3.8; the obsolete backport is unnecessary.
    sys.modules.setdefault("pickle5", pickle)
    from src.config import config_dict
    from src import dataset
    from src.util import seed_torch
    from src import train_helper
    train_helper.torch = CheckpointTorchProxy(args.checkpoint_dir, args.checkpoint_selection)
    if args.bf16:
        train_helper.autocast = functools.partial(torch.cuda.amp.autocast, dtype=torch.bfloat16)
        from src import models_1d
        def stable_gem(self, value, p=3, eps=1e-6):
            with torch.cuda.amp.autocast(enabled=False):
                return torch.nn.functional.avg_pool1d(value.float().clamp(min=eps).pow(p.float()), self.kernel_size).pow(1./p.float())
        models_1d.GeM.gem = stable_gem
    if args.stable_gem:
        from src import models_1d
        from team3_stable_training import patch_gem
        patch_gem(models_1d)
    if args.audited_training:
        from team3_stable_training import install_training
        install_training(train_helper, args)
    from src import optim as source_optim
    from team3_optimizer_compat import ranger_factory
    train_helper.RangerLars = ranger_factory(source_optim)
    original_train_epoch = train_helper.Trainer.train_epoch
    class BudgetFinished(Exception):
        pass
    def checked_train_epoch(self, loader):
        if args.stop_at is not None and time.time() >= args.stop_at:
            raise BudgetFinished('Training budget reached at an epoch boundary')
        if args.max_local_epochs is not None and getattr(self, 'completed_local_epochs', 0) >= args.max_local_epochs:
            raise BudgetFinished('Local epoch cap reached; original scheduler horizon preserved')
        loss = original_train_epoch(self, loader)
        self.completed_local_epochs = getattr(self, 'completed_local_epochs', 0)+1
        if not torch.isfinite(torch.as_tensor(loss)).item():
            raise FloatingPointError('Non-finite epoch training loss; this run cannot be promoted')
        return loss
    train_helper.Trainer.train_epoch = checked_train_epoch
    original_valid_epoch = train_helper.Trainer.valid_epoch
    def record_last_epoch(self, loader, model):
        loss, predictions = original_valid_epoch(self, loader, model)
        if args.keep_last and model is self.model:
            score = float(train_helper.get_score(self.valid_labels, predictions))
            if not np.isfinite(score) or not np.isfinite(predictions).all():
                raise FloatingPointError('Invalid validation predictions')
            last_state = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_valid_score': score, 'n_epoch': getattr(self, 'completed_local_epochs', 1)-1,
                        'scheduler': self.scheduler.state_dict(), 'valid_preds': predictions,
                        'checkpoint_selection': 'last', 'score_scope': 'Actual last epoch validation, not historical peak'}
            if args.audited_training:
                last_state['actual_optimizer_updates'] = self.actual_optimizer_updates
            train_helper.torch.save(last_state, output / f'Fold_{args.fold}_last_model.pth')
            if args.save_epoch_checkpoints:
                train_helper.torch.save(last_state, output / f'Fold_{args.fold}_epoch_{last_state["n_epoch"]+1:03d}_model.pth')
        return loss, predictions
    train_helper.Trainer.valid_epoch = record_last_epoch
    training_loop = train_helper.training_loop
    base = config_dict[args.config]
    name = args.experiment_name or (("fixture" if args.fixture else "pilot") + "_team3_" + args.config.replace("-", "_"))
    if Path(name).name != name or (args.fixture and not name.startswith('fixture')):
        raise ValueError('Invalid experiment name')
    if args.synthetic and args.synthetic_bank:
        raise ValueError('Use one synthetic-data source')
    output = ROOT / "experiments" / name / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    arrays = {split: np.load(args.packed / f"{split}.npy", mmap_mode="r") for split in ["train", "test"]}
    dataset.np = DatasetNumpyProxy(arrays)
    dataset.pd = DatasetPandasProxy()
    table = pd.read_csv(args.packed / "train.csv", dtype={"id": str})
    id_maps = {split: {value: index for index, value in enumerate(pd.read_csv(args.packed / f"{split}.csv", dtype={"id": str}).id)}
               for split in ["train", "test"]}
    def packed_path(sample_id, config, train=True):
        split = "train" if train else "test"
        return f"packed:{split}:{id_maps[split][sample_id]}"
    dataset.id_2_path_wave = packed_path
    fold_seed = args.fold_seed or base.seed
    table["fold"] = table[f"fold_{fold_seed}"]
    table["file_path"] = [f"packed:train:{index}" for index in range(len(table))]
    spectrum_path = output / "average_spectrum.pt"
    negative_indices = np.flatnonzero((table.target.values == 0) & (table.fold.values != args.fold))
    estimate_spectrum(arrays["train"], negative_indices, spectrum_path, device)
    overrides = dict(device=device, train_folds=[args.fold], output_dir=str(output), model_output_folder=str(output),
                     use_wandb=False, use_dp=False, num_workers=args.workers, avr_w0_path=str(spectrum_path),
                     checkpoint_folder=str(args.checkpoint_dir) if args.checkpoint_dir else None,
                     PL_folder=str(args.pseudo_label_dir) if args.pseudo_label_dir else None,
                     synthetic=args.synthetic is not None or args.synthetic_bank is not None)
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides['lr'] = args.lr
    if args.training_seed is not None:
        overrides['seed'] = args.training_seed
    if args.amp:
        overrides['use_autocast'] = True
    if args.fp32:
        overrides['use_autocast'] = False
    if args.bf16:
        overrides['use_autocast'] = True
    if args.criterion:
        overrides['crit'] = args.criterion
    if args.spectral_dropout is not None:
        overrides['sdrop'] = args.spectral_dropout
    if args.optimizer == 'SGD':
        def sgd_factory(parameters, lr, weight_decay, **unused):
            return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        train_helper.AdamW = sgd_factory
        overrides['optim'] = 'SGD'
    config = type("LocalReproduction", (base,), overrides)
    if args.rank_weight < 0 or (args.rank_weight and config.crit != 'bce'):
        raise ValueError('Composite rank loss requires BCE and a nonnegative rank weight')
    if args.pseudo_weight < 1:
        if not args.pseudo_label_dir or config.crit != 'bce':
            raise ValueError('Weighted soft pseudo labels require BCE and a pseudo-label source')
        teacher_info = json.loads((args.pseudo_label_dir / 'teacher_scope.json').read_text())
        if bool(teacher_info.get('fixture', False)) != args.fixture:
            raise ValueError('Fixture pseudo labels cannot be used in an official run')
        if teacher_info['fold_seed'] != fold_seed or teacher_info['excluded_fold'] != args.fold:
            raise ValueError('Pseudo-label teacher must exclude the student validation fold')
        if name.startswith(('campaign14h', 'campaign_final')) and not teacher_info.get('test_teacher_weights_exclude_student_validation_fold'):
            raise ValueError('Campaign14h teacher test weights must also exclude the student validation fold')
    if args.pseudo_weight < 1 or args.rank_weight > 0:
        import torch.nn.functional as functional
        original_bce = functional.binary_cross_entropy_with_logits
        def weighted_bce(predicted, target, **unused):
            loss = original_bce(predicted, target, reduction='none')
            weights = torch.where((target > 0) & (target < 1), args.pseudo_weight, 1.)
            total = (loss*weights).sum()/weights.sum()
            if args.rank_weight > 0:
                positive = predicted[target == 1].float()
                negative = predicted[target == 0].float()
                if len(positive) and len(negative):
                    total = total-args.rank_weight*functional.logsigmoid(positive[:, None]-negative[None, :]).mean()
            return total
        # The source trainer looks up F.binary_cross_entropy_with_logits.
        train_helper.F.binary_cross_entropy_with_logits = weighted_bce
    info = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    info.update({"trainer": "original third-place training_loop", "spectrum_negatives": len(negative_indices),
                 "spectrum_scope": "training fold only", "full_ensemble_reproduction": False,
                 "competition_score": None, "fold_seed": fold_seed, "output": str(output)})
    info['checkpoint_deserialization_device'] = 'cpu; model is copied to the explicitly assigned training GPU'
    if args.bf16:
        info['precision_policy'] = 'BF16 convolutions; FP32 GeM power and pooling; FP32 parameters and validation'
    if args.stable_gem:
        info['gem_policy'] = 'Exact generalized mean in FP32 log space; training and inference use the same adapter'
    if args.audited_training:
        info['trainer'] = 'Third-place model/dataset/fit with audited BF16/FP32 optimizer loop; every backward checked'
        info['optimizer_resume_policy'] = 'Moments and step restored; new LR/scheduler; not exact RNG continuation' if args.resume_optimizer_moments else 'Fresh optimizer and scheduler'
    if config.optim == 'RangerLars':
        info['optimizer_compatibility'] = 'Original Ralamb and Lookahead updates; modern Optimizer bookkeeping initialized, slow weights checkpointed'
    (output / "launch.json").write_text(json.dumps(info, indent=2))
    table.to_csv(output / "train_metadata.csv", index=False)
    synthetic = None
    if args.synthetic:
        with args.synthetic.open("rb") as handle:
            synthetic = pickle.load(handle)
    if args.synthetic_bank:
        synthetic = PackedSyntheticBank(args.synthetic_bank)
    physical = isinstance(synthetic, PackedSyntheticBank) and synthetic.metadata.get('injection') == 'physical'
    snr_values = None
    if args.synthetic_snr_range:
        low, high = args.synthetic_snr_range
        if not physical or not (0 < low < high):
            raise ValueError('SNR control requires an audited physical signal bank')
        snr_values = np.load(args.synthetic_bank / 'network_snr.npy', mmap_mode='r')
        if snr_values.shape != (len(synthetic.waves),) or not np.isfinite(snr_values).all() or np.any(snr_values <= 0):
            raise ValueError('Invalid template SNR normalization')
        info['snr_sampling'] = 'Uniform network optimal SNR using published adjusted PSD; waveform morphology/time unchanged'
    if physical and (synthetic.metadata.get('fold_seed') != fold_seed
                     or synthetic.metadata.get('excluded_prior_fold') != args.fold):
        raise ValueError('Physical prior bank must exclude the exact validation fold being evaluated')
    noise_bank = None
    if args.noise_bank:
        if not physical or args.pseudo_label_dir:
            raise ValueError('Independent synthetic noise requires physical signals and hard training labels')
        noise_info = json.loads((args.noise_bank / 'metadata.json').read_text())
        if noise_info['amplitude_scale'] != 1e20:
            raise ValueError('Noise bank has an incompatible amplitude scale')
        noise_bank = np.load(args.noise_bank / 'noise.npy', mmap_mode='r')
        if noise_bank.shape[1:] != (3, 4096) or len(noise_bank) != noise_info['count']:
            raise ValueError('Noise bank shape does not match metadata')
        info['synthetic_noise_source'] = noise_info
        (output / 'launch.json').write_text(json.dumps(info, indent=2))
    if physical or args.no_augmentation:
        class PhysicalDataRetriever(torch.utils.data.Dataset):
            def __init__(self, paths, targets, synthetic=None, Config=None):
                if not Config.use_raw_wave:
                    raise ValueError('Physical injection requires raw waveforms')
                self.paths, self.targets, self.synthetic = paths, targets, synthetic
                self.negative = np.flatnonzero(targets == 0).tolist()

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, index):
                target = self.targets[index]
                source = random.choice(self.negative) if target > 0 and self.synthetic is not None else index
                if noise_bank is None or random.random() < args.noise_real_probability:
                    wave = torch.tensor(dataset.np.load(self.paths[source])*1e20, dtype=torch.float32)
                else:
                    wave = torch.tensor(np.array(noise_bank[random.randrange(len(noise_bank))], copy=True))
                if target > 0 and self.synthetic is not None:
                    # KDL already samples physical amplitude and arrival time.
                    signal_index = random.randrange(len(self.synthetic.waves))
                    controlled = snr_values is not None and (args.synthetic_snr_probability == 1. or random.random() < args.synthetic_snr_probability)
                    scale = random.uniform(*args.synthetic_snr_range) / float(snr_values[signal_index]) if controlled else 1.
                    wave += torch.tensor(np.array(self.synthetic[signal_index], copy=True), dtype=torch.float32)*scale
                return wave, torch.tensor(target, dtype=torch.float32)
        train_helper.DataRetriever = PhysicalDataRetriever
        info['dataset_adaptation'] = 'Physical amplitude/time injection without legacy rescaling or shifts; no augmentation'
        (output / 'launch.json').write_text(json.dumps(info, indent=2))
    if args.checkpoint_dir:
        initial_launch_path = args.checkpoint_dir / 'launch.json'
        if initial_launch_path.exists():
            initial_launch = json.loads(initial_launch_path.read_text())
            if initial_launch.get('fold') != args.fold or initial_launch.get('fold_seed', 48) != fold_seed:
                raise ValueError('Initial checkpoint must exclude the exact current validation fold')
            if bool(initial_launch.get('fixture', False)) != args.fixture:
                raise ValueError('Fixture checkpoints cannot initialize an official competition run')
        initial = args.checkpoint_dir / f'Fold_{args.fold}_{args.checkpoint_selection}_model.pth'
        fallback = output / f'Fold_{args.fold}_best_model.pth'
        if not fallback.exists():
            shutil.copy2(initial, fallback)
            info['initial_checkpoint_retained_if_no_validation_improvement'] = True
            (output / 'launch.json').write_text(json.dumps(info, indent=2))
    seed_torch(config.seed)
    torch.backends.cudnn.benchmark = config.use_cudnn
    try:
        training_loop(table, config, synthetic=synthetic)
    except BudgetFinished as error:
        print(str(error), flush=True)
        info['training_stopped_at_epoch_boundary'] = True
        (output / 'launch.json').write_text(json.dumps(info, indent=2))
    checkpoint = output / f"Fold_{args.fold}_best_model.pth"
    if not checkpoint.exists():
        raise RuntimeError("The original trainer did not produce a checkpoint")


if __name__ == "__main__":
    main()
