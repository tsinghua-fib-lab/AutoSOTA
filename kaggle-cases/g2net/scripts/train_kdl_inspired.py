"""Budgeted physical pretraining and real-data fine tuning of a new KDL model."""
import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from kdl_inspired_model import SplitMultiScaleResNet
from upstream_models import ROOT

STOP = False


def request_stop(*unused):
    global STOP
    STOP = True


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix('.partial.pth')
    torch.save(value, temporary)
    temporary.replace(path)


class WaveDataset(Dataset):
    def __init__(self, waves, indices, labels, bank=None, noise=None, real_probability=0., snr_range=None):
        self.waves, self.indices, self.labels = waves, indices, labels
        self.bank = bank
        self.noise = noise
        self.real_probability = real_probability
        self.snr_range = snr_range
        self.negatives = indices[labels[indices] == 0]
        if bank:
            self.signals = np.load(bank / 'signals.npy', mmap_mode='r')
            self.parameters = np.load(bank / 'parameters.npy', mmap_mode='r')
            self.snr = np.load(bank / 'network_snr.npy', mmap_mode='r')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        target = float(self.labels[index])
        params = np.zeros(3, np.float32)
        if self.bank:
            if self.noise is not None and random.random() >= self.real_probability:
                wave = np.array(self.noise[random.randrange(len(self.noise))], copy=True)
            else:
                wave = np.array(self.waves[random.choice(self.negatives)], copy=True)
            if target == 1:
                template = random.randrange(len(self.signals))
                rho = random.uniform(*self.snr_range) if self.snr_range else float(self.snr[template])
                wave += self.signals[template]*(rho/float(self.snr[template]))
                params[:] = [math.log(float(self.parameters[template, 2])/20),
                             math.log(float(self.parameters[template, 3])),
                             math.log(rho/10)]
        else:
            wave = np.array(self.waves[index], copy=True)
        return torch.from_numpy(wave), torch.tensor(target), torch.from_numpy(params)


def seed_worker(index):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, labels = [], []
    for wave, target, _ in loader:
        predictions.append(model(wave.to(device, non_blocking=True)).sigmoid().cpu().numpy())
        labels.append(target.numpy())
    predicted = np.concatenate(predictions)
    if not np.isfinite(predicted).all():
        raise FloatingPointError('Non-finite validation probabilities')
    return float(roc_auc_score(np.concatenate(labels), predicted)), predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--noise-bank', type=Path)
    parser.add_argument('--real-noise-probability', type=float, default=0.)
    parser.add_argument('--auxiliary-weight', type=float, default=.05)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--stop-at', type=float, required=True)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--snr-range', type=float, nargs=2)
    args = parser.parse_args()
    if args.snr_range and (not args.bank or not 0 < args.snr_range[0] < args.snr_range[1]):
        raise ValueError('Controlled SNR requires physical templates and a positive increasing range')
    if Path(args.tag).name != args.tag or not 0 <= args.fold <= 4 or not 0 <= args.real_noise_probability <= 1:
        raise ValueError('Invalid experiment arguments')
    output = ROOT / 'experiments' / args.tag
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'history.json').exists():
        raise FileExistsError('Existing experiment must not be overwritten')
    if args.bank:
        metadata = json.loads((args.bank / 'metadata.json').read_text())
        audit = json.loads((args.bank / 'snr_audit.json').read_text())
        if metadata['excluded_prior_fold'] != args.fold or metadata['fold_seed'] != 48 or not audit['passed']:
            raise ValueError('Physical prior/SNR audit does not match validation fold')
    else:
        args.auxiliary_weight = 0.
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    seed = 9148+args.fold
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(6)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    torch.backends.cudnn.benchmark = True
    table = pd.read_csv(ROOT / 'data/packed/train.csv')
    waves = np.load(ROOT / 'data/packed/train.npy', mmap_mode='r')
    training = np.flatnonzero(table.fold_48.values != args.fold)
    validation = np.flatnonzero(table.fold_48.values == args.fold)
    labels = table.target.values
    noise = np.load(args.noise_bank / 'noise.npy', mmap_mode='r') if args.noise_bank else None
    if noise is not None:
        noise_meta = json.loads((args.noise_bank / 'metadata.json').read_text())
        if noise_meta['amplitude_scale'] != 1e20 or noise.shape != (noise_meta['count'], 3, 4096):
            raise ValueError('Invalid noise bank')
    train_loader = DataLoader(WaveDataset(waves, training, labels, args.bank, noise, args.real_noise_probability, args.snr_range),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, worker_init_fn=seed_worker, persistent_workers=True)
    valid_loader = DataLoader(WaveDataset(waves, validation, labels), batch_size=args.batch_size*2,
                              shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    model = SplitMultiScaleResNet(args.width).to(device)
    best = -math.inf
    if args.checkpoint:
        initial = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if initial['fold'] != args.fold or initial['fold_seed'] != 48 or initial['width'] != args.width:
            raise ValueError('Initial checkpoint validation scope/model mismatch')
        model.load_state_dict(initial['model_state_dict'], strict=True)
        best, _ = evaluate(model, valid_loader, device)
        initial['best_valid_score'] = best
        atomic_checkpoint(output / 'best.pth', initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader)*args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        max(.05, .5*(1+math.cos(math.pi*step/max(1, total_steps)))))
    launch = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    launch.update(model='KDL-inspired split multi-scale encoders with unpretrained ResNet34',
                  exact_winner_reproduction=False, seed=seed, fold_seed=48, precision=args.precision,
                  augmentation='none', train_samples=len(training), validation_samples=len(validation),
                  preprocessing='20.43Hz order-5 Butterworth magnitude squared in FFT with odd reflection; no whitening',
                  historical_prior_scope='External published chirp-mass estimates and PSD; historical estimator scope unknown')
    atomic_json(output / 'launch.json', launch)
    history = []
    began = time.time()
    for epoch in range(args.epochs):
        if STOP or time.time() >= args.stop_at:
            break
        model.train()
        started = time.time()
        total, samples, auxiliary_total = 0., 0, 0.
        for wave, target, physical in train_loader:
            if STOP or time.time() >= args.stop_at:
                break
            wave = wave.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            physical = physical.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.precision == 'bf16', dtype=torch.bfloat16):
                predicted, regression = model(wave, auxiliary=True)
                loss = F.binary_cross_entropy_with_logits(predicted, target)
                positives = target > .5
                aux = F.smooth_l1_loss(regression[positives], physical[positives]) if args.bank and positives.any() else loss*0
                loss = loss+args.auxiliary_weight*aux
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Non-finite gradient norm')
            optimizer.step(); scheduler.step()
            total += loss.item()*len(target); auxiliary_total += aux.item()*len(target); samples += len(target)
        if not samples:
            break
        auc, predictions = evaluate(model, valid_loader, device)
        entry = dict(epoch=epoch+1, samples=samples, loss=total/samples, auxiliary_loss=auxiliary_total/samples,
                     validation_auc=auc, seconds=time.time()-started, lr=optimizer.param_groups[0]['lr'],
                     complete_epoch=samples == len(training))
        history.append(entry)
        state = dict(model_state_dict=model.state_dict(), width=args.width, fold=args.fold, fold_seed=48,
                     epoch=epoch+1, best_valid_score=max(best, auc), last_valid_score=auc,
                     optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                     launch=launch)
        atomic_checkpoint(output / 'last.pth', state)
        if auc > best:
            best = auc
            atomic_checkpoint(output / 'best.pth', state)
            np.save(output / 'best_validation.npy', predictions)
        atomic_json(output / 'history.json', history)
        print(json.dumps(entry), flush=True)
    if not (output / 'best.pth').exists():
        raise RuntimeError('No usable checkpoint produced within the training budget')
    atomic_json(output / 'training_result.json', dict(best_validation_auc=best, epochs=len(history),
                seconds=time.time()-began, stopped_by_budget=STOP or time.time() >= args.stop_at,
                finished_at=time.time()))


if __name__ == '__main__':
    main()
