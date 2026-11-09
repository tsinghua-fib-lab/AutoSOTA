'''Training with static pseudo label
'''
import argparse
from pathlib import Path
from pprint import pprint
import sys
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.utils.data as D
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import pickle
import traceback

from kuma_utils.torch import TorchTrainer, TorchLogger
from kuma_utils.torch.utils import get_time, seed_everything, fit_state_dict
from kuma_utils.utils import sigmoid

from configs import *
from utils import print_config, notify_me
from training_extras import make_tta_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='Baseline',
                        help="config name in configs.py")
    parser.add_argument("--hardware", type=str, default='A100',
                        help="hardware name (this determines num of cpus and gpus)")
    parser.add_argument("--limit_fold", type=int, default=-1,
                        help="train only specified fold")
    parser.add_argument("--inference", action='store_true',
                        help="inference")
    parser.add_argument("--tta", action='store_true', 
                        help="test time augmentation ")
    parser.add_argument("--gpu", nargs="+", default=[])
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--silent", action='store_true')
    parser.add_argument("--progress_bar", action='store_true')
    parser.add_argument("--skip_existing", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--wait", type=int, default=0,
                        help="time (sec) to wait before execution")
    opt = parser.parse_args()
    pprint(opt)

    ''' Configure hardware '''
    N_CPU, N_RAM, N_GPU, N_GRAM = HW_CFG[opt.hardware]
    if len(opt.gpu) == 0:
        opt.gpu = None # use all GPUs
    elif len(opt.gpu) > N_GPU:
        raise ValueError(f'Maximum GPUs allowed is {N_GPU}')

    ''' Configure path '''
    cfg = eval(opt.config)
    assert cfg.pseudo_labels is not None
    export_dir = Path('results') / cfg.name
    export_dir.mkdir(parents=True, exist_ok=True)

    ''' Configure logger '''
    log_items = [
        'epoch', 'train_loss', 'train_metric', 'train_monitor', 
        'valid_loss', 'valid_metric', 'valid_monitor', 
        'learning_rate', 'early_stop'
    ]
    if opt.debug:
        log_items += ['gpu_memory']
    if opt.limit_fold >= 0:
        logger_path = f'{cfg.name}_fold{opt.limit_fold}_{get_time("%y%m%d%H%M")}.log'
    else:
        logger_path = f'{cfg.name}_{get_time("%y%m%d%H%M")}.log'
    LOGGER = TorchLogger(
        export_dir / logger_path, 
        log_items=log_items, file=not opt.silent
    )
    if opt.wait > 0:
        LOGGER(f'Waiting for {opt.wait} sec.')
        time.sleep(opt.wait)

    ''' Prepare data '''
    seed_everything(cfg.seed, cfg.deterministic)
    print_config(cfg, LOGGER)
    train = pd.read_csv(cfg.train_path)
    test = pd.read_csv(cfg.test_path)
    if cfg.debug:
        train = train.iloc[:10000]
        test = test.iloc[:1000]
    splitter = cfg.splitter
    fold_iter = list(splitter.split(X=train, y=train['target']))
    test_labels = sigmoid(np.load(cfg.pseudo_labels['path']))
    if cfg.pseudo_labels['confident_samples'] is not None:
        lower, upper = cfg.pseudo_labels['confident_samples']
        order = test_labels.mean(0).reshape(-1).argsort()
        confident_idx = np.concatenate([
            order[:int(len(order) * lower)], 
            order[int(len(order) * upper):]
        ])
        LOGGER(f'{len(confident_idx)} confident samples extracted.')
    else:
        confident_idx = slice(None)

    '''
    Training
    '''
    scores = []
    if cfg.train_cache is None:
        train_cache = None
    else:
        with open(cfg.train_cache, 'rb') as f:
            train_cache = pickle.load(f)
    if cfg.test_cache is None:
        test_cache = None
    else:
        with open(cfg.test_cache, 'rb') as f:
            test_cache = pickle.load(f)
    
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):
        
        if opt.limit_fold >= 0 and fold != opt.limit_fold:
            continue  # skip fold

        if opt.inference:
            continue

        if opt.skip_existing and (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'checkpoint fold{fold}.pt already exists.')
            continue

        LOGGER(f'===== TRAINING FOLD {fold} =====')

        train_fold = train.iloc[train_idx]
        valid_fold = train.iloc[valid_idx]

        LOGGER(f'train positive: {train_fold.target.values.mean(0)} ({len(train_fold)})')
        LOGGER(f'valid positive: {valid_fold.target.values.mean(0)} ({len(valid_fold)})')
        if cfg.pseudo_labels['hard_label'] == 'mix':
            test_labels_fold = test_labels[fold].reshape(-1)
            test_labels_fold[confident_idx] = test_labels_fold[confident_idx].round()
            test_paths_fold = test['path'].values
        elif cfg.pseudo_labels['hard_label']:
            test_labels_fold = test_labels[fold][confident_idx].reshape(-1)
            test_paths_fold = test['path'].values[confident_idx]
            test_labels_fold = test_labels_fold.round()
        else:
            test_labels_fold = test_labels[fold].reshape(-1)
            test_paths_fold = test['path'].values

        train_data = cfg.dataset(
            paths=train_fold['path'].values, targets=train_fold['target'].values,
            transforms=cfg.transforms['train'], cache=train_cache, is_test=False,
            pseudo_label=True, 
            test_targets=test_labels_fold, 
            test_paths=test_paths_fold, 
            test_cache=test_cache, 
            **cfg.dataset_params)
        valid_data = cfg.dataset(
            paths=valid_fold['path'].values, targets=valid_fold['target'].values,
            transforms=cfg.transforms['test'], cache=train_cache, is_test=True,
            **cfg.dataset_params)

        train_loader = D.DataLoader(
            train_data, batch_size=cfg.batch_size, shuffle=True,
            num_workers=0, pin_memory=False)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=0, pin_memory=False)

        model = cfg.model(**cfg.model_params)

        # Load snapshot
        if cfg.weight_path is not None:
            if cfg.weight_path.is_dir():
                weight_path = cfg.weight_path / f'fold{fold}.pt'
            else:
                weight_path = cfg.weight_path
            LOGGER(f'{weight_path} loaded.')
            weight = torch.load(weight_path, 'cpu')['model']
            fit_state_dict(weight, model)
            model.load_state_dict(weight, strict=False)
            del weight; gc.collect()
        # Load SeqCNN model
        if hasattr(model, 'cnn_path'):
            checkpoint = torch.load(model.cnn_path / f'fold{fold}.pt')['model']
            model.load_cnn(checkpoint)
        if hasattr(model, 'seq_path'):
            checkpoint = torch.load(model.seq_path / f'fold{fold}.pt')['model']
            model.load_seq(checkpoint)
        if hasattr(model, 'freeze'):
            model.freeze_seq()
            model.freeze_cnn()

        optimizer = cfg.optimizer(model.parameters(), **cfg.optimizer_params)
        scheduler = cfg.scheduler(optimizer, **cfg.scheduler_params)
        FIT_PARAMS = {
            'loader': train_loader,
            'loader_valid': valid_loader,
            'criterion': cfg.criterion,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'scheduler_target': cfg.scheduler_target,
            'batch_scheduler': cfg.batch_scheduler, 
            'num_epochs': cfg.num_epochs,
            'callbacks': deepcopy(cfg.callbacks),
            'hook': cfg.hook,
            'export_dir': export_dir,
            'eval_metric': cfg.eval_metric,
            'monitor_metrics': cfg.monitor_metrics,
            'fp16': cfg.amp,
            'parallel': cfg.parallel,
            'deterministic': cfg.deterministic, 
            'clip_grad': cfg.clip_grad, 
            'max_grad_norm': cfg.max_grad_norm,
            'random_state': cfg.seed,
            'logger': LOGGER,
            'progress_bar': opt.progress_bar, 
            'resume': opt.resume
        }
        if not cfg.debug:
            notify_me(f'[{cfg.name}:fold{opt.limit_fold}]\nTraining started.')
        try:
            trainer = TorchTrainer(model, serial=f'fold{fold}', device=opt.gpu)
            trainer.fit(**FIT_PARAMS)
        except Exception as e:
            err = traceback.format_exc()
            LOGGER(err)
            if not opt.silent:
                notify_me('\n'.join([
                    f'[{cfg.name}:fold{opt.limit_fold}]', 
                    'Training stopped due to:', 
                    f'{traceback.format_exception_only(type(e), e)}'
                ]))
        del model, trainer, train_data, valid_data; gc.collect()
        torch.cuda.empty_cache()


    '''
    Inference
    '''
    predictions = np.full((cfg.cv, len(test), 1), 0.5, dtype=np.float32)
    outoffolds = np.full((len(train), 1), 0.5, dtype=np.float32)
    test_data = cfg.dataset(
        paths=test['path'].values, transforms=cfg.transforms['test'], 
        cache=test_cache, is_test=True, **cfg.dataset_params)
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):

        if opt.limit_fold >= 0:
            if fold == 0:
                checkpoint = torch.load(export_dir/f'fold{opt.limit_fold}.pt', 'cpu')
                scores.append(checkpoint['state']['best_score'])
            continue

        if not (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'fold{fold}.pt missing. No target to predict.')
            continue

        LOGGER(f'===== INFERENCE FOLD {fold} =====')

        valid_fold = train.iloc[valid_idx]
        valid_data = cfg.dataset(
            paths=valid_fold['path'].values, targets=valid_fold['target'].values,
            cache=train_cache, transforms=cfg.transforms['test'], is_test=True,
            **cfg.dataset_params)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=0, pin_memory=False)
        test_loader = D.DataLoader(
            test_data, batch_size=cfg.batch_size, shuffle=False, 
            num_workers=0, pin_memory=False)

        model = cfg.model(**cfg.model_params)
        checkpoint = torch.load(export_dir/f'fold{fold}.pt', 'cpu')
        fit_state_dict(checkpoint['model'], model)
        try:
            model.load_state_dict(checkpoint['model'])
        except: # drop preprocess module for compatibility
            model.cnn = nn.Sequential(
                *[model.cnn[i+1] for i in range(len(model.cnn)-1)])
            model.load_state_dict(checkpoint['model'])
        scores.append(checkpoint['state']['best_score'])
        del checkpoint; gc.collect()

        trainer = TorchTrainer(model, serial=f'fold{fold}', device=opt.gpu)
        trainer.register(hook=cfg.hook, callbacks=cfg.callbacks)

        if opt.tta: # flip wave TTA
            tta_transform = Compose(
                cfg.transforms['test'].transforms + [FlipWave(always_apply=True)])
            LOGGER(f'[{fold}] pred0 {test_loader.dataset.transforms}')
            prediction0 = trainer.predict(test_loader, progress_bar=opt.progress_bar)
            test_loader = make_tta_dataloader(test_loader, cfg.dataset, dict(
                paths=test['path'].values, transforms=tta_transform, 
                cache=test_cache, is_test=True, **cfg.dataset_params
            ))
            LOGGER(f'[{fold}] pred1 {test_loader.dataset.transforms}')
            prediction1 = trainer.predict(test_loader, progress_bar=opt.progress_bar)
            prediction_fold = (prediction0 + prediction1) / 2

            LOGGER(f'[{fold}] oof0 {valid_loader.dataset.transforms}')
            outoffold0 = trainer.predict(valid_loader, progress_bar=opt.progress_bar)
            valid_loader = make_tta_dataloader(valid_loader, cfg.dataset, dict(
                paths=valid_fold['path'].values, targets=valid_fold['target'].values,
                cache=train_cache, transforms=tta_transform, is_test=True,
                **cfg.dataset_params))
            LOGGER(f'[{fold}] oof1 {valid_loader.dataset.transforms}')
            outoffold1 = trainer.predict(valid_loader, progress_bar=opt.progress_bar)
            outoffold = (outoffold0 + outoffold1) / 2
        else:
            prediction_fold = trainer.predict(test_loader, progress_bar=opt.progress_bar)
            outoffold = trainer.predict(valid_loader, progress_bar=opt.progress_bar)

        predictions[fold] = prediction_fold
        outoffolds[valid_idx] = outoffold

        del model, trainer, valid_data; gc.collect()
        torch.cuda.empty_cache()

    if opt.limit_fold < 0:
        if opt.tta:
            np.save(export_dir/'outoffolds_tta', outoffolds)
            np.save(export_dir/'predictions_tta', predictions)
        else:
            np.save(export_dir/'outoffolds', outoffolds)
            np.save(export_dir/'predictions', predictions)

    LOGGER(f'scores: {scores}')
    LOGGER(f'mean +- std: {np.mean(scores):.5f} +- {np.std(scores):.5f}')
    if not cfg.debug:
        notify_me('\n'.join([
            f'[{cfg.name}:fold{opt.limit_fold}]',
            'Training has finished successfully.',
            f'mean +- std: {np.mean(scores):.5f} +- {np.std(scores):.5f}'
        ]))
