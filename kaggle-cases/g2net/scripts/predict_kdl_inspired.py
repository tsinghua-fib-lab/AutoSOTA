"""Aligned predictions for the explicitly reconstructed KDL-inspired model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from kdl_inspired_model import SplitMultiScaleResNet
from predict_upstream import predict
from upstream_models import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--tta-mode', choices=['none', 'sign'], default='none')
    args = parser.parse_args()
    torch.set_num_threads(6); torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = SplitMultiScaleResNet(state['width'])
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.to(device).eval()
    train = pd.read_csv(ROOT / 'data/packed/train.csv', dtype={'id': str})
    test = pd.read_csv(ROOT / 'data/packed/test.csv', dtype={'id': str})
    valid = np.flatnonzero(train[f"fold_{state['fold_seed']}"].values == state['fold'])
    waves = np.load(ROOT / 'data/packed/train.npy', mmap_mode='r')
    test_waves = np.load(ROOT / 'data/packed/test.npy', mmap_mode='r')
    heldout = predict(model, waves, valid, None, args.batch_size, device, False, 'kdl_inspired', tta_mode=args.tta_mode)
    test_predicted = predict(model, test_waves, np.arange(len(test)), None, args.batch_size, device, False, 'kdl_inspired', tta_mode=args.tta_mode)
    if not np.isfinite(heldout).all() or not np.isfinite(test_predicted).all():
        raise FloatingPointError('Invalid predicted probabilities')
    args.output.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(train), np.nan, dtype=np.float32); oof[valid] = heldout
    np.save(args.output / 'oof.npy', oof); np.save(args.output / 'test_predictions.npy', test_predicted)
    pd.DataFrame({'id': test.id, 'target': test_predicted}).to_csv(args.output / 'submission.csv', index=False)
    evidence = dict(family='new_kdl_inspired', fixture=False, fold=state['fold'], fold_seed=state['fold_seed'],
                    checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    train_id_sha256=hashlib.sha256('\n'.join(train.id).encode()).hexdigest(),
                    test_id_sha256=hashlib.sha256('\n'.join(test.id).encode()).hexdigest(),
                    validation_samples=len(valid), test_samples=len(test), unpredicted_oof_rows=int(np.isnan(oof).sum()),
                    competition_validation_auc=float(roc_auc_score(train.target.values[valid], heldout)),
                    data_scope='full official training data', full_competition_score_reproduction=False,
                    kaggle_public_auc=None, kaggle_private_auc=None, tta=args.tta_mode != 'none', tta_mode=args.tta_mode)
    (args.output / 'metrics.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == '__main__':
    main()
