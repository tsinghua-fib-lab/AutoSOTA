"""Average same-fold checkpoints and optionally refresh BN on training rows only."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from upstream_models import ROOT, TEAM3, make_team3
from predict_upstream import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--bn-samples', type=int, default=16384)
    args = parser.parse_args()
    if len(args.checkpoints) < 2 or args.output.exists():
        raise ValueError('Use at least two checkpoints and a new output directory')
    launches = [json.loads((p.parent / 'launch.json').read_text()) for p in args.checkpoints]
    signature = lambda j: (j['fold'], j['fold_seed'], j['config'], bool(j.get('fixture', False)))
    if len(set(map(signature, launches))) != 1 or launches[0].get('fixture'):
        raise ValueError('Checkpoint averaging requires matching official validation exclusions and architecture')
    fold, seed, config_name, _ = signature(launches[0])
    states = [torch.load(p, map_location='cpu', weights_only=False)['model_state_dict'] for p in args.checkpoints]
    if any(set(s) != set(states[0]) for s in states):
        raise ValueError('Checkpoint keys differ')
    averaged = {}
    for key, value in states[0].items():
        if any(s[key].shape != value.shape for s in states):
            raise ValueError('Checkpoint shapes differ')
        if value.is_floating_point():
            averaged[key] = torch.stack([s[key].float() for s in states]).mean(0).to(value.dtype)
            if not torch.isfinite(averaged[key]).all():
                raise ValueError('Nonfinite averaged state')
        else:
            averaged[key] = states[-1][key].clone()
    sys.path.insert(0, str(TEAM3))
    from src.config import config_dict
    cfg = config_dict[config_name]
    torch.set_num_threads(8); torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(7148); torch.cuda.manual_seed_all(7148)
    model = make_team3(args.checkpoints[0].parent / 'average_spectrum.pt', cfg.channels,
                      cfg.proba_final_layer, cfg.model_module, cfg.sdrop,
                      stable_gem=any(j.get('stable_gem', False) for j in launches))
    model.load_state_dict(averaged, strict=True); model.to(device).eval()
    train = pd.read_csv(ROOT / 'data/packed/train.csv', dtype={'id': str})
    waves = np.load(ROOT / 'data/packed/train.npy', mmap_mode='r')
    training_rows = np.flatnonzero(train[f'fold_{seed}'].values != fold)
    indices = np.random.default_rng(7148+fold).choice(training_rows, min(args.bn_samples, len(training_rows)), replace=False)
    if np.any(train[f'fold_{seed}'].values[indices] == fold):
        raise ValueError('BN recalibration included held-out data')
    bn = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if args.bn_samples:
        for module in bn:
            module.reset_running_stats(); module.train()
        seen = 0
        with torch.no_grad():
            for start in range(0, len(indices), 128):
                batch = torch.from_numpy(np.array(waves[indices[start:start+128]], copy=True)).to(device)
                for module in bn:
                    module.momentum = len(batch)/(seen+len(batch))
                result = model(batch)
                if not torch.isfinite(result).all():
                    raise ValueError('Nonfinite BN recalibration forward')
                seen += len(batch)
        model.eval()
    valid = np.flatnonzero(train[f'fold_{seed}'].values == fold)
    heldout = predict(model, waves, valid, None, 128, device, False, 'team3', 0, 'none')
    if not np.isfinite(heldout).all():
        raise ValueError('Nonfinite averaged checkpoint predictions')
    score = float(roc_auc_score(train.target.values[valid], heldout))
    args.output.mkdir(parents=True)
    destination = args.output / f'Fold_{fold}_best_model.pth'
    torch.save(dict(model_state_dict=model.state_dict(), best_valid_score=score, valid_preds=heldout,
                    n_epoch=-1, checkpoint_selection='parameter_average', inference_only=True), destination)
    shutil.copy2(args.checkpoints[0].parent / 'average_spectrum.pt', args.output / 'average_spectrum.pt')
    launch = dict(launches[0], checkpoint_dir=None, experiment_name=args.output.parent.name, output=str(args.output),
                  stable_gem=any(j.get('stable_gem', False) for j in launches),
                  parameter_average_sources=[str(p) for p in args.checkpoints],
                  source_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in args.checkpoints],
                  bn_recalibration_samples=len(indices), bn_scope='Current fold training rows only; stochastic depth/dropout disabled',
                  bn_training_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                  no_tta_averaged_validation_auc=score)
    np.save(args.output / 'bn_training_indices.npy', indices)
    (args.output / 'launch.json').write_text(json.dumps(launch, indent=2))
    print(json.dumps(launch, indent=2), flush=True)


if __name__ == '__main__':
    main()
