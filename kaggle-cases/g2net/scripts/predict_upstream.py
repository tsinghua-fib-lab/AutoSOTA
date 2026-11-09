"""Produce correctly aligned held-out and test predictions from original models."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from upstream_models import ROOT, TEAM3, make_eventhorizon, make_team3, apply_eventhorizon_transform


@torch.no_grad()
def predict(model, array, indices, transform, batch_size, device, tta, family, mc_folds=0, tta_mode=None):
    predictions = []
    for start in range(0, len(indices), batch_size):
        waves = torch.from_numpy(np.array(array[indices[start:start + batch_size]], copy=True))
        if transform is not None:
            waves = apply_eventhorizon_transform(waves, transform)
        waves = waves.to(device)
        variants = [waves]
        if tta_mode is None:
            mode = ('sign_swap' if family == 'team3' else 'sign') if tta else 'none'
        else:
            mode = tta_mode
        if mode != 'none':
            variants.append(-waves)
            if mode == 'sign_swap':
                variants.extend([waves[:, [1, 0, 2]], -waves[:, [1, 0, 2]]])
        def forward(value):
            if family == 'team3' and mc_folds > 0:
                return model(value, use_MC=True, MC_folds=mc_folds)
            return model(value)
        result = torch.stack([forward(value).sigmoid().reshape(-1) for value in variants]).mean(0)
        predictions.append(result.cpu().numpy())
    return np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["eventhorizon", "team3"], required=True)
    parser.add_argument("--config", default="Seq03aug3")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--psd", type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--survival", type=float)
    parser.add_argument("--packed", type=Path, default=ROOT / "data/packed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument('--tta-mode', choices=['none', 'sign', 'sign_swap'])
    parser.add_argument('--mc-folds', type=int, default=0)
    parser.add_argument('--fold-seed', type=int)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    metadata = json.loads((args.packed / "metadata.json").read_text())
    if bool(metadata.get("fixture", False)) != args.fixture:
        raise ValueError("Fixture data must be explicitly marked")
    torch.set_num_threads(8)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    if args.family == "eventhorizon":
        model, config = make_eventhorizon(args.config, pretrained=False)
        transform = config.transforms["test"]
        seed = config.seed
        state_key = "model"
    else:
        psd = args.psd or args.checkpoint.parent / "average_spectrum.pt"
        launch_path = args.checkpoint.parent / 'launch.json'
        launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
        sys.path.insert(0, str(TEAM3))
        from src.config import config_dict
        config_name = launch.get('config', args.config)
        source_config = config_dict[config_name]
        args.config = config_name
        args.width = args.width or source_config.channels
        args.survival = args.survival if args.survival is not None else source_config.proba_final_layer
        model = make_team3(psd, args.width, args.survival, source_config.model_module, source_config.sdrop,
                           stable_gem=launch.get('stable_gem', False))
        transform = None
        seed = launch.get('fold_seed', 48)
        state_key = "model_state_dict"
    seed = args.fold_seed or seed
    prediction_seed = seed+101*args.fold
    torch.manual_seed(prediction_seed)
    torch.cuda.manual_seed_all(prediction_seed)
    np.random.seed(prediction_seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint[state_key], strict=True)
    model = model.to(device).eval()
    train = pd.read_csv(args.packed / "train.csv", dtype={"id": str})
    test = pd.read_csv(args.packed / "test.csv", dtype={"id": str})
    train_array = np.load(args.packed / "train.npy", mmap_mode="r")
    test_array = np.load(args.packed / "test.npy", mmap_mode="r")
    valid = np.flatnonzero(train[f"fold_{seed}"].values == args.fold)
    heldout = predict(model, train_array, valid, transform, args.batch_size, device, args.tta, args.family, args.mc_folds, args.tta_mode)
    test_preds = predict(model, test_array, np.arange(len(test)), transform, args.batch_size, device, args.tta, args.family, args.mc_folds, args.tta_mode)
    if not np.isfinite(heldout).all() or not np.isfinite(test_preds).all():
        raise ValueError("Non-finite predictions")
    if np.any((test_preds < 0) | (test_preds > 1)) or test.id.duplicated().any():
        raise ValueError("Invalid submission")
    args.output.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(train), np.nan, dtype=np.float32)
    oof[valid] = heldout
    np.save(args.output / "oof.npy", oof)
    np.save(args.output / "test_predictions.npy", test_preds)
    pd.DataFrame({"id": test.id, "target": test_preds}).to_csv(args.output / "submission.csv", index=False)
    auc = float(roc_auc_score(train.target.iloc[valid], heldout))
    report = {"family": args.family, "config": args.config, "checkpoint": str(args.checkpoint),
              "fixture": args.fixture, "fold_seed": seed, "fold": args.fold, "tta": args.tta,
              "validation_samples": len(valid), "test_samples": len(test),
              "competition_validation_auc": None if args.fixture else auc,
              "fixture_validation_auc": auc if args.fixture else None,
              "kaggle_public_auc": None, "kaggle_private_auc": None,
              "unpredicted_oof_rows": int(np.isnan(oof).sum()), 'mc_folds': args.mc_folds,
              'train_id_sha256': hashlib.sha256('\n'.join(train.id).encode()).hexdigest(),
              'test_id_sha256': hashlib.sha256('\n'.join(test.id).encode()).hexdigest()}
    report['prediction_seed'] = prediction_seed
    report['tta_mode'] = args.tta_mode or ('sign_swap' if args.tta and args.family == 'team3' else ('sign' if args.tta else 'none'))
    report['checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.family == 'team3' and launch.get('checkpoint_dir'):
        initial_path = Path(launch['checkpoint_dir']) / f"Fold_{args.fold}_{launch.get('checkpoint_selection', 'best')}_model.pth"
        if not initial_path.is_absolute():
            initial_path = ROOT / initial_path
        if initial_path.exists():
            report['initial_checkpoint_sha256'] = hashlib.sha256(initial_path.read_bytes()).hexdigest()
            report['checkpoint_byte_identical_to_initial'] = report['checkpoint_sha256'] == report['initial_checkpoint_sha256']
            report['training_improvement_claim'] = 'No checkpoint change' if report['checkpoint_byte_identical_to_initial'] else 'Checkpoint changed; assess aligned validation separately'
    report['data_scope'] = 'official subset pilot' if metadata.get('partial_pilot') else 'full official training data'
    report['full_competition_score_reproduction'] = False
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
