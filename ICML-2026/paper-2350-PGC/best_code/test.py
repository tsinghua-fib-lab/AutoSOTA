import torch

from data.eval_dataset import UniversalFakeDetectDataset
from data.transforms import MEAN, STD, create_eval_transforms
from engine.evaluator import evaluate_model
from models.pgc import PGCNetwork
from utils.cli import build_test_parser, parse_devices
from utils.logging_utils import log_test_config, setup_logging
from utils.metrics import compute_mean_metrics, log_metrics


def main():
    # ---------------- CLI parsing & global setup ----------------
    parser = build_test_parser()
    args = parser.parse_args()

    devices, device = parse_devices(args.devices)
    args.devices = devices

    logger_inst, log_path = setup_logging(args.checkpoints_dir, args.name, log_type='test')
    logger_inst.info('Logging to file: %s', log_path)
    log_test_config(logger_inst, args)
    logger_inst.info('Using device: %s', device)
    if args.tta:
        logger_inst.info('TTA enabled: mode=%s', args.tta_mode)

    # ---------------- Build dataset + transform ----------------
    transform = create_eval_transforms(
        image_size=args.cropSize, mean=MEAN['dino'], std=STD['dino'],
    )
    logger_inst.info('Loading test dataset: %s', args.test_root)
    dataset = UniversalFakeDetectDataset(args.test_root, transform=transform)

    # ---------------- Build & load model ----------------
    model = PGCNetwork(
        dino_variant=args.dino_variant,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
        pretrained_root=args.dino_pretrained_root,
        tau_rgb=args.tau_rgb,
        tau_res=args.tau_res,
    )
    logger_inst.info('Loading checkpoint: %s', args.checkpoint)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model'])
    model.to(device).eval()

    # ---------------- Evaluation ----------------
    logger_inst.info('Starting evaluation...')
    subset_metrics, _overall_micro = evaluate_model(
        model=model,
        device=device,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_tta=args.tta,
        tta_mode=args.tta_mode,
    )
    mean_metrics = compute_mean_metrics(subset_metrics)
    log_metrics(mean_metrics, 'Overall test set mean')
    logger_inst.info('Evaluation completed. Subsets evaluated: %d', len(subset_metrics))


if __name__ == '__main__':
    main()
