"""
Streaming Prototype OOD Detection -- main entry point.

Usage
-----
# Run with the default ImageNet config:
python main.py --config configs/streaming_ood_imagenet.yaml

# Override data root and batch size on the command line:
python main.py --config configs/streaming_ood_imagenet.yaml \
               --data_root /your/data/path \
               --batch_size 128

# Or use the provided shell script:
bash scripts/run_streaming_ood_imagenet.sh --data_root /your/data/path
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openood.pipelines import get_pipeline
from openood.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Streaming Prototype OOD Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to a YAML config file, e.g. configs/streaming_ood_imagenet.yaml',
    )
    # Common per-run overrides. Any key present in the YAML can be added here.
    parser.add_argument('--data_root',  type=str, default=None,
                        help='Override data_root in the config.')
    parser.add_argument('--backbone',   type=str, default=None,
                        help='Override backbone, e.g. ViT-B/32.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch_size.')
    parser.add_argument('--seed',       type=int, default=None,
                        help='Override random seed.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output_dir in the config.')
    parser.add_argument('--result_name', type=str, default=None,
                        help='Override result_name in the config.')
    parser.add_argument('--allow_mock_data', action='store_true', default=None,
                        help='Allow synthetic mock data if image lists are missing.')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # Apply any command-line overrides on top of the YAML values.
    for key in (
        'data_root',
        'backbone',
        'batch_size',
        'seed',
        'output_dir',
        'result_name',
        'allow_mock_data',
    ):
        val = getattr(args, key)
        if val is not None:
            cfg_dict[key] = val

    cfg = Config(cfg_dict)
    pipeline = get_pipeline(cfg)
    pipeline.run()


if __name__ == '__main__':
    main()
