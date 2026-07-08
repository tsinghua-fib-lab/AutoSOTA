import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from meta_trainers.distill_trainer import DistillTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fp:
        config = json.load(fp)

    trainer = DistillTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()

