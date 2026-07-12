#!/usr/bin/env python
"""Shell entry point for cramming runs. Thin wrapper around progressive_cramming.run.main.

Examples
--------
Full cramming (default trainer)::

    python scripts/run_cramming.py \
        --model_checkpoint HuggingFaceTB/SmolLM2-135M \
        --dataset_name LarryLovestein/pg19_1k \
        --max_sequence_length 64 --limit_dataset_items 4 \
        --max_optimization_steps_per_sample 2000 --learning_rate 0.01

Progressive cramming::

    python scripts/run_cramming.py --progressive_train 1 \
        --model_checkpoint HuggingFaceTB/SmolLM2-135M \
        --dataset_name LarryLovestein/pg19_1k \
        --max_sequence_length 64 --limit_dataset_items 4

Low-dim projection::

    python scripts/run_cramming.py --low_dim_train --low_dim_size 32 \
        --model_checkpoint HuggingFaceTB/SmolLM2-135M \
        --dataset_name LarryLovestein/pg19_1k \
        --max_sequence_length 64 --limit_dataset_items 4
"""

from progressive_cramming.run import main

if __name__ == "__main__":
    main()
