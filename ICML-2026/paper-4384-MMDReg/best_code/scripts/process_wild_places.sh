#!/bin/bash

mkdir -p datasets/processed

# Process sequences K-03, K-04, V-03, and V-04.

uv run python -u preprocessing/process_wild_places.py --save_path datasets/processed/wild_places_k_03.hdf5 --sequence K-03
uv run python -u preprocessing/process_wild_places.py --save_path datasets/processed/wild_places_k_04.hdf5 --sequence K-04
uv run python -u preprocessing/process_wild_places.py --save_path datasets/processed/wild_places_v_03.hdf5 --sequence V-03
uv run python -u preprocessing/process_wild_places.py --save_path datasets/processed/wild_places_v_04.hdf5 --sequence V-04
