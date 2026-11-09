Hello!

Below you can find a outline of how to reproduce our solution for the CHAMPS competition.
If you run into any trouble with the setup/code or have any questions please contact us at Zico.Kolter@us.bosch.com 

Copyright 2019 Robert Bosch GmbH

Code authors: Zico Kolter, Shaojie Bai, Devin Wilmott, Mordechai Kornbluth, Jonathan Mailoa, part of Bosch Research (CR).

## Archive Contents
  - `config/`             : Configuration files
  - `data/`               : Raw data
  - `models/`             : Saved models
  - `processed/`          : Processed data
  - `src/`                : Source code for preprocessing, training, and predicting.
  - `submission/`         : Directory for the actual predictions

## Hardware (The following specs were used to create the original solution)

The variety of models were trained on different machines, each running a Linux OS:
  - 5 machines had 4 GPUs, each a NVIDIA GeForce RTX 2080 Ti
  - 2 machines had 1 GPU NVIDIA Tesla V100 with 32 GB memory
  - 6 machines had 1 GPU NVIDIA Tesla V100 with 16 GB memory

## Software
  - Python 3.5+
  - CUDA 10.1
  - [NVIDIA APEX](https://github.com/NVIDIA/apex) (Only available through the repo at this phase)

Python packages are detailed separately in `requirements.txt`.

Note: Though listed in `requirements.txt`, `rdkit` is not available with `pip`. We strongly suggest installing `rdkit` via conda:
```sh
conda install -c rdkit rdkit
```

## Data Setup

We use only the `train.csv`, `test.csv`, and `structures.csv` files of the competition. They should be (unzipped and) placed in the `data/` directory. All of the commands below are executed from the `src/` directory.

## Data Processing

0. `cd src/`
1. `python pipeline_pre.py 1` (This could take 1-2 hours)
2. `python pipeline_pre.py 2`

(You may need to change the permission to the `.csv` files to use the two scripts above via `chmod`.)

## Model Build - There are three options to produce the solution.

While in `src/`:
1. Very fast prediction: `predictor.py fast` to use the precomputed results for ensembling.
2. Ordinary prediction: `predictor.py` to use the precomputed checkpoints for predicting and ensembling.
3. Re-train models: `train.py` to train a new model from scratch. See `train.py -h` for allowed arguments, and `config` files for each model for the arguments used.

The `config/models.json` file contains the following important keys:

- names: List of the names we will ensemble
- output file: The name of the ensembled output file
- num atom types, bond types, triplet types, quad types: These are arguments to pass to the GraphTransformer instantiator. Note that in the default setting, quadruplet information is not used by GTs.
- `model_dir`: The directory in `models/` associated with each model. Each directory must have 
    1) `graph_transformer.py` with a `GraphTransformer` class (and any modules it needs); 
    2) `config` file with the kwargs to instantiate the `GraphTransformer` class; 
    3) `[MODEL_NAME].ckpt` that can be loaded via `load_state_dict(torch.load('[MODEL_NAME].ckpt').state_dict())` (to avoid PyTorch version conflict).

## Notes on (Pre-trained) Model Loading

All pretrained models are stored in `models/`. However, different models may have slightly different architecture (e.g., some GT models are followed by a 2-layer grouped residual network, while some others only have one residual block). The training script (`train.py`), when initiated without the `--debug` flag, will automatically create a log folder in `CHAMPS-GT/` that contains the code for the GT used. When loading the model, use the `graph_transformer.py` in that log folder (instead of the default one in `src/`).

## Notes on Model Training

When trained from scratch, the default parameters should lead to a model achieving a score of around -3.06 to -3.07. Using `--debug` flag will prevent the program from creating a log folder.

## Notes on Saving Memory

What if you got a `CUDA out of memory` error? We suggest a few solutions:
  - If you have a multi-GPU machine, use the `--multi_gpu` flag, and tune the `--gpu0_bsz` flag (which controls the minibatch size passed to GPU device 0). For instance, on a 4-GPU machine, you can do `python train.py [...] --batch_size 47 --multi_gpu --gpu0_bsz 11`, which assigns a batch size of 12 to GPU `1,2,3` and a batch size of 11 to GPU `0`.
  - Use the `--fp16` option, which applies NVIDIA APEX's mixed precision training.
  - Use the `--batch_chunk` option, which chunks a larger batch into a few smaller (equal) shares. The gradients from the smaller minibatches will accumulate, so the effective batch size is still the same as `--batch_size`.
  - Use fewer `--n_layer`, or smaller `--batch_size` :P

