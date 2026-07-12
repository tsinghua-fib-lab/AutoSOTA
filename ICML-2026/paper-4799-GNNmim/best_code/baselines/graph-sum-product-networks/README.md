# Graph-Induced Sum-Product Networks (GSPN)

Official Repository of the [ICLR 2024 paper](https://openreview.net/forum?id=h7nOCxFsPg) _"Tractable Probabilistic Graph Representation Learning with Graph-Induced Sum-Product Networks"_.

![GSPN](gspn.png)

### Citing us

Please consider citing us if you find the code and paper useful:

    @inproceedings{errica_tractable_2024,
      title={Tractable Probabilistic Graph Representation Learning with Graph-Induced Sum-Product Networks},
      author={Errica, Federico and Niepert, Mathias},
      booktitle={The 12th International Conference on Learning Representations (ICLR)},
      year={2024},
    }

## Requirements

An environment with [PyTorch](https://pytorch.org/get-started/locally/) (>=2.0.0), [PytorchGeometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) (>=2.3.0) and [PyDGN](https://github.com/diningphil/PyDGN/tree/main) (==1.5.0) installed.

You can install PyDGN using `pip install pydgn==1.5.0`

# How to reproduce the results

Remove the `--debug` option to run experiments in parallel. Please refer to the
[PyDGN tutorial](https://pydgn.readthedocs.io/en/latest/tutorial.html) for an in-depth explanation.

## Scarce supervision Experiments

### Prepare data (e.g., for benzene)

    pydgn-dataset --config-file DATA_CONFIGS/config_benzene.yml

Run the same command for different data configuration files to create the datasets.

### Launch Exp (e.g., for benzene)

First, build unsupervised embeddings

    pydgn-train  --config-file WEAK_SUP_MODEL_CONFIGS/unsup_model_embedding_generation_categorical.yml --debug

Then, train a classifier on top of them

    pydgn-train  --config-file WEAK_SUP_MODEL_CONFIGS/unsup_model_embedding_regression_mlp_weak_supervision.yml --debug

Modify the configuration files accordingly (`dataset_name` and `data_splits_file` fields) to run experiments on different datasets. Note that
OGBG-molpcba has different configuration files (`unsup_model_embedding_generation_multicategorical.yml` and `unsup_model_embedding_regression_mlp_weak_supervision_ogbg.yml`).

## Graph Classification Experiments

### Prepare data (e.g., for NCI1)

    pydgn-dataset --config-file DATA_CONFIGS/config_NCI1.yml


Run the same command for different data configuration files to create the datasets.

### Launch Exp (e.g., for NCI1)

#### Unsupervised GSPN

First, build unsupervised embeddings

    pydgn-train  --config-file MODEL_CONFIGS/unsup_model_embedding_generation_categorical.yml --debug


Then, train a classifier on top of them

    pydgn-train  --config-file MODEL_CONFIGS/unsup_model_embedding_classification_CHEMICAL.yml --debug

#### Supervised GSPN

    pydgn-train  --config-file MODEL_CONFIGS/sup_model_embedding_classification_CHEMICAL.yml --debug


Modify the configuration files accordingly (`dataset_name` and `data_splits_file` fields) to run experiments on different datasets.

## Missing Data Experiments

### Prepare data (e.g., for benzene)

Run the first part of the `Dataset Creation and Model Analysis` notebook using jupyter to generate the raw dataset.

Then

    pydgn-dataset --config-file DATA_CONFIGS/config_benzene_missing_data.yml

### Launch Exp (e.g., for benzene)

    pydgn-train  --config-file MODEL_CONFIGS/missing_gaussian_molecular.yml --debug

Modify the configuration files accordingly (`dataset_name` and `data_splits_file` fields) to run experiments on different datasets.

## Remarks

Once more, these commands show how to run experiments for GPSN, but not all of them. By easily changing the path of the configuration files, you can run all experiments (please have a look at the folders)
and reproduce the results for all baselines and datasets.
