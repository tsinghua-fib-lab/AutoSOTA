# Oldie but Goodie: When Label Propagation meets Feature Propagation on Graphs with Partial Features

## Overall Framework of GOODIE

<img src="https://user-images.githubusercontent.com/68312164/216514092-c993fb3e-85ea-4d89-96d8-45f79eced9d5.png">

### Requirements
- python version: 3.7.11
- numpy version: 1.19.2
- pytorch version: 1.8.0
- torch-geometric version: 2.0.1

### How to run
Following Options can be passed to `main.py`

`--dataset:` Name of the dataset. Cora, CiteSeer, PubMed, wikics, cs, physics, OGBN-Arxiv are available.  
usage example :`--dataset Cora`

`--mask_type:`
Feature missing type. uniform, structural are available.
usage example :`--mask_type structural`

`--missing_rate:`
Feature missing rate. Any value in range 0.0 ~ 1.0 is available (0.0: No features, 1.0: Full features).
usage example :`--missing_rate 0.9999`

`--lp_alpha:`
Relative amount absorbing neighbors' labels.  
usage example :`--lp_alpha 0.99`

`--lamb:`
PseudoCon loss controlling parameter.  
usage example :`--lamb 1.0`

### How to Run

```
python main.py --dataset Cora --mask_type structural --missing_rate 0.9999 --lp_alpha 0.99 --lamb 1.0
```

