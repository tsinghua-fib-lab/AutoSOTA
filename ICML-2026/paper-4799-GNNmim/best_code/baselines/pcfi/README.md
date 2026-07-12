# PCFI (ICLR2023)
This repository is the official PyTorch implementation of "Confidence-Based Feature Imputation for Graphs with Partially Known Features" [[Paper](https://openreview.net/forum?id=YPKBIILy-Kt)]. The codes are built on [Feature Propagation](https://github.com/twitter-research/feature-propagation).

## Requirements
python >= 3.9 <br />
torch == 1.10.2 <br />
pyg == 2.0.3

## To run the code
Semi-supervised node classification
```
python run_node.py
```
Link prediction
```
python run_link.py
```


## Citation
```
@inproceedings{umconfidence,
  title={Confidence-Based Feature Imputation for Graphs with Partially Known Features},
  author={Um, Daeho and Park, Jiwoong and Park, Seulki and young Choi, Jin},
  booktitle={The Eleventh International Conference on Learning Representations}
}
```
