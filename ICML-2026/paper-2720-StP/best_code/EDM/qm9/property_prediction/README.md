# E(n) Equivariant Graph Neural Networks

Official implementation (Pytorch 1.7.1) of:  

**E(n) Equivariant Graph Neural Networks**  
Victor Garcia Satorras, Emiel Hogeboom, Max Welling  
https://arxiv.org/abs/2102.09844

<img src="models/egnn.png" width="400">




**Abstract**: This paper introduces a new model to learn graph neural networks equivariant to rotations, translations, reflections and permutations called E(n)-Equivariant Graph Neural Networks (EGNNs). In contrast with existing methods, our work does not require computationally expensive higher-order representations in intermediate layers while it still achieves competitive or better performance. In addition, whereas existing methods are limited to equivariance on 3 dimensional spaces, our model is easily scaled to higher-dimensional spaces. We demonstrate the effectiveness of our method on dynamical systems modelling, representation learning in graph autoencoders and predicting molecular properties.


### Example code
For a simple example of a EGNN implementation [click here](https://github.com/vgsatorras/egnn/blob/3c079e7267dad0aa6443813ac1a12425c3717558/models/egnn_clean/egnn_clean.py#L106). Or copy the file `models/egnn_clean/egnn_clean.py` into your working directory and run:

```python
import egnn_clean as eg
import torch

# Dummy parameters
batch_size = 8
n_nodes = 4
n_feat = 1
x_dim = 3

# Dummy variables h, x and fully connected edges
h = torch.ones(batch_size * n_nodes, n_feat)
x = torch.ones(batch_size * n_nodes, x_dim)
edges, edge_attr = eg.get_edges_batch(n_nodes, batch_size)

# Initialize EGNN
egnn = eg.EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)

# Run EGNN
h, x = egnn(h, x, edges, edge_attr)
```

If you are using the EGNN in a new application we recommend checking the EGNN [attributes description](https://github.com/vgsatorras/egnn/blob/3c079e7267dad0aa6443813ac1a12425c3717558/models/egnn_clean/egnn_clean.py#L119) that contains some upgrades not included in the paper.


### N-body system experiment

#### Create N-body dataset
```
cd n_body_system/dataset
python -u generate_dataset.py --num-train 10000 --seed 43 --sufix small
```

#### Run experiments

*EGNN model*  
```
python -u main_nbody.py --exp_name exp_1_egnn_vel --model egnn_vel --max_training_samples 3000 --lr 5e-4
```
  

*GNN model*  
```
python -u main_nbody.py --exp_name exp_1_gnn --model gnn --max_training_samples 3000 --lr 1e-3
```
  
  
*Radial Field*  
```
python -u main_nbody.py --exp_name exp_1_gnn --model rf_vel --n_layers 4 --max_training_samples 3000 --lr 2e-4 
```
  
  
*Tensor Field Networks*
```
python -u main_nbody.py --exp_name exp_1_tfn --model tfn --max_training_samples 3000 --lr 1e-3 --degree 2 --nf 32
``` 
  
  
*SE3 Transformer*  
```
python -u main_nbody.py --exp_name exp_1_se3 --model se3_transformer --max_training_samples 3000 --div 1 --degree 3 --nf 64 --lr 5e-3
```    

#### N-body system sweep experiment
For the experiment where we sweep over different amounts of training samples you should create a larger training dataset
```
cd n_body_system/dataset
python -u generate_dataset.py  --num-train 50000 --sample-freq 500 
```
Then you can train on in this new partition by adding `--dataset nbody` to the above training commands. You can choose the number of training samples by modifying the argument `--max_training_samples <number of training samples>` 

E.g. for the EGNN for 10.000 samples
```
python -u main_nbody.py --exp_name exp_debug --model egnn_vel --max_training_samples 10000 --lr 5e-4 --dataset nbody
```

 




### Graph Autoencoder experiment

*GNN Erdos & Renyi*  
```  
python -u main_ae.py --exp_name exp1_gnn_erdosrenyi --model ae --dataset erdosrenyinodes_0.25_none --K 8 --emb_nf 8 --noise_dim 0
```  

*GNN Community*  
```  
python -u main_ae.py --exp_name exp1_gnn_community --model ae --dataset community_ours --K 8 --emb_nf 8 --noise_dim 0
```  

*Noise-GNN Erdos&Renyi*  
```  
python -u main_ae.py --exp_name exp1_gnn_noise_erdosrenyi --model ae --dataset erdosrenyinodes_0.25_none --K 8 --emb_nf 8 --noise_dim 1
```  
  
*Noise GNN Community*  
``` 
python -u main_ae.py --exp_name exp1_noise_gnn_community --model ae --dataset community_ours --K 8 --emb_nf 8 --noise_dim 1
``` 
  
*Radial Field Erdos&Renyi*  
``` 
python -u main_ae.py --exp_name exp1_rf_erdosrenyi --model ae_rf --dataset erdosrenyinodes_0.25_none --K 8 --emb_nf 8
``` 

*Radial Field Community*  
``` 
python -u main_ae.py --exp_name exp1_rf_community --model ae_rf --dataset community_ours --K 8 --emb_nf 8
``` 
  
*EGNN Erdos&Renyi*  
```
python -u main_ae.py --exp_name exp1_egnn_erdosrenyi --model ae_egnn --dataset erdosrenyinodes_0.25_none --K 8 --emb_nf 8
```

*EGNN Community*  
```
python -u main_ae.py --exp_name exp1_egnn_community --model ae_egnn --dataset community_ours --K 8 --emb_nf 8
```

----------------------------------------------------
The following overfit eperiments are for (p=0.2). p can be modified by replacing the 0.2 value from the dataset name (e.g. erdosrenyinodes_0.2_overfit) to other values.  

*GNN Erdos&Renyi overfit*
```
python -u main_ae.py --model ae --dataset erdosrenyinodes_0.2_overfit --epochs 10001 --test_interval 200 --K 16 --emb_nf 16 2>&1 | tee outputs_ae/$EXP.log &
```
*Noise-GNN Erdos&Renyi overfit*  
```
python -u main_ae.py --model ae --dataset erdosrenyinodes_0.2_overfit --epochs 10001 --test_interval 200 --noise_dim 1 --K 16 --emb_nf 16 2>&1 | tee outputs_ae/$EXP.log &
```
*EGNN Erdos&Renyi overfit*
```
python -u main_ae.py --model ae_egnn --dataset erdosrenyinodes_0.2_overfit --epochs 10001 --test_interval 200 --K 16 --emb_nf 16 2>&1 | tee outputs_ae/$EXP.log &
```


### QM9 experiment
properties --> [alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve]  
learning rate --> 1e-3 for [gap, homo lumo], 5r-4 for the rest
```
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property alpha --exp_name exp_1_alpha
python -u main_qm9.py --num_workers 2 --lr 1e-3 --property gap --exp_name exp_1_gap
python -u main_qm9.py --num_workers 2 --lr 1e-3 --property homo --exp_name exp_1_homo
python -u main_qm9.py --num_workers 2 --lr 1e-3 --property lumo --exp_name exp_1_lumo
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property mu --exp_name exp_1_mu
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property Cv --exp_name exp_1_Cv
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property G --exp_name exp_1_G
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property H --exp_name exp_1_H
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property r2 --exp_name exp_1_r2
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property U --exp_name exp_1_U
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property U0 --exp_name exp_1_U0
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property zpve --exp_name exp_1_zpve
```

#### Acknowledgements
The Robert Bosch GmbH is acknowledged for financial support.

