## homophilic datasets

python main.py --gnn gcn --dataset amazon-computer --hidden_channels 512 --epochs 1000 --lr 0.001 --runs 3 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device $1 --ln
python main.py --gnn sage --dataset amazon-computer --hidden_channels 64 --epochs 1000 --lr 0.001 --runs 3 --local_layers 4 --weight_decay 5e-5 --dropout 0.3 --device $1 --ln
python main.py --gnn gat --dataset amazon-computer --hidden_channels 64 --epochs 1000 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-5 --dropout 0.5 --device $1 --ln

python main.py --gnn gcn --dataset amazon-photo --hidden_channels 256 --epochs 1000 --lr 0.001 --runs 3 --local_layers 6 --weight_decay 5e-5 --dropout 0.5 --device $1 --ln --res
python main.py --gnn sage --dataset amazon-photo --hidden_channels 64 --epochs 1000 --lr 0.001 --runs 3 --local_layers 6 --weight_decay 5e-5 --dropout 0.2 --device $1 --ln --res
python main.py --gnn gat --dataset amazon-photo --hidden_channels 64 --epochs 1000 --lr 0.001 --runs 3 --local_layers 3 --weight_decay 5e-5 --dropout 0.5 --device $1 --ln --res

python main.py --gnn gcn --dataset coauthor-cs --hidden_channels 512 --epochs 1500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device $1 --ln --res
python main.py --gnn sage --dataset coauthor-cs --hidden_channels 512 --epochs 1500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.5 --device $1 --ln --res
python main.py --gnn gat --dataset coauthor-cs --hidden_channels 256 --epochs 1500 --lr 0.001 --runs 3 --local_layers 1 --weight_decay 5e-4 --dropout 0.3 --device $1 --ln --res

python main.py --gnn gcn --dataset coauthor-physics --hidden_channels 64 --epochs 1500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.3 --device $1 --ln --res
python main.py --gnn sage --dataset coauthor-physics --hidden_channels 64 --epochs 1500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.7 --device $1 --bn --res
python main.py --gnn gat --dataset coauthor-physics --hidden_channels 256 --epochs 1500 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 5e-4 --dropout 0.7 --device $1 --bn --res

python main.py --gnn gcn --dataset wikics --hidden_channels 256 --epochs 1000 --lr 0.001 --runs 3 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --device $1 --ln
python main.py --gnn sage --dataset wikics --hidden_channels 256 --epochs 1000 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 0.0 --dropout 0.7 --device $1 --ln
python main.py --gnn gat --dataset wikics --hidden_channels 512 --epochs 1000 --lr 0.001 --runs 3 --local_layers 2 --weight_decay 0.0 --dropout 0.7 --device $1 --ln --res

python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3  --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5
python main.py --gnn sage --dataset cora --lr 0.001 --local_layers 3  --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5
python main.py --gnn gat --dataset cora --lr 0.001 --local_layers 3  --hidden_channels 512 --weight_decay 5e-4 --dropout 0.2 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5 --res

python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5
python main.py --gnn sage --dataset citeseer --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 0.01 --dropout 0.2 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5
python main.py --gnn gat --dataset citeseer --lr 0.001 --local_layers 3 --hidden_channels 256 --weight_decay 0.01 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5 --res

python main.py --gnn gcn --dataset pubmed --lr 0.005 --local_layers 2 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5
python main.py --gnn sage --dataset pubmed --lr 0.005 --local_layers 4 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5
python main.py --gnn gat --dataset pubmed --lr 0.01 --local_layers 2 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.5 --rand_split_class --valid_num 500 --test_num 1000 --seed 123 --device $1 --runs 5

## heterophilic datasets

python main.py --gnn gcn --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 4 --weight_decay 0.0 --dropout 0.5 --device $1 --bn --res
python main.py --gnn sage --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 9 --weight_decay 0.0 --dropout 0.5 --device $1 --bn --res
python main.py --gnn gat --dataset amazon-ratings --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 4 --weight_decay 0.0 --dropout 0.5 --device $1 --bn --res

python main.py --gnn gcn --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 3 --local_layers 12 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device $1 --bn --res
python main.py --gnn sage --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 3 --local_layers 15 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device $1 --bn --res
python main.py --gnn gat --dataset minesweeper --hidden_channels 64 --epochs 2000 --lr 0.01 --runs 3 --local_layers 15 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device $1 --bn --res

python main.py --gnn gcn --dataset roman-empire --pre_linear --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 9 --weight_decay 0.0 --dropout 0.5 --device $1 --bn --res
python main.py --gnn sage --dataset roman-empire --pre_linear --hidden_channels 256 --epochs 2500 --lr 0.001 --runs 3 --local_layers 9 --weight_decay 0.0 --dropout 0.3 --device $1 --bn
python main.py --gnn gat --dataset roman-empire --pre_linear --hidden_channels 512 --epochs 2500 --lr 0.001 --runs 3 --local_layers 10 --weight_decay 0.0 --dropout 0.3 --device $1 --bn --res

python main.py --gnn gcn --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 3 --local_layers 10 --weight_decay 0.0 --dropout 0.3 --metric rocauc --device $1 --res
python main.py --gnn sage --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 3 --local_layers 6 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device $1 --ln
python main.py --gnn gat --dataset questions --pre_linear --hidden_channels 512 --epochs 1500 --lr 3e-5 --runs 3 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --metric rocauc --device $1 --ln --res

python main.py --gnn gcn  --dataset squirrel --lr 0.01 --local_layers 4 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --device $1 --runs 10 --bn --res
python main.py --gnn sage  --dataset squirrel --lr 0.01 --local_layers 3 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --device $1 --runs 10 --bn --res
python main.py --gnn gat  --dataset squirrel --lr 0.005 --local_layers 7 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.5 --device $1 --runs 10 --bn --res

python main.py --gnn gcn --dataset chameleon --lr 0.005 --local_layers 5 --hidden_channels 512 --weight_decay 0.001 --dropout 0.2 --device $1 --runs 10 --epochs 200
python main.py --gnn sage --dataset chameleon --lr 0.01 --local_layers 4 --hidden_channels 256 --weight_decay 0.001 --dropout 0.7 --device $1 --runs 10 --epochs 200 --bn --res
python main.py --gnn gat --dataset chameleon --lr 0.01 --local_layers 2 --hidden_channels 256 --weight_decay 0.001 --dropout 0.7 --device $1 --runs 10 --epochs 200 --bn --res
