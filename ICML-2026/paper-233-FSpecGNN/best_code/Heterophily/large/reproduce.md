# Reproduce Best Hyperparameters

## Roman Empire

```bash
python train.py --net Cheb_GAT --dataset Roman_empire --semi_rnd --hidden 512 --K 5 --lr 0.005 --wd 0 --dropout 0.700 --prop_lr 0.02 --prop_wd 0.0001 --dprate 0.300 --heads 4 --alpha_init 0.000
python train.py --net ChebII_GAT --dataset Roman_empire --semi_rnd --hidden 512 --K 5 --lr 0.005 --wd 0.0001 --dropout 0.700 --prop_lr 0.01 --prop_wd 0.0001 --dprate 0.100 --heads 4 --alpha_init 0.000
python train.py --net Bern_GAT --dataset Roman_empire --semi_rnd --hidden 512 --K 5 --lr 0.02 --wd 0.0001 --dropout 0.700 --prop_lr 0.02 --prop_wd 0.0001 --dprate 0.100 --heads 4 --alpha_init -2.000
```

## Minesweeper

```bash
python train.py --net Cheb_GAT --dataset Minesweeper --semi_rnd --hidden 512 --K 5 --lr 0.005 --wd 0 --dropout 0.100 --prop_lr 0.01 --prop_wd 0.0005 --dprate 0.100 --heads 4 --alpha_init -2.000
python train.py --net ChebII_GAT --dataset Minesweeper --semi_rnd --hidden 512 --K 5 --lr 0.005 --wd 0.0005 --dropout 0.100 --prop_lr 0.02 --prop_wd 0.0001 --dprate 0.100 --heads 4 --alpha_init 0.000
python train.py --net Bern_GAT --dataset Minesweeper --semi_rnd --hidden 512 --K 5 --lr 0.005 --wd 0.0001 --dropout 0.300 --prop_lr 0.005 --prop_wd 0 --dprate 0.100 --heads 4 --alpha_init 0.000
```

## Tolokers

```bash
python train.py --net Cheb_GAT --dataset Tolokers --semi_rnd --hidden 512 --K 5 --lr 0.005 --wd 0.0005 --dropout 0.100 --prop_lr 0.02 --prop_wd 0.0005 --dprate 0.100 --heads 4 --alpha_init -6.000
python train.py --net ChebII_GAT --dataset Tolokers --semi_rnd --hidden 512 --K 5 --lr 0.005 --wd 0.0005 --dropout 0.100 --prop_lr 0.02 --prop_wd 0.0001 --dprate 0.100 --heads 4 --alpha_init 0.000
python train.py --net Bern_GAT --dataset Tolokers --semi_rnd --hidden 512 --K 5 --lr 0.005 --wd 0.0001 --dropout 0.100 --prop_lr 0.005 --prop_wd 0.0001 --dprate 0.100 --heads 1 --alpha_init 0.000
```

## Questions

```bash
python train.py --net Cheb_GAT --dataset Questions --semi_rnd --hidden 512 --K 5 --lr 0.005 --wd 0.0005 --dropout 0.700 --prop_lr 0.02 --prop_wd 0.0005 --dprate 0.300 --heads 4 --alpha_init -6.000
python train.py --net ChebII_GAT --dataset Questions --semi_rnd --hidden 512 --K 5 --lr 0.01 --wd 0.0005 --dropout 0.700 --prop_lr 0.01 --prop_wd 0 --dprate 0.100 --heads 2 --alpha_init 0.000
python train.py --net Bern_GAT --dataset Questions --semi_rnd --hidden 512 --K 5 --lr 0.005 --wd 0 --dropout 0.700 --prop_lr 0.01 --prop_wd 0.0005 --dprate 0.100 --heads 1 --alpha_init -2.000
```
