# Reproduce Best Hyperparameters

## Texas

```bash
python train.py --net Cheb_GAT --dataset Texas --semi_rnd --hidden 64 --K 2 --lr 0.02 --wd 0 --dropout 0.700 --prop_lr 0.01 --prop_wd 0 --dprate 0.100 --heads 1 --alpha_init -2.000
python train.py --net ChebII_GAT --dataset Texas --semi_rnd --hidden 64 --K 2 --lr 0.02 --wd 0 --dropout 0.900 --prop_lr 0.02 --prop_wd 0.0005 --dprate 0.500 --heads 1 --alpha_init 0.000
python train.py --net Bern_GAT --dataset Texas --semi_rnd --hidden 64 --K 2 --lr 0.02 --wd 0.0001 --dropout 0.700 --prop_lr 0.01 --prop_wd 0.0005 --dprate 0.500 --heads 1 --alpha_init -2.000
```

## Wisconsin

```bash
python train.py --net Cheb_GAT --dataset Wisconsin --semi_rnd --hidden 64 --K 2 --lr 0.02 --wd 0.0001 --dropout 0.100 --prop_lr 0.02 --prop_wd 0.0001 --dprate 0.100 --heads 2 --alpha_init -4.000
python train.py --net ChebII_GAT --dataset Wisconsin --semi_rnd --hidden 64 --K 2 --lr 0.02 --wd 0.0001 --dropout 0.500 --prop_lr 0.005 --prop_wd 0.0005 --dprate 0.100 --heads 1 --alpha_init 0.000
python train.py --net Bern_GAT --dataset Wisconsin --semi_rnd --hidden 64 --K 2 --lr 0.02 --wd 0.0001 --dropout 0.300 --prop_lr 0.005 --prop_wd 0.0005 --dprate 0.300 --heads 1 --alpha_init -2.000
```

## Chameleon

```bash
python train.py --net Cheb_GAT --dataset Chameleon --semi_rnd --hidden 64 --K 2 --lr 0.02 --wd 0.0001 --dropout 0.300 --prop_lr 0.02 --prop_wd 0.0001 --dprate 0.900 --heads 2 --alpha_init 0.000
python train.py --net ChebII_GAT --dataset Chameleon --semi_rnd --hidden 64 --K 2 --lr 0.02 --wd 0 --dropout 0.900 --prop_lr 0.005 --prop_wd 0.0005 --dprate 0.300 --heads 1 --alpha_init -4.000
python train.py --net Bern_GAT --dataset Chameleon --semi_rnd --hidden 64 --K 2 --lr 0.02 --wd 0.0001 --dropout 0.900 --prop_lr 0.005 --prop_wd 0 --dprate 0.100 --heads 2 --alpha_init -2.000
```

## Squirrel

```bash
python train.py --net Cheb_GAT --dataset Squirrel --semi_rnd --hidden 64 --K 2 --lr 0.02 --wd 0.0005 --dropout 0.900 --prop_lr 0.01 --prop_wd 0.0001 --dprate 0.900 --heads 1 --alpha_init -2.000
python train.py --net ChebII_GAT --dataset Squirrel --semi_rnd --hidden 64 --K 2 --lr 0.01 --wd 0.0001 --dropout 0.900 --prop_lr 0.02 --prop_wd 0.0005 --dprate 0.100 --heads 4 --alpha_init 0.000
python train.py --net Bern_GAT --dataset Squirrel --semi_rnd --hidden 64 --K 2 --lr 0.005 --wd 0.0005 --dropout 0.900 --prop_lr 0.02 --prop_wd 0.0001 --dprate 0.100 --heads 4 --alpha_init -4.000
```
