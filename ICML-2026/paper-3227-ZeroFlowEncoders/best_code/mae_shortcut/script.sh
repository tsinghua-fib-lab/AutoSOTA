
# # python run_mae.py
# python run_finetune.py --job mae


# python run_zfe.py
# python run_finetune.py --job zfe

python mae_mnist.py --task mnist
python mae_mnist.py --task fmnist
python mae_mnist.py --task cifar10

python mae_mnist_wm.py --task mnist
python mae_mnist_wm.py --task fmnist
python mae_mnist_wm.py --task cifar10