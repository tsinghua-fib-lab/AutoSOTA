import subprocess
import sys
import argparse

def run_experiment(dataset, drop_percent, lr,downlr,epochs,shot_num,gpu, skip_pretrain):
    cmd = [
        sys.executable,
        'MDGMIX.py',
        '--dataset', str(dataset),
        '--drop_percent', str(drop_percent),
        '--lr',str(lr),
        '--downstreamlr',str(downlr),
        '--epochs',str(epochs),
        '--shot_num',str(shot_num),
        '--gpu', str(gpu),
        '--skip_pretrain', str(skip_pretrain)
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment with specific parameters.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name.')
    parser.add_argument('--drop_percent', type=float, default=0.5, help='Drop percentage.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use.')
    parser.add_argument('--lr', type=float, default=0.002, help='pretrain lr')
    parser.add_argument('--downstreamlr', type=float, default=0.03, help='downstream lr')
    parser.add_argument('--epochs', type=int, default=60, help='epoch')
    parser.add_argument('--shot_num', type=int, default=1, help='shotnum')
    parser.add_argument('--skip_pretrain', type=int, default=0, help='try to use trained models')

    args = parser.parse_args()

    run_experiment(args.dataset, args.drop_percent, args.lr, args.downstreamlr, args.epochs, args.shot_num, args.gpu, args.skip_pretrain)