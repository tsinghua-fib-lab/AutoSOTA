from evaluator import Evaluator
from rgvi import RGVI
from argparse import ArgumentParser
import os
import torch
import warnings
warnings.filterwarnings('ignore')


parser = ArgumentParser()
parser.add_argument('--root', default='../DB/VI/HQVI', type=str, help='root directory of videos')
parser.add_argument('--res', default='240p', choices=['240p', '480p', '2K'], help='input resolution')
parser.add_argument('--seq', default=None, type=str, help='name of video sequence')
parser.add_argument('--prompt', default=None, type=str, help='text prompt for generative model')
args = parser.parse_args()


if __name__ == '__main__':

    # set device
    torch.cuda.set_device(0)

    # define model
    model = RGVI().eval()

    # testing stage
    with torch.no_grad():
        if args.seq is not None:
            evaluator = Evaluator(args.root, args.res, args.seq)
            evaluator.evaluate(model, args.prompt, os.path.join('outputs', args.root.split('/')[-1]))

        if args.seq is None:
            for seq in sorted(os.listdir(os.path.join(args.root, 'JPEGImages', args.res))):
                evaluator = Evaluator(args.root, args.res, seq)
                evaluator.evaluate(model, args.prompt, os.path.join('outputs', args.root.split('/')[-1]))
