import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MUTAG', type=str, help='Provide the dataset name',
                    choices=['PROTEINS', 'PROTEINS_full', 'ENZYMES', 'SYNTHETIC', 'BZR', 'COX2', 'NCI1', 'MUTAG',
                             'IMDB-BINARY', 'IMDB-MULTI', 'COLLAB', 'DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K',
                             'REDDIT-MULTI-12K', 'PTC_MR', 'reddit_threads', 'SYNTHETICnew'])
parser.add_argument('--epoch', default=2000, type=int)
parser.add_argument('--device', default='cpu')

args = parser.parse_args()
