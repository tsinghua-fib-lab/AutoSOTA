from train_NAVAR import train_NAVAR
from evaluate import calculate_AUROC, dream_file_to_causal_matrix
import pandas as pd
import argparse


# parse arguments
parser = argparse.ArgumentParser(description='Train NAVAR on your own data')
parser.add_argument('--filename',  type=str, help='name of the file')
parser.add_argument('--maxlags', nargs='?', type=int, default=5, help='Maximum number of lags (K)')
parser.add_argument('--hidden_nodes', nargs='?', type=int, default=10, help='Number of nodes per hidden layer')
parser.add_argument('--hidden_layers', nargs='?', type=int, default=1, help='Number of hidden layers')
parser.add_argument('--epochs', type=int, nargs='?', default=2000, help='Number of training epochs')
parser.add_argument('--batch_size', nargs='?', type=int, default=32, help='Batch size')
parser.add_argument('--sparsity_penalty', nargs='?', type=float, default=0.1, help='Sparsity penalty')
parser.add_argument('--weight_decay', nargs='?', type=float, default=0.001, help='Weight Decay')
parser.add_argument('--dropout', nargs='?', type=float, default=0.5, help='Dropout')
parser.add_argument('--learning_rate', nargs='?', type=float, default=3e-4, help='Learning Rate')
parser.add_argument('--validation_proportion', nargs='?', type=float, default=0, help="Proportion of data used for validation")
parser.add_argument('--lstm', action='store_true')

args = parser.parse_args()
filename = args.filename
maxlags = args.maxlags
hidden_nodes = args.hidden_nodes
hl = args.hidden_layers
epochs = args.epochs
batch_size = args.batch_size
lambda1 = args.sparsity_penalty
weight_decay = args.weight_decay
dropout = args.dropout
learning_rate = args.learning_rate
val_proportion = args.validation_proportion
lstm = args.lstm
check_every = int(epochs/10)


# load the data
data = pd.read_csv(filename, sep=',')
data = data.values

# start training
print(f"Starting training on the data from experiment {filename}, training for {epochs} iterations.")
score_matrix, _, _, _ = train_NAVAR(data, maxlags=maxlags, hidden_nodes=hidden_nodes, dropout=dropout, epochs=epochs,
                                 learning_rate=learning_rate, batch_size=batch_size, lambda1=lambda1,
                                 val_proportion=val_proportion, weight_decay=weight_decay, check_every=check_every, hidden_layers=hl, normalize=True,
                                 split_timeseries=False, lstm=lstm)
# evaluate
print('Done training!')
print(score_matrix)
