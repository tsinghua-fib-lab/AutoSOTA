import argparse
from train_utils import train, choose_lr
from generate import set_task_specific_parameters
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument('--task_name', required=True, help="Task to train the model")
    parser.add_argument('--run_number', required=True, type=int, help="Which run we currently are on")
    
    parser.add_argument('--sequence_len', default=100, type=int, help="Length of the sequences to train on")
    parser.add_argument('--batch_size', default=64, type=int, help="Size of the training and testing batches")
    parser.add_argument('--batches_per_epoch', default=1000, type=int, help="Number of training batches seen per epoch")

    parser.add_argument("--ood_eval", default=False, type=bool, help="If true, perform OOD evaluation after each epoch")
    parser.add_argument("--eval_batches_per_epoch", default=100, type=int, help="Number of evaluation batches per epoch")

    parser.add_argument('--num_vocab', default=30, help="Number of vocab. Can be modified based on task")
    parser.add_argument('--num_numbers', default=5, help="Number of number tokens. Can be modified based on task")
    parser.add_argument('--p', default=0.1, type=float, help="Probability, based on the task")
    parser.add_argument('--num_bits', default=4, type=float, help="Number of bits, based on the task")
        
    parser.add_argument('--eval_num_vocab', default=30, help="Number of vocab. Can be modified based on task. Ignored if ood_data is false")
    parser.add_argument('--eval_num_numbers', default=5, help="Number of number tokens. Can be modified based on task. Ignored if ood_data is false")
    parser.add_argument('--eval_p', default=0.1, type=float, help="Probability, based on the task. Ignored if ood_data is false")
    parser.add_argument('--eval_num_bits', default=4, type=float, help="Number of bits, based on the task. Ignored if ood_data is false")

    # Model parameters
    parser.add_argument("--num_epochs", default=10, type=int, help="Number of epochs to train for")
    
    parser.add_argument('--layer1', choices=['SSM', 'TF'], default='SSM', help="First layer of the 2-layer model")
    parser.add_argument('--layer2', choices=['SSM', 'TF'], default='TF', help="Second layer of the 2-layer model")
    
    parser.add_argument('--embed_dim', default=12, type=int, help="Embedding dimension of the tokens")
    parser.add_argument('--num_heads', default=1, type=int, help="Number of heads for the transformer layers")
    parser.add_argument('--state_dim', default=1, type=int, help="State dimension fo the mamba layers")
    parser.add_argument('--window', default=20, type=int, help="Width of the windowing for the transformer attention")

    parser.add_argument('--lr', default=0, type=float, help="Model learning rate. Defaults to hyperparameter tuning")
    parser.add_argument('--lr_epochs', default=3, type=int, help="The number of epochs to train the model to determine the lr")
    parser.add_argument('--lr_low', default=1e-3, type=float, help="Smallest lr tried. Unused if lr defined")
    parser.add_argument('--lr_high', default=1e-0, type=float, help="Largest lr tried. Unused if lr defined")
    parser.add_argument('--lr_num', default=13, type=int, help="Number of lrs tried. Unused if lr defined")

    parser.add_argument('--pytorch_transformer', default=False, type=bool, help="If true, use a pytorch TF. Otherwise, use a handwritten one")
    parser.add_argument('--positional_encoding', default='learned', choices=["learned", "sine", "none"], help="If true, use a pytorch TF. Otherwise, use a handwritten one")

    parser.add_argument("--save", default=False, type=bool, help="If true, will save the resuls data")
    parser.add_argument("--save_path", default="", help="Path to save the model. No path provided does not save the model")

    parser.add_argument('--lr_only', default=False, type=bool, help="If true, will not train a model, just determine the best learning rate")
    parser.add_argument("--train_only", default=False, type=bool, help="If true, will train the model and not save it")
    parser.add_argument("--lr_no_save", default=False, type=bool, help="If true, find a new lr but also not save it")
    parser.add_argument("--force_learn", default=False, type=bool, help="If true, overwrite the existing results when saving")

    parser.add_argument("--expand", type=int, help="The expand parameter for the SSM")
    
    return parser.parse_args()

args = parse_args()
args.layers = [args.layer1, args.layer2]
args.run_name = "run%d.pt" % args.run_number
set_task_specific_parameters(args)

if args.lr == 0:
    args.lr = choose_lr(args)

if (not args.lr_only) or args.train_only:
    model = train(args)

if len(args.save_path) > 0:
    torch.save(model, "saved_models/" + args.save_path)