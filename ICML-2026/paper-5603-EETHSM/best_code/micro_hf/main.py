import json
import argparse
import os
import numpy as np
import random
import torch

from model_utils import get_model
from data_utils import get_train_dataset, get_tokenizer
from train_utils import train, save_model, make_dir, get_ident_name, get_data_ident_name, get_task_dir_name, get_lrs, add_lr
from test_utils import evaluation
from generate import force_args, task_choices

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_number', default=-1, type=int, help="The current run number. Will not save if the run has already been saved")

    # Task parameters
    parser.add_argument('--train_task', choices=task_choices,
                        required=True, help="Task to train the model")
    parser.add_argument('--eval_task', choices=task_choices,
                        required=True, help="tasks to evaluate the model")
    
    parser.add_argument('--num_vocab', default=26, type=int, help="vocabulary size in the strings. maximum is 26.")
    parser.add_argument('--num_numbers', default=5, type=int, help="vocabulary (number) size in the strings. maximum is 9.")
    parser.add_argument('--min_number', default=0, type=int, help="The smallest number token")

    parser.add_argument('--ood_eval', default=False, type=bool, help="If true, perform out-of-distribution evaluation.")
    parser.add_argument('--p', default=0.2, type=float, help="proportion, depends on task")    
    parser.add_argument('--eval_p', default=None, type=float, help="proportion, depends on task")    

    parser.add_argument('--mixed', default=False, type=bool, help="If true, use mixed distribution when generating data.")

    # Model
    parser.add_argument('--nope', default=False, type=bool, help="If true, use the no positional encoding version of the hybrid model")

    parser.add_argument('--model', type=str, choices=['hybrid', 'TF', 'SSM'], default=None, help='The model architecture. Cannot specify layers.')
    parser.add_argument('--num_layers', type=int, default=None, help="Number of layers in the model. Cannot specify layers.")

    parser.add_argument('--layer1', type=str, choices=['TF', 'SSM'], default=None, help='The first layer of the trained model. Cannot specify a model.')
    parser.add_argument('--layer2', type=str, choices=['TF', 'SSM'], default=None, help='The second layer of the trained model. Cannot specify a model.')
    parser.add_argument('--layer3', type=str, choices=['TF', 'SSM'], default=None, help='The (optional) third layer of the trained model. Cannot specify a model.')

    parser.add_argument('--hidden_size', default=8, type=int, help="Hidden size of the models")
    parser.add_argument('--heads', default=1, type=int, help="Number of heads in the transformer models.")
    parser.add_argument('--num_masked_heads', default=1, type=int, help='''Only when model = ''T_hard_alibi''. 
            Number of heads where we apply hard alibi. The remaining heads are set to nope.''')
    parser.add_argument('--state_dim', default=1, type=int, help='''Only when model = ''mamba'' or ''hybrid''. 
            Sets the state dimension of the model.''')

    # Optimization
    parser.add_argument('--lr', default=1e-3, type=float, help="choice of learning rate")
    parser.add_argument('--auto_lr', default=False, type=bool, help="If true, find the best lr with some training")
    parser.add_argument('--force_do_lr', default=False, type=bool, help="If true, learn a new lr")

    parser.add_argument('--epochs', default=4, type=int, help="number of epochs")
    parser.add_argument('--num_examples', default=1000, type=int, help="number of samples for each epoch")
    parser.add_argument('--num_eval_examples', default=100, type=int, help="number of evaluation examples per length")
    parser.add_argument('--window', default=20, type=int, help="width of the sliding window attention")

    parser.add_argument('--train_batch_size', default=8, type=int, help="training batch size")
    parser.add_argument('--eval_batch_size', default=8, type=int, help="evaluation batch size")
    parser.add_argument('--eval_num_batches', default=1, type=int, help='''number of batches to use for evaluation.
            useful to have a mean + std over results.''')
    
    parser.add_argument('--pack_examples', default=False, type=bool, help='If true, fill context with multiple examples, deliniated')
    parser.add_argument('--min_train_length', default=97, type=int, help="minimum length of a training example")
    parser.add_argument('--max_train_length', default=98, type=int, help="maximum length of a training example")
    parser.add_argument('--min_eval_length', default=97, type=int, help="minimum length of an evaluation example")
    parser.add_argument('--max_eval_length', default=98, type=int, help="maximum length of an evaluation example")

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help="number of gradient accumulation steps")

    # Context length
    parser.add_argument('--sequence_length', default=100, type=int, help="context length during training")
    parser.add_argument('--eval_sequence_length', default=100, type=int, help="context length at evaluation time")

    # Saving parameters
    parser.add_argument('--save_model', default=False, type=bool, help="If true, save the model after training")
    parser.add_argument('--save_results', default=False, type=bool, help="If true, save the results after training")
    parser.add_argument('--run_anyways', default=False, type=bool, help="If true, run even if the results have already been saved")
    
    # Visual parameters
    parser.add_argument('--print', default=False, type=bool, help="If true, show helpful print statements")
    parser.add_argument('--progress_bar', default=False, type=bool, help="If true, show the process of each epoch")
    parser.add_argument('--num_log_steps', default=50, type=int, help="number of steps between each log when training")

    parser.add_argument("--test_generate", default=False, type=bool, help="If true, test the synthetic tasks generation")
    parser.add_argument("--curriculum", default=False, type=bool, help="If true, use curriculum learning: progressively longer sequences")
    
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    return parser.parse_args()

args = parse_args()
# Set random seeds for reproducibility (IDEA-009)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


# Check the user is specifying either model or layers
do_layers = args.layer1 and args.layer2
do_model = args.model and args.num_layers
if do_model and (args.layer1 or args.layer2):
    assert False, "Cannot specify both model and layers"
if do_layers and (args.model or args.num_layers):
    assert False, "Cannot specify both model and layers"
if not do_layers and not do_model:
    assert False, "Must specify either model or layers"


# Set the layers based on either the model or the specified layers
if do_model:
    if args.model in ['TF', 'SSM']:
        args.layers = [args.model] * args.num_layers
    elif args.model == 'hybrid':
        args.layers = ['SSM', 'TF'] * (args.num_layers // 2)
        if args.num_layers % 2 == 1:
            args.layers.append('SSM')
else:
    if args.layer3 is not None:
        args.layers = [args.layer1, args.layer2, args.layer3]
    else:
        args.layers = [args.layer1, args.layer2]

# Set the eval dataset to be the same as the train dataset if not specified
if args.eval_p is None:
    args.eval_p = args.p

# Force task specific arguments
force_args(args)

if not args.auto_lr and args.save_results and args.run_number >= 0 and not args.run_anyways:
    result_filename = 'results/' + get_task_dir_name(args) + '/%d.json' % args.run_number
    if os.path.exists(result_filename):
        exit(0)

args.data_name = get_data_ident_name(args)

if args.print:
    print(args)


## Get train dataset & tokenizer
tokenizer = get_tokenizer(args)
train_dataset = get_train_dataset(args, tokenizer) 

batch = next(iter(train_dataset))

if args.print:
    print("v"*100)
    print("EXAMPLE:", batch['input'][0])
    # print("STRUNG:", tokenizer.to_string(batch['input_ids'][0]))
    print("-"*100)
    print("TOKENIZED:", batch['input_ids'][0][batch['mask'][0]==1])
    print("^"*100)

if args.test_generate:
    i = batch['input'][0].index("#0")
    print(batch['input'][0][i])
    print(batch['input'][0][i+1])
    print(batch['output'][0][-1])
    exit(0)


## Find the best LR
if args.auto_lr:
    lrs = get_lrs(args)
    key = get_data_ident_name(args) + "_" + get_ident_name(args)
    if args.print:
        print(key)

    if key not in lrs.keys() or (args.force_do_lr and args.run_number == 0):
        losses = []
        for itr in range(2):
            for lr in np.geomspace(1e-4, 1e-0, num=9):
                model = get_model(args, tokenizer)
                args.lr = lr
                if args.print:
                    print("Testing LR:", lr)
                # _, final_loss = train(args, model, tokenizer, train_dataset)
                _, final_loss = train(args, model, tokenizer, train_dataset, one_epoch=True)
                if args.print:
                    print("Final loss:", final_loss)

                losses.append((final_loss, lr))
        losses.sort()
        best_lr = losses[0][1]
        add_lr(args, best_lr)

        args.lr = best_lr
    else:
        args.lr = lrs[key]

if args.save_results and args.run_number >= 0 and not args.run_anyways:
    result_filename = 'results/' + get_task_dir_name(args) + '/%d.json' % args.run_number
    if os.path.exists(result_filename):
        exit(0)

## Get model
model = get_model(args, tokenizer)

if args.print:
    print()
    print("v"*100)
    print(model)
    print(f"Number of parameters of the model: {count_parameters(model)}")
    print("^"*100)
    print()


## train the model
if args.curriculum:
    # IDEA-001: Curriculum learning
    curriculum_schedule = [(20, 30), (50, 60), (80, 90), (97, 98)]
    if args.print:
        print("Using curriculum learning with schedule:", curriculum_schedule)
    all_accs = []
    final_loss = 0.0
    for phase, (min_len, max_len) in enumerate(curriculum_schedule):
        args.min_train_length = min_len
        args.max_train_length = max_len
        phase_dataset = get_train_dataset(args, tokenizer)
        if args.print:
            print(f"Phase {phase}: training on lengths [{min_len}, {max_len}]")
        phase_accs, phase_loss = train(args, model, tokenizer, phase_dataset, one_epoch=True)
        all_accs.extend(phase_accs)
        final_loss = phase_loss
    accs = all_accs
else:
    accs, final_loss = train(args, model, tokenizer, train_dataset)

## save model
if args.save_model:
    save_model(args, model)

## evaluation of the model
if args.print:
    print("###EVALUATION")

model.eval()

str_acc_mean_list, str_acc_std_list, char_accuracy_list = evaluation(args, model, tokenizer)

if args.print:
    print(args)

    print("DONE")

    print("String")
    print(str_acc_mean_list)
    print("Char")
    print(char_accuracy_list)


if args.save_results and args.run_number >= 0:
    # assert False, "Decide what we want to actually save"
    results = {
        "train_accs": accs,
        "final_acc": char_accuracy_list,
        "final_loss": final_loss,
        "params": count_parameters(model),
        "args": vars(args)
    }

    make_dir(args)

    save_path = 'results/' + get_task_dir_name(args)
    with open(save_path + '/%d.json' % args.run_number, 'w') as f:
        json.dump(results, f)
