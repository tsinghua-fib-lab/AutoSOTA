import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import json
import os
import numpy as np
import math

from generate import generate_data
from models.hybrid import HybridModel
from models.transformer import generate_mask



def get_ident_name(args):
    dashed_task_name = "-".join(args.task_name.split("_"))
    return "run_%s_%s_%s_w%d_d%d_nh%d_sd%d_nn%d_nv%d" % (dashed_task_name, args.layer1, args.layer2, args.window, args.embed_dim, \
                                           args.num_heads, args.state_dim, args.num_numbers, args.num_vocab)

def get_task_dir_name(args):
    return args.task_name + "/" + args.data_name + "/" + get_ident_name(args)

def make_dir(args):
    if args.ood_eval:
        if 'results_ood' not in os.listdir('.'):
            os.mkdir('results_ood')
        base_path = 'results_ood'
    else:
        if 'results' not in os.listdir('.'):
            os.mkdir('results')
        base_path = 'results'
    
    if args.task_name not in os.listdir(base_path + ''):
        os.mkdir(base_path + '/' + args.task_name)

    if args.data_name not in os.listdir(base_path + '/' + args.task_name):
        os.mkdir(base_path + '/' + args.task_name + "/" + args.data_name)
    
    if get_ident_name(args) not in os.listdir(base_path + '/' + args.task_name + "/" + args.data_name):
        os.mkdir(base_path + '/' + args.task_name + "/" + args.data_name + "/" + get_ident_name(args))



class CustomDataset(Dataset):
    def __init__(self, data_in, data_out):
        self.data_in = data_in
        self.data_out = data_out

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, idx):
        sample_in = self.data_in[idx]
        sample_out = self.data_out[idx]

        return sample_in, sample_out
    


def get_scheduler(args, optimizer):
    total_steps = args.batches_per_epoch * args.num_epochs
    warmup_steps = args.batches_per_epoch // 10 # Make the warmup be 10% of an epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



def train_epoch(model, optimizer, lr_scheduler, criterion, mask, train_loader, args, device="cuda"):
    model.train()

    avg_loss = 0
    avg_acc = 0
    
    # for itr in range(args.batches_per_epoch):
    #     print("Batch:", itr)
    #     input_seqs, target_seqs = generate_data(args)
    
    for itr, (input_seqs, target_seqs) in enumerate(train_loader):
        
        input_seqs = input_seqs.to(device)
        target_seqs = target_seqs.type(torch.LongTensor).to(device)

        outputs = model(input_seqs, mask)  # (batch, seq-1, vocab)

        # Masked loss, ignore positions which have null tokens (== vocab_size-1)
        loss_mask = (target_seqs != args.vocab_size-1)
        loss = loss_mask.reshape(-1) * criterion(outputs.view(-1, args.vocab_size), target_seqs.reshape(-1))
        loss = loss.sum() / loss_mask.sum()

        acc = torch.sum(loss_mask & ((torch.argmax(outputs, dim=-1) - target_seqs) == 0)).item()
        acc /= loss_mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        avg_loss += loss.item()
        avg_acc += acc

    avg_loss /= args.batches_per_epoch
    avg_acc /= args.batches_per_epoch
    
    return avg_loss, avg_acc


def eval_epoch(model, criterion, mask, eval_loader, args, device="cuda"):
    model.eval()

    avg_loss = 0
    avg_acc = 0
    
    for itr, (input_seqs, target_seqs) in enumerate(eval_loader):
        
        input_seqs = input_seqs.to(device)
        target_seqs = target_seqs.type(torch.LongTensor).to(device)

        outputs = model(input_seqs, mask)  # (batch, seq-1, vocab)

        # Masked loss, ignore positions which have null tokens (== vocab_size-1)
        loss_mask = (target_seqs != args.vocab_size-1)
        loss = loss_mask.reshape(-1) * criterion(outputs.view(-1, args.vocab_size), target_seqs.reshape(-1))
        loss = loss.sum() / loss_mask.sum()

        acc = torch.sum(loss_mask & ((torch.argmax(outputs, dim=-1) - target_seqs) == 0)).item()
        acc /= loss_mask.sum()

        avg_loss += loss.item()
        avg_acc += acc

    avg_loss /= args.batches_per_epoch
    avg_acc /= args.batches_per_epoch
    
    return avg_loss, avg_acc



def choose_lr(args):
    # If found already, use what is cached
    ident_name = get_ident_name(args)
    with open('models/lrs.json') as f:
        d = json.load(f)
    if (not args.lr_no_save) and ident_name in d.keys():
        return d[ident_name]

    train_dataset = CustomDataset(*generate_data(args, all_at_once=True))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    assert args.window <= args.sequence_len
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Starting to find lr for results/" + args.data_name + "/" + args.run_name)

    lrs = np.geomspace(args.lr_low, args.lr_high, args.lr_num)
    criterion = nn.CrossEntropyLoss(reduction='none')
    all_losses = np.zeros_like(lrs)

    for itr, lr in enumerate(lrs):
        curr_losses = []
        print("- Current learning rate: %.4f" % lr)
        for run in range(3):
            model = HybridModel(args).to(device)
        
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            lr_scheduler = get_scheduler(args, optimizer)
        
            mask = generate_mask(args.sequence_len, window=args.window).to(device)
        
            # Training loop
            for epoch in range(args.lr_epochs):
                loss, acc = train_epoch(model, optimizer, lr_scheduler, criterion, mask, train_loader, args, device)
        
            print(f"- - Loss: {loss:.4f}, Acc: {acc:.4f}")
            
            curr_losses.append(loss)
            
        all_losses[itr] = np.median(np.array(curr_losses))

    all_losses = np.nan_to_num(all_losses, nan=100) # Nan needs to be a big number
    lr = lrs[np.argmin(all_losses)]
    print("Loss array:", all_losses)
    print("Determined a lr of", lr)

    if not args.lr_no_save:
        # Saving the lr
        with open('models/lrs.json') as f:
            d = json.load(f)
        d[ident_name] = lr
        with open('models/lrs.json', 'w') as f:
            json.dump(d, f, indent=4)

    return lr


    
def train(args):
    assert args.window <= args.sequence_len

    if args.save:
        dir_name = get_task_dir_name(args)
        if not args.force_learn:
            if args.ood_eval:
                if os.path.isdir("results_ood/" + dir_name) and "run%d.pt" % args.run_number in os.listdir("results_ood/" + dir_name):
                    return
            else:
                if os.path.isdir("results/" + dir_name) and "run%d.pt" % args.run_number in os.listdir("results/" + dir_name):
                    return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CustomDataset(*generate_data(args, all_at_once=True))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    print("Starting work for results/" + args.data_name + "/" + args.run_name)

    model = HybridModel(args).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    lr_scheduler = get_scheduler(args, optimizer)

    mask = generate_mask(args.sequence_len, window=args.window).to(device)

    # Training loop
    losses = []
    accs = []
        
    for epoch in range(args.num_epochs):
        loss, acc = train_epoch(model, optimizer, lr_scheduler, criterion, mask, train_loader, args, device)

        losses.append(loss)
        accs.append(acc)

    print(f"Loss: {losses[-1]:.4f}, Acc: {accs[-1]:.4f}")
    
    losses = torch.Tensor(losses)
    accs = torch.Tensor(accs)

    # Evaluation
    if args.ood_eval:
        args.num_vocab = args.eval_num_vocab
        args.num_numbers = args.eval_num_numbers
        args.p = args.eval_p
        args.num_bits = args.eval_num_bits

    eval_dataset = CustomDataset(*generate_data(args, all_at_once=True))
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    
    eval_loss, eval_acc = eval_epoch(model, criterion, mask, eval_loader, args, device)
    print(f"Eval - Loss: {eval_loss:.4f}, Acc: {eval_acc:.4f}")
    
    # Save the model
    if args.save and not args.train_only:
        make_dir(args)
        dir_name = get_task_dir_name(args)
        if args.ood_eval:
            run_filename = "results_ood/" + dir_name + "/run%d.pt" % args.run_number
        else:
            run_filename = "results/" + dir_name + "/run%d.pt" % args.run_number
        
        # Count of the parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        torch.save({"args": args, "losses": losses, "accs": accs, "param_count": params, "eval_loss": eval_loss, "eval_acc": eval_acc}, run_filename)
        print("Saved and finished for %s" % run_filename)

    return model