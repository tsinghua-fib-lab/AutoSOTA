from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
from tqdm import tqdm
from pathlib import Path
from torch.optim import AdamW
import torch
import os
import json

from test_utils import evaluation

from accelerate import Accelerator


def get_lrs(args):
    if not os.path.exists("results"):
        return {}
    if not os.path.exists("results/" + args.train_task):
        return {}
    if not os.path.exists("results/" + args.train_task + "/lrs.json"):
        return {}
    with open("results/" + args.train_task + "/lrs.json")  as f:
        lrs = json.load(f)
    return lrs

def add_lr(args, lr):
    lrs = get_lrs(args)
    ident_name = get_data_ident_name(args) + "_" + get_ident_name(args)
    lrs[ident_name] = lr
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("results/" + args.train_task):
        os.mkdir("results/" + args.train_task)
    with open("results/" + args.train_task + "/lrs.json", "w")  as f:
        json.dump(lrs, f)


def ce_loss(inputs, logits, mask):
    # Shift so that tokens < n predict n
    if type(logits) != torch.Tensor:
        logits = logits['logits']
    shift_labels = inputs.contiguous()
    shift_logits = logits.contiguous()
    # mask = mask.contiguous().view(-1)

    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(*shift_labels.shape)
    return torch.sum(loss*mask)/torch.sum(mask)



def custom_get_scheduler(optimizer, num_training_steps):
    # IDEA-012: Cosine schedule with 20% warmup ratio (better for Mamba)
    num_warmup_steps = 100
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler


def train_epoch(model, epoch, accelerator, optimizer, lr_scheduler, attention_mask, train_dataset, args, device="cuda"):
    avg_loss = 0
    count = 0

    if args.progress_bar:
        # progress_bar = tqdm(
        #     enumerate(train_dataset, start=1), total=args.epochs * args.num_examples,
        #     desc=f'Epoch {epoch + 1}/{args.epochs}'
        # )
        pass
    else:
        progress_bar = enumerate(train_dataset, start=1)
        progress_bar = enumerate(train_dataset)
    
    for step, batch in progress_bar:
        x = batch['input_ids'].to(device)
        y = batch['output_ids'].to(device)
        loss_mask = batch['mask'].to(device)

        logits = model(x, attention_mask=attention_mask, return_dict=True)['logits']

        loss = ce_loss(y, logits, loss_mask)
        # if (step+1) % args.num_log_steps == 0:
        #     avg_loss.append(0)
        #     count.append(0)
        loss = loss / args.gradient_accumulation_steps
        
        avg_loss += loss.item()
        count += 1
        accelerator.backward(loss)
        
        if step % args.gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # completed_steps += 1

        # This should never happen, but just in case
        if step > args.epochs * args.num_examples:
        # if step > args.num_examples:
            assert False, "This should never happen"
            # break
            
        if args.progress_bar:
            # Update tqdm description with the current loss
            progress_bar.set_postfix({'Loss': loss.item()})

    return avg_loss / count


def train(args, model, tokenizer, train_dataset, one_epoch=False):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # Put model on GPU
    accelerator = Accelerator()
    model, optimizer = accelerator.prepare(model, optimizer)

    lr_scheduler = custom_get_scheduler(optimizer, args.epochs * args.num_examples // args.gradient_accumulation_steps)

    attention_mask = torch.ones((args.sequence_length, args.sequence_length))
    attention_mask = (torch.triu(attention_mask, diagonal=0) - torch.triu(attention_mask, diagonal=args.window)).T.to('cuda' if torch.cuda.is_available() else 'cpu')

    losses = []
    accs = []
    for epoch in range(args.epochs):
        model.train()
        avg_loss = train_epoch(model, epoch, accelerator, optimizer, lr_scheduler, attention_mask, train_dataset, args, device="cuda")
        losses.append(avg_loss)
        print(epoch, optimizer.param_groups[0]["lr"], avg_loss)

        model.eval()
        _, _, char_accuracy_list = evaluation(args, model, tokenizer, do_print=False)
        accs.append(char_accuracy_list[0])

        if one_epoch:
            break

    # return losses
    return accs, losses[-1]


##################################################################################################################################

# Saving helpers

def get_data_ident_name(args):
    return "data_%d_%d_%d" % (args.sequence_length, args.num_numbers, args.num_vocab)

def get_ident_name(args):
    dashed_task_name = "-".join(args.train_task.split("_"))
    # Depth tests
    if args.num_layers is not None:
        return "run_%s_%s-%d_w%d_d%d_nh%d_sd%d" % (dashed_task_name, args.model, args.num_layers, args.window, args.hidden_size, \
                                            args.heads, args.state_dim)
    elif args.layer3 is not None:
        return "run_%s_%s-%s-%s_w%d_d%d_nh%d_sd%d" % (dashed_task_name, args.layer1, args.layer2, args.layer3, args.window, args.hidden_size, \
                                           args.heads, args.state_dim)
    else:
        if args.mixed:
            return "run-mixed_%s_%s-%s_w%d_d%d_nh%d_sd%d" % (dashed_task_name, args.layer1, args.layer2, args.window, args.hidden_size, \
                                                        args.heads, args.state_dim)
        else:
            return "run_%s_%s-%s_w%d_d%d_nh%d_sd%d" % (dashed_task_name, args.layer1, args.layer2, args.window, args.hidden_size, \
                                            args.heads, args.state_dim)

def get_task_dir_name(args):
    return args.train_task + "/" + args.data_name + "/" + get_ident_name(args)

def make_dir(args, saving='results'):
    if saving == 'results':
        if args.ood_eval:
            if 'results_ood' not in os.listdir('.'):
                os.mkdir('results_ood')
            base_path = 'results_ood'
        else:
            if 'results' not in os.listdir('.'):
                os.mkdir('results')
            base_path = 'results'

    if saving == 'model_results':
        if 'model_results' not in os.listdir('.'):
            os.mkdir('model_results')
        base_path = 'model_results'
    
    if args.train_task not in os.listdir(base_path + ''):
        os.mkdir(base_path + '/' + args.train_task)

    if args.data_name not in os.listdir(base_path + '/' + args.train_task):
        os.mkdir(base_path + '/' + args.train_task + "/" + args.data_name)
    
    if get_ident_name(args) not in os.listdir(base_path + '/' + args.train_task + "/" + args.data_name):
        os.mkdir(base_path + '/' + args.train_task + "/" + args.data_name + "/" + get_ident_name(args))


def load_model(args, model):
    path = 'model_results/' + get_task_dir_name(args)

    # Load model
    model = model.from_pretrained(path)

    return model
    

def save_model(args, model):
    path = 'model_results/' + get_task_dir_name(args)

    if not os.path.exists(path):
        make_dir(args, saving='model_results')

    # Save model
    model.save_pretrained(path)
