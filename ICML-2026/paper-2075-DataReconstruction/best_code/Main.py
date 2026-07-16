import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import sys

import threadpoolctl
import torch
import numpy as np
import datetime
import wandb

import common_utils
from common_utils.common import AverageValueMeter, load_weights, now, save_weights
from CreateData import setup_problem
from CreateModel import create_model
from extraction import calc_extraction_loss, evaluate_extraction, get_trainable_params, viz_nns
from evaluations import l2_dist, transform_vmin_vmax_batch
from GetParams import get_args
import math
import kornia
import matplotlib.pyplot as plt
import csv
from split import *
from analysis import find_best_ssim_scores_batch


thread_limit = threadpoolctl.threadpool_limits(limits=8)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

###############################################################################
#                               Train                                         #
###############################################################################
def get_loss_ce(args, model, x, y):
    p = model(x)
    if args.output_dim == 1:  # binary version
        p = p.view(-1)
        loss = torch.nn.BCEWithLogitsLoss()(p, y)
    else:
        loss = torch.nn.CrossEntropyLoss()(p, y)
    return loss, p


def get_total_err(args, p, y):
    # BCEWithLogitsLoss needs 0,1
    if args.output_dim == 1:  # binary version
        # labels are 0,1
        err = (p.sign().view(-1).add(1).div(2) != y).float().mean().item()
    else:
        err = (p.softmax(dim=1).argmax(dim=1) != y.long()).float().mean().item()    
    return err


def epoch_ce(args, dataloader, model, epoch, device, opt=None):
    total_loss, total_err = AverageValueMeter(), AverageValueMeter()
    model.train()
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        loss, p = get_loss_ce(args, model, x, y)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        err = get_total_err(args, p, y)
        total_err.update(err)

        total_loss.update(loss.item())
    return total_err.avg, total_loss.avg, p.data


def train(args, train_loader, test_loader, val_loader, model):
    if args.model_type == 'convfc':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=1e-3)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.train_lr)
    print('Model:')
    print(model)

    # Handle Reduce Mean
    if args.data_reduce_mean:
        print('Reducing Trainset-Mean from Trainset and Testset')
        Xtrn, Ytrn = next(iter(train_loader))
        ds_mean = Xtrn.mean(dim=0, keepdims=True)
        Xtrn = Xtrn - ds_mean
        train_loader = [(Xtrn, Ytrn)]

        Xtst, Ytst = next(iter(test_loader))
        Xtst = Xtst - ds_mean
        test_loader = [(Xtst, Ytst)]

    for epoch in range(args.train_epochs + 1):
        # if args.train_SGD:
        #     train_error, train_loss, output = epoch_ce_sgd(args, train_loader, model, epoch, args.device, args.train_SGD_batch_size, optimizer)
        # else:
        train_error, train_loss, output = epoch_ce(args, train_loader, model, epoch, args.device, optimizer)

        if epoch % args.train_evaluate_rate == 0:
            test_error, test_loss, _ = epoch_ce(args, test_loader, model, args.device, None, None)
            if val_loader is not None:
                validation_error, validation_loss, _ = epoch_ce(args, val_loader, model, args.device, None, None)
                print(now(), f'Epoch {epoch}: train-loss = {train_loss:.8g} ; train-error = {train_error:.4g} ; test-loss = {test_loss:.8g} ; test-error = {test_error:.4g} ; validation-loss = {validation_loss:.8g} ; validation-error = {validation_error:.4g} ; p-std = {output.abs().std()}; p-val = {output.abs().mean()}')
            else:
                print(now(),
                      f'Epoch {epoch}: train-loss = {train_loss:.8g} ; train-error = {train_error:.4g} ; test-loss = {test_loss:.8g} ; test-error = {test_error:.4g} ; p-std = {output.abs().std()}; p-val = {output.abs().mean()}')

            if args.wandb_active:
                wandb.log({"epoch": epoch, "train loss": train_loss, 'train error': train_error, 'p-val':output.abs().mean(), 'p-std': output.abs().std()})
                if val_loader is not None:
                    wandb.log({'validation loss': validation_loss, 'validation error': validation_error})
                wandb.log({'test loss': test_loss, 'test error': test_error})

        if np.isnan(train_loss):
            raise ValueError('Optimizer diverged')

        if train_loss < args.train_threshold:
            print(f'Reached train threshold {args.train_threshold} (train_loss={train_loss})')
            break

        if args.train_save_model_every > 0 and epoch % args.train_save_model_every == 0:
            save_weights(os.path.join(args.output_dir, 'weights'), model, ext_text=args.model_name, epoch=epoch)

    print(now(), 'ENDED TRAINING')
    return model


###############################################################################
#                               Extraction                                    #
###############################################################################

def log_score(csv_path, row):
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'split_epochs','loss', 'kkt_loss', 'loss_verify', 'extraction_score', 'dssim_score', 'psnr_score', 'num_atoms'])
    
    with open(csv_path, 'a', newline='') as f: 
        writer = csv.writer(f)
        writer.writerow(row)


def decay_lr(opt_x, opt_l, epoch, args, gamma=0.1):
    restart_period = args.extraction_evaluate_rate * 20
    base_lr = args.extraction_lr * (0.5 ** (epoch // restart_period))
    for g in opt_x.param_groups:
        g['lr'] = base_lr * 0.5 * (1 + math.cos(math.pi * (epoch % restart_period) / restart_period))
    for g in opt_l.param_groups:
        g['lr'] = base_lr * 0.5 * (1 + math.cos(math.pi * (epoch % restart_period) / restart_period))

def update_params_after_split(A, args, old_split_x, split_count, opt_method="SGD"):
    """
    This function updates the parameters after splitting, including x, y, l, source_x, opt_x, opt_l,
    and the split_x record that tracks the splitting rounds.
    """
    # Update x, y, l, and source_x
    x = torch.stack([a[0] for a in A]).to(args.device).detach().requires_grad_(True)
    y = torch.tensor([a[1] for a in A], device=args.device)
    l = torch.stack([a[2] for a in A]).to(args.device).detach().requires_grad_(True)
    source_x = torch.tensor([a[3] for a in A], device=args.device)  # Track the original source of each sample
    parent_id = torch.tensor([a[4] for a in A], device=args.device)
    sample_id = torch.tensor([a[5] for a in A], device=args.device)

    # Initialize new optimizers
    if opt_method == "Adam":
        opt_x = torch.optim.Adam([x], lr=args.extraction_lr)
        opt_l = torch.optim.Adam([l], lr=args.extraction_lambda_lr)
    else:
        opt_x = torch.optim.SGD([x], lr=args.extraction_lr, momentum=0.9)
        opt_l = torch.optim.SGD([l], lr=args.extraction_lambda_lr, momentum=0.9)

    # Update the split_x
    num_old = old_split_x.shape[0]
    num_new = len(A) - num_old
    split_x = torch.cat([
        old_split_x,
        torch.full((num_new,), split_count, dtype=torch.long, device=args.device)  # Track split round
    ])

    return x, y, l, source_x, parent_id, sample_id, split_x, opt_x, opt_l


def improved_data_extraction(args, dataset_loader, model, csv_path="extraction_log.csv", top_k=10, log_x=False, log_split=False, lr_decay=False,max_iter=100000,method="Haim",model_init=None,opt_method="SGD"):
    # we use dataset only for shapes and post-visualization (adding mean if it was reduced)
    x0, y0 = next(iter(dataset_loader))
    print('X:', x0.shape, x0.device)
    print('y:', y0.shape, y0.device)
    print('model device:', model.layers[0].weight.device)
    if args.data_reduce_mean:
        ds_mean = x0.mean(dim=0, keepdims=True)
        x0 = x0 - ds_mean

    # create labels
    y = torch.zeros(args.extraction_data_amount).type(torch.get_default_dtype()).to(args.device)
    if args.output_dim == 1: # binary (equal number of 1/-1)
        y[:y.shape[0] // 2] = -1
        y[y.shape[0] // 2:] = 1
    elif args.output_dim > 1:
        for c in range(args.num_classes):
            y[c * args.extraction_data_amount_per_class:(c+1) * args.extraction_data_amount_per_class] = c
    y = y.long()

    # trainable parameters
    l, opt_l, opt_x, x = get_trainable_params(args, x0, opt_method=opt_method, method=method)

    print('y type,shape:', y.type(), y.shape)
    print('l type,shape:', l.type(), l.shape)

    # record splitting information
    split_count = 0
    split_x = torch.zeros(x.shape[0], dtype=torch.long, device=args.device)
    source_x = torch.arange(x.shape[0], dtype=torch.long, device=args.device)  # Track original sources
    sample_id = torch.arange(x.shape[0], device=args.device)  # Global ID of the current sample
    parent_id = torch.full_like(sample_id, -1)  # Root nodes have parent = -1

    # extraction phase
    for epoch in range(max_iter):
        values = model(x).squeeze()
        loss, kkt_loss, loss_verify = calc_extraction_loss(args, l, model, values, x, y, method=method,model_init=model_init)
        if np.isnan(kkt_loss.item()):
            raise ValueError('Optimizer diverged during extraction')
        # optimization step
        opt_x.zero_grad()
        opt_l.zero_grad()
        loss.backward()
        opt_x.step()
        opt_l.step()
        
        # Learning rate decay
        if lr_decay == True:
            decay_lr(opt_x, opt_l, epoch, args)

        save_cicle = 10 * args.extraction_evaluate_rate if max_iter <= 100000 else 20*args.extraction_evaluate_rate
        if epoch == 0 or epoch == max_iter - 1 or (log_x == True and epoch % (10*args.extraction_evaluate_rate) == 0):
            torch.save(x.cpu(), os.path.join(args.output_dir, 'x', f'{epoch}_x.pth'))
        if epoch == 0 and log_split == True:
            torch.save(sample_id.cpu(), os.path.join(args.output_dir, 'l', f'{epoch}_sample_id.pth'))
        if epoch == max_iter - 1 and log_split == True:
            perform_log_split(args, x, x0, l, source_x, sample_id, parent_id, epoch, split_count,split_x, ds_mean,lr_decay=lr_decay)
        # Periodic splitting
        if epoch % args.extraction_epochs == 0 and epoch > 0 and epoch < max_iter:
            split_count += 1
            # before splitting
            print(f"Before splitting at epoch {epoch}")
            extraction_score, dssim_score, psnr_score,_ = evaluate_extraction(args, epoch, kkt_loss, loss_verify, x, x0, y0, ds_mean, top_k=top_k)
            log_score(csv_path, [epoch, args.extraction_epochs, loss.item(), kkt_loss.item(), loss_verify.item(), extraction_score, dssim_score, psnr_score, x.shape[0]])
            if log_split == True:
                perform_log_split(args, x, x0, l, source_x, sample_id, parent_id, epoch, split_count,split_x, ds_mean,lr_decay=lr_decay)
            old_split_x = split_x
            
            print(f"Splitting atoms at epoch {epoch}")
            A = list(zip(x, y, l, source_x, parent_id, sample_id))
            if len(A) >= 1000:
                growth_rate = 0.1
            else:
                growth_rate = 0.3
            A = sample_splitting(
                args, model, A,
                epsilon= 0.02,
                growth_rate=growth_rate,
                x0=x0, y0=y0, ds_mean=ds_mean, model_init=model_init,method=method
            )

            # after splitting
            print(f"After splitting at epoch {epoch}")
            x, y, l, source_x, parent_id, sample_id, split_x, opt_x, opt_l = update_params_after_split(A, args, old_split_x, split_count, opt_method=opt_method)
            if log_x == True:
                torch.save(sample_id.cpu(), os.path.join(args.output_dir, 'l', f'{epoch+10*args.extraction_evaluate_rate}_sample_id.pth'))
            
            values = model(x).squeeze()
            loss, kkt_loss, loss_verify = calc_extraction_loss(args, l, model, values, x, y, method=method, model_init=model_init)
            if np.isnan(kkt_loss.item()):
                raise ValueError('Optimizer diverged during extraction')
            opt_x.zero_grad()
            opt_l.zero_grad()
            loss.backward()
            extraction_score, dssim_score, psnr_score,_ = evaluate_extraction(args, epoch, kkt_loss, loss_verify, x, x0, y0, ds_mean, top_k=top_k)
            
        if epoch % args.extraction_evaluate_rate == 0:
            extraction_score, dssim_score, psnr_score,_ = evaluate_extraction(args, epoch, kkt_loss, loss_verify, x, x0, y0, ds_mean, top_k=top_k)
            log_score(csv_path, [epoch, args.extraction_epochs, loss.item(), kkt_loss.item(), loss_verify.item(), extraction_score, dssim_score, psnr_score, x.shape[0]])
            if epoch >= args.extraction_stop_threshold and extraction_score > 3300:
                print('Extraction Score is too low. Epoch:', epoch, 'Score:', extraction_score)
                break
        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return extraction_score

def perform_log_split(args, x, x0, l, source_id, sample_id, parent_id, epoch, split_count, split_x, ds_mean, lr_decay=False):
    if lr_decay == True:
        csv_file_top = os.path.join(args.output_dir, f"top_lr_decay_initial_{args.extraction_data_amount_per_class}_epochs{args.extraction_epochs}.csv")
        csv_file_split = os.path.join(args.output_dir, f"path_lr_decay_initial_{args.extraction_data_amount_per_class}_epochs{args.extraction_epochs}.csv")
    else:
        csv_file_top = os.path.join(args.output_dir, f"top_initial_{args.extraction_data_amount_per_class}_epochs{args.extraction_epochs}.csv")
        csv_file_split = os.path.join(args.output_dir, f"path_initial_{args.extraction_data_amount_per_class}_epochs{args.extraction_epochs}.csv")
   
    if not os.path.exists(csv_file_split):
        with open(csv_file_split, "w") as f:
            f.write("epoch,split_round,source_id,parent_id,sample_id,score,lambda\n")
    if not os.path.exists(csv_file_top):
        with open(csv_file_top, "w") as f:
            f.write("epoch,split_round,source_id,parent_id,sample_id,score,lambda\n")
    
    xx = x.data.clone()
    yy = x0.clone()
    ll = l.data.squeeze().clone().cpu().numpy()
    l2 = l2_dist(xx,yy).min(dim=1)[0]
    x0_mean = x0 + ds_mean
    xx_mean = transform_vmin_vmax_batch(xx + ds_mean)
    ssims = find_best_ssim_scores_batch(xx_mean, x0_mean)
    print(ssims[:10])
    if args.dataset == 'mnist':
        v_np = l2.detach().cpu().numpy()
    else:
        v_np = ssims.detach().cpu().numpy()
    src_np = source_id.cpu().numpy()
    sid_np = sample_id.cpu().numpy()
    pid_np = parent_id.cpu().numpy()

    with open(csv_file_top, "a") as f:
        for sa_id, so_id, p_id, v, l in zip(sid_np, src_np, pid_np, v_np, ll):
            if args.dataset == 'mnist' and v < 10:
                f.write(f"{epoch},{split_count},{so_id},{p_id},{sa_id},{v},{l}\n")
            elif args.dataset != 'mnist' and v > 0.4:
                f.write(f"{epoch},{split_count},{so_id},{p_id},{sa_id},{v},{l}\n")
    with open(csv_file_split, "a") as f:
        for sa_id, so_id, p_id, v, l in zip(sid_np, src_np, pid_np, v_np, ll):
            if sa_id != p_id and p_id != -1:
                f.write(f"{epoch},{split_count},{so_id},{p_id},{sa_id},{v},{l}\n")

###############################################################################
#                               MAIN                                          #
###############################################################################
def create_dirs_save_files(args):
    if args.train_save_model or args.extract_save_results or args.extract_save_results:
        # create dirs
        os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'x'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'l'), exist_ok=True)

    if args.save_args_files:
        # save args
        common_utils.common.dump_obj_with_dict(args, f"{args.output_dir}/args.txt")
        # save command line
        with open(f"{args.output_dir}/sys.args.txt", 'w') as f:
            f.write(" ".join(sys.argv))

def setup_args(args):
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    from settings import datasets_dir, models_dir, results_base_dir
    args.results_base_dir = results_base_dir
    args.datasets_dir = datasets_dir
    if args.pretrained_model_path:
        args.pretrained_model_path = os.path.join(models_dir, args.pretrained_model_path)
    if args.initial_model_path:
        args.initial_model_path = os.path.join(models_dir, args.initial_model_path)
    args.model_name = f'{args.problem}_d{args.data_per_class_train}'
    if args.proj_name:
        args.model_name += f'_{args.proj_name}'

    torch.manual_seed(args.seed)

    if args.wandb_active:
        wandb.init(project=args.wandb_project_name, entity='dataset_reconsruction')
        wandb.config.update(args)

    if args.wandb_active:
        args.output_dir = wandb.run.dir
    else:
        import dateutil.tz
        timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
        run_name = f'{timestamp}_{args.model_name}_{args.extraction_method}' 
        args.output_dir = os.path.join(args.results_base_dir, run_name)
    print('OUTPUT_DIR:', args.output_dir)

    args.wandb_base_path = './'

    return args


def main_train(args, train_loader, test_loader, val_loader):
    print('TRAINING A MODEL')
    model = create_model(args, extraction=False)
    save_weights(args.output_dir, model, ext_text=f'{args.model_name}_initial')
    if args.wandb_active:
        wandb.watch(model)

    trained_model = train(args, train_loader, test_loader, val_loader, model)
    if args.train_save_model:
        save_weights(args.output_dir, trained_model, ext_text=args.model_name)


def main_reconstruct(args, train_loader):
    print('USING PRETRAINED MODEL AT:', args.pretrained_model_path)
    extraction_model = create_model(args, extraction=True)
    extraction_model.eval()
    extraction_model = load_weights(extraction_model, args.pretrained_model_path, device=args.device)
    print('EXTRACTION MODEL:')
    print(extraction_model)

    if args.extraction_method == 'Loo' and args.initial_model_path:
        initial_model = create_model(args, extraction=True)
        initial_model.eval()
        initial_model = load_weights(initial_model, args.initial_model_path, device=args.device)
    top_k=10
    lr_decay=args.lr_decay
    max_iter=args.max_extraction_iter
    #max_iter = 200000  # using args.max_extraction_iter instead
    log_split=args.log_split
    log_x=args.log_x
    os.makedirs(args.problem, exist_ok=True)
    if lr_decay == True:
        csv_path = os.path.join(args.output_dir, f'lr_decay_initial_{args.extraction_data_amount_per_class}_epochs{args.extraction_epochs}_top{top_k}.csv')
    else:
        csv_path = os.path.join(args.output_dir, f'initial_{args.extraction_data_amount_per_class}_epochs{args.extraction_epochs}_top{top_k}.csv')
    if args.extraction_method == 'Loo' and args.initial_model_path:
        return improved_data_extraction(args, train_loader, extraction_model,csv_path=csv_path,top_k=top_k,log_x=log_x,log_split=log_split, lr_decay=lr_decay,max_iter=max_iter,model_init=initial_model,method=args.extraction_method,opt_method=args.extraction_opt_method)
    else:
        return improved_data_extraction(args, train_loader, extraction_model,csv_path=csv_path,top_k=top_k,log_x=log_x,log_split=log_split, lr_decay=lr_decay,max_iter=max_iter,opt_method=args.extraction_opt_method)



def validate_settings_exists():
    if os.path.isfile("settings.py"):
        return
    raise FileNotFoundError("You should create a 'settings.py' file with the contents of 'settings.deafult.py', " + 
                            "adjusted according to your system")


def main():
    print(now(), 'STARTING!')
    validate_settings_exists()
    args = get_args(sys.argv[1:])
    args = setup_args(args)
    create_dirs_save_files(args)

    if args.precision == 'double':
        torch.set_default_dtype(torch.float64)

    train_loader, test_loader, val_loader = setup_problem(args)

    # train
    if args.run_mode == 'train':
        main_train(args, train_loader, test_loader, val_loader)
    # reconstruct
    elif args.run_mode == 'reconstruct':
        main_reconstruct(args, train_loader)
    else:
        raise ValueError(f'no such args.run_mode={args.run_mode}')

if __name__ == '__main__':
    main()
