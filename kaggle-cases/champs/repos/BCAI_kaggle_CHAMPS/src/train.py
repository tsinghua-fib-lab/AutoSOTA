## Copyright (c) 2017 Robert Bosch GmbH
## All rights reserved.
##
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from graph_transformer import *
import json
import time
from setproctitle import setproctitle
import numpy as np

from utils import radam
from utils.csv_to_pkl import get_datasets
from utils.exp_utils import *
from utils.data_parallel import BalancedDataParallel
from utils.filters import *
import argparse

root = '../'  # This should be the root of the archive
with open(os.path.join(root,'SETTINGS.json')) as f:
    settings = json.load(f)

parser = argparse.ArgumentParser(description='Graph Transformer on Predicting Molecular properties')
parser.add_argument('--data', type=str, default=os.path.join(root,settings['PROCESSED_DATA_DIR']),
                    help='location of the (processed) data')
parser.add_argument('--n_layer', type=int, default=14,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=650,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=3800,
                    help='inner dimension in posFF')
parser.add_argument('--feature_dim', type=int, default=200,
                    help='extra feature (e.g., charge, angle) dimension, if learnable')
parser.add_argument('--final_dim', type=int, default=280,
                    help='final layer hidden dimension')
parser.add_argument('--dropout', type=float, default=0.03,
                    help='global dropout rate (applies to residual blocks in transformer)')
parser.add_argument('--cutout', type=float, default=0.0,
                    help='frequency to use cutout (randomly throw away an atom and its nearest neighbor)')
parser.add_argument('--final_dropout', type=float, default=0.04,
                    help='final layer dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')
parser.add_argument('--optim', default='Adam', type=str,
                    choices=['Adam', 'SGD', 'Adagrad', 'RAdam'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate (0.0001|5 for adam|sgd)')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=5000,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--max_epoch', type=int, default=250,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=48,
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--patience', type=int, default=5,
                    help='patience')
parser.add_argument('--champs_loss', action='store_true',
                    help='use CHAMPS loss')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--fp16', action='store_true',
                    help='use mixed precision from apex to save memory')
parser.add_argument('--wnorm', action='store_true',
                    help='use weight normalization')
parser.add_argument('--max_bond_count', type=int, default=250,
                    help='maximum bond usage')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='batch chunking')
parser.add_argument('--work_dir', default=os.path.join(root,settings['MODEL_DIR'],'CHAMP-GT'), type=str,
                    help='experiment directory.')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--load', type=str, default='',
                    help='the path to the model you want to load')
parser.add_argument('--dist_embed_type', type=str, default='sine',
                    choices=['sine', 'learnable'],
                    help='the embedding type for distance features')
parser.add_argument('--angle_embed_type', type=str, default='sine',
                    choices=['sine', 'learnable'],
                    help='the embedding type for angle features')
parser.add_argument('--quad_angle_embed_type', type=str, default='sine',
                    choices=['sine', 'learnable'],
                    help='the embedding type for angle features')
parser.add_argument('--use_quad', action='store_true',
                    help='use quadruplet information')
parser.add_argument('--name', type=str, default='',
                    help='name of the experiment')
parser.add_argument('--mode', type=str, default='_full',
                    choices=['', '_full'],
                    help='mode of the dataset')
parser.add_argument('--eta_min', type=float, default=1e-7,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')

args = parser.parse_args()

if args.d_embed < 0:
    args.d_embed = args.d_model

args.work_dir = '{}'.format(args.work_dir)
args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
logging = create_exp_dir(args.work_dir,
    scripts_to_save=['train.py', 'graph_transformer.py'], debug=args.debug)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')
setproctitle('CHAMP-GT' if len(args.name) == 0 else args.name)

APEX_AVAILABLE = False
if args.fp16:
    try:
        from apex import amp
        APEX_AVAILABLE = True
    except ModuleNotFoundError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
        APEX_AVAILABLE = False

if len(args.name) == 0:
    args.name = f'{args.d_model}x{args.n_layer}_dr{int(args.dropout*100)}_nhead{args.n_head}'

#####################################################################
#
# Loading dataset, set macros
#
#####################################################################

batch_size = args.batch_size
train_dataset, val_dataset = get_datasets(args.data, mode=args.mode)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
if val_dataset is not None:
    assert args.mode != "_full", "You should NOT use the validation set when not in _full mode"
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

NUM_ATOM_TYPES = [int(train_dataset.tensors[1][:,:,i].max()) for i in range(3)]   # Atom hierarchy has 3 levels
NUM_BOND_TYPES = [int(train_dataset.tensors[3][:,:,i].max()) for i in range(3)]   # Bond hierarchy has 3 levels
NUM_TRIPLET_TYPES = [int(train_dataset.tensors[5][:,:,i].max()) for i in range(2)]  # Triplet hierarchy has 2 levels
NUM_QUAD_TYPES = [int(train_dataset.tensors[7][:,:,i].max()) for i in range(1)]   # Quad hierarchy has only 1 level
NUM_BOND_ORIG_TYPES = 8
MAX_BOND_COUNT = args.max_bond_count
print(f"Atom hierarchy: {NUM_ATOM_TYPES}")
print(f"Bond hierarchy: {NUM_BOND_TYPES}")
print(f"Triplet hierarchy: {NUM_TRIPLET_TYPES}")
if args.use_quad:
    print(f"Quad hierarchy: {NUM_QUAD_TYPES}")

#####################################################################
#
# Build the model, optimizer and the objective(s)
#
#####################################################################

# 1. Create a model or load a model
if args.restart:
    with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
if len(args.load) > 0:
    model = torch.load(args.load)
else:
    model = GraphTransformer(dim=args.d_model, n_layers=args.n_layer, d_inner=args.d_inner,
                             fdim = args.feature_dim, final_dim=args.final_dim, dropout=args.dropout,
                             dropatt=args.dropatt, final_dropout=args.final_dropout, n_head=args.n_head,
                             num_atom_types=NUM_ATOM_TYPES,
                             num_bond_types=NUM_BOND_TYPES,
                             num_triplet_types=NUM_TRIPLET_TYPES,
                             num_quad_types=NUM_QUAD_TYPES,
                             dist_embedding=args.dist_embed_type,
                             atom_angle_embedding=args.angle_embed_type,
                             trip_angle_embedding=args.quad_angle_embed_type,
                             quad_angle_embedding=args.quad_angle_embed_type,
                             wnorm=args.wnorm,
                             use_quad=args.use_quad).to(device)
args.n_all_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])


# 2. Initialize optimizer and learning rate scheduler
optimizer = getattr(optim if args.optim != "RAdam" else radam, args.optim)(model.parameters(), lr=args.lr)
if args.optim == "RAdam":
    print("Using RAdam optimizer!")

args.max_step = args.max_epoch * len(train_loader)
if args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_step, eta_min=args.eta_min)
elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step else step / (args.warmup_step ** 1.5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay_rate,
                                                     patience=args.patience, min_lr=args.lr_min)

# 3. Load the optimizer if we are restarting from somewhere
if args.restart:
    if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
        with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
            opt_state_dict = torch.load(f)
            optimizer.load_state_dict(opt_state_dict)
    else:
        print('Optimizer was not saved. Start from scratch.')

# 4. Handle mixed precision stuff (which MUST be before the DataParallel call, if applicable)
if APEX_AVAILABLE:
    # Currently, only 'O1' is supported with DataParallel. See here: https://github.com/NVIDIA/apex/issues/227
    opt_level = "O1" if args.multi_gpu else "O2"
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, loss_scale="dynamic")

# 5. Handle data parallelism
if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0:
        para_model = BalancedDataParallel(args.gpu0_bsz, model).to(device)
    else:
        para_model = nn.DataParallel(model).to(device)
else:
    para_model = model.to(device)

logging('=' * 60)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 60)
logging(f'#params = {args.n_all_param/1e6:.2f}M')


#####################################################################
#
# Training script
#
#####################################################################

def loss(y_pred, y, x_bond):
    y_pred_pad = torch.cat([torch.zeros(y_pred.shape[0], 1, y_pred.shape[2], device=y_pred.device), y_pred], dim=1)

    # Note: The [:,:,1] below should match the num_bond_types[1]*final_dim in graph transformer
    y_pred_scaled = y_pred_pad.gather(1,x_bond[:,:,1][:,None,:])[:,0,:] * y[:,:,2] + y[:,:,1]
    abs_dy = (y_pred_scaled - y[:,:,0]).abs()
    loss_bonds = (x_bond[:,:,0] > 0)
    abs_err = abs_dy.masked_select(loss_bonds & (y[:,:,3] > 0)).sum()

    type_dy = [abs_dy.masked_select(x_bond[:,:,0] == i) for i in range(1,NUM_BOND_ORIG_TYPES+1)]
    if args.champs_loss:
        type_err = torch.cat([t.sum().view(1) for t in type_dy], dim=0)
        type_cnt = torch.cat([torch.sum(x_bond[:,:,0] == i).view(1) for i in range(1,NUM_BOND_ORIG_TYPES+1)])
    else:
        type_err = torch.tensor([t.sum() for t in type_dy])
        type_cnt = torch.tensor([len(t) for t in type_dy])
    return abs_err, type_err, type_cnt


def epoch(loader, model, opt=None, ep=-1):
    global train_step
    model.eval() if opt is None else model.train()
    dev = next(model.parameters()).device
    abs_err, type_err, type_cnt = 0.0, torch.zeros(NUM_BOND_ORIG_TYPES), torch.zeros(NUM_BOND_ORIG_TYPES, dtype=torch.long)
    log_interval = args.log_interval

    with torch.enable_grad() if opt else torch.no_grad():
        batch_id = 0
        total_loss = torch.zeros(NUM_BOND_ORIG_TYPES)
        for x_idx, x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, x_quad, x_quad_angle, y in loader:
            x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, x_quad, x_quad_angle, y = \
                x_atom.to(dev), x_atom_pos.to(dev), x_bond.to(dev), x_bond_dist.to(dev), \
                x_triplet.to(dev), x_triplet_angle.to(dev), x_quad.to(dev), x_quad_angle.to(dev), y.to(dev)

            x_bond, x_bond_dist, y = x_bond[:, :MAX_BOND_COUNT], x_bond_dist[:, :MAX_BOND_COUNT], y[:,:MAX_BOND_COUNT]

            if opt:
                # Put this here so that the batch_chunk setting will work
                opt.zero_grad()

                # Perform cutout on the molecule (i.e., for a large molecule, randomly remove an atom and its nearest
                # neighbor; then remove all bonds/triplets related to this atom)
                x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, _ = \
                    subgraph_filter(x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, args)

            if args.batch_chunk > 1:
                mbsz = args.batch_size // args.batch_chunk
                b_abs_err = 0
                b_type_err = 0
                b_type_cnt = 0
                types_cnt = sum([(x_bond[:,:,0] == i).sum() for i in range(1,NUM_BOND_ORIG_TYPES+1)])
                for i in range(args.batch_chunk):
                    mini = slice(i*mbsz,(i+1)*mbsz)
                    y_pred_mb, _ = para_model(x_atom[mini], x_atom_pos[mini], x_bond[mini], x_bond_dist[mini],
                                              x_triplet[mini], x_triplet_angle[mini], x_quad[mini], x_quad_angle[mini])
                    mb_abs_err, mb_type_err, mb_type_cnt = loss(y_pred_mb, y[mini], x_bond[mini])
                    b_abs_err += mb_abs_err.detach()       # No need to average, as it's sum
                    b_type_err += mb_type_err.detach()
                    b_type_cnt += mb_type_cnt.detach()
                    mb_raw_loss = mb_abs_err / types_cnt.float()
                    if args.champs_loss:
                        raise ValueError("CHAMPS loss not supported yet with batch_chunk mode")
                    if APEX_AVAILABLE:
                        with amp.scale_loss(mb_raw_loss, opt) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        mb_raw_loss.backward()
            else:
                y_pred, _ = para_model(x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, x_quad, x_quad_angle)
                b_abs_err, b_type_err, b_type_cnt = loss(y_pred, y, x_bond)

            abs_err += b_abs_err.detach()
            type_err += b_type_err.detach()
            type_cnt += b_type_cnt.detach()
            batch_id += 1
            total_loss += b_type_err / b_type_cnt.float()

            if opt:
                train_step += 1
                if train_step <= args.warmup_step:
                    curr_lr = args.lr * train_step / args.warmup_step
                    opt.param_groups[0]['lr'] = curr_lr
                elif args.scheduler == 'cosine':
                    scheduler.step(train_step)
                if args.batch_chunk == 1:
                    raw_loss = b_abs_err/b_type_cnt.sum()
                    if args.champs_loss:
                        nonzero_indices = b_type_cnt.nonzero()
                        raw_loss = torch.log((b_type_err[nonzero_indices] / b_type_cnt[nonzero_indices].float()) + 1e-9).mean()
                    if APEX_AVAILABLE:
                        with amp.scale_loss(raw_loss, opt) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        raw_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()

                if batch_id % log_interval == 0:
                    avg_loss = torch.log(total_loss / log_interval).mean().item()
                    logging(f"Epoch {ep:2d} | Step {train_step} | lr {opt.param_groups[0]['lr']:.7f} | Error {avg_loss:.5f}")

                    total_loss = 0

    torch.cuda.empty_cache()
    return abs_err / type_cnt.sum(), torch.log(type_err / type_cnt.float()).mean(), torch.log(type_err / type_cnt.float())


if __name__ == '__main__':
    train_step = 0
    start_epoch = 0
    best_val_err = 1e8

    for i in range(start_epoch, args.max_epoch):
        start = time.time()
        _, _, _ = epoch(train_loader, model, optimizer, ep=i)
        if val_dataset is not None:
            _, err, type_err = epoch(val_loader, model)
            end = time.time()
            logging(f"Epoch {i:2d} | Time {end-start:.2f} sec | Validation Error {err:.5f}")
            type_err_str = [eval(f"{elem:.2f}") for elem in type_err.tolist()]
            logging(f"Epoch {i:2d} | Validation Error (by Type) \n {type_err_str}")
            if err < best_val_err:
                with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                    torch.save(model, f)
                with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                    torch.save(optimizer.state_dict(), f)
                save_path = os.path.join(root,settings['MODEL_DIR'],f"ckpt/graph_transformer_part_{args.name}.ckpt")
                logging(f"Saving model at {save_path}!")
                best_val_err = err
                torch.save(model, save_path)
        else:
            end = time.time()
            save_path = os.path.join(root,settings['MODEL_DIR'],f"ckpt/graph_transformer_{args.name}.ckpt")
            logging(f"Saving model at {save_path} (without validation error)! Time: {end-start:.2f} sec")
            torch.save(model, save_path)
        if args.scheduler == 'dev_perf':
            scheduler.step(err)
