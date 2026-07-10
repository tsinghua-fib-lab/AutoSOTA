# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import build_geom_dataset
from configs.datasets_config import geom_with_h
import copy
import utils
import argparse
import wandb
from os.path import join
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion
from equivariant_diffusion import en_diffusion

from equivariant_diffusion import utils as diffusion_utils
import torch
import time
import pickle

from qm9.utils import prepare_context, compute_mean_mad
import train_test
import os

parser = argparse.ArgumentParser(description='e3_diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')

# Latent Diffusion args
parser.add_argument('--train_diffusion', action='store_true', 
                    help='Train second stage LatentDiffusionModel model')
parser.add_argument('--ae_path', type=str, default=None,
                    help='Specify first stage model path')
parser.add_argument('--trainable_ae', action='store_true',
                    help='Train first stage AutoEncoder model')

# VAE args
parser.add_argument('--latent_nf', type=int, default=4,
                    help='number of latent features')
parser.add_argument('--kl_weight', type=float, default=0.01,
                    help='weight of KL term in ELBO')

parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)

parser.add_argument('--n_epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=192,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='geom',
                    help='dataset name')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=50)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
parser.add_argument('--save_model', type=eval, default=True, help='save model')
parser.add_argument('--generate_epochs', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=1)
parser.add_argument('--data_augmentation', type=eval, default=False,
                    help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='multiple arguments can be passed, '
                         'including: homo | onehot | lumo | num_atoms | etc. '
                         'usage: "--conditioning H_thermo homo onehot H_thermo"')
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0,           # TODO
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=20,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=False, help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=5000)
parser.add_argument('--normalization_factor', type=float,
                    default=100, help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean" aggregation for the graph network')
parser.add_argument('--filter_molecule_size', type=int, default=None,
                    help="Only use molecules below this size.")
parser.add_argument('--sequential', action='store_true',
                    help='Organize data by size to reduce average memory usage.')
parser.add_argument('--StP', action='store_true', default=False, help='Enable StP')
parser.add_argument('--data_norm', action='store_true', default=False, help='Enable data normalization')
args = parser.parse_args()
if args.StP and args.data_norm:
    parser.error("Only one of --StP or --norm_data can be True at the same time.")
    
exp_dir = os.path.join("outputs", args.exp_name)
if os.path.exists(exp_dir):
    ans = input(f"The folder {exp_dir} already exists. Overwrite? [y/N]: ")
    if ans.lower() != 'y':
        print("Aborting to avoid overwriting existing folder.")
        exit(1)
    
if args.StP:
    print(">>> [USING StP] <<<")
elif args.data_norm:
    print(">>> [USING DATA NORMALIZATION] <<<")

data_file = '../processed_dataset/geom/geom_drugs_30.npy'

if args.remove_h:
    raise NotImplementedError()
else:
    dataset_info = geom_with_h

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

split_data = build_geom_dataset.load_split_data(data_file, val_proportion=0.1, test_proportion=0.1, filter_size=args.filter_molecule_size)
transform = build_geom_dataset.GeomDrugsTransform(dataset_info, args.include_charges, device, args.sequential)
dataloaders = {}
for key, data_list in zip(['train', 'val', 'test'], split_data):
    dataset = build_geom_dataset.GeomDrugsDataset(data_list, transform=transform)
    shuffle = (key == 'train') and not args.sequential

    # Sequential dataloading disabled for now.
    dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
        sequential=args.sequential, dataset=dataset, batch_size=args.batch_size,
        shuffle=shuffle)
del split_data

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    args.resume = resume
    args.break_train_epoch = False
    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr
    print(args)

utils.create_folders(args)
print(args)

# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion_geom', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

data_dummy = next(iter(dataloaders['train']))


if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf


# Create Latent Diffusion Model or Audoencoder
if args.train_diffusion:
    model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, dataloaders['train'])
else:
    model, nodes_dist, prop_dist = get_autoencoder(args, device, dataset_info, dataloaders['train'])

model = model.to(device)
optim = get_optim(args, model)
# print(model)


gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def main():
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
        dequantizer_state_dict = torch.load(join(args.resume, 'dequantizer.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1 and args.cuda:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = diffusion_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        train_test.train_epoch(args, dataloaders['train'], epoch, model, model_dp, model_ema, ema, device, dtype,
                               property_norms, optim, nodes_dist, gradnorm_queue, dataset_info,
                               prop_dist)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch > 5 and epoch % args.test_epochs == 0:
            nll_val = train_test.test(args, dataloaders['val'], epoch, model_ema_dp, device, dtype,
                                      property_norms, nodes_dist, partition='Val')
            with open(f"outputs/{args.exp_name}/log.txt", "a") as f:
                f.write(f"Epoch {epoch}: NLL = {nll_val}\n")
                
            if nll_val < best_nll_val:
                best_nll_val = nll_val
                if args.save_model:
                    args.current_epoch = epoch + 1
                    epoch_dir = f"outputs/{args.exp_name}/epoch_{epoch}"
                    os.makedirs(epoch_dir, exist_ok=True)
                    # save current best
                    utils.save_model(optim, f"outputs/{args.exp_name}/epoch_{epoch}/optim.npy")
                    utils.save_model(model, f"outputs/{args.exp_name}/epoch_{epoch}/generative_model.npy")
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, f"outputs/{args.exp_name}/epoch_{epoch}/generative_model_ema.npy")
                    with open(f"outputs/{args.exp_name}/epoch_{epoch}/args.pickle", "wb") as f:
                        pickle.dump(args, f)
                    # save universal best
                    utils.save_model(optim, f"outputs/{args.exp_name}/optim.npy")
                    utils.save_model(model, f"outputs/{args.exp_name}/generative_model.npy")
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, f"outputs/{args.exp_name}/generative_model_ema.npy")
                    with open(f"outputs/{args.exp_name}/args.pickle", "wb") as f:
                        pickle.dump(args, f)
                    # skip periodic saving for this epoch
                    continue

            # periodic checkpoint saving 
            if args.save_model and epoch > 30 and epoch % 10 == 0:
                args.current_epoch = epoch + 1
                epoch_dir = f"outputs/{args.exp_name}/epoch_{epoch}"
                os.makedirs(epoch_dir, exist_ok=True)
                utils.save_model(optim, f"outputs/{args.exp_name}/epoch_{epoch}/optim.npy")
                utils.save_model(model, f"outputs/{args.exp_name}/epoch_{epoch}/generative_model.npy")
                if args.ema_decay > 0:
                    utils.save_model(model_ema, f"outputs/{args.exp_name}/epoch_{epoch}/generative_model_ema.npy")
                with open(f"outputs/{args.exp_name}/epoch_{epoch}/args.pickle", "wb") as f:
                    pickle.dump(args, f)




if __name__ == "__main__":
    main()
