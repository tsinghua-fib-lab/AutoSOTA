# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import utils
import argparse
from configs.datasets_config import qm9_with_h, qm9_without_h
from qm9 import dataset
from qm9.models import get_model, get_autoencoder, get_latent_diffusion

from equivariant_diffusion.utils import assert_correctly_masked
import torch
import pickle
import qm9.visualizer as vis
from qm9.analyze import check_stability
from os.path import join
from qm9.sampling import sample_chain, sample
from configs.datasets_config import get_dataset_info


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def save_and_sample_chain(args, eval_args, device, flow,
                          n_tries, n_nodes, dataset_info, id_from=0,
                          num_chains=100):

    for i in range(num_chains):
        target_path = f'eval/chain_{i}/'

        one_hot, charges, x = sample_chain(
            args, device, flow, n_tries, dataset_info)

        vis.save_xyz_file(
            join(eval_args.model_path, target_path), one_hot, charges, x,
            dataset_info, id_from, name='chain')

        vis.visualize_chain_uncertainty(
            join(eval_args.model_path, target_path), dataset_info,
            spheres_3d=True)

    return one_hot, charges, x


def sample_different_sizes_and_save(args, eval_args, device, generative_model,
                                    nodes_dist, dataset_info, n_samples=10):
    nodesxsample = nodes_dist.sample(n_samples)
    one_hot, charges, x, node_mask = sample(
        args, device, generative_model, dataset_info,
        nodesxsample=nodesxsample)

    vis.save_xyz_file(
        join(eval_args.model_path, 'eval/molecules/'), one_hot, charges, x,
        id_from=0, name='molecule', dataset_info=dataset_info,
        node_mask=node_mask)


def sample_only_stable_different_sizes_and_save(
        args, eval_args, device, flow, nodes_dist,
        dataset_info, n_samples=10, n_tries=50):
    assert n_tries > n_samples

    nodesxsample = nodes_dist.sample(n_tries)
    one_hot, charges, x, node_mask = sample(
        args, device, flow, dataset_info,
        nodesxsample=nodesxsample)

    counter = 0
    for i in range(n_tries):
        num_atoms = int(node_mask[i:i+1].sum().item())
        atom_type = one_hot[i:i+1, :num_atoms].argmax(2).squeeze(0).cpu().detach().numpy()
        x_squeeze = x[i:i+1, :num_atoms].squeeze(0).cpu().detach().numpy()
        mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

        num_remaining_attempts = n_tries - i - 1
        num_remaining_samples = n_samples - counter

        if mol_stable or num_remaining_attempts <= num_remaining_samples:
            if mol_stable:
                print('Found stable mol.')
            vis.save_xyz_file(
                join(eval_args.model_path, 'eval/molecules/'),
                one_hot[i:i+1], charges[i:i+1], x[i:i+1],
                id_from=counter, name='molecule_stable',
                dataset_info=dataset_info,
                node_mask=node_mask[i:i+1])
            counter += 1

            if counter >= n_samples:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument(
        '--n_tries', type=int, default=10,
        help='N tries to find stable molecule for gif animation')
    parser.add_argument('--n_nodes', type=int, default=19,
                        help='number of atoms in molecule for gif animation')

    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    flow, nodes_dist, prop_dist = get_latent_diffusion(
        args, device, dataset_info, dataloaders['train'])
    flow.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn),
                                 map_location=device)

    flow.load_state_dict(flow_state_dict)

    print('Sampling handful of molecules.')
    sample_different_sizes_and_save(
        args, eval_args, device, flow, nodes_dist,
        dataset_info=dataset_info, n_samples=30)

    print('Sampling stable molecules.')
    sample_only_stable_different_sizes_and_save(
        args, eval_args, device, flow, nodes_dist,
        dataset_info=dataset_info, n_samples=10, n_tries=2*10)
    print('Visualizing molecules.')
    vis.visualize(
        join(eval_args.model_path, 'eval/molecules/'), dataset_info,
        max_num=100, spheres_3d=True)


if __name__ == "__main__":
    main()
