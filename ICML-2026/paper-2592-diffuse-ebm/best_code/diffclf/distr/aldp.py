# Implementation of alanine dipeptide

# Libraries
import boltzgen as bg
import mdtraj
import numpy as np
import torch
from ..metrics.kld import kl_divergence
from ..metrics.wasserstein import wasserstein_distance_1d
from ..utils.plot_utils import plot_ramachandran_hist2d, plot_phi_psi_train_pred_hist1d, \
    plot_free_energy_projection, filter_nan
from ..utils.se3_utils import remove_mean, compute_intersection, compute_correlation
from .base import Distribution
from .utils_aldp import torch_to_mdtraj, filter_chirality, compute_phi_psi
from openmmtools import testsystems

class AlanineDipeptide(Distribution):
    def __init__(self, data_path, temperature=300, energy_cut=1e8, energy_max=1e20,
            n_threads=4, env='vacuum'):
        """Boltzmann distribution of Alanine dipeptide

        Args:
            * data_path (str): Path to the trajectory file used to initialize the transformation.
            * temperature (int): Temperature of the system. (default is 1000)
            * energy_cut (float): Value after which the energy is logarithmically scaled. (default is 1e8)
            * energy_max (float): Maximum energy allowed; higher energies are cut. (default is 1e20)
            * n_threads (int): Number of threads used to evaluate the log probability for batches.
                (default is 4)
        """

        # Call the parent constructor
        super().__init__(build_score=False, build_log_prob_and_grad=False, build_laplacian=False,
                 build_log_prob_and_grad_and_laplacian=False)
        # Define molecule parameters
        self.n_particles = 22
        self.n_dimensions = 3
        self.data_shape = (self.n_particles, self.n_dimensions)
        # System setup
        if env == 'vacuum':
            system = testsystems.AlanineDipeptideVacuum(constraints=None)
        elif env == 'implicit':
            system = testsystems.AlanineDipeptideImplicit(constraints=None)
        else:
            raise NotImplementedError('This environment is not implemented.')
        # Load trajectory
        traj = mdtraj.load(data_path)
        traj.center_coordinates()
        ind = traj.top.select("backbone")
        traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)
        self.topology = traj.topology
        # Gather the training data
        self.has_data = True
        self.load_data(remove_mean(
            torch.from_numpy(traj.xyz).view((-1, *self.data_shape))
        ))
        # Define the Boltzmann distribution
        if n_threads > 1:
            self.boltz = bg.distributions.BoltzmannParallel(system, temperature,
                energy_cut=energy_cut, energy_max=energy_max, n_threads=n_threads)
        else:
            self.boltz = bg.distributions.Boltzmann(system, temperature,
                energy_cut=energy_cut, energy_max=energy_max)
        
    def build_dist(self):
        """Builds the inner dist object"""
        pass

    def get_bonds(self):
        """Returns the bonds"""
        return [
            (0, 1), (1, 2), (1, 3), (1, 4),
            (4, 5), (4, 6), (6, 7), (6, 8),
            (8, 9), (8, 10), (10, 11), (10, 12),
            (10, 13), (8, 14), (14, 15), (14, 16),
            (16, 17), (16, 18), (18, 19), (18, 20),
            (18, 21)
        ]

    def get_atom_chemical_types(self):
        """Returns the atom chemical types"""
        chemical_elements = ['H', 'C', 'N', 'O']
        table = self.topology.to_dataframe()[0]
        chemical_to_index = {element: idx for idx, element in enumerate(chemical_elements)}
        return [chemical_to_index[element] for element in table["element"].values]
    
    def log_prob(self, x: torch.tensor):
        """Evaluates the log-likelihood of the distribution at x"""
        return self.boltz.log_prob(x)
    
    def log_prob_and_grad(self, x):
        """Evaluates the log-likelihood and the score of the distribution at x"""
        x_ = torch.autograd.Variable(x, requires_grad=True)
        log_prob_x = self.boltz.log_prob(x_.view((-1, self.n_particles * self.n_dimensions)))
        return log_prob_x, torch.autograd.grad(log_prob_x.sum(), x_)[0].detach().view(x.shape)

    def score(self, x):
        """Evaluates the score of the distribution at x"""
        x_ = torch.autograd.Variable(x, requires_grad=True)
        log_prob_x = self.boltz.log_prob(x_.view((-1, self.n_particles * self.n_dimensions)))
        return torch.autograd.grad(log_prob_x.sum(), x_)[0].detach().view(x.shape)

    def filter_chirality(self, samples):
        """Replace the chiral forms"""
        data_shape = samples.shape[1:]
        ref_samples = self.sample((samples.shape[0],)).cpu()
        target_traj = torch_to_mdtraj(ref_samples.view(-1, self.n_particles, self.n_dimensions), self.topology)
        samples, D_form_counter = filter_chirality(samples, target_traj)
        print(f"Number of D-form samples are changed from {D_form_counter[0] / samples.shape[0]} to {D_form_counter[1] / samples.shape[0]}")
        return samples.view(-1, *data_shape)

    def compute_psi_phi(self, samples):
        """Compute the diheral angles"""
        return filter_nan(*compute_phi_psi(torch_to_mdtraj(samples, self.topology)))

    def compute_ramachandran_kld(self, phi_source, psi_source, phi_target, psi_target, source_weights=None,
            bins_1d=256, bins_2d=128):
        """Compute the KL between histograms in Ramachandran coordinates"""
        dihedral_source = np.stack([phi_source, psi_source], axis=1)
        dihedral_target = np.stack([phi_target, psi_target], axis=1)
        metrics = {
            "kld_phi": kl_divergence(phi_source, phi_target, num_bins=bins_1d, ranges=[[-np.pi, np.pi]],
                source_weights=source_weights.detach().cpu().numpy() if source_weights is not None else None),
            "kld_psi": kl_divergence(psi_source, psi_target, num_bins=bins_1d, ranges=[[-np.pi, np.pi]],
                source_weights=source_weights.detach().cpu().numpy() if source_weights is not None else None),
            "kld_ramachandran": kl_divergence(dihedral_source, dihedral_target, num_bins=bins_2d,
                ranges=[[-np.pi, np.pi], [-np.pi, np.pi]],
                source_weights=source_weights.detach().cpu().numpy() if source_weights is not None else None)
        }
        return metrics

    def compute_energy_histograms(self, x, bins, range=None, return_en=False, weights=None):
        """Compute the histograms of energies"""
        ens = -self.log_prob(x).detach().cpu().flatten()
        hist = torch.histogram(ens, bins=bins, density=True, range=range,
            weight=weights.detach().cpu().flatten() if weights is not None else None)[0]
        en_min, en_max = ens.min().item(), ens.max().item()
        if return_en:
            return hist, (en_min, en_max), ens
        else:
            return hist, (en_min, en_max)

    def compute_metrics(self, samples, weights=None, ref_samples=None, compute_standard_metrics=False,
                        skip_costly_metrics=True, bins=128, already_filtered=False):
        """Compute various metrics based on samples"""
        # Filter the chirality of the samples
        if not already_filtered:
            samples = self.filter_chirality(samples)
        # Get the standard statistics
        ret = super().compute_metrics(samples, weights=weights, ref_samples=ref_samples,
            compute_standard_metrics=compute_standard_metrics, skip_costly_metrics=skip_costly_metrics)
        # Get reference samples
        if ref_samples is None:
            ref_samples = self.sample((samples.shape[0],)).to(samples.device)
        # Compute the various histograms
        ref_en_hist, (en_min, en_max), en_ref = self.compute_energy_histograms(ref_samples, bins,
            return_en=True)
        samples_en_hist, _, en_samples = self.compute_energy_histograms(samples, bins,
            range=(en_min, en_max), return_en=True, weights=weights)
        # Compute the histogram distances
        ret['correlation_en_hist'] = compute_correlation(ref_en_hist, samples_en_hist).item()
        ret['intersection_en_hist'] = compute_intersection(ref_en_hist, samples_en_hist).item()
        # Compute the energy wasserstein distance
        ret['energy_w2'] = wasserstein_distance_1d(en_ref, en_samples,
            v_weights=weights.cpu() if weights is not None else None).item()
        # Compute distances in the Ramachandran space
        ret.update(self.compute_ramachandran_kld(
            *self.compute_psi_phi(samples), *self.compute_psi_phi(ref_samples), source_weights=weights
        ))
        return ret

    def plot_samples(self, ax, samples, weights=None, label="model", plot_type="ramachandran",
            bins=128, already_filtered=False):
        """Display the samples"""
        # Get the samples
        target_samples = self.sample((samples.shape[0],))
        if not already_filtered:
            samples = self.filter_chirality(samples)
        # Make the plot
        if plot_type in ["ramachandran", "marginal_angles", "fep_psi", "fep_phi"]:
            phi_source, psi_source = self.compute_psi_phi(samples)
            phi_target, psi_target = self.compute_psi_phi(target_samples)
            if weights is not None:
                weights = weights.detach().cpu().numpy()
            if plot_type == "ramachandran":
                plot_ramachandran_hist2d(ax, phi_source, psi_source, weights=weights)
            elif plot_type == "marginal_angles":
                plot_phi_psi_train_pred_hist1d(ax, phi_source, psi_source,
                    phi_target, psi_target, weights_source=weights)
            elif plot_type == "fep_phi":
                plot_free_energy_projection(ax, phi_source, weights=weights)
            elif plot_type == "fep_psi":
                plot_free_energy_projection(ax, psi_source, weights=weights)
        else:
            true_hist, (val_min, val_max) = self.compute_energy_histograms(target_samples, bins)
            model_hist = self.compute_energy_histograms(samples, bins, range=(val_min, val_max),
                weights=weights)[0]
            hist_pairwise_linespace = torch.linspace(val_min, val_max, bins)
            width = torch.min(torch.diff(hist_pairwise_linespace))
            ax.bar(hist_pairwise_linespace, true_hist, label='True', align='edge', alpha=0.5, width=width)
            ax.bar(hist_pairwise_linespace, model_hist, label=label, align='edge', alpha=0.5, width=width)
            ax.set_ylabel('Density')
            ax.set_xlabel('Energy')
            ax.legend()
