from typing import Literal

import ase
import numpy as np
import seaborn as sns
import torch
import torch.distributions as dists
import torch.nn as nn
from clease.calculator import attach_calculator
from clease.settings import CEBulk
from icet import ClusterExpansion
from matplotlib import pyplot as plt
from mchammer.calculators import ClusterExpansionCalculator
from tqdm import tqdm

from baselines.constants import K_B


class ClusterExpansionModel:
    """Cluster expansion energy model with Ising-like API.

    This wraps the existing ClusterExpansionCalculator-based helpers and exposes
    methods compatible with `LatticeIsingModel` so samplers and training code can
    treat alloys the same way as Ising lattices.
    """

    # NOTE: This class currently returns energy values in units of k_B T.
    def __init__(
        self,
        structure: ase.Atoms,
        cluster_expansion: ClusterExpansion,
    ) -> None:
        """Initialize the ClusterExpansionModel.

        Args:
            structure: ASE Atoms object representing the reference supercell structure.
            cluster_expansion: An icet ClusterExpansion object containing cluster
                definitions and effective cluster interactions (ECIs).

        Attributes:
            calc: ClusterExpansionCalculator instance for energy computations.
            atomic_numbers: Numpy array mapping CE occupation indices to atomic numbers.
            num_elements: Number of distinct chemical elements in the cluster expansion.
            energy_conversion_factor: Conversion factor from eV to unit
            num_sites: Number of lattice sites in the structure.
        """
        self.calc = ClusterExpansionCalculator(
            structure=structure,
            cluster_expansion=cluster_expansion,
            name="cluster_expansion",
            use_local_energy_calculator=True,
        )
        atomic_numbers = [
            ase.data.chemical_symbols.index(el)
            for el in cluster_expansion.chemical_symbols[0]
        ]
        self.atomic_numbers = np.array(atomic_numbers)
        self.num_elements = len(self.atomic_numbers)
        # Need to divide by T in units of T
        self.energy_conversion_factor = ase.units.eV / ase.units.kB

        # number of lattice sites is inferred from provided structure
        self.num_sites = len(structure)

    def _index_to_numbers(self, indices: np.ndarray) -> np.ndarray:
        """Convert CE occupation indices to atomic numbers.

        Args:
            indices: 1D numpy array of occupation indices (integers in [0, num_elements)).

        Returns:
            Numpy array of atomic numbers corresponding to the input indices.
        """
        return self.atomic_numbers[indices]

    def _energy(
        self, occupations: np.ndarray, temp: np.ndarray, mus: np.ndarray
    ) -> float:
        """Compute total energy for a single occupation configuration.

        This method computes the cluster expansion energy using the C++ backend and
        adds the grand potential contribution from chemical potentials.

        Args:
            occupations: 1D numpy array of occupation indices for each lattice site.
            temp: temperature in K (float).
            mus: 1D numpy array of chemical potentials (length num_elements).

        Returns:
            Total energy in units of k_B*T (float).
        """
        grand_potential = np.sum(mus[occupations])
        occupations = self._index_to_numbers(occupations)
        cv = self.calc.cpp_calc.get_cluster_vector(occupations)
        energy = (
            np.sum(cv * self.calc.cluster_expansion.parameters)
            * self.calc._property_scaling
        )
        return self.energy_conversion_factor / temp * (energy + grand_potential)

    def _energy_change(
        self,
        occupations: np.ndarray,
        temp: np.ndarray,
        mus: np.ndarray,
        site_index: int,
        new_occ: int,
    ) -> float:
        """Compute energy change for flipping a single site to a new occupation.

        Uses the efficient cluster vector change calculation from the C++ backend
        to avoid recomputing the full cluster expansion.

        Args:
            occupations: 1D numpy array of current occupation indices.
            temp: temperature in K (float).
            mus: 1D numpy array of chemical potentials (length num_elements).
            site_index: Index of the site to flip (integer).
            new_occ: New occupation index for the site (integer).

        Returns:
            Energy change (new - old) in units of k_B*T (float).
        """
        grand_potential_change = mus[new_occ] - mus[occupations[site_index]]
        occupations = self._index_to_numbers(occupations)
        new_occ = self._index_to_numbers(new_occ)
        cv = self.calc.cpp_calc.get_cluster_vector_change(
            occupations=occupations,
            flip_index=site_index,
            new_occupation=new_occ,
        )
        energy_change = (
            np.sum(cv * self.calc.cluster_expansion.parameters)
            * self.calc._property_scaling
        )
        return (
            self.energy_conversion_factor
            / temp
            * (energy_change + grand_potential_change)
        )

    def _all_energy_changes(
        self, occupations: np.ndarray, temp: np.ndarray, mus: np.ndarray
    ) -> np.ndarray:
        """Compute energy changes for flipping each site to every possible element.

        Args:
            occupations: 1D numpy array of current occupation indices.
            temp: temperature in K (float).
            mus: 1D numpy array of chemical potentials (length num_elements).

        Returns:
            2D numpy array with shape (num_sites, num_elements) where entry [i, j]
            is the energy change when site i is set to element j. Diagonal entries
            (no change) are computed but may not represent meaningful energy changes.
        """
        grand_potential_changes = mus[None, :] - mus[occupations[:, None]]
        occupations = self._index_to_numbers(occupations)
        energy_changes = np.zeros((len(occupations), self.num_elements))
        for site_index in range(len(occupations)):
            for elem_idx, new_occ in enumerate(self.atomic_numbers):
                if new_occ == occupations[site_index]:
                    continue
                cv = self.calc.cpp_calc.get_cluster_vector_change(
                    occupations=occupations,
                    flip_index=site_index,
                    new_occupation=new_occ,
                )
                energy_changes[site_index, elem_idx] = (
                    np.sum(cv * self.calc.cluster_expansion.parameters)
                    * self.calc._property_scaling
                )
        return (
            self.energy_conversion_factor
            / temp
            * (energy_changes + grand_potential_changes)
        )

    # ---- Public API following LatticeIsingModel conventions ----
    def init_sample(self, num_samples: int) -> torch.Tensor:
        """Randomly initialize occupations for `num_samples`.

        Returns a tensor of shape (num_samples, num_sites) with integer occupation
        indices in [0, num_elements).
        """
        samples = torch.randint(
            low=0,
            high=self.num_elements,
            size=(num_samples, self.num_sites),
            dtype=torch.long,
        )
        return samples

    def compute(
        self, x: torch.Tensor, temps: torch.Tensor, mus: torch.Tensor
    ) -> torch.Tensor:
        """Compute total energy for a batch of occupation configurations.

        x: (B, num_sites) integer tensor with occupation indices.
        temps: (B,) tensor of temperatures.
        mus: (B, num_elements) tensor of chemical potentials.
        Returns: (B,) tensor of energies.
        """
        energies = []
        device = x.device
        x_np = x.detach().cpu().numpy()
        temps_np = temps.detach().cpu().numpy()
        mus_np = mus.detach().cpu().numpy()
        for batch_idx in range(x_np.shape[0]):
            energy = self._energy(
                x_np[batch_idx], temps_np[batch_idx], mus_np[batch_idx]
            )
            energies.append(energy)
        return torch.tensor(energies, device=device)

    def get_energy(
        self,
        x: torch.Tensor,
        temps: torch.Tensor = None,
        fields: torch.Tensor = None,
        time: torch.Tensor = None,
    ) -> torch.Tensor:
        """Ising-compatible wrapper returning energy per sample.

        This method provides the same interface as LatticeIsingModel.get_energy()
        to ensure compatibility with existing samplers and training code.

        Args:
            x: (B, num_sites) tensor of occupation indices.
            temps: (B,) tensor of temperatures (not used in CE calculations).
            fields: (B, num_elements) tensor of chemical potentials.
            time: (B,) tensor of times; if provided, energy is scaled by time.

        Returns:
            (B,) tensor of energies in units of k_B*T.
        """
        mus_tensor = (
            fields
            if fields is not None
            else torch.zeros((x.size(0), self.num_elements), device=x.device)
        )
        energies = self.compute(x, temps=temps, mus=mus_tensor)
        if time is not None:
            return time * energies
        return energies

    def site_energy_change(
        self,
        x: torch.Tensor,
        temps: torch.Tensor,
        mus: torch.Tensor,
        site_index: torch.Tensor,
        new_occ: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy change when flipping a single site for each batch sample.

        Args:
            x: (B, num_sites) tensor of current occupation indices.
            temps: (B,) tensor of temperatures.
            mus: (B, num_elements) tensor of chemical potentials.
            site_index: (B,) tensor of site indices to flip for each sample.
            new_occ: (B,) tensor of new occupation indices for each sample.

        Returns:
            (B,) tensor of energy changes.
        """
        energy_changes = []
        device = x.device
        x_np = x.detach().cpu().numpy()
        temps_np = temps.detach().cpu().numpy()
        mus_np = mus.detach().cpu().numpy()
        site_idx_np = site_index.detach().cpu().numpy()
        new_occ_np = new_occ.detach().cpu().numpy()
        for batch_idx in range(x_np.shape[0]):
            energy_change = self._energy_change(
                x_np[batch_idx],
                temps_np[batch_idx],
                mus_np[batch_idx],
                int(site_idx_np[batch_idx]),
                int(new_occ_np[batch_idx]),
            )
            energy_changes.append(energy_change)
        return torch.tensor(energy_changes, device=device)

    def all_changes(
        self, x: torch.Tensor, temps: torch.Tensor, mus: torch.Tensor
    ) -> torch.Tensor:
        """Return energy change for changing each site to each possible element.

        Args:
            x: (B, num_sites) tensor of current occupation indices.
            temps: (B,) tensor of temperatures.
            mus: (B, num_elements) tensor of chemical potentials.

        Returns:
            (B, num_sites, num_elements) tensor of energy changes.
        """
        energy_changes = []
        device = x.device
        x_np = x.detach().cpu().numpy()
        temps_np = temps.detach().cpu().numpy()
        mus_np = mus.detach().cpu().numpy()
        for batch_idx in range(x_np.shape[0]):
            energy_changes.append(
                self._all_energy_changes(
                    x_np[batch_idx], temps_np[batch_idx], mus_np[batch_idx]
                )
            )
        return torch.tensor(energy_changes, device=device)

    def time_derivative(
        self,
        x: torch.Tensor,
        temps: torch.Tensor,
        mus: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Compute time derivative dU/dt for the energy function.

        For cluster expansion we assume U_t(x) = t * U(x), so dU/dt = U(x).
        This convention matches the Ising model where time scaling is handled
        externally by the sampler or AIS schedule.

        Args:
            x: (B, num_sites) tensor of occupation indices.
            temps: (B,) tensor of temperatures.
            mus: (B, num_elements) tensor of chemical potentials.
            time: (B,) tensor of times.

        Returns:
            (B,) tensor with dU/dt for each sample (same units as get_energy).
        """
        return self.compute(x, temps=temps, mus=mus)

    def get_energy_change_per_site(
        self,
        x: torch.Tensor,
        temps: torch.Tensor = None,
        fields: torch.Tensor = None,
        time: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute energy change for flipping each site to every element.

        Args:
            x: (B, num_sites) tensor of occupation indices.
            temps: (B,) tensor of temperatures (unused but kept for compatibility).
            fields: (B, num_elements) tensor treated as chemical potentials.
            time: (B,) tensor of times (unused but kept for compatibility).
        """
        mus = (
            fields
            if fields is not None
            else torch.zeros((x.size(0), self.num_elements), device=x.device)
        )
        changes = self.all_changes(x, temps, mus)
        if time is not None:
            return changes * time[:, None, None]
        return changes

    def single_partial(
        self,
        x: torch.Tensor,
        temps: torch.Tensor,
        fields: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample partial derivative with respect to time.

        This is a lightweight helper that maps the sampler's expected signature
        to the CE model's time_derivative method. For cluster expansion, we reuse
        the time_derivative behavior which returns U(x).

        Args:
            x: (B, num_sites) tensor of occupation indices.
            temps: (B,) tensor of temperatures (unused but kept for compatibility).
            fields: (B, num_elements) tensor treated as chemical potentials.
            time: (B,) tensor of times (unused but kept for compatibility).

        Returns:
            (B,) tensor of partial derivatives dU/dt.
        """
        mus = (
            fields
            if fields is not None
            else torch.zeros((x.size(0), self.num_elements), device=x.device)
        )
        return self.time_derivative(x, temps, mus, time)

    def partial_t(
        self,
        x: torch.Tensor,
        temps: torch.Tensor,
        fields: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample partial derivatives using vmap-like iteration.

        This implements a simple loop-based vmap to produce the same output shape
        that samplers expect. For large batches this is not optimized but maintains
        API consistency with LatticeIsingModel.

        Args:
            x: (B, num_sites) tensor of occupation indices.
            temps: (B,) tensor of temperatures (unused here).
            fields: (B, num_elements) tensor of chemical potentials.
            time: (B,) tensor of times (unused except for shape compatibility).

        Returns:
            (B,) tensor of per-sample partial derivatives dU/dt.
        """
        B, L = x.size()
        x, temps, fields, time = (
            x.view(B, 1, L),
            temps.view(-1, 1),
            fields.view(-1, 1),
            time.view(-1, 1),
        )
        # vmap-like behavior: compute per-sample derivative using single_partial
        out = []
        for i in range(B):
            out.append(
                self.single_partial(
                    x[i].squeeze(0),
                    temps[i].squeeze(0),
                    fields[i].squeeze(0),
                    time[i].squeeze(0),
                )
            )
        return torch.stack(out)

    def step(
        self,
        x: torch.Tensor,
        temps: torch.Tensor,
        fields: torch.Tensor,
        time: torch.Tensor = None,
        criterion: str = "glauber",
    ) -> torch.Tensor:
        """Sampler-compatible step method placeholder.

        The ClusterExpansionModel does not implement its own Gibbs sampler.
        The sampling logic (proposals and acceptance) is expected to be handled
        by external sampler classes (e.g., PerDimGibbsSampler). This method
        exists so the model conforms to the same API as LatticeIsingModel.

        Args:
            x: (B, num_sites) tensor of occupation indices.
            temps: (B,) tensor of temperatures (accepted for compatibility).
            fields: (B, num_elements) tensor of chemical potentials.
            time: (B,) tensor of times (optional, accepted for compatibility).
            criterion: Sampling criterion string (accepted for compatibility).

        Returns:
            The input tensor x unchanged (no sampling performed).
        """
        return x

    def forward(
        self,
        x: torch.Tensor,
        temps: torch.Tensor,
        fields: torch.Tensor,
        time: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass method for PyTorch-like API compatibility.

        This method simply delegates to get_energy() and is provided for
        consistency with PyTorch nn.Module conventions.

        Args:
            x: (B, num_sites) tensor of occupation indices.
            temps: (B,) tensor of temperatures.
            fields: (B,) tensor of chemical potentials.
            time: (B,) tensor of times (optional).

        Returns:
            (B,) tensor of energies in units of k_B*T.
        """
        # TODO: generalize for more than 2 elements
        mus = torch.stack([torch.zeros_like(fields), fields], dim=1)
        return self.get_energy(x, temps, mus, time)

    def __call__(self, *args: torch.Tensor, **kwds: torch.Tensor) -> torch.Tensor:
        return self.forward(*args, **kwds)


class AuCuGibbsSampler(nn.Module):
    """Gibbs sampler for Au-Cu binary alloy model with per-dimension updates.

    This sampler performs single-site flips (0 <-> 1) for binary alloys,
    similar to PerDimGibbsSampler for Ising models.
    """

    def __init__(self, data_dim: int, batch_size: int, rand: bool = False) -> None:
        """Initialize the Gibbs sampler.

        Args:
            data_dim: The total number of lattice sites.
            batch_size: The batch size.
            rand: Whether to use random updates (if True) or sequential updates (if False).
                Defaults to False.
        """
        super().__init__()
        self.data_dim = data_dim
        self.batch_size = batch_size
        self.num_changes = torch.zeros((data_dim,))
        self._i = 0
        self._ar = 0.0
        self._hops = 0.0
        self.rand = rand
        if self.rand:
            self.register_buffer("rand_logits", torch.zeros((self.data_dim,)))

    def step(
        self,
        x: torch.Tensor,
        model: "AuCuAlloyModel",
        temps: torch.Tensor = None,
        fields: torch.Tensor = None,
        time: torch.Tensor = None,
        criterion: Literal["glauber", "metropolis"] = "glauber",
    ) -> torch.Tensor:
        """Performs a single step of the Gibbs sampling algorithm.

        Args:
            x: The input sample with shape (B, L) with binary occupations {0, 1}.
            model: The AuCuAlloyModel instance.
            temps: The temperature with shape (B,).
            fields: The field (chemical potential) with shape (B,).
            time: The time with shape (B,) (optional, for AIS).
            criterion: The sampling criterion, either "glauber" or "metropolis".
                Defaults to "glauber".

        Returns:
            The updated sample tensor.
        """
        sample = x
        # Compute log probability of current state (negative free energy)
        lp_keep = -model(sample, temps, fields, time).squeeze()

        # Propose a flip at one site
        if self.rand:
            changes = dists.OneHotCategorical(
                logits=self.rand_logits
            ).sample(  # random proposal
                (x.size(0),)
            )
        else:
            changes = torch.zeros((x.size(0), self.data_dim), device=x.device)
            changes[:, self._i] = 1.0

        # Flip the selected site: 0 -> 1, 1 -> 0
        sample_change = (1.0 - changes) * sample + changes * (1.0 - sample)

        # Compute log probability of new state
        lp_change = -model(sample_change, temps, fields, time).squeeze()
        lp_update = lp_change - lp_keep  # log probability difference

        if criterion == "glauber":
            # Glauber dynamics: accept with probability sigmoid(lp_update)
            update_dist = dists.Bernoulli(logits=lp_update)
            updates = update_dist.sample()
        else:
            # Metropolis-Hastings: accept if rand < exp(lp_update)
            updates = (
                torch.rand_like(lp_update)
                < torch.exp(lp_update).clamp(min=0.0, max=1.0)
            ).float()

        # Apply updates
        sample = (
            sample_change * updates[:, None] + sample * (1.0 - updates[:, None])
        ).type_as(x)
        self.num_changes[self._i] = updates.mean()
        self._i = (self._i + 1) % self.data_dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def forward(
        self, x: torch.Tensor, model: "AuCuAlloyModel", **kwargs
    ) -> torch.Tensor:
        """Performs a single step of the Gibbs sampling algorithm.

        Args:
            x: The input sample with shape (B, L).
            model: The AuCuAlloyModel instance.
            kwargs: Additional keyword arguments.

        Returns:
            The updated sample tensor.
        """
        return self.step(x, model, **kwargs)


class AuCuAlloyModel:
    """Au-Cu binary alloy energy model with cluster expansion backend.

    This class provides similar API to ClusterExpansionModel but is specialized
    for binary Au-Cu alloys using the CLEASE framework.
    """

    def __init__(
        self,
        structure: ase.Atoms = None,
        settings: CEBulk = None,
        eci: ClusterExpansion | None = None,
    ) -> None:
        """Initialize the AuCuAlloyModel.

        Args:
            structure: ASE Atoms object (for compatibility with ClusterExpansionModel).
            settings: CEBulk object defining the alloy configuration.
            eci: Cluster expansion object (for compatibility).

        Attributes:
            settings: CEBulk object for alloy configuration.
            atoms: ASE Atoms object with attached calculator.
            k_b: Boltzmann constant.
            num_sites: Number of lattice sites (equivalent to Nz).
            num_elements: Number of elements (always 2 for Au-Cu).
            U_cu: Reference energy for pure Cu configuration.
            U_au: Reference energy for pure Au configuration.
            bias: Bias tensor for initialization.
            init_dist: Initial distribution for sampling.
            temperature: Temperature used for energy scaling.
        """
        # Store parameters for API compatibility
        self.num_elements = 2  # Binary Au-Cu alloy

        # Initialize from config (legacy path)
        self.atoms = attach_calculator(settings, atoms=structure, eci=eci)
        self.num_sites = len(structure)

        # Common initialization
        self.k_b = K_B
        self.Nz = self.num_sites  # Legacy attribute name

        # Calculate reference energies for pure compositions
        self.atoms.numbers = np.ones(self.num_sites) * 29
        self.U_cu = self.atoms.get_potential_energy()  # potential energy of pure Cu
        print("PURE CU ENERGY:", self.U_cu)

        self.atoms.numbers = np.ones(self.num_sites) * 79
        self.U_au = self.atoms.get_potential_energy()  # potential energy of pure Au
        print("PURE AU ENERGY:", self.U_au)
        self.bias = torch.ones(self.num_sites).float() * 0.0
        self.init_dist = dists.Bernoulli(logits=2 * self.bias)

        # Initialize Gibbs sampler for MCMC
        self.batch_size = 1250  # Default batch size, can be overridden
        self.rand = True  # Use random site selection by default
        self.sampler = AuCuGibbsSampler(
            data_dim=self.num_sites, batch_size=self.batch_size, rand=self.rand
        )

    def set_scaled_positions(self, scaled_positions: torch.Tensor) -> None:
        """Sets the scaled positions of the atoms.

        Args:
            scaled_positions: A torch tensor representing the scaled positions.
        """
        self.atoms.set_scaled_positions(scaled_positions.detach().cpu().numpy())

    def init_sample(self, num_samples: int) -> torch.Tensor:
        """Initializes samples from the initial distribution.

        Args:
            num_samples: The number of samples to initialize.

        Returns:
            A torch tensor representing the initialized samples.
        """
        return self.init_dist.sample((num_samples,))

    def compute(self, x: torch.Tensor, mus: torch.Tensor) -> torch.Tensor:
        """Compute total energy for a batch of occupation configurations.

        Args:
            x: (B, num_sites) tensor with binary occupations {0,1}.
            mus: (B, num_elements) tensor of chemical potentials (unused in this implementation).

        Returns:
            (B,) tensor of energies.
        """
        return self.get_energy(x)

    def get_concentrations(self, lattices: torch.Tensor) -> torch.Tensor:
        """Calculates the concentrations of Au in the lattices.

        Args:
            lattices: A torch tensor representing the lattices.

        Returns:
            A torch tensor representing the concentrations of Au in the lattices.
        """
        return torch.sum(lattices, dim=1) / (self.Nz)

    def get_energy(self, lattices: torch.Tensor) -> torch.Tensor:
        """Calculates the energy for each lattice.

        Args:
            lattices: A torch tensor representing the lattices with atom types {0, 1} of shape
                (B, L).

        Returns:
            A torch tensor representing the energy of the lattices.
        """
        # TODO parallelize this; it is the bottleneck
        energies = torch.zeros(lattices.shape[0]).to(lattices.device)
        for lattice_num in range(lattices.shape[0]):
            energies[lattice_num] = self.get_cluster_energy(lattices[lattice_num, :])
        return energies

    def get_free_energies(
        self, lattices: torch.Tensor, temps: torch.Tensor, fields: torch.Tensor
    ) -> torch.Tensor:
        """Get the free energies for the given inputs in terms of k_B*T.

        Args:
            lattices: A torch tensor representing the lattices with atom types {0, 1} of shape
                (B, L).
            temps: A torch tensor representing the temperatures with shape (B,).
            fields: A torch tensor representing the fields with shape (B,).

        Returns:
            The free energies.
        """
        energies = self.get_energy(lattices)  # shape (B,)
        lattices_factor = (lattices * 2.0 - 1.0).sum(
            dim=1
        )  # (B,), convert to {-1, 1} first for binary alloy, so that the sum is over {-L, ..., L}
        return (energies - fields * lattices_factor) / (
            temps * K_B
        )  # shape (B,), free energies

    def get_cluster_energy(self, lattice: torch.Tensor) -> float:
        """Calculates the energy of a cluster in the lattice.

        Args:
            lattice: A torch tensor representing the lattice.

        Returns:
            A float representing the energy of the cluster.
        """
        # Convert from {0,1} to {Cu = 29, Au = 79}
        self.atoms.numbers = ((lattice * 50) + 29).int().detach().cpu().numpy()
        energy_t = self.atoms.get_potential_energy()
        Au_conc = torch.sum(lattice, dim=0) / (self.Nz)  # Au concentration
        # Returned energy is the formation energy of the cluster at standard state
        return energy_t - (1 - Au_conc) * (self.U_cu) - (Au_conc) * (self.U_au)

    def get_hull_fe_energies(self, lattices: torch.Tensor, cpot: float) -> np.array:
        """Calculates the energy above hull of the lattices.

        Args:
            lattices: A torch tensor representing the lattices.
            cpot: A float representing the chemical potential.

        Returns:
            A numpy array representing the hull formation energies of the lattices.
        """
        size = lattices.shape[0]
        out = np.zeros(size)

        def get_min_fe(ch_pot: float) -> float:
            hull = {}
            hull[0.0] = 0.0

            hull[0.25] = -0.0341671942368
            hull[0.50] = -0.0361824880
            hull[0.75] = -0.018754351039661
            hull[1.0] = 0.0
            if ch_pot < hull[0.25] * 2.0:
                return 0.0
            if ch_pot < (hull[0.50] - hull[0.25]) * 2.0:
                return -0.0341671942368 * self.Nz - 2 * ch_pot * (0.25 * self.Nz)
            if ch_pot < (hull[0.75] - hull[0.5]) * 2.0:
                return -0.0361824880 * self.Nz - 2 * ch_pot * (0.5 * self.Nz)
            if ch_pot < (hull[1.0] - hull[0.75]) * 2.0:
                return -0.01875435 * self.Nz - 2 * ch_pot * (0.75 * self.Nz)
            return -2 * ch_pot * self.Nz

        act_min = get_min_fe(cpot)
        Au_conc = self.get_concentrations(lattices)
        energies = self.get_energy(lattices)
        vals = energies - (2 * cpot * Au_conc) * self.Nz
        for i in range(size):
            curr_val = vals[i].cpu().detach().numpy()
            out[i] = curr_val - act_min
        return out

    def as_ase_atoms(self, lattice: torch.Tensor) -> ase.Atoms:
        """Converts a lattice to an ASE atoms object.

        Args:
            lattice: A torch tensor representing the lattice.

        Returns:
            An ASE atoms object representing the lattice.
        """
        if lattice.ndimension() > 1:
            raise ValueError("Only one lattice can be converted at a time.")
        atoms = self.atoms.copy()
        atoms.set_atomic_numbers(((lattice * 50) + 29).int().cpu().detach().numpy())
        return atoms

    def plot_energies(self, energies: torch.Tensor, save_path: str) -> None:
        """Plot the energies and save the plot to the specified path.

        Args:
            energies: The energies to plot.
            save_path: The path to save the plot.
        """
        sns.kdeplot(energies, fill=True)
        plt.savefig(save_path)
        plt.close()

    def site_energy_change(
        self,
        x: torch.Tensor,
        mus: torch.Tensor,
        site_index: torch.Tensor,
        new_occ: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy change for flipping a single site for each batch sample.

        Args:
            x: (B, num_sites) tensor of current binary occupations {0,1}.
            mus: (B, num_elements) tensor of chemical potentials (unused).
            site_index: (B,) tensor of site indices to flip.
            new_occ: (B,) tensor of new occupation values {0,1}.

        Returns:
            (B,) tensor of energy changes.
        """
        # For binary alloy, compute energy change by evaluating both states
        energy_changes = []
        device = x.device

        for batch_idx in range(x.size(0)):
            # Current energy
            current_energy = self.get_cluster_energy(x[batch_idx])

            # Create modified configuration
            x_modified = x[batch_idx].clone()
            x_modified[site_index[batch_idx]] = new_occ[batch_idx]

            # New energy
            new_energy = self.get_cluster_energy(x_modified)

            energy_changes.append(new_energy - current_energy)

        return torch.tensor(energy_changes, device=device)

    def all_changes(self, x: torch.Tensor, mus: torch.Tensor) -> torch.Tensor:
        """Compute energy changes for flipping each site to every possible element.

        Args:
            x: (B, num_sites) tensor of current binary occupations {0,1}.
            mus: (B, num_elements) tensor of chemical potentials (unused).

        Returns:
            (B, num_sites, num_elements) tensor of energy changes.
        """
        B, num_sites = x.size()
        energy_changes = torch.zeros(B, num_sites, 2, device=x.device)

        for batch_idx in range(B):
            current_energy = self.get_cluster_energy(x[batch_idx])

            for site_idx in range(num_sites):
                for new_occ in [0, 1]:
                    if x[batch_idx, site_idx] != new_occ:
                        x_modified = x[batch_idx].clone()
                        x_modified[site_idx] = new_occ
                        new_energy = self.get_cluster_energy(x_modified)
                        energy_changes[batch_idx, site_idx, new_occ] = (
                            new_energy - current_energy
                        )

        return energy_changes

    def time_derivative(self, x: torch.Tensor, mus: torch.Tensor) -> torch.Tensor:
        """Compute time derivative dU/dt for the energy function.

        Args:
            x: (B, num_sites) tensor of binary occupations {0,1}.
            mus: (B, num_elements) tensor of chemical potentials.

        Returns:
            (B,) tensor with dU/dt for each sample.
        """
        return self.compute(x, mus)

    def get_energy_change_per_site(
        self,
        x: torch.Tensor,
        temps: torch.Tensor = None,
        fields: torch.Tensor = None,
        time: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute energy change for flipping each site to every element.

        Args:
            x: (B, num_sites) tensor of binary occupations {0,1}.
            temps: (B,) tensor of temperatures (unused).
            fields: (B,) tensor of fields (treated as chemical potential for Au).
            time: (B,) tensor of times (unused).

        Returns:
            (B, num_sites, num_elements) tensor of energy changes.
        """
        # Convert fields to chemical potentials format
        if fields is not None:
            mus = torch.stack(
                [torch.zeros_like(fields), fields], dim=1
            )  # [mu_Cu, mu_Au]
        else:
            mus = torch.zeros((x.size(0), 2), device=x.device)

        changes = self.all_changes(x, mus)
        if time is not None:
            return changes * time[:, None, None]
        return changes

    def single_partial(
        self,
        x: torch.Tensor,
        temps: torch.Tensor,
        fields: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample partial derivative with respect to time.

        Args:
            x: (B, num_sites) tensor of binary occupations {0,1}.
            temps: (B,) tensor of temperatures (unused).
            fields: (B,) tensor of fields (treated as chemical potentials).
            time: (B,) tensor of times (unused).

        Returns:
            (B,) tensor of partial derivatives dU/dt.
        """
        mus = (
            torch.stack([torch.zeros_like(fields), fields], dim=1)
            if fields is not None
            else torch.zeros((x.size(0), 2), device=x.device)
        )
        return self.time_derivative(x, mus)

    def partial_t(
        self,
        x: torch.Tensor,
        temps: torch.Tensor,
        fields: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample partial derivatives using vmap-like iteration.

        Args:
            x: (B, num_sites) tensor of binary occupations {0,1}.
            temps: (B,) tensor of temperatures.
            fields: (B,) tensor of fields.
            time: (B,) tensor of times.

        Returns:
            (B,) tensor of per-sample partial derivatives dU/dt.
        """
        B, L = x.size()
        x, temps, fields, time = (
            x.view(B, 1, L),
            temps.view(-1, 1),
            fields.view(-1, 1),
            time.view(-1, 1),
        )

        out = []
        for i in range(B):
            out.append(
                self.single_partial(
                    x[i].squeeze(0),
                    temps[i].squeeze(0),
                    fields[i].squeeze(0),
                    time[i].squeeze(0),
                )
            )
        return torch.stack(out)

    def step(
        self,
        x: torch.Tensor,
        temps: torch.Tensor,
        fields: torch.Tensor,
        time: torch.Tensor = None,
        criterion: Literal["glauber", "metropolis"] = "glauber",
    ) -> torch.Tensor:
        """Perform a single step of the Gibbs sampling algorithm.

        Args:
            x: (B, num_sites) tensor of binary occupations {0,1}.
            temps: (B,) tensor of temperatures.
            fields: (B,) tensor of fields (chemical potential for Au).
            time: (B,) tensor of times (optional, for AIS).
            criterion: The sampling criterion, either "glauber" or "metropolis".
                Defaults to "glauber".

        Returns:
            The updated sample tensor.
        """
        return self.sampler.step(x, self, temps, fields, time, criterion)

    def generate_samples(
        self,
        n_samples: int,
        temps: torch.Tensor,
        fields: torch.Tensor,
        time: torch.Tensor = None,
        gt_steps: int = 1000000,
        rand: bool = False,
        starting_samples: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate samples using Gibbs sampling.

        Args:
            n_samples: The number of samples to generate in each batch, aka batch size.
            temps: Temperature torch tensor with length n_samples.
            fields: Field (chemical potential) torch tensor with length n_samples.
            gt_steps: The number of Gibbs sampling steps. Defaults to 1000000.
            rand: Whether to use random updates or sequential updates. Defaults to False.
            starting_samples: Optional starting samples for the Gibbs sampler.

        Returns:
            The generated samples.
        """
        if starting_samples is not None:
            samples = starting_samples
        else:
            samples = self.init_sample(n_samples)

        # Update sampler's random flag if needed
        if rand != self.sampler.rand:
            self.sampler.rand = rand
            if rand and not hasattr(self.sampler, "rand_logits"):
                self.sampler.register_buffer(
                    "rand_logits", torch.zeros((self.num_sites,))
                )

        print(f"Generating {n_samples:d} samples from {self!s}")
        for _ in tqdm(range(gt_steps)):
            samples = self.sampler.step(
                samples, self, temps, fields, time=time
            ).detach()
        return samples.detach().cpu()

    def forward(
        self,
        x: torch.Tensor,
        temps: torch.Tensor,
        fields: torch.Tensor,
        time: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass method for PyTorch-like API compatibility.

        Args:
            x: (B, num_sites) tensor of binary occupations {0,1}.
            temps: (B,) tensor of temperatures.
            fields: (B,) tensor of fields.
            time: (B,) tensor of times (optional).

        Returns:
            (B,) tensor of free energies.
        """
        return self.get_free_energies(x, temps, fields)

    def __call__(
        self,
        lattices: torch.Tensor,
        temps: torch.Tensor,
        fields: torch.Tensor,
        time: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calls the `get_free_energies` method.

        Args:
            lattices: A torch tensor representing the lattices with atom types {0, 1} of shape
                (B, L).
            temps: A torch tensor representing the temperatures with shape (B,).
            fields: A torch tensor representing the fields with shape (B,).

        Returns:
            The free energies normalized by k_B*T (energy/(k_B*T)).
        """
        return self.forward(lattices, temps, fields, time)
