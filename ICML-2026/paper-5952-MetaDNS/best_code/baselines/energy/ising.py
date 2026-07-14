"""Energy model and energy-based sampling for Ising model."""

from typing import Literal

import igraph as ig
import numpy as np
import seaborn as sns
import torch
import torch.distributions as dists
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.func import jacfwd, vmap
from tqdm import tqdm


# TODO: Generalize base energy distribution
class LatticeIsingModel(nn.Module):
    """Represents an Ising model on a lattice."""

    def __init__(
        self,
        dim: int,
        init_sigma: float = 1,
        init_field: float = 0.0,
        learn_G: bool = False,
        learn_sigma: bool = False,
        learn_field: bool = False,
        lattice_dim: int = 2,
        batch_size: int = 1250,
        n_samples: int = 2000,
        rand: bool = True,
    ) -> None:
        """Initialize the Ising model.

        Args:
            dim: The length of the lattice in each dimension.
            init_sigma: The initial value for the interaction strength. Defaults to 0.15.
            init_field: The initial value for the field strength. Defaults to 0.0.
            learn_G: Whether to learn the adjacency matrix. Defaults to False.
            learn_sigma: Whether to learn the interaction strength. Defaults to False.
            learn_field: Whether to learn the field strength. Defaults to False.
            lattice_dim: The number of dimensions. Defaults to 2.
            batch_size: The batch size. Defaults to 1250.
            n_samples: The number of samples. Defaults to 2000.
            rand: Whether to use random updates. Defaults to True.

        Attributes:
            G (nn.Parameter): The adjacency matrix.
            sigma (nn.Parameter): The interaction strength.
            bias (nn.Parameter): The field strength.
            init_dist (dists.Bernoulli): The initial distribution.
            data_dim (int): The dimension of the data.
            n_samples (int): The number of samples.
            rand (bool): Whether to use random updates.
        """
        super().__init__()
        g = ig.Graph.Lattice(dim=[dim] * lattice_dim, circular=True)  # boundary conditions
        A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()
        self.G = nn.Parameter(torch.tensor(A).float(), requires_grad=learn_G)
        self.sigma = nn.Parameter(
            torch.tensor(init_sigma).float(), requires_grad=learn_sigma
        )  # interaction strength
        self.field = nn.Parameter(
            torch.ones((dim**lattice_dim,)).float() * init_field,
            requires_grad=learn_field,
        )  # field
        self.init_dist = dists.Bernoulli(logits=2 * self.field)
        self.data_dim = dim**lattice_dim
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.rand = rand
        self.sampler = PerDimGibbsSampler(
            data_dim=self.data_dim, batch_size=batch_size, rand=self.rand
        )

    def init_sample(self, num_samples: int) -> torch.Tensor:
        """Initializes samples from the initial distribution.

        Args:
            num_samples: The number of samples to initialize.

        Returns:
            A torch tensor representing the initialized samples.
        """
        return self.init_dist.sample((num_samples,))

    @property
    def J(self) -> torch.Tensor:
        """Calculate the interaction matrix, J.

        Returns:
            The interaction matrix.
        """
        return self.G * self.sigma

    def get_energy(
        self,
        x: torch.Tensor,
        temps: torch.Tensor = None,
        fields: torch.Tensor = None,
        time: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculates the energy for all samples in terms of T.

        Args:
            x: The input tensor with shape atom types {0, 1} of shape (B, D).
            temps: A torch tensor representing the temperatures with shape (B,).
            fields: A torch tensor representing the fields with shape (B,).
            time: A torch tensor representing the time with shape (B,).

        Returns:
            A torch tensor representing the energy.
        """
        if x.ndim > 2:
            print(f"Reshaping x from {x.size()} to {(x.size(0), -1)}")
            x = x.reshape(x.size(0), -1)

        x = (2 * x) - 1  # from {0,1} to {-1,1}
        x = x.type_as(self.J)
        xg = x @ self.J
        xgx = (xg * x).sum(-1)  # energy with interactions but without bias

        if fields is not None:
            b = (fields[:, None] * x).sum(-1)
        else:
            b = (self.field[None, :] * x).sum(-1)  # add bias

        denominator = temps if temps is not None else torch.Tensor([1.0], device=x.device)

        if time is not None:
            assert time.size(0) == x.size(0)
            return time * (-(0.5 * xgx + b) / denominator)  # time-dependent energy for AIS
        return -(0.5 * xgx + b) / denominator

    def get_energy_change_per_site(
        self,
        x: torch.Tensor,
        temps: torch.Tensor = None,
        fields: torch.Tensor = None,
        time: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculates the energy change per site.

        Args:
            x: The input tensor with shape atom types {0, 1} of shape (B, D).
            temps: A torch tensor representing the temperatures with shape (B,).
            fields: A torch tensor representing the fields with shape (B,).
            time: A torch tensor representing the time with shape (B,).

        Returns:
            A torch tensor representing the energy change per site.
        """
        if x.ndim > 2:
            print(f"Reshaping x from {x.size()} to {(x.size(0), -1)}")
            x = x.reshape(x.size(0), -1)

        x_idx = x.clone()

        all_energy_changes = torch.zeros(*x.size(), 2, device=x.device)

        x = (2 * x) - 1  # from {0,1} to {-1,1}
        x = x.type_as(self.J)
        xg = x @ self.J
        interaction_change = 2 * (x * xg)  # Energy change due to interactions

        # Compute field energy change
        if fields is not None:
            field_change = 2 * fields[:, None] * x
        else:
            field_change = 2 * self.field[None, :] * x

        # Combine interaction and field changes
        energy_change = interaction_change + field_change

        # Adjust for temperature if provided
        if temps is not None:
            energy_change /= temps[:, None]

        if time is not None:
            assert time.size(0) == x.size(0)
            energy_change *= time[:, None]

        # Populate corresponding energy changes for flipping
        # x_idx == 0.0 corresponds to flipping to 1.0
        # x_idx == 1.0 corresponds to flipping to 0.0
        # Spins that are not flipped are 0.0
        # all_energy_changes[x_idx == 0, 1] = energy_change[x_idx == 0]
        # all_energy_changes[x_idx == 1, 0] = energy_change[x_idx == 1]
        x_idx_inv = (1.0 - x_idx).long()
        all_energy_changes.scatter_(2, x_idx_inv[..., None], energy_change[..., None])
        return all_energy_changes

    def single_partial(
        self, x: torch.Tensor, temps: torch.Tensor, fields: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the partial derivative with respect to time for a single sample.

        Args:
            x: The input tensor with shape atom types {0, 1} of shape (D,).
            temps: A torch tensor representing the temperature (scalar).
            fields: A torch tensor representing the field (scalar).
            time: A torch tensor representing the time (scalar).

        Returns:
            The partial derivative with respect to time.
        """
        return jacfwd(self.get_energy, 3)(x, temps, fields, time).squeeze()

    def partial_t(
        self,
        x: torch.Tensor,
        temps: torch.Tensor,
        fields: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the partial derivative with respect to time.

        Args:
            x: The input tensor with shape atom types {0, 1} of shape (B, L).
            temps: A torch tensor representing the temperatures with shape (B,).
            fields: A torch tensor representing the fields with shape (B,).
            time: A torch tensor representing the time with shape (B,).

        Returns:
            The partial derivative with respect to time.
        """
        # return jacfwd(self.get_energy, 3)(x, temps, fields, time).diag()
        B, L = x.size()
        x, temps, fields, time = (
            x.view(B, 1, L),
            temps.view(-1, 1),
            fields.view(-1, 1),
            time.view(-1, 1),
        )
        return vmap(
            self.single_partial, in_dims=(0, 0, 0, 0), out_dims=(0), randomness="different"
        )(x, temps, fields, time)

    def plot_scores(self, scores: torch.Tensor, save_path: str) -> None:
        """Plot the scores and save the plot to the specified path.

        Args:
            scores: The scores to plot.
            save_path: The path to save the plot.
        """
        sns.kdeplot(scores, fill=True)
        plt.savefig(save_path)
        plt.close()

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
            x: The input tensor with shape atom types {0, 1} of shape (B, L).
            temps: A torch tensor representing the temperatures with shape (B,).
            fields: A torch tensor representing the fields with shape (B,).
            time: A torch tensor representing the time with shape (B,).
            criterion: The sampling criterion, either "glauber" or "metropolis".
                Defaults to "glauber".
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
            fields: Field torch tensor with length n_samples.
            gt_steps: The number of Gibbs sampling steps. Defaults to 1000000.
            rand: Whether to use random updates or sequential updates. Defaults to False.
            starting_samples: Optional starting samples for the Gibbs sampler.

        Returns:
            The generated samples.
        """
        if starting_samples is not None:
            samples = starting_samples.type_as(self.J)
        else:
            samples = self.init_sample(n_samples)
        print(f"Generating {n_samples:d} samples from {self!s:s}")
        for _ in tqdm(range(gt_steps)):
            samples = self.sampler.step(samples, self, temps, fields, time=time).detach()
        return samples.detach().cpu()

    def forward(
        self,
        x: torch.Tensor,
        temps: torch.Tensor,
        fields: torch.Tensor,
        time: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calls the `get_energy` method to compute the energy of the input tensor.

        Args:
            x: The input tensor with shape atom types {0, 1} of shape (B, L).
            temps: A torch tensor representing the temperatures with shape (B,).
            fields: A torch tensor representing the fields with shape (B,).
            time: A torch tensor representing the time with shape (B,).
        """
        return self.get_energy(x, temps, fields, time)


class PerDimGibbsSampler(nn.Module):
    """A class representing a Gibbs sampler for the Ising model with per-dimension updates."""

    def __init__(self, data_dim: int, batch_size: int, rand: bool = False) -> None:
        """Initialize the Gibbs sampler.

        Args:
            data_dim: The total number of lattice sites.
            batch_size: The batch size.
            rand: Whether to use random updates (if True) or sequential updates (if False).
            Defaults to False.

        Attributes:
            data_dim (int): The total number of lattice sites.
            changes (torch.Tensor): A tensor to store the average changes made in each dimension.
            change_rate (float): The rate of change in the model.
            p (nn.Parameter): A learnable parameter representing the probabilities of
                each dimension.
            _i (int): The index of the current dimension being updated.
            _ar (float): The acceptance rate of the updates.
            _hops (float): The average number of changes made per sample.
            _phops (float): The previous average number of changes made per sample.
            rand (bool): Whether to use random updates (if True) or sequential updates (if False).
        """
        super().__init__()
        self.data_dim = data_dim
        self.batch_size = batch_size
        self.num_changes = torch.zeros((data_dim,))
        # self.change_rate = 0.0
        # self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._ar = 0.0
        self._hops = 0.0
        # self._phops = 1.0
        self.rand = rand
        if self.rand:
            self.register_buffer("rand_logits", torch.zeros((self.data_dim,)))
        else:
            self.register_buffer("changes", torch.zeros((self.batch_size, self.data_dim)))

    def step(
        self,
        x: torch.Tensor,
        model: LatticeIsingModel,
        temps: torch.Tensor = None,
        fields: torch.Tensor = None,
        time: torch.Tensor = None,
        criterion: Literal["glauber", "metropolis"] = "glauber",
    ) -> torch.Tensor:
        """Performs a single step of the Gibbs sampling algorithm.

        Args:
            x: The input sample with shape atom types {0, 1} of shape (B, L).
            model: The Ising model.
            temps: The temperature with shape (B,).
            fields: The field with shape (B,).
            time: The time with shape (B,).
            criterion: The sampling criterion, either "glauber" or "metropolis".
                Defaults to "glauber".
        """
        sample = x
        lp_keep = -model(sample, temps, fields, time).squeeze()  # log_p of current sample
        if self.rand:
            self.changes = dists.OneHotCategorical(
                logits=self.rand_logits
            ).sample(  # random proposal
                (x.size(0),)
            )
        else:
            self.changes = torch.zeros_like(self.changes)
            self.changes[:, self._i] = 1.0
        sample_change = (1.0 - self.changes) * sample + self.changes * (1.0 - sample)

        lp_change = -model(sample_change, temps, fields, time).squeeze()  # log_p of new sample
        lp_update = lp_change - lp_keep  # -energy change/temperature

        if criterion == "glauber":
            # Option 1: Glauber dynamics
            update_dist = dists.Bernoulli(logits=lp_update)
            updates = update_dist.sample()
        else:
            # Option 2: Metropolis-Hastings
            updates = (
                torch.rand_like(lp_update) < torch.exp(lp_update).clamp(min=0.0, max=1.0)
            ).float()
        sample = (sample_change * updates[:, None] + sample * (1.0 - updates[:, None])).type_as(x)
        self.num_changes[self._i] = updates.mean()
        self._i = (self._i + 1) % self.data_dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample  # return to original type

    def forward(self, x: torch.Tensor, model: LatticeIsingModel, **kwargs) -> torch.Tensor:
        """Performs a single step of the Gibbs sampling algorithm.

        Args:
            x: The input sample with shape atom types {0, 1} of shape (B, L).
            model: The Ising model.
            kwargs: Additional keyword arguments.
        """
        return self.step(x, model, **kwargs)

    # def logp_accept(self, xhat, x, model):
    #     """Computes the log probability of accepting a proposed sample.

    #     Args:
    #         xhat (torch.Tensor): The proposed sample.
    #         x (torch.Tensor): The current sample.
    #         model (nn.Module): The Ising model.

    #     Returns:
    #         float: The log probability of accepting the proposed sample.

    #     """
    #     # only true if xhat was generated from self.step(x, model)
    #     return 0


class LatticePottsModel(nn.Module):
    """Represents a Potts model on a lattice with q states."""

    def __init__(
        self,
        dim: int,
        q: int = 3,
        init_sigma: float = 1.0,
        init_field: float = 0.0,
        learn_G: bool = False,
        learn_sigma: bool = False,
        learn_field: bool = False,
        lattice_dim: int = 2,
        batch_size: int = 1250,
        n_samples: int = 2000,
        rand: bool = True,
    ) -> None:
        """Initialize the Potts model.

        Args:
            dim: The length of the lattice in each dimension.
            q: The number of spin states (0, 1, ..., q-1). Defaults to 3.
            init_sigma: The initial value for the interaction strength. Defaults to 1.0.
            init_field: The initial value for the field strength. Defaults to 0.0.
            learn_G: Whether to learn the adjacency matrix. Defaults to False.
            learn_sigma: Whether to learn the interaction strength. Defaults to False.
            learn_field: Whether to learn the field strength. Defaults to False.
            lattice_dim: The number of dimensions. Defaults to 2.
            batch_size: The batch size. Defaults to 1250.
            n_samples: The number of samples. Defaults to 2000.
            rand: Whether to use random updates. Defaults to True.
        """
        super().__init__()
        self.q = q
        g = ig.Graph.Lattice(dim=[dim] * lattice_dim, circular=True)  # boundary conditions
        A = np.asarray(g.get_adjacency().data)
        self.G = nn.Parameter(torch.tensor(A).float(), requires_grad=learn_G)
        self.sigma = nn.Parameter(
            torch.tensor(init_sigma).float(), requires_grad=learn_sigma
        )  # interaction strength
        self.field = nn.Parameter(
            torch.ones((dim**lattice_dim, q)).float() * init_field,
            requires_grad=learn_field,
        )  # field for each state
        # Initialize with uniform distribution over q states
        self.init_dist = dists.Categorical(probs=torch.ones((dim**lattice_dim, q)) / q)
        self.data_dim = dim**lattice_dim
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.rand = rand
        self.sampler = PottsGibbsSampler(
            data_dim=self.data_dim, batch_size=batch_size, q=q, rand=self.rand
        )

    def init_sample(self, num_samples: int) -> torch.Tensor:
        """Initializes samples from the initial distribution.

        Args:
            num_samples: The number of samples to initialize.

        Returns:
            A torch tensor representing the initialized samples with values in {0, 1, ..., q-1}.
        """
        samples = []
        for _ in range(num_samples):
            sample = self.init_dist.sample()  # (data_dim,)
            samples.append(sample)
        return torch.stack(samples).long()

    @property
    def J(self) -> torch.Tensor:
        """Calculate the interaction matrix, J.

        Returns:
            The interaction matrix.
        """
        return self.G * self.sigma

    def get_energy(
        self,
        x: torch.Tensor,
        temps: torch.Tensor = None,
        fields: torch.Tensor = None,
        time: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculates the energy for all samples.

        For Potts model: E = -J * sum_{<i,j>} delta(s_i, s_j) - sum_i h_{s_i}
        where delta is the Kronecker delta.

        Args:
            x: The input tensor with shape (B, D) with values in {0, 1, ..., q-1}.
            temps: A torch tensor representing the temperatures with shape (B,).
            fields: A torch tensor representing the fields with shape (B,).
            time: A torch tensor representing the time with shape (B,).

        Returns:
            A torch tensor representing the energy.
        """
        if x.ndim > 2:
            x = x.reshape(x.size(0), -1)

        x = x.long()  # Ensure integer values
        B, D = x.shape

        # Reshape to 2D lattice for neighbor computation
        L = int(D**0.5)
        x_2d = x.view(B, L, L)

        # Compute interaction energy: -J * sum of Kronecker deltas for neighbors
        # Using periodic boundary conditions
        x_left = torch.roll(x_2d, shifts=1, dims=2)
        x_right = torch.roll(x_2d, shifts=-1, dims=2)
        x_up = torch.roll(x_2d, shifts=1, dims=1)
        x_down = torch.roll(x_2d, shifts=-1, dims=1)

        # Count matching neighbors (Kronecker delta)
        matches = (
            (x_2d == x_left).long()
            + (x_2d == x_right).long()
            + (x_2d == x_up).long()
            + (x_2d == x_down).long()
        )
        interaction_energy = -self.sigma * matches.sum(dim=(1, 2)) / 2.0

        # Field energy: -sum_i h_{s_i}
        if fields is not None:
            # Fields is (B,) - apply same field to all states
            field_energy = -fields.sum() * 0.0  # No field contribution for now
        else:
            # Use per-state fields if available
            field_energy = torch.zeros(B, device=x.device)
            for b in range(B):
                for d in range(D):
                    field_energy[b] -= self.field[d, x[b, d]]

        denominator = temps if temps is not None else torch.tensor([1.0], device=x.device)

        if time is not None:
            assert time.size(0) == x.size(0)
            return time * ((interaction_energy + field_energy) / denominator)
        return (interaction_energy + field_energy) / denominator

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
            fields: Field torch tensor with length n_samples.
            gt_steps: The number of Gibbs sampling steps. Defaults to 1000000.
            rand: Whether to use random updates or sequential updates. Defaults to False.
            starting_samples: Optional starting samples for the Gibbs sampler.

        Returns:
            The generated samples.
        """
        if starting_samples is not None:
            samples = starting_samples.long()
        else:
            samples = self.init_sample(n_samples)
        print(f"Generating {n_samples:d} samples from {self!s:s}")
        for _ in tqdm(range(gt_steps)):
            samples = self.sampler.step(samples, self, temps, fields, time=time).detach()
        return samples.detach().cpu()

    def forward(
        self,
        x: torch.Tensor,
        temps: torch.Tensor,
        fields: torch.Tensor,
        time: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calls the `get_energy` method to compute the energy of the input tensor.

        Args:
            x: The input tensor with shape (B, D) with values in {0, 1, ..., q-1}.
            temps: A torch tensor representing the temperatures with shape (B,).
            fields: A torch tensor representing the fields with shape (B,).
            time: A torch tensor representing the time with shape (B,).
        """
        return self.get_energy(x, temps, fields, time)


class PottsGibbsSampler(nn.Module):
    """A class representing a Gibbs sampler for the Potts model with q states."""

    def __init__(self, data_dim: int, batch_size: int, q: int = 3, rand: bool = False) -> None:
        """Initialize the Potts Gibbs sampler.

        Args:
            data_dim: The total number of lattice sites.
            batch_size: The batch size.
            q: The number of spin states. Defaults to 3.
            rand: Whether to use random updates (if True) or sequential updates (if False).
                Defaults to False.
        """
        super().__init__()
        self.data_dim = data_dim
        self.batch_size = batch_size
        self.q = q
        self.num_changes = torch.zeros((data_dim,))
        self._i = 0
        self._ar = 0.0
        self._hops = 0.0
        self.rand = rand
        if self.rand:
            self.register_buffer("rand_logits", torch.zeros((self.data_dim,)))
        # Note: changes is created dynamically in step() method, not as a buffer

    def step(
        self,
        x: torch.Tensor,
        model: LatticePottsModel,
        temps: torch.Tensor = None,
        fields: torch.Tensor = None,
        time: torch.Tensor = None,
        criterion: Literal["glauber", "metropolis"] = "glauber",
    ) -> torch.Tensor:
        """Performs a single step of the Gibbs sampling algorithm.

        Args:
            x: The input sample with shape (B, D) with values in {0, 1, ..., q-1}.
            model: The Potts model.
            temps: The temperature with shape (B,).
            fields: The field with shape (B,).
            time: The time with shape (B,).
            criterion: The sampling criterion, either "glauber" or "metropolis".
                Defaults to "glauber".

        Returns:
            Updated samples with shape (B, D).
        """
        B, D = x.shape
        x = x.long()
        sample = x.clone()

        # Select which site to update
        if self.rand:
            # Random site selection
            site_idx = torch.randint(0, D, (B,), device=x.device)
        else:
            # Sequential site selection
            site_idx = torch.full((B,), self._i, device=x.device, dtype=torch.long)

        # Compute energy for each possible state at the selected site
        # Create proposals: for each sample, try all q states at the selected site
        energies = torch.zeros(B, self.q, device=x.device)
        for state in range(self.q):
            proposal = sample.clone()
            proposal[torch.arange(B), site_idx] = state
            energies[:, state] = model(proposal, temps, fields, time).squeeze()

        # Compute log probabilities (negative energy / temperature)
        if temps is not None:
            # model(..., temps) already returns E/T, so we just negate it
            log_probs = -energies
        else:
            log_probs = -energies

        if criterion == "glauber":
            # Glauber dynamics: sample from softmax distribution
            dist = dists.Categorical(logits=log_probs)
            new_states = dist.sample()
        else:
            # Metropolis-Hastings: accept/reject based on energy difference
            current_energies = energies[torch.arange(B), sample[torch.arange(B), site_idx]]
            # For MH, we need to propose uniformly and accept based on energy
            proposed_states = torch.randint(0, self.q, (B,), device=x.device)
            proposed_energies = energies[torch.arange(B), proposed_states]

            if temps is not None:
                # energies are already E/T, so difference is (E' - E)/T
                accept_prob = torch.exp(-(proposed_energies - current_energies))
            else:
                accept_prob = torch.exp(-(proposed_energies - current_energies))
            accept = torch.rand(B, device=x.device) < accept_prob.clamp(min=0.0, max=1.0)
            new_states = torch.where(accept, proposed_states, sample[torch.arange(B), site_idx])

        # Update the selected sites
        sample[torch.arange(B), site_idx] = new_states

        if not self.rand:
            self.num_changes[self._i] = (sample[:, self._i] != x[:, self._i]).float().mean()
            self._i = (self._i + 1) % self.data_dim

        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def forward(self, x: torch.Tensor, model: LatticePottsModel, **kwargs) -> torch.Tensor:
        """Performs a single step of the Gibbs sampling algorithm.

        Args:
            x: The input sample with shape (B, D) with values in {0, 1, ..., q-1}.
            model: The Potts model.
            kwargs: Additional keyword arguments.
        """
        return self.step(x, model, **kwargs)
