import numpy as np
import torch
from anomaly_tpp import simulate
from anomaly_tpp.data import Sequence, SequenceDataset
from anomaly_tpp.data.sequence import dequantize, split_into_shorter_sequences

from typing import List


__all__ = [
    "ServerStop",
    "ServerOverload",
    "Latency",
    "Connectome",
]


class Scenario:
    def get_id_train(self):
        """Return a train dataset of in-distribution (ID) sequences"""
        raise NotImplementedError

    def get_id_test(self):
        """Return a test dataset of in-distribution (ID) sequences"""
        raise NotImplementedError

    def sample_ood(self, detectability: float, **kwargs):
        """Return a test dataset of out-of-distribution (OOD) sequences"""
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__


class MultivariateHawkesFailure(Scenario):
    """Multivariate Hawkes process with 3 marks.
    In OOD sequences one entry of the influence matrix is set to zero."""

    def __init__(
        self,
        t_max: float,
        num_sequences: int,
        base_rate: float = 3.0,
        base_influence: float = 1.0,
        overload: bool = False,
        max_failure_time: float = 0.5,
        **kwargs,
    ):
        self.t_max = t_max
        self.num_sequences = num_sequences
        self.overload = overload
        self.num_marks = 3
        self.max_failure_time = max_failure_time

        # Create the Hawkes processes for ID and OOD sequences
        self.base_rates = np.array([base_rate, 0.0, 0.0])
        self.influence_id = np.zeros([self.num_marks, self.num_marks])
        self.influence_id[1, 0] = base_influence
        self.influence_id[2, 0] = base_influence

        self.influence_ood = np.zeros([self.num_marks, self.num_marks])
        self.influence_ood[1, 0] = base_influence
        if overload:
            self.influence_ood[1, 0] *= 2

    def sample_id(self, num_sequences: int):
        def _single_sample():
            """Generate a single ID sequence."""
            return simulate.hawkes_exp_kernels(
                self.t_max, base_rates=self.base_rates, influence=self.influence_id
            )

        sequences = [_single_sample() for _ in range(num_sequences)]
        return SequenceDataset(sequences, num_marks=self.num_marks)

    def get_id_train(self):
        return self.sample_id(self.num_sequences)

    def get_id_test(self):
        return self.sample_id(self.num_sequences)

    def sample_ood(self, detectability: float, **kwargs):
        if detectability < 0 or detectability > 1:
            raise ValueError(
                f"{self.name} experiment requires detectability to be a float in [0, 1] (got {detectability})"
            )
        failure_time = self.t_max - self.t_max * self.max_failure_time * detectability

        def _single_sample():
            """Generate a single OOD sequence."""
            before_failure = simulate.hawkes_exp_kernels(
                t_max=failure_time,
                base_rates=self.base_rates,
                influence=self.influence_id,
            )
            if detectability == 0:
                return before_failure
            else:
                after_failure = simulate.hawkes_exp_kernels(
                    t_max=(self.t_max - failure_time),
                    base_rates=self.base_rates,
                    influence=self.influence_ood,
                )
                return before_failure + after_failure

        sequences = [_single_sample() for _ in range(self.num_sequences)]
        return SequenceDataset(sequences, num_marks=self.num_marks)


class ServerStop(MultivariateHawkesFailure):
    def __init__(
        self,
        t_max: float,
        num_sequences: int,
        base_rate: float = 3.0,
        base_influence: float = 1.0,
        max_failure_time: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            t_max=t_max,
            num_sequences=num_sequences,
            base_rate=base_rate,
            base_influence=base_influence,
            overload=False,
            max_failure_time=max_failure_time,
        )


class ServerOverload(MultivariateHawkesFailure):
    def __init__(
        self,
        t_max: float,
        num_sequences: int,
        base_rate: float = 3.0,
        base_influence: float = 1.0,
        max_failure_time: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            t_max=t_max,
            num_sequences=num_sequences,
            base_rate=base_rate,
            base_influence=base_influence,
            overload=True,
            max_failure_time=max_failure_time,
        )


class Latency(Scenario):
    def __init__(
        self,
        t_max: float,
        num_sequences: int,
        base_rate: float = 3.0,
        mean_delay: float = 1.0,
        std_delay: float = 0.1,
        max_delay: float = 0.3,
        **kwargs,
    ):
        self.t_max = t_max
        self.num_sequences = num_sequences
        self.base_rate = base_rate
        self.mean_delay = mean_delay
        self.std_delay = std_delay
        self.max_delay = max_delay

    def _create_marked_from_unmarked(self, sequences: List[Sequence]) -> Sequence:
        """Combine several unmarked sequences into a marked sequence."""
        assert (
            len(set(s.t_max.item() for s in sequences)) == 1
        ), "All sequences must have the same t_max"
        all_marks = torch.cat(
            [
                torch.ones_like(s.arrival_times.cpu()) * i
                for (i, s) in enumerate(sequences)
            ]
        )
        all_times = torch.cat([s.arrival_times for s in sequences]).cpu().numpy()
        order = np.argsort(all_times)
        arrival_times = torch.as_tensor(all_times[order], dtype=torch.float32)
        marks = torch.as_tensor(all_marks[order], dtype=torch.long)
        return Sequence(
            arrival_times=arrival_times, marks=marks, t_max=sequences[0].t_max
        )

    def _get_response(self, seq: Sequence, lag_mean, lag_std) -> Sequence:
        """Obtain the arrival times of 'response' given the arrival times of 'trigger'."""
        response_times = seq.arrival_times + torch.normal(
            float(lag_mean), float(lag_std), [len(seq)]
        )
        response_times = response_times[response_times < self.t_max]
        return Sequence(arrival_times=response_times, t_max=self.t_max)

    def _single_sample(self, extra_delay: float = 0.0) -> Sequence:
        trig = simulate.homogeneous_poisson(t_max=self.t_max, rate=self.base_rate)
        resp = self._get_response(trig, self.mean_delay + extra_delay, self.std_delay)
        return self._create_marked_from_unmarked([trig, resp])

    def sample_id(self, num_sequences: int):
        return SequenceDataset(
            [self._single_sample(extra_delay=0.0) for _ in range(num_sequences)],
            num_marks=2,
        )

    def get_id_train(self):
        return self.sample_id(self.num_sequences)

    def get_id_test(self):
        return self.sample_id(self.num_sequences)

    def sample_ood(self, detectability: float, **kwargs):
        if detectability < 0:
            raise ValueError(
                f"{self.__class__.__name__} experiment requires detectability to be a float >= 0 (got {detectability})"
            )
        extra_delay = detectability * self.max_delay
        return SequenceDataset(
            [self._single_sample(extra_delay) for _ in range(self.num_sequences)],
            num_marks=2,
        )
    

class Connectome(Scenario):
    def __init__(
        self,
        random_seed=123,
        step_size=300,
        window_size=1000,
        num_train=500,
        num_marks_to_keep=50,
        **kwargs,
    ):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        dset = SequenceDataset.from_file("./data/connectome/connectome_1.pt")
        full_seq = dset[0]

        # Only keep the first num_marks_to_keep marks
        events_to_keep = np.isin(full_seq.marks.numpy(), np.arange(num_marks_to_keep))
        seq = Sequence(
            arrival_times=full_seq.arrival_times[events_to_keep],
            t_max=full_seq.t_max,
            marks=full_seq.marks[events_to_keep],
        )
        # Add uniform noise to convert discrete arrival times into continuous
        seq = dequantize(seq, dt=1.0).float()

        # Split long sequence into shorter subsequences
        subsequences = split_into_shorter_sequences(
            seq, step_size=step_size, window_size=window_size
        )
        self.id_train = SequenceDataset(
            subsequences[:num_train], num_marks=dset.num_marks,
        ) 
        self.id_test = SequenceDataset(
            subsequences[num_train:], num_marks=dset.num_marks,
        )

        # Permutation used to generate remappings
        self.num_marks = dset.num_marks
        self.permutation = np.random.permutation(self.num_marks)

    def set_seed(self, seed):
        np.random.seed(seed)
        self.permutation = np.random.permutation(self.num_marks)

    def get_id_train(self):
        return self.id_train

    def get_id_test(self):
        return self.id_test

    def sample_ood(self, detectability, seed, **kwargs):
        self.set_seed(seed)
        num_flips = int(detectability * self.num_marks)
        return SequenceDataset(
            [self.switch_marks(seq, num_flips) for seq in self.id_test],
            num_marks=self.num_marks,
        )

    def switch_marks(self, seq, num_flips):
        marks = seq.marks
        remap = self.get_remap_dict(num_flips)
        new_marks = torch.tensor(
            [remap[m.item()] for m in marks], device=marks.device
        ).long()
        return Sequence(
            arrival_times=seq.arrival_times, t_max=seq.t_max, marks=new_marks
        )

    def get_remap_dict(self, num_flips):
        remap = {}
        for i in range(num_flips // 2):
            a = self.permutation[2 * i]
            b = self.permutation[2 * i + 1]
            remap[a] = b
            remap[b] = a
        for i in range(self.num_marks):
            if i not in remap:
                remap[i] = i
        return remap