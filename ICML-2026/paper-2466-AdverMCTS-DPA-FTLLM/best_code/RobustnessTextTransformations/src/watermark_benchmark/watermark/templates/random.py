import random
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import hash_cpp
import torch

from watermark_benchmark.utils import adapt_arguments


class BaseRandomness(ABC):
    """Base class for randomness with helper functions.

    Attributes:
        * self.secret_key: Secret key used for seeding the watermark random number generator.
        Can either be a single integer if all generations use the same key, or a list of
        integers if each generation uses a different key. In that case, the length of the list G
        is the number of generations.

    * self.device: Device on which to generate random numbers.

    Minimal set of methods to implements:
        * _get_seed:
        Returns the seed for the given previous values and IDs.

        * _rand_index:
        Returns a tensor of random numbers determined by the list of seeds and a token index.

    No other methods need to be implemented for a minimal functional example.
    However, the other methods are used by the watermarking algorithm and
    can be implemented for better performance.

    Methods:
        * reset:
        Resets the state of the randomness generator after each generation.

        * get_secret:
        Returns the secret key for the given offset (if the secret key is a single value, returns that value)

        * normalize_previous_values:
        Normalizes the previous values by padding them with 1s to make them of equal length, and converting them to a tensor.

        * get_seed, rand_index:
        Wrappers around _get_seed and _rand_index that adapt the arguments to the expected format.

        * rand_range and _rand_range:
        Returns a tensor of random values of shape N x L, where N is the number of seeds and L is the length of the range. Calls rand_index for each index in the range.

    """

    def __init__(self, secret_key: Union[List[int], int], device: str = "cpu"):
        self.secret_key = secret_key
        self.device = device

    def reset(self):
        pass

    def get_secret(self, offset):
        return (
            self.secret_key[offset]
            if isinstance(self.secret_key, list)
            else self.secret_key
        )

    def reinit(self):
        pass

    def normalize_previous_values(
        self, previous_values: Union[List[int], torch.Tensor]
    ):
        """
        Normalize the previous values by padding them with 1s to make them of equal length, and converting them to a tensor.

        Args:
            previous_values (list or torch.Tensor): The previous values to normalize.

        Returns:
            torch.Tensor: The normalized previous values as a tensor.
        """
        if not isinstance(previous_values, torch.Tensor):
            max_len = max(len(p) for p in previous_values)
            previous_values = [
                [1 for _ in range(max_len - len(p))] + p
                for p in previous_values
            ]
            previous_values = torch.tensor(previous_values)

        if len(previous_values.shape) == 1:
            previous_values = previous_values.unsqueeze(0)

        return previous_values

    def get_seed(
        self,
        previous_values: Union[List[int], torch.Tensor],
        ids: Union[List[int], torch.Tensor] = None,
        **kwargs
    ) -> Union[str, List[str], torch.Tensor]:
        previous_values = self.normalize_previous_values(previous_values)
        kwargs["ids"] = ids
        return adapt_arguments(self._get_seed, kwargs, previous_values)

    def rand_index(
        self, seeds: Union[str, List[str], torch.Tensor], index: int, **kwargs
    ) -> torch.Tensor:
        if isinstance(seeds, str):
            seeds = [seeds]
        return adapt_arguments(self._rand_index, kwargs, seeds, index)

    def rand_range(
        self, seeds: Union[str, List[str], torch.Tensor], length: int, **kwargs
    ) -> torch.Tensor:
        if isinstance(seeds, str):
            seeds = [seeds]
        return adapt_arguments(self._rand_range, kwargs, seeds, length)

    @abstractmethod
    def _get_seed(
        self, previous_values: torch.Tensor, ids: Optional[List[int]] = None
    ):
        """Returns the seed for the given previous values and IDs.

        previous_values: tensor of previously generated tokens, shape N x K (N batch size, K sequence length)
        ids: integer IDs of the each generation in the batch. Used to recover the generation-specific key length. Length N. Values range from 0 to G-1 (G number of generations)

        returns a list of seeds (strings) of length N
        """

    @abstractmethod
    def _rand_index(
        self, seeds: Union[List[str], torch.Tensor], index: int, device=None
    ) -> torch.Tensor:
        """
        Returns a tensor of random values of shape N, where N is the number of seeds.
        Each value is a random number generated from the seed and the index.

        seeds: list of N seeds, one for each generation in the batch.
        index: index of the token.
        """

    def _rand_range(
        self, seeds: Union[List[str], torch.Tensor], length: int, **kwargs
    ) -> torch.Tensor:
        rand_per_index = [
            self.rand_index(seeds, i, **kwargs) for i in range(length)
        ]
        return torch.stack(rand_per_index)


class SimpleRandomness(BaseRandomness):
    """Simple randomness implementation, only requires a secret key and does not depend on generated text"""

    def _get_seed(self, previous_values, ids=None):
        N, _ = previous_values.shape
        if ids is None:
            ids = [0 for _ in range(N)]

        return [str(self.get_secret(i)) for i in ids]

    def _rand_index(self, seeds, index, device=None):
        if device is None:
            device = self.device
        return hash_cpp.index_hash(seeds, index).to(device)


class Randomness(BaseRandomness):
    """
    Base class for generating random numbers that depend on sequence information. Extends BaseRandomness

    Additional Attributes:
        * devices (Union[torch.device, List[torch.device]]): A device or list of devices
        on which to generate random numbers. The first in the list is the main device,
        but others can be specified if the RNG can be accessed on multiple devices.
        * vocab_size (int): The size of the vocabulary of the LLM.

    Additional Methods:
        * update_state:
        Updates the internal state to keep track of how many tokens have been generated for each generation.

        * set_permutation:
        Sets the permutation used for downstream tasks. One permutation is created for each secret key.

        * get_permutation:
        Returns the permutation used for downstream tasks.

        * green_list:
        Returns the green list for the given seeds and gamma value (Specific for distibution-shift)
    """

    @abstractmethod
    def __init__(
        self,
        secret_key: Union[List[int], int],
        devices: Optional[Union[str, List[str]]] = None,
        vocab_size: int = 1,
    ):
        if isinstance(devices, list):
            self.devices = devices
        elif devices is not None:
            self.devices = [devices]
        else:
            self.devices = [i for i in range(torch.cuda.device_count())]

        super().__init__(secret_key, devices[0])

        self.vocab_size = vocab_size
        self.state = None
        self.reset()
        self.set_permutation()

        self.generator = torch.Generator("cpu")

    def reinit(self):
        self.state = None
        self.reset()
        self.set_permutation()
        self.generator = torch.Generator("cpu")

    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size

    def reset(self):
        """
        Resets the state of the watermark generator to all zeros.
        """
        le = (
            1 if not isinstance(self.secret_key, list) else len(self.secret_key)
        )
        self.state = torch.zeros((le,)).long()

    def update_state(self, ids):
        """
        Update the state of the watermark generator.
        """
        self.state[ids] += 1

    def set_permutation(self):
        """
        Creates one permutation for each secret key, stores it in a tensor, copied around each device.
        Shape is G x V, where G is the number of secret keys and V is the vocabulary size.
        """
        if self.vocab_size < 2:
            return

        if isinstance(self.secret_key, list):
            shuf = [
                list(range(self.vocab_size))
                for _ in range(len(self.secret_key))
            ]
            for idx, key in enumerate(self.secret_key):
                random.Random(key).shuffle(shuf[idx])
            permutation = torch.tensor(shuf)
        else:
            shuf = list(range(self.vocab_size))
            random.Random(self.secret_key).shuffle(shuf)
            permutation = torch.tensor(shuf).unsqueeze(0)

        inv_permutation = torch.zeros_like(permutation)
        indices = (
            torch.arange(permutation.shape[0])
            .repeat(permutation.shape[1], 1)
            .t()
        )
        indices = torch.cat(
            (indices.unsqueeze(2), permutation.unsqueeze(2)), dim=2
        )
        inv_permutation[indices[:, :, 0], indices[:, :, 1]] = torch.arange(
            self.vocab_size
        ).repeat(permutation.shape[0], 1)

        self.permutation = {
            device: permutation.to(device) for device in self.devices
        }
        self.inv_permutation = {
            device: inv_permutation.to(device) for device in self.devices
        }

    def get_permutation(self, device: Optional[str], inv: bool = False):
        """
        Returns the permutation used for downstream tasks.
        If inv is specified, the inverse permutation is returned.
        """
        if device not in self.permutation:
            if (
                type(device) == torch.device
                and device.index in self.permutation
            ):
                device = device.index
            else:
                print(
                    "Device not initialized for random number generator. The sampling procedure is occuring on device {}, while only {} are available. Copying over".format(
                        device, self.devices
                    )
                )
                self.permutation[device] = self.permutation[self.device].to(
                    device
                )
                self.inv_permutation[device] = self.inv_permutation[
                    self.device
                ].to(device)
        if inv:
            return self.inv_permutation[device]
        else:
            return self.permutation[device]

    def green_list(self, seeds: List[str], gamma: float, inv: bool = False):
        """
        Return a green list seeded by the given seeds. Its size is gamma*vocab_size.
        The green list is a boolean vector of size N x V, where N is the number of
        seeds and V is the vocabulary size.
        """
        gl_size = int(gamma * self.vocab_size)
        permutation = torch.cat(
            tuple(
                torch.randperm(
                    self.vocab_size,
                    generator=self.generator.manual_seed(
                        int(h.item() * 2147483647)
                    ),
                ).unsqueeze(0)
                for h in seeds
            )
        )
        if not inv:
            return permutation[:, :gl_size]
        else:
            permutation = permutation.to(self.device)
            inv_permutation = torch.zeros_like(permutation)
            indices = (
                torch.arange(permutation.shape[0], device=self.device)
                .repeat(permutation.shape[1], 1)
                .t()
            )
            indices = torch.cat(
                (indices.unsqueeze(2), permutation.unsqueeze(2)), dim=2
            )
            inv_permutation[indices[:, :, 0], indices[:, :, 1]] = torch.arange(
                self.vocab_size, device=self.device
            ).repeat(permutation.shape[0], 1)
            return inv_permutation <= gl_size


class EmbeddedRandomness(Randomness):
    """
    A class that represents text-dependent randomness.

    Additional Attributes:
        hash_len (int): The length of the hash.
        min_hash (bool): A flag indicating whether to use the minimum hash.
    """

    def __init__(
        self, secret_key, device=None, vocab_size=1, hash_len=1, min_hash=False
    ):
        super().__init__(secret_key, device, vocab_size)
        self.hash_len = hash_len
        self.min_hash = min_hash

    def _get_seed(
        self, previous_values: torch.Tensor, ids: Optional[List[int]] = None
    ):
        N, _ = previous_values.shape
        if ids is None:
            ids = [0 for _ in range(N)]

        if not self.hash_len:
            tmp = [[] for _ in range(previous_values.shape[0])]
        else:
            tmp = [
                [v.item() for v in prev[-self.hash_len :]]
                for prev in previous_values
            ]
            tmp = [
                [-1 for _ in range(self.hash_len - len(value))] + value
                for value in tmp
            ]

        if self.min_hash:
            h = [
                str(
                    round(
                        min(
                            hash_cpp.index_hash(
                                ["{}SEED{}".format(t, self.get_secret(ids[k]))],
                                0,
                            )
                            .cpu()
                            .item()
                            for t in (tmp[k] if len(tmp[k]) else [0])
                        ),
                        8,
                    )
                )
                for k in range(N)
            ]
        else:
            tmp = ["_".join(str(i) for i in t) if len(t) else "" for t in tmp]
            h = [
                "{}SEED{}".format(t, self.get_secret(ids[k]))
                for k, t in enumerate(tmp)
            ]

        self.update_state(ids)
        return h

    def _rand_range(self, seeds: List[str], length: int, device=None, **kwargs):
        return hash_cpp.all_index_hash(
            seeds,
            torch.zeros((len(seeds), length), dtype=torch.float32).to(
                self.device if device is None else device
            ),
        )

    def _rand_index(self, seeds: List[str], index: int, device=None):
        return hash_cpp.index_hash(seeds, index).to(
            self.device if device is None else device
        )


class ExternalRandomness(Randomness):
    """
    A class representing an external source of randomness for generating watermarks.

    Additional Attributes:
        * key_len (int): The length of the key.
        * random_size (int): The number of random numbers to generate per position in the key, if different from the vocab size.

    Functions _get_index and _rand_index are re-interpreted to use the external source of randomness.
    """

    def __init__(
        self,
        secret_key,
        device=None,
        vocab_size=1,
        key_len=512,
        random_size=None,
    ):
        self.key_len = key_len
        super().__init__(secret_key, device, vocab_size)
        if random_size is None:
            self.random_size_is_vocab_size = True
            self.random_size = vocab_size
        else:
            self.random_size_is_vocab_size = False
            self.random_size = random_size

        self._init_xi()

    def _init_xi(self):
        self.rng = (
            [random.Random(self.secret_key)]
            if not isinstance(self.secret_key, list)
            else [random.Random(key) for key in self.secret_key]
        )
        self.xi = torch.tensor(
            [
                [r.random() for _ in range(self.key_len * self.random_size)]
                for r in self.rng
            ],
            dtype=torch.float64,
        ).reshape(len(self.rng), self.key_len, self.random_size)

    def reinit(self):
        if self.random_size_is_vocab_size:
            self.random_size = self.vocab_size
        super().reinit()
        self._init_xi()
        self.reset()

    def reset(self):
        super().reset()
        le = (
            1 if not isinstance(self.secret_key, list) else len(self.secret_key)
        )
        self.shift = torch.randint(self.key_len, (le,))

    def _get_seed(
        self, previous_values: torch.Tensor, ids: Optional[List[int]] = None
    ):
        N, _ = previous_values.shape
        if ids is None:
            ids = torch.zeros((N,)).long()
        elif not isinstance(ids, torch.Tensor):
            ids = torch.Tensor(ids).long()
        self.update_state(ids)

        rtn = torch.cat(
            (
                ids.unsqueeze(0),
                (
                    (self.shift[ids] + self.state[ids] - 1) % self.key_len
                ).unsqueeze(0),
            ),
            axis=0,
        ).t()
        return rtn

    def _rand_range(
        self, seeds: torch.Tensor, length: int, device=None, **kwargs
    ):
        """The seeds argument is used as an index into the xi tensor."""
        index = seeds
        if length:
            return self.xi[index[:, 0], index[:, 1], :length].to(
                self.device if device is None else device
            )
        else:
            return self.xi[index[:, 0], index[:, 1], :].to(
                self.device if device is None else device
            )

    def _rand_index(self, seeds: torch.Tensor, index: int, device=None):
        """The seeds argument is used as an index into the xi tensor."""
        xi_index = seeds
        return self.xi[xi_index[:, 0], xi_index[:, 1], index].to(
            self.device if device is None else device
        )
