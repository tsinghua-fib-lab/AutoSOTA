import torch
from torch.utils.data import Sampler, RandomSampler, DistributedSampler
import numpy as np


class AlternatingSampler(Sampler):
    def __init__(self, dataset_a, dataset_b, batch_size, m, n, generator=None):
        """
        Args:
            dataset_a (Dataset): 第一个数据集.
            dataset_b (Dataset): 第二个数据集.
            batch_size (int): 每个批次的大小.
            m (int): 从 dataset_a 中连续采样的批次数.
            n (int): 从 dataset_b 中连续采样的批次数.
            generator (Generator): 用于可复现的随机数生成器.
        """
        super().__init__()
        self.len_a = len(dataset_a)
        self.len_b = len(dataset_b)

        self.batch_size = batch_size
        self.m = m
        self.n = n

        # 为每个数据集创建独立的随机采样器
        self.sampler_a = RandomSampler(dataset_a, generator=generator)
        self.sampler_b = RandomSampler(dataset_b, generator=generator)

        if np.ceil(self.len_a / m) <= np.ceil(self.len_b / n):
            self.num_samples = int(self.len_a * (1 + n / m))
        else:
            self.num_samples = int(self.len_b * (1 + m / n))

    def __iter__(self):
        iter_a = iter(self.sampler_a)
        iter_b = iter(self.sampler_b)

        finished = False
        while not finished:
            # 从 loader_a 中读取 m 个 batch 的索引
            for _ in range(self.m):
                batch_indices = []
                for _ in range(self.batch_size):
                    try:
                        idx = next(iter_a)
                        batch_indices.append(idx)
                    except StopIteration:
                        # 如果 sampler_a 耗尽，重新初始化
                        iter_a = iter(self.sampler_a)
                        idx = next(iter_a)
                        batch_indices.append(idx)

                # 如果 batch_indices 为空，说明可能迭代已结束
                if not batch_indices:
                    finished = True
                    break
                yield from batch_indices

            if finished: break

            # 从 loader_b 中读取 n 个 batch 的索引
            for _ in range(self.n):
                batch_indices = []
                for _ in range(self.batch_size):
                    try:
                        # 关键：为 b 的索引加上 a 的长度偏移
                        idx = next(iter_b) + self.len_a
                        batch_indices.append(idx)
                    except StopIteration:
                        # 如果 sampler_b 耗尽，重新初始化
                        iter_b = iter(self.sampler_b)
                        idx = next(iter_b) + self.len_a
                        batch_indices.append(idx)

                if not batch_indices:
                    finished = True
                    break
                yield from batch_indices

    def __len__(self):
        return self.num_samples


class DistributedAlternatingSampler(Sampler):
    def __init__(self, dataset_a, dataset_b, batch_size, m, n,
                 num_replicas=None, rank=None, shuffle=True, seed=0):
        """
        兼容分布式训练的交替采样器。

        Args:
            dataset_a (Dataset): 第一个数据集.
            dataset_b (Dataset): 第二个数据集.
            batch_size (int): 每个批次的大小.
            m (int): 从 dataset_a 中连续采样的批次数.
            n (int): 从 dataset_b 中连续采样的批次数.
            num_replicas (int, optional): 分布式训练中的进程总数. Defaults to None.
            rank (int, optional): 当前进程的排名. Defaults to None.
            shuffle (bool, optional): 是否打乱数据. Defaults to True.
            seed (int, optional): 随机种子. Defaults to 0.
        """
        super().__init__()
        if num_replicas is None or rank is None:
            # 尝试从 torch.distributed 获取，如果失败则为单机模式
            try:
                num_replicas = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
            except (ValueError, RuntimeError):  # Not initialized
                num_replicas = 1
                rank = 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.total_len_a = len(dataset_a)
        self.total_len_b = len(dataset_b)

        self.batch_size = batch_size
        self.m = m
        self.n = n

        # 为每个子数据集创建独立的分布式采样器
        self.sampler_a = DistributedSampler(
            dataset_a, num_replicas=self.num_replicas, rank=self.rank, shuffle=self.shuffle, seed=self.seed
        )
        self.sampler_b = DistributedSampler(
            dataset_b, num_replicas=self.num_replicas, rank=self.rank, shuffle=self.shuffle, seed=self.seed
        )

        # 计算当前 rank 分到的样本总数
        if np.ceil(len(self.sampler_a) / m) <= np.ceil(len(self.sampler_b) / n):
            self.num_samples = int(len(self.sampler_a) * (1 + n / m))
        else:
            self.num_samples = int(len(self.sampler_b) * (1 + m / n))

    def __iter__(self):
        iter_a = iter(self.sampler_a)
        iter_b = iter(self.sampler_b)

        finished = False
        while not finished:
            # 从 sampler_a 中读取 m 个 batch 的索引
            for _ in range(self.m):
                batch_indices = []
                for _ in range(self.batch_size):
                    try:
                        idx = next(iter_a)
                        batch_indices.append(idx)
                    except StopIteration:
                        # 分布式采样器耗尽时，不应该手动重启，因为一个 epoch 的数据已经采样完毕
                        # 我们可以选择提前结束或者用完另一个采样器的数据
                        # 这里我们简单地跳出内层循环
                        break

                if not batch_indices:
                    finished = True  # 如果一个 batch 都没取到，说明 sampler_a 彻底空了
                    break
                yield from batch_indices

            if finished: break

            # 从 sampler_b 中读取 n 个 batch 的索引
            for _ in range(self.n):
                batch_indices = []
                for _ in range(self.batch_size):
                    try:
                        # 关键：为 b 的索引加上 a 的总长度偏移
                        idx = next(iter_b) + self.total_len_a
                        batch_indices.append(idx)
                    except StopIteration:
                        break

                if not batch_indices:
                    finished = True
                    break
                yield from batch_indices

    def __len__(self) -> int:
        # 返回当前 rank 分到的样本数量
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        设置当前 epoch，这对于确保 shuffle 在多个 epoch 中正常工作至关重要。
        Trainer 会在每个 epoch 开始时自动调用此方法。
        """
        self.epoch = epoch
        self.sampler_a.set_epoch(epoch)
        self.sampler_b.set_epoch(epoch)