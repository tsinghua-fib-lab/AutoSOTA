import torch
from tqdm import tqdm
from sdot import rebalance


class SdotRebalanceDataset(torch.utils.data.Dataset):
    def __init__(self, ot_sampler, length, batch_size=1024):
        super().__init__()
        assert length % len(ot_sampler.data_tensor) == 0
        self.noise = torch.randn(length, ot_sampler.data_tensor_flatten.shape[1])
        self.target_idx = torch.randint(high=1, size=(length,))
        self.batch_size = batch_size
        self.ot_sampler = ot_sampler

    def __getitem__(self, idx):
        target_idx = self.target_idx[idx]
        return self.noise[idx], target_idx

    def __len__(self):
        return len(self.target_idx)

    def refresh(self):
        self.noise.normal_()
        batch_size = self.batch_size
        with tqdm(range(0, len(self), batch_size), desc="refreshing") as pbar:
            for i in pbar:
                current_noise = self.noise[i : min(i + batch_size, len(self))]
                _, _, target_idx = self.ot_sampler.sample_plan(
                    current_noise.to(self.ot_sampler.dual_weight.device), None
                )
                self.target_idx[i : min(i + batch_size, len(self))] = (
                    target_idx.squeeze().cpu()
                )

        # Apply rebalancing
        x = self.target_idx
        x_rebalance = rebalance(x, num_targets=len(self.ot_sampler.data_tensor))[0]
        sdot_ratio = torch.sum(x_rebalance == x) / len(x)
        print("Refreshed! SDOT ratio: {:3f}".format(sdot_ratio.item()))
        self.target_idx = x_rebalance
