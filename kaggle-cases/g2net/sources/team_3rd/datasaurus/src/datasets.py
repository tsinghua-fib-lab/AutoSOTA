import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from scipy import signal
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchaudio.functional import lowpass_biquad

from src.config import INPUT_PATH
from src.preprocessing import apply_whiten, biquad_bandpass_filter, get_design_curves


class GWDataset(Dataset):
    def __init__(
        self,
        df,
        tukey_alpha=0.2,
        bp_lf=25,
        bp_hf=500,
        bp_order=4,
        whiten=False,
        folder="train",
        channel_shuffle=False,
        **kwargs,
    ):
        self.df = df.reset_index(drop=True)
        self.folder = folder
        self.window = torch.tensor(signal.tukey(4096, tukey_alpha))
        self.design_curves = torch.tensor(get_design_curves(("tukey", tukey_alpha)))
        self.lf = bp_lf
        self.hf = bp_hf
        self.order = bp_order
        self.whiten = whiten
        self.channel_shuffle = channel_shuffle

    def load_file(self, id_):
        path = INPUT_PATH / self.folder / id_[0] / id_[1] / id_[2] / f"{id_}.npy"
        waves = np.load(path)
        return waves / np.max(np.abs(waves), axis=1).reshape(3, 1)
        # return waves / np.max(np.abs(waves))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        target = torch.tensor([self.df.loc[index, "target"]], dtype=torch.float32)
        data = self.load_file(self.df.loc[index, "id"])
        data = torch.tensor(data) * self.window

        if self.whiten:
            data = apply_whiten(data, self.design_curves)
        else:
            data = data.to(dtype=torch.float32)

        if self.lf and self.hf:
            data = biquad_bandpass_filter(data, self.lf, self.hf, 2048)

        return data, target


class GWSyntheticDataset(Dataset):
    def __init__(
        self,
        df,
        tukey_alpha=0.2,
        bp_lf=25,
        bp_hf=500,
        bp_order=4,
        folder="train",
        amplitude_min=0.1,
        amplitude_max=0.5,
        **kwargs,
    ):
        self.df = df.query("target == 0").reset_index(drop=True)
        self.folder = folder
        self.window = torch.tensor(signal.tukey(4096, tukey_alpha))
        self.lf = bp_lf
        self.hf = bp_hf
        self.order = bp_order
        self.gw_paths = list((INPUT_PATH / "gw_sim").glob("*.npy"))
        self.amp_min = amplitude_min
        self.amp_max = amplitude_max

    def load_file(self, id_):
        path = INPUT_PATH / self.folder / id_[0] / id_[1] / id_[2] / f"{id_}.npy"
        waves = np.load(path)
        return waves

    def prepare_gw(self, sig, sr=2048, error_ms=5):
        if len(sig) < 4096:
            pad = 4096 - len(sig)
            sig = np.pad(sig, (0, pad))
        # Time lags https://arxiv.org/abs/1706.04191
        # https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/251934#1387136

        # Start the GW from Hanford
        if np.random.rand() > 0.5:
            hanford_delay = 0 + np.random.normal(scale=error_ms)
            livingston_delay = 10 + np.random.normal(scale=error_ms)
            virgo_delay = 26 + np.random.normal(scale=error_ms)
        # Start the GW from Virgo
        else:
            hanford_delay = 27 + np.random.normal(scale=error_ms)
            livingston_delay = 26 + np.random.normal(scale=error_ms)
            virgo_delay = 0 + np.random.normal(scale=error_ms)

        random_pad = np.random.randint(30, int(4096 * 0.85))

        hanford_pad = int(sr * hanford_delay / 1000) + random_pad
        livingston_pad = int(sr * livingston_delay / 1000) + random_pad
        virgo_pad = int(sr * virgo_delay / 1000) + random_pad

        hanford_sig = np.pad(sig, (0, hanford_pad))[-4096:]
        livingston_sig = np.pad(sig, (0, livingston_pad))[-4096:]
        virgo_sig = np.pad(sig, (0, virgo_pad))[-4096:]
        return np.stack([hanford_sig, livingston_sig, virgo_sig])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.load_file(self.df.loc[index, "id"])
        data = torch.tensor(data, dtype=torch.float32)
        scale = torch.abs(data).max()

        # if np.random.rand() > 0.5:
        target = torch.tensor([1], dtype=torch.float32)
        gw = np.load(np.random.choice(self.gw_paths))[1]
        data_clean = torch.tensor(self.prepare_gw(gw), dtype=torch.float32)
        data_clean *= np.random.uniform(low=self.amp_min, high=self.amp_max)
        data += data_clean
        data_clean /= scale
        # else:
        #     target = torch.tensor([0], dtype=torch.float32)
        #     data_clean = torch.normal(mean=0, std=1e-10, size=data.shape)

        data /= scale

        data *= self.window
        data = biquad_bandpass_filter(data, self.lf, self.hf, 2048)

        return data, data_clean, target


class GWDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        seed: int = 48,
        tukey_alpha=0.2,
        bp_lf=25,
        bp_hf=500,
        bp_order=4,
        whiten=False,
        denoising=False,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        num_workers: int = 4,
        pseudo_label: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.alpha = tukey_alpha
        self.lf = bp_lf
        self.hf = bp_hf
        self.order = bp_order
        self.whiten = whiten
        self.denoising = denoising
        self.train_transforms = train_transforms
        self.valid_transforms = val_transforms
        self.test_transforms = test_transforms
        self.num_workers = num_workers
        self.pseudo_label = pseudo_label

        self.df = pd.read_csv(INPUT_PATH / "training_labels.csv")
        self.df_test = pd.read_csv(INPUT_PATH / "sample_submission.csv")
        self.df_pl = pd.read_csv(INPUT_PATH / "submission_power2_weight.csv")
        # self.df_pl = self.df_pl.query("(mean < 0.4) | (mean > 0.8 & mean < 0.99)")
        self.create_folds()

        # Harden PLs
        # temperature = 2
        # idxs = self.df_pl["mean"] > 0.75  # Only harden above a threshold
        # cols = [f"fold_{i}" for i in range(5)]
        # self.df_pl.loc[idxs, cols] = expit(
        #     logit(self.df_pl.loc[idxs, cols]) * temperature
        # )

    def create_folds(self):
        skf = StratifiedKFold(5, shuffle=True, random_state=self.seed)
        splits = skf.split(self.df["id"], self.df["target"])
        self.df["fold"] = -1
        for fold, (_, val_idx) in enumerate(splits):
            self.df.loc[val_idx, "fold"] = fold

    def setup(self, stage=None, fold_n: int = 0):
        trn_df = self.df.query(f"fold != {fold_n}")
        val_df = self.df.query(f"fold == {fold_n}")

        params = {
            "tukey_alpha": self.alpha,
            "bp_lf": self.lf,
            "bp_hf": self.hf,
            "bp_order": self.order,
            "whiten": self.whiten,
        }

        if stage == "fit" or stage is None:
            if self.denoising:
                self.gw_train = GWSyntheticDataset(trn_df, **params)
                self.gw_valid = GWSyntheticDataset(val_df, **params)
            else:
                self.gw_train = GWDataset(trn_df, **params)
                self.gw_valid = GWDataset(val_df, **params)

                if self.pseudo_label:
                    # pl_fold = self.df_pl[["id", f"fold_{fold_n}"]]
                    # pl_fold.columns = ["id", "target"]
                    pl_dataset = GWDataset(self.df_pl, folder="test", **params)
                    self.gw_train = ConcatDataset([self.gw_train, pl_dataset])

            print(
                f"Setup fold {fold_n} with {len(self.gw_train)} train",
                f"and {len(self.gw_valid)} valid samples",
            )

        if stage == "test":
            self.gw_test = GWDataset(self.df_test, folder="test", **params)

    def train_dataloader(self):
        return DataLoader(
            self.gw_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.gw_valid,
            batch_size=self.batch_size * 4,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.gw_test,
            batch_size=self.batch_size * 4,
            num_workers=self.num_workers,
            pin_memory=True,
        )
