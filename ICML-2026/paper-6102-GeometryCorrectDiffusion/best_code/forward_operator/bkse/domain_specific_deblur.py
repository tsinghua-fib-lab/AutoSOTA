import argparse
from math import ceil, log10
from pathlib import Path

import torchvision
import yaml
from PIL import Image
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset


class Images(Dataset):
    def __init__(self, root_dir, duplicates):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob("*.png"))
        self.duplicates = (
            duplicates  # Number of times to duplicate the image in the dataset to produce multiple HR images
        )

    def __len__(self):
        return self.duplicates * len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx // self.duplicates]
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        if self.duplicates == 1:
            return image, img_path.stem
        else:
            return image, img_path.stem + f"_{(idx % self.duplicates)+1}"


parser = argparse.ArgumentParser(description="PULSE")

# I/O arguments
parser.add_argument("--input_dir", type=str, default="imgs/blur_faces", help="input data directory")
parser.add_argument(
    "--output_dir", type=str, default="experiments/domain_specific_deblur/results", help="output data directory"
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default="experiments/domain_specific_deblur/cache",
    help="cache directory for model weights",
)
parser.add_argument(
    "--yml_path", type=str, default="options/domain_specific_deblur/stylegan2.yml", help="configuration file"
)

kwargs = vars(parser.parse_args())

with open(kwargs["yml_path"], "rb") as f:
    opt = yaml.safe_load(f)

dataset = Images(kwargs["input_dir"], duplicates=opt["duplicates"])
out_path = Path(kwargs["output_dir"])
out_path.mkdir(parents=True, exist_ok=True)

dataloader = DataLoader(dataset, batch_size=opt["batch_size"])

if opt["stylegan_ver"] == 1:
    from models.dsd.dsd_stylegan import DSDStyleGAN

    model = DSDStyleGAN(opt=opt, cache_dir=kwargs["cache_dir"])
else:
    from models.dsd.dsd_stylegan2 import DSDStyleGAN2

    model = DSDStyleGAN2(opt=opt, cache_dir=kwargs["cache_dir"])

model = DataParallel(model)

toPIL = torchvision.transforms.ToPILImage()

for ref_im, ref_im_name in dataloader:
    if opt["save_intermediate"]:
        padding = ceil(log10(100))
        for i in range(opt["batch_size"]):
            int_path_HR = Path(out_path / ref_im_name[i] / "HR")
            int_path_LR = Path(out_path / ref_im_name[i] / "LR")
            int_path_HR.mkdir(parents=True, exist_ok=True)
            int_path_LR.mkdir(parents=True, exist_ok=True)
        for j, (HR, LR) in enumerate(model(ref_im)):
            for i in range(opt["batch_size"]):
                toPIL(HR[i].cpu().detach().clamp(0, 1)).save(int_path_HR / f"{ref_im_name[i]}_{j:0{padding}}.png")
                toPIL(LR[i].cpu().detach().clamp(0, 1)).save(int_path_LR / f"{ref_im_name[i]}_{j:0{padding}}.png")
    else:
        # out_im = model(ref_im,**kwargs)
        for j, (HR, LR) in enumerate(model(ref_im)):
            for i in range(opt["batch_size"]):
                toPIL(HR[i].cpu().detach().clamp(0, 1)).save(out_path / f"{ref_im_name[i]}.png")
