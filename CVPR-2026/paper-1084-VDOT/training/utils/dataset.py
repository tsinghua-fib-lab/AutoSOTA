import os
import sys
sys.path.append(os.path.abspath('.'))
import torchvision.transforms.functional as F
from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
import decord
import pandas as pd


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        file = prompt_path
        df = pd.read_csv(file)
        self.prompt_list = df['prompt'].tolist()

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TextMultiTaskVideoDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        # with open(prompt_path, encoding="utf-8") as f:
        #     self.prompt_list = [line.rstrip() for line in f]
        file = prompt_path
        df = pd.read_csv(file)
        self.task_list = df['task_name'].tolist()
        self.src_video_path = df['src_video_path'].tolist()
        self.mask_video_path = df['mask_video_path'].tolist()
        self.ref_image_path = df['ref_image_path'].tolist()
        self.prompt_list = df['internvl_short'].tolist()

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "task_name": self.task_list[idx],
            "prompts": self.prompt_list[idx],
            "video_path": self.src_video_path[idx],
            "mask_video_path": self.mask_video_path[idx],
            "ref_image_path": self.ref_image_path[idx],
            "idx": idx,
        }
        return batch


class TextPoseGTVideoDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        # with open(prompt_path, encoding="utf-8") as f:
        #     self.prompt_list = [line.rstrip() for line in f]
        file = prompt_path
        df = pd.read_csv(file)
        self.video_list = df['video_path'].tolist()
        self.prompt_list = df['internvl_short'].tolist()
        self.gt_video_list = df['gt_video_path'].tolist()


    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "video_path": self.video_list[idx],
            "gt_video_path": self.gt_video_list[idx],
            "idx": idx,
        }
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }

import io
class ODEfromCeph(Dataset):
    def __init__(self, json_path):
        self.json_path = json_path
        with open(self.json_path, "r") as f:
            self.data_anno = json.load(f)

    def __len__(self):
        return len(self.data_anno)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        proxy_url = "xxx"
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        # latent_file_path = self.data_anno[str(idx)]
        latent_file_path = self.data_anno[idx]
        latent_byte = io.BytesIO(client.get(latent_file_path))
        latent_json = torch.load(
            latent_byte,
            map_location="cpu",
            weights_only=True,
        )
        latent = latent_json['denoise_latents']
        context = latent_json['context']

        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        os.environ['http_proxy'] = proxy_url
        os.environ['https_proxy'] = proxy_url

        return {
                "prompt_embeds": context,
                "ode_latent": latent
            }


def collate_fn(batch):
    """
    Collate function to handle batching of custom dataset items.
    Each item in the batch is a dict with keys "prompt_embeds" and "ode_latent".
    """
    # Extract prompt_embeds and ode_latent from each item in the batch
    # for item in batch:
    #     print(len(item["prompt_embeds"]))
    prompt_embeds = [item["prompt_embeds"][0] for item in batch]
    ode_latent = [item["ode_latent"][-1] for item in batch]

    # Stack them along the first dimension to create a batch
    # prompt_embeds_batch = torch.stack(prompt_embeds, dim=0)
    ode_latent_batch = torch.stack(ode_latent, dim=0)

    # Return as a dictionary
    return {
        "prompt_embeds": prompt_embeds,
        "ode_latent": ode_latent_batch
    }

class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str

def smart_resize_and_crop(video, target_height=480, target_width=832):
    """
    Resize video so that one side == target, and the other >= target,
    then center crop to (target_height, target_width)
    Args:
        video (torch.Tensor): shape (T, C, H, W)
    Returns:
        (T, C, target_height, target_width)
    """
    T, C, H, W = video.shape

    # Compute scale
    scale = max(target_height / H, target_width / W)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))

    # Resize each frame
    resized_video = torch.stack([
        F.resize(frame, [new_h, new_w]) for frame in video
    ])

    # Center crop
    cropped_video = torch.stack([
        F.center_crop(frame, [target_height, target_width]) for frame in resized_video
    ])

    return cropped_video

class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


def cycle(dl):
    while True:
        for data in dl:
            yield data

if __name__ == '__main__':
    dataset = ODEfromCeph("./train_data/wanx_gan.json")
    # sampler = torch.utils.data.distributed.DistributedSampler(
    #     dataset, shuffle=True, drop_last=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        # sampler=sampler,
        collate_fn=collate_fn,
        num_workers=8)

    dataloader = cycle(dataloader)
    batch = next(dataloader)
    print(len(batch['prompt_embeds']), len(batch['ode_latent']), batch['prompt_embeds'][0].shape, batch['ode_latent'].shape)
    # print(batch["encode_latent"].shape, batch["condition_image"].shape, batch["camera_embedding"].shape)
    # print(batch["caption"])

    # "encode_latent": encode_latent,
    # "caption": caption,
    # "condition_image": first_image,
    # "camera_embedding": camera_embedding,
    # }
