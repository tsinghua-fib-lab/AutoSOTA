import argparse
import os
import torch
import sys
import time
import random
import numpy as np
from tqdm import tqdm

from biggan_models.model import BigGAN
from biggan_models.utils import (
    truncated_noise_sample,
    one_hot_from_int,
    save_as_images_by_class,
)
from biggan_models.model_kdeformer import KDEformerBigGAN
from biggan_models.model_performer import PerformerBigGAN
from biggan_models.model_reformer import ReformerBigGAN
from biggan_models.model_thinformer import ThinformerBigGAN
from biggan_models.model_wildcat import WildCatBigGAN

ALL_ATTENTION_TYPES = ['exact', 'kdeformer', 'performer', 'reformer', 'thinformer', 'wildcat']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='biggan-deep-512')
    parser.add_argument("--labels", "-l", type=int, nargs='+', required=True,
                        help="Target ImageNet class labels to generate images for")
    parser.add_argument("--n_images", "-n", type=int, default=5,
                        help="Number of images to generate per label")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--truncation", type=float, default=0.4)
    parser.add_argument("--attention", type=str, default=None,
                        choices=ALL_ATTENTION_TYPES + ['all'],
                        help="Attention type to use. Use 'all' to run all types (default: all)")
    # WildCat-specific args
    parser.add_argument("--r", "-r", type=int, default=96, help="WildCat rank parameter")
    parser.add_argument("--bins", "-b", type=int, default=8, help="WildCat number of bins")
    return parser.parse_args()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model(attention, model_name, r=96, num_bins=8):
    if attention == 'exact':
        return BigGAN.from_pretrained(model_name)
    elif attention == 'kdeformer':
        return KDEformerBigGAN.from_pretrained(model_name)
    elif attention == 'performer':
        return PerformerBigGAN.from_pretrained(model_name)
    elif attention == 'reformer':
        return ReformerBigGAN.from_pretrained(model_name)
    elif attention == 'sblocal':
        try:
            from biggan_models.model_sblocal import SBlocalBigGAN
        except ImportError:
            print("To run scatterbrain, install fast-transformers: pip install --no-build-isolation fast-transformers/")
            raise
        return SBlocalBigGAN.from_pretrained(model_name)
    elif attention == 'thinformer':
        return ThinformerBigGAN.from_pretrained(model_name)
    elif attention == 'wildcat':
        return WildCatBigGAN.from_pretrained(model_name, r=r, num_bins=num_bins)
    else:
        raise NotImplementedError(f"Invalid attention option: {attention}")


@torch.no_grad()
def generate_for_attention(attention, args, labels, noise_vector, class_vector):
    print(f"\n=== Attention: {attention} ===")
    model = load_model(attention, args.model_name, r=args.r, num_bins=args.bins)
    print(model.__class__)

    if torch.cuda.is_available():
        model = model.to('cuda')

    model.eval()
    output_all = []
    num_batches = len(labels) // args.batch_size + 1
    tic = time.time()
    for idx in tqdm(range(num_batches)):
        batch_idx = list(range(idx * args.batch_size, min(len(labels), (idx + 1) * args.batch_size)))
        if len(batch_idx) == 0:
            continue
        n_vec = noise_vector[batch_idx]
        c_vec = class_vector[batch_idx]
        output = model(n_vec, c_vec, args.truncation)
        output_all.append(output.to('cpu'))

    print(f"Generation time: {time.time() - tic:.4f} sec")
    output_all = torch.cat(output_all)

    # Build output path and save
    attn_tag = f"wildcat_r{args.r}_b{args.bins}" if attention == 'wildcat' else attention
    model_tag = args.model_name.replace('-', '_')
    output_path = f"./generations/{model_tag}/{attn_tag}-s{args.seed}"
    os.makedirs(output_path, exist_ok=True)

    save_as_images_by_class(output_all, labels, output_path)

    print(f"Saved images to {output_path}")
    del model


@torch.no_grad()
def main():
    args = get_args()
    seed_everything(args.seed)

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")

    attention_types = ALL_ATTENTION_TYPES if args.attention is None or args.attention == 'all' else [args.attention]

    # Build label and noise/class vectors (repeated per image count)
    labels = np.repeat(args.labels, args.n_images).tolist()
    class_vector = one_hot_from_int(labels, batch_size=len(labels))
    noise_vector = truncated_noise_sample(truncation=args.truncation, batch_size=len(labels), seed=args.seed)

    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)

    if torch.cuda.is_available():
        noise_vector = noise_vector.to('cuda')
        class_vector = class_vector.to('cuda')

    for attention in attention_types:
        generate_for_attention(attention, args, labels, noise_vector, class_vector)


if __name__ == "__main__":
    main()
