import argparse
import os
import torch
import time
import random
import numpy as np
from tqdm import tqdm

from torch.nn import functional as F

from biggan_models.model import BigGAN
from biggan_models.utils import (
    truncated_noise_sample,
    one_hot_from_int,
    save_as_images
)
from biggan_models.model_wildcat import WildCatBigGAN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str, default='biggan-deep-512')
    parser.add_argument("--num_classes",type=int, default=1000)
    parser.add_argument("--data_per_class",type=int, default=5)
    parser.add_argument("--seed",type=int, default=1)
    parser.add_argument("--num_splits", "-ns",type=int, default=10)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--truncation",type=float, default=0.4)
    parser.add_argument("--no_store",action='store_true')
    parser.add_argument("--fid",action='store_true')
    parser.add_argument("--debug",action='store_true')
    parser.add_argument("--postfix", type=str, default='')
    # Fast attention arguments
    parser.add_argument("--attention",type=str, default='exact', choices=['exact', 'kdeformer', 'performer', 'reformer', 'sblocal', 'thinformer', 'wildcat'])
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

@torch.no_grad()
def main():
    args = get_args()
    seed_everything(args.seed)

    for aa, bb in args.__dict__.items():
        print(f"{aa}: {bb}")

    model_name = args.model_name
    num_classes = args.num_classes
    data_per_class = args.data_per_class
    batch_size = args.batch_size
    attention = args.attention
    truncation = args.truncation

    # Load pre-trained model tokenizer (vocabulary)
    if attention == 'exact':
        model = BigGAN.from_pretrained(model_name)
    elif attention == 'kdeformer':
        from biggan_models.model_kdeformer import KDEformerBigGAN
        model = KDEformerBigGAN.from_pretrained(model_name)
    elif attention == 'performer':
        from biggan_models.model_performer import PerformerBigGAN
        model = PerformerBigGAN.from_pretrained(model_name)
    elif attention == 'reformer':
        from biggan_models.model_reformer import ReformerBigGAN
        model = ReformerBigGAN.from_pretrained(model_name)
    elif attention == 'sblocal':
        try:
            from biggan_models.model_sblocal import SBlocalBigGAN
        except:
            print("To run scatterbrain, install fast-transformers package: pip install --no-build-isolation fast-transformers/")
        model = SBlocalBigGAN.from_pretrained(model_name)
    elif attention == 'thinformer':
        from biggan_models.model_thinformer import ThinformerBigGAN
        model = ThinformerBigGAN.from_pretrained(model_name)
    elif attention == 'wildcat':
        model = WildCatBigGAN.from_pretrained(model_name, r=args.r, num_bins=args.bins)
    else:
        raise NotImplementedError("Invalid attention option")

    print(model.__class__)

    # Prepare a input
    labels = np.repeat(np.arange(num_classes), data_per_class).tolist()
    class_vector = one_hot_from_int(labels, batch_size=len(labels))
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=len(labels), seed=args.seed)

    # All in tensors
    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)

    if torch.cuda.is_available():
        # If you have a GPU, put everything on cuda
        noise_vector = noise_vector.to('cuda')
        class_vector = class_vector.to('cuda')
        model = model.to('cuda')

    tic = time.time()
    model.eval()
    output_all = []
    num_batches = len(labels) // batch_size + 1
    for idx in tqdm(range(num_batches)):
        batch_idx = list(range(idx * batch_size, min(len(labels), (idx+1) * batch_size)))
        if len(batch_idx) == 0:
            continue

        n_vec = noise_vector[batch_idx]
        c_vec = class_vector[batch_idx]

        # Generate an image
        output = model(n_vec, c_vec, truncation)
        output = output.to('cpu')

        output_all.append(output)

    time_generation = time.time() - tic

    output_all = torch.cat(output_all)
    print(f"output_all.shape: {output_all.shape}")
    print(f"generation time : {time_generation:.4f} sec")
    del model, noise_vector, class_vector

    if args.fid:
        print("computing FID & Inception scores ...")
        import inception_utils
        
        pool, logits = get_logits(output_all)
        is_mean_fake, is_std_fake = inception_utils.calculate_inception_score(logits.cpu().numpy(), num_splits=args.num_splits)
        print(f"Inception score : {is_mean_fake:.5f} (std : {is_std_fake:.5f})", flush=True)

        mu, sigma = np.mean(pool.cpu().numpy(), axis=0), np.cov(pool.cpu().numpy(), rowvar=False)
        data_mu = np.load('imagenet_val_inception_moments.npz')['mu']
        data_sigma = np.load('imagenet_val_inception_moments.npz')['sigma']

        fid_value = inception_utils.numpy_calculate_frechet_distance(mu, sigma, data_mu, data_sigma)
        print(f"FID  : {fid_value}", flush=True)

        print("Saving results to file", flush=True)
        if attention == 'wildcat':
            attention = f"wildcat_r{args.r}_b{args.bins}"
        res_str = f"model: {args.model_name}, data_per_class: {data_per_class}, num_splits: {args.num_splits}, seed: {args.seed}, attention: {attention:<30}, fid: {fid_value}, is_mean_fake: {is_mean_fake}, is_std_fake: {is_std_fake}\n"
        with open("./fid_score_results.txt", "a") as f:
            f.write(res_str)

        if not args.no_store:
            print("Saving images to disk", flush=True)
            output_path = f"./generations/{model_name.replace('-','_')}/{attention}-n{len(labels)}{args.postfix}-ns{args.num_splits}-s{args.seed}"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            tic = time.time()
            print(f"saving {data_per_class} images....")
            save_as_images(output_all[282*data_per_class:283*data_per_class], output_path + "/img")
            print(f"done. ({time.time() - tic:.4f} sec)")


def get_logits(imgs, batch_size=128, net=None):
    import inception_utils

    if net is None:
        net = inception_utils.load_inception_net()
        net = net.to('cuda')

    pool, logits = [], []
    with torch.no_grad():
        num_batches = len(imgs) // batch_size + 1
        for idx in tqdm(range(num_batches)):
            batch_idx = list(range(idx * batch_size, min(len(imgs), (idx+1) * batch_size)))
            if len(batch_idx) == 0:
                continue

            # Generate an image
            pool_val, logits_val = net(imgs[batch_idx].to('cuda'))
            pool += [pool_val]
            logits += [F.softmax(logits_val, 1)]
    pool, logits = torch.cat(pool, 0), torch.cat(logits, 0)
    return pool , logits

if __name__ == "__main__":
    main()
 