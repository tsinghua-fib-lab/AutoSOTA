import os
import torch
import glob
import tifffile
import heapq
import torchvision
import numpy as np
from tqdm import tqdm
import argparse
from scanpy import read_visium
import scanpy as sc
import pandas as pd
from joblib import Parallel, delayed
import sys
from PIL import ImageFile, Image

import timm
from huggingface_hub import login
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset import Alex_Nat_Slice
from sklearn.model_selection import LeaveOneOut
sys.path.insert(0, "../")

XFOLD = [i for i in range(4)]
loo = LeaveOneOut()
KFOLD = []
for x in loo.split(XFOLD):
    KFOLD.append(x)


parser = argparse.ArgumentParser()
parser.add_argument("--index_dir", default="Alex_Nat", type=str)  #
parser.add_argument("--save_dir", default=r"./refer/Alex_Nat/uni",
                    type=str)
parser.add_argument("--data", default=r"D:\dataset\Alex_NatGen/",
                    type=str)
parser.add_argument("--fold", default=0, type=int)

args = parser.parse_args()

index_dir = args.index_dir
save_dir = args.save_dir
os.makedirs(os.path.join(save_dir, str(args.fold), index_dir), exist_ok=True)

login("<REDACTED:HF_TOKEN>")
encoder = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
transform = create_transform(**resolve_data_config(encoder.pretrained_cfg, model=encoder))
encoder.to("cuda")
encoder.eval()
features = 1024

num_cores = 12
batch_size = 128
patch_size = 224
reference = KFOLD[args.fold][0]

root = args.data

data = []
for file in Alex_Nat_Slice:
    path = os.path.join(root, file)
    matrix_file = os.path.join(path, 'filtered_count_matrix', 'matrix.mtx')
    genes_file = os.path.join(path, 'filtered_count_matrix', 'features.tsv')
    barcodes_file = os.path.join(path, 'filtered_count_matrix', 'barcodes.tsv')
    positions_file = os.path.join(path, 'spatial', 'tissue_positions_list.csv')
    meta_file = os.path.join(path, 'metadata.csv')
    img_fild = os.path.join(path, 'image.tif')

    img = np.array(Image.open(img_fild))

    adata = sc.read_mtx(matrix_file).T

    genes = pd.read_csv(genes_file, header=None, sep='\t')
    barcodes = pd.read_csv(barcodes_file, header=None, sep='\t')

    adata.var['gene_ids'] = genes[0].values
    adata.obs['barcodes'] = barcodes[0].values
    adata.var_names = adata.var['gene_ids']
    adata.obs_names = adata.obs['barcodes']

    positions = pd.read_csv(positions_file, sep=',', header=None)
    positions.columns = ['Barcode', 'InMatrix', 'X', 'Y', 'Pixel_X', 'Pixel_Y']


    positions = positions.set_index('Barcode').reindex(adata.obs_names)

    adata.obs['x'] = positions['X'].values
    adata.obs['y'] = positions['Y'].values

    spatial_coords = np.vstack((positions['Pixel_X'].values, positions['Pixel_Y'].values)).T
    adata.obsm['spatial'] = spatial_coords

    meta = pd.read_csv(meta_file, sep=',', index_col=0)
    meta = meta.reindex(adata.obs_names)
    adata.obs['nCount_RNA'] = meta['nCount_RNA'].values
    adata.obs['nFeature_RNA'] = meta['nFeature_RNA'].values
    adata.obs['subtype'] = meta['subtype'].values
    adata.obs['classification'] = meta['Classification'].values

    # adata.write(rf"D:\dataset\Alex_NatGen\{file}/{file}.h5ad")

    coord = pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)
    data.append([img, coord.values.astype(int)])
    print(coord.values.astype(int).shape)

mapping = []
for i, (img, coord) in enumerate(data):
    mappings = []
    for j in range(len(coord)):
        mappings.append(j)
    mapping.append(mappings)


def generate():
    def get_slide_gene(idx, ):

        img, coord = data[idx[0]]
        coord = coord[idx[1]]

        y, x = coord
        img = img[(y + (-patch_size // 2)):(y + (patch_size // 2)), (x + (-patch_size // 2)):(x + (patch_size // 2)), :]

        code = idx[1]
        img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1) / 255
        img = img.cuda()
        return (img - 0.5) / 0.5, code

    def extract(imgs):
        with torch.no_grad():
            batch = torch.stack(imgs)
            return encoder(batch).view(-1, features)

    for i, k in tqdm(enumerate(mapping)):
        batch_img, codes = [], []
        img_embedding = []
        for j in k:
            img, code = get_slide_gene([i, j])
            batch_img.append(img)
            codes.append(code)

            if len(batch_img) == batch_size:
                img_embedding += [extract(batch_img)]
                batch_img = []

        if len(batch_img) != 0:
            img_embedding += [extract(batch_img)]

        img_embedding = torch.cat(img_embedding).contiguous()

        assert (np.array(codes) == np.sort(codes)).all()

        assert img_embedding.size(0) == len(codes)

        assert img_embedding.size(1) == features
        print(img_embedding.size())
        torch.save(img_embedding, f"{save_dir}/{i}.pt")
        del img_embedding, batch_img, codes
        torch.cuda.empty_cache()


def create_search_index():
    class Queue:
        def __init__(self, max_size=2):
            self.max_size = max_size
            self.list = []

        def add(self, item):
            heapq.heappush(self.list, item)

            while len(self.list) > self.max_size:
                heapq.heappop(self.list)

        def __repr__(self):
            return str(self.list)

    for i, _ in tqdm(enumerate(mapping)):
        p = f"{save_dir}/{i}.pt"
        p = torch.load(p).cuda()

        Q = [Queue(max_size=128) for _ in range(p.size(0))]

        for op_i, _ in enumerate(mapping):
            if op_i == i or not op_i in reference:
                continue

            op = torch.load(f"{save_dir}/{op_i}.pt").cuda()
            dist = torch.cdist(p.unsqueeze(0), op.unsqueeze(0), p=1).squeeze(0)
            topk = min(len(dist), 100)
            knn = dist.topk(topk, dim=1, largest=False)

            q_values = knn.values.cpu().numpy()
            q_infos = knn.indices.cpu().numpy()

            def add(q_value, q_info, myQ):
                for idx in range(len(q_value)):
                    myQ.add((-q_value[idx], q_info[idx], op_i))
                return myQ

            Q = Parallel(n_jobs=num_cores)(
                delayed(add)(q_values[f], q_infos[f], Q[f]) for f in range(q_values.shape[0]))
        np.save(f"{save_dir}/{args.fold}/{index_dir}/{i}.npy", [myq.list for myq in Q])


if args.fold == 0:
    generate()
create_search_index()
