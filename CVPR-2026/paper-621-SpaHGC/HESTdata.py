import os

import numpy as np
import h5py
from hest import iter_hest
import torch
import timm
from huggingface_hub import login
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
import heapq
from joblib import Parallel, delayed
from dataset import  HEST_Lymph_Node, HEST_Pancreas_xenium, HEST_Pancreas_Visium
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torchvision

from sklearn.model_selection import LeaveOneOut
XFOLD = [i for i in range(4)]
loo = LeaveOneOut()
KFOLD = []
for x in loo.split(XFOLD):
    KFOLD.append(x)

def load_data(ids, base_path):
    data = []
    for idx, st in enumerate(iter_hest(base_path, id_list=ids)):
        img = st.wsi
        adata = st.adata
        coord = pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)

        data.append([img, coord.values.astype(int)])
        print(coord.values.astype(int).shape)
    return data


def generate_mappings(data):
    mapping = []
    for i, (_, coord) in enumerate(data):
        mappings = list(range(len(coord)))
        mapping.append(mappings)
    return mapping


def extract_features(batch, encoder, features):
    with torch.no_grad():
        batch = torch.stack(batch)
        return encoder(batch).view(-1, features)


def generate_embeddings(patch_size, data, mapping, encoder, save_dir, batch_size, features):
    def get_slide_gene(idx, ):

        img, coord = data[idx[0]]
        coord = coord[idx[1]]

        x, y = coord
        patch = img.read_region((x, y), 0, (patch_size, patch_size))
        patch = patch[:, :, :3]
        code = idx[1]
        patch = torch.as_tensor(patch, dtype=torch.float).permute(2, 0, 1) / 255
        patch = patch.cuda()
        return patch, code

    for i, k in tqdm(enumerate(mapping)):
        batch_img, codes = [], []
        img_embedding = []
        for j in k:
            img, code = get_slide_gene([i, j])
            batch_img.append(img)
            codes.append(code)

            if len(batch_img) == batch_size:
                img_embedding += [extract_features(batch_img, encoder, features)]
                batch_img = []

        if len(batch_img) != 0:
            img_embedding += [extract_features(batch_img, encoder, features)]

        img_embedding = torch.cat(img_embedding).contiguous()
        assert (np.array(codes) == np.sort(codes)).all()

        assert img_embedding.size(0) == len(codes)

        assert img_embedding.size(1) == features

        torch.save(img_embedding, f"{save_dir}/{i}.pt")
        del img_embedding, batch_img, codes
        torch.cuda.empty_cache()


class Queue:
    def __init__(self, max_size=128):
        self.max_size = max_size
        self.list = []

    def add(self, item):
        heapq.heappush(self.list, item)
        while len(self.list) > self.max_size:
            heapq.heappop(self.list)


def create_search_index(mapping, save_dir, index_dir, reference, num_cores):
    for i, _ in tqdm(enumerate(mapping)):
        p = torch.load(f"{save_dir}/{i}.pt").cuda()
        Q = [Queue() for _ in range(p.size(0))]

        for op_i in reference:
            if op_i == i:
                continue
            op = torch.load(f"{save_dir}/{op_i}.pt").cuda()
            dist = torch.cdist(p.unsqueeze(0), op.unsqueeze(0), p=1).squeeze(0)
            topk = min(len(dist), 100)
            knn = dist.topk(topk, dim=1, largest=False)

            def add(q_value, q_info, myQ):
                for idx in range(len(q_value)):
                    myQ.add((-q_value[idx], q_info[idx], op_i))
                return myQ

            q_values, q_infos = knn.values.cpu().numpy(), knn.indices.cpu().numpy()
            Q = Parallel(n_jobs=num_cores)(
                delayed(add)(q_values[f], q_infos[f], Q[f]) for f in range(q_values.shape[0])
            )

        np.save(f"{save_dir}/{args.fold}/{index_dir}/{i}.npy", [myq.list for myq in Q])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="HEST_Lymph_Node", type=str)  # her2+ HEST_Cervix HEST_Lymph_Node HEST_Pancreas_xenium
    parser.add_argument("--save_dir", default=r"./refer/HEST_Lymph_Node/uni",
                        type=str)  # HEST_Cervix,HER2+
    parser.add_argument("--fold", default=0, type=int)

    args = parser.parse_args()

    login("<REDACTED:HF_TOKEN>")
    encoder = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    transform = create_transform(**resolve_data_config(encoder.pretrained_cfg, model=encoder))
    encoder.to("cuda")
    encoder.eval()
    features = 1024



    Tissue = HEST_Lymph_Node  # HEST_Pancreas HEST_Lymph_Node HEST_Pancreas_Visium HEST_Pancreas_xenium

    data = load_data(Tissue, "./hest_data")

    mapping = generate_mappings(data)

    reference = KFOLD[args.fold][0]
    num_cores = 12
    batch_size = 128
    patch_size = 224

    os.makedirs(os.path.join(args.save_dir, str(args.fold), args.index_dir), exist_ok=True)

    if args.fold == 0:
        generate_embeddings(patch_size, data, mapping, encoder, args.save_dir, batch_size, features)

    create_search_index(mapping, args.save_dir, args.index_dir, reference, num_cores)
