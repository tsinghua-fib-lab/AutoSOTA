from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pickle
import pandas
import tifffile
from PIL import Image
from scanpy import read_visium
import scanpy as sc
import pandas as pd
import glob
from PIL import ImageFile, Image
import scprep as scp
from huggingface_hub import login
import datasets
import h5py
from hest import iter_hest

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

'''
code is based on https://github.com/bryanhe/ST-Net
'''

SquamousCellCarcinoma_patients = ['P2', 'P5', 'P9', 'P10']
SquamousCellCarcinoma_reps = ['rep1', 'rep2', 'rep3']
SquamousCellCarcinoma = []
for i in SquamousCellCarcinoma_patients:
    for j in SquamousCellCarcinoma_reps:
        SquamousCellCarcinoma.append(i + '_ST_' + j)

HER2Add_cnt_dir = "D:\dataset\Her2st\data\ST-cnts"
HER2Add_patients = os.listdir(HER2Add_cnt_dir)
HER2Add_patients.sort()
HER2Add_patients = [i[:2] for i in HER2Add_patients]

login(token="<REDACTED:HF_TOKEN>")
local_dir = 'hest_data'  # hest will be dowloaded to this folder

meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")

Lymph_Node_df = meta_df[(meta_df['organ'] == 'Lymph node') & (meta_df['disease_state'] != 'Healthy')]

HEST_Lymph_Node = list(Lymph_Node_df['id'].values)

Pancreas_xenium_df = meta_df[
    (meta_df['organ'] == 'Pancreas') & (meta_df['disease_state'] != 'Healthy') & (meta_df['st_technology'] == 'Xenium')]

HEST_Pancreas_xenium = list(Pancreas_xenium_df['id'].values)

Pancreas_Visium_df = meta_df[
    (meta_df['organ'] == 'Pancreas') & (meta_df['disease_state'] != 'Healthy') & (meta_df['st_technology'] == 'Visium')]

HEST_Pancreas_Visium = list(Pancreas_Visium_df['id'].values)

Alex_Nat_Slice = ['1142243F', '1160920F', 'CID4465', 'CID44971']

Visium_BC_Slice = ['block1', 'block2', 'FFPE']

DLPFC_Slice = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674',
               '151675', '151676']


class Visium_BC_Dataset(Dataset):
    def __init__(self, Visium_BC_Slice, index_filter, transform, args, train=None):
        self.Visium_BC_Slice = Visium_BC_Slice
        self.args = args
        self.train = train
        self.index_filter = index_filter
        self.data = self.load_raw(args.data)
        self.meta_info(args.data)
        self.transform = transform

        keep_idx = list(range(len(self.mean)))
        keep = set(keep_idx)

        self.filter_name = [j for i, j in enumerate(self.gene_names) if i in keep]
        self.gene_filter = np.array([i in keep for i in range(len(self.gene_names))])
        self.max = torch.log10(torch.as_tensor(self.max[self.gene_filter], dtype=torch.float) + 1)
        self.min = torch.log10(torch.as_tensor(self.min[self.gene_filter], dtype=torch.float) + 1)
        del self.data
        mapping = []
        for i in Visium_BC_Slice:
            img, counts, coord, emb, index = self.all_data[i]
            for j in range(len(counts)):
                mapping.append([i, j])
        self.map = mapping

    def load_raw(self, data_root):

        data = []
        for idx, file in enumerate(Visium_BC_Slice):
            index = os.path.join(self.args.index_path, f"{idx}.npy")
            index = np.load(index)

            path = os.path.join(data_root, file)
            img_fild = os.path.join(path, 'image.tif')
            h5 = read_visium(path, count_file=f"filtered_feature_bc_matrix.h5")
            img = np.array(Image.open(img_fild))
            h5.var_names_make_unique()

            data.append([img, h5, idx, index])

        return data

    def meta_info(self, root):

        from tqdm import tqdm
        if not os.path.exists(r'D:\baseline\SpaMGCL-main\v2\Visium_BC/com_gene.npy'):
            gene_names = None
            for _, p, _, _ in tqdm(self.data):
                sc.pp.highly_variable_genes(p, flavor="seurat_v3", n_top_genes=1000)
                hvg = set(p.var[p.var.highly_variable].index.tolist())
                if gene_names is None:
                    gene_names = hvg
                else:
                    gene_names = gene_names.intersection(hvg)

            gene_names = list(gene_names)
            np.save(r'D:\baseline\SpaMGCL-main\v2\Visium_BC/com_gene.npy', np.array(gene_names))
        else:
            gene_names = list(np.load(r'D:\baseline\SpaMGCL-main\v2\Visium_BC/com_gene.npy', allow_pickle=True))

        all_data = {}
        all_gene = []
        part_gene = []
        for img, p, idx, index in tqdm(self.data):
            counts = pd.DataFrame(p.X.todense(), columns=p.var_names, index=p.obs_names)
            coord = pd.DataFrame(p.obsm['spatial'], columns=['x_coord', 'y_coord'], index=p.obs_names)

            c = counts[gene_names].values.astype(float)

            emb = torch.load(f"{self.args.emb_path}/{idx}.pt", map_location=torch.device("cpu"))

            assert emb.size(0) == c.shape[0]

            all_data[idx] = [img, c, coord.values.astype(int), emb, index]

            for i in c:
                all_gene.append(i)
                if idx in self.Visium_BC_Slice:
                    part_gene.append(i)

        all_gene = np.array(all_gene)
        part_gene = np.array(part_gene)
        print(all_gene.shape, part_gene.shape)

        self.mean = np.mean(all_gene, 0)
        self.max = np.max(part_gene, 0)
        self.min = np.min(part_gene, 0)
        self.gene_names = gene_names
        self.all_data = all_data

        gene_path = r"D:\baseline\SpaMGCL-main\v2\Visium_BC/ComGene.txt"
        if not os.path.exists(gene_path):
            with open(gene_path, "w") as f:
                for name in self.gene_names:
                    f.write(name + "\n")

    def retrive_similer(self, index):
        numk = self.args.numk
        index = np.array([[i, j, k] for i, j, k in sorted(index, key=lambda x: float(x[0]))])
        index = index[-numk:]

        op_emb = []
        op_counts = []
        gene_num = 0
        for _, op_name, op_n in index:
            op_name = int(op_name)
            op_n = int(op_n)
            op_emb.append(self.all_data[op_n][3][op_name])
            op_count = torch.as_tensor(self.all_data[op_n][1][op_name], dtype=torch.float)
            op_count = torch.log10(op_count[self.gene_filter] + 1)
            op_count = (op_count - self.min) / (self.max - self.min + 1e-8)
            op_counts.append(op_count)
            gene_num = len(op_count)

        return torch.stack(op_emb).view(numk, -1), torch.stack(op_counts).view(numk, gene_num)

    def generate(self, idx):

        idx = self.map[idx]
        img, counts, coord, emb, index = self.all_data[idx[0]]
        counts, coord, emb, index = counts[idx[1]], coord[idx[1]], emb[idx[1]], index[idx[1]]

        emb = emb.unsqueeze(0)

        x, y = coord
        window = self.args.size
        pos = [x // window, y // window]

        op_emb, op_counts = self.retrive_similer(index)

        img = img[(y + (-window // 2)):(y + (window // 2)), (x + (-window // 2)):(x + (window // 2)), :]

        if self.transform != None:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1) / 255

        counts = torch.log10(torch.as_tensor(counts[self.gene_filter], dtype=torch.float) + 1)
        counts = (counts - self.min) / (self.max - self.min + 1e-8)

        return {
            "img": img,
            "count": counts,
            "p_feature": emb,
            "pos_count": op_counts,
            "pos_feature": op_emb,
            "pos": torch.LongTensor(pos),
        }

    def __getitem__(self, index):
        return self.generate(index)

    def __len__(self):
        return len(self.map)


class DLPFC_Dataset(Dataset):
    def __init__(self, DLPFC_Slice, index_filter, transform, args, train=None):
        self.DLPFC_Slice = DLPFC_Slice
        self.args = args
        self.train = train
        self.index_filter = index_filter
        self.data = self.load_raw(args.data)
        self.meta_info(args.data)
        self.transform = transform

        keep_idx = list(range(len(self.mean)))
        keep = set(keep_idx)

        self.filter_name = [j for i, j in enumerate(self.gene_names) if i in keep]
        self.gene_filter = np.array([i in keep for i in range(len(self.gene_names))])
        self.max = torch.log10(torch.as_tensor(self.max[self.gene_filter], dtype=torch.float) + 1)
        self.min = torch.log10(torch.as_tensor(self.min[self.gene_filter], dtype=torch.float) + 1)
        del self.data
        mapping = []
        for i in DLPFC_Slice:
            img, counts, coord, emb, index = self.all_data[i]
            for j in range(len(counts)):
                mapping.append([i, j])
        self.map = mapping

    def load_raw(self, data_root):

        data = []
        for idx, file in enumerate(DLPFC_Slice):
            index = os.path.join(self.args.index_path, f"{idx}.npy")
            index = np.load(index)

            path = os.path.join(data_root, file)
            img_fild = os.path.join(path, 'spatial', f'{file}_full_image.tif')
            h5 = read_visium(path, count_file=f"{file}_filtered_feature_bc_matrix.h5")
            img = np.array(Image.open(img_fild))
            h5.var_names_make_unique()

            data.append([img, h5, idx, index])

        return data

    def meta_info(self, root):

        from tqdm import tqdm
        if not os.path.exists(r'D:\baseline\SpaMGCL-main\v2\DLPFC/com_gene.npy'):
            gene_names = None
            for _, p, _, _ in tqdm(self.data):
                sc.pp.highly_variable_genes(p, flavor="seurat_v3", n_top_genes=1000)
                hvg = set(p.var[p.var.highly_variable].index.tolist())
                if gene_names is None:
                    gene_names = hvg
                else:
                    gene_names = gene_names.intersection(hvg)

            gene_names = list(gene_names)
            np.save(r'D:\baseline\SpaMGCL-main\v2\DLPFC/com_gene.npy', np.array(gene_names))
        else:
            gene_names = list(np.load(r'D:\baseline\SpaMGCL-main\v2\DLPFC/com_gene.npy', allow_pickle=True))

        all_data = {}
        all_gene = []
        part_gene = []
        for img, p, idx, index in tqdm(self.data):
            counts = pd.DataFrame(p.X.todense(), columns=p.var_names, index=p.obs_names)
            coord = pd.DataFrame(p.obsm['spatial'], columns=['x_coord', 'y_coord'], index=p.obs_names)

            c = counts[gene_names].values.astype(float)

            emb = torch.load(f"{self.args.emb_path}/{idx}.pt", map_location=torch.device("cpu"))

            assert emb.size(0) == c.shape[0]

            all_data[idx] = [img, c, coord.values.astype(int), emb, index]

            for i in c:
                all_gene.append(i)
                if idx in self.DLPFC_Slice:
                    part_gene.append(i)

        all_gene = np.array(all_gene)
        part_gene = np.array(part_gene)
        print(all_gene.shape, part_gene.shape)

        self.mean = np.mean(all_gene, 0)
        self.max = np.max(part_gene, 0)
        self.min = np.min(part_gene, 0)
        self.gene_names = gene_names
        self.all_data = all_data

        gene_path = r"D:\baseline\SpaMGCL-main\v2\DLPFC/ComGene.txt"
        if not os.path.exists(gene_path):
            with open(gene_path, "w") as f:
                for name in self.gene_names:
                    f.write(name + "\n")

    def retrive_similer(self, index):
        numk = self.args.numk
        index = np.array([[i, j, k] for i, j, k in sorted(index, key=lambda x: float(x[0]))])
        index = index[-numk:]

        op_emb = []
        op_counts = []
        gene_num = 0
        for _, op_name, op_n in index:
            op_name = int(op_name)
            op_n = int(op_n)
            op_emb.append(self.all_data[op_n][3][op_name])
            op_count = torch.as_tensor(self.all_data[op_n][1][op_name], dtype=torch.float)
            op_count = torch.log10(op_count[self.gene_filter] + 1)
            op_count = (op_count - self.min) / (self.max - self.min + 1e-8)
            op_counts.append(op_count)
            gene_num = len(op_count)

        return torch.stack(op_emb).view(numk, -1), torch.stack(op_counts).view(numk, gene_num)

    def generate(self, idx):

        idx = self.map[idx]
        img, counts, coord, emb, index = self.all_data[idx[0]]
        counts, coord, emb, index = counts[idx[1]], coord[idx[1]], emb[idx[1]], index[idx[1]]

        emb = emb.unsqueeze(0)

        y, x = coord
        window = self.args.size
        pos = [x // window, y // window]

        op_emb, op_counts = self.retrive_similer(index)

        img = img[(y + (-window // 2)):(y + (window // 2)), (x + (-window // 2)):(x + (window // 2)), :]

        if self.transform != None:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1) / 255

        counts = torch.log10(torch.as_tensor(counts[self.gene_filter], dtype=torch.float) + 1)
        counts = (counts - self.min) / (self.max - self.min + 1e-8)

        return {
            "img": img,
            "count": counts,
            "p_feature": emb,
            "pos_count": op_counts,
            "pos_feature": op_emb,
            "pos": torch.LongTensor(pos),
        }

    def __getitem__(self, index):
        return self.generate(index)

    def __len__(self):
        return len(self.map)


class CSCCDataset(Dataset):
    def __init__(self, SquamousCellCarcinoma, index_filter, transform, args, train=None):
        self.SquamousCellCarcinoma = SquamousCellCarcinoma
        self.gene_names = list(np.load(r'D:\dataset\CSCC_data\GSE144240_RAW/skin_hvg_cut_1000.npy', allow_pickle=True))
        self.args = args
        self.train = train
        self.index_filter = index_filter
        self.data = self.load_raw(args.data)
        self.meta_info(args.data)
        self.transform = transform

        keep_idx = list(range(len(self.mean)))
        keep = set(keep_idx)

        self.filter_name = [j for i, j in enumerate(self.gene_names) if i in keep]
        self.gene_filter = np.array([i in keep for i in range(len(self.gene_names))])
        self.max = torch.log10(torch.as_tensor(self.max[self.gene_filter], dtype=torch.float) + 1)
        self.min = torch.log10(torch.as_tensor(self.min[self.gene_filter], dtype=torch.float) + 1)
        del self.data

        mapping = []
        for i in SquamousCellCarcinoma:
            img, counts, coord, emb, pos_index = self.all_data[i]
            for j in range(len(counts)):
                mapping.append([i, j])
        self.map = mapping

    def load_raw(self, data_root):

        data = []
        for idx, file in enumerate(SquamousCellCarcinoma):
            pos_index_path = os.path.join(self.args.index_path, f"{idx}.npy")
            pos_index = np.load(pos_index_path)

            image_path = glob.glob(data_root + '*' + file + '.jpg')[0]
            img = np.array(Image.open(image_path))

            exp_path = glob.glob(data_root + '*' + file + '_stdata.tsv')[0]
            exp_df = pd.read_csv(exp_path, sep='\t', index_col=0)

            pos_path = glob.glob(data_root + '*spot*' + file + '.tsv')[0]
            pos_df = pd.read_csv(pos_path, sep='\t')
            x = pos_df['x'].values
            y = pos_df['y'].values
            x = np.around(x).astype(int)
            y = np.around(y).astype(int)
            id = []
            for i in range(len(x)):
                id.append(str(x[i]) + 'x' + str(y[i]))
            pos_df['id'] = id

            meta = exp_df.join(pos_df.set_index('id'), how='inner')

            data.append([img, meta, idx, pos_index])

        return data

    def meta_info(self, root):

        from tqdm import tqdm

        all_data = {}
        all_gene = []
        part_gene = []
        for img, p, idx, pos_index in tqdm(self.data):
            coord = pd.DataFrame(p[['pixel_x', 'pixel_y']].values, columns=['x_coord', 'y_coord'], index=p.index)
            c = p[self.gene_names].values.astype(float)

            emb = torch.load(f"{self.args.emb_path}/{idx}.pt", map_location=torch.device("cpu"))

            assert emb.size(0) == c.shape[0]

            all_data[idx] = [img, c, coord.values.astype(int), emb, pos_index]

            for i in c:
                all_gene.append(i)
                if idx in self.SquamousCellCarcinoma:
                    part_gene.append(i)

        all_gene = np.array(all_gene)
        part_gene = np.array(part_gene)
        print(all_gene.shape, part_gene.shape)

        self.mean = np.mean(all_gene, 0)
        self.max = np.max(part_gene, 0)
        self.min = np.min(part_gene, 0)
        self.all_data = all_data

    def retrive_similer(self, pos_index):
        numk = self.args.numk
        pos_index = np.array([[i, j, k] for i, j, k in sorted(pos_index, key=lambda x: float(x[0]))])
        pos_index = pos_index[-numk:]

        pos_emb = []
        pos_counts = []
        gene_num = 0
        for _, op_name, op_n in pos_index:
            op_name = int(op_name)
            op_n = int(op_n)
            pos_emb.append(self.all_data[op_n][3][op_name])
            op_count = torch.as_tensor(self.all_data[op_n][1][op_name], dtype=torch.float)
            op_count = torch.log10(op_count[self.gene_filter] + 1)
            op_count = (op_count - self.min) / (self.max - self.min + 1e-8)
            gene_num = len(op_count)
            pos_counts.append(op_count)
        pos_emb = torch.stack(pos_emb).view(numk, -1)
        pos_counts = torch.stack(pos_counts).view(numk, gene_num)

        # pos_index = sorted(pos_index, key=lambda x: float(x[0]))[-1:]  # 取最后一个相似度最高的
        # _, op_name, op_n = pos_index[0]
        # op_name, op_n = int(op_name), int(op_n)
        #
        # # 获取该节点对应的表达值
        # op_count = torch.as_tensor(self.all_data[int(op_n)][1][int(op_name)], dtype=torch.float)
        # op_count = torch.log10(op_count[self.gene_filter] + 1)
        # op_count = (op_count - self.min) / (self.max - self.min + 1e-8)  # (gene_num,)
        #
        # adata_sing_cell = sc.read(
        #     rf"D:\dataset\CSCC_data\GSE144240_RAW/{SquamousCellCarcinoma[self.SquamousCellCarcinoma[0]][:2]}.h5ad")
        #
        # ref_expr = op_count.unsqueeze(0)  # shape (1, gene_num)
        # sc_expr = torch.tensor(adata_sing_cell.X)  # shape (n_cells, gene_num)
        # sc_expr = torch.log10(sc_expr + 1)
        # sc_expr = (sc_expr - self.min) / (self.max - self.min + 1e-8)
        #
        # ref_norm = torch.nn.functional.normalize(ref_expr, dim=1)
        # sc_norm = torch.nn.functional.normalize(sc_expr, dim=1)
        # sim_scores = torch.mm(ref_norm, sc_norm.T).squeeze(0)  # shape: (n_cells,)
        #
        # topk_indices = torch.topk(sim_scores, self.args.numk).indices
        #
        # sc_counts = sc_expr[topk_indices]

        return pos_emb, pos_counts  # , sc_counts

    def generate(self, idx):

        idx = self.map[idx]
        img, counts, coord, emb, pos_index = self.all_data[idx[0]]
        counts, coord, emb, pos_index = counts[idx[1]], coord[idx[1]], emb[idx[1]], pos_index[idx[1]]

        emb = emb.unsqueeze(0)

        x, y = coord
        window = self.args.size
        pos = [x // window, y // window]

        pos_emb, pos_counts = self.retrive_similer(pos_index)  # , sc_counts

        img = img[(y + (-window // 2)):(y + (window // 2)), (x + (-window // 2)):(x + (window // 2)), :]

        if self.transform != None:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1) / 255

        counts = torch.log10(torch.as_tensor(counts[self.gene_filter], dtype=torch.float) + 1)
        counts = (counts - self.min) / (self.max - self.min + 1e-8)
        return {
            "img": img,
            "count": counts,
            "p_feature": emb,
            "pos_count": pos_counts,
            "pos_feature": pos_emb,
            "pos": torch.LongTensor(pos)
            # "sc_counts": sc_counts,
        }

    def __getitem__(self, index):
        return self.generate(index)

    def __len__(self):
        return len(self.map)


class HER2AddDataset(Dataset):
    def __init__(self, HER2Add_patients, index_filter, transform, args, train=None):
        self.HER2Add_patients = HER2Add_patients
        self.cnt_dir = 'D:\dataset\Her2st\data/ST-cnts'
        self.img_dir = 'D:\dataset\Her2st\data/ST-imgs'
        self.pos_dir = 'D:\dataset\Her2st\data/ST-spotfiles'
        self.lbl_dir = 'D:\dataset\Her2st\data/her2st/data/ST-pat/lbl'
        self.args = args
        self.train = train
        self.index_filter = index_filter
        self.data = self.load_raw(args.data)
        self.meta_info(args.data)
        self.transform = transform

        keep_idx = list(range(len(self.mean)))
        keep = set(keep_idx)

        self.filter_name = [j for i, j in enumerate(self.gene_names) if i in keep]
        self.gene_filter = np.array([i in keep for i in range(len(self.gene_names))])
        self.max = torch.log10(torch.as_tensor(self.max[self.gene_filter], dtype=torch.float) + 1)
        self.min = torch.log10(torch.as_tensor(self.min[self.gene_filter], dtype=torch.float) + 1)
        del self.data
        mapping = []
        for i in HER2Add_patients:
            img, counts, coord, emb, pos_index = self.all_data[i]
            for j in range(len(counts)):
                mapping.append([i, j])
        self.map = mapping

    def load_raw(self, data_root):

        data = []
        for idx, file in enumerate(HER2Add_patients):
            pos_index_path = os.path.join(self.args.index_path, f"{idx}.npy")
            pos_index = np.load(pos_index_path)

            pre = self.img_dir + '/' + file[0] + '/' + file
            fig_name = os.listdir(pre)[0]
            image_path = pre + '/' + fig_name
            img = np.array(Image.open(image_path))

            exp_path = self.cnt_dir + '/' + file + '.tsv'
            exp_df = pd.read_csv(exp_path, sep='\t', index_col=0)

            pos_path = self.pos_dir + '/' + file + '_selection.tsv'
            pos_df = pd.read_csv(pos_path, sep='\t')
            x = pos_df['x'].values
            y = pos_df['y'].values
            x = np.around(x).astype(int)
            y = np.around(y).astype(int)
            id = []
            for i in range(len(x)):
                id.append(str(x[i]) + 'x' + str(y[i]))
            pos_df['id'] = id

            meta = exp_df.join(pos_df.set_index('id'), how='inner')

            data.append([img, meta, idx, pos_index])

        return data

    def meta_info(self, root):

        from tqdm import tqdm
        gene_names = list(np.load(r'D:\dataset\Her2st\data/her_hvg_cut_1000.npy', allow_pickle=True))

        all_data = {}
        all_gene = []
        part_gene = []
        for img, p, idx, pos_index in tqdm(self.data):

            coord = pd.DataFrame(p[['pixel_x', 'pixel_y']].values, columns=['x_coord', 'y_coord'], index=p.index)
            c = scp.transform.log(scp.normalize.library_size_normalize(p)).values.astype(float)

            emb = torch.load(f"{self.args.emb_path}/{idx}.pt", map_location=torch.device("cpu"))

            assert emb.size(0) == c.shape[0]

            all_data[idx] = [img, c, coord.values.astype(int), emb, pos_index]

            for i in c:
                all_gene.append(i)
                if idx in self.HER2Add_patients:
                    part_gene.append(i)

        all_gene = np.array(all_gene)
        part_gene = np.array(part_gene)
        print(all_gene.shape, part_gene.shape)

        self.mean = np.mean(all_gene, 0)
        self.max = np.max(part_gene, 0)
        self.min = np.min(part_gene, 0)
        self.gene_names = gene_names
        self.all_data = all_data

    def retrive_similer(self, pos_index):
        numk = self.args.numk
        pos_index = np.array([[i, j, k] for i, j, k in sorted(pos_index, key=lambda x: float(x[0]))])
        pos_index = pos_index[-numk:]

        pos_emb = []
        pos_counts = []
        gene_num = 0
        for _, op_name, op_n in pos_index:
            op_name = int(op_name)
            op_n = int(op_n)
            pos_emb.append(self.all_data[op_n][3][op_name])
            op_count = torch.as_tensor(self.all_data[op_n][1][op_name], dtype=torch.float)
            op_count = torch.log10(op_count[self.gene_filter] + 1)
            op_count = (op_count - self.min) / (self.max - self.min + 1e-8)
            gene_num = len(op_count)
            pos_counts.append(op_count)

        return torch.stack(pos_emb).view(numk, -1), torch.stack(pos_counts).view(numk, gene_num)

    def generate(self, idx):

        idx = self.map[idx]
        img, counts, coord, emb, pos_index = self.all_data[idx[0]]
        counts, coord, emb, pos_index = counts[idx[1]], coord[idx[1]], emb[idx[1]], pos_index[idx[1]]

        emb = emb.unsqueeze(0)

        x, y = coord
        window = self.args.size
        pos = [x // window, y // window]

        pos_emb, pos_counts = self.retrive_similer(pos_index)

        img = img[(y + (-window // 2)):(y + (window // 2)), (x + (-window // 2)):(x + (window // 2)), :]

        if self.transform != None:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1) / 255

        counts = torch.log10(torch.as_tensor(counts[self.gene_filter], dtype=torch.float) + 1)
        counts = (counts - self.min) / (self.max - self.min + 1e-8)

        return {
            "img": img,
            "count": counts,
            "p_feature": emb,
            "pos_count": pos_counts,
            "pos_feature": pos_emb,
            "pos": torch.LongTensor(pos),
        }

    def __getitem__(self, index):
        return self.generate(index)

    def __len__(self):
        return len(self.map)


class HESTCerDataset(Dataset):
    def __init__(self, HEST_Lymph_Node, index_filter, transform, args, train=None):
        self.HEST_Lymph_Node = HEST_Lymph_Node

        self.args = args
        self.train = train
        self.index_filter = index_filter
        self.root = './hest_data'
        self.data = self.load_raw(args.data)
        self.meta_info(args.data)
        self.transform = transform

        keep_idx = list(range(len(self.mean)))
        keep = set(keep_idx)

        self.filter_name = [j for i, j in enumerate(self.gene_names) if i in keep]
        self.gene_filter = np.array([i in keep for i in range(len(self.gene_names))])
        self.max = torch.log10(torch.as_tensor(self.max[self.gene_filter], dtype=torch.float) + 1)
        self.min = torch.log10(torch.as_tensor(self.min[self.gene_filter], dtype=torch.float) + 1)
        del self.data

        mapping = []
        for i in HEST_Lymph_Node:
            img, counts, coord, emb, index = self.all_data[i]
            for j in range(len(counts)):
                mapping.append([i, j])
        self.map = mapping

    def load_raw(self, data_root):

        data = []

        for idx, st in enumerate(iter_hest(self.root, id_list=HEST_Lymph_Node)):
            index = os.path.join(self.args.index_path, f"{idx}.npy")
            index = np.load(index)

            adata = st.adata
            img = st.wsi
            adata.var_names_make_unique()

            data.append([img, adata, idx, index])

        return data

    def meta_info(self, root):

        from tqdm import tqdm
        if not os.path.exists(r'D:\baseline\SpaMGCL-main\v2\HEST_Lymph_Node/com_gene.npy'):
            gene_names = None
            for _, p, _, _ in tqdm(self.data):
                sc.pp.highly_variable_genes(p, flavor="seurat_v3", n_top_genes=1000)
                hvg = set(p.var[p.var.highly_variable].index.tolist())
                if gene_names is None:
                    gene_names = hvg
                else:
                    gene_names = gene_names.intersection(hvg)

            gene_names = list(gene_names)
            np.save(r'D:\baseline\SpaMGCL-main\v2\HEST_Lymph_Node/com_gene.npy', np.array(gene_names))
        else:
            gene_names = list(
                np.load(r'D:\baseline\SpaMGCL-main\v2\HEST_Lymph_Node/com_gene.npy', allow_pickle=True))

        all_data = {}
        all_gene = []
        part_gene = []
        for img, p, idx, index in tqdm(self.data):
            counts = pd.DataFrame(p.X.todense(), columns=p.var_names, index=p.obs_names)  # .todense()
            coord = pd.DataFrame(p.obsm['spatial'], columns=['x_coord', 'y_coord'], index=p.obs_names)

            c = counts[gene_names].values.astype(np.float32)

            emb = torch.load(f"{self.args.emb_path}/{idx}.pt", map_location=torch.device("cpu"))

            assert emb.size(0) == c.shape[0]

            all_data[idx] = [img, c, coord.values.astype(int), emb, index]

            for i in c:
                all_gene.append(i)
                if idx in self.HEST_Lymph_Node:
                    part_gene.append(i)

        all_gene = np.array(all_gene)
        part_gene = np.array(part_gene)
        print(all_gene.shape, part_gene.shape)

        self.mean = np.mean(all_gene, 0)
        self.max = np.max(part_gene, 0)
        self.min = np.min(part_gene, 0)
        self.gene_names = gene_names
        self.all_data = all_data

        gene_path = r"D:\baseline\SpaMGCL-main\v2\HEST_Lymph_Node/ComGene.txt"
        if not os.path.exists(gene_path):
            with open(gene_path, "w") as f:
                for name in self.gene_names:
                    f.write(name + "\n")

    def retrive_similer(self, index):
        numk = self.args.numk
        index = np.array([[i, j, k] for i, j, k in sorted(index, key=lambda x: float(x[0]))])
        index = index[-numk:]

        op_emb = []
        op_counts = []
        gene_num = 0
        for _, op_name, op_n in index:
            op_name = int(op_name)
            op_n = int(op_n)
            op_emb.append(self.all_data[op_n][3][op_name])
            op_count = torch.as_tensor(self.all_data[op_n][1][op_name], dtype=torch.float)
            op_count = torch.log10(op_count[self.gene_filter] + 1)
            op_count = (op_count - self.min) / (self.max - self.min + 1e-8)
            op_counts.append(op_count)
            gene_num = len(op_count)

        return torch.stack(op_emb).view(numk, -1), torch.stack(op_counts).view(numk, gene_num)

    def generate(self, idx):

        idx = self.map[idx]
        img, counts, coord, emb, index = self.all_data[idx[0]]
        counts, coord, emb, index = counts[idx[1]], coord[idx[1]], emb[idx[1]], index[idx[1]]

        emb = emb.unsqueeze(0)
        x, y = coord
        window = self.args.size
        pos = [x // window, y // window]
        op_emb, op_counts = self.retrive_similer(index)
        counts = torch.log10(torch.as_tensor(counts[self.gene_filter], dtype=torch.float) + 1)
        counts = (counts - self.min) / (self.max - self.min + 1e-8)

        return {
            "count": counts,
            "p_feature": emb,
            "pos_count": op_counts,
            "pos_feature": op_emb,
            "pos": torch.LongTensor(pos),
        }

    def __getitem__(self, index):
        return self.generate(index)

    def __len__(self):
        return len(self.map)


class AlexDataset(Dataset):
    def __init__(self, Alex_Nat_Slice, index_filter, transform, args, train=None):
        self.Alex_Nat_Slice = Alex_Nat_Slice
        self.args = args
        self.train = train
        self.index_filter = index_filter
        self.data = self.load_raw(args.data)
        self.meta_info(args.data)
        self.transform = transform

        keep_idx = list(range(len(self.mean)))
        keep = set(keep_idx)

        self.filter_name = [j for i, j in enumerate(self.gene_names) if i in keep]
        self.gene_filter = np.array([i in keep for i in range(len(self.gene_names))])
        self.max = torch.log10(torch.as_tensor(self.max[self.gene_filter], dtype=torch.float) + 1)
        self.min = torch.log10(torch.as_tensor(self.min[self.gene_filter], dtype=torch.float) + 1)
        del self.data
        mapping = []
        for i in Alex_Nat_Slice:
            img, counts, coord, emb, index = self.all_data[i]
            for j in range(len(counts)):
                mapping.append([i, j])
        self.map = mapping

    def load_raw(self, data_root):

        data = []
        for idx, file in enumerate(Alex_Nat_Slice):
            index = os.path.join(self.args.index_path, f"{idx}.npy")
            index = np.load(index)

            path = os.path.join(data_root, file)

            img_fild = os.path.join(path, 'image.tif')
            img = np.array(Image.open(img_fild))
            adata_path = os.path.join(path, f"{file}.h5ad")
            adata = sc.read_h5ad(adata_path)
            adata.var_names_make_unique()

            data.append([img, adata, idx, index])

        return data

    def meta_info(self, root):

        from tqdm import tqdm
        if not os.path.exists(r'D:\baseline\SpaMGCL-main\v2\Alex_Nat/com_gene.npy'):
            gene_names = None
            for _, p, _, _ in tqdm(self.data):
                sc.pp.highly_variable_genes(p, flavor="seurat_v3", n_top_genes=1000)
                hvg = set(p.var[p.var.highly_variable].index.tolist())
                if gene_names is None:
                    gene_names = hvg
                else:
                    gene_names = gene_names.intersection(hvg)

            gene_names = list(gene_names)
            np.save(r'D:\baseline\SpaMGCL-main\v2\Alex_Nat/com_gene.npy', np.array(gene_names))
        else:
            gene_names = list(np.load(r'D:\baseline\SpaMGCL-main\v2\Alex_Nat/com_gene.npy', allow_pickle=True))

        all_data = {}
        all_gene = []
        part_gene = []
        for img, p, idx, index in tqdm(self.data):
            counts = pd.DataFrame(p.X.todense(), columns=p.var_names, index=p.obs_names)
            coord = pd.DataFrame(p.obsm['spatial'], columns=['x_coord', 'y_coord'], index=p.obs_names)

            c = counts[gene_names].values.astype(float)

            emb = torch.load(f"{self.args.emb_path}/{idx}.pt", map_location=torch.device("cpu"))

            assert emb.size(0) == c.shape[0]

            all_data[idx] = [img, c, coord.values.astype(int), emb, index]

            for i in c:
                all_gene.append(i)
                if idx in self.Alex_Nat_Slice:
                    part_gene.append(i)

        all_gene = np.array(all_gene)
        part_gene = np.array(part_gene)
        print(all_gene.shape, part_gene.shape)

        self.mean = np.mean(all_gene, 0)
        self.max = np.max(part_gene, 0)
        self.min = np.min(part_gene, 0)
        self.gene_names = gene_names
        self.all_data = all_data

        gene_path = r"D:\baseline\SpaMGCL-main\v2\Alex_Nat/ComGene.txt"
        if not os.path.exists(gene_path):
            with open(gene_path, "w") as f:
                for name in self.gene_names:
                    f.write(name + "\n")

    def retrive_similer(self, index):
        numk = self.args.numk
        index = np.array([[i, j, k] for i, j, k in sorted(index, key=lambda x: float(x[0]))])
        index = index[-numk:]

        op_emb = []
        op_counts = []
        gene_num = 0
        for _, op_name, op_n in index:
            op_name = int(op_name)
            op_n = int(op_n)
            op_emb.append(self.all_data[op_n][3][op_name])
            op_count = torch.as_tensor(self.all_data[op_n][1][op_name], dtype=torch.float)
            op_count = torch.log10(op_count[self.gene_filter] + 1)
            op_count = (op_count - self.min) / (self.max - self.min + 1e-8)
            op_counts.append(op_count)
            gene_num = len(op_count)

        return torch.stack(op_emb).view(numk, -1), torch.stack(op_counts).view(numk, gene_num)

    def generate(self, idx):

        idx = self.map[idx]
        img, counts, coord, emb, index = self.all_data[idx[0]]
        counts, coord, emb, index = counts[idx[1]], coord[idx[1]], emb[idx[1]], index[idx[1]]

        emb = emb.unsqueeze(0)

        y, x = coord
        window = self.args.size
        pos = [y // window, x // window]

        op_emb, op_counts = self.retrive_similer(index)

        img = img[(y + (-window // 2)):(y + (window // 2)), (x + (-window // 2)):(x + (window // 2)), :]

        if self.transform != None:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1) / 255

        counts = torch.log10(torch.as_tensor(counts[self.gene_filter], dtype=torch.float) + 1)
        counts = (counts - self.min) / (self.max - self.min + 1e-8)

        return {
            "img": img,
            "count": counts,
            "p_feature": emb,
            "pos_count": op_counts,
            "pos_feature": op_emb,
            "pos": torch.LongTensor(pos),
        }

    def __getitem__(self, index):
        return self.generate(index)

    def __len__(self):
        return len(self.map)


class HESTXeniumDataset(Dataset):
    def __init__(self, HEST_Pancreas_xenium, index_filter, transform, args, train=None):
        self.HEST_Pancreas_xenium = HEST_Pancreas_xenium

        self.args = args
        self.train = train
        self.index_filter = index_filter
        self.root = './hest_data'
        self.data = self.load_raw(args.data)
        self.meta_info(args.data)
        self.transform = transform

        keep_idx = list(range(len(self.mean)))
        keep = set(keep_idx)

        self.filter_name = [j for i, j in enumerate(self.gene_names) if i in keep]
        self.gene_filter = np.array([i in keep for i in range(len(self.gene_names))])
        self.max = torch.log10(torch.as_tensor(self.max[self.gene_filter], dtype=torch.float) + 1)
        self.min = torch.log10(torch.as_tensor(self.min[self.gene_filter], dtype=torch.float) + 1)
        del self.data

        mapping = []
        for i in HEST_Pancreas_xenium:
            img, counts, coord, emb, index = self.all_data[i]
            for j in range(len(counts)):
                mapping.append([i, j])
        self.map = mapping

    def load_raw(self, data_root):

        data = []
        root = r"D:\dataset\Xenium/"
        for idx, st in enumerate(iter_hest(self.root, id_list=HEST_Pancreas_xenium)):
            index = os.path.join(self.args.index_path, f"{idx}.npy")
            index = np.load(index)

            adata = sc.read_h5ad(root + f'{HEST_Pancreas_xenium[idx]}.h5ad')
            img = st.wsi
            adata.var_names_make_unique()

            data.append([img, adata, idx, index])

        return data

    def meta_info(self, root):

        from tqdm import tqdm
        if not os.path.exists(r'D:\baseline\SpaMGCL-main\v2\HEST_Pancreas_xenium2/com_gene.npy'):
            gene_names = None
            for _, p, _, _ in tqdm(self.data):
                sc.pp.highly_variable_genes(p, flavor="seurat_v3", n_top_genes=1000)
                hvg = set(p.var[p.var.highly_variable].index.tolist())
                if gene_names is None:
                    gene_names = hvg
                else:
                    gene_names = gene_names.intersection(hvg)

            gene_names = list(gene_names)
            np.save(r'D:\baseline\SpaMGCL-main\v2\HEST_Pancreas_xenium2/com_gene.npy', np.array(gene_names))
        else:
            gene_names = list(
                np.load(r'D:\baseline\SpaMGCL-main\v2\HEST_Pancreas_xenium2/com_gene.npy', allow_pickle=True))

        all_data = {}
        all_gene = []
        part_gene = []
        for img, p, idx, index in tqdm(self.data):
            counts = pd.DataFrame(p.X, columns=p.var_names, index=p.obs_names)  # .todense()
            coord = pd.DataFrame(p.obsm['spatial'], columns=['x_coord', 'y_coord'], index=p.obs_names)

            c = counts[gene_names].values.astype(np.float32)

            emb = torch.load(f"{self.args.emb_path}/{idx}.pt", map_location=torch.device("cpu"))

            assert emb.size(0) == c.shape[0]

            all_data[idx] = [img, c, coord.values.astype(int), emb, index]

            for i in c:
                all_gene.append(i)
                if idx in self.HEST_Pancreas_xenium:
                    part_gene.append(i)

        all_gene = np.array(all_gene)
        part_gene = np.array(part_gene)
        print(all_gene.shape, part_gene.shape)

        self.mean = np.mean(all_gene, 0)
        self.max = np.max(part_gene, 0)
        self.min = np.min(part_gene, 0)
        self.gene_names = gene_names
        self.all_data = all_data

        gene_path = r"D:\baseline\SpaMGCL-main\v2\HEST_Pancreas_xenium2/ComGene.txt"
        if not os.path.exists(gene_path):
            with open(gene_path, "w") as f:
                for name in self.gene_names:
                    f.write(name + "\n")

    def retrive_similer(self, index):
        numk = self.args.numk
        index = np.array([[i, j, k] for i, j, k in sorted(index, key=lambda x: float(x[0]))])
        index = index[-numk:]

        op_emb = []
        op_counts = []
        gene_num = 0
        for _, op_name, op_n in index:
            op_name = int(op_name)
            op_n = int(op_n)
            op_emb.append(self.all_data[op_n][3][op_name])
            op_count = torch.as_tensor(self.all_data[op_n][1][op_name], dtype=torch.float)
            op_count = torch.log10(op_count[self.gene_filter] + 1)
            op_count = (op_count - self.min) / (self.max - self.min + 1e-8)
            op_counts.append(op_count)
            gene_num = len(op_count)

        return torch.stack(op_emb).view(numk, -1), torch.stack(op_counts).view(numk, gene_num)

    def generate(self, idx):

        idx = self.map[idx]
        img, counts, coord, emb, index = self.all_data[idx[0]]
        counts, coord, emb, index = counts[idx[1]], coord[idx[1]], emb[idx[1]], index[idx[1]]

        emb = emb.unsqueeze(0)
        x, y = coord
        window = self.args.size
        pos = [x // window, y // window]
        op_emb, op_counts = self.retrive_similer(index)
        counts = torch.log10(torch.as_tensor(counts[self.gene_filter], dtype=torch.float) + 1)
        counts = (counts - self.min) / (self.max - self.min + 1e-8)

        return {
            "count": counts,
            "p_feature": emb,
            "pos_count": op_counts,
            "pos_feature": op_emb,
            "pos": torch.LongTensor(pos),
        }

    def __getitem__(self, index):
        return self.generate(index)

    def __len__(self):
        return len(self.map)
