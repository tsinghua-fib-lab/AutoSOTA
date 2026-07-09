import json
import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from pymatgen.core.structure import Molecule

# -------------------------- 配置参数 --------------------------
CUTOFF_RADIUS = 5.0       # 截断半径
NUM_BOND_FEATURES = 100   # 边特征高斯展开维度
MU_MAX = 6.0              # 高斯中心最大值
SIGMA = 0.5               # 高斯标准差

# -------------------------- 工具函数 --------------------------
def gaussian_expand_distance(distances: np.ndarray) -> np.ndarray:
    """对距离进行高斯展开，用作边特征"""
    mus = np.linspace(0, MU_MAX, NUM_BOND_FEATURES)
    expanded = np.exp(-((distances[:, np.newaxis] - mus) ** 2) / (SIGMA ** 2))
    return expanded.astype(np.float32)

def build_molecule_graph(mol: Molecule, max_atomic_num=16):
    """
    将pymatgen Molecule对象转换为PyG图数据
    max_atomic_num: QM7b原子类型最大值，默认16足够
    """
    atomic_numbers = np.array([site.specie.Z for site in mol.sites], dtype=np.int64)
    
    # 节点特征 one-hot
    x = np.zeros((len(atomic_numbers), max_atomic_num), dtype=np.float32)
    for i, z in enumerate(atomic_numbers):
        if z - 1 < max_atomic_num:
            x[i, z - 1] = 1.0
    x = torch.tensor(x, dtype=torch.float32)

    # 构建边索引和边特征
    num_atoms = len(mol.sites)
    edge_indices = []
    distances = []

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = mol.get_distance(i, j)
            if dist <= CUTOFF_RADIUS:
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                distances.append(dist)
                distances.append(dist)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    if len(distances) == 0:
        edge_attr = torch.zeros(0, NUM_BOND_FEATURES, dtype=torch.float32)
    else:
        edge_attr = torch.tensor(gaussian_expand_distance(np.array(distances, dtype=np.float32)))

    return x, edge_index, edge_attr
from torch_geometric.data.data import DataEdgeAttr
# -------------------------- 自定义 PyG Dataset --------------------------
class QM7bDataset(InMemoryDataset):
    def __init__(self, root, raw_json_path=None, transform=None, pre_transform=None):
        self.raw_json_path = raw_json_path
        super().__init__(root, transform, pre_transform)
        torch.serialization.add_safe_globals([DataEdgeAttr])
        processed_file = os.path.join(root, "data.pt")
        if os.path.exists(processed_file):
            self.data_list = torch.load(processed_file, weights_only=False)
        else:
            self.data_list = self.process_data()
            os.makedirs(root, exist_ok=True)
            torch.save(self.data_list, processed_file)

    def process_data(self):
        with open(self.raw_json_path, "r") as f:
            raw_data = json.load(f)

        molecules = raw_data["molecules"]
        targets = raw_data["targets"]
        mol_ids = sorted(molecules.keys())

        data_list = []
        for mol_id in mol_ids:
            mol = Molecule.from_dict(molecules[mol_id])
            x, edge_index, edge_attr = build_molecule_graph(mol)
            tgt_dict = targets[mol_id]
            flat_targets = []
            target_names = []
            for method, basis_dict in tgt_dict.items():
                for basis, value in basis_dict.items():
                    flat_targets.append(value)
                    target_names.append(f"{method}_{basis}")

            y = torch.tensor(flat_targets, dtype=torch.float32)  # [num_targets]

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,  # 保留边特征
                y=y,
                mol_id=f"qm7b_{mol_id}",
                target_names=target_names
            )
            data_list.append(data)

        print(f"处理完成 {len(data_list)} 个分子，每个分子有 {len(data_list[0].y)} 个标签")
        return data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset


class FidelitySubset(Dataset):
    """
    包装 Dataset,按指定索引和 fidelity 返回对应 y
    """
    def __init__(self, dataset, indices, target_cols, fidelity):
        self.dataset = dataset
        self.indices = indices
        self.target_cols = target_cols  # 可以是单个列或者列表
        self.fidelity = fidelity        # 单个整数或列表，长度和 indices 相同

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        data = self.dataset[orig_idx]
        # 根据 target_cols 选择对应 fidelity 的 y
        if isinstance(self.target_cols, int):
            y = torch.tensor(data.y[self.target_cols])
        else:
            y = torch.tensor(data.y[self.target_cols[idx]])
        new_data = data.clone() if hasattr(data, "clone") else data  # 避免修改原始 data
        new_data.y = y
        new_data.fidelity = self.fidelity[idx] if isinstance(self.fidelity, (list, np.ndarray)) else self.fidelity
        return new_data

class QM7bDataModule:
    def __init__(self, dataset, mode="3_fi", ccsd_num=100, ratios=None, batch_size=32, seed=42, test_num=1000):
        self.dataset = dataset
        self.mode = mode
        self.ccsd_num = ccsd_num
        self.ratios = ratios if ratios else ([1,2] if mode=="2_fi" else [1,2,4])
        self.batch_size = batch_size
        self.seed = seed
        self.test_num = test_num

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.pretrain_loader = None
        self.pretrain_val_loader = None
        self.finetune_loader = None
        self.finetune_val_loader = None

        self.prepare_data()

    def prepare_data(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        N = len(self.dataset)
        all_idx = np.arange(N)

        # ------------------ 测试集（CCSD） ------------------
        ccsd_col = self.dataset[0].target_names.index("CCSD(T)_ccpvdz")
        test_idx = np.random.choice(all_idx, size=self.test_num, replace=False)

        remaining_idx = np.setdiff1d(all_idx, test_idx)

        if self.mode == "1_fi":
            self._prepare_1fi(remaining_idx, test_idx, ccsd_col)
        elif self.mode == "2_fi":
            mp2_col = self.dataset[0].target_names.index("MP2_ccpvdz")
            self._prepare_2fi(remaining_idx, test_idx, ccsd_col, mp2_col)
        elif self.mode == "3_fi":
            mp2_col = self.dataset[0].target_names.index("MP2_ccpvdz")
            hf_col = self.dataset[0].target_names.index("HF_ccpvdz")
            self._prepare_3fi(remaining_idx, test_idx, ccsd_col, mp2_col, hf_col)
        elif self.mode in ["mp2_transfer", "hf_transfer"]:
            self._prepare_transfer(remaining_idx, test_idx,self.mode)
        else:
            raise ValueError("Invalid mode!")

    # --------------------------- 1_fi ---------------------------
    def _prepare_1fi(self, remaining_idx, test_idx, ccsd_col):
        np.random.shuffle(remaining_idx)
        ccsd_idx = remaining_idx[:self.ccsd_num]
        n_train = int(0.8*len(ccsd_idx))
        train_idx = ccsd_idx[:n_train]
        val_idx = ccsd_idx[n_train:]

        self.train_loader = DataLoader(FidelitySubset(self.dataset, train_idx, ccsd_col, fidelity=0),
                                       batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(FidelitySubset(self.dataset, val_idx, ccsd_col, fidelity=0),
                                     batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(FidelitySubset(self.dataset, test_idx, ccsd_col, fidelity=0),
                                      batch_size=self.batch_size, shuffle=False)

    # --------------------------- 2_fi ---------------------------
    def _prepare_2fi(self, remaining_idx, test_idx, ccsd_col, mp2_col):
        np.random.shuffle(remaining_idx)
        ccsd_idx = remaining_idx[:self.ccsd_num]
        n_train_ccsd = int(0.8*len(ccsd_idx))
        train_ccsd_idx = ccsd_idx[:n_train_ccsd]
        val_ccsd_idx = ccsd_idx[n_train_ccsd:]

        mp2_num = int(self.ccsd_num * self.ratios[1]/self.ratios[0])
        mp2_idx = np.random.choice(np.arange(len(self.dataset)), size=mp2_num, replace=False)
        n_train_mp2 = int(0.8*mp2_num)
        train_mp2_idx = mp2_idx[:n_train_mp2]
        val_mp2_idx = mp2_idx[n_train_mp2:]

        train_idx = np.concatenate([train_ccsd_idx, train_mp2_idx])
        val_idx = np.concatenate([val_ccsd_idx, val_mp2_idx])
        train_cols = [ccsd_col]*len(train_ccsd_idx) + [mp2_col]*len(train_mp2_idx)
        val_cols = [ccsd_col]*len(val_ccsd_idx) + [mp2_col]*len(val_mp2_idx)
        train_fidelity = [0]*len(train_ccsd_idx) + [1]*len(train_mp2_idx)
        val_fidelity = [0]*len(val_ccsd_idx) + [1]*len(val_mp2_idx)

        self.train_loader = DataLoader(FidelitySubset(self.dataset, train_idx, train_cols, train_fidelity),
                                       batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(FidelitySubset(self.dataset, val_idx, val_cols, val_fidelity),
                                     batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(FidelitySubset(self.dataset, test_idx, ccsd_col, fidelity=0),
                                      batch_size=self.batch_size, shuffle=False)

    # --------------------------- 3_fi ---------------------------
    def _prepare_3fi(self, remaining_idx, test_idx, ccsd_col, mp2_col, hf_col):
        np.random.shuffle(remaining_idx)
        ccsd_idx = remaining_idx[:self.ccsd_num]
        n_train_ccsd = int(0.8*self.ccsd_num)
        train_ccsd_idx = ccsd_idx[:n_train_ccsd]
        val_ccsd_idx = ccsd_idx[n_train_ccsd:]

        mp2_num = int(self.ccsd_num * self.ratios[1]/self.ratios[0])
        mp2_idx = np.random.choice(np.arange(len(self.dataset)), size=mp2_num, replace=False)
        n_train_mp2 = int(0.8*mp2_num)
        train_mp2_idx = mp2_idx[:n_train_mp2]
        val_mp2_idx = mp2_idx[n_train_mp2:]

        hf_num = int(self.ccsd_num * self.ratios[2]/self.ratios[0])
        hf_idx = np.random.choice(np.arange(len(self.dataset)), size=hf_num, replace=False)
        n_train_hf = int(0.8*hf_num)
        train_hf_idx = hf_idx[:n_train_hf]
        val_hf_idx = hf_idx[n_train_hf:]

        train_idx = np.concatenate([train_ccsd_idx, train_mp2_idx, train_hf_idx])
        val_idx = np.concatenate([val_ccsd_idx, val_mp2_idx, val_hf_idx])
        train_cols = [ccsd_col]*len(train_ccsd_idx) + [mp2_col]*len(train_mp2_idx) + [hf_col]*len(train_hf_idx)
        val_cols = [ccsd_col]*len(val_ccsd_idx) + [mp2_col]*len(val_mp2_idx) + [hf_col]*len(val_hf_idx)
        train_fidelity = [0]*len(train_ccsd_idx) + [1]*len(train_mp2_idx) + [2]*len(train_hf_idx)
        val_fidelity = [0]*len(val_ccsd_idx) + [1]*len(val_mp2_idx) + [2]*len(val_hf_idx)

        self.train_loader = DataLoader(FidelitySubset(self.dataset, train_idx, train_cols, train_fidelity),
                                       batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(FidelitySubset(self.dataset, val_idx, val_cols, val_fidelity),
                                     batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(FidelitySubset(self.dataset, test_idx, ccsd_col, fidelity=0),
                                      batch_size=self.batch_size, shuffle=False)
        
    def _prepare_transfer(self, trainval_idx, test_idx, transfer_mode="mp2_transfer"):
        ccsd_col = self.dataset[0].target_names.index("CCSD(T)_ccpvdz")
        transfer_col = self.dataset[0].target_names.index("MP2_ccpvdz") if transfer_mode == "mp2_transfer" else self.dataset[0].target_names.index("HF_ccpvdz")

        np.random.shuffle(trainval_idx)
        ccsd_idx = trainval_idx[:self.ccsd_num]
        n_train = int(0.8 * len(ccsd_idx))
        train_ccsd_idx = ccsd_idx[:n_train]
        val_ccsd_idx = ccsd_idx[n_train:]

        num_transfer = len(ccsd_idx) * 2
        transfer_idx = np.random.choice(np.arange(len(self.dataset)), size=num_transfer, replace=False)
        n_train_transfer = int(0.8 * num_transfer)
        train_transfer_idx = transfer_idx[:n_train_transfer]
        val_transfer_idx = transfer_idx[n_train_transfer:]

        self.pretrain_loader = DataLoader(FidelitySubset(self.dataset, train_transfer_idx, transfer_col, 1), batch_size=self.batch_size, shuffle=True)
        self.pretrain_val_loader = DataLoader(FidelitySubset(self.dataset, val_transfer_idx, transfer_col, 1), batch_size=self.batch_size, shuffle=False)
        self.finetune_loader = DataLoader(FidelitySubset(self.dataset, train_ccsd_idx, ccsd_col, 0), batch_size=self.batch_size, shuffle=True)
        self.finetune_val_loader = DataLoader(FidelitySubset(self.dataset, val_ccsd_idx, ccsd_col, 0), batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(FidelitySubset(self.dataset, test_idx, ccsd_col, 0), batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    import torch
    from collections import Counter

    RAW_JSON_PATH = "./qm7b.json"
    SAVE_DIR = "./qm7b_pyg"

    # 初始化数据集
    dataset = QM7bDataset(root=SAVE_DIR, raw_json_path=RAW_JSON_PATH)

    def print_dataloader_info(loader, name):
        ys = [data.y.item() if isinstance(data.y, torch.Tensor) else data.y for data in loader.dataset]
        fidelities = [data.fidelity for data in loader.dataset]
        print(f"{name} 样本数: {len(loader.dataset)}")
        print(f"{name} fidelity分布: {Counter(fidelities)}")
        print(f"{name} y值范围: min={min(ys):.4f}, max={max(ys):.4f}\n")

    # ------------------- 示例 1: 多保真训练 1_fi -------------------
    dm_1fi = QM7bDataModule(dataset, mode="1_fi", ccsd_num=100, batch_size=32, seed=123)
    print("===== 多保真训练 1_fi =====")
    print_dataloader_info(dm_1fi.train_loader, "训练集")
    print_dataloader_info(dm_1fi.val_loader, "验证集")
    print_dataloader_info(dm_1fi.test_loader, "测试集")

    # ------------------- 示例 2: 多保真训练 2_fi -------------------
    dm_2fi = QM7bDataModule(dataset, mode="2_fi", ccsd_num=100, batch_size=32, seed=123)
    print("===== 多保真训练 2_fi =====")
    print_dataloader_info(dm_2fi.train_loader, "训练集")
    print_dataloader_info(dm_2fi.val_loader, "验证集")
    print_dataloader_info(dm_2fi.test_loader, "测试集")

    # ------------------- 示例 3: 多保真训练 3_fi -------------------
    dm_3fi = QM7bDataModule(dataset, mode="3_fi", ccsd_num=100, batch_size=32, seed=123)
    print("===== 多保真训练 3_fi =====")
    print_dataloader_info(dm_3fi.train_loader, "训练集")
    print_dataloader_info(dm_3fi.val_loader, "验证集")
    print_dataloader_info(dm_3fi.test_loader, "测试集")

    # ------------------- 示例 4: 迁移学习 MP2 -------------------
    dm_transfer_mp2 = QM7bDataModule(dataset, mode="mp2_transfer", ccsd_num=100, batch_size=32, seed=123)
    print("===== 迁移学习 MP2 =====")
    print_dataloader_info(dm_transfer_mp2.pretrain_loader, "预训练训练集")
    print_dataloader_info(dm_transfer_mp2.pretrain_val_loader, "预训练验证集")
    print_dataloader_info(dm_transfer_mp2.finetune_loader, "微调训练集")
    print_dataloader_info(dm_transfer_mp2.finetune_val_loader, "微调验证集")
    print_dataloader_info(dm_transfer_mp2.test_loader, "测试集")

    # ------------------- 示例 5: 迁移学习 HF -------------------
    dm_transfer_hf = QM7bDataModule(dataset, mode="hf_transfer", ccsd_num=100, batch_size=32, seed=123)
    print("===== 迁移学习 HF =====")
    print_dataloader_info(dm_transfer_hf.pretrain_loader, "预训练训练集")
    print_dataloader_info(dm_transfer_hf.pretrain_val_loader, "预训练验证集")
    print_dataloader_info(dm_transfer_hf.finetune_loader, "微调训练集")
    print_dataloader_info(dm_transfer_hf.finetune_val_loader, "微调验证集")
    print_dataloader_info(dm_transfer_hf.test_loader, "测试集")
