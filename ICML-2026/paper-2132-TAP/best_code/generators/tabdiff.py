"""TabDiff 生成器实现"""
import os
import sys
import json
import pickle
import shutil
from pathlib import Path
from typing import List, Optional, Union
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

tabdiff_path = Path(__file__).parent.parent / 'TabDiff'
if str(tabdiff_path) in sys.path:
    sys.path.remove(str(tabdiff_path))
sys.path.insert(0, str(tabdiff_path))

import src

import importlib.util
_utils_train_spec = importlib.util.spec_from_file_location("tabdiff_utils_train", tabdiff_path / "utils_train.py")
_utils_train_module = importlib.util.module_from_spec(_utils_train_spec)
sys.modules["tabdiff_utils_train"] = _utils_train_module
_utils_train_spec.loader.exec_module(_utils_train_module)
TabDiffDataset = _utils_train_module.TabDiffDataset
update_ema = _utils_train_module.update_ema

from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion

from .base import BaseGenerator


class TabDiffGenerator(BaseGenerator):
    """
    TabDiff (Tabular Diffusion) 生成器包装类
    """
    
    def __init__(
        self,
        model_dir: str,
        data_dir: str,
        device: str = "cuda"
    ):
        """
        Args:
            model_dir: 模型保存目录（包含 model.pt, config.pkl 等）
            data_dir: 训练数据目录（包含 .npy 文件和 info.json）
            device: 设备
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 加载数据集信息
        self.info = src.load_json(self.data_dir / 'info.json')
        
        # 加载配置
        config_path = self.model_dir / 'config.pkl'
        if config_path.exists():
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
        else:
            raise FileNotFoundError(f"No config file found in {self.model_dir}")
        
        # 加载数据集以获取逆转换器
        self.dataset = TabDiffDataset(
            dataname=self.info.get('dataname', 'custom'),
            data_dir=str(self.data_dir),
            info=self.info,
            isTrain=True,
            y_only=False,
            dequant_dist=self.config.get('data', {}).get('dequant_dist', 'none'),
            int_dequant_factor=self.config.get('data', {}).get('int_dequant_factor', 0.0)
        )
        
        d_numerical = self.dataset.d_numerical
        categories = self.dataset.categories
        
        # 构建模型
        unimodmlp_params = self.config['unimodmlp_params'].copy()
        unimodmlp_params['d_numerical'] = d_numerical
        unimodmlp_params['categories'] = (categories + 1).tolist()
        
        backbone = UniModMLP(**unimodmlp_params)
        model = Model(backbone, **self.config['diffusion_params']['edm_params'])
        model.to(self.device)
        
        # 创建扩散模型
        diffusion_params = self.config['diffusion_params'].copy()
        diffusion_params.pop('edm_params', None)
        
        self.diffusion = UnifiedCtimeDiffusion(
            num_classes=categories,
            num_numerical_features=d_numerical,
            denoise_fn=model,
            y_only_model=None,
            **diffusion_params,
            device=self.device
        )
        self.diffusion.to(self.device)
        
        # 加载权重
        ckpt_path = self.model_dir / 'model.pt'
        if ckpt_path.exists():
            state_dicts = torch.load(ckpt_path, map_location=self.device)
            self.diffusion._denoise_fn.load_state_dict(state_dicts['denoise_fn'])
            self.diffusion.num_schedule.load_state_dict(state_dicts['num_schedule'])
            self.diffusion.cat_schedule.load_state_dict(state_dicts['cat_schedule'])
        
        self.diffusion.eval()
        
        # 保存列名信息
        self._columns = self.info.get('column_names', [])
        self._target_col = self._get_target_col_name()
        
        # 保存编码后的训练数据供 inpainting 使用
        self._encoded_train_data = self.dataset.X.cpu().numpy()
    
    def _get_target_col_name(self) -> str:
        """获取目标列名"""
        target_col_idx = self.info.get('target_col_idx', [])
        column_names = self.info.get('column_names', [])
        if target_col_idx and column_names:
            return column_names[target_col_idx[0]]
        return self.info.get('target_col', 'target')
    
    def sample(
        self,
        n_samples: int,
        temperature: float = 1.0,
        device: str = "cuda"
    ) -> pd.DataFrame:
        """
        生成合成样本
        
        Args:
            n_samples: 生成样本数量
            temperature: 温度参数（TabDiff 不使用，保留接口兼容性）
            device: 设备
        """
        batch_size = min(n_samples, 10000)
        
        # 生成样本
        syn_data = self.diffusion.sample_all(n_samples, batch_size, keep_nan_samples=True)
        print(f"[TabDiff] 生成样本 shape: {syn_data.shape}")
        
        # 逆转换
        df = self._inverse_transform(syn_data)
        print(f"[TabDiff] 逆转换后 DataFrame shape: {df.shape}")
        
        return df
    
    def _inverse_transform(self, syn_data: torch.Tensor) -> pd.DataFrame:
        """将生成的数据逆转换为原始 DataFrame 格式（使用 TabDiff 原始逻辑）"""
        from tabdiff.trainer import split_num_cat_target, recover_data
        
        # 将 Tensor 转换为 numpy（必须在传入 inverse 函数前转换）
        if isinstance(syn_data, torch.Tensor):
            syn_data = syn_data.cpu().numpy()
        
        num_inverse = self.dataset.num_inverse
        int_inverse = self.dataset.int_inverse
        cat_inverse = self.dataset.cat_inverse
        
        syn_num, syn_cat, syn_target = split_num_cat_target(
            syn_data, self.info, num_inverse, int_inverse, cat_inverse
        )
        syn_df = recover_data(syn_num, syn_cat, syn_target, self.info)
        
        # 重命名列
        idx_name_mapping = self.info.get('idx_name_mapping', {})
        idx_name_mapping = {int(k): v for k, v in idx_name_mapping.items()}
        if idx_name_mapping:
            syn_df.rename(columns=idx_name_mapping, inplace=True)
        
        # 将标签从整数编码转换回原始值
        target_col = self.info.get('target_col')
        label_classes = self.info.get('label_classes')
        if target_col and label_classes and target_col in syn_df.columns:
            # 将整数标签映射回原始字符串
            syn_df[target_col] = syn_df[target_col].astype(int).clip(0, len(label_classes)-1)
            syn_df[target_col] = syn_df[target_col].map(lambda x: label_classes[int(x)])
        
        return syn_df
    
    def sample_inpaint(
        self,
        anchor_indices: Union[List[int], np.ndarray],
        num_mask: List[bool],
        cat_mask: List[bool],
        n_samples_per_anchor: int = 1,
        stochasticity: float = 1.0
    ) -> pd.DataFrame:
        """
        条件生成：固定部分列，生成其余列
        
        Args:
            anchor_indices: 锚点在训练数据中的索引
            num_mask: 数值列的 mask，True=生成，False=固定
            cat_mask: 类别列的 mask，True=生成，False=固定
            n_samples_per_anchor: 每个锚点生成的样本数
            stochasticity: 随机重启强度 (0~1)
        
        Returns:
            生成的 DataFrame（原始格式）
        """
        # 直接从编码后的训练数据获取 anchor
        anchor_encoded = self._encoded_train_data[anchor_indices]
        
        # 分离数值和类别特征
        d_num = self.dataset.d_numerical
        anchor_num = torch.tensor(anchor_encoded[:, :d_num], dtype=torch.float32)
        anchor_cat = torch.tensor(anchor_encoded[:, d_num:], dtype=torch.long)
        
        num_mask_t = torch.tensor(num_mask, dtype=torch.bool)
        cat_mask_t = torch.tensor(cat_mask, dtype=torch.bool)
        
        # 调用底层 inpaint
        syn_data = self.diffusion.sample_inpaint(
            anchor_num=anchor_num,
            anchor_cat=anchor_cat,
            num_mask=num_mask_t,
            cat_mask=cat_mask_t,
            n_samples_per_anchor=n_samples_per_anchor,
            stochasticity=stochasticity
        )
        
        # 逆转换
        return self._inverse_transform(syn_data)
    
    def get_column_masks(self, fix_cols: List[str]) -> tuple:
        """
        根据要固定的列名生成 num_mask 和 cat_mask
        
        注意：TabDiff 的数据格式是 [num_features, target+cat_features]
        对于分类任务，目标列放在类别特征的最前面
        
        Args:
            fix_cols: 要固定的列名列表
        
        Returns:
            (num_mask, cat_mask): bool 列表，True=生成，False=固定
        """
        info = self.info
        num_col_idx = info.get('num_col_idx', [])
        cat_col_idx = info.get('cat_col_idx', [])
        column_names = info.get('column_names', [])
        target_col = info.get('target_col', '')
        task_type = info.get('task_type', 'binclass')
        
        # 数值列
        num_cols = [column_names[i] for i in num_col_idx]
        num_mask = [col not in fix_cols for col in num_cols]
        
        # 类别列（TabDiff 格式：对于分类任务，target 在最前面）
        cat_cols = [column_names[i] for i in cat_col_idx]
        
        if task_type in ['binclass', 'multiclass']:
            # target 在类别特征最前面
            cat_mask = [target_col not in fix_cols]  # 目标列的 mask
            cat_mask.extend([col not in fix_cols for col in cat_cols])
        else:
            # regression: target 在数值特征最前面（由 y 拼接），需补齐 mask
            if target_col:
                num_mask = [target_col not in fix_cols] + num_mask
            cat_mask = [col not in fix_cols for col in cat_cols]
        
        return num_mask, cat_mask
    
    @property
    def columns(self) -> List[str]:
        """返回数据集的所有列名"""
        return self._columns
    
    @property
    def conditional_col(self) -> Optional[str]:
        """返回条件列名"""
        return self._target_col


def train_tabdiff(
    train_data: Union[str, pd.DataFrame],
    target_col: str,
    save_path: str = 'runs/tabdiff_model',
    steps: int = 8000,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 4096,
    num_timesteps: int = 50,
    device: str = "cuda",
    seed: int = 0,
    task_type: Optional[str] = None,
) -> TabDiffGenerator:
    """
    训练 TabDiff 模型
    
    Args:
        train_data: 训练数据（CSV 路径或 DataFrame）
        target_col: 目标列名
        save_path: 模型保存路径
        steps: 训练步数（默认 8000）
        lr: 学习率（默认 0.001）
        weight_decay: 权重衰减（默认 0.0）
        batch_size: 批次大小（默认 4096）
        num_timesteps: 扩散步数（默认 50）
        device: 设备
        seed: 随机种子
        task_type: "classification" or "regression"; if omitted, infer from target values
        
    Returns:
        训练好的 TabDiffGenerator 实例
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 加载数据
    if isinstance(train_data, str):
        df = pd.read_csv(train_data)
    else:
        df = train_data.copy()
    
    print(f"Training TabDiff on {len(df)} rows")
    
    # 创建保存目录
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    data_dir = save_path / 'data'
    data_dir.mkdir(exist_ok=True)
    
    info = _prepare_tabdiff_data(df, target_col, data_dir, task_type=task_type)
    
    # 设置设备
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 加载配置
    config = _get_default_config()
    config['train']['main']['steps'] = steps
    config['train']['main']['lr'] = lr
    config['train']['main']['weight_decay'] = weight_decay
    config['train']['main']['batch_size'] = batch_size
    config['diffusion_params']['num_timesteps'] = num_timesteps
    
    # 加载数据集
    dataset = TabDiffDataset(
        dataname='custom',
        data_dir=str(data_dir),
        info=info,
        isTrain=True,
        y_only=False,
        dequant_dist=config['data']['dequant_dist'],
        int_dequant_factor=config['data']['int_dequant_factor']
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    d_numerical = dataset.d_numerical
    categories = dataset.categories
    
    print(f"数值特征: {d_numerical}, 类别特征: {len(categories)}")
    
    # 构建模型
    config['unimodmlp_params']['d_numerical'] = d_numerical
    config['unimodmlp_params']['categories'] = (categories + 1).tolist()
    
    backbone = UniModMLP(**config['unimodmlp_params'])
    model = Model(backbone, **config['diffusion_params']['edm_params'])
    model.to(device_obj)
    
    # 创建扩散模型
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        num_timesteps=config['diffusion_params']['num_timesteps'],
        scheduler=config['diffusion_params']['scheduler'],
        cat_scheduler=config['diffusion_params']['cat_scheduler'],
        noise_dist=config['diffusion_params']['noise_dist'],
        edm_params=config['diffusion_params']['edm_params'],
        noise_dist_params=config['diffusion_params']['noise_dist_params'],
        noise_schedule_params=config['diffusion_params']['noise_schedule_params'],
        sampler_params=config['diffusion_params']['sampler_params'],
        device=device_obj
    )
    diffusion.to(device_obj)
    diffusion.train()
    
    # EMA 模型
    ema_model = deepcopy(diffusion._denoise_fn)
    for param in ema_model.parameters():
        param.detach_()
    ema_num_schedule = deepcopy(diffusion.num_schedule)
    for param in ema_num_schedule.parameters():
        param.detach_()
    ema_cat_schedule = deepcopy(diffusion.cat_schedule)
    for param in ema_cat_schedule.parameters():
        param.detach_()
    
    # 优化器
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50)
    
    # 训练循环
    print(f"开始训练: {steps} 步...")
    best_loss = float('inf')
    ema_decay = config['train']['main'].get('ema_decay', 0.997)
    
    for epoch in range(steps):
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{steps}")
        
        curr_dloss = 0.0
        curr_closs = 0.0
        curr_count = 0
        
        for batch in pbar:
            x = batch.float().to(device_obj)
            
            diffusion.train()
            optimizer.zero_grad()
            
            dloss, closs = diffusion.mixed_loss(x)
            loss = dloss + closs
            loss.backward()
            optimizer.step()
            
            curr_dloss += dloss.item() * len(x)
            curr_closs += closs.item() * len(x)
            curr_count += len(x)
            
            pbar.set_postfix({
                "DLoss": f"{curr_dloss/curr_count:.4f}",
                "CLoss": f"{curr_closs/curr_count:.4f}",
            })
        
        total_loss = (curr_dloss + curr_closs) / curr_count
        scheduler.step(total_loss)
        
        # 更新 EMA
        update_ema(ema_model.parameters(), diffusion._denoise_fn.parameters(), rate=ema_decay)
        update_ema(ema_num_schedule.parameters(), diffusion.num_schedule.parameters(), rate=ema_decay)
        update_ema(ema_cat_schedule.parameters(), diffusion.cat_schedule.parameters(), rate=ema_decay)
        
        # 保存最佳模型
        if total_loss < best_loss:
            best_loss = total_loss
            state_dicts = {
                'denoise_fn': ema_model.state_dict(),
                'num_schedule': ema_num_schedule.state_dict(),
                'cat_schedule': ema_cat_schedule.state_dict(),
            }
            torch.save(state_dicts, save_path / 'model.pt')
    
    # 保存配置
    with open(save_path / 'config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    # 复制 info.json
    shutil.copy(data_dir / 'info.json', save_path / 'info.json')
    
    print(f"✓ 模型已保存至: {save_path}\n")
    
    return TabDiffGenerator(str(save_path), str(data_dir), device)


def _get_default_config() -> dict:
    """获取默认配置"""
    return {
        'data': {
            'dequant_dist': 'none',
            'int_dequant_factor': 0,
        },
        'unimodmlp_params': {
            'num_layers': 2,
            'd_token': 4,
            'n_head': 1,
            'factor': 32,
            'bias': True,
            'dim_t': 1024,
            'use_mlp': True,
        },
        'diffusion_params': {
            'num_timesteps': 50,
            'scheduler': 'power_mean',
            'cat_scheduler': 'log_linear',
            'noise_dist': 'uniform_t',
            'sampler_params': {
                'stochastic_sampler': True,
                'second_order_correction': True,
            },
            'edm_params': {
                'precond': True,
                'sigma_data': 1.0,
                'net_conditioning': 'sigma',
            },
            'noise_dist_params': {
                'P_mean': -1.2,
                'P_std': 1.2,
            },
            'noise_schedule_params': {
                'sigma_min': 0.002,
                'sigma_max': 80,
                'rho': 7,
                'eps_max': 1e-3,
                'eps_min': 1e-5,
                'rho_init': 7.0,
                'rho_offset': 5.0,
                'k_init': -6.0,
                'k_offset': 1.0,
            },
        },
        'train': {
            'main': {
                'steps': 8000,
                'lr': 0.001,
                'weight_decay': 0,
                'ema_decay': 0.997,
                'batch_size': 4096,
            },
        },
    }


def _prepare_tabdiff_data(
    df: pd.DataFrame,
    target_col: str,
    save_dir: Path,
    task_type: Optional[str] = None,
) -> dict:
    """将 DataFrame 转换为 TabDiff 所需的 .npy 格式"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 分离特征和目标
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    
    column_names = df.columns.tolist()
    
    # 区分数值列和类别列
    # 参考 TabDiff 原始逻辑：根据 dtype 判断，不依赖唯一值数量
    num_cols = []
    cat_cols = []
    for col in X.columns:
        # 检查是否为类别类型或字符串类型
        if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
            cat_cols.append(col)
        else:
            # 尝试转换为数值，如果成功则为数值列
            try:
                pd.to_numeric(X[col], errors='raise')
                num_cols.append(col)
            except (ValueError, TypeError):
                cat_cols.append(col)
    
    # 获取列索引
    num_col_idx = [column_names.index(c) for c in num_cols]
    cat_col_idx = [column_names.index(c) for c in cat_cols]
    target_col_idx = [column_names.index(target_col)]
    
    # 处理 NaN 值
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
    if len(cat_cols) > 0:
        for col in cat_cols:
            mode_val = X[col].mode()
            if len(mode_val) > 0:
                X[col] = X[col].fillna(mode_val[0])
    
    # 确定任务类型
    if task_type == "regression":
        tabdiff_task_type = "regression"
    elif task_type == "classification":
        tabdiff_task_type = "binclass" if len(np.unique(y)) == 2 else "multiclass"
    elif y.dtype == 'object' or len(np.unique(y)) < 20:
        tabdiff_task_type = 'binclass' if len(np.unique(y)) == 2 else 'multiclass'
    else:
        tabdiff_task_type = 'regression'

    if tabdiff_task_type in {"binclass", "multiclass"}:
        n_classes = len(np.unique(y))
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        label_classes = le.classes_.tolist()
    else:
        n_classes = None
        y_encoded = y.astype(np.float32)
        label_classes = None
    
    # 检测整数列
    int_col_idx = []
    int_col_idx_wrt_num = []
    for i, col in enumerate(num_cols):
        col_data = X[col].dropna()
        if len(col_data) > 0 and (col_data % 1 == 0).all():
            int_col_idx.append(column_names.index(col))
            int_col_idx_wrt_num.append(i)
    
    # 构建 idx_mapping
    idx_mapping = {}
    inverse_idx_mapping = {}
    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)
    
    for idx in range(len(column_names)):
        if idx in num_col_idx:
            idx_mapping[idx] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[idx] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[idx] = curr_target_idx
            curr_target_idx += 1
    
    for k, v in idx_mapping.items():
        inverse_idx_mapping[v] = k
    
    idx_name_mapping = {i: column_names[i] for i in range(len(column_names))}
    
    # 保存数值特征
    if len(num_cols) > 0:
        X_num = X[num_cols].values.astype(np.float32)
        np.save(save_dir / 'X_num_train.npy', X_num)
        np.save(save_dir / 'X_num_test.npy', X_num[:min(10, len(X_num))])
    else:
        # 创建空的数值数组
        X_num_empty = np.array([]).reshape(len(X), 0).astype(np.float32)
        np.save(save_dir / 'X_num_train.npy', X_num_empty)
        np.save(save_dir / 'X_num_test.npy', X_num_empty[:min(10, len(X))])
    
    # 保存类别特征（保持原始字符串格式，让 TabDiff 自己编码）
    if len(cat_cols) > 0:
        X_cat = X[cat_cols].astype(str).values
        np.save(save_dir / 'X_cat_train.npy', X_cat)
        np.save(save_dir / 'X_cat_test.npy', X_cat[:min(10, len(X_cat))])
    else:
        # 创建空的类别数组
        X_cat_empty = np.array([]).reshape(len(X), 0)
        np.save(save_dir / 'X_cat_train.npy', X_cat_empty)
        np.save(save_dir / 'X_cat_test.npy', X_cat_empty[:min(10, len(X))])
    
    # 保存目标
    np.save(save_dir / 'y_train.npy', y_encoded)
    np.save(save_dir / 'y_test.npy', y_encoded[:min(10, len(y_encoded))])
    
    # 创建 info.json
    info = {
        'task_type': tabdiff_task_type,
        'n_classes': n_classes,
        'column_names': column_names,
        'target_col': target_col,
        'num_col_idx': num_col_idx,
        'cat_col_idx': cat_col_idx,
        'target_col_idx': target_col_idx,
        'int_col_idx': int_col_idx,
        'int_col_idx_wrt_num': int_col_idx_wrt_num,
        'idx_mapping': idx_mapping,
        'inverse_idx_mapping': inverse_idx_mapping,
        'idx_name_mapping': idx_name_mapping,
        'label_classes': label_classes,
        'train_num': len(df),
        'test_num': min(10, len(df)),
        'dataname': 'custom',
    }
    
    with open(save_dir / 'info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    return info
