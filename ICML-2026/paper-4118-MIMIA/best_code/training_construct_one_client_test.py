import sys
import os
import numpy as np
import torch
from collections import ChainMap
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import copy
import scipy
import time
import json
import math
import random
import pandas as pd # <--- 新增

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np


def select_client_specific_parts(global_data, K, client_ids_to_select):
    """
    将全局数据按 K 个客户端均匀切分，并提取指定客户端的部分。
    """
    if len(global_data) == 0: return global_data

    # 1. 均匀切分 (处理无法整除的情况)
    chunks = np.array_split(global_data, K)

    # 2. 提取指定部分
    selected_chunks = []
    for cid in client_ids_to_select:
        if 0 <= cid < len(chunks):
            selected_chunks.append(chunks[cid])

    # 3. 拼接
    if len(selected_chunks) > 0:
        if isinstance(global_data, torch.Tensor):
            return torch.cat(selected_chunks, dim=0)
        else:
            return np.concatenate(selected_chunks, axis=0)
    else:
        return global_data[:0]  # 返回空对象
def reorder_all_clients_data(training_res, target_client_id, val_client_ids):
    """
    重写所有客户端的 train/mix 数据

    核心逻辑：
    1. 从 target_client_id 获取其真实的训练索引（client_train_indices）
    2. 在每个客户端的 mix_indices 中查找这些索引的位置
    3. 从 mix 的对应位置提取数据作为新的 train
    4. 对 val_client_ids 做类似操作（如果包含0则从train提取，否则从mix提取）

    参数:
    - training_res: 已加载的所有客户端数据列表
    - target_client_id: 目标客户端ID（其训练数据将作为新的train）
    - val_client_ids: 验证客户端ID列表（其训练数据将作为新的mix）

    返回:
    - reordered_training_res: 重写后的所有客户端数据
    """

    print(f"\n{'=' * 70}")
    print(f"🔄 开始重排所有客户端数据")
    print(f"{'=' * 70}")
    print(f"   目标客户端: {target_client_id} → 提取其训练数据作为新的 train (成员)")
    print(f"   验证客户端: {val_client_ids} → 提取其训练数据作为新的 mix (非成员)")
    print(f"   将重写 {len(training_res)} 个客户端文件\n")

    # ========== 1. 获取目标客户端的真实训练索引 ==========
    target_real_indices = np.array(training_res[target_client_id]['client_train_indices'])
    print(f"  📌 目标客户端 {target_client_id} 的真实训练索引:")
    print(f"     索引范围: [{target_real_indices.min()}, {target_real_indices.max()}]")
    print(f"     样本数量: {len(target_real_indices)}\n")

    # ========== 2. 获取验证客户端的真实训练索引 ==========
    val_real_indices_dict = {}
    for val_id in val_client_ids:
        val_real_indices_dict[val_id] = np.array(training_res[val_id]['client_train_indices'])
        print(f"  📌 验证客户端 {val_id} 的真实训练索引:")
        print(f"     索引范围: [{val_real_indices_dict[val_id].min()}, {val_real_indices_dict[val_id].max()}]")
        print(f"     样本数量: {len(val_real_indices_dict[val_id])}\n")

    # ========== 3. 逐个重写每个客户端的数据 ==========
    reordered_training_res = []
    is_default_view = (target_client_id == 0)
    for client_id, client_data in enumerate(training_res):
        print(f"  🔧 正在重写客户端 {client_id} 的数据...")

        # 深拷贝避免修改原数据
        new_client_data = copy.deepcopy(client_data)

        # 获取当前客户端的原始索引
        original_train_indices = np.array(client_data['index']['train_indices'])
        original_mix_indices = np.array(client_data['index']['mix_indices'])

        # --- 3.1 提取新的 train 数据（目标客户端的训练数据）---
        # 在 mix_indices 中找到 target_real_indices 的位置
        train_positions_in_mix = []  # 初始化
        if not is_default_view:
            # 【重排模式】在 Mix 中查找目标数据
            new_train_indices, train_positions_in_mix = find_indices_positions(
                target_real_indices, original_mix_indices, "mix"
            )
            # 覆盖索引
            new_client_data['index']['train_indices'] = new_train_indices
        else:
            # 【默认模式】保留原索引
            new_train_indices = original_train_indices
        if len(new_train_indices) == 0:
            print(f"     ⚠️  警告: 在 mix 中未找到目标客户端 {target_client_id} 的数据")
        else:
            print(f"     ✅ 从 mix 中提取 {len(new_train_indices)} 个样本作为新 train")

        # --- 3.2 提取新的 mix 数据（验证客户端的训练数据）---
        new_mix_indices_list = []
        mix_positions_dict = {}  # 记录每个验证客户端在原数据中的位置

        for val_id in val_client_ids:
            val_indices = val_real_indices_dict[val_id]

            if val_id == 0:
                # 特殊情况：客户端0的数据在原始 train 中
                found_indices, positions = find_indices_positions(
                    val_indices, original_train_indices, "train"
                )
                mix_positions_dict[val_id] = ('train', positions)
            else:
                # 其他客户端：从原始 mix 中提取
                found_indices, positions = find_indices_positions(
                    val_indices, original_mix_indices, "mix"
                )
                mix_positions_dict[val_id] = ('mix', positions)

            new_mix_indices_list.append(found_indices)

            if len(found_indices) > 0:
                source = "train" if val_id == 0 else "mix"
                print(f"     ✅ 从 {source} 中提取 {len(found_indices)} 个样本 (客户端{val_id})")

        # 合并所有验证客户端的索引
        new_mix_indices = np.concatenate(new_mix_indices_list, axis=0) if new_mix_indices_list else np.array([])

        # --- 3.3 更新 index 字典 ---
        new_client_data['index']['train_indices'] = new_train_indices
        new_client_data['index']['mix_indices'] = new_mix_indices
        # test_indices 保持不变

        # --- 3.4 重写各模态的分数数据 ---
        for modality in ['full', 'audio', 'visual']:
            if modality not in new_client_data:
                continue

            # 重写 train 相关的分数（从原 mix 中提取）
            if not is_default_view:
                new_client_data = rewrite_scores_from_positions(
                    new_client_data, client_data, modality,
                    train_positions_in_mix, 'train', source='mix'
                )

            # 重写 mix 相关的分数（从原 train 或 mix 中提取并合并）
            new_client_data = rewrite_mix_scores_from_multiple_sources(
                new_client_data, client_data, modality, mix_positions_dict
            )

        # --- 3. 5 重写 res 数据（logit, labels, loss等）---
        new_client_data = rewrite_res_data_from_positions(
            new_client_data, client_data,
            train_positions_in_mix, mix_positions_dict,
            is_default_view=is_default_view  # <--- 显式传入
        )

        reordered_training_res.append(new_client_data)
        print(f"     ✅ 客户端 {client_id} 重写完成\n")

    print(f"{'=' * 70}")
    print(f"🎉 所有客户端数据重写完成！")
    print(f"{'=' * 70}\n")

    return reordered_training_res


def find_indices_positions(target_indices, source_indices, source_name):
    """
    在 source_indices 中查找 target_indices 的位置

    参数:
    - target_indices: 要查找的索引数组
    - source_indices: 源索引数组
    - source_name: 源名称（用于日志）

    返回:
    - found_indices: 找到的索引
    - positions: 在source_indices中的位置
    """
    # 使用 np.isin 找到哪些目标索引存在于源中
    mask = np.isin(target_indices, source_indices)
    found_indices = target_indices[mask]

    if len(found_indices) == 0:
        return np.array([]), np.array([])

    # 使用 np.searchsorted 找到位置（要求source_indices已排序）
    # 如果未排序，需要使用其他方法
    positions = []
    for idx in found_indices:
        pos = np.where(source_indices == idx)[0]
        if len(pos) > 0:
            positions.append(pos[0])

    positions = np.array(positions)

    return found_indices, positions


def rewrite_scores_from_positions(new_data, original_data, modality, positions, target_key_prefix, source='mix'):
    """
    从指定位置提取分数并重写

    参数:
    - new_data: 要更新的数据
    - original_data: 原始数据
    - modality: 模态名称
    - positions: 要提取的位置数组
    - target_key_prefix: 目标键前缀（'train' 或 'mix'）
    - source: 数据来源（'train' 或 'mix'）

    返回:
    - 更新后的 new_data
    """

    if len(positions) == 0:
        return new_data

    # 确定源键前缀
    if source == 'train':
        source_key_prefix = 'tarin'  # 注意拼写
    elif source == 'mix':
        source_key_prefix = 'mix'
    else:
        return new_data

    # 确定目标键前缀
    if target_key_prefix == 'train':
        target_key_prefix = 'tarin'  # 注意拼写

    # 需要重写的键名后缀
    key_suffixes = ['_cos', '_diffs', '_grad_norm']

    for suffix in key_suffixes:
        source_key = source_key_prefix + suffix
        target_key = target_key_prefix + suffix

        if source_key not in original_data[modality]:
            continue

        original_scores = original_data[modality][source_key]

        # 处理嵌套字典结构（多模态）
        if isinstance(original_scores, dict):
            if target_key not in new_data[modality]:
                new_data[modality][target_key] = {}

            for sub_modality, scores in original_scores.items():
                # 转换为 tensor
                if isinstance(scores, np.ndarray):
                    scores = torch.from_numpy(scores)
                elif not isinstance(scores, torch.Tensor):
                    scores = torch.tensor(scores)

                # 提取指定位置的分数
                extracted_scores = scores[positions]
                new_data[modality][target_key][sub_modality] = extracted_scores
        else:
            # 直接是 tensor/array
            if isinstance(original_scores, np.ndarray):
                original_scores = torch.from_numpy(original_scores)
            elif not isinstance(original_scores, torch.Tensor):
                original_scores = torch.tensor(original_scores)

            extracted_scores = original_scores[positions]
            new_data[modality][target_key] = extracted_scores

    return new_data


def rewrite_mix_scores_from_multiple_sources(new_data, original_data, modality, positions_dict):
    """
    从多个源（train或mix）合并数据作为新的mix

    参数:
    - new_data: 要更新的数据
    - original_data: 原始数据
    - modality: 模态名称
    - positions_dict: {client_id: (source, positions)} 字典

    返回:
    - 更新后的 new_data
    """

    key_suffixes = ['_cos', '_diffs', '_grad_norm']

    for suffix in key_suffixes:
        target_key = 'mix' + suffix

        merged_scores_dict = {}  # {sub_modality: [scores_list]}

        for client_id, (source, positions) in positions_dict.items():
            if len(positions) == 0:
                continue

            # 确定源键
            if source == 'train':
                source_key = 'tarin' + suffix  # 注意拼写
            else:
                source_key = 'mix' + suffix

            if source_key not in original_data[modality]:
                continue

            original_scores = original_data[modality][source_key]

            # 处理嵌套字典
            if isinstance(original_scores, dict):
                for sub_modality, scores in original_scores.items():
                    if sub_modality not in merged_scores_dict:
                        merged_scores_dict[sub_modality] = []

                    # 转换并提取
                    if isinstance(scores, np.ndarray):
                        scores = torch.from_numpy(scores)
                    elif not isinstance(scores, torch.Tensor):
                        scores = torch.tensor(scores)

                    extracted = scores[positions]
                    merged_scores_dict[sub_modality].append(extracted)
            else:
                # 非字典结构
                if 'default' not in merged_scores_dict:
                    merged_scores_dict['default'] = []

                if isinstance(original_scores, np.ndarray):
                    original_scores = torch.from_numpy(original_scores)
                elif not isinstance(original_scores, torch.Tensor):
                    original_scores = torch.tensor(original_scores)

                extracted = original_scores[positions]
                merged_scores_dict['default'].append(extracted)

        # 合并所有分数
        if len(merged_scores_dict) > 0:
            if 'default' in merged_scores_dict:
                # 非字典结构
                new_data[modality][target_key] = torch.cat(merged_scores_dict['default'], dim=0)
            else:
                # 字典结构
                new_data[modality][target_key] = {}
                for sub_modality, scores_list in merged_scores_dict.items():
                    if len(scores_list) > 0:
                        new_data[modality][target_key][sub_modality] = torch.cat(scores_list, dim=0)

    return new_data


def rewrite_res_data_from_positions(new_data, original_data, train_positions, mix_positions_dict,
                                    is_default_view=False):
    """
    重写 train_res 和 mix_res 数据
    参数:
    - is_default_view:
        True  -> 目标为0，跳过 train_res 的覆盖（保留原值）。
        False -> 目标非0，需要从 mix_res 提取数据覆盖 train_res。
    """

    # --- 1. 重写 train_res ---
    # 逻辑：只有当【不是】默认视图时，才去执行“移花接木”的操作
    if not is_default_view:
        if 'mix_res' in original_data and 'train_res' in new_data and len(train_positions) > 0:
            for key in original_data['mix_res'].keys():
                data = original_data['mix_res'][key]

                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                elif not isinstance(data, torch.Tensor):
                    data = torch.tensor(data)

                # 提取 Mix 中的数据 覆盖到 Train 中
                new_data['train_res'][key] = data[train_positions]

    # --- 2. 重写 mix_res (保持不变) ---
    if 'mix_res' in new_data:
        merged_res = {}
        for client_id, (source, positions) in mix_positions_dict.items():
            if len(positions) == 0: continue

            if source == 'train':
                source_res_key = 'train_res'
            else:
                source_res_key = 'mix_res'

            if source_res_key not in original_data: continue

            for key in original_data[source_res_key].keys():
                data = original_data[source_res_key][key]
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                elif not isinstance(data, torch.Tensor):
                    data = torch.tensor(data)

                if key not in merged_res: merged_res[key] = []
                merged_res[key].append(data[positions])

        for key, data_list in merged_res.items():
            if len(data_list) > 0:
                if data_list[0].dim() == 0:
                    new_data['mix_res'][key] = torch.stack(data_list)
                else:
                    new_data['mix_res'][key] = torch.cat(data_list, dim=0)

    return new_data



def validate_reorder_config(training_res, target_client_id, val_client_ids):
    """
    验证重排配置的有效性
    """
    MAX_K = len(training_res)

    if target_client_id < 0 or target_client_id >= MAX_K:
        return False, f"目标客户端ID {target_client_id} 超出范围 [0, {MAX_K - 1}]"

    if 'client_train_indices' not in training_res[target_client_id]:
        return False, f"目标客户端 {target_client_id} 缺少 'client_train_indices'"

    for val_id in val_client_ids:
        if val_id < 0 or val_id >= MAX_K:
            return False, f"验证客户端ID {val_id} 超出范围 [0, {MAX_K - 1}]"

        if 'client_train_indices' not in training_res[val_id]:
            return False, f"验证客户端 {val_id} 缺少 'client_train_indices'"

    if target_client_id in val_client_ids:
        return False, f"目标客户端ID {target_client_id} 不能同时出现在验证客户端列表中"

    return True, ""


def create_wide_format_dataframe(attack_type, epch, scores_per_modality, indices_per_modality=None):
    """
    将攻击分数转换为宽格式的 Pandas DataFrame。
    【修复】强制将索引转换为一维整数，防止合并时出现 NaN。
    """
    if not scores_per_modality:
        return pd.DataFrame()

    all_data = []

    # 获取第一个模态作为参照 (例如 'audio')
    first_modality = list(scores_per_modality.keys())[0]

    # --- 1. 准备 Index 数据 ---
    # 先默认使用 0, 1, 2...
    train_idx_data = np.arange(len(scores_per_modality[first_modality][0]))
    test_idx_data = np.arange(len(scores_per_modality[first_modality][1]))

    # 如果传入了真实 Index，则覆盖默认值
    if indices_per_modality is not None and first_modality in indices_per_modality:
        raw_train = indices_per_modality[first_modality][0]
        raw_test = indices_per_modality[first_modality][1]

        # ⭐【关键修复】强制展平并转为整数
        # 这样 1.0 和 1 都会变成 1，且 [1] 也会变成 1
        train_idx_data = np.array(raw_train).flatten().astype(int)
        test_idx_data = np.array(raw_test).flatten().astype(int)

    # --- 2. 成员 (Member / Train) ---
    # 确保分数也是一维的
    member_scores_ref = np.array(scores_per_modality[first_modality][0]).flatten()
    member_scores_len = len(member_scores_ref)

    # 安全检查：截断以防长度不一致
    min_len_train = min(len(train_idx_data), member_scores_len)

    member_data = {
        'Sample_Type': ['Member'] * min_len_train,
        'Sample_Idx': train_idx_data[:min_len_train],  # 使用清洗过的整数索引
        'Epoch': [epch] * min_len_train,
    }

    for modality, (member_scores, _) in scores_per_modality.items():
        flat_scores = np.array(member_scores).flatten()[:min_len_train]
        member_data[f'{attack_type}_{modality}_Score'] = flat_scores

    all_data.append(pd.DataFrame(member_data))

    # --- 3. 非成员 (Non-Member / Test/Val) ---
    non_member_scores_ref = np.array(scores_per_modality[first_modality][1]).flatten()
    non_member_scores_len = len(non_member_scores_ref)

    min_len_test = min(len(test_idx_data), non_member_scores_len)

    non_member_data = {
        'Sample_Type': ['Non_Member'] * min_len_test,
        'Sample_Idx': test_idx_data[:min_len_test],  # 使用清洗过的整数索引
        'Epoch': [epch] * min_len_test,
    }

    for modality, (_, non_member_scores) in scores_per_modality.items():
        flat_scores = np.array(non_member_scores).flatten()[:min_len_test]
        non_member_data[f'{attack_type}_{modality}_Score'] = flat_scores

    all_data.append(pd.DataFrame(non_member_data))

    return pd.concat(all_data, ignore_index=True)


def save_all_attacks_to_single_excel(save_dir, epch, all_attack_dataframes,MODE):
    """
    将所有攻击结果横向合并到一个 DataFrame，并保存到 Excel 文件的单个 Sheet 中。
    """

    list_of_dfs = list(all_attack_dataframes.values())
    if not list_of_dfs:
        print("警告: 没有收集到任何攻击 DataFrame，跳过保存。")
        return

    # 定义用于横向合并的公共键
    merge_keys = ['Sample_Type', 'Sample_Idx', 'Epoch']

    # 1. 初始化合并结果为第一个 DataFrame
    final_combined_df = list_of_dfs[0]

    # 2. 循环合并其余的 DataFrame
    print(f"🔄 正在基于键 {merge_keys} 横向合并所有 {len(list_of_dfs)} 个攻击类型的数据...")

    for i, df in enumerate(list_of_dfs[1:]):
        # 使用内连接（'inner'）确保只有在所有攻击中都存在的样本才被保留
        # 如果样本集完全相同，'outer' 或 'inner' 都可以
        final_combined_df = pd.merge(
            final_combined_df,
            df,
            on=merge_keys,
            how='outer'  # 使用 'outer' 以保留所有样本，缺失分数用 NaN 填充
        )
        print(f"   -> 完成合并第 {i + 2} 个 DataFrame (当前总列数: {len(final_combined_df.columns)})")

    # --------------------------------------
    global TARGET_CLIENT_ID, VAL_CLIENT_IDS,SHAWDOW_LIST
    # 构建保存路径
    save_path = f"{save_dir}/train_excel/training_construct_{TARGET_CLIENT_ID}_{VAL_CLIENT_IDS}_{epch}_{MODE}_lira_opti_one_attack_client_test.xlsx"

    print(f"📚 合并后的总样本数: {len(final_combined_df)}")

    try:
        # ==========================================
        # [新增] 检查并自动创建目录
        # ==========================================
        # 获取 save_path 的父目录 (即 .../train_excel/)
        directory = os.path.dirname(save_path)

        # 如果目录不存在，这就创建它 (exist_ok=True 表示如果目录已存在也不报错)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"📁 检测到目录不存在，已自动创建: {directory}")
        # ==========================================

        # 将合并后的 DataFrame 保存到 Excel 的第一个 Sheet
        final_combined_df.to_excel(save_path, sheet_name='All_Attacks_Combined', index=False)
        print(f"\n🎉 所有攻击结果已成功合并并保存到单个 Sheet: {save_path}")

    except Exception as e:
        print(f"\n❌ 保存 Excel 失败: {e}")

def liratio(mu_in,mu_out,var_in,var_out,new_samples):
    #l_in=np.sqrt(var_in)*np.exp(-((new_samples-mu_in)*(new_samples-mu_in))/(2*var_in+1e-3) )
    #l_out=np.sqrt(var_out)*np.exp(-((new_samples-mu_out)*(new_samples-mu_out))/(2*var_out+1e-3))
    l_out=scipy.stats.norm.cdf(new_samples,mu_out,np.sqrt(var_out))
    return l_out

@ torch.no_grad()
def hinge_loss_fn(x,y):
    x,y=copy.deepcopy(x).cuda(),copy.deepcopy(y).cuda()
    mask=torch.eye(x.shape[1],device="cuda")[y].bool()
    tmp1=x[mask]
    x[mask]=-1e10
    tmp2=torch.max(x,dim=1)[0]
    # print(tmp1.shape,tmp2.shape)
    return (tmp1-tmp2).cpu().numpy()

def ce_loss_fn(x,y):
    loss_fn=torch.nn.CrossEntropyLoss(reduction='none')
    return loss_fn(x,y)


def plot_auc(name,target_val_score,target_train_score,epoch):
    # print('target_val_score.shape:',target_val_score.shape)
    # indices = random.sample([i for i in range(0,target_val_score.shape[0])], target_train_score.shape[0])
    # target_val_score = torch.index_select(target_val_score, 0, torch.tensor(indices))
    # print('after sampling target_val_score.shape:',target_val_score.shape)


    fpr, tpr, thresholds = metrics.roc_curve(torch.cat( [torch.zeros_like(target_val_score),torch.ones_like(target_train_score)] ).cpu().numpy(), torch.cat([target_val_score,target_train_score]).cpu().numpy())
    auc=metrics.auc(fpr, tpr)
    log_tpr,log_fpr=np.log10(tpr),np.log10(fpr)
    log_tpr[log_tpr<-5]=-5
    log_fpr[log_fpr<-5]=-5
    log_fpr=(log_fpr+5)/5.0
    log_tpr=(log_tpr+5)/5.0
    log_auc=metrics.auc( log_fpr,log_tpr )

    tprs={}
    for fpr_thres in [10, 1, 0.1,0.02,0.01,0.001,0.0001]:
        tpr_index = np.sum(fpr<fpr_thres)
        tprs[str(fpr_thres)]=tpr[tpr_index-1]
    return auc,log_auc,tprs

def common_attack(f,K,epch,extract_fn=None):
    accs=[]
    target_res=torch.load(f.format(0,epch))

    # target_train_loss=hinge_loss_fn(target_res["train_res"]["logit"] , target_res["train_res"]["labels"] )
    # target_test_loss=hinge_loss_fn(target_res["test_res"]["logit"] , target_res["test_res"]["labels"] )

    target_train_loss=-ce_loss_fn(target_res["train_res"]["logit"] , target_res["train_res"]["labels"] )
    if MODE=="test":
        target_test_loss=-ce_loss_fn(target_res["test_res"]["logit"] , target_res["test_res"]["labels"] )
    elif MODE=="val":
        target_test_loss=-ce_loss_fn(target_res["val_res"]["logit"] , target_res["val_res"]["labels"] )

    auc,log_auc,tprs=plot_auc("common",torch.tensor(target_test_loss),torch.tensor(target_train_loss),epch)
    print("__"*10,"common")
    print(f"tprs:{tprs}", log_auc)
    # print("test_acc:",target_res[taret_idx])
    print("__"*10,)

    return accs,tprs,auc,log_auc,(target_test_loss,target_train_loss)


def lira_attack_ldh_cosine(f, epch, K, save_dir, extract_fn=None, attack_mode="cos", sample_modality='full'):
    # attack_mode="cos"

    # 标题美化
    print('\n' + '=' * 70)
    print(f'⭐ LIRA Attack (LDH-Cosine) Started ⭐')
    print(f'🎯 Target Epoch: {epch} | Attack Mode: {attack_mode.upper()} | Modality Mode: {sample_modality.upper()}')
    print('=' * 70)

    # 变量初始化... (保持不变)
    save_log = save_dir + '/' + f'attack_sel{select_mode}_{select_method}_{attack_mode}.log'
    accs = []
    training_res = []

    # 加载模型结果 (保持不变)
    print(f"🔄 Loading results from {K} models...")
    for i in range(K):
        training_res.append(torch.load(f.format(i, epch), weights_only=False))
        accs.append(training_res[-1]["test_acc"])

    target_idx = 0
    val_idx = 1
    target_res = training_res[target_idx]
    shadow_res = training_res[val_idx:]
    global TARGET_CLIENT_ID, VAL_CLIENT_IDS,SHAWDOW_LIST

    if 'TARGET_CLIENT_ID' in globals() and 'VAL_CLIENT_IDS' in globals():
        if TARGET_CLIENT_ID is not None and VAL_CLIENT_IDS is not None:
            # 验证配置
            is_valid, error_msg = validate_reorder_config(training_res, TARGET_CLIENT_ID, VAL_CLIENT_IDS)
            if not is_valid:
                raise ValueError(f"❌ 配置验证失败: {error_msg}")

            print(f"\n🎯 检测到自定义配置:")
            print(f"   目标客户端: {TARGET_CLIENT_ID}")
            print(f"   验证客户端: {VAL_CLIENT_IDS}\n")

            # 🔥 重写所有客户端数据
            training_res = reorder_all_clients_data(training_res, TARGET_CLIENT_ID, VAL_CLIENT_IDS)
            target_res = training_res[TARGET_CLIENT_ID]
            shadow_res = [training_res[i] for i in SHAWDOW_LIST]
        else:
            print(f"\nℹ️  使用默认配置\n")
    else:
        print(f"\nℹ️  使用默认配置\n")



    # 模态检测
    try:
        modalities = list(target_res[sample_modality]["tarin_cos"].keys())
    except KeyError:
        # 尝试从 loss 键中获取，以防 'cos' 不存在
        modalities = ['full']
        if target_res[sample_modality].get('tarin_diffs'):
            modalities = list(target_res[sample_modality]["tarin_diffs"].keys())
        elif attack_mode == 'loss':
            modalities = ['full', 'audio', 'visual']  # 假设 loss 模式支持

    if sample_modality == 'audio':
        modalities = ['audio']
        print("🔊 Only Audio Modality Detected.")
    elif sample_modality == 'visual':
        modalities = ['visual']
        print("🎥 Only Visual Modality Detected.")

    print(f"🔍 Found and processing modalities: {', '.join(modalities)}")

    tprs_per_modality = {}
    auc_per_modality = {}
    log_auc_per_modality = {}
    scores_per_modality = {}
    indices_per_modality = {}
    for modality in modalities:
        # 模态攻击开始提示
        print(f"\n✨ Attacking Modality: **{modality.upper()}** (Feature: {attack_mode.upper()}) ✨")
        print('-' * 40)

        # ... (数据提取逻辑保持不变) ...
        # LIRA CDF Score 计算逻辑保持不变

        if attack_mode == "cos":
            target_train_loss = target_res[sample_modality]["tarin_cos"][modality].cpu().numpy()
            if MODE == "test":
                target_test_loss = target_res[sample_modality]["test_cos"][modality].cpu().numpy()
            elif MODE == "val":
                target_test_loss = target_res[sample_modality]["val_cos"][modality].cpu().numpy()
            elif MODE == 'mix':
                target_test_loss_part = target_res[sample_modality]["test_cos"][modality].cpu().numpy()
                mix_test_loss = target_res[sample_modality]["mix_cos"][modality].cpu().numpy()
                # target_test_loss = np.concatenate([ mix_test_loss,target_test_loss_part], axis=0)
                target_test_loss_part = select_client_specific_parts(target_test_loss_part, K, VAL_CLIENT_IDS + [TARGET_CLIENT_ID])
                target_test_loss = np.concatenate([mix_test_loss, target_test_loss_part], axis=0)
        if attack_mode == "diff":
            target_train_loss = target_res[sample_modality]["tarin_diffs"][modality].cpu().numpy()
            if MODE == "test":
                target_test_loss = target_res[sample_modality]["test_diffs"][modality].cpu().numpy()
            elif MODE == "val":
                target_test_loss = target_res[sample_modality]["val_diffs"][modality].cpu().numpy()

        if attack_mode == 'loss':
            logit_key = 'logit'
            if modality == 'audio':
                logit_key = 'logit_a'
            elif modality == 'visual':
                logit_key = 'logit_v'
            target_train_loss = -ce_loss_fn(target_res["train_res"][logit_key],
                                            target_res["train_res"]["labels"]).cpu().numpy()
            if MODE == "test":
                target_test_loss = -ce_loss_fn(target_res["test_res"][logit_key],
                                               target_res["test_res"]["labels"]).cpu().numpy()
            elif MODE == "val":
                target_test_loss = -ce_loss_fn(target_res["val_res"][logit_key],
                                               target_res["val_res"]["labels"]).cpu().numpy()
            elif MODE == 'mix':

                target_test_loss_part = -ce_loss_fn(target_res["test_res"][logit_key],
                                                    target_res["test_res"]["labels"]).cpu().numpy()
                mix_test_loss = -ce_loss_fn(target_res["mix_res"][logit_key],
                                            target_res["mix_res"]["labels"]).cpu().numpy()
                # target_test_loss = np.concatenate([ mix_test_loss,target_test_loss_part], axis=0)
                target_test_loss_part = select_client_specific_parts(target_test_loss_part, K,
                                                                     VAL_CLIENT_IDS + [TARGET_CLIENT_ID])
                target_test_loss = np.concatenate([mix_test_loss, target_test_loss_part], axis=0)

        target_train_idx = target_res['index']['train_indices']

        # 2. Non-Member (Test/Mix) Indices
        if MODE == "test":
            target_test_idx = target_res['index']['test_indices']

        elif MODE == "val":
            # 如果没有 val_indices，可能需要检查 target_res 的键名
            target_test_idx = target_res['index']['val_indices']

        elif MODE == 'mix':
            # 🚨 关键：Mix 模式下的拼接
            # Loss 的拼接逻辑是: [mix_test_loss, target_test_loss_part]
            # 所以 Index 必须也是: [mix_indices, test_indices]

            idx_mix = target_res['index']['mix_indices']
            idx_test = target_res['index']['test_indices']
            # target_test_idx = np.concatenate([idx_mix, idx_test_part], axis=0)
            idx_test_part = select_client_specific_parts(idx_test, K, VAL_CLIENT_IDS + [TARGET_CLIENT_ID])
            target_test_idx = np.concatenate([idx_mix, idx_test_part], axis=0)

        else:
            # 防止 MODE 未定义时的报错，给个空值或报错
            target_test_idx = np.array([])
        shadow_train_losses = []
        shadow_test_losses = []
        for i in shadow_res:
            if attack_mode == "cos":
                shadow_train_losses.append(i[sample_modality]["tarin_cos"][modality].cpu().numpy())
                if MODE == "val":
                    shadow_test_losses.append(i[sample_modality]["val_cos"][modality].cpu().numpy())
                elif MODE == "test":
                    shadow_test_losses.append(i[sample_modality]["test_cos"][modality].cpu().numpy())
                elif MODE == 'mix':
                    shadow_test_loss_part = i[sample_modality]["test_cos"][modality].cpu().numpy()
                    mix_test_loss = i[sample_modality]["mix_cos"][modality].cpu().numpy()
                    # shadow_test_losses.append(np.concatenate([shadow_test_loss_part, mix_test_loss], axis=0))
                    shadow_test_loss_part = select_client_specific_parts(shadow_test_loss_part, K, VAL_CLIENT_IDS + [TARGET_CLIENT_ID])
                    # 注意顺序：必须是 [Mix, Test] 以匹配 Target 的顺序
                    shadow_test_losses.append(np.concatenate([mix_test_loss, shadow_test_loss_part], axis=0))
            elif attack_mode == "diff":
                shadow_train_losses.append(i[sample_modality]["tarin_diffs"][modality].cpu().numpy())
                if MODE == "val":
                    shadow_test_losses.append(i[sample_modality]["val_diffs"][modality].cpu().numpy())
                elif MODE == "test":
                    shadow_test_losses.append(i[sample_modality]["test_diffs"][modality].cpu().numpy())

            elif attack_mode == "loss":
                loss_key = 'loss'
                if modality == 'audio':
                    loss_key = 'loss_a'
                elif modality == 'visual':
                    loss_key = 'loss_v'

                shadow_train_losses.append(-i["train_res"][loss_key])
                if MODE == "val":
                    shadow_test_losses.append(-i["val_res"][loss_key])
                elif MODE == "test":
                    shadow_test_losses.append(-i["test_res"][loss_key])
                elif MODE == 'mix':
                    shadow_test_loss_part = -i["test_res"][loss_key]
                    mix_test_loss = -i["mix_res"][loss_key]
                    # shadow_test_losses.append(np.concatenate([shadow_test_loss_part, mix_test_loss], axis=0))
                    shadow_test_loss_part = select_client_specific_parts(shadow_test_loss_part, K, VAL_CLIENT_IDS + [TARGET_CLIENT_ID])
                    # 注意顺序：必须是 [Mix, Test] 以匹配 Target 的顺序
                    shadow_test_losses.append(np.concatenate([mix_test_loss, shadow_test_loss_part], axis=0))


        shadow_train_losses_stack = np.vstack(shadow_train_losses)
        shadow_test_losses_stack = np.vstack(shadow_test_losses)
        train_mu_out = shadow_train_losses_stack.mean(axis=0)
        train_var_out = shadow_train_losses_stack.var(axis=0) + 1e-8
        test_mu_out = shadow_test_losses_stack.mean(axis=0)
        test_var_out = shadow_test_losses_stack.var(axis=0) + 1e-8

        train_l_out = scipy.stats.norm.cdf(target_train_loss, train_mu_out, np.sqrt(train_var_out))
        test_l_out = scipy.stats.norm.cdf(target_test_loss, test_mu_out, np.sqrt(test_var_out))

        # 6. 计算当前模态的攻击效果（AUC和TPRs）
        auc, log_auc, tprs = plot_auc(f"lira_{modality}", torch.tensor(test_l_out), torch.tensor(train_l_out), epch)

        # 结果输出美化
        print(f"📈 Modality **{modality.upper()}** LIRA Attack Results:")
        print(f"  - AUC Score: **{auc:.4f}**")
        print(f"  - LogAUC Score: {log_auc:.4f}")

        # 仅在特定周期打印 TPRs
        if epch % 50 == 0:
            tpr_output = ", ".join(
                # 强制将 tpr 转换为 float() 以确保格式化正常
                [f"TPR@{fpr * 100:.1f}%FPR: {float(tpr):.4f}" for fpr, tpr in zip([0.001, 0.01, 0.05, 0.1], tprs)])

            # 注意：fpr 已经是浮点数，fpr * 100 也是浮点数，不需要额外转换

            print(f"  - TPRs (Select): {tpr_output}")

        # 7. 结果存储 (保持不变)
        tprs_per_modality[modality] = tprs
        auc_per_modality[modality] = auc
        log_auc_per_modality[modality] = log_auc
        scores_per_modality[modality] = (train_l_out, test_l_out)
        indices_per_modality[modality] = (target_train_idx, target_test_idx)

    print('\n' + '-' * 70)
    print(f'✅ LIRA Attack (Epch {epch}) Finished.')
    print('=' * 70)

    return accs, tprs_per_modality, auc_per_modality, log_auc_per_modality, scores_per_modality,indices_per_modality


def cos_attack(f, K, epch, attack_mode, extract_fn=None, sample_modality='full'):
    # 标题美化
    print('\n' + '=' * 70)
    print(f'⭐ Direct Score Attack Started ⭐')
    print(f'🎯 Target Epoch: {epch} | Attack Mode: {attack_mode.upper()} | Modality Mode: {sample_modality.upper()}')
    print('=' * 70)

    accs = []
    training_res = []

    # 加载模型结果 (保持不变)
    print(f"🔄 Loading results from {K} models...")
    K=6
    for i in range(K):
        training_res.append(torch.load(f.format(i, epch), weights_only=False))
    # 加载目标模型结果 (保持不变)
    # try:
    #     target_res = torch.load(f.format(0, epch), weights_only=False)
    # except FileNotFoundError:
    #     print(f"❌ Error: Target model file not found for epoch {epch}.")
    #     return [], {}, {}, {}, {}
    global TARGET_CLIENT_ID, VAL_CLIENT_IDS

    if 'TARGET_CLIENT_ID' in globals() and 'VAL_CLIENT_IDS' in globals():
        if TARGET_CLIENT_ID is not None and VAL_CLIENT_IDS is not None:
            # 验证配置
            is_valid, error_msg = validate_reorder_config(training_res, TARGET_CLIENT_ID, VAL_CLIENT_IDS)
            if not is_valid:
                raise ValueError(f"❌ 配置验证失败: {error_msg}")

            print(f"\n🎯 检测到自定义配置:")
            print(f"   目标客户端: {TARGET_CLIENT_ID}")
            print(f"   验证客户端: {VAL_CLIENT_IDS}\n")

            # 🔥 重写所有客户端数据
            training_res = reorder_all_clients_data(training_res, TARGET_CLIENT_ID, VAL_CLIENT_IDS)
            target_res = training_res[TARGET_CLIENT_ID]
        else:
            print(f"\nℹ️  使用默认配置\n")
    else:
        print(f"\nℹ️  使用默认配置\n")

    # 模态检测 (逻辑不变，输出美化)
    if attack_mode == "loss based":
        modalities = ['full', 'audio', 'visual']  # 假设支持
    else:
        key_map = {
            "cosine attack": "tarin_cos",
            "grad diff": "tarin_diffs",
            "grad norm": "tarin_grad_norm"
        }
        key_to_peek = key_map.get(attack_mode)
        try:
            modalities = list(target_res[sample_modality][key_to_peek].keys()) if key_to_peek else []
        except KeyError:
            print(f"❌ Error: Key '{key_to_peek}' not found in results for modality '{sample_modality}'.")
            return [], {}, {}, {}, {}

    if sample_modality == 'audio':
        modalities = ['audio']
        print("🔊 Only Audio Modality Detected.")
    elif sample_modality == 'visual':
        modalities = ['visual']
        print("🎥 Only Visual Modality Detected.")

    print(f"🔍 Found and processing modalities: {', '.join(modalities)}")

    tprs_per_modality = {}
    auc_per_modality = {}
    log_auc_per_modality = {}
    scores_per_modality = {}
    indices_per_modality = {}
    for modality in modalities:
        # 模态攻击开始提示
        print(f"\n✨ Attacking Modality: **{modality.upper()}** (Method: {attack_mode.upper()}) ✨")
        print('-' * 40)

        train_liratios, val_liratios = None, None

        # ... (数据提取逻辑保持不变) ...
        # 注意：这里需要确保 val_liratios 和 train_liratios 被正确填充

        if attack_mode == "cosine attack":
            train_liratios = target_res[sample_modality]['tarin_cos'][modality]
            if MODE == "test":
                val_liratios = target_res[sample_modality]['test_cos'][modality]
            elif MODE == "val":
                val_liratios = target_res[sample_modality]['val_cos'][modality]
            elif MODE == 'mix':
                val_liratios = target_res[sample_modality]["test_cos"][modality]
                mix_test_loss = target_res[sample_modality]["mix_cos"][modality]
                # val_liratios = torch.cat([mix_test_loss, val_liratios_part], axis=0)
                val_liratios_part = select_client_specific_parts(val_liratios, K, VAL_CLIENT_IDS + [TARGET_CLIENT_ID])
                # 拼接
                val_liratios = torch.cat([mix_test_loss, val_liratios_part], axis=0)

            val_liratios = np.array([i.cpu().item() for i in val_liratios])
            train_liratios = np.array([i.cpu().item() for i in train_liratios])
        elif attack_mode == "grad diff":
            train_liratios = target_res[sample_modality]['tarin_diffs'][modality]
            if MODE == "test":
                val_liratios = target_res[sample_modality]['test_diffs'][modality]
            elif MODE == "val":
                val_liratios = target_res[sample_modality]['val_diffs'][modality]
            elif MODE == 'mix':
                val_liratios = target_res[sample_modality]["test_diffs"][modality]
                mix_test_loss = target_res[sample_modality]["mix_diffs"][modality]
                # val_liratios = torch.cat([mix_test_loss, val_liratios_part], axis=0)
                val_liratios_part = select_client_specific_parts(val_liratios, K, VAL_CLIENT_IDS + [TARGET_CLIENT_ID])
                # 拼接
                val_liratios = torch.cat([mix_test_loss, val_liratios_part], axis=0)

            val_liratios = np.array([i.cpu().item() for i in val_liratios])
            train_liratios = np.array([i.cpu().item() for i in train_liratios])
        elif attack_mode == "grad norm":
            train_liratios = target_res[sample_modality]['tarin_grad_norm'][modality]
            if MODE == "test":
                val_liratios = target_res[sample_modality]['test_grad_norm'][modality]
            elif MODE == "val":
                val_liratios = target_res[sample_modality]['val_grad_norm'][modality]
            elif MODE == 'mix':
                val_liratios = target_res[sample_modality]["test_grad_norm"][modality]
                mix_test_loss = target_res[sample_modality]["mix_grad_norm"][modality]

                # val_liratios = select_client_specific_parts(val_liratios, K, VAL_CLIENT_IDS + [TARGET_CLIENT_ID])
                # 拼接
                # val_liratios = torch.cat([mix_test_loss, val_liratios_part], axis=0)
                val_liratios_part = select_client_specific_parts(val_liratios, K, VAL_CLIENT_IDS + [TARGET_CLIENT_ID])
                # 拼接
                val_liratios = torch.cat([mix_test_loss, val_liratios_part], axis=0)
            val_liratios = -np.array([i.cpu().item() for i in val_liratios])
            train_liratios = -np.array([i.cpu().item() for i in train_liratios])
        elif attack_mode == "loss based":
            logit_key = 'logit'
            if modality == 'audio':
                logit_key = 'logit_a'
            elif modality == 'visual':
                logit_key = 'logit_v'

            train_liratios = -ce_loss_fn(target_res["train_res"][logit_key],
                                         target_res["train_res"]["labels"]).cpu().numpy()
            if MODE == "test":
                val_liratios = -ce_loss_fn(target_res["test_res"][logit_key],
                                           target_res["test_res"]["labels"]).cpu().numpy()
            elif MODE == "val":
                val_liratios = -ce_loss_fn(target_res["val_res"][logit_key],
                                           target_res["val_res"]["labels"]).cpu().numpy()
            elif MODE == 'mix':
                val_liratios = -ce_loss_fn(target_res["test_res"][logit_key],
                                                target_res["test_res"]["labels"]).cpu().numpy()
                mix_test_loss = -ce_loss_fn(target_res["mix_res"][logit_key],
                                            target_res["mix_res"]["labels"]).cpu().numpy()
                # val_liratios = np.concatenate([mix_test_loss, val_liratios_part], axis=0)
                val_liratios_part = select_client_specific_parts(val_liratios, K, VAL_CLIENT_IDS + [TARGET_CLIENT_ID])
                # 拼接
                val_liratios = np.concatenate([mix_test_loss, val_liratios_part], axis=0)

        target_train_idx = np.array([])
        target_test_idx = np.array([])
        if 'index' in target_res:
            idx_data = target_res['index']

            # Train Index
            target_train_idx = idx_data['train_indices']

            # Test/Val/Mix Index
            if MODE == "test":
                target_test_idx = idx_data['test_indices']
            elif MODE == "val":
                target_test_idx = idx_data['val_indices']
            elif MODE == 'mix':
                # 🚨 关键：Mix 模式下，Index 的拼接顺序必须是 [Mix, Test]
                idx_mix = idx_data['mix_indices']
                idx_test_full = idx_data['test_indices']
                # target_test_idx = np.concatenate([idx_mix, idx_test_part], axis=0)
                idx_test_part = select_client_specific_parts(idx_test_full, K, VAL_CLIENT_IDS + [TARGET_CLIENT_ID])
                target_test_idx = np.concatenate([idx_mix, idx_test_part], axis=0)
        else:
            print("⚠️ Warning: 'index' key not found in target_res.")


        if train_liratios is not None and val_liratios is not None:
            # 使用提取到的分数计算AUC和TPR
            auc, log_auc, tprs = plot_auc(f"{attack_mode}_{modality}", torch.tensor(val_liratios),
                                          torch.tensor(train_liratios), epch)

            # 结果输出美化
            print(f"📈 Modality **{modality.upper()}** Direct Attack Results:")
            print(f"  - AUC Score: **{auc:.4f}**")
            print(f"  - LogAUC Score: {log_auc:.4f}")

            # 仅在特定周期打印 TPRs
            if epch % 50 == 0:
                tpr_output = ", ".join(
                    # 强制将 tpr 转换为 float() 以确保格式化正常
                    [f"TPR@{fpr * 100:.1f}%FPR: {float(tpr):.4f}" for fpr, tpr in zip([0.001, 0.01, 0.05, 0.1], tprs)])

                # 注意：fpr 已经是浮点数，fpr * 100 也是浮点数，不需要额外转换

                print(f"  - TPRs (Select): {tpr_output}")

            # 结果存储 (保持不变)
            tprs_per_modality[modality] = tprs
            auc_per_modality[modality] = auc
            log_auc_per_modality[modality] = log_auc
            scores_per_modality[modality] = (train_liratios, val_liratios)
            indices_per_modality[modality] = (target_train_idx, target_test_idx)
    print('\n' + '-' * 70)
    print(f'✅ Direct Score Attack (Epch {epch}) Finished.')
    print('=' * 70)

    return accs, tprs_per_modality, auc_per_modality, log_auc_per_modality, scores_per_modality,indices_per_modality


@torch.no_grad()
def attack_comparison(p, log_path, save_dir, epochs, MAX_K, defence, seed,MODE):
    """
    一个多模态版本的主协调函数。
    """
    # 假设我们已知要处理的模态
    all_possible_modalities  = ['full', 'audio', 'visual']

    # --- 初始化用于存储结果的嵌套字典 ---
    # 外层键是模态名，内层键是攻击方法名
    attack_types = attack_modes + ["lira", "lira_loss"]
    scores = {m: {k: [] for k in attack_types} for m in all_possible_modalities}
    auc_dict = {m: {k: [] for k in attack_types} for m in all_possible_modalities}
    reses_lira = {m: [] for m in all_possible_modalities}
    reses_lira_loss = {m: [] for m in all_possible_modalities}
    reses_common = {m: {k: [] for k in attack_modes} for m in all_possible_modalities}
    single_score = {m: {} for m in all_possible_modalities}
    avg_scores = {m: {} for m in all_possible_modalities}
    other_scores = {m: {} for m in all_possible_modalities}


    # 遍历所有指定的epoch
    for epch in epochs:
        print(f"\n========================= Processing Epoch: {epch} =========================")
        modalities=['full', 'audio', 'visual']
        all_attack_dfs = {}
        for sample_modality in modalities:
            lira_accs, lira_tprs, lira_aucs, lira_log_aucs, lira_scores, lira_indices = lira_attack_ldh_cosine(
                p, epch, MAX_K, save_dir, sample_modality=sample_modality
            )

            lira_loss_accs, lira_loss_tprs, lira_loss_aucs, lira_loss_log_aucs, lira_loss_scores, lira_loss_indices = lira_attack_ldh_cosine(
                p, epch, MAX_K, save_dir, attack_mode='loss', sample_modality=sample_modality
            )

            # 2. 保存 DataFrame
            if sample_modality == 'full':
                # --- 保存 Loss 结果 ---
                df_loss = create_wide_format_dataframe(
                    'loss'.replace(" ", "_").upper(),
                    epch,
                    lira_loss_scores,
                    indices_per_modality=lira_loss_indices  # <--- ⭐ 新增传入参数
                )
                all_attack_dfs[f"LIRA_{'loss'.upper()}"] = df_loss

                # --- 保存 Cosine 结果 ---
                df_cos = create_wide_format_dataframe(
                    'cos'.replace(" ", "_").upper(),
                    epch,
                    lira_scores,
                    indices_per_modality=lira_indices  # <--- ⭐ 新增传入参数
                )
                all_attack_dfs[f"LIRA_{'cos'.upper()}"] = df_cos
            # --- 动态地解包并保存LIRA结果 ---
            for returned_modality, tprs_data in lira_tprs.items():
                scores[returned_modality]["lira"].append(tprs_data.get('0.001', 0))
                auc_dict[returned_modality]["lira"].append(lira_aucs[returned_modality])
                reses_lira[returned_modality].append(lira_scores[returned_modality])

            for returned_modality, tprs_data in lira_loss_tprs.items():
                scores[returned_modality]["lira_loss"].append(tprs_data.get('0.001', 0))
                auc_dict[returned_modality]["lira_loss"].append(lira_loss_aucs[returned_modality])
                reses_lira_loss[returned_modality].append(lira_loss_scores[returned_modality])
        # --- 调用白盒攻击函数 ---
        print(f"\n--- [White-Box Attacks] Starting  ---")
        common_results = {}
        for sample_modality in modalities:
            for attack_mode in attack_modes:
                common_accs, common_tprs, common_aucs, common_log_aucs, common_raw_scores,common_raw_index = cos_attack(p, 0, epch,
                                                                                                       attack_mode,sample_modality=sample_modality)
                if sample_modality == 'full':
                    df_lira_cos = create_wide_format_dataframe(
                        attack_mode.replace(" ", "_").upper(),
                        epch,
                        common_raw_scores,
                        indices_per_modality=common_raw_index
                    )
                    all_attack_dfs[f"{attack_mode.upper()}"] = df_lira_cos

                for returned_modality, tprs_data in common_tprs.items():
                    scores[returned_modality][attack_mode].append(tprs_data.get('0.001', 0))
                    auc_dict[returned_modality][attack_mode].append(common_aucs[returned_modality])
                    reses_common[returned_modality][attack_mode].append(common_raw_scores[returned_modality])
        save_all_attacks_to_single_excel(save_dir, epch, all_attack_dfs,MODE)



    for m in modalities:
        for attack_type in attack_types:
            if scores[m][attack_type]:
                # 找到分数最高的那个epoch的索引
                sorted_id = sorted(range(len(scores[m][attack_type])), key=lambda k: scores[m][attack_type][k],
                                   reverse=True)
                best_epoch_idx = sorted_id[0]
                # 记录最佳TPR分数和对应的AUC
                single_score[m][attack_type] = scores[m][attack_type][best_epoch_idx]
                single_score[m][f'single_{attack_type}_auc'] = auc_dict[m][attack_type][best_epoch_idx]

    print('\n------------ Sequential Attack (Averaged over all epochs) -------------')
    for m in modalities:
        print(f"\n--- Averaging for Modality: {m} ---")
        # LIRA
        if reses_lira[m]:
            train_score = np.vstack([i[0].reshape(1, -1) for i in reses_lira[m]]).mean(axis=0)
            test_score = np.vstack([i[1].reshape(1, -1) for i in reses_lira[m]]).mean(axis=0)
            auc, log_auc, tprs = plot_auc(f"averaged_lira_{m}", torch.tensor(test_score), torch.tensor(train_score),
                                          999)
            avg_scores[m]["lira"] = tprs
            other_scores[m]["lira_auc"] = [auc, log_auc]
            print(f"Modality [{m}] - Averaged LIRA TPRs: {tprs}")

        # LIRA Loss
        if reses_lira_loss[m]:
            train_score = np.vstack([i[0].reshape(1, -1) for i in reses_lira_loss[m]]).mean(axis=0)
            test_score = np.vstack([i[1].reshape(1, -1) for i in reses_lira_loss[m]]).mean(axis=0)
            auc, log_auc, tprs = plot_auc(f"averaged_lira_loss_{m}", torch.tensor(test_score),
                                          torch.tensor(train_score), 999)
            avg_scores[m]["lira_loss"] = tprs
            other_scores[m]["lira_loss_auc"] = [auc, log_auc]
            print(f"Modality [{m}] - Averaged LIRA-Loss TPRs: {tprs}")

        # 白盒攻击
        for attack_mode in attack_modes:
            if reses_common[m][attack_mode]:
                train_score = np.vstack([i[0].reshape(1, -1) for i in reses_common[m][attack_mode]]).mean(axis=0)
                test_score = np.vstack([i[1].reshape(1, -1) for i in reses_common[m][attack_mode]]).mean(axis=0)
                auc, log_auc, tprs = plot_auc(f"averaged_{attack_mode}_{m}", torch.tensor(test_score),
                                              torch.tensor(train_score), 999)
                avg_scores[m][attack_mode] = tprs
                other_scores[m][f"{attack_mode}_auc"] = [auc, log_auc]
                print(f"Modality [{m}] - Averaged {attack_mode} TPRs: {tprs}")


    final_acc_storage = []

    if 'lira_accs' in locals():
        final_acc = lira_accs
    else:
        final_acc = []  # 提供一个默认值

    # --- 调用绘图函数 ---
    # fig_out(epochs, MAX_K, defence, seed, log_path, scores, avg_scores, single_score, other_scores, final_acc)

def main(argv=None, custom_target=None, custom_val=None,shadow_list=None):
    # 引入所有需要使用的全局变量
    global MODE, attack_modes, PATH, p_folder, device, select_mode, select_method, SHADOW_NUM, SEED, mix_length
    global SAVE_DIR
    global TARGET_CLIENT_ID, VAL_CLIENT_IDS,SHAWDOW_LIST

    if custom_target is not None and custom_val is not None:
        TARGET_CLIENT_ID = custom_target
        VAL_CLIENT_IDS = custom_val
        print(f"\n🔄 [批处理模式] 启动任务: Target={TARGET_CLIENT_ID}, Val={VAL_CLIENT_IDS}")
        SHAWDOW_LIST=shadow_list
    else:
        # 如果没有传入参数，则使用默认值 (保留原有逻辑作为 fallback)
        TARGET_CLIENT_ID = 0
        VAL_CLIENT_IDS = [1, 2]
        print(f"\nℹ️ [默认模式] 使用默认配置: Target={TARGET_CLIENT_ID}, Val={VAL_CLIENT_IDS}")

    # 💡 或者使用默认行为（注释掉上面两行，取消注释下面）
    # TARGET_CLIENT_ID = None
    # VAL_CLIENT_IDS = None
    attack_modes=["cosine attack","grad diff","loss based","grad norm"]
    epochs=list(range(10,int(60)+1,10))
    p_folder="saved_mia_models"
    PATH="log_fedmia/iid"
    device=1
    MODE='mix'
    SEED = int(52)
    MAX_K=10

    configurable_path = r"E:\machang\code\FedMIA-main\FedMIA-main\saved_mia_models\cremad_K6_N1100_AVClassifer_defnone_iid$1_$1.0_$sgd_local3_s42_mod$multimodal_random"
    PATH = os.path.join(configurable_path, "client_{}_losses_epoch{}.pkl")
    name = os.path.basename(configurable_path)
    MAX_K=int(name.split("_K")[1].split("_")[0])
    model=name.split("_")[3]
    defence=name.split("_")[-5].strip('def').strip('0.0')
    seed=name.split("_")[-1]
    save_dir=p_folder + '/'+name
    SAVE_DIR = configurable_path

    if 'iid$1' in name:
        select_mode = 0
        select_method='none'
        SHADOW_NUM = 9
    else:
        select_mode = 1
        select_method ='outlier'
        SHADOW_NUM = 4
    print(configurable_path)

    print('MODE\tattack_modes\tPATH\tp_folder\tselect_mode\tselect_method\tSHADOW_NUM\tSEED')
    print(f'{MODE}\t{attack_modes}\t{PATH}\t{p_folder}\t{select_mode}\t{select_method}\t{SHADOW_NUM}\t{SEED}')
    print("name:",name)


    log_path="logs/log_res"
    # print(MAX_K,PATH)
    try:
        attack_comparison(PATH, log_path, save_dir, epochs, MAX_K, defence,seed,MODE)
        print("success!")

    except IOError:
        print("error:",MAX_K,PATH)
        pass




if __name__ == "__main__":
    # ================= 用户配置区域 =================

    # 1. 训练集构造用户组 (Colluding Users)
    # 例如: [0, 1, 2, 3] 意味着我们将生成 4 份文件：
    # Target=0, Val=[1,2,3]
    # Target=1, Val=[0,2,3] ...以此类推
    # COLLUDING_USERS = [3,4,5]
    #
    # # 2. 测试集构造用户 (Test Run)
    # # 单独指定测试集的目标和验证集
    # TEST_RUN = {'target': 2, 'val': [0,1]}
    COLLUDING_USERS = [5]

    # 2. 测试集构造用户 (Test Run)
    # 单独指定测试集的目标和验证集
    TEST_RUN = {'target': 0, 'val': [1,2,3,4]}
    # ================= 自动化执行逻辑 =================

    print("🚀 [System] 开始批量生成数据...")

    # --- 循环 1: 生成训练集数据 (One-vs-Rest) ---
    # for target_id in COLLUDING_USERS:
    #     # 逻辑：从列表中选一个做 Target，其余剩下的做 Validation
    #     val_ids = [u for u in COLLUDING_USERS if u != target_id]
    #
    #     # 🔥 关键：调用 main 函数，传入当前轮次的参数
    #     main(argv=sys.argv, custom_target=target_id, custom_val=val_ids,shadow_list=[TEST_RUN['target']]+TEST_RUN['val']+val_ids)
    #
    #     print("-" * 50)  # 分隔线

    # --- 循环 2: 生成测试集数据 ---
    print("🚀 [System] 开始生成测试集数据...")
    main(argv=sys.argv, custom_target=TEST_RUN['target'], custom_val=TEST_RUN['val'],shadow_list=TEST_RUN['val']+COLLUDING_USERS)

    print("\n🎉🎉🎉 所有任务全部执行完毕！")

